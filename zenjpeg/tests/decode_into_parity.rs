//! `decode_into()` must produce exactly what `decode()` produces.
//!
//! `decode_into` has a "direct" fast path that decodes straight into the
//! caller's buffer, and a fallback that runs the ordinary `decode()` and
//! memcpys. The direct path skips several post-decode stages, so it is only
//! correct for configurations that do not use them — anything it fails to
//! exclude is silently dropped, and the caller gets pixels that do not match
//! what they asked for with no error to tell them.
//!
//! Every test here is the same shape: configure a `Decoder`, run both APIs,
//! demand byte equality. That is the contract; the fast path is an
//! optimisation under it, not a different feature set.
//!
//! Run: `cargo test -p zenjpeg --test decode_into_parity`

use enough::Unstoppable;
use zenjpeg::decode::{CropRegion, DeblockMode, Decoder};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::types::PixelFormat;

/// Deterministic noise + saturated patches. Not a gradient: block-boundary
/// artefacts (what deblocking acts on) need real high-frequency content.
fn source_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    let mut state = 0x9E3779B97F4A7C15u64;
    let pal = [
        (220u8, 20u8, 20u8),
        (20, 200, 60),
        (30, 60, 230),
        (230, 200, 20),
        (200, 30, 200),
        (20, 210, 210),
    ];
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let n = (state >> 32) as u32;
            let (bx, by) = (x / 16, y / 16);
            if (bx + by) % 3 == 0 {
                v.extend_from_slice(&[n as u8, (n >> 8) as u8, (n >> 16) as u8]);
            } else {
                let base = pal[((bx * 3 + by * 5) % 6) as usize];
                v.extend_from_slice(&[
                    base.0.saturating_add((n & 7) as u8),
                    base.1.saturating_add(((n >> 3) & 7) as u8),
                    base.2.saturating_add(((n >> 6) & 7) as u8),
                ]);
            }
        }
    }
    v
}

fn encode(w: u32, h: u32, subsampling: ChromaSubsampling, quality: f32) -> Vec<u8> {
    let rgb = source_rgb(w, h);
    let mut enc = EncoderConfig::ycbcr(quality, subsampling)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

/// Run both APIs with the same config and demand byte equality.
#[track_caller]
fn assert_parity(decoder: &Decoder, jpeg: &[u8], format: PixelFormat, label: &str) {
    let via_decode = decoder
        .clone()
        .output_format(format)
        .decode(jpeg, Unstoppable)
        .unwrap_or_else(|e| panic!("{label}: decode() failed: {e:?}"));
    let expect = via_decode
        .pixels_u8()
        .unwrap_or_else(|| panic!("{label}: decode() produced non-u8 output"));

    // Size the destination from what decode() actually produced, so a
    // dimension change (crop) is part of what is compared.
    let mut dst = vec![0xA5u8; expect.len() + 64];
    let written = decoder
        .decode_into(jpeg, format, &mut dst, Unstoppable)
        .unwrap_or_else(|e| panic!("{label}: decode_into() failed: {e:?}"));

    assert_eq!(
        written,
        expect.len(),
        "{label}: decode_into wrote {written} bytes, decode() produced {} \
         ({}x{} at {} B/px)",
        expect.len(),
        via_decode.width,
        via_decode.height,
        format.bytes_per_pixel(),
    );

    let ndiff = dst[..written]
        .iter()
        .zip(expect.iter())
        .filter(|(a, b)| a != b)
        .count();
    let maxdiff = dst[..written]
        .iter()
        .zip(expect.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert_eq!(
        ndiff, 0,
        "{label}: decode_into disagrees with decode — {ndiff} of {written} bytes differ, \
         max delta {maxdiff}. The direct path dropped a configured stage.",
    );

    // The guard bytes past the written region must be untouched.
    assert!(
        dst[written..].iter().all(|&b| b == 0xA5),
        "{label}: decode_into wrote past the {written} bytes it reported"
    );
}

const W: u32 = 96;
const H: u32 = 64;

/// Baseline: no extra configuration. This is the case the direct path is for.
#[test]
fn plain_decode_into_matches_decode() {
    for sub in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        let jpeg = encode(W, H, sub, 85.0);
        for format in [
            PixelFormat::Rgb,
            PixelFormat::Bgr,
            PixelFormat::Rgba,
            PixelFormat::Bgra,
        ] {
            assert_parity(&Decoder::new(), &jpeg, format, &format!("plain {format:?}"));
        }
    }
}

/// `.crop(..).decode_into(..)` must return the crop, not the full image.
#[test]
fn crop_is_honoured_by_decode_into() {
    let jpeg = encode(W, H, ChromaSubsampling::Quarter, 85.0);
    for (region, label) in [
        (
            CropRegion::Pixels {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            },
            "top-left aligned",
        ),
        (
            CropRegion::Pixels {
                x: 17,
                y: 9,
                width: 40,
                height: 30,
            },
            "unaligned offset",
        ),
        (
            CropRegion::Percent {
                x: 0.25,
                y: 0.25,
                width: 0.5,
                height: 0.5,
            },
            "percent",
        ),
    ] {
        assert_parity(
            &Decoder::new().crop(region),
            &jpeg,
            PixelFormat::Rgb,
            &format!("crop {label}"),
        );
    }
}

/// Deblocking must reach `decode_into`. `Knusperli` and `Auto` were already
/// excluded from the direct path; `Boundary4Tap` and `AutoStreamable` are the
/// streaming-compatible modes and were not.
#[test]
fn deblocking_is_honoured_by_decode_into() {
    // Low quality so the deblock filter has visible blocking to act on.
    let jpeg = encode(W, H, ChromaSubsampling::Quarter, 25.0);
    for mode in [
        DeblockMode::Off,
        DeblockMode::Boundary4Tap,
        DeblockMode::AutoStreamable,
        DeblockMode::Auto,
        DeblockMode::Knusperli,
    ] {
        assert_parity(
            &Decoder::new().deblock(mode),
            &jpeg,
            PixelFormat::Rgb,
            &format!("deblock {mode:?}"),
        );
    }
}

/// A deblocking mode that is silently dropped would be indistinguishable from
/// `Off`. Prove the filter actually changes pixels at this quality, so the
/// parity test above is not comparing two identical no-ops.
#[test]
fn boundary4tap_actually_changes_pixels() {
    let jpeg = encode(W, H, ChromaSubsampling::Quarter, 25.0);
    let off = Decoder::new()
        .deblock(DeblockMode::Off)
        .decode(&jpeg, Unstoppable)
        .expect("decode off");
    let on = Decoder::new()
        .deblock(DeblockMode::Boundary4Tap)
        .decode(&jpeg, Unstoppable)
        .expect("decode boundary4tap");
    let a = off.pixels_u8().unwrap();
    let b = on.pixels_u8().unwrap();
    let ndiff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    assert!(
        ndiff > a.len() / 100,
        "Boundary4Tap changed only {ndiff} of {} bytes at Q25 — the parity test \
         above would not detect it being dropped",
        a.len()
    );
}

/// Cropping combined with a deblocking mode: both stages must survive.
#[test]
fn crop_and_deblock_together_are_honoured() {
    let jpeg = encode(W, H, ChromaSubsampling::Quarter, 25.0);
    assert_parity(
        &Decoder::new()
            .crop(CropRegion::Pixels {
                x: 8,
                y: 8,
                width: 48,
                height: 40,
            })
            .deblock(DeblockMode::Boundary4Tap),
        &jpeg,
        PixelFormat::Rgb,
        "crop + Boundary4Tap",
    );
}

/// The matrix/TRC profile carried by `tests/images/ultrahdr_sample.jpg`.
///
/// Committed in-repo (no submodule, no corpus download), so this test never
/// silently skips. Its primaries are BT.709/sRGB (`cicp` = 1/13/1).
fn embedded_srgb_icc() -> Vec<u8> {
    let jpeg = std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/images/ultrahdr_sample.jpg"),
    )
    .expect("read tests/images/ultrahdr_sample.jpg");

    let mut i = 2usize;
    while i + 3 < jpeg.len() {
        if jpeg[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = jpeg[i + 1];
        if m == 0xD8 || m == 0xD9 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        if m == 0xFF {
            i += 1;
            continue;
        }
        let len = ((jpeg[i + 2] as usize) << 8) | jpeg[i + 3] as usize;
        if m == 0xE2 && jpeg[i + 4..].starts_with(b"ICC_PROFILE\0") {
            // 12-byte signature + 2 sequence bytes, then the profile.
            return jpeg[i + 18..i + 2 + len].to_vec();
        }
        if m == 0xDA {
            break;
        }
        i += 2 + len;
    }
    panic!("no ICC profile in tests/images/ultrahdr_sample.jpg");
}

/// The same profile with its three colorant tags replaced by Display-P3
/// primaries (D50-adapted, as ICC's PCS requires). Everything else — the
/// `chad` Bradford matrix, the sRGB TRCs, the white point — is unchanged, so
/// the result is a well-formed wide-gamut profile that differs from the
/// original in exactly one respect: gamut.
///
/// Built rather than shipped so the test carries no extra binary, and so the
/// difference from sRGB is visible in the source. A P3-primaries source
/// corrected to sRGB moves saturated colours a long way, which is what makes
/// silently dropping the correction (while labelling the output sRGB) a
/// colour-accuracy bug rather than a rounding difference.
fn wide_gamut_icc() -> Vec<u8> {
    let mut p = embedded_srgb_icc();

    // s15Fixed16 encoding of the Display P3 colorants in the D50 PCS.
    let p3: [(&[u8; 4], [f64; 3]); 3] = [
        (b"rXYZ", [0.515_12, 0.241_20, -0.001_05]),
        (b"gXYZ", [0.291_98, 0.692_25, 0.041_89]),
        (b"bXYZ", [0.157_10, 0.066_57, 0.784_07]),
    ];

    let tag_count = u32::from_be_bytes(p[128..132].try_into().unwrap()) as usize;
    let mut patched = 0;
    for k in 0..tag_count {
        let e = 132 + k * 12;
        let sig: [u8; 4] = p[e..e + 4].try_into().unwrap();
        let off = u32::from_be_bytes(p[e + 4..e + 8].try_into().unwrap()) as usize;
        for (want, xyz) in &p3 {
            if &sig == *want {
                // XYZType: 4-byte sig + 4 reserved + three s15Fixed16 values.
                for (c, v) in xyz.iter().enumerate() {
                    let fixed = (v * 65536.0).round() as i32;
                    p[off + 8 + c * 4..off + 12 + c * 4].copy_from_slice(&fixed.to_be_bytes());
                }
                patched += 1;
            }
        }
    }
    assert_eq!(patched, 3, "expected three colorant tags to patch");
    p
}

/// ICC correction must reach `decode_into`, or the sink receives pixels in the
/// source gamut while `codec::info::decode_descriptor` labels them sRGB —
/// a mislabelling nothing downstream can detect.
///
/// Gated on `moxcms` because that is the feature that makes `correct_color` do
/// anything: without it `apply_icc_transform` returns the input unchanged, so
/// there would be no difference to detect. The gate is a compile-time feature,
/// not a runtime skip.
#[cfg(feature = "moxcms")]
#[test]
fn icc_correction_is_honoured_by_decode_into() {
    use zenjpeg::color::icc::TargetColorSpace;

    let rgb = source_rgb(W, H);
    let icc = wide_gamut_icc();
    let cfg = EncoderConfig::ycbcr(95.0, ChromaSubsampling::None);
    let jpeg = cfg
        .request()
        .icc_profile(&icc)
        .encode(bytemuck::cast_slice::<u8, rgb::RGB<u8>>(&rgb), W, H)
        .expect("encode with ICC");

    // Guard: the correction must be a real change, else parity is vacuous.
    let plain = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("plain decode");
    let corrected = Decoder::new()
        .correct_color(Some(TargetColorSpace::Srgb))
        .decode(&jpeg, Unstoppable)
        .expect("corrected decode");
    let ndiff = plain
        .pixels_u8()
        .unwrap()
        .iter()
        .zip(corrected.pixels_u8().unwrap().iter())
        .filter(|(a, b)| a != b)
        .count();
    let maxdelta = plain
        .pixels_u8()
        .unwrap()
        .iter()
        .zip(corrected.pixels_u8().unwrap().iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        ndiff > plain.pixels_u8().unwrap().len() / 4 && maxdelta > 16,
        "P3 -> sRGB correction changed only {ndiff} bytes (max delta {maxdelta}); \
         the parity check below would be vacuous"
    );

    assert_parity(
        &Decoder::new().correct_color(Some(TargetColorSpace::Srgb)),
        &jpeg,
        PixelFormat::Rgb,
        "correct_color(Srgb) on a wide-gamut source",
    );
}
