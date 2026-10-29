//! Regression test for issue #149: `auto_orient(true)` (the default) must
//! produce exactly the same pixels as decoding upright and applying the
//! orientation in the pixel domain.
//!
//! The native decoder used to apply EXIF orientation as a DCT-coefficient-
//! domain transform. The decoder's coefficient storage is MCU-padded;
//! transforms relocate the trailing right/bottom padding, and when it lands
//! on a leading (top/left) edge the visible image starts at a nonzero offset
//! inside the transformed plane. That offset was computed at 8-px
//! granularity instead of the stored grid's MCU granularity, shifting the
//! whole output by 8 px for 4:2:0 images whose height mod 16 was in 1..=8
//! (e.g. every 4000x3000 EXIF-6 phone photo: 88% of bytes wrong, max abs
//! diff 252). Dimension-swapping transforms additionally forced the f32
//! IDCT, diverging from the integer-IDCT upright decode by up to 9 channel
//! steps even on MCU-aligned images.
//!
//! Decode-time orientation is now a pixel-domain permutation of the upright
//! decode, so path A (auto_orient) and path B (decode upright + bake) must
//! be byte-identical for every orientation, subsampling, and size.
//!
//! Matrix: all 8 EXIF orientations x {4:4:4, 4:2:0} x sizes
//! {64x48 (MCU-aligned), 67x45, 80x53 (non-aligned, partial MCUs)}.

use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::encode::encoder_config::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};
use zenjpeg::encode::exif::{Exif, Orientation};
use zenjpeg::types::PixelFormat;
use zenpixels::{PixelDescriptor, PixelSlice};

/// All 8 EXIF orientation values (EXIF tags 1-8).
const ALL_ORIENTATIONS: &[Orientation] = &[
    Orientation::Normal,
    Orientation::FlipHorizontal,
    Orientation::Rotate180,
    Orientation::FlipVertical,
    Orientation::Transpose,
    Orientation::Rotate90,
    Orientation::Transverse,
    Orientation::Rotate270,
];

/// Deterministic noise + solid patches (NOT gradients — gradients produce
/// degenerate DCT coefficients and hide block-placement bugs).
fn noise_patches_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 3];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for v in px.iter_mut() {
        // xorshift* PRNG — deterministic across platforms
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *v = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8;
    }
    // Solid patches at asymmetric offsets so every orientation produces a
    // visually distinct layout (catches direction mix-ups, not just shifts).
    let patches: &[(usize, usize, usize, usize, [u8; 3])] = &[
        (1, 1, w / 3, h / 4, [255, 0, 0]),
        (w / 2, h / 3, w / 3, h / 3, [0, 255, 0]),
        (w / 4, (h * 2) / 3, w / 2, h / 4, [40, 40, 220]),
    ];
    for &(x0, y0, pw, ph, c) in patches {
        for y in y0..(y0 + ph).min(h) {
            for x in x0..(x0 + pw).min(w) {
                px[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
            }
        }
    }
    px
}

/// Encode a noise+patches RGB image with the given EXIF orientation tag.
fn encode_with_orientation(
    w: usize,
    h: usize,
    orient: Orientation,
    ss: ChromaSubsampling,
) -> Vec<u8> {
    let pixels = noise_patches_rgb(w, h);
    EncoderConfig::ycbcr(90.0, ss)
        .request()
        .exif(Exif::build().orientation(orient))
        .encode_bytes(&pixels, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode failed")
}

/// Path A: the native decoder with auto_orient(true) — the default.
fn decode_auto_orient(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let result = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .auto_orient(true)
        .decode(jpeg, Unstoppable)
        .expect("auto-orient decode failed");
    let (w, h) = (result.width(), result.height());
    (w, h, result.into_pixels_u8().expect("u8 pixels"))
}

/// Path B: decode upright, then bake the orientation in the pixel domain
/// with zenpixels_convert::orient::apply_orientation (the reference
/// workaround from issue #149, which matches ImageMagick -auto-orient).
fn decode_then_bake(jpeg: &[u8], orient: Orientation) -> (u32, u32, Vec<u8>) {
    let result = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .auto_orient(false)
        .decode(jpeg, Unstoppable)
        .expect("upright decode failed");
    let (w, h) = (result.width(), result.height());
    let pixels = result.into_pixels_u8().expect("u8 pixels");
    let zp_orient =
        zenpixels::Orientation::from_exif(orient as u8).expect("valid EXIF orientation");
    let slice = PixelSlice::new(&pixels, w, h, w as usize * 3, PixelDescriptor::RGB8_SRGB)
        .expect("pixel slice");
    let oriented = zenpixels_convert::orient::apply_orientation(slice, zp_orient);
    let (ow, oh) = (oriented.width(), oriented.height());
    let oslice = oriented.as_slice();
    let mut out = Vec::with_capacity(ow as usize * oh as usize * 3);
    for y in 0..oh {
        out.extend_from_slice(oslice.row(y));
    }
    (ow, oh, out)
}

#[test]
fn auto_orient_matches_pixel_domain_bake() {
    let sizes: &[(u32, u32)] = &[
        (64, 48), // MCU-aligned for both 4:4:4 and 4:2:0
        (67, 45), // partial MCUs on both axes (odd pads)
        (80, 53), // width 16-aligned, height unaligned (odd pad)
    ];
    let subsamplings: &[(ChromaSubsampling, &str)] = &[
        (ChromaSubsampling::None, "4:4:4"),
        (ChromaSubsampling::Quarter, "4:2:0"),
    ];

    let mut failures = Vec::new();
    for &(w, h) in sizes {
        for &(ss, ss_name) in subsamplings {
            for &orient in ALL_ORIENTATIONS {
                let jpeg = encode_with_orientation(w as usize, h as usize, orient, ss);
                let (aw, ah, a) = decode_auto_orient(&jpeg);
                let (bw, bh, b) = decode_then_bake(&jpeg, orient);

                if (aw, ah) != (bw, bh) {
                    failures.push(format!(
                        "{w}x{h} {ss_name} {orient:?}: dims {aw}x{ah} vs {bw}x{bh}"
                    ));
                    continue;
                }

                // Decode-time orientation is a pure pixel permutation of the
                // upright decode, identical to what path B performs — the
                // outputs must be byte-identical. Do NOT loosen this bound:
                // any divergence means the decoder is no longer permuting
                // the upright pixels (issue #149).
                if a != b {
                    let mut max_d = 0u32;
                    let mut diff_count = 0u64;
                    for (&av, &bv) in a.iter().zip(b.iter()) {
                        let d = u32::from(av.abs_diff(bv));
                        if d > 0 {
                            diff_count += 1;
                        }
                        max_d = max_d.max(d);
                    }
                    failures.push(format!(
                        "{w}x{h} {ss_name} {orient:?}: not byte-identical \
                         (max abs diff {max_d}, {diff_count} of {} bytes differ)",
                        a.len()
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "auto_orient(true) diverges from decode-upright-then-orient:\n{}",
        failures.join("\n")
    );
}

/// The scanline reader must agree with decode() for oriented output
/// (issue #149 also affected the from_coefficients scanline path, which
/// served the transformed grid without the leading-edge crop).
#[test]
fn scanline_reader_auto_orient_matches_decode() {
    let sizes: &[(u32, u32)] = &[(64, 48), (67, 45), (80, 53)];
    let mut failures = Vec::new();
    for &(w, h) in sizes {
        for &orient in ALL_ORIENTATIONS {
            let jpeg =
                encode_with_orientation(w as usize, h as usize, orient, ChromaSubsampling::Quarter);
            let (dw, dh, expected) = decode_auto_orient(&jpeg);

            let mut reader = Decoder::new()
                .output_format(PixelFormat::Rgb)
                .auto_orient(true)
                .scanline_reader(&jpeg)
                .expect("scanline reader");
            assert_eq!(
                (reader.width(), reader.height()),
                (dw, dh),
                "{w}x{h} {orient:?}: scanline dims"
            );
            let stride = dw as usize * 3;
            let mut pixels = vec![0u8; stride * dh as usize];
            let mut row = 0usize;
            while row < dh as usize {
                let buf = &mut pixels[row * stride..(row + 1) * stride];
                let output = imgref::ImgRefMut::new(buf, stride, 1);
                let n = reader.read_rows_rgb8(output).expect("read_rows_rgb8");
                if n == 0 {
                    break;
                }
                row += n;
            }
            assert_eq!(row, dh as usize, "{w}x{h} {orient:?}: rows served");

            if pixels != expected {
                let max_d = pixels
                    .iter()
                    .zip(expected.iter())
                    .map(|(&a, &b)| u32::from(a.abs_diff(b)))
                    .max()
                    .unwrap_or(0);
                failures.push(format!(
                    "{w}x{h} 4:2:0 {orient:?}: scanline output differs from decode() \
                     (max abs diff {max_d})"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "scanline auto-orient diverges from decode():\n{}",
        failures.join("\n")
    );
}
