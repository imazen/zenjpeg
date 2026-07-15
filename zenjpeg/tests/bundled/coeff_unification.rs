//! Coefficient-centric buffered decode (issue #187).
//!
//! Progressive/arithmetic JPEGs can't stream, so they're decoded up front. The
//! old buffered path stored pre-converted RGB and then re-derived YCbCr/gray
//! *lossily* (RGB→YCbCr / RGB→Y) on those reads. The coefficient-centric path
//! stores the DCT coefficients and runs the same strip pipeline as streaming,
//! so every output format is native.
//!
//! Wired for 3-component color with symmetric Cb/Cr sampling: 4:4:4, 4:2:2,
//! 4:2:0 and 4:4:0 (see `Decoder::coeff_strip_compatible`). CMYK, the
//! exotic-sampling normalization, and asymmetric chroma stay on the RGB
//! buffered path.
//!
//! ## Why some tests use flat chroma
//!
//! [`color_noise`] gives every 8×8 chroma block high-frequency content, so its
//! blocks always quantize with `coeff_count > 1`. That silently skipped the
//! DC-only branch of the vertical-context peek, which is the branch real photos
//! take constantly (smooth chroma → DC-only chroma blocks) and which was wrong
//! by 128 levels. See [`smooth_chroma_bands`] — a generator that *does* produce
//! DC-only chroma blocks is the only reason these paths are covered.

use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};

fn color_noise(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut s = 0x1234_5678u32;
    for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Distinct, structured per-channel content so chroma isn't flat.
        px[0] = (s >> 24) as u8;
        px[1] = ((s >> 16) as u8).wrapping_add((i * 5) as u8);
        px[2] = ((s >> 8) as u8).wrapping_add((i * 11) as u8);
    }
    rgb
}

/// Chroma that is **flat within each 8×8 chroma block** but changes across MCU
/// boundaries, so chroma blocks quantize to DC-only (`coeff_count <= 1`) while
/// luma keeps high-frequency detail.
///
/// This is the shape of ordinary photographic content, and the shape
/// [`color_noise`] cannot produce. The coefficient path's vertical-context peek
/// had a hand-rolled DC-only IDCT (`(dc + 1024) >> 11` instead of
/// `(dc + 4 + 1024) >> 3`) that only fires on DC-only blocks — so every
/// noise-based test passed while a real photo (waterhouse.jpg, 2048×1153 4:2:0
/// progressive) was wrong by 76/255 on the last row of every MCU row.
fn smooth_chroma_bands(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Band changes every 16 rows => chroma differs across the MCU
            // boundary, so a wrong bottom-row context is visible.
            let (r, g, b) = match (y / 16) % 4 {
                0 => (200i32, 90i32, 70i32),
                1 => (70, 180, 90),
                2 => (80, 90, 210),
                _ => (190, 190, 70),
            };
            // Luma detail rides on top without perturbing chroma much: the same
            // delta on all three channels moves Y, leaves Cb/Cr ~unchanged.
            let luma = ((x * 7 + y * 3) % 90) as i32 / 3;
            rgb[i] = (r + luma).clamp(0, 255) as u8;
            rgb[i + 1] = (g + luma).clamp(0, 255) as u8;
            rgb[i + 2] = (b + luma).clamp(0, 255) as u8;
        }
    }
    rgb
}

fn read_all_ycbcr(jpeg: &[u8]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Decoder::new()
        .scanline_reader(jpeg)
        .expect("scanline_reader");
    let w = r.width() as usize;
    let h = r.height() as usize;
    let mut y = vec![0f32; w * h];
    let mut cb = vec![0f32; w * h];
    let mut cr = vec![0f32; w * h];
    let mut row = 0;
    while row < h {
        let n = r
            .read_rows_ycbcr_f32(
                &mut y[row * w..],
                &mut cb[row * w..],
                &mut cr[row * w..],
                w,
                h - row,
            )
            .expect("read_rows_ycbcr_f32");
        if n == 0 {
            break;
        }
        row += n;
    }
    (y, cb, cr)
}

/// Progressive 4:4:4 color decodes NATIVE YCbCr — byte-identical to the
/// baseline decode of the same source — not the lossy RGB→YCbCr round-trip the
/// old buffered path produced. Baseline and progressive of one source have
/// identical quantized coefficients, so native YCbCr must match exactly.
#[test]
fn progressive_444_ycbcr_is_native_not_lossy() {
    let (w, h) = (72usize, 56usize);
    let rgb = color_noise(w, h);

    let base = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
        .progressive(false)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode baseline 4:4:4");
    let prog = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
        .progressive(true)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode progressive 4:4:4");

    // The fix: native YCbCr. Baseline (streaming) and progressive (coefficient
    // path) share identical quantized coefficients, so their native YCbCr
    // planes are byte-identical. The old lossy path would differ by the
    // YCbCr→RGB→YCbCr round-trip error.
    let (yb, cbb, crb) = read_all_ycbcr(&base);
    let (yp, cbp, crp) = read_all_ycbcr(&prog);
    assert_eq!(
        yb, yp,
        "progressive 4:4:4 Y must be native (match baseline)"
    );
    assert_eq!(
        cbb, cbp,
        "progressive 4:4:4 Cb must be native (match baseline)"
    );
    assert_eq!(
        crb, crp,
        "progressive 4:4:4 Cr must be native (match baseline)"
    );
}

/// The coefficient path must not regress the common RGB read: progressive
/// 4:4:4 RGB stays byte-identical to the one-shot `decode()`.
#[test]
fn progressive_444_rgb_matches_decode() {
    let (w, h) = (80usize, 64usize);
    let rgb = color_noise(w, h);
    let prog = EncoderConfig::ycbcr(88.0, ChromaSubsampling::None)
        .progressive(true)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode progressive 4:4:4");

    let p1 = Decoder::new()
        .decode(&prog, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap();

    let mut reader = Decoder::new().scanline_reader(&prog).unwrap();
    let rw = reader.width() as usize;
    let rh = reader.height() as usize;
    let stride = rw * 3;
    let mut p2 = vec![0u8; rh * stride];
    let mut row = 0;
    while row < rh {
        let buf = imgref::ImgRefMut::new_stride(&mut p2[row * stride..], rw * 3, rh - row, stride);
        let n = reader.read_rows_rgb8(buf).unwrap();
        if n == 0 {
            break;
        }
        row += n;
    }
    let max = p1
        .iter()
        .zip(&p2)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    assert_eq!(max, 0, "progressive 4:4:4 RGB: decode() vs scanline_reader");
}

fn read_all_rgb8(jpeg: &[u8]) -> Vec<u8> {
    let mut r = Decoder::new()
        .scanline_reader(jpeg)
        .expect("scanline_reader");
    let (w, h) = (r.width() as usize, r.height() as usize);
    let stride = w * 3;
    let mut p = vec![0u8; h * stride];
    let mut row = 0;
    while row < h {
        let buf = imgref::ImgRefMut::new_stride(&mut p[row * stride..], w * 3, h - row, stride);
        let n = r.read_rows_rgb8(buf).expect("read_rows_rgb8");
        if n == 0 {
            break;
        }
        row += n;
    }
    p
}

/// **Upsampling correctness for 4:2:2 through the coefficient path.**
///
/// Baseline 4:2:2 *streams* — it runs the years-validated, libjpeg-turbo-matching
/// horizontal chroma-upsampling `StripProcessor::upsample_chroma`. Progressive
/// 4:2:2 of the same source can't stream, so it takes the coefficient path
/// (`from_coefficients` → the *same* strip pipeline). Baseline and progressive
/// share identical quantized coefficients, so the streaming-decoder RGB must be
/// byte-identical — if it weren't, the coefficient path's upsampling would
/// diverge from the reference (the 4:2:0 vertical-boundary failure, but
/// horizontal). It is byte-identical: the unification preserves upsampling.
///
/// NOTE: both sides read through [`Decoder::scanline_reader`], the canonical
/// streaming decoder. The whole-image `Decoder::decode` convenience wrapper has a
/// *separate, pre-existing* right-edge chroma quirk for 4:2:2 that diverges from
/// `scanline_reader` by up to 11/255 on the rightmost column — present on
/// baseline images too, unrelated to issue #187. Using `decode` here would test
/// that unrelated wrapper bug, not the coefficient path.
#[test]
fn progressive_422_rgb_matches_baseline_upsampling() {
    let (w, h) = (74usize, 58usize);
    let rgb = color_noise(w, h);

    let base = EncoderConfig::ycbcr(90.0, ChromaSubsampling::HalfHorizontal)
        .progressive(false)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode baseline 4:2:2");
    let prog = EncoderConfig::ycbcr(90.0, ChromaSubsampling::HalfHorizontal)
        .progressive(true)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode progressive 4:2:2");

    // Baseline streams (reference upsampling); progressive takes the coefficient
    // path (unified upsampling). Same coefficients ⇒ byte-identical RGB.
    let ref_rgb = read_all_rgb8(&base);
    let coeff_rgb = read_all_rgb8(&prog);

    assert_eq!(ref_rgb.len(), coeff_rgb.len(), "4:2:2 RGB length mismatch");
    let (max, count) = ref_rgb
        .iter()
        .zip(&coeff_rgb)
        .fold((0i32, 0usize), |(m, c), (a, b)| {
            let d = (*a as i32 - *b as i32).abs();
            (m.max(d), c + usize::from(d != 0))
        });
    assert_eq!(
        max, 0,
        "4:2:2 horizontal upsampling diverges on the coefficient path: \
         max pixel delta {max} over {count} bytes (baseline-stream vs progressive-coeff)"
    );
}

/// **Vertical chroma upsampling through the coefficient path (4:2:0 / 4:4:0).**
///
/// Same contract as the 4:2:2 test above, but for the vertically-subsampled
/// modes, which need the *next* MCU row's first chroma row as bottom context.
/// Baseline streams (reference); progressive of the same source takes the
/// coefficient path. Identical quantized coefficients ⇒ byte-identical RGB.
///
/// Swept over both chroma regimes deliberately: `color_noise` (every chroma
/// block high-frequency) passed even when the DC-only peek was wrong by 128
/// levels, so `smooth_chroma_bands` (DC-only chroma blocks, i.e. what real
/// photos produce) is the arm that actually exercises the bug. Sizes cover an
/// MCU-aligned height and a partial bottom MCU.
#[test]
fn progressive_420_440_vertical_upsampling_matches_baseline() {
    for (sub_name, sub) in [
        ("4:2:0", ChromaSubsampling::Quarter),
        ("4:4:0", ChromaSubsampling::HalfVertical),
    ] {
        for (gen_name, make) in [
            ("hf-chroma", color_noise as fn(usize, usize) -> Vec<u8>),
            ("smooth-chroma(DC-only)", smooth_chroma_bands),
        ] {
            // 64×48: MCU-aligned. 74×58: partial bottom MCU + partial right MCU.
            for (w, h) in [(64usize, 48usize), (74, 58)] {
                let rgb = make(w, h);
                let base = EncoderConfig::ycbcr(90.0, sub)
                    .progressive(false)
                    .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
                    .expect("encode baseline");
                let prog = EncoderConfig::ycbcr(90.0, sub)
                    .progressive(true)
                    .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
                    .expect("encode progressive");

                let ref_rgb = read_all_rgb8(&base);
                let coeff_rgb = read_all_rgb8(&prog);
                assert_eq!(ref_rgb.len(), coeff_rgb.len(), "RGB length mismatch");

                // Report the worst row so a regression names the MCU boundary
                // it broke on (the old bug hit exactly row_in_mcu == 15).
                let stride = w * 3;
                let mut worst = (0i32, 0usize);
                for y in 0..h {
                    let m = (0..stride)
                        .map(|x| {
                            let i = y * stride + x;
                            (ref_rgb[i] as i32 - coeff_rgb[i] as i32).abs()
                        })
                        .max()
                        .unwrap_or(0);
                    if m > worst.0 {
                        worst = (m, y);
                    }
                }
                assert_eq!(
                    worst.0,
                    0,
                    "{sub_name} {gen_name} {w}x{h}: vertical upsampling diverges on the \
                     coefficient path — max delta {} at row {} (row_in_mcu {}); \
                     baseline-stream vs progressive-coeff",
                    worst.0,
                    worst.1,
                    worst.1 % 16
                );
            }
        }
    }
}

/// Native YCbCr for 4:2:0 through the coefficient path, with DC-only chroma:
/// baseline (streaming) and progressive (coefficient) must yield byte-identical
/// *upsampled* YCbCr planes. Isolates chroma-plane handling + vertical upsample
/// from the YCbCr→RGB matrix.
#[test]
fn progressive_420_ycbcr_matches_baseline() {
    let (w, h) = (74usize, 58usize);
    let rgb = smooth_chroma_bands(w, h);

    let base = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode baseline 4:2:0");
    let prog = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .progressive(true)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode progressive 4:2:0");

    let (yb, cbb, crb) = read_all_ycbcr(&base);
    let (yp, cbp, crp) = read_all_ycbcr(&prog);
    assert_eq!(yb, yp, "4:2:0 Y must match baseline (native)");
    assert_eq!(cbb, cbp, "4:2:0 Cb must match baseline (native upsample)");
    assert_eq!(crb, crp, "4:2:0 Cr must match baseline (native upsample)");
}

/// Native YCbCr for 4:2:2 through the coefficient path: baseline (streaming) and
/// progressive (coefficient) of one source must yield byte-identical *upsampled*
/// YCbCr planes. This isolates the chroma-plane handling + horizontal upsample
/// from the YCbCr→RGB matrix.
#[test]
fn progressive_422_ycbcr_matches_baseline() {
    let (w, h) = (74usize, 58usize);
    let rgb = color_noise(w, h);

    let base = EncoderConfig::ycbcr(90.0, ChromaSubsampling::HalfHorizontal)
        .progressive(false)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode baseline 4:2:2");
    let prog = EncoderConfig::ycbcr(90.0, ChromaSubsampling::HalfHorizontal)
        .progressive(true)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode progressive 4:2:2");

    let (yb, cbb, crb) = read_all_ycbcr(&base);
    let (yp, cbp, crp) = read_all_ycbcr(&prog);
    assert_eq!(yb, yp, "4:2:2 Y must match baseline (native)");
    assert_eq!(cbb, cbp, "4:2:2 Cb must match baseline (native upsample)");
    assert_eq!(crb, crp, "4:2:2 Cr must match baseline (native upsample)");
}
