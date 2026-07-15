//! Coefficient-centric buffered decode (issue #187).
//!
//! Progressive/arithmetic JPEGs can't stream, so they're decoded up front. The
//! old buffered path stored pre-converted RGB and then re-derived YCbCr/gray
//! *lossily* (RGB→YCbCr / RGB→Y) on those reads. The coefficient-centric path
//! stores the DCT coefficients and runs the same strip pipeline as streaming,
//! so every output format is native.
//!
//! This is currently wired for 3-component color with no vertical chroma
//! subsampling (4:4:4, 4:2:2) — 4:2:0's vertical chroma-upsampling boundary
//! handling on the coefficient path is not yet unified (see
//! `Decoder::coeff_strip_compatible`).

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
