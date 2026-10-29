//! Whole-image `Decoder::decode` must agree with `Decoder::scanline_reader` on
//! chroma upsampling (issue #188).
//!
//! Both are streaming decoders, but historically they used different upsamplers:
//! `decode()` (baseline_streaming) upsampled the **MCU-padded** chroma strip
//! tightly-packed, so for horizontally-subsampled images with even width the
//! final visible column was interpolated against MCU-*padding* chroma instead of
//! replicating the last real column. That corrupted the rightmost column by up
//! to ~11/255 for 4:2:2 (h2v1) — while `scanline_reader` (the strip pipeline)
//! upsampled the **real** widths with padded strides and was correct
//! (byte-identical to libjpeg-turbo's `h2v1_fancy_upsample`).
//!
//! The fix routes `decode()`'s h2v1 through the same strided kernel with real
//! widths, so the two paths are byte-identical. These tests are pure Rust (no
//! C++), so they run in the default suite.

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, Decoder};
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};

fn color_noise(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut s = 0x9e37_79b9u32;
    for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        px[0] = (s >> 24) as u8;
        px[1] = ((s >> 16) as u8).wrapping_add((i * 5) as u8);
        px[2] = ((s >> 8) as u8).wrapping_add((i * 11) as u8);
    }
    rgb
}

fn decode_whole(jpeg: &[u8], up: ChromaUpsampling) -> Vec<u8> {
    Decoder::new()
        .chroma_upsampling(up)
        .decode(jpeg, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .expect("pixels")
}

fn decode_scanline(jpeg: &[u8], up: ChromaUpsampling) -> Vec<u8> {
    let mut r = Decoder::new()
        .chroma_upsampling(up)
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

fn max_delta(a: &[u8], b: &[u8]) -> (i32, usize) {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold((0i32, 0usize), |(m, c), (x, y)| {
        let d = (*x as i32 - *y as i32).abs();
        (m.max(d), c + usize::from(d != 0))
    })
}

/// The fix: `decode()` == `scanline_reader` byte-for-byte for 4:2:2 across a
/// span of widths (odd, MCU-aligned, and even non-aligned — the last is where
/// the padding-vs-real-edge bug lived) and both upsampling filters.
#[test]
fn decode_matches_scanline_422_all_widths() {
    // 64 = MCU-aligned; 63/129 = odd; 74/130 = even non-aligned (buggy case).
    for &(w, h) in &[(63, 40), (64, 40), (74, 58), (129, 48), (130, 47)] {
        let rgb = color_noise(w, h);
        let jpeg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::HalfHorizontal)
            .progressive(false)
            .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .expect("encode 4:2:2");
        for up in [
            ChromaUpsampling::Triangle,
            ChromaUpsampling::NearestNeighbor,
        ] {
            let (m, c) = max_delta(&decode_whole(&jpeg, up), &decode_scanline(&jpeg, up));
            assert_eq!(
                m, 0,
                "4:2:2 {w}x{h} {up:?}: decode() vs scanline max delta {m} over {c} bytes"
            );
        }
    }
}

/// Guard that the untouched 4:2:0 / 4:4:0 paths still agree closely with
/// `scanline_reader` (they share the same fused/strip kernels; the tiny residual
/// is pre-existing IDCT/round noise, not an edge bug). Only the h2v1 branch was
/// changed, so these must not regress.
#[test]
fn decode_matches_scanline_420_440_unchanged() {
    for (name, ss, tol) in [
        ("4:2:0", ChromaSubsampling::Quarter, 2),
        ("4:4:0", ChromaSubsampling::HalfVertical, 0),
    ] {
        let (w, h) = (74usize, 58usize);
        let rgb = color_noise(w, h);
        let jpeg = EncoderConfig::ycbcr(90.0, ss)
            .progressive(false)
            .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .expect("encode");
        let (m, _c) = max_delta(
            &decode_whole(&jpeg, ChromaUpsampling::Triangle),
            &decode_scanline(&jpeg, ChromaUpsampling::Triangle),
        );
        assert!(
            m <= tol,
            "{name}: decode() vs scanline max delta {m} > {tol}"
        );
    }
}

/// Cross-check against a pure-Rust libjpeg-compatible reference (`jpeg-decoder`):
/// after the fix, `decode()`'s rightmost 4:2:2 column matches the reference to
/// within normal decoder rounding (≤3), where it was off by up to 11 before.
#[test]
fn decode_422_right_edge_matches_reference() {
    let (w, h) = (74usize, 58usize);
    let rgb = color_noise(w, h);
    let jpeg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::HalfHorizontal)
        .progressive(false)
        .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode 4:2:2");

    let zen = decode_whole(&jpeg, ChromaUpsampling::Triangle);
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
    let refpx = d.decode().expect("jpeg-decoder decode");
    assert_eq!(
        d.info().unwrap().pixel_format,
        jpeg_decoder::PixelFormat::RGB24
    );

    let mut edge_max = 0i32;
    for row in 0..h {
        for ch in 0..3 {
            let i = (row * w + (w - 1)) * 3 + ch;
            edge_max = edge_max.max((zen[i] as i32 - refpx[i] as i32).abs());
        }
    }
    assert!(
        edge_max <= 3,
        "4:2:2 rightmost column vs jpeg-decoder reference: max {edge_max} (>3 means the \
         padding-edge regression is back)"
    );
}
