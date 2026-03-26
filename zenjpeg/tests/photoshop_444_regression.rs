//! Regression test for issue #2: Baseline 4:4:4 Photoshop files produce
//! max pixel diff 100-144 vs jpeg-decoder reference.
//!
//! The bug affects buf-int decode mode but NOT scanline mode.
//! Test file: photoshop-444-scrubbed.jpg (1666x1111, baseline SOF0, 4:4:4,
//! Q100 all-1 quant tables, JFIF + Photoshop APP13 + AdobeRGB ICC + EXIF + XMP)

use enough::Unstoppable;

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/testdata/");

fn load_test_file() -> Vec<u8> {
    let path = format!("{TESTDATA}photoshop-444-scrubbed.jpg");
    match std::fs::read(&path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Skipping: {path} not found");
            Vec::new()
        }
    }
}

/// Compute max and mean absolute pixel difference between two RGB buffers.
fn pixel_diff_stats(a: &[u8], b: &[u8]) -> (u8, f64) {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    let mut max_diff: u8 = 0;
    let mut sum_diff: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i16 - *y as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let mean = sum_diff as f64 / a.len() as f64;
    (max_diff, mean)
}

/// Decode with jpeg-decoder crate (reference implementation).
fn decode_reference(data: &[u8]) -> (Vec<u8>, u16, u16) {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    let pixels = decoder.decode().expect("jpeg-decoder decode failed");
    let info = decoder.info().unwrap();
    assert_eq!(
        info.pixel_format,
        jpeg_decoder::PixelFormat::RGB24,
        "expected RGB24 output"
    );
    (pixels, info.width, info.height)
}

/// Decode with zenjpeg buffered mode (the default decode() path).
fn decode_zenjpeg_buffered(data: &[u8]) -> (Vec<u8>, u32, u32) {
    let decoder = zenjpeg::decoder::Decoder::new();
    let result = decoder.decode(data, Unstoppable).expect("zenjpeg decode failed");
    let w = result.width;
    let h = result.height;
    (result.into_pixels_u8().unwrap(), w, h)
}

/// Decode with zenjpeg scanline mode.
fn decode_zenjpeg_scanline(data: &[u8]) -> (Vec<u8>, u32, u32) {
    let mut reader = zenjpeg::decoder::Decoder::new()
        .scanline_reader(data)
        .expect("scanline_reader failed");
    let w = reader.width();
    let h = reader.height();
    let stride = w as usize * 3;
    let mut pixels = vec![0u8; stride * h as usize];
    let out = imgref::ImgRefMut::new(&mut pixels, stride, h as usize);
    reader
        .read_rows_rgb8(out)
        .expect("read_rows_rgb8 failed");
    (pixels, w, h)
}

/// Decode with zune-jpeg (second reference).
fn decode_zune(data: &[u8]) -> (Vec<u8>, u16, u16) {
    use zune_jpeg::{JpegDecoder, zune_core::bytestream::ZCursor};
    let mut decoder = JpegDecoder::new(ZCursor::new(data));
    let pixels = decoder.decode().expect("zune-jpeg decode failed");
    let info = decoder.info().unwrap();
    (pixels, info.width, info.height)
}

/// Issue #2 repro: buffered decode should match jpeg-decoder within ±1.
/// Currently fails with max_diff=128.
#[test]
fn photoshop_444_buffered_vs_reference() {
    let data = load_test_file();
    if data.is_empty() {
        return;
    }

    let (ref_pixels, rw, rh) = decode_reference(&data);
    let (zen_pixels, zw, zh) = decode_zenjpeg_buffered(&data);

    assert_eq!(
        (rw as u32, rh as u32),
        (zw, zh),
        "dimension mismatch: ref={}x{} zen={}x{}",
        rw,
        rh,
        zw,
        zh
    );

    let (max_diff, mean_diff) = pixel_diff_stats(&ref_pixels, &zen_pixels);
    eprintln!("buffered vs jpeg-decoder: max={max_diff} mean={mean_diff:.2}");

    // This is the regression assertion. Before fix, max_diff=128.
    // After fix, should be <=1 (IDCT rounding only).
    assert!(
        max_diff <= 1,
        "buffered decode max_diff={max_diff} (mean={mean_diff:.2}) exceeds threshold of 1"
    );
}

/// Scanline decode should match jpeg-decoder within ±3 (known working).
#[test]
fn photoshop_444_scanline_vs_reference() {
    let data = load_test_file();
    if data.is_empty() {
        return;
    }

    let (ref_pixels, rw, rh) = decode_reference(&data);
    let (zen_pixels, zw, zh) = decode_zenjpeg_scanline(&data);

    assert_eq!(
        (rw as u32, rh as u32),
        (zw, zh),
        "dimension mismatch"
    );

    let (max_diff, mean_diff) = pixel_diff_stats(&ref_pixels, &zen_pixels);
    eprintln!("scanline vs jpeg-decoder: max={max_diff} mean={mean_diff:.2}");

    assert!(
        max_diff <= 3,
        "scanline decode max_diff={max_diff} (mean={mean_diff:.2}) exceeds threshold of 3"
    );
}

/// Cross-check: buffered and scanline should produce similar output.
#[test]
fn photoshop_444_buffered_vs_scanline() {
    let data = load_test_file();
    if data.is_empty() {
        return;
    }

    let (buf_pixels, bw, bh) = decode_zenjpeg_buffered(&data);
    let (scan_pixels, sw, sh) = decode_zenjpeg_scanline(&data);

    assert_eq!((bw, bh), (sw, sh), "dimension mismatch");

    let (max_diff, mean_diff) = pixel_diff_stats(&buf_pixels, &scan_pixels);
    eprintln!("buffered vs scanline: max={max_diff} mean={mean_diff:.2}");

    // If both paths are correct, they should match within ±1 (IDCT rounding).
    // Current bug: max_diff ≈ 128.
    assert!(
        max_diff <= 1,
        "buffered vs scanline max_diff={max_diff} (mean={mean_diff:.2})"
    );
}

/// Verify zune-jpeg also matches jpeg-decoder (sanity check on the reference).
#[test]
fn photoshop_444_zune_vs_reference() {
    let data = load_test_file();
    if data.is_empty() {
        return;
    }

    let (ref_pixels, rw, rh) = decode_reference(&data);
    let (zune_pixels, zw, zh) = decode_zune(&data);

    assert_eq!(
        (rw, rh),
        (zw, zh),
        "dimension mismatch"
    );

    let (max_diff, mean_diff) = pixel_diff_stats(&ref_pixels, &zune_pixels);
    eprintln!("zune vs jpeg-decoder: max={max_diff} mean={mean_diff:.2}");

    // zune-jpeg and jpeg-decoder both use integer IDCT but may differ by ±3
    assert!(
        max_diff <= 3,
        "zune vs jpeg-decoder max_diff={max_diff} — reference disagreement!"
    );
}
