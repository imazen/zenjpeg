//! Regression test for issue #3: CMYK scanline transform panic.
//!
//! `scanline_reader_with_transform()` did not check for CMYK (4-component)
//! images, causing an index-out-of-bounds panic in `StripProcessor` where
//! `h_samp`/`v_samp` are `[u8; 3]` but CMYK has `comp_idx=3`.

use zenjpeg::decoder::Decoder;
use zenjpeg::lossless::LosslessTransform;

static CMYK_DATA: &[u8] = include_bytes!("../testdata/cymk.jpg");

/// Non-dimension-swapping transform on CMYK — this panicked before the fix.
#[test]
fn cmyk_scanline_flip_horizontal() {
    let mut reader = Decoder::new()
        .transform(LosslessTransform::FlipHorizontal)
        .scanline_reader(CMYK_DATA)
        .expect("scanline_reader should accept CMYK with transform");

    let w = reader.width() as usize;
    let mut buf = vec![0u8; w * 3];
    let output = imgref::ImgRefMut::new(&mut buf, w * 3, 1);
    let rows = reader
        .read_rows_rgb8(output)
        .expect("read_rows_rgb8 should succeed for CMYK");
    assert_eq!(rows, 1);
}

/// Dimension-swapping transform on CMYK (already hit the buffered fallback
/// for dimension swaps, but verify CMYK doesn't break it).
#[test]
fn cmyk_scanline_rotate90() {
    let mut reader = Decoder::new()
        .transform(LosslessTransform::Rotate90)
        .scanline_reader(CMYK_DATA)
        .expect("scanline_reader should accept CMYK with rotate90");

    let w = reader.width() as usize;
    let mut buf = vec![0u8; w * 3];
    let output = imgref::ImgRefMut::new(&mut buf, w * 3, 1);
    let rows = reader
        .read_rows_rgb8(output)
        .expect("read_rows_rgb8 should succeed for CMYK rotate90");
    assert_eq!(rows, 1);
}

/// Buffered decode on CMYK with transform — verify it produces valid output.
#[test]
fn cmyk_buffered_decode_with_transform() {
    let result = Decoder::new()
        .transform(LosslessTransform::FlipHorizontal)
        .decode(CMYK_DATA, enough::Unstoppable);
    assert!(
        result.is_ok(),
        "buffered decode failed: {}",
        result.unwrap_err()
    );
    let decoded = result.unwrap();
    assert_eq!(decoded.width(), 600);
    assert_eq!(decoded.height(), 397);
}

/// Plain scanline reader (no transform) on CMYK — already worked, but verify.
#[test]
fn cmyk_scanline_no_transform() {
    let mut reader = Decoder::new()
        .scanline_reader(CMYK_DATA)
        .expect("scanline_reader should accept CMYK without transform");

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    assert_eq!(w, 600);
    assert_eq!(h, 397);

    let mut buf = vec![0u8; w * 3];
    let output = imgref::ImgRefMut::new(&mut buf, w * 3, 1);
    let rows = reader
        .read_rows_rgb8(output)
        .expect("read_rows_rgb8 should succeed");
    assert_eq!(rows, 1);
}
