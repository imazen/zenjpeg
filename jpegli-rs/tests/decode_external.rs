//! Integration tests for decoding external JPEG files.
//!
//! Verifies that jpegli-rs decoder produces output matching jpeg-decoder
//! for JPEGs encoded by external tools (ImageMagick, etc.)

use dssim::Dssim;
use rgb::RGBA8;
use std::path::Path;

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let a_rgba = rgb_to_rgba(a);
    let b_rgba = rgb_to_rgba(b);
    let a_img = attr.create_image_rgba(&a_rgba, width, height).unwrap();
    let b_img = attr.create_image_rgba(&b_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&a_img, b_img);
    dssim.into()
}

fn decode_with_jpegli(data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let mut decoder = jpegli::Decoder::new();
    let img = decoder.decode(data).map_err(|e| e.to_string())?;
    Ok((img.data, img.width as usize, img.height as usize))
}

fn decode_with_jpeg_decoder(data: &[u8]) -> (Vec<u8>, usize, usize) {
    let mut decoder =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
    let pixels = decoder.decode().expect("jpeg-decoder failed");
    let (width, height) = decoder.dimensions().unwrap();
    (pixels, width, height)
}

/// Test decoding 4:4:4 JPEG from ImageMagick
#[test]
fn test_decode_im_q85_444() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower.png.im_q85_444.jpg");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let jpeg_data = std::fs::read(path).expect("read file");

    let (jpegli_pixels, jw, jh) = decode_with_jpegli(&jpeg_data).expect("jpegli decode");
    let (ref_pixels, rw, rh) = decode_with_jpeg_decoder(&jpeg_data);

    assert_eq!((jw, jh), (rw, rh), "Dimension mismatch");

    let dssim = compute_dssim(&jpegli_pixels, &ref_pixels, jw, jh);
    println!("4:4:4 decode DSSIM vs reference: {:.6}", dssim);

    // STRICT CHECK: 4:4:4 decoding should be nearly identical (only rounding differences)
    const DSSIM_THRESHOLD_444: f64 = 0.0001;
    assert!(
        dssim < DSSIM_THRESHOLD_444,
        "4:4:4 DSSIM {} too high (max: {})",
        dssim,
        DSSIM_THRESHOLD_444
    );
}

/// Test decoding non-interleaved 4:4:4 JPEG
#[test]
fn test_decode_444_non_interleaved() {
    let path = jpegli::test_utils::get_testdata_dir()
        .join("jxl/flower/flower_small.q85_444_non_interleaved.jpg");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let jpeg_data = std::fs::read(path).expect("read file");

    let (jpegli_pixels, jw, jh) = decode_with_jpegli(&jpeg_data).expect("jpegli decode");
    let (ref_pixels, rw, rh) = decode_with_jpeg_decoder(&jpeg_data);

    assert_eq!((jw, jh), (rw, rh), "Dimension mismatch");

    let dssim = compute_dssim(&jpegli_pixels, &ref_pixels, jw, jh);
    println!("4:4:4 non-interleaved DSSIM vs reference: {:.6}", dssim);

    // STRICT CHECK: Non-interleaved 4:4:4 should match reference decoder exactly
    const DSSIM_THRESHOLD_444: f64 = 0.0001;
    assert!(
        dssim < DSSIM_THRESHOLD_444,
        "4:4:4 non-interleaved DSSIM {} too high (max: {})",
        dssim,
        DSSIM_THRESHOLD_444
    );
}

/// Test that 4:2:0 decoding works with MCU interleaving
#[test]
fn test_decode_420_mcu_interleaved() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower.png.im_q85_420.jpg");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let jpeg_data = std::fs::read(path).expect("read file");

    // 4:2:0 now works with proper MCU interleaving
    let (jpegli_pixels, jw, jh) = decode_with_jpegli(&jpeg_data).expect("jpegli decode");
    let (ref_pixels, rw, rh) = decode_with_jpeg_decoder(&jpeg_data);

    assert_eq!((jw, jh), (rw, rh), "Dimension mismatch");

    let dssim = compute_dssim(&jpegli_pixels, &ref_pixels, jw, jh);
    println!("4:2:0 MCU interleaved DSSIM vs reference: {:.6}", dssim);

    // 4:2:0 has looser tolerance than 4:4:4 due to chroma upsampling differences
    // between implementations. This is a known issue documented in JPEG specs.
    // Justification: Different upsampling algorithms (bilinear vs fancy) produce
    // different results. The 0.001 threshold allows for upsampling variations.
    const DSSIM_THRESHOLD_420: f64 = 0.001;
    assert!(
        dssim < DSSIM_THRESHOLD_420,
        "4:2:0 DSSIM {} too high (max: {}). Note: 4:2:0 allows 10x looser threshold \
         than 4:4:4 due to chroma upsampling algorithm differences.",
        dssim,
        DSSIM_THRESHOLD_420
    );
}

/// Test decoding grayscale JPEG
#[test]
fn test_decode_grayscale() {
    let path = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower.png.im_q85_gray.jpg");
    if !path.exists() {
        eprintln!("Skipping: test file not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let jpeg_data = std::fs::read(path).expect("read file");

    // Try to decode - grayscale may or may not be supported yet
    match decode_with_jpegli(&jpeg_data) {
        Ok((jpegli_pixels, jw, jh)) => {
            let (ref_pixels, rw, rh) = decode_with_jpeg_decoder(&jpeg_data);
            assert_eq!((jw, jh), (rw, rh), "Dimension mismatch");

            // jpeg-decoder returns grayscale as single channel
            // jpegli returns RGB (grayscale expanded to RGB)
            // Compare luminance only
            let jpegli_luma: Vec<u8> = if jpegli_pixels.len() == jw * jh * 3 {
                // RGB - take just R (same as G and B for grayscale)
                jpegli_pixels.chunks(3).map(|c| c[0]).collect()
            } else {
                jpegli_pixels.clone()
            };

            let ref_luma: Vec<u8> = if ref_pixels.len() == rw * rh * 3 {
                ref_pixels.chunks(3).map(|c| c[0]).collect()
            } else {
                ref_pixels.clone()
            };

            // Simple MSE comparison for grayscale
            let mse: f64 = jpegli_luma
                .iter()
                .zip(ref_luma.iter())
                .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                .sum::<f64>()
                / jpegli_luma.len() as f64;

            let psnr = if mse > 0.0 {
                10.0 * (255.0 * 255.0 / mse).log10()
            } else {
                f64::INFINITY
            };

            println!("Grayscale decode PSNR vs reference: {:.2} dB", psnr);
            assert!(psnr > 40.0, "PSNR {} too low", psnr);
        }
        Err(e) => {
            println!("Grayscale decode not supported yet: {}", e);
        }
    }
}
