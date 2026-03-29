#![cfg(feature = "__ffi-tests")]
//! Tests for grayscale JPEG decoding with both streaming and non-streaming interfaces.
//!
//! This tests the decoder with:
//! 1. A standard grayscale test image (flower_gray.jpg)
//! 2. An extracted gain map from an UltraHDR image (if available)
use enough::Unstoppable;

use imgref::ImgRefMut;
use zenjpeg::decoder::{Decoder, PixelFormat};
#[cfg(feature = "ultrahdr")]
use zenjpeg::ultrahdr::UltraHdrExtras;

/// Path to the grayscale test image
const GRAY_TEST_IMAGE: &str = "internal/jpegli-cpp/testdata/jxl/flower/flower.png.im_q85_gray.jpg";

/// Path to UltraHDR test image (may not exist on all systems)
#[allow(dead_code)]
fn ultrahdr_test_image_path() -> std::path::PathBuf {
    zenjpeg_bench_utils::ultrahdr_test_image()
}

fn load_test_image(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

#[test]
fn test_grayscale_decode_basic() {
    let Some(data) = load_test_image(GRAY_TEST_IMAGE) else {
        eprintln!("Skipping test: {} not found", GRAY_TEST_IMAGE);
        return;
    };

    let decoder = Decoder::new().output_format(PixelFormat::Gray);
    let decoded = decoder
        .decode(&data, Unstoppable)
        .expect("decode grayscale image");

    println!(
        "Grayscale image: {}x{}, {} bytes",
        decoded.width(),
        decoded.height(),
        decoded.pixels_u8().unwrap().len()
    );

    // Verify dimensions make sense
    assert!(decoded.width() > 0);
    assert!(decoded.height() > 0);

    // Verify output size matches dimensions (1 byte per pixel for grayscale)
    let expected_size = (decoded.width() * decoded.height()) as usize;
    assert_eq!(
        decoded.pixels_u8().unwrap().len(),
        expected_size,
        "grayscale output should have 1 byte per pixel"
    );
}

#[test]
fn test_grayscale_decode_to_rgb() {
    let Some(data) = load_test_image(GRAY_TEST_IMAGE) else {
        eprintln!("Skipping test: {} not found", GRAY_TEST_IMAGE);
        return;
    };

    // Decode grayscale as RGB (should expand gray to R=G=B)
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let decoded = decoder
        .decode(&data, Unstoppable)
        .expect("decode grayscale as RGB");

    println!(
        "Grayscale→RGB: {}x{}, {} bytes",
        decoded.width(),
        decoded.height(),
        decoded.pixels_u8().unwrap().len()
    );

    // Verify output size (3 bytes per pixel for RGB)
    let expected_size = (decoded.width() * decoded.height() * 3) as usize;
    assert_eq!(
        decoded.pixels_u8().unwrap().len(),
        expected_size,
        "RGB output should have 3 bytes per pixel"
    );

    // Verify R=G=B for grayscale content
    let pixels = decoded.pixels_u8().unwrap();
    for chunk in pixels.chunks_exact(3).take(100) {
        assert_eq!(chunk[0], chunk[1], "R should equal G for grayscale content");
        assert_eq!(chunk[1], chunk[2], "G should equal B for grayscale content");
    }
}

#[test]
#[ignore] // Grayscale scanline reading not yet supported
fn test_grayscale_scanline_reader() {
    let Some(data) = load_test_image(GRAY_TEST_IMAGE) else {
        eprintln!("Skipping test: {} not found", GRAY_TEST_IMAGE);
        return;
    };

    // Try scanline reader
    let decoder = Decoder::new();
    let result = decoder.scanline_reader(&data);

    match result {
        Ok(mut reader) => {
            println!("Scanline reader: {}x{}", reader.width(), reader.height());

            let width = reader.width() as usize;
            let height = reader.height() as usize;
            let mut pixels = vec![0u8; width * height];

            let mut rows_read = 0;
            while rows_read < height {
                let output =
                    ImgRefMut::new(&mut pixels[rows_read * width..], width, height - rows_read);
                // Note: scanline reader only supports RGB output currently
                let n = reader.read_rows_rgb8(output).expect("read rows");
                if n == 0 {
                    break;
                }
                rows_read += n;
            }

            assert_eq!(rows_read, height, "should read all rows");
        }
        Err(e) => {
            // Grayscale scanline reading may not be supported yet
            eprintln!("Scanline reader not supported for grayscale: {}", e);
        }
    }
}

#[test]
#[cfg(feature = "ultrahdr")]
fn test_ultrahdr_gainmap_extraction() {
    let ultrahdr_path = ultrahdr_test_image_path();
    let Some(data) = load_test_image(ultrahdr_path.to_str().unwrap()) else {
        eprintln!("Skipping test: {} not found", ultrahdr_path.display());
        return;
    };

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&data, Unstoppable)
        .expect("decode UltraHDR image");

    println!(
        "Primary image: {}x{}, format: {:?}",
        decoded.width(),
        decoded.height(),
        decoded.format
    );

    let extras = decoded.extras().expect("UltraHDR should have extras");

    println!("Is UltraHDR: {}", extras.is_ultrahdr());
    println!("Has XMP: {}", extras.xmp().is_some());
    println!("Secondary images: {}", extras.secondary_images().len());

    if let Some(xmp) = extras.xmp() {
        println!("XMP length: {} bytes", xmp.len());
        if xmp.contains("hdrgm:") {
            println!("XMP contains hdrgm namespace (UltraHDR metadata)");
        }
    }

    if extras.is_ultrahdr() {
        // Extract and decode the gain map
        if let Some(gainmap_jpeg) = extras.gainmap() {
            println!("Gain map JPEG: {} bytes", gainmap_jpeg.len());

            // Decode the gain map
            let gm_decoded = Decoder::new()
                .output_format(PixelFormat::Gray)
                .decode(gainmap_jpeg, Unstoppable)
                .expect("decode gain map");

            println!(
                "Gain map decoded: {}x{}, {} bytes",
                gm_decoded.width(),
                gm_decoded.height(),
                gm_decoded.pixels_u8().unwrap().len()
            );

            // Gain maps are typically grayscale or RGB
            assert!(gm_decoded.width() > 0);
            assert!(gm_decoded.height() > 0);
        } else {
            println!("No gain map found in secondary images");
        }
    }
}

#[test]
#[cfg(feature = "ultrahdr")]
fn test_gainmap_grayscale_decode_streaming() {
    let ultrahdr_path = ultrahdr_test_image_path();
    let Some(data) = load_test_image(ultrahdr_path.to_str().unwrap()) else {
        eprintln!("Skipping test: {} not found", ultrahdr_path.display());
        return;
    };

    let decoded = Decoder::new().decode(&data, Unstoppable).expect("decode");
    let extras = match decoded.extras() {
        Some(e) if e.is_ultrahdr() => e,
        _ => {
            eprintln!("Skipping: not an UltraHDR image");
            return;
        }
    };

    let gainmap_jpeg = match extras.gainmap() {
        Some(gm) => gm,
        None => {
            eprintln!("Skipping: no gain map found");
            return;
        }
    };

    println!(
        "Testing gain map streaming decode ({} bytes)",
        gainmap_jpeg.len()
    );

    // Try scanline reader on the gain map
    let decoder = Decoder::new();
    match decoder.scanline_reader(gainmap_jpeg) {
        Ok(mut reader) => {
            println!(
                "Gain map scanline reader: {}x{}",
                reader.width(),
                reader.height()
            );

            // Note: scanline reader requires 3-component YCbCr, so grayscale may fail
            let width = reader.width() as usize;
            let height = reader.height() as usize;
            let mut pixels = vec![0u8; width * height * 3];

            let mut rows_read = 0;
            while rows_read < height {
                let output = ImgRefMut::new(
                    &mut pixels[rows_read * width * 3..],
                    width * 3,
                    height - rows_read,
                );
                match reader.read_rows_rgb8(output) {
                    Ok(n) if n > 0 => rows_read += n,
                    Ok(_) => break,
                    Err(e) => {
                        eprintln!("Streaming decode error at row {}: {}", rows_read, e);
                        break;
                    }
                }
            }

            println!("Streaming decode read {} of {} rows", rows_read, height);
        }
        Err(e) => {
            // Expected: scanline reader requires 3-component image
            println!(
                "Scanline reader not supported for gain map (expected): {}",
                e
            );
        }
    }

    // Non-streaming decode should always work
    let gm_decoded = Decoder::new()
        .output_format(PixelFormat::Gray)
        .decode(gainmap_jpeg, Unstoppable)
        .expect("non-streaming decode should work");

    println!(
        "Non-streaming decode: {}x{} grayscale",
        gm_decoded.width(),
        gm_decoded.height()
    );
}
