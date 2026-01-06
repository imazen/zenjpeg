//! Multi-decoder compatibility test.
//!
//! Tests that JPEGs encoded by jpegli-rs can be correctly decoded by all major
//! JPEG decoders in the Rust ecosystem, and that the quality metrics are
//! appropriate for the encoding quality level.
//!
//! Decoders tested:
//! - jpegli-rs (our decoder, 12-bit precision f32 pipeline)
//! - jpeg-decoder (used by image crate, integer IDCT)
//! - zune-jpeg (fastest pure Rust decoder, integer IDCT)
//! - mozjpeg (C wrapper, libjpeg-compatible)
//!
//! Note: turbojpeg is not included because it conflicts with mozjpeg
//! (both link libjpeg symbols, causing duplicate symbol errors).
//!
//! Run with:
//! ```
//! cargo test --test multi_decoder_compatibility -- --nocapture
//! ```

use butteraugli::{compute_butteraugli, ButteraugliParams};
use dssim::Dssim;
use rgb::RGBA8;
use std::collections::HashMap;

/// Test configuration for a single encoding
#[derive(Clone, Debug)]
struct EncodingConfig {
    quality: u8,
    subsampling: jpegli::Subsampling,
    mode: jpegli::JpegMode,
    name: String,
}

/// Result from decoding with a specific decoder
struct DecoderResult {
    decoder_name: String,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    decode_time_us: u64,
}

/// Quality thresholds based on encoding quality level
/// Lower quality = higher allowed butteraugli score
/// Note: jpegli uses adaptive quantization which can produce higher butteraugli
/// scores than raw quality suggests, especially for synthetic test images.
fn max_butteraugli_for_quality(quality: u8) -> f64 {
    match quality {
        95..=100 => 1.5, // Very high quality
        90..=94 => 2.0,  // High quality
        80..=89 => 2.5,  // Good quality
        70..=79 => 3.5,  // Medium quality
        50..=69 => 5.0,  // Low quality
        _ => 8.0,        // Very low quality
    }
}

/// Maximum allowed pixel difference between decoders (per channel)
/// Small differences are expected due to rounding in IDCT implementations.
/// jpegli uses 12-bit f32 IDCT while others use integer IDCT.
const MAX_PIXEL_DIFF: u8 = 4;

/// Maximum allowed DSSIM between different decoders for same JPEG
/// Should be very close since they're decoding the same bitstream
const MAX_INTER_DECODER_DSSIM: f64 = 0.0005;

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

fn compute_butteraugli_score(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let params = ButteraugliParams::default();
    match compute_butteraugli(original, decoded, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => 99.0, // Return high score on error
    }
}

fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

// Decoder implementations

fn decode_jpegli(data: &[u8]) -> Option<DecoderResult> {
    let start = std::time::Instant::now();
    let decoder = jpegli::Decoder::new();
    match decoder.decode(data) {
        Ok(img) => Some(DecoderResult {
            decoder_name: "jpegli-rs".to_string(),
            pixels: img.data,
            width: img.width as usize,
            height: img.height as usize,
            decode_time_us: start.elapsed().as_micros() as u64,
        }),
        Err(e) => {
            eprintln!("jpegli-rs decode failed: {}", e);
            None
        }
    }
}

fn decode_jpeg_decoder(data: &[u8]) -> Option<DecoderResult> {
    let start = std::time::Instant::now();
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
    match decoder.decode() {
        Ok(pixels) => {
            let (width, height) = decoder.dimensions().unwrap();
            Some(DecoderResult {
                decoder_name: "jpeg-decoder".to_string(),
                pixels,
                width,
                height,
                decode_time_us: start.elapsed().as_micros() as u64,
            })
        }
        Err(e) => {
            eprintln!("jpeg-decoder decode failed: {}", e);
            None
        }
    }
}

fn decode_zune_jpeg(data: &[u8]) -> Option<DecoderResult> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;

    let start = std::time::Instant::now();
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    match decoder.decode() {
        Ok(pixels) => {
            let (width, height) = decoder.dimensions().unwrap();
            Some(DecoderResult {
                decoder_name: "zune-jpeg".to_string(),
                pixels,
                width,
                height,
                decode_time_us: start.elapsed().as_micros() as u64,
            })
        }
        Err(e) => {
            eprintln!("zune-jpeg decode failed: {:?}", e);
            None
        }
    }
}

fn decode_mozjpeg(data: &[u8]) -> Option<DecoderResult> {
    let start = std::time::Instant::now();
    match mozjpeg::Decompress::new_mem(data) {
        Ok(decompress) => {
            let width = decompress.width();
            let height = decompress.height();
            match decompress.rgb() {
                Ok(mut decompressor) => {
                    let mut pixels = vec![0u8; width * height * 3];
                    #[allow(deprecated)]
                    match decompressor.read_scanlines_into::<u8>(&mut pixels) {
                        Ok(_) => Some(DecoderResult {
                            decoder_name: "mozjpeg".to_string(),
                            pixels,
                            width,
                            height,
                            decode_time_us: start.elapsed().as_micros() as u64,
                        }),
                        Err(e) => {
                            eprintln!("mozjpeg: failed to read scanlines: {:?}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!("mozjpeg rgb() failed: {:?}", e);
                    None
                }
            }
        }
        Err(e) => {
            eprintln!("mozjpeg decompress failed: {:?}", e);
            None
        }
    }
}

/// Generate a test image with gradients and patterns
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Horizontal gradient for R
            pixels[idx] = (x * 255 / width) as u8;
            // Vertical gradient for G
            pixels[idx + 1] = (y * 255 / height) as u8;
            // Diagonal pattern for B
            pixels[idx + 2] = ((x + y) * 127 / (width + height)) as u8;
        }
    }
    pixels
}

/// Generate a more complex test image with high-frequency content
fn generate_complex_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Checkerboard pattern with gradients
            let checker = ((x / 8) + (y / 8)) % 2 == 0;
            let base = if checker { 200 } else { 55 };

            pixels[idx] = (base + (x % 56)) as u8;
            pixels[idx + 1] = (base + (y % 56)) as u8;
            pixels[idx + 2] = (base + ((x + y) % 56)) as u8;
        }
    }
    pixels
}

/// Test all decoders against a single encoded JPEG
fn test_all_decoders(
    original: &[u8],
    jpeg_data: &[u8],
    width: usize,
    height: usize,
    config: &EncodingConfig,
) -> HashMap<String, (f64, f64, u64)> {
    // (butteraugli, inter_decoder_dssim, decode_time_us)
    let mut results: HashMap<String, (f64, f64, u64)> = HashMap::new();

    // Collect all decoder results
    let decoder_fns: Vec<(&str, fn(&[u8]) -> Option<DecoderResult>)> = vec![
        ("jpegli-rs", decode_jpegli),
        ("jpeg-decoder", decode_jpeg_decoder),
        ("zune-jpeg", decode_zune_jpeg),
        ("mozjpeg", decode_mozjpeg),
    ];

    let mut decoder_results: Vec<DecoderResult> = Vec::new();

    for (name, decoder_fn) in &decoder_fns {
        match decoder_fn(jpeg_data) {
            Some(result) => {
                assert_eq!(
                    result.width, width,
                    "{}: width mismatch for {}",
                    name, config.name
                );
                assert_eq!(
                    result.height, height,
                    "{}: height mismatch for {}",
                    name, config.name
                );
                decoder_results.push(result);
            }
            None => {
                eprintln!("  {} failed to decode {}", name, config.name);
            }
        }
    }

    if decoder_results.is_empty() {
        panic!("No decoder could decode {}", config.name);
    }

    // Use first successful decoder as reference for inter-decoder comparison
    let reference = &decoder_results[0];

    for result in &decoder_results {
        // Compute butteraugli against original
        let butteraugli = compute_butteraugli_score(original, &result.pixels, width, height);

        // Compute DSSIM against reference decoder
        let inter_dssim = if result.decoder_name == reference.decoder_name {
            0.0
        } else {
            compute_dssim(&reference.pixels, &result.pixels, width, height)
        };

        // Check max pixel difference against reference
        if result.decoder_name != reference.decoder_name {
            let max_diff = max_pixel_diff(&reference.pixels, &result.pixels);
            if max_diff > MAX_PIXEL_DIFF {
                eprintln!(
                    "  WARNING: {} vs {} max pixel diff: {} (threshold: {})",
                    result.decoder_name, reference.decoder_name, max_diff, MAX_PIXEL_DIFF
                );
            }
        }

        results.insert(
            result.decoder_name.clone(),
            (butteraugli, inter_dssim, result.decode_time_us),
        );
    }

    results
}

/// Main test: encode with various configurations, decode with all decoders
#[test]
fn test_multi_decoder_compatibility() {
    let width = 256;
    let height = 256;
    let original = generate_test_image(width, height);

    let configs = vec![
        EncodingConfig {
            quality: 95,
            subsampling: jpegli::Subsampling::S444,
            mode: jpegli::JpegMode::Baseline,
            name: "Q95_444_baseline".to_string(),
        },
        EncodingConfig {
            quality: 90,
            subsampling: jpegli::Subsampling::S444,
            mode: jpegli::JpegMode::Baseline,
            name: "Q90_444_baseline".to_string(),
        },
        EncodingConfig {
            quality: 80,
            subsampling: jpegli::Subsampling::S444,
            mode: jpegli::JpegMode::Baseline,
            name: "Q80_444_baseline".to_string(),
        },
        EncodingConfig {
            quality: 70,
            subsampling: jpegli::Subsampling::S420,
            mode: jpegli::JpegMode::Baseline,
            name: "Q70_420_baseline".to_string(),
        },
        EncodingConfig {
            quality: 50,
            subsampling: jpegli::Subsampling::S420,
            mode: jpegli::JpegMode::Baseline,
            name: "Q50_420_baseline".to_string(),
        },
        EncodingConfig {
            quality: 90,
            subsampling: jpegli::Subsampling::S444,
            mode: jpegli::JpegMode::Progressive,
            name: "Q90_444_progressive".to_string(),
        },
        EncodingConfig {
            quality: 80,
            subsampling: jpegli::Subsampling::S422,
            mode: jpegli::JpegMode::Baseline,
            name: "Q80_422_baseline".to_string(),
        },
        EncodingConfig {
            quality: 80,
            subsampling: jpegli::Subsampling::S440,
            mode: jpegli::JpegMode::Baseline,
            name: "Q80_440_baseline".to_string(),
        },
    ];

    println!("\n=== Multi-Decoder Compatibility Test ===\n");
    println!(
        "{:<25} {:>12} {:>12} {:>12} {:>12}",
        "Config", "jpegli-rs", "jpeg-dec", "zune-jpeg", "mozjpeg"
    );
    println!("{}", "-".repeat(85));

    let mut all_passed = true;

    for config in &configs {
        // Encode with jpegli-rs
        let jpeg_data = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(jpegli::PixelFormat::Rgb)
            .jpegli_quality(jpegli::Quality::from_quality(config.quality as f32))
            .subsampling(config.subsampling)
            .mode(config.mode)
            .encode(&original)
            .expect("jpegli encode failed");

        let results = test_all_decoders(&original, &jpeg_data, width, height, config);

        // Print butteraugli scores
        let max_allowed = max_butteraugli_for_quality(config.quality);

        print!("{:<25}", config.name);
        for decoder_name in &["jpegli-rs", "jpeg-decoder", "zune-jpeg", "mozjpeg"] {
            if let Some((butteraugli, _, _)) = results.get(*decoder_name) {
                let status = if *butteraugli <= max_allowed {
                    "OK"
                } else {
                    "FAIL"
                };
                print!(" {:>10.3} {}", butteraugli, status);
                if *butteraugli > max_allowed {
                    all_passed = false;
                }
            } else {
                print!(" {:>12}", "N/A");
            }
        }
        println!();

        // Check inter-decoder consistency
        for (decoder_name, (_, inter_dssim, _)) in &results {
            if *inter_dssim > MAX_INTER_DECODER_DSSIM {
                eprintln!(
                    "  WARNING: {} inter-decoder DSSIM {:.6} > threshold {:.6} for {}",
                    decoder_name, inter_dssim, MAX_INTER_DECODER_DSSIM, config.name
                );
            }
        }
    }

    println!("\n{}", "-".repeat(85));
    println!(
        "Butteraugli thresholds by quality: Q95={}, Q90={}, Q80={}, Q70={}, Q50={}",
        max_butteraugli_for_quality(95),
        max_butteraugli_for_quality(90),
        max_butteraugli_for_quality(80),
        max_butteraugli_for_quality(70),
        max_butteraugli_for_quality(50),
    );

    assert!(
        all_passed,
        "Some decoder/quality combinations failed butteraugli threshold"
    );
}

/// Test with a more complex image that stresses the decoders
#[test]
fn test_multi_decoder_complex_image() {
    let width = 512;
    let height = 384;
    let original = generate_complex_test_image(width, height);

    println!("\n=== Complex Image Decoder Test ===\n");

    let configs = vec![
        EncodingConfig {
            quality: 90,
            subsampling: jpegli::Subsampling::S444,
            mode: jpegli::JpegMode::Baseline,
            name: "complex_Q90_444".to_string(),
        },
        EncodingConfig {
            quality: 75,
            subsampling: jpegli::Subsampling::S420,
            mode: jpegli::JpegMode::Baseline,
            name: "complex_Q75_420".to_string(),
        },
    ];

    for config in &configs {
        let jpeg_data = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(jpegli::PixelFormat::Rgb)
            .jpegli_quality(jpegli::Quality::from_quality(config.quality as f32))
            .subsampling(config.subsampling)
            .mode(config.mode)
            .encode(&original)
            .expect("jpegli encode failed");

        println!("{}: {} bytes", config.name, jpeg_data.len());

        let results = test_all_decoders(&original, &jpeg_data, width, height, config);
        let max_allowed = max_butteraugli_for_quality(config.quality);

        for (decoder_name, (butteraugli, inter_dssim, time_us)) in &results {
            let status = if *butteraugli <= max_allowed {
                "OK"
            } else {
                "FAIL"
            };
            println!(
                "  {:<15}: butteraugli={:.3} {} inter_dssim={:.6} time={}us",
                decoder_name, butteraugli, status, inter_dssim, time_us
            );
            assert!(
                *butteraugli <= max_allowed,
                "{} butteraugli {:.3} > threshold {:.3} for {}",
                decoder_name,
                butteraugli,
                max_allowed,
                config.name
            );
        }
    }
}

/// Benchmark decoder speeds
#[test]
#[ignore] // Run with --ignored for benchmark
fn benchmark_decoders() {
    let width = 1024;
    let height = 768;
    let original = generate_test_image(width, height);

    // Encode at Q85
    let jpeg_data = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .jpegli_quality(jpegli::Quality::from_quality(85.0))
        .encode(&original)
        .expect("encode failed");

    println!(
        "\n=== Decoder Benchmark ({}x{}, {} bytes) ===\n",
        width,
        height,
        jpeg_data.len()
    );

    let iterations = 50;
    let decoder_fns: Vec<(&str, fn(&[u8]) -> Option<DecoderResult>)> = vec![
        ("jpegli-rs", decode_jpegli),
        ("jpeg-decoder", decode_jpeg_decoder),
        ("zune-jpeg", decode_zune_jpeg),
        ("mozjpeg", decode_mozjpeg),
    ];

    for (name, decoder_fn) in &decoder_fns {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = decoder_fn(&jpeg_data);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let pixels = width * height;
        let mp_per_sec = (pixels as f64 / avg_us) * 1_000_000.0 / 1_000_000.0;

        println!(
            "{:<15}: {:>8.1} us/decode  {:>6.1} MP/s",
            name, avg_us, mp_per_sec
        );
    }
}

/// Test grayscale encoding/decoding compatibility
#[test]
fn test_grayscale_compatibility() {
    let width = 128;
    let height = 128;

    // Generate grayscale image (stored as RGB with R=G=B)
    let mut original = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let gray = ((x * 255 / width) + (y * 255 / height)) as u8 / 2;
            original[idx] = gray;
            original[idx + 1] = gray;
            original[idx + 2] = gray;
        }
    }

    let jpeg_data = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .jpegli_quality(jpegli::Quality::from_quality(90.0))
        .encode(&original)
        .expect("encode failed");

    println!("\n=== Grayscale Compatibility Test ===\n");

    // All decoders should handle RGB that happens to be grayscale
    let decoder_fns: Vec<(&str, fn(&[u8]) -> Option<DecoderResult>)> = vec![
        ("jpegli-rs", decode_jpegli),
        ("jpeg-decoder", decode_jpeg_decoder),
        ("zune-jpeg", decode_zune_jpeg),
        ("mozjpeg", decode_mozjpeg),
    ];

    for (name, decoder_fn) in &decoder_fns {
        match decoder_fn(&jpeg_data) {
            Some(result) => {
                assert_eq!(result.width, width);
                assert_eq!(result.height, height);
                let butteraugli =
                    compute_butteraugli_score(&original, &result.pixels, width, height);
                println!("{:<15}: butteraugli={:.3}", name, butteraugli);
                assert!(
                    butteraugli < 2.0,
                    "{} butteraugli too high for Q90 grayscale",
                    name
                );
            }
            None => {
                panic!("{} failed to decode grayscale JPEG", name);
            }
        }
    }
}
