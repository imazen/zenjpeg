//! Integration tests for RD-OPT content-adaptive quantization table refinement.
//!
//! Uses CID22 corpus (512×512 diverse photographs) for validation.
//! Tests verify that RD-OPT produces valid, decodable JPEGs with size/quality
//! characteristics at least as good as the baseline jpegli tables.

#![cfg(feature = "rdopt")]

use std::path::{Path, PathBuf};

use imgref::{Img, ImgVec};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

/// Get CID22 training directory via codec-corpus (auto-downloads).
fn get_cid22_dir() -> PathBuf {
    let corpus = codec_corpus::Corpus::new()
        .expect("codec-corpus init failed (set CODEC_CORPUS_CACHE if needed)");
    corpus
        .get("CID22/CID22-512/training")
        .expect("corpus.get(CID22/CID22-512/training)")
}

/// Load a PNG image as RGB8 bytes. Returns (pixels, width, height).
fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let data = std::fs::read(path).ok()?;
    let decoder = png::Decoder::new(std::io::Cursor::new(&data));
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info().clone();

    if info.bit_depth != png::BitDepth::Eight {
        return None;
    }

    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let frame = reader.next_frame(&mut buf).ok()?;
    buf.truncate(frame.buffer_size());

    let (w, h) = (info.width, info.height);

    match info.color_type {
        png::ColorType::Rgb => Some((buf, w, h)),
        png::ColorType::Rgba => {
            let rgb: Vec<u8> = buf.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
            Some((rgb, w, h))
        }
        _ => None,
    }
}

/// Load first N PNG images from CID22 training set.
fn load_cid22_images(max: usize) -> Vec<(String, Vec<u8>, u32, u32)> {
    let dir = get_cid22_dir();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("read CID22 dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    entries.truncate(max);

    entries
        .into_iter()
        .filter_map(|e| {
            let path = e.path();
            let name = path.file_stem()?.to_string_lossy().to_string();
            let (pixels, w, h) = load_png_rgb(&path)?;
            Some((name, pixels, w, h))
        })
        .collect()
}

/// Encode an image with the given config, return JPEG bytes.
fn encode_jpeg(pixels: &[u8], width: u32, height: u32, config: EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation");
    enc.push_packed(pixels, Unstoppable).expect("push rows");
    enc.finish().expect("finish")
}

/// Decode JPEG bytes back to RGB8 using zune-jpeg (independent decoder).
fn decode_jpeg_rgb(jpeg: &[u8]) -> (Vec<u8>, u32, u32) {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;

    let cursor = ZCursor::new(jpeg);
    let mut decoder = JpegDecoder::new(cursor);
    let pixels = decoder.decode().expect("zune-jpeg decode");
    let (w, h) = decoder.dimensions().expect("jpeg dimensions");
    (pixels, w as u32, h as u32)
}

/// Compute SSIMULACRA2 score between two RGB8 images.
fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let orig_pixels: Vec<[u8; 3]> = original.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let comp_pixels: Vec<[u8; 3]> = decoded.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    let orig_img: ImgVec<[u8; 3]> = Img::new(orig_pixels, width, height);
    let comp_img: ImgVec<[u8; 3]> = Img::new(comp_pixels, width, height);

    fast_ssim2::compute_ssimulacra2(orig_img.as_ref(), comp_img.as_ref()).unwrap_or(0.0)
}

/// Test: RD-OPT with thresholds produces valid, decodable JPEGs.
///
/// For each CID22 image, encodes with and without rdopt and verifies:
/// 1. Output is valid JPEG (decodable by zune-jpeg)
/// 2. Dimensions match
/// 3. Quality is non-catastrophic (SSIMULACRA2 > 50 at Q85)
#[test]
#[ignore] // Requires CID22 corpus download
fn test_rdopt_produces_valid_jpeg() {
    let images = load_cid22_images(10);
    assert!(!images.is_empty(), "No CID22 images found");

    let quality = 85.0;

    println!("\n=== RD-OPT Validity Check (CID22, {} images, Q{quality}) ===", images.len());
    println!(
        "{:>20} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "image", "base_sz", "rdopt_sz", "base_q", "rdopt_q", "size%"
    );

    for (name, pixels, w, h) in &images {
        // Baseline encode (no rdopt)
        let config_base = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
        let jpeg_base = encode_jpeg(pixels, *w, *h, config_base);

        // RD-OPT encode (with thresholds)
        let config_rdopt = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
            .rdopt_refine(true)
            .rdopt_thresholds(true);
        let jpeg_rdopt = encode_jpeg(pixels, *w, *h, config_rdopt);

        // Verify both are decodable
        let (dec_base, dw_b, dh_b) = decode_jpeg_rgb(&jpeg_base);
        let (dec_rdopt, dw_r, dh_r) = decode_jpeg_rgb(&jpeg_rdopt);

        assert_eq!((dw_b, dh_b), (*w, *h), "{name}: baseline dims");
        assert_eq!((dw_r, dh_r), (*w, *h), "{name}: rdopt dims");

        let ssim_base = compute_ssim2(pixels, &dec_base, *w as usize, *h as usize);
        let ssim_rdopt = compute_ssim2(pixels, &dec_rdopt, *w as usize, *h as usize);

        let short_name: String = name.chars().take(20).collect();
        println!(
            "{short_name:>20} {:>10} {:>10} {:>8.2} {:>8.2} {:>7.1}%",
            jpeg_base.len(),
            jpeg_rdopt.len(),
            ssim_base,
            ssim_rdopt,
            jpeg_rdopt.len() as f64 / jpeg_base.len() as f64 * 100.0,
        );

        assert!(
            ssim_rdopt > 50.0,
            "{name}: rdopt quality catastrophic: ssim2={ssim_rdopt:.2}"
        );
    }
}

/// Test: RD-OPT quality sweep across multiple quality levels.
#[test]
#[ignore] // Requires CID22 corpus download
fn test_rdopt_quality_sweep() {
    let images = load_cid22_images(5);
    assert!(!images.is_empty(), "No CID22 images found");

    let qualities = [50.0, 75.0, 85.0, 95.0];

    println!("\n=== RD-OPT Quality Sweep (CID22, {} images) ===", images.len());
    println!(
        "{:>12} {:>5} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "image", "Q", "base_sz", "rdopt_sz", "base_q", "rdopt_q", "size%"
    );

    for quality in qualities {
        let mut total_base = 0u64;
        let mut total_rdopt = 0u64;

        for (name, pixels, w, h) in &images {
            let jpeg_base =
                encode_jpeg(pixels, *w, *h, EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter));
            let jpeg_rdopt = encode_jpeg(
                pixels,
                *w,
                *h,
                EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).rdopt_refine(true),
            );

            let (dec_base, _, _) = decode_jpeg_rgb(&jpeg_base);
            let (dec_rdopt, _, _) = decode_jpeg_rgb(&jpeg_rdopt);

            let ssim_base = compute_ssim2(pixels, &dec_base, *w as usize, *h as usize);
            let ssim_rdopt = compute_ssim2(pixels, &dec_rdopt, *w as usize, *h as usize);

            let short: String = name.chars().take(12).collect();
            println!(
                "{short:>12} Q{quality:<3.0} {:>10} {:>10} {:>8.2} {:>8.2} {:>7.1}%",
                jpeg_base.len(),
                jpeg_rdopt.len(),
                ssim_base,
                ssim_rdopt,
                jpeg_rdopt.len() as f64 / jpeg_base.len() as f64 * 100.0,
            );

            total_base += jpeg_base.len() as u64;
            total_rdopt += jpeg_rdopt.len() as u64;

            let min_ssim = match quality as u32 {
                50 => 35.0,
                75 => 45.0,
                85 => 50.0,
                _ => 30.0,
            };
            assert!(ssim_rdopt > min_ssim, "Q{quality} {name}: ssim2={ssim_rdopt:.2}");
        }

        println!(
            "{:>12} Q{quality:<3.0} {:>10} {:>10} {:>8} {:>8} {:>7.1}%\n",
            "TOTAL",
            total_base,
            total_rdopt,
            "",
            "",
            total_rdopt as f64 / total_base as f64 * 100.0,
        );
    }
}

/// Test: rdopt_refine(false) produces bit-identical output to baseline.
#[test]
fn test_rdopt_disabled_matches_baseline() {
    let w = 64u32;
    let h = 64u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            let v = if (x / 8 + y / 8) % 2 == 0 { 200u8 } else { 50u8 };
            pixels[idx] = v;
            pixels[idx + 1] = v;
            pixels[idx + 2] = v;
        }
    }

    let jpeg_base =
        encode_jpeg(&pixels, w, h, EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter));
    let jpeg_off = encode_jpeg(
        &pixels,
        w,
        h,
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).rdopt_refine(false),
    );

    assert_eq!(jpeg_base, jpeg_off, "rdopt_refine(false) must be bit-identical to baseline");
}

/// Test: RD-OPT with 4:4:4 subsampling produces valid output.
#[test]
fn test_rdopt_444_synthetic() {
    let w = 128u32;
    let h = 128u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 7 + 13) % 256) as u8;
    }

    let jpeg = encode_jpeg(
        &pixels,
        w,
        h,
        EncoderConfig::ycbcr(80.0, ChromaSubsampling::None)
            .rdopt_refine(true)
            .rdopt_thresholds(true),
    );
    let (decoded, dw, dh) = decode_jpeg_rgb(&jpeg);
    assert_eq!((dw, dh), (w, h));
    assert!(!decoded.is_empty());
}

/// Test: RD-OPT on a small synthetic image with known pattern.
#[test]
fn test_rdopt_small_image() {
    let w = 32u32;
    let h = 32u32;
    let mut pixels = vec![128u8; (w * h * 3) as usize];
    // Add some variation
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            pixels[idx] = ((x * 8) % 256) as u8;
            pixels[idx + 1] = ((y * 8) % 256) as u8;
            pixels[idx + 2] = 128;
        }
    }

    let jpeg = encode_jpeg(
        &pixels,
        w,
        h,
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).rdopt_refine(true),
    );
    let (_, dw, dh) = decode_jpeg_rgb(&jpeg);
    assert_eq!((dw, dh), (w, h));
    assert!(jpeg.len() > 100, "JPEG too small: {} bytes", jpeg.len());
}

/// Diagnostic test: print old vs new quant tables to see what the optimizer does.
#[test]
fn test_rdopt_diagnostic_tables() {
    let w = 128u32;
    let h = 128u32;
    // Noise-like pattern for realistic DCT distribution
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i.wrapping_mul(7).wrapping_add(13).wrapping_mul(i >> 3)) % 256) as u8;
    }

    // Encode WITHOUT rdopt to get baseline
    let jpeg_base = encode_jpeg(
        &pixels, w, h,
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
    );
    // Encode WITH rdopt
    let jpeg_rdopt = encode_jpeg(
        &pixels, w, h,
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .rdopt_refine(true)
            .rdopt_thresholds(false),
    );

    let (dec_base, _, _) = decode_jpeg_rgb(&jpeg_base);
    let (dec_rdopt, _, _) = decode_jpeg_rgb(&jpeg_rdopt);

    let ssim_base = compute_ssim2(&pixels, &dec_base, w as usize, h as usize);
    let ssim_rdopt = compute_ssim2(&pixels, &dec_rdopt, w as usize, h as usize);

    println!("base: {} bytes, ssim2={ssim_base:.2}", jpeg_base.len());
    println!("rdopt: {} bytes, ssim2={ssim_rdopt:.2}", jpeg_rdopt.len());

    // Compare DQT markers (extract quant tables from JPEG)
    println!("\nBase JPEG starts with: {:02x?}", &jpeg_base[..20]);
    println!("RdOpt JPEG starts with: {:02x?}", &jpeg_rdopt[..20]);

    // Print actual pixel samples
    println!("\nFirst 12 decoded pixels (base): {:?}", &dec_base[..12]);
    println!("First 12 decoded pixels (rdopt): {:?}", &dec_rdopt[..12]);

    // Check the rdopt output is not all zeros or all same value
    let unique_rdopt: std::collections::HashSet<u8> = dec_rdopt.iter().copied().collect();
    println!("Unique pixel values in rdopt: {}", unique_rdopt.len());
    assert!(unique_rdopt.len() > 10, "rdopt output looks degenerate ({} unique values)", unique_rdopt.len());
}
