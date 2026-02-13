//! Characterize the effective resampling filter of zenjpeg's shrink-on-load
//! using resamplescope's dot pattern analysis.
//!
//! For each DctScale (Half, Quarter, Eighth), we:
//! 1. Generate resamplescope's standard dot pattern (557×275)
//! 2. NN-upscale to dimensions that produce exactly 555×275 at that scale
//! 3. Encode as high-quality JPEG
//! 4. Decode at the target DctScale
//! 5. Analyze the output with resamplescope's dot filter reconstruction
//! 6. Score against known filters and render scope graphs

use enough::Unstoppable;
use imgref::ImgVec;
use resamplescope::pattern::{
    generate_dot_pattern, DOT_DST_HEIGHT, DOT_DST_WIDTH, DOT_SRC_HEIGHT, DOT_SRC_WIDTH,
};
use zenjpeg::decoder::{DctScale, Decoder, ShrinkHint};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

const OUTPUT_DIR: &str = "/mnt/v/output/zenjpeg/resamplescope";

/// Nearest-neighbor upscale for grayscale. Each source pixel maps to the
/// nearest output pixel — no filtering, just pixel repetition.
fn nn_upscale_gray(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh];
    for y in 0..dh {
        let sy = ((y as f64 + 0.5) * sh as f64 / dh as f64) as usize;
        let sy = sy.min(sh - 1);
        for x in 0..dw {
            let sx = ((x as f64 + 0.5) * sw as f64 / dw as f64) as usize;
            let sx = sx.min(sw - 1);
            dst[y * dw + x] = src[sy * sw + sx];
        }
    }
    dst
}

/// Convert grayscale to packed RGB (R=G=B=gray).
fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(gray.len() * 3);
    for &g in gray {
        rgb.extend_from_slice(&[g, g, g]);
    }
    rgb
}

/// Encode RGB pixels as JPEG.
fn encode_jpeg(pixels: &[u8], w: u32, h: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// Compute source width that produces exactly `target` at given DctScale.
/// DctScale::scaled_dimension(W) = (W * numerator + 7) / 8
fn source_dim_for_target(target: usize, scale: DctScale) -> u32 {
    let num = scale.numerator() as usize;
    // (W * num + 7) / 8 = target
    // W * num = target * 8 - 7 (approximately, need to search)
    // Start from the approximate value and adjust
    let approx = (target * 8 + num - 1) / num;
    // Search nearby for exact match
    for w in (approx.saturating_sub(2))..=(approx + 2) {
        if scale.scaled_dimension(w as u32) == target as u32 {
            return w as u32;
        }
    }
    panic!(
        "Cannot find source dimension for target={target} at scale={scale}: tried {}-{}",
        approx.saturating_sub(2),
        approx + 2
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn save_scope_png(img: &ImgVec<rgb::RGB8>, path: &str) {
    let file = std::fs::File::create(path).unwrap();
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, img.width() as u32, img.height() as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let data: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    writer.write_image_data(&data).unwrap();
}

/// Run resamplescope dot pattern analysis for a given DctScale.
fn analyze_scale(scale: DctScale, quality: f32) -> resamplescope::analyze::FilterCurve {
    // Compute JPEG source dimensions that produce exactly 555×275 at this scale
    let jpeg_w = source_dim_for_target(DOT_DST_WIDTH, scale);
    let jpeg_h = source_dim_for_target(DOT_DST_HEIGHT, scale);

    // Verify the dimensions
    assert_eq!(scale.scaled_dimension(jpeg_w), DOT_DST_WIDTH as u32);
    assert_eq!(scale.scaled_dimension(jpeg_h), DOT_DST_HEIGHT as u32);

    let upscale_ratio = jpeg_w as f64 / DOT_SRC_WIDTH as f64;
    eprintln!(
        "\n=== {scale} scale ===\nJPEG source: {jpeg_w}×{jpeg_h} (upscale ratio: {upscale_ratio:.3}x)"
    );

    // Generate standard dot pattern
    let dot = generate_dot_pattern();

    // NN upscale to JPEG source dimensions
    let upscaled = nn_upscale_gray(
        dot.buf(),
        DOT_SRC_WIDTH,
        DOT_SRC_HEIGHT,
        jpeg_w as usize,
        jpeg_h as usize,
    );
    let rgb = gray_to_rgb(&upscaled);

    // Encode as JPEG
    let jpeg = encode_jpeg(&rgb, jpeg_w, jpeg_h, quality);
    eprintln!("JPEG size: {} bytes (Q{quality})", jpeg.len());

    // Decode at target scale
    let result = Decoder::new()
        .shrink(ShrinkHint::ExactScale(scale))
        .decode(&jpeg, Unstoppable)
        .unwrap();

    let pixels = result.pixels_u8().unwrap();
    let hw = result.width() as usize;
    let hh = result.height() as usize;
    eprintln!("Decoded output: {hw}×{hh}");

    assert_eq!(
        hw, DOT_DST_WIDTH,
        "Width mismatch: expected {DOT_DST_WIDTH}, got {hw}"
    );
    assert_eq!(
        hh, DOT_DST_HEIGHT,
        "Height mismatch: expected {DOT_DST_HEIGHT}, got {hh}"
    );

    // Extract R channel as grayscale
    let gray: Vec<u8> = (0..hw * hh).map(|i| pixels[i * 3]).collect();
    let gray_img = ImgVec::new(gray, hw, hh);

    // Analyze with resamplescope
    let curve = resamplescope::analyze::analyze_dot(&gray_img.as_ref(), false);
    eprintln!(
        "Filter curve: {} points, scale_factor={:.4}",
        curve.points.len(),
        curve.scale_factor
    );

    // Score against known filters
    let scores = resamplescope::score::score_against_all(&curve);
    eprintln!("\nTop filter matches:");
    for (i, s) in scores.iter().take(5).enumerate() {
        eprintln!("  {}: {s}", i + 1);
    }

    curve
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resamplescope_shrink_half() {
    let curve = analyze_scale(DctScale::Half, 99.0);

    let scores = resamplescope::score::score_against_all(&curve);

    // Render scope graph with best-match reference overlay
    let best_filter = scores.first().map(|s| s.filter);
    let graph = resamplescope::graph::render(Some(&curve), None, best_filter);

    std::fs::create_dir_all(OUTPUT_DIR).unwrap();
    save_scope_png(&graph, &format!("{OUTPUT_DIR}/half_scale.png"));
    eprintln!("\nScope graph: {OUTPUT_DIR}/half_scale.png");

    // Also render with common reference filters for comparison
    for filter in [
        resamplescope::filters::KnownFilter::Box,
        resamplescope::filters::KnownFilter::Triangle,
        resamplescope::filters::KnownFilter::Lanczos2,
    ] {
        let graph = resamplescope::graph::render(Some(&curve), None, Some(filter));
        let name = filter.name().to_lowercase().replace('-', "_");
        save_scope_png(&graph, &format!("{OUTPUT_DIR}/half_vs_{name}.png"));
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resamplescope_shrink_quarter() {
    let curve = analyze_scale(DctScale::Quarter, 99.0);

    let scores = resamplescope::score::score_against_all(&curve);
    let best_filter = scores.first().map(|s| s.filter);
    let graph = resamplescope::graph::render(Some(&curve), None, best_filter);

    std::fs::create_dir_all(OUTPUT_DIR).unwrap();
    save_scope_png(&graph, &format!("{OUTPUT_DIR}/quarter_scale.png"));
    eprintln!("\nScope graph: {OUTPUT_DIR}/quarter_scale.png");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resamplescope_shrink_eighth() {
    let curve = analyze_scale(DctScale::Eighth, 99.0);

    let scores = resamplescope::score::score_against_all(&curve);
    let best_filter = scores.first().map(|s| s.filter);
    let graph = resamplescope::graph::render(Some(&curve), None, best_filter);

    std::fs::create_dir_all(OUTPUT_DIR).unwrap();
    save_scope_png(&graph, &format!("{OUTPUT_DIR}/eighth_scale.png"));
    eprintln!("\nScope graph: {OUTPUT_DIR}/eighth_scale.png");
}

/// Compare JPEG shrink at multiple quality levels to see how JPEG
/// compression quality affects the effective filter kernel.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resamplescope_shrink_half_quality_sweep() {
    std::fs::create_dir_all(OUTPUT_DIR).unwrap();

    for quality in [75.0, 85.0, 95.0, 99.0] {
        let curve = analyze_scale(DctScale::Half, quality);
        let scores = resamplescope::score::score_against_all(&curve);
        let best_filter = scores.first().map(|s| s.filter);
        let graph = resamplescope::graph::render(Some(&curve), None, best_filter);
        save_scope_png(
            &graph,
            &format!("{OUTPUT_DIR}/half_q{}.png", quality as u32),
        );
    }
}
