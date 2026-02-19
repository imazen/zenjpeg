//! JPEG re-encoding calibration tool.
//!
//! Generates source JPEGs using libjpeg-turbo, mozjpeg, and cjpegli, then re-encodes
//! with zenjpeg at various quality levels. Measures quality degradation and size changes
//! to find optimal re-encoding parameters.
//!
//! Three modes:
//! - **Match**: Don't degrade quality, don't increase size (within tolerance)
//! - **Shrink**: Minimize size with configurable quality loss tolerance
//! - **Resize+re-encode**: How downscaling interacts with optimal quality selection
//!
//! Usage:
//! ```bash
//! # Default: gb82 corpus, 10 images
//! cargo run --release -p zenjpeg --example reencode_calibration --features trellis
//!
//! # Smoke test
//! cargo run --release -p zenjpeg --example reencode_calibration --features trellis -- --images 2
//!
//! # With resize experiments
//! cargo run --release -p zenjpeg --example reencode_calibration --features trellis -- --resize
//!
//! # Full sweep (more quality levels + 4:4:4 sources for turbo/mozjpeg)
//! cargo run --release -p zenjpeg --example reencode_calibration --features trellis -- --full-sweep
//! ```

use enough::Unstoppable;
use rayon::prelude::*;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use zenjpeg::detect;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout};
use zenjpeg_bench_utils::{
    bytes_to_rgb, decode_jpeg_to_rgb, decode_jpeg_with_icc, rgb_to_bytes, write_ppm, ImageData,
    QualityMetrics, RgbImage,
};

const DEFAULT_OUTPUT_DIR: &str = "/mnt/v/output/zenjpeg/reencode_calibration";

const SRC_QUALITIES: [u8; 10] = [10, 20, 30, 40, 50, 65, 75, 80, 85, 90];
const SRC_QUALITIES_FULL: [u8; 14] = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95];
const ZEN_QUALITIES: [f32; 19] = [
    20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 88.0,
    90.0, 93.0, 95.0, 97.0,
];
const RESIZE_QUALITIES: [f32; 8] = [55.0, 65.0, 75.0, 80.0, 85.0, 88.0, 90.0, 95.0];
const RESIZE_RATIOS: [f64; 4] = [1.5, 2.0, 3.0, 4.0];

/// Preset quality profiles for offset calibration.
/// Progressive variants used since BA delta is identical to baseline counterparts.
/// "auto" = auto_optimize(true) which is the calibrated baseline for the grids.
const OFFSET_PRESETS: &[(&str, Option<OptimizationPreset>)] = &[
    ("auto", None),
    ("jpegli", Some(OptimizationPreset::JpegliProgressive)),
    ("mozjpeg", Some(OptimizationPreset::MozjpegProgressive)),
    ("moz-max", Some(OptimizationPreset::MozjpegMaxCompression)),
    ("hybrid", Some(OptimizationPreset::HybridProgressive)),
    ("hyb-max", Some(OptimizationPreset::HybridMaxCompression)),
];

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

struct Args {
    corpus: PathBuf,
    output: PathBuf,
    max_images: usize,
    ba_tolerance: f64,
    shrink_tolerance: f64,
    size_tolerance: f64,
    full_sweep: bool,
    resize: bool,
    preset_offsets: bool,
    no_turbo: bool,
    no_mozjpeg: bool,
    no_cjpegli: bool,
    verbose: bool,
}

fn default_corpus_dir() -> PathBuf {
    codec_corpus::Corpus::new()
        .expect("codec-corpus unavailable")
        .get("gb82")
        .expect("gb82 corpus not available")
}

fn expand_tilde(s: &str) -> PathBuf {
    if s.starts_with('~') {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(&s[2..]);
        }
    }
    PathBuf::from(s)
}

fn parse_args() -> Args {
    let mut args = Args {
        corpus: default_corpus_dir(),
        output: PathBuf::from(DEFAULT_OUTPUT_DIR),
        max_images: 10,
        ba_tolerance: 0.0,
        shrink_tolerance: 0.5,
        size_tolerance: 0.02,
        full_sweep: false,
        resize: false,
        preset_offsets: false,
        no_turbo: false,
        no_mozjpeg: false,
        no_cjpegli: false,
        verbose: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--corpus" => {
                if let Some(s) = iter.next() {
                    args.corpus = expand_tilde(&s);
                }
            }
            "--output" => {
                if let Some(s) = iter.next() {
                    args.output = PathBuf::from(s);
                }
            }
            "--images" => {
                args.max_images = iter.next().and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--ba-tolerance" => {
                args.ba_tolerance = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            "--shrink-tolerance" => {
                args.shrink_tolerance = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
            }
            "--size-tolerance" => {
                args.size_tolerance = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0.02);
            }
            "--full-sweep" => args.full_sweep = true,
            "--resize" => args.resize = true,
            "--preset-offsets" => args.preset_offsets = true,
            "--no-turbo" => args.no_turbo = true,
            "--no-mozjpeg" => args.no_mozjpeg = true,
            "--no-cjpegli" => args.no_cjpegli = true,
            "--verbose" | "-v" => args.verbose = true,
            "--help" | "-h" => {
                eprintln!("Usage: reencode_calibration [OPTIONS]");
                eprintln!("  --corpus <dir>          Image directory (default: gb82)");
                eprintln!("  --output <dir>          Output dir (default: {DEFAULT_OUTPUT_DIR})");
                eprintln!("  --images <N>            Max images (default: 10)");
                eprintln!("  --ba-tolerance <f>      Match BA tolerance (default: 0.0)");
                eprintln!("  --shrink-tolerance <f>  Shrink BA tolerance (default: 0.5)");
                eprintln!("  --size-tolerance <f>    Size ratio tolerance (default: 0.02)");
                eprintln!("  --full-sweep            All quality+subsampling combos");
                eprintln!("  --resize                Enable Phase 3 resize experiments");
                eprintln!("  --preset-offsets        Measure quality offsets for all presets");
                eprintln!("  --no-turbo              Skip libjpeg-turbo");
                eprintln!("  --no-mozjpeg            Skip mozjpeg");
                eprintln!("  --no-cjpegli            Skip cjpegli");
                eprintln!("  --verbose               Per-image output");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }
    args
}

// ---------------------------------------------------------------------------
// Source JPEG encoding
// ---------------------------------------------------------------------------

fn check_binary(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Encode with libjpeg-turbo cjpeg CLI (4:2:0 default). Returns JPEG bytes.
fn encode_turbo(ppm_path: &Path, quality: u8) -> io::Result<Vec<u8>> {
    let output = Command::new("cjpeg")
        .arg("-quality")
        .arg(quality.to_string())
        .arg(ppm_path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("cjpeg failed: {}", String::from_utf8_lossy(&output.stderr)),
        ));
    }
    Ok(output.stdout)
}

/// Encode with libjpeg-turbo at 4:4:4 (for full-sweep).
fn encode_turbo_444(ppm_path: &Path, quality: u8) -> io::Result<Vec<u8>> {
    let output = Command::new("cjpeg")
        .arg("-quality")
        .arg(quality.to_string())
        .arg("-sample")
        .arg("1x1")
        .arg(ppm_path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "cjpeg 444 failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }
    Ok(output.stdout)
}

/// Encode with mozjpeg-rs (pure Rust, in-process).
fn encode_mozjpeg(
    pixels: &[u8],
    w: usize,
    h: usize,
    quality: u8,
    sub_444: bool,
) -> Option<Vec<u8>> {
    let sub = if sub_444 {
        mozjpeg_rs::Subsampling::S444
    } else {
        mozjpeg_rs::Subsampling::S420
    };
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(sub)
        .encode_rgb(pixels, w as u32, h as u32)
        .ok()
}

/// Encode with cjpegli CLI (4:4:4 default). Returns JPEG bytes.
fn encode_cjpegli(png_path: &Path, quality: u8, tmp_dir: &Path) -> io::Result<Vec<u8>> {
    let stem = png_path.file_stem().unwrap_or_default().to_string_lossy();
    let tmp_out = tmp_dir.join(format!("cjpegli_{stem}_q{quality}.jpg"));
    let output = Command::new("cjpegli")
        .arg(png_path)
        .arg(&tmp_out)
        .arg("-q")
        .arg(quality.to_string())
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "cjpegli failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }
    let data = std::fs::read(&tmp_out)?;
    std::fs::remove_file(&tmp_out).ok();
    Ok(data)
}

// ---------------------------------------------------------------------------
// Re-encoding with zenjpeg
// ---------------------------------------------------------------------------

fn encode_zen(
    pixels: &[u8],
    w: usize,
    h: usize,
    quality: f32,
    sub: ChromaSubsampling,
) -> Option<Vec<u8>> {
    encode_zen_preset(pixels, w, h, quality, sub, None)
}

fn encode_zen_preset(
    pixels: &[u8],
    w: usize,
    h: usize,
    quality: f32,
    sub: ChromaSubsampling,
    preset: Option<OptimizationPreset>,
) -> Option<Vec<u8>> {
    let config = EncoderConfig::ycbcr(quality, sub);
    let config = match preset {
        Some(p) => config.optimization(p),
        None => config.auto_optimize(true),
    };
    let mut e = config
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    e.push_packed(pixels, Unstoppable).ok()?;
    e.finish().ok()
}

// ---------------------------------------------------------------------------
// Resize (Phase 3)
// ---------------------------------------------------------------------------

fn resize_rgb(pixels: &[u8], w: u32, h: u32, out_w: u32, out_h: u32) -> Vec<u8> {
    // Simple area-average downscale (no zenresize dependency needed)
    let in_w = w as usize;
    let in_h = h as usize;
    let ow = out_w as usize;
    let oh = out_h as usize;
    let mut out = vec![0u8; ow * oh * 3];
    let x_ratio = in_w as f64 / ow as f64;
    let y_ratio = in_h as f64 / oh as f64;
    for oy in 0..oh {
        for ox in 0..ow {
            let src_y0 = (oy as f64 * y_ratio) as usize;
            let src_y1 = ((oy + 1) as f64 * y_ratio).ceil() as usize;
            let src_x0 = (ox as f64 * x_ratio) as usize;
            let src_x1 = ((ox + 1) as f64 * x_ratio).ceil() as usize;
            let mut r = 0u32;
            let mut g = 0u32;
            let mut b = 0u32;
            let mut count = 0u32;
            for sy in src_y0..src_y1.min(in_h) {
                for sx in src_x0..src_x1.min(in_w) {
                    let idx = (sy * in_w + sx) * 3;
                    r += pixels[idx] as u32;
                    g += pixels[idx + 1] as u32;
                    b += pixels[idx + 2] as u32;
                    count += 1;
                }
            }
            let oidx = (oy * ow + ox) * 3;
            out[oidx] = (r / count) as u8;
            out[oidx + 1] = (g / count) as u8;
            out[oidx + 2] = (b / count) as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

fn measure_rgb(reference: &RgbImage, distorted: &RgbImage) -> (f64, f64) {
    let ba = QualityMetrics::butteraugli(reference.as_ref(), distorted.as_ref());
    let ss2 = QualityMetrics::ssimulacra2(reference.as_ref(), distorted.as_ref());
    (ba, ss2)
}

fn measure_jpeg(reference: &RgbImage, jpeg: &[u8]) -> Option<(f64, f64)> {
    // Use zenjpeg's decoder — zune-jpeg fails on zenjpeg's progressive scan structure
    let dec = decode_jpeg_with_icc(jpeg).ok()?;
    Some(measure_rgb(reference, &dec))
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct RawResult {
    image: String,
    src_encoder: String,
    src_quality: u8,
    src_sub: String,
    src_ba: f64,
    src_ss2: f64,
    src_size: usize,
    resize_ratio: f64,
    zen_preset: String,
    zen_quality: f32,
    zen_sub: String,
    reenc_ba: f64,
    reenc_ss2: f64,
    reenc_size: usize,
    ba_delta: f64,
    ss2_delta: f64,
    size_ratio: f64,
}

struct SummaryRow {
    src_encoder: String,
    src_quality: u8,
    src_sub: String,
    mean_src_ba: f64,
    /// Recommended: lowest Q where ba_delta ≤ 0.3 (barely perceptible re-encoding loss)
    rec_zen_q: Option<f32>,
    rec_ba_delta: f64,
    rec_size_ratio: f64,
    /// Match: highest Q where ba_delta ≤ ba_tolerance AND size ≤ 1.0 + size_tolerance
    match_zen_q: Option<f32>,
    match_ba_delta: f64,
    match_size_ratio: f64,
    ci95_zen_q: Option<f32>,
    shrink_zen_q: Option<f32>,
    shrink_size_ratio: f64,
}

/// Quality ceiling for a given resize ratio.
struct ResizeCeiling {
    ratio: f64,
    /// The quality above which marginal BA improvement per % size is < threshold
    ceiling_q: Option<f32>,
    /// BA at ceiling quality
    ceiling_ba: f64,
    /// Size ratio at ceiling (vs unreized source)
    ceiling_size_ratio: f64,
    /// BA at the next quality above ceiling (showing diminishing returns)
    next_ba: f64,
    next_size_ratio: f64,
}

#[derive(Clone)]
struct SourceConfig {
    encoder: String,
    sub: String,
}

// ---------------------------------------------------------------------------
// Per-image processing
// ---------------------------------------------------------------------------

fn process_source(
    img_name: &str,
    reference: &RgbImage,
    source_jpeg: &[u8],
    src_encoder: &str,
    src_quality: u8,
    src_sub: &str,
    zen_qualities: &[f32],
    presets: &[(&str, Option<OptimizationPreset>)],
    verbose: bool,
) -> Vec<RawResult> {
    let mut results = Vec::new();

    // Validate with probe
    if verbose {
        if let Ok(probe) = detect::probe(source_jpeg) {
            eprintln!(
                "    {} Q{}: detected {:?} Q{:.0} {:?}",
                src_encoder, src_quality, probe.encoder, probe.quality.value, probe.subsampling
            );
        }
    }

    // Decode source JPEG
    let decoded = match decode_jpeg_to_rgb(source_jpeg) {
        Ok(d) => d,
        Err(e) => {
            if verbose {
                eprintln!("    decode failed: {e}");
            }
            return results;
        }
    };

    let (src_ba, src_ss2) = measure_rgb(reference, &decoded);
    let src_size = source_jpeg.len();
    let decoded_bytes = rgb_to_bytes(decoded.as_ref());
    let w = decoded.width();
    let h = decoded.height();

    // Determine which zenjpeg subsampling modes to try
    let zen_subs: Vec<(&str, ChromaSubsampling)> = if src_sub == "444" {
        vec![
            ("444", ChromaSubsampling::None),
            ("420", ChromaSubsampling::Quarter),
        ]
    } else {
        vec![("420", ChromaSubsampling::Quarter)]
    };

    for &(preset_name, preset) in presets {
        for &zen_q in zen_qualities {
            for &(zen_sub_str, zen_sub) in &zen_subs {
                if let Some(reenc) =
                    encode_zen_preset(&decoded_bytes, w, h, zen_q, zen_sub, preset)
                {
                    if let Some((reenc_ba, reenc_ss2)) = measure_jpeg(reference, &reenc) {
                        let reenc_size = reenc.len();
                        results.push(RawResult {
                            image: img_name.to_string(),
                            src_encoder: src_encoder.to_string(),
                            src_quality,
                            src_sub: src_sub.to_string(),
                            src_ba,
                            src_ss2,
                            src_size,
                            resize_ratio: 1.0,
                            zen_preset: preset_name.to_string(),
                            zen_quality: zen_q,
                            zen_sub: zen_sub_str.to_string(),
                            reenc_ba,
                            reenc_ss2,
                            reenc_size,
                            ba_delta: reenc_ba - src_ba,
                            ss2_delta: reenc_ss2 - src_ss2,
                            size_ratio: reenc_size as f64 / src_size as f64,
                        });
                    }
                }
            }
        }
    }

    results
}

fn process_resize(
    img: &ImageData,
    _reference: &RgbImage,
    source_jpeg: &[u8],
    src_encoder: &str,
    src_quality: u8,
    src_sub: &str,
) -> Vec<RawResult> {
    let mut results = Vec::new();

    let decoded = match decode_jpeg_to_rgb(source_jpeg) {
        Ok(d) => d,
        Err(_) => return results,
    };
    let decoded_bytes = rgb_to_bytes(decoded.as_ref());
    let src_size = source_jpeg.len();
    let (orig_w, orig_h) = (img.width as u32, img.height as u32);

    for &ratio in &RESIZE_RATIOS {
        let out_w = (orig_w as f64 / ratio).round() as u32;
        let out_h = (orig_h as f64 / ratio).round() as u32;
        if out_w < 8 || out_h < 8 {
            continue;
        }

        // Resize reference (original PNG → smaller)
        let resized_ref_bytes = resize_rgb(&img.pixels, orig_w, orig_h, out_w, out_h);
        let resized_ref = bytes_to_rgb(&resized_ref_bytes, out_w as usize, out_h as usize);

        // Resize decoded source JPEG → smaller
        let resized_dec_bytes = resize_rgb(
            &decoded_bytes,
            decoded.width() as u32,
            decoded.height() as u32,
            out_w,
            out_h,
        );

        // Source BA for resized: compare resized decoded vs resized reference
        let resized_dec = bytes_to_rgb(&resized_dec_bytes, out_w as usize, out_h as usize);
        let (rsrc_ba, rsrc_ss2) = measure_rgb(&resized_ref, &resized_dec);

        for &zen_q in &RESIZE_QUALITIES {
            if let Some(reenc) = encode_zen(
                &resized_dec_bytes,
                out_w as usize,
                out_h as usize,
                zen_q,
                ChromaSubsampling::Quarter,
            ) {
                if let Some((reenc_ba, reenc_ss2)) = measure_jpeg(&resized_ref, &reenc) {
                    let reenc_size = reenc.len();
                    results.push(RawResult {
                        image: img.name.clone(),
                        src_encoder: src_encoder.to_string(),
                        src_quality,
                        src_sub: src_sub.to_string(),
                        src_ba: rsrc_ba,
                        src_ss2: rsrc_ss2,
                        src_size,
                        resize_ratio: ratio,
                        zen_preset: "auto".to_string(),
                        zen_quality: zen_q,
                        zen_sub: "420".to_string(),
                        reenc_ba,
                        reenc_ss2,
                        reenc_size,
                        ba_delta: reenc_ba - rsrc_ba,
                        ss2_delta: reenc_ss2 - rsrc_ss2,
                        size_ratio: reenc_size as f64 / src_size as f64,
                    });
                }
            }
        }
    }

    results
}

fn process_image(
    img: &ImageData,
    png_path: &Path,
    tmp_dir: &Path,
    source_configs: &[SourceConfig],
    src_qualities: &[u8],
    presets: &[(&str, Option<OptimizationPreset>)],
    args: &Args,
) -> Vec<RawResult> {
    let reference = bytes_to_rgb(&img.pixels, img.width, img.height);
    let mut results = Vec::new();

    // Sanity check: encode original PNG directly with zenjpeg at Q90
    if args.verbose {
        if let Some(direct) = encode_zen(
            &img.pixels,
            img.width,
            img.height,
            90.0,
            ChromaSubsampling::Quarter,
        ) {
            if let Some((ba, ss2)) = measure_jpeg(&reference, &direct) {
                eprintln!(
                    "  [sanity] {} direct zen Q90: BA={:.2}, SS2={:.2}, size={}",
                    img.name,
                    ba,
                    ss2,
                    direct.len()
                );
            }
        }
    }

    // Write PPM for turbo (once per image)
    let ppm_path = tmp_dir.join(format!("{}.ppm", img.name));
    let has_turbo = source_configs.iter().any(|c| c.encoder == "turbo");
    if has_turbo {
        if let Err(e) = write_ppm(&ppm_path, reference.as_ref()) {
            eprintln!("  warning: cannot write PPM for {}: {e}", img.name);
        }
    }

    for src_cfg in source_configs {
        for &sq in src_qualities {
            // Generate source JPEG
            let source_jpeg = match (src_cfg.encoder.as_str(), src_cfg.sub.as_str()) {
                ("turbo", "420") => encode_turbo(&ppm_path, sq).ok(),
                ("turbo", "444") => encode_turbo_444(&ppm_path, sq).ok(),
                ("mozjpeg", sub) => {
                    encode_mozjpeg(&img.pixels, img.width, img.height, sq, sub == "444")
                }
                ("cjpegli", _) => encode_cjpegli(png_path, sq, tmp_dir).ok(),
                _ => None,
            };

            let source_jpeg = match source_jpeg {
                Some(j) if !j.is_empty() => j,
                _ => continue,
            };

            // Phase 2: Re-encoding sweep (all presets)
            results.extend(process_source(
                &img.name,
                &reference,
                &source_jpeg,
                &src_cfg.encoder,
                sq,
                &src_cfg.sub,
                &ZEN_QUALITIES,
                presets,
                args.verbose,
            ));

            // Phase 3: Resize experiments
            if args.resize {
                results.extend(process_resize(
                    img,
                    &reference,
                    &source_jpeg,
                    &src_cfg.encoder,
                    sq,
                    &src_cfg.sub,
                ));
            }
        }
    }

    // Cleanup PPM
    if has_turbo {
        std::fs::remove_file(&ppm_path).ok();
    }

    results
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

fn compute_summary(results: &[RawResult], args: &Args) -> Vec<SummaryRow> {
    // Only non-resize, auto-preset results (preserves existing behavior)
    let non_resize: Vec<&RawResult> = results
        .iter()
        .filter(|r| r.resize_ratio == 1.0 && r.zen_preset == "auto")
        .collect();

    // Group by (src_encoder, src_quality, src_sub)
    let mut groups: std::collections::BTreeMap<(String, u8, String), Vec<&RawResult>> =
        std::collections::BTreeMap::new();
    for r in &non_resize {
        groups
            .entry((r.src_encoder.clone(), r.src_quality, r.src_sub.clone()))
            .or_default()
            .push(r);
    }

    let mut summary = Vec::new();
    for ((enc, sq, sub), group) in &groups {
        // Compute mean source BA (deduplicate by image — each image has same src_ba for this group)
        let mut seen_images: Vec<&str> = Vec::new();
        let mut src_ba_sum = 0.0;
        for r in group.iter() {
            if !seen_images.contains(&r.image.as_str()) {
                seen_images.push(&r.image);
                src_ba_sum += r.src_ba;
            }
        }
        let mean_src_ba = if seen_images.is_empty() {
            0.0
        } else {
            src_ba_sum / seen_images.len() as f64
        };

        let mut rec_zen_q = None;
        let mut rec_ba_delta = 0.0;
        let mut rec_size_ratio = 0.0;
        let mut match_zen_q = None;
        let mut match_ba_delta = 0.0;
        let mut match_size_ratio = 0.0;
        let mut ci95_zen_q = None;
        let mut shrink_zen_q = None;
        let mut shrink_size_ratio = 0.0;

        // Collect (zen_q → mean_ba_delta, mean_size_ratio) for matching subsampling
        let mut q_stats: Vec<(f32, f64, f64)> = Vec::new();
        for &zq in &ZEN_QUALITIES {
            let matching: Vec<&&RawResult> = group
                .iter()
                .filter(|r| r.zen_quality == zq && r.zen_sub == *sub)
                .collect();
            if matching.is_empty() {
                continue;
            }
            let n = matching.len() as f64;
            let mean_bd = matching.iter().map(|r| r.ba_delta).sum::<f64>() / n;
            let mean_sr = matching.iter().map(|r| r.size_ratio).sum::<f64>() / n;
            q_stats.push((zq, mean_bd, mean_sr));

            // 95% CI (scan high→low later)
            let passing = matching
                .iter()
                .filter(|r| {
                    r.ba_delta <= args.ba_tolerance && r.size_ratio <= 1.0 + args.size_tolerance
                })
                .count();
            if ci95_zen_q.is_none() && passing as f64 / n >= 0.95 {
                ci95_zen_q = Some(zq);
            }
        }

        // Recommended: lowest Q where ba_delta ≤ 0.3 (barely perceptible)
        for &(zq, bd, sr) in &q_stats {
            if bd <= 0.3 {
                rec_zen_q = Some(zq);
                rec_ba_delta = bd;
                rec_size_ratio = sr;
                break;
            }
        }

        // Match: highest Q where ba_delta ≤ ba_tolerance AND size ≤ 1.0+tol
        for &(zq, bd, sr) in q_stats.iter().rev() {
            if match_zen_q.is_none() && bd <= args.ba_tolerance && sr <= 1.0 + args.size_tolerance {
                match_zen_q = Some(zq);
                match_ba_delta = bd;
                match_size_ratio = sr;
            }
        }

        // Shrink: lowest Q where ba_delta ≤ shrink_tolerance
        for &(zq, bd, sr) in &q_stats {
            if bd <= args.shrink_tolerance {
                shrink_zen_q = Some(zq);
                shrink_size_ratio = sr;
                break;
            }
        }

        summary.push(SummaryRow {
            src_encoder: enc.clone(),
            src_quality: *sq,
            src_sub: sub.clone(),
            mean_src_ba,
            rec_zen_q,
            rec_ba_delta,
            rec_size_ratio,
            match_zen_q,
            match_ba_delta,
            match_size_ratio,
            ci95_zen_q,
            shrink_zen_q,
            shrink_size_ratio,
        });
    }

    summary
}

/// Compute quality ceilings for each resize ratio.
///
/// For each ratio, finds the quality above which you're wasting bytes:
/// the marginal BA improvement per additional % file size drops below
/// a threshold (0.01 BA per 1% size increase).
fn compute_resize_ceilings(results: &[RawResult]) -> Vec<ResizeCeiling> {
    let resize_results: Vec<&RawResult> = results.iter().filter(|r| r.resize_ratio > 1.0).collect();
    let mut ceilings = Vec::new();

    for &ratio in &RESIZE_RATIOS {
        let at_ratio: Vec<&&RawResult> = resize_results
            .iter()
            .filter(|r| (r.resize_ratio - ratio).abs() < 0.01)
            .collect();
        if at_ratio.is_empty() {
            continue;
        }

        // Compute (zen_q, mean_ba, mean_size_ratio) sorted by quality
        let mut q_points: Vec<(f32, f64, f64)> = Vec::new();
        for &zq in &RESIZE_QUALITIES {
            let at_q: Vec<&&&RawResult> = at_ratio.iter().filter(|r| r.zen_quality == zq).collect();
            if at_q.is_empty() {
                continue;
            }
            let n = at_q.len() as f64;
            let mean_ba = at_q.iter().map(|r| r.reenc_ba).sum::<f64>() / n;
            let mean_sr = at_q.iter().map(|r| r.size_ratio).sum::<f64>() / n;
            q_points.push((zq, mean_ba, mean_sr));
        }

        // Find ceiling: last Q where efficiency (BA improvement / % size increase) ≥ 0.01
        // In other words: scan pairs, when the next step gives < 0.01 BA per 1% size, stop.
        let mut ceiling_q = None;
        let mut ceiling_ba = 0.0;
        let mut ceiling_sr = 0.0;
        let mut next_ba = 0.0;
        let mut next_sr = 0.0;

        for i in 0..q_points.len().saturating_sub(1) {
            let (q_lo, ba_lo, sr_lo) = q_points[i];
            let (_q_hi, ba_hi, sr_hi) = q_points[i + 1];

            let ba_improvement = ba_lo - ba_hi; // positive = quality improved
            let size_increase_pct = (sr_hi - sr_lo) / sr_lo * 100.0;

            if size_increase_pct > 0.0 {
                let efficiency = ba_improvement / size_increase_pct;
                if efficiency < 0.01 {
                    // This step is wasteful — ceiling is at q_lo
                    ceiling_q = Some(q_lo);
                    ceiling_ba = ba_lo;
                    ceiling_sr = sr_lo;
                    next_ba = ba_hi;
                    next_sr = sr_hi;
                    break;
                }
            }
        }

        // If no ceiling found (all steps are efficient), ceiling is the highest Q
        if ceiling_q.is_none() {
            if let Some(&(q, ba, sr)) = q_points.last() {
                ceiling_q = Some(q);
                ceiling_ba = ba;
                ceiling_sr = sr;
                next_ba = ba;
                next_sr = sr;
            }
        }

        ceilings.push(ResizeCeiling {
            ratio,
            ceiling_q,
            ceiling_ba,
            ceiling_size_ratio: ceiling_sr,
            next_ba,
            next_size_ratio: next_sr,
        });
    }

    ceilings
}

fn print_summary(results: &[RawResult], args: &Args) {
    let summary = compute_summary(results, args);

    // Group by (src_encoder, src_sub) for display
    let mut display_groups: Vec<(String, String, Vec<&SummaryRow>)> = Vec::new();
    for row in &summary {
        if let Some(g) = display_groups
            .iter_mut()
            .find(|(e, s, _)| *e == row.src_encoder && *s == row.src_sub)
        {
            g.2.push(row);
        } else {
            display_groups.push((row.src_encoder.clone(), row.src_sub.clone(), vec![row]));
        }
    }

    // === Recommended settings (primary output) ===
    println!("\n{}", "=".repeat(90));
    println!("\nRecommended Re-encode Settings (barely perceptible, BA delta \u{2264} 0.3):");

    for (enc, sub, rows) in &display_groups {
        println!("\n  {} {}:", enc, sub);
        for row in rows {
            let rec_str = match row.rec_zen_q {
                Some(q) => format!(
                    "zen Q{:.0}  \u{0394}BA {:.2}  size {:+.0}%",
                    q,
                    row.rec_ba_delta,
                    (row.rec_size_ratio - 1.0) * 100.0
                ),
                None => "no Q achieves \u{0394}BA \u{2264} 0.3".to_string(),
            };
            println!(
                "    Q{:2} (BA ~{:.1}) \u{2192} {}",
                row.src_quality, row.mean_src_ba, rec_str
            );
        }
    }

    // === Resize quality ceilings ===
    let resize_results: Vec<&RawResult> = results.iter().filter(|r| r.resize_ratio > 1.0).collect();
    if !resize_results.is_empty() {
        let ceilings = compute_resize_ceilings(results);

        println!("\n{}", "=".repeat(90));
        println!("\nResize Quality Ceilings (above ceiling, bytes wasted on imperceptible gain):");
        println!(
            "  {:>6}  {:>10}  {:>8}  {:>12}  {}",
            "ratio", "ceiling Q", "BA", "size vs src", "reason"
        );
        println!(
            "  {:>6}  {:>10}  {:>8}  {:>12}  {}",
            "-----", "---------", "------", "-----------", "------"
        );

        for c in &ceilings {
            let q_str = c
                .ceiling_q
                .map_or("-".to_string(), |q| format!("Q{:.0}", q));
            let reason = if (c.next_ba - c.ceiling_ba).abs() < 0.001 {
                "highest tested".to_string()
            } else {
                let ba_gain = c.ceiling_ba - c.next_ba;
                let size_cost =
                    (c.next_size_ratio - c.ceiling_size_ratio) / c.ceiling_size_ratio * 100.0;
                format!("next step: {:.2} BA for +{:.0}% size", ba_gain, size_cost)
            };
            println!(
                "  {:>5.1}x  {:>10}  {:>6.2}  {:>10.0}%  {}",
                c.ratio,
                q_str,
                c.ceiling_ba,
                c.ceiling_size_ratio * 100.0,
                reason
            );
        }

        // Also show the full R-D curve per ratio for context
        println!("\n  Full R-D curves:");
        for &ratio in &RESIZE_RATIOS {
            let at_ratio: Vec<&&RawResult> = resize_results
                .iter()
                .filter(|r| (r.resize_ratio - ratio).abs() < 0.01)
                .collect();
            if at_ratio.is_empty() {
                continue;
            }

            println!("\n  {:.1}x downscale:", ratio);
            for &zq in &RESIZE_QUALITIES {
                let at_q: Vec<&&&RawResult> =
                    at_ratio.iter().filter(|r| r.zen_quality == zq).collect();
                if at_q.is_empty() {
                    continue;
                }
                let n = at_q.len() as f64;
                let mean_ba = at_q.iter().map(|r| r.reenc_ba).sum::<f64>() / n;
                let mean_bd = at_q.iter().map(|r| r.ba_delta).sum::<f64>() / n;
                let mean_sr = at_q.iter().map(|r| r.size_ratio).sum::<f64>() / n;
                println!(
                    "    zen Q{:.0}: BA {:.2}, \u{0394} {:.2}, size {:.0}%",
                    zq,
                    mean_ba,
                    mean_bd,
                    mean_sr * 100.0
                );
            }
        }
    }

    // === Detailed match/shrink (secondary) ===
    println!("\n{}", "=".repeat(90));
    println!("\nDetailed Match/Shrink Analysis:");

    for (enc, sub, rows) in &display_groups {
        println!("\n  {} {}:", enc, sub);
        for row in rows {
            let match_str = match row.match_zen_q {
                Some(q) => format!(
                    "match=Q{:.0} (\u{0394}{:.2}, {:+.0}%)",
                    q,
                    row.match_ba_delta,
                    (row.match_size_ratio - 1.0) * 100.0
                ),
                None => "match=NONE".to_string(),
            };
            let shrink_str = match row.shrink_zen_q {
                Some(q) => format!(
                    "shrink=Q{:.0} ({:+.0}%)",
                    q,
                    (row.shrink_size_ratio - 1.0) * 100.0
                ),
                None => "shrink=NONE".to_string(),
            };
            println!(
                "    Q{:2} (BA ~{:.1}): {}  {}",
                row.src_quality, row.mean_src_ba, match_str, shrink_str
            );
        }
    }
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

fn write_raw_csv(path: &Path, results: &[RawResult]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot create {}: {e}", path.display());
            return;
        }
    };
    writeln!(
        f,
        "image,src_encoder,src_quality,src_sub,src_ba,src_ss2,src_size,\
         resize_ratio,zen_preset,zen_quality,zen_sub,reenc_ba,reenc_ss2,reenc_size,\
         ba_delta,ss2_delta,size_ratio"
    )
    .ok();
    for r in results {
        writeln!(
            f,
            "{},{},{},{},{:.4},{:.2},{},{:.1},{},{:.0},{},{:.4},{:.2},{},{:.4},{:.2},{:.4}",
            r.image,
            r.src_encoder,
            r.src_quality,
            r.src_sub,
            r.src_ba,
            r.src_ss2,
            r.src_size,
            r.resize_ratio,
            r.zen_preset,
            r.zen_quality,
            r.zen_sub,
            r.reenc_ba,
            r.reenc_ss2,
            r.reenc_size,
            r.ba_delta,
            r.ss2_delta,
            r.size_ratio,
        )
        .ok();
    }
    println!("  raw_data.csv: {} rows", results.len());
}

fn write_summary_csv(path: &Path, summary: &[SummaryRow]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot create {}: {e}", path.display());
            return;
        }
    };
    writeln!(
        f,
        "src_encoder,src_quality,src_sub,mean_src_ba,\
         rec_zen_q,rec_ba_delta,rec_size_ratio,\
         match_zen_q,match_mean_ba_delta,match_mean_size_ratio,\
         ci95_zen_q,shrink_zen_q,shrink_mean_size_ratio"
    )
    .ok();
    for r in summary {
        writeln!(
            f,
            "{},{},{},{:.4},{},{:.4},{:.4},{},{:.4},{:.4},{},{},{:.4}",
            r.src_encoder,
            r.src_quality,
            r.src_sub,
            r.mean_src_ba,
            r.rec_zen_q.map_or("-".to_string(), |q| format!("{:.0}", q)),
            r.rec_ba_delta,
            r.rec_size_ratio,
            r.match_zen_q
                .map_or("-".to_string(), |q| format!("{:.0}", q)),
            r.match_ba_delta,
            r.match_size_ratio,
            r.ci95_zen_q
                .map_or("-".to_string(), |q| format!("{:.0}", q)),
            r.shrink_zen_q
                .map_or("-".to_string(), |q| format!("{:.0}", q)),
            r.shrink_size_ratio,
        )
        .ok();
    }
    println!("  summary.csv: {} rows", summary.len());
}

// ---------------------------------------------------------------------------
// Preset offset analysis
// ---------------------------------------------------------------------------

/// For each (preset, src_encoder, src_quality), find the recommended Q at the
/// given BA tolerance. Returns (preset, family, src_q, rec_q) tuples.
fn find_recommended_q_per_preset(
    results: &[RawResult],
    ba_tolerance: f64,
) -> Vec<(String, String, u8, Option<f32>)> {
    let non_resize: Vec<&RawResult> = results.iter().filter(|r| r.resize_ratio == 1.0).collect();

    // Group by (zen_preset, src_encoder, src_quality, src_sub)
    let mut groups: std::collections::BTreeMap<(String, String, u8, String), Vec<&RawResult>> =
        std::collections::BTreeMap::new();
    for r in &non_resize {
        groups
            .entry((
                r.zen_preset.clone(),
                r.src_encoder.clone(),
                r.src_quality,
                r.src_sub.clone(),
            ))
            .or_default()
            .push(r);
    }

    let mut out = Vec::new();
    for ((preset, enc, sq, sub), group) in &groups {
        // Compute (zen_q → mean_ba_delta) for matching subsampling
        let mut q_stats: Vec<(f32, f64)> = Vec::new();
        for &zq in &ZEN_QUALITIES {
            let matching: Vec<&&RawResult> = group
                .iter()
                .filter(|r| r.zen_quality == zq && r.zen_sub == *sub)
                .collect();
            if matching.is_empty() {
                continue;
            }
            let n = matching.len() as f64;
            let mean_bd = matching.iter().map(|r| r.ba_delta).sum::<f64>() / n;
            q_stats.push((zq, mean_bd));
        }

        // Find lowest Q where mean ba_delta ≤ tolerance
        let rec_q = q_stats
            .iter()
            .find(|&&(_, bd)| bd <= ba_tolerance)
            .map(|&(q, _)| q);

        out.push((preset.clone(), enc.clone(), sq.to_owned(), rec_q));
    }

    out
}

fn print_preset_offsets(results: &[RawResult]) {
    let has_presets = results.iter().any(|r| r.zen_preset != "auto");
    if !has_presets {
        return;
    }

    // Collect preset names
    let mut preset_names: Vec<String> = results
        .iter()
        .map(|r| r.zen_preset.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    preset_names.sort();

    println!("\n{}", "=".repeat(100));
    println!("\nPreset Quality Offsets (vs auto_optimize)");
    println!("=========================================\n");

    // For each BA tolerance, show the offset table
    for &tol in &[0.3_f64, 0.5, 1.0] {
        let recs = find_recommended_q_per_preset(results, tol);

        // Build lookup: (preset, encoder, src_q) → rec_q
        let mut lookup: std::collections::BTreeMap<(String, String, u8), Option<f32>> =
            std::collections::BTreeMap::new();
        for (preset, enc, sq, rec_q) in &recs {
            lookup.insert((preset.clone(), enc.clone(), *sq), *rec_q);
        }

        // Get unique (encoder, src_q) pairs
        let mut source_points: Vec<(String, u8)> = recs
            .iter()
            .map(|(_, enc, sq, _)| (enc.clone(), *sq))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        source_points.sort();

        println!("  Tolerance {tol:.1} (BA delta ≤ {tol:.1}):");
        println!();

        // Header
        print!("  {:>8} {:>5}", "encoder", "srcQ");
        for p in &preset_names {
            print!("  {:>8}", p);
        }
        println!("  {:>8}", "offset");
        print!("  {:>8} {:>5}", "-------", "----");
        for _ in &preset_names {
            print!("  {:>8}", "-------");
        }
        println!("  {:>8}", "------");

        // Per-source-point offset tracking
        let mut all_offsets: std::collections::BTreeMap<String, Vec<f32>> =
            std::collections::BTreeMap::new();

        for (enc, sq) in &source_points {
            let auto_q = lookup
                .get(&("auto".to_string(), enc.clone(), *sq))
                .and_then(|q| *q);

            print!("  {:>8} {:>5}", enc, sq);
            for p in &preset_names {
                let q = lookup
                    .get(&(p.clone(), enc.clone(), *sq))
                    .and_then(|q| *q);
                match q {
                    Some(q) => print!("  {:>8.0}", q),
                    None => print!("  {:>8}", "-"),
                }
            }

            // Show offset vs auto for non-auto presets
            if let Some(aq) = auto_q {
                let offsets: Vec<String> = preset_names
                    .iter()
                    .filter(|p| *p != "auto")
                    .filter_map(|p| {
                        let q = lookup
                            .get(&(p.clone(), enc.clone(), *sq))
                            .and_then(|q| *q)?;
                        let off = q - aq;
                        all_offsets.entry(p.clone()).or_default().push(off);
                        Some(format!("{off:+.0}"))
                    })
                    .collect();
                print!("  {}", offsets.join("/"));
            }
            println!();
        }

        // Summary: mean offset per preset
        println!();
        print!("  {:>14}", "mean offset:");
        // Skip auto column
        print!("  {:>8}", "");
        for p in &preset_names {
            if p == "auto" {
                continue;
            }
            if let Some(offsets) = all_offsets.get(p) {
                if !offsets.is_empty() {
                    let mean = offsets.iter().sum::<f32>() / offsets.len() as f32;
                    print!("  {:>+8.1}", mean);
                } else {
                    print!("  {:>8}", "-");
                }
            } else {
                print!("  {:>8}", "-");
            }
        }
        println!("\n");
    }

    // Produce copy-paste Rust code
    println!("  Copy-paste for process.rs (using tolerance=0.3 mean offsets):");
    println!("  -------------------------------------------------------------");

    let recs = find_recommended_q_per_preset(results, 0.3);
    let mut lookup: std::collections::BTreeMap<(String, String, u8), Option<f32>> =
        std::collections::BTreeMap::new();
    for (preset, enc, sq, rec_q) in &recs {
        lookup.insert((preset.clone(), enc.clone(), *sq), *rec_q);
    }

    // Compute mean offset per preset across all (encoder, src_q) pairs
    let mut mean_offsets: std::collections::BTreeMap<String, f32> =
        std::collections::BTreeMap::new();
    for p in &preset_names {
        if p == "auto" {
            continue;
        }
        let mut offsets = Vec::new();
        for (_, enc, sq, _) in recs.iter().filter(|(pr, _, _, _)| pr == p) {
            let auto_q = lookup
                .get(&("auto".to_string(), enc.clone(), *sq))
                .and_then(|q| *q);
            let preset_q = lookup
                .get(&(p.clone(), enc.clone(), *sq))
                .and_then(|q| *q);
            if let (Some(aq), Some(pq)) = (auto_q, preset_q) {
                offsets.push(pq - aq);
            }
        }
        if !offsets.is_empty() {
            mean_offsets.insert(
                p.clone(),
                offsets.iter().sum::<f32>() / offsets.len() as f32,
            );
        }
    }

    let jpegli_off = mean_offsets.get("jpegli").copied().unwrap_or(1.0);
    let moz_off = mean_offsets.get("mozjpeg").copied().unwrap_or(3.0);
    let moz_max_off = mean_offsets.get("moz-max").copied().unwrap_or(3.0);
    let hyb_off = mean_offsets.get("hybrid").copied().unwrap_or(0.0);
    let hyb_max_off = mean_offsets.get("hyb-max").copied().unwrap_or(0.0);

    println!();
    println!("  fn preset_quality_offset(preset: Option<crate::PresetArg>) -> f32 {{");
    println!("      use crate::PresetArg::*;");
    println!("      match preset {{");
    println!("          None => 0.0,");
    println!(
        "          Some(Hybrid | HybridProg) => {:.1},",
        hyb_off.max(0.0)
    );
    println!(
        "          Some(HybridMax) => {:.1},",
        hyb_max_off.max(0.0)
    );
    println!(
        "          Some(Jpegli | JpegliProg) => {:.1},",
        jpegli_off.max(0.0)
    );
    println!(
        "          Some(Mozjpeg | MozjpegProg) => {:.1},",
        moz_off.max(0.0)
    );
    println!(
        "          Some(MozjpegMax) => {:.1},",
        moz_max_off.max(0.0)
    );
    println!("      }}");
    println!("  }}");
    println!();

    // Write offset CSV
    println!("  (Offsets also written to preset_offsets.csv)");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    // Create output + tmp dirs
    let tmp_dir = args.output.join("tmp");
    std::fs::create_dir_all(&tmp_dir).unwrap_or_else(|e| {
        eprintln!("Cannot create output dir {}: {e}", args.output.display());
        std::process::exit(1);
    });

    // Check available encoders
    let has_turbo = !args.no_turbo && check_binary("cjpeg");
    let has_cjpegli = !args.no_cjpegli && check_binary("cjpegli");
    let has_mozjpeg = !args.no_mozjpeg;

    if !has_turbo && !args.no_turbo {
        eprintln!("warning: cjpeg not found, skipping libjpeg-turbo");
    }
    if !has_cjpegli && !args.no_cjpegli {
        eprintln!("warning: cjpegli not found, skipping cjpegli");
    }

    // Build source configs
    let mut source_configs = Vec::new();
    if has_turbo {
        source_configs.push(SourceConfig {
            encoder: "turbo".to_string(),
            sub: "420".to_string(),
        });
        if args.full_sweep {
            source_configs.push(SourceConfig {
                encoder: "turbo".to_string(),
                sub: "444".to_string(),
            });
        }
    }
    if has_mozjpeg {
        source_configs.push(SourceConfig {
            encoder: "mozjpeg".to_string(),
            sub: "420".to_string(),
        });
        if args.full_sweep {
            source_configs.push(SourceConfig {
                encoder: "mozjpeg".to_string(),
                sub: "444".to_string(),
            });
        }
    }
    if has_cjpegli {
        source_configs.push(SourceConfig {
            encoder: "cjpegli".to_string(),
            sub: "444".to_string(),
        });
    }

    if source_configs.is_empty() {
        eprintln!("No encoders available. Nothing to do.");
        std::process::exit(1);
    }

    // Load images
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.corpus)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read {}: {e}", args.corpus.display());
            std::process::exit(1);
        })
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    paths.sort();
    paths.truncate(args.max_images);

    let images: Vec<(PathBuf, ImageData)> = paths
        .iter()
        .filter_map(|p| ImageData::from_path(p).map(|img| (p.clone(), img)))
        .collect();

    if images.is_empty() {
        eprintln!("No PNG images found in {}", args.corpus.display());
        std::process::exit(1);
    }

    let corpus_name = args
        .corpus
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();

    let src_qualities: &[u8] = if args.full_sweep {
        &SRC_QUALITIES_FULL
    } else {
        &SRC_QUALITIES
    };

    // Build preset list
    let presets: Vec<(&str, Option<OptimizationPreset>)> = if args.preset_offsets {
        OFFSET_PRESETS.to_vec()
    } else {
        vec![("auto", None)]
    };

    // Estimate work
    let configs_per_img = source_configs.len() * src_qualities.len();
    let zen_combos = ZEN_QUALITIES.len() * 2; // rough max (some have 1 sub, some have 2)
    let est_encodes = images.len() * configs_per_img * zen_combos * presets.len();

    // Print header
    println!("=== Re-encoding Calibration ===");
    println!(
        "{} images ({}), BA tol={:.2}, shrink tol={:.2}, size tol={:.2}",
        images.len(),
        corpus_name,
        args.ba_tolerance,
        args.shrink_tolerance,
        args.size_tolerance
    );
    println!(
        "Encoders: {}",
        source_configs
            .iter()
            .map(|c| format!("{} {}", c.encoder, c.sub))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Source qualities: {src_qualities:?}");
    println!("Zen qualities: {:?}", &ZEN_QUALITIES);
    if args.preset_offsets {
        println!(
            "Presets: {}",
            presets.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
        );
    }
    if args.resize {
        println!("Resize ratios: {:?}", &RESIZE_RATIOS);
    }
    println!("Estimated re-encodes: ~{est_encodes}");
    println!();

    // Process all images in parallel
    let progress = AtomicUsize::new(0);
    let total = images.len();

    let all_results: Vec<Vec<RawResult>> = images
        .par_iter()
        .map(|(png_path, img)| {
            let r = process_image(
                img,
                png_path,
                &tmp_dir,
                &source_configs,
                src_qualities,
                &presets,
                &args,
            );
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            eprint!("\r  [{done}/{total}] {:<40}", img.name);
            io::stderr().flush().ok();
            r
        })
        .collect();

    eprintln!("\r  [{total}/{total}] done{:40}", "");

    let results: Vec<RawResult> = all_results.into_iter().flatten().collect();
    println!("Total results: {}", results.len());

    // Write raw CSV
    write_raw_csv(&args.output.join("raw_data.csv"), &results);

    // Compute summary and write CSV (auto-preset only)
    let summary = compute_summary(&results, &args);
    write_summary_csv(&args.output.join("summary.csv"), &summary);

    // Print summary to stdout (auto-preset only)
    print_summary(&results, &args);

    // Print preset offset analysis if enabled
    if args.preset_offsets {
        print_preset_offsets(&results);
    }

    // Cleanup tmp dir
    std::fs::remove_dir_all(&tmp_dir).ok();

    println!("\nOutput written to {}", args.output.display());
}
