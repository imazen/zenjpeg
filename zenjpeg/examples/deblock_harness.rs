//! Deblocking experiment harness.
//!
//! Encodes source images with libjpeg-turbo, mozjpeg, and cjpegli at multiple
//! quality levels, caching the encoded JPEGs to disk. Then measures decode quality
//! with pluggable deblocking strategies.
//!
//! The cache is reusable across runs — encode once, measure many times.
//!
//! Usage:
//! ```bash
//! # Generate cached encoded JPEGs (first run)
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --generate
//!
//! # Measure baseline (no deblocking)
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --measure
//!
//! # Both in one shot
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder
//!
//! # Limit to N images for quick testing
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --images 3
//!
//! # Use specific corpus
//! cargo run --release -p zenjpeg --example deblock_harness --features decoder -- --corpus cid22
//! ```

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use zenjpeg::detect::{self, EncoderFamily};
use zenjpeg_bench_utils::{decode_jpeg_with_icc, ImageData, QualityMetrics, RgbImage};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const CACHE_DIR: &str = "/mnt/v/output/zenjpeg/deblock";
const RESULTS_DIR: &str = "/mnt/v/output/zenjpeg/deblock/results";

const QUALITY_LEVELS: [u8; 17] = [
    5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 93, 95, 97,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Encoder {
    Turbo420,
    Mozjpeg420,
    Cjpegli,
}

impl Encoder {
    fn dir_name(self) -> &'static str {
        match self {
            Self::Turbo420 => "turbo-420",
            Self::Mozjpeg420 => "mozjpeg-420",
            Self::Cjpegli => "cjpegli",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Turbo420 => "libjpeg-turbo 4:2:0",
            Self::Mozjpeg420 => "mozjpeg 4:2:0",
            Self::Cjpegli => "cjpegli",
        }
    }

    fn expected_family(self) -> EncoderFamily {
        match self {
            Self::Turbo420 => EncoderFamily::LibjpegTurbo,
            Self::Mozjpeg420 => EncoderFamily::Mozjpeg,
            Self::Cjpegli => EncoderFamily::CjpegliYcbcr,
        }
    }

    fn all() -> &'static [Encoder] {
        &[Self::Turbo420, Self::Mozjpeg420, Self::Cjpegli]
    }
}

/// A deblocking strategy. Takes JPEG bytes, returns decoded (possibly enhanced) RGB.
trait DeblockStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage>;
}

/// Baseline: standard zenjpeg integer IDCT decode, no enhancements.
struct BaselineDecode;

impl DeblockStrategy for BaselineDecode {
    fn name(&self) -> &str {
        "baseline"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        decode_jpeg_with_icc(jpeg_bytes).ok()
    }
}

/// Dequant bias: zenjpeg f32 IDCT with Laplacian dequantization biases.
struct DequantBiasDecode;

impl DeblockStrategy for DequantBiasDecode {
    fn name(&self) -> &str {
        "dequant_bias"
    }

    fn decode(&self, jpeg_bytes: &[u8]) -> Option<RgbImage> {
        use enough::Unstoppable;
        use zenjpeg::decoder::Decoder;

        let decoded = Decoder::new()
            .dequant_bias(true)
            .decode(jpeg_bytes, Unstoppable)
            .ok()?;

        let w = decoded.width() as usize;
        let h = decoded.height() as usize;

        // dequant_bias uses SrgbF32Precise → f32 output in [0.0, 1.0], convert to u8
        let f32_pixels = decoded.pixels_f32()?;
        let u8_pixels: Vec<u8> = f32_pixels
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            .collect();
        Some(zenjpeg_bench_utils::bytes_to_rgb(&u8_pixels, w, h))
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    generate: bool,
    measure: bool,
    corpus: String,
    max_images: usize,
    strategies: Vec<Box<dyn DeblockStrategy>>,
    verbose: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        generate: false,
        measure: false,
        corpus: "gb82+cid22".to_string(),
        max_images: usize::MAX,
        strategies: vec![Box::new(BaselineDecode), Box::new(DequantBiasDecode)],
        verbose: false,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--generate" => args.generate = true,
            "--measure" => args.measure = true,
            "--corpus" => {
                if let Some(s) = iter.next() {
                    args.corpus = s;
                }
            }
            "--images" => {
                args.max_images = iter
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(usize::MAX);
            }
            "--verbose" | "-v" => args.verbose = true,
            "--help" | "-h" => {
                eprintln!("Usage: deblock_harness [OPTIONS]");
                eprintln!("  --generate       Encode cached JPEGs (skip if already cached)");
                eprintln!("  --measure        Run measurements with deblock strategies");
                eprintln!("  --corpus <name>  gb82, cid22, gb82-sc, or gb82+cid22 (default)");
                eprintln!("  --images <N>     Max images per corpus");
                eprintln!("  --verbose        Per-image output");
                eprintln!();
                eprintln!("With no flags, runs both --generate and --measure.");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    // Default: both phases
    if !args.generate && !args.measure {
        args.generate = true;
        args.measure = true;
    }

    args
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

fn load_corpus_images(corpus_name: &str, max_images: usize) -> Vec<ImageData> {
    let cc = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let mut images = Vec::new();

    let corpora: Vec<&str> = match corpus_name {
        "gb82+cid22" => vec!["gb82", "cid22"],
        other => vec![other],
    };

    for name in corpora {
        let dir = match name {
            "gb82" => cc.get("gb82").expect("gb82 not found"),
            "cid22" => cc
                .get("CID22")
                .expect("CID22 not found")
                .join("CID22-512/validation"),
            "gb82-sc" => cc.get("gb82-sc").expect("gb82-sc not found"),
            other => {
                eprintln!("Unknown corpus: {other}");
                continue;
            }
        };

        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("cannot read corpus dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .is_some_and(|ext| ext == "png" || ext == "PNG")
            })
            .collect();
        paths.sort();

        for path in paths.into_iter().take(max_images) {
            match ImageData::from_path(&path) {
                Some(img) => images.push(img),
                None => eprintln!("  skip {}: load failed", path.display()),
            }
        }
    }

    images
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Encode with libjpeg-turbo cjpeg CLI (4:2:0). Returns JPEG bytes.
fn encode_turbo_420(ppm_path: &Path, quality: u8) -> Option<Vec<u8>> {
    let output = Command::new("cjpeg")
        .arg("-quality")
        .arg(quality.to_string())
        .arg(ppm_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(output.stdout)
}

/// Encode with mozjpeg-rs (in-process, progressive 4:2:0).
fn encode_mozjpeg_420(pixels: &[u8], w: usize, h: usize, quality: u8) -> Option<Vec<u8>> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w as u32, h as u32)
        .ok()
}

/// Encode with cjpegli CLI. Returns JPEG bytes.
fn encode_cjpegli(png_path: &Path, quality: u8, tmp_dir: &Path) -> Option<Vec<u8>> {
    let stem = png_path.file_stem().unwrap_or_default().to_string_lossy();
    let tmp_out = tmp_dir.join(format!("cjpegli_{stem}_q{quality}.jpg"));
    let output = Command::new("cjpegli")
        .arg(png_path)
        .arg(&tmp_out)
        .arg("-q")
        .arg(quality.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let data = std::fs::read(&tmp_out).ok()?;
    std::fs::remove_file(&tmp_out).ok();
    Some(data)
}

/// Write PPM for cjpeg CLI input.
fn write_ppm(path: &Path, pixels: &[u8], w: usize, h: usize) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(pixels)?;
    Ok(())
}

/// Cache key: encoder/image_name_qXX.jpg
fn cache_path(encoder: Encoder, image_name: &str, quality: u8) -> PathBuf {
    Path::new(CACHE_DIR)
        .join("sources")
        .join(encoder.dir_name())
        .join(format!("{image_name}_q{quality}.jpg"))
}

/// Generate all cached encoded JPEGs. Skips files that already exist.
fn generate_cache(images: &[ImageData]) {
    let sources_dir = Path::new(CACHE_DIR).join("sources");

    // Create directories
    for enc in Encoder::all() {
        let dir = sources_dir.join(enc.dir_name());
        std::fs::create_dir_all(&dir).expect("cannot create cache dir");
    }

    let tmp_dir = Path::new(CACHE_DIR).join("tmp");
    std::fs::create_dir_all(&tmp_dir).expect("cannot create tmp dir");

    // Count what needs encoding
    let mut needed = 0u64;
    let mut cached = 0u64;
    for img in images {
        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                if cache_path(enc, &img.name, q).exists() {
                    cached += 1;
                } else {
                    needed += 1;
                }
            }
        }
    }

    eprintln!(
        "Cache: {cached} existing, {needed} to encode ({} images x {} encoders x {} qualities)",
        images.len(),
        Encoder::all().len(),
        QUALITY_LEVELS.len()
    );

    if needed == 0 {
        eprintln!("All JPEGs cached, nothing to encode.");
        return;
    }

    let progress = AtomicUsize::new(0);
    let total = needed as usize;
    let start = Instant::now();

    // Build work items
    struct EncodeJob {
        encoder: Encoder,
        quality: u8,
        image_idx: usize,
    }

    let mut jobs = Vec::new();
    for (i, img) in images.iter().enumerate() {
        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                if !cache_path(enc, &img.name, q).exists() {
                    jobs.push(EncodeJob {
                        encoder: enc,
                        quality: q,
                        image_idx: i,
                    });
                }
            }
        }
    }

    // Pre-write PPM files for turbo (cjpeg needs file input)
    let ppm_dir = tmp_dir.join("ppm");
    std::fs::create_dir_all(&ppm_dir).expect("cannot create ppm dir");

    let needs_ppm: Vec<usize> = jobs
        .iter()
        .filter(|j| j.encoder == Encoder::Turbo420)
        .map(|j| j.image_idx)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for &idx in &needs_ppm {
        let img = &images[idx];
        let ppm_path = ppm_dir.join(format!("{}.ppm", img.name));
        if !ppm_path.exists() {
            write_ppm(&ppm_path, &img.pixels, img.width, img.height)
                .unwrap_or_else(|e| panic!("write PPM {}: {e}", ppm_path.display()));
        }
    }

    // Encode in parallel
    jobs.par_iter().for_each(|job| {
        let img = &images[job.image_idx];
        let out_path = cache_path(job.encoder, &img.name, job.quality);

        let jpeg = match job.encoder {
            Encoder::Turbo420 => {
                let ppm_path = ppm_dir.join(format!("{}.ppm", img.name));
                encode_turbo_420(&ppm_path, job.quality)
            }
            Encoder::Mozjpeg420 => {
                encode_mozjpeg_420(&img.pixels, img.width, img.height, job.quality)
            }
            Encoder::Cjpegli => {
                // cjpegli needs PNG input — find the original
                let cc = codec_corpus::Corpus::new().unwrap();
                let png_path = find_source_png(&cc, &img.name);
                match png_path {
                    Some(p) => encode_cjpegli(&p, job.quality, &tmp_dir),
                    None => {
                        eprintln!("  cannot find PNG for {}", img.name);
                        None
                    }
                }
            }
        };

        if let Some(data) = jpeg {
            if let Err(e) = std::fs::write(&out_path, &data) {
                eprintln!("  write error {}: {e}", out_path.display());
            }
        } else {
            eprintln!(
                "  encode failed: {} q{} {}",
                img.name,
                job.quality,
                job.encoder.display_name()
            );
        }

        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 50 == 0 || done == total {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            let remaining = (total - done) as f64 / rate;
            eprint!(
                "\r  Encoded {done}/{total} ({:.0}/s, {:.0}s remaining)    ",
                rate, remaining
            );
        }
    });

    eprintln!("\n  Done in {:.1}s", start.elapsed().as_secs_f64());

    // Cleanup PPMs
    std::fs::remove_dir_all(&ppm_dir).ok();
}

/// Find the original PNG source for an image name.
/// `name` may include the .png extension (e.g., "baby-lossless.png").
fn find_source_png(cc: &codec_corpus::Corpus, name: &str) -> Option<PathBuf> {
    // Strip .png/.PNG extension if present (ImageData.name includes extension)
    let stem = name
        .strip_suffix(".png")
        .or_else(|| name.strip_suffix(".PNG"))
        .unwrap_or(name);

    // Try gb82
    if let Ok(gb82) = cc.get("gb82") {
        let path = gb82.join(format!("{stem}.png"));
        if path.exists() {
            return Some(path);
        }
    }

    // Try CID22
    if let Ok(cid22) = cc.get("CID22") {
        let path = cid22.join(format!("CID22-512/validation/{stem}.png"));
        if path.exists() {
            return Some(path);
        }
        // Some CID22 images may use .PNG extension
        let path = cid22.join(format!("CID22-512/validation/{stem}.PNG"));
        if path.exists() {
            return Some(path);
        }
    }

    // Try gb82-sc
    if let Ok(sc) = cc.get("gb82-sc") {
        let path = sc.join(format!("{stem}.png"));
        if path.exists() {
            return Some(path);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Boundary discontinuity metric: mean |p0-q0| at 8x8 block boundaries.
/// Computed on the Y (luma) channel. Returns (smooth_mean, edge_mean, overall_mean).
fn boundary_discontinuity(img: &RgbImage) -> (f64, f64, f64) {
    let w = img.width();
    let h = img.height();
    let buf = img.buf();

    // Convert to Y channel (BT.601)
    let mut y = vec![0f32; w * h];
    for row in 0..h {
        for col in 0..w {
            let px = buf[row * img.stride() + col];
            y[row * w + col] = 0.299 * px.r as f32 + 0.587 * px.g as f32 + 0.114 * px.b as f32;
        }
    }

    let mut smooth_sum = 0.0f64;
    let mut smooth_count = 0u64;
    let mut edge_sum = 0.0f64;
    let mut edge_count = 0u64;

    // Edge threshold: if local gradient > this, it's an edge region
    let edge_thresh = 20.0f32;

    // Vertical boundaries (between columns 7-8, 15-16, etc.)
    for by in 0..h {
        for bx_idx in 1..(w / 8) {
            let col = bx_idx * 8;
            if col >= w {
                break;
            }
            let p0 = y[by * w + col - 1];
            let q0 = y[by * w + col];
            let disc = (p0 - q0).abs();

            // Check if this is an edge region: gradient magnitude around boundary
            let p1 = if col >= 2 { y[by * w + col - 2] } else { p0 };
            let q1 = if col + 1 < w { y[by * w + col + 1] } else { q0 };
            let grad = ((p0 - p1).abs() + (q1 - q0).abs()) * 0.5;

            if grad > edge_thresh {
                edge_sum += disc as f64;
                edge_count += 1;
            } else {
                smooth_sum += disc as f64;
                smooth_count += 1;
            }
        }
    }

    // Horizontal boundaries (between rows 7-8, 15-16, etc.)
    for bx in 0..w {
        for by_idx in 1..(h / 8) {
            let row = by_idx * 8;
            if row >= h {
                break;
            }
            let p0 = y[(row - 1) * w + bx];
            let q0 = y[row * w + bx];
            let disc = (p0 - q0).abs();

            let p1 = if row >= 2 { y[(row - 2) * w + bx] } else { p0 };
            let q1 = if row + 1 < h {
                y[(row + 1) * w + bx]
            } else {
                q0
            };
            let grad = ((p0 - p1).abs() + (q1 - q0).abs()) * 0.5;

            if grad > edge_thresh {
                edge_sum += disc as f64;
                edge_count += 1;
            } else {
                smooth_sum += disc as f64;
                smooth_count += 1;
            }
        }
    }

    let smooth_mean = if smooth_count > 0 {
        smooth_sum / smooth_count as f64
    } else {
        0.0
    };
    let edge_mean = if edge_count > 0 {
        edge_sum / edge_count as f64
    } else {
        0.0
    };
    let total = smooth_sum + edge_sum;
    let total_count = smooth_count + edge_count;
    let overall = if total_count > 0 {
        total / total_count as f64
    } else {
        0.0
    };

    (smooth_mean, edge_mean, overall)
}

#[derive(Debug)]
struct Measurement {
    image: String,
    encoder: Encoder,
    quality: u8,
    strategy: String,
    ssim2: f64,
    butteraugli: f64,
    boundary_smooth: f64,
    boundary_edge: f64,
    boundary_overall: f64,
    file_size: usize,
    detected_encoder: String,
    detected_quality: String,
}

/// Run measurements for all cached JPEGs with all strategies.
fn run_measurements(
    images: &[ImageData],
    strategies: &[Box<dyn DeblockStrategy>],
    verbose: bool,
) -> Vec<Measurement> {
    let results_dir = Path::new(RESULTS_DIR);
    std::fs::create_dir_all(results_dir).expect("cannot create results dir");

    // Build work items: (image_idx, encoder, quality)
    struct MeasureJob {
        image_idx: usize,
        encoder: Encoder,
        quality: u8,
    }

    let mut jobs = Vec::new();
    for (i, img) in images.iter().enumerate() {
        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                let path = cache_path(enc, &img.name, q);
                if path.exists() {
                    jobs.push(MeasureJob {
                        image_idx: i,
                        encoder: enc,
                        quality: q,
                    });
                }
            }
        }
    }

    eprintln!(
        "Measuring {} cached JPEGs x {} strategies = {} decode+measure ops",
        jobs.len(),
        strategies.len(),
        jobs.len() * strategies.len()
    );

    let progress = AtomicUsize::new(0);
    let total = jobs.len() * strategies.len();
    let start = Instant::now();

    // Measure in parallel over jobs (strategies are fast, parallelize over images)
    let measurements: Vec<Vec<Measurement>> = jobs
        .par_iter()
        .map(|job| {
            let img = &images[job.image_idx];
            let jpeg_path = cache_path(job.encoder, &img.name, job.quality);
            let jpeg_bytes = match std::fs::read(&jpeg_path) {
                Ok(b) => b,
                Err(_) => return vec![],
            };

            // Probe encoder detection
            let probe = detect::probe(&jpeg_bytes).ok();
            let detected_encoder = probe
                .as_ref()
                .map(|p| format!("{:?}", p.encoder))
                .unwrap_or_else(|| "ProbeError".to_string());
            let detected_quality = probe
                .as_ref()
                .map(|p| format!("{:?}", p.quality))
                .unwrap_or_else(|| "?".to_string());

            // Verify encoder detection
            if let Some(ref p) = probe {
                let expected = job.encoder.expected_family();
                let actual = p.encoder;
                if actual != expected {
                    eprintln!(
                        "  DETECTION MISMATCH: {} q{} {} — expected {:?}, got {:?}",
                        img.name,
                        job.quality,
                        job.encoder.display_name(),
                        expected,
                        actual
                    );
                }
            }

            // Build reference RgbImage from source pixels
            let reference = {
                use imgref::ImgVec;
                use rgb::RGB8;
                let px: Vec<RGB8> = img
                    .pixels
                    .chunks_exact(3)
                    .map(|c| RGB8 {
                        r: c[0],
                        g: c[1],
                        b: c[2],
                    })
                    .collect();
                ImgVec::new(px, img.width, img.height)
            };

            let mut results = Vec::with_capacity(strategies.len());

            for strategy in strategies {
                let decoded = match strategy.decode(&jpeg_bytes) {
                    Some(d) => d,
                    None => {
                        eprintln!(
                            "  decode failed: {} q{} {} [{}]",
                            img.name,
                            job.quality,
                            job.encoder.display_name(),
                            strategy.name()
                        );
                        continue;
                    }
                };

                // Quality metrics vs source
                let ssim2 = QualityMetrics::ssimulacra2(reference.as_ref(), decoded.as_ref());
                let ba = QualityMetrics::butteraugli(reference.as_ref(), decoded.as_ref());

                // Boundary discontinuity
                let (bsmooth, bedge, boverall) = boundary_discontinuity(&decoded);

                if verbose {
                    eprintln!(
                        "  {:<20} {:>10} q{:<3} [{:<13}] SS2={:6.2} BA={:5.2} BD={:.2}",
                        img.name,
                        job.encoder.display_name(),
                        job.quality,
                        strategy.name(),
                        ssim2,
                        ba,
                        bsmooth
                    );
                }

                results.push(Measurement {
                    image: img.name.clone(),
                    encoder: job.encoder,
                    quality: job.quality,
                    strategy: strategy.name().to_string(),
                    ssim2,
                    butteraugli: ba,
                    boundary_smooth: bsmooth,
                    boundary_edge: bedge,
                    boundary_overall: boverall,
                    file_size: jpeg_bytes.len(),
                    detected_encoder: detected_encoder.clone(),
                    detected_quality: detected_quality.clone(),
                });

                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 20 == 0 || done == total {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = done as f64 / elapsed;
                    let remaining = (total - done) as f64 / rate;
                    eprint!(
                        "\r  Measured {done}/{total} ({:.1}/s, {:.0}s remaining)    ",
                        rate, remaining
                    );
                }
            }

            results
        })
        .collect();

    let measurements: Vec<Measurement> = measurements.into_iter().flatten().collect();
    eprintln!(
        "\n  Done: {} measurements in {:.1}s",
        measurements.len(),
        start.elapsed().as_secs_f64()
    );

    measurements
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn write_csv(measurements: &[Measurement], path: &Path) {
    let mut f = std::fs::File::create(path).expect("cannot create CSV");
    writeln!(
        f,
        "image,encoder,quality,strategy,ssim2,butteraugli,bd_smooth,bd_edge,bd_overall,\
         file_size,detected_encoder,detected_quality"
    )
    .unwrap();

    for m in measurements {
        writeln!(
            f,
            "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{}",
            m.image,
            m.encoder.dir_name(),
            m.quality,
            m.strategy,
            m.ssim2,
            m.butteraugli,
            m.boundary_smooth,
            m.boundary_edge,
            m.boundary_overall,
            m.file_size,
            m.detected_encoder,
            m.detected_quality,
        )
        .unwrap();
    }
}

/// Print summary table: mean metrics per encoder × quality × strategy.
fn print_summary(measurements: &[Measurement]) {
    // Group by (strategy, encoder, quality) → collect metrics
    let mut groups: BTreeMap<(String, Encoder, u8), Vec<(f64, f64, f64)>> = BTreeMap::new();

    for m in measurements {
        groups
            .entry((m.strategy.clone(), m.encoder, m.quality))
            .or_default()
            .push((m.ssim2, m.butteraugli, m.boundary_smooth));
    }

    // Get all strategies
    let strategies: Vec<String> = measurements
        .iter()
        .map(|m| m.strategy.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for strategy in &strategies {
        eprintln!("\n=== Strategy: {} ===", strategy);
        eprintln!(
            "{:<14} {:>3}  {:>7} {:>7} {:>7}  {:>3}",
            "Encoder", "Q", "SS2", "BA", "BD_sm", "N"
        );
        eprintln!("{}", "-".repeat(52));

        for &enc in Encoder::all() {
            for &q in &QUALITY_LEVELS {
                if let Some(vals) = groups.get(&(strategy.clone(), enc, q)) {
                    let n = vals.len();
                    let mean_ss2: f64 = vals.iter().map(|v| v.0).sum::<f64>() / n as f64;
                    let mean_ba: f64 = vals.iter().map(|v| v.1).sum::<f64>() / n as f64;
                    let mean_bd: f64 = vals.iter().map(|v| v.2).sum::<f64>() / n as f64;

                    eprintln!(
                        "{:<14} {:>3}  {:>7.2} {:>7.2} {:>7.2}  {:>3}",
                        enc.dir_name(),
                        q,
                        mean_ss2,
                        mean_ba,
                        mean_bd,
                        n
                    );
                }
            }
        }
    }

    // Delta table: strategy improvements over baseline
    if strategies.len() > 1 {
        eprintln!("\n=== Deltas vs baseline ===");
        for strategy in strategies.iter().filter(|s| *s != "baseline") {
            eprintln!("\n--- {} vs baseline ---", strategy);
            eprintln!(
                "{:<14} {:>3}  {:>8} {:>8} {:>8}",
                "Encoder", "Q", "dSS2", "dBA", "dBD_sm"
            );
            eprintln!("{}", "-".repeat(55));

            for &enc in Encoder::all() {
                for &q in &QUALITY_LEVELS {
                    let baseline_key = ("baseline".to_string(), enc, q);
                    let strategy_key = (strategy.clone(), enc, q);

                    if let (Some(base), Some(strat)) =
                        (groups.get(&baseline_key), groups.get(&strategy_key))
                    {
                        let n = base.len().min(strat.len());
                        let d_ss2 = strat.iter().map(|v| v.0).sum::<f64>() / n as f64
                            - base.iter().map(|v| v.0).sum::<f64>() / n as f64;
                        let d_ba = strat.iter().map(|v| v.1).sum::<f64>() / n as f64
                            - base.iter().map(|v| v.1).sum::<f64>() / n as f64;
                        let d_bd = strat.iter().map(|v| v.2).sum::<f64>() / n as f64
                            - base.iter().map(|v| v.2).sum::<f64>() / n as f64;

                        eprintln!(
                            "{:<14} {:>3}  {:>+8.3} {:>+8.3} {:>+8.3}",
                            enc.dir_name(),
                            q,
                            d_ss2,
                            d_ba,
                            d_bd,
                        );
                    }
                }
            }
        }
    }

    // Detection accuracy
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut mismatches = Vec::new();
    // Deduplicate: check once per (encoder, quality, image)
    let mut seen = std::collections::HashSet::new();
    for m in measurements {
        if m.strategy != "baseline" {
            continue;
        }
        let key = (m.encoder, m.quality, m.image.clone());
        if !seen.insert(key) {
            continue;
        }
        total += 1;
        let expected = format!("{:?}", m.encoder.expected_family());
        if m.detected_encoder == expected {
            correct += 1;
        } else {
            mismatches.push(format!(
                "  {} q{} {}: expected {}, got {}",
                m.image,
                m.quality,
                m.encoder.dir_name(),
                expected,
                m.detected_encoder
            ));
        }
    }

    eprintln!("\n=== Encoder Detection ===");
    eprintln!("Correct: {correct}/{total}");
    if !mismatches.is_empty() {
        eprintln!("Mismatches:");
        for mm in &mismatches[..mismatches.len().min(20)] {
            eprintln!("{mm}");
        }
        if mismatches.len() > 20 {
            eprintln!("  ... and {} more", mismatches.len() - 20);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    eprintln!("Loading corpus: {}", args.corpus);
    let images = load_corpus_images(&args.corpus, args.max_images);
    eprintln!("Loaded {} images", images.len());

    if images.is_empty() {
        eprintln!("No images found!");
        return;
    }

    if args.generate {
        eprintln!("\n--- Phase 1: Generate cached JPEGs ---");
        generate_cache(&images);
    }

    if args.measure {
        eprintln!("\n--- Phase 2: Measure with strategies ---");
        let measurements = run_measurements(&images, &args.strategies, args.verbose);

        if measurements.is_empty() {
            eprintln!("No measurements collected! Run --generate first.");
            return;
        }

        // Write CSV
        let csv_path = Path::new(RESULTS_DIR).join("baseline.csv");
        write_csv(&measurements, &csv_path);
        eprintln!("\nCSV written to: {}", csv_path.display());

        // Print summary
        print_summary(&measurements);
    }
}
