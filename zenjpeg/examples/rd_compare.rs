//! Rate-distortion comparison CLI for zenjpeg encoder configurations.
//!
//! Measures BD-rate, mean perpendicular distance, and per-point wins of
//! a candidate configuration vs a baseline, across a small corpus of
//! photos, screenshots, and synthetic images.
//!
//! Example:
//! ```bash
//! cargo run --release --example rd_compare -- \
//!     --baseline default \
//!     --candidate auto_optimize \
//!     --corpus cid22:3,screenshots:2,synthetic:2 \
//!     --qualities 50,65,75,85,95 \
//!     --metrics ssim2,bbs \
//!     --output-dir benchmarks/rd_compare/demo/
//! ```
//!
//! Available named configs:
//! - `default` — `EncoderConfig::ycbcr(q, Quarter)`
//! - `auto_optimize` — default + `.auto_optimize(true)` at q ≥ 70
//! - `mozjpeg_progressive` — `.optimization(MozjpegProgressive)`
//! - `progressive` — default + `.progressive(true)`
//! - `default_444` / `auto_optimize_444` / `mozjpeg_progressive_444` —
//!   same with `ChromaSubsampling::None`
//!
//! Corpus specs:
//! - `cid22:N` — first N images from `~/work/codec-eval/codec-corpus/CID22/CID22-512/validation`
//! - `screenshots:N` — first N PNGs from `~/work/codec-eval/codec-corpus/gb82-sc`
//!   (scaled down via center-crop to 512px max side so the run stays bounded)
//! - `synthetic:N` — N deterministically-generated synthetic images
//! - `<path>/<image.png>` — explicit path (class inferred by dir if
//!   it's under cid22/gb82-sc, otherwise Photo)

use enough::Unstoppable;
use imgref::{ImgRef, ImgVec};
use rgb::RGB;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout,
};
use zenjpeg::metrics::rd;
use zenjpeg::metrics::sweep::{
    self, CorpusImage, ImageClass, MetricKind, SampleOutput, SweepResult,
};

#[derive(Debug, Clone)]
struct Args {
    baseline: String,
    candidate: String,
    corpus_specs: Vec<CorpusSpec>,
    qualities: Vec<u8>,
    metrics: Vec<MetricKind>,
    output_dir: PathBuf,
    max_corpus: usize,
    run_id: String,
}

#[derive(Debug, Clone)]
struct CorpusSpec {
    kind: String,
    count: usize,
}

fn parse_args() -> Args {
    let mut baseline = String::from("default");
    let mut candidate = String::from("auto_optimize");
    let mut corpus_specs: Vec<CorpusSpec> = vec![
        CorpusSpec { kind: "cid22".into(), count: 3 },
        CorpusSpec { kind: "screenshots".into(), count: 2 },
        CorpusSpec { kind: "synthetic".into(), count: 2 },
    ];
    let mut qualities: Vec<u8> = vec![50, 65, 75, 85, 95];
    let mut metrics: Vec<MetricKind> = vec![MetricKind::Ssim2, MetricKind::Bbs];
    let mut output_dir = PathBuf::from("benchmarks/rd_compare/");
    let mut run_id = chrono_timestamp();
    let max_corpus = usize::MAX;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--baseline" => {
                i += 1;
                baseline = argv[i].clone();
            }
            "--candidate" => {
                i += 1;
                candidate = argv[i].clone();
            }
            "--corpus" => {
                i += 1;
                corpus_specs = parse_corpus_specs(&argv[i]);
            }
            "--qualities" => {
                i += 1;
                qualities = argv[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--metrics" => {
                i += 1;
                metrics = argv[i]
                    .split(',')
                    .map(|s| match s.trim().to_ascii_lowercase().as_str() {
                        "ssim2" => MetricKind::Ssim2,
                        "bbs" => MetricKind::Bbs,
                        "butteraugli" => MetricKind::Butteraugli,
                        "dssim" => MetricKind::Dssim,
                        other => MetricKind::Custom(other.to_owned()),
                    })
                    .collect();
            }
            "--output-dir" => {
                i += 1;
                output_dir = PathBuf::from(&argv[i]);
            }
            "--run-id" => {
                i += 1;
                run_id = argv[i].clone();
            }
            "--help" | "-h" => {
                eprintln!("rd_compare — BD-rate / mean-distance comparison of two encoder configs");
                eprintln!();
                eprintln!("  --baseline <name>      (default: default)");
                eprintln!("  --candidate <name>     (default: auto_optimize)");
                eprintln!("  --corpus cid22:N,screenshots:N,synthetic:N");
                eprintln!("  --qualities 50,65,75,85,95");
                eprintln!("  --metrics ssim2,bbs");
                eprintln!("  --output-dir benchmarks/rd_compare/");
                eprintln!("  --run-id <stamp>       (default: today's date)");
                eprintln!();
                eprintln!("Named configs:");
                eprintln!("  default, default_444, progressive, progressive_444");
                eprintln!("  auto_optimize, auto_optimize_444");
                eprintln!("  mozjpeg_progressive, mozjpeg_progressive_444");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown flag: {}", other);
                std::process::exit(2);
            }
        }
        i += 1;
    }

    Args {
        baseline,
        candidate,
        corpus_specs,
        qualities,
        metrics,
        output_dir,
        max_corpus,
        run_id,
    }
}

fn chrono_timestamp() -> String {
    // Local-date stamp via `chrono` — already a zenjpeg dev-dep.
    // Keep it simple: YYYY-MM-DD + short rand-ish suffix from nanos.
    let now = chrono::Utc::now();
    format!("{}", now.format("%Y-%m-%d"))
}

fn parse_corpus_specs(s: &str) -> Vec<CorpusSpec> {
    s.split(',')
        .filter_map(|item| {
            let item = item.trim();
            if item.is_empty() {
                return None;
            }
            let (kind, count) = match item.split_once(':') {
                Some((k, n)) => (k.to_owned(), n.parse().ok()?),
                None => (item.to_owned(), usize::MAX),
            };
            Some(CorpusSpec { kind, count })
        })
        .collect()
}

/// Load a PNG or JPEG into a tightly-packed RGB8 buffer.
fn load_rgb(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let buf = rgb.into_raw();
    debug_assert_eq!(buf.len(), w * h * 3);
    Some((buf, w, h))
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME must be set"))
}

fn collect_corpus(specs: &[CorpusSpec], max: usize) -> Vec<CorpusImage> {
    let mut out = Vec::new();
    for spec in specs {
        let images = match spec.kind.as_str() {
            "cid22" => collect_cid22(spec.count),
            "screenshots" => collect_screenshots(spec.count),
            "synthetic" => generate_synthetic(spec.count),
            other => {
                // Treat as a literal path — either a single file or a
                // directory of files.
                let p = PathBuf::from(other);
                if p.is_file() {
                    load_image_as_corpus(&p, ImageClass::Photo, None)
                        .map(|c| vec![c])
                        .unwrap_or_default()
                } else if p.is_dir() {
                    dir_pngs(&p, spec.count, ImageClass::Photo)
                } else {
                    eprintln!("warning: ignoring unrecognised corpus spec '{}'", spec.kind);
                    Vec::new()
                }
            }
        };
        for img in images {
            if out.len() >= max {
                return out;
            }
            out.push(img);
        }
    }
    out
}

fn collect_cid22(n: usize) -> Vec<CorpusImage> {
    let root = home().join("work/codec-eval/codec-corpus/CID22/CID22-512/validation");
    dir_pngs(&root, n, ImageClass::Photo)
}

fn collect_screenshots(n: usize) -> Vec<CorpusImage> {
    // gb82-sc screenshots can be huge (2560×1664); center-crop to
    // max 512 per side so the demo fits in budget.
    let root = home().join("work/codec-eval/codec-corpus/gb82-sc");
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&root) else {
        eprintln!(
            "warning: screenshots dir missing: {}",
            root.display()
        );
        return out;
    };
    let mut paths: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|e| e.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    for p in paths.into_iter().take(n) {
        if let Some(mut img) = load_image_as_corpus(&p, ImageClass::Screenshot, None) {
            // Center-crop to 512×512 (or whatever both dims allow) so
            // big screenshots don't dominate the run.
            let tgt = 512usize;
            if img.width > tgt || img.height > tgt {
                let cw = img.width.min(tgt);
                let ch = img.height.min(tgt);
                img = crop_center(&img, cw, ch);
            }
            out.push(img);
        }
    }
    out
}

fn generate_synthetic(n: usize) -> Vec<CorpusImage> {
    // Deterministic set: checkerboard, text-like stripes, line art,
    // noise+edges. Classes: the first is Synthetic (checkerboard),
    // second is LineArt, rest are Synthetic again.
    let generators: Vec<(&str, ImageClass, fn(usize, usize) -> Vec<u8>)> = vec![
        ("synth_checkerboard", ImageClass::Synthetic, gen_checkerboard),
        ("synth_stripes", ImageClass::LineArt, gen_stripes),
        ("synth_grid", ImageClass::LineArt, gen_grid),
        ("synth_noise_edges", ImageClass::Synthetic, gen_noise_edges),
    ];
    let size = 384usize; // small enough to encode fast, big enough for real seams
    generators
        .into_iter()
        .take(n)
        .map(|(label, class, generator)| CorpusImage {
            label: label.to_owned(),
            width: size,
            height: size,
            rgb8: generator(size, size),
            class,
        })
        .collect()
}

fn gen_checkerboard(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 230 } else { 30 };
            let idx = (y * w + x) * 3;
            buf[idx] = v;
            buf[idx + 1] = v;
            buf[idx + 2] = v;
        }
    }
    buf
}

fn gen_stripes(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![255u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // Horizontal + vertical thin lines at irregular spacing
            let on = y % 7 == 0 || x % 11 == 0 || (x + y) % 13 == 3;
            if on {
                let idx = (y * w + x) * 3;
                buf[idx] = 20;
                buf[idx + 1] = 20;
                buf[idx + 2] = 20;
            }
        }
    }
    buf
}

fn gen_grid(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![245u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let on = (x % 16 == 0) || (y % 16 == 0);
            if on {
                let idx = (y * w + x) * 3;
                buf[idx] = 50;
                buf[idx + 1] = 50;
                buf[idx + 2] = 80;
            }
        }
    }
    buf
}

fn gen_noise_edges(w: usize, h: usize) -> Vec<u8> {
    // Blocky noise patches with sharp rectangular edges — should
    // produce visible seam artifacts at low Q.
    let mut buf = vec![128u8; w * h * 3];
    let mut state: u64 = 0xdeadbeefcafebabe;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (state >> 24) as u8;
            let block_x = x / 24;
            let block_y = y / 24;
            let patch = (block_x + block_y) % 3;
            let base: u8 = match patch {
                0 => 80,
                1 => 180,
                _ => 210,
            };
            let idx = (y * w + x) * 3;
            let v: u8 = base.saturating_add(noise / 10);
            buf[idx] = v;
            buf[idx + 1] = v.saturating_sub(10);
            buf[idx + 2] = v.saturating_sub(20);
        }
    }
    buf
}

fn load_image_as_corpus(
    path: &Path,
    class: ImageClass,
    label_override: Option<String>,
) -> Option<CorpusImage> {
    let (rgb8, w, h) = load_rgb(path)?;
    let label = label_override.unwrap_or_else(|| {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string())
    });
    Some(CorpusImage {
        label,
        width: w,
        height: h,
        rgb8,
        class,
    })
}

fn dir_pngs(root: &Path, n: usize, class: ImageClass) -> Vec<CorpusImage> {
    let Ok(entries) = fs::read_dir(root) else {
        eprintln!("warning: corpus dir missing: {}", root.display());
        return Vec::new();
    };
    let mut paths: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|e| e.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    paths
        .into_iter()
        .take(n)
        .filter_map(|p| load_image_as_corpus(&p, class, None))
        .collect()
}

fn crop_center(img: &CorpusImage, new_w: usize, new_h: usize) -> CorpusImage {
    let x0 = (img.width.saturating_sub(new_w)) / 2;
    let y0 = (img.height.saturating_sub(new_h)) / 2;
    let mut out = Vec::with_capacity(new_w * new_h * 3);
    for y in 0..new_h {
        let row_start = ((y + y0) * img.width + x0) * 3;
        out.extend_from_slice(&img.rgb8[row_start..row_start + new_w * 3]);
    }
    CorpusImage {
        label: img.label.clone(),
        width: new_w,
        height: new_h,
        rgb8: out,
        class: img.class,
    }
}

/// Build an [`EncoderConfig`] from a name like `"default"` or
/// `"auto_optimize_444"`.
///
/// Returns `None` if the name is unknown.
fn build_config(name: &str, quality: u8) -> Option<EncoderConfig> {
    let q = quality as f32;
    let mk = |subs: ChromaSubsampling, mutator: fn(EncoderConfig) -> EncoderConfig| {
        mutator(EncoderConfig::ycbcr(q, subs))
    };
    match name {
        "default" => Some(mk(ChromaSubsampling::Quarter, |c| c)),
        "default_444" => Some(mk(ChromaSubsampling::None, |c| c)),
        "progressive" => Some(mk(ChromaSubsampling::Quarter, |c| c.progressive(true))),
        "progressive_444" => Some(mk(ChromaSubsampling::None, |c| c.progressive(true))),
        "auto_optimize" => Some(mk(ChromaSubsampling::Quarter, |c| c.auto_optimize(true))),
        "auto_optimize_444" => Some(mk(ChromaSubsampling::None, |c| c.auto_optimize(true))),
        "mozjpeg_progressive" => Some(mk(ChromaSubsampling::Quarter, |c| {
            c.optimization(OptimizationPreset::MozjpegProgressive)
        })),
        "mozjpeg_progressive_444" => Some(mk(ChromaSubsampling::None, |c| {
            c.optimization(OptimizationPreset::MozjpegProgressive)
        })),
        // Phase 2 of #91 — boundary-continuity refinement, non-trellis.
        "boundary_rd" => Some(mk(ChromaSubsampling::Quarter, |c| c.boundary_rd(true))),
        "boundary_rd_444" => Some(mk(ChromaSubsampling::None, |c| c.boundary_rd(true))),
        "auto_optimize_boundary_rd" => Some(mk(ChromaSubsampling::Quarter, |c| {
            c.auto_optimize(true).boundary_rd(true)
        })),
        // Phase 4 of #91 — left+above boundary-continuity refinement.
        "boundary_rd_left_above" => Some(mk(ChromaSubsampling::Quarter, |c| {
            c.boundary_rd(true).boundary_rd_above(true)
        })),
        "boundary_rd_left_above_444" => Some(mk(ChromaSubsampling::None, |c| {
            c.boundary_rd(true).boundary_rd_above(true)
        })),
        "auto_optimize_boundary_rd_left_above" => Some(mk(ChromaSubsampling::Quarter, |c| {
            c.auto_optimize(true).boundary_rd(true).boundary_rd_above(true)
        })),
        _ => None,
    }
}

fn encode_jpeg(config: EncoderConfig, img: &CorpusImage) -> Option<Vec<u8>> {
    let mut enc = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    enc.push_packed(&img.rgb8, Unstoppable).ok()?;
    enc.finish().ok()
}

fn decode_jpeg_rgb(data: &[u8], w: usize, h: usize) -> Option<Vec<u8>> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder.decode().ok()?;
    if pixels.len() != w * h * 3 {
        return None;
    }
    Some(pixels)
}

/// Compute the configured metrics on (original, reconstructed).
fn compute_distortions(
    metrics: &[MetricKind],
    orig: &[u8],
    recon: &[u8],
    w: usize,
    h: usize,
) -> BTreeMap<MetricKind, f64> {
    let mut out = BTreeMap::new();
    let orig_rgb_vec: Vec<RGB<u8>> = orig
        .chunks_exact(3)
        .map(|c| RGB { r: c[0], g: c[1], b: c[2] })
        .collect();
    let recon_rgb_vec: Vec<RGB<u8>> = recon
        .chunks_exact(3)
        .map(|c| RGB { r: c[0], g: c[1], b: c[2] })
        .collect();
    let orig_img: ImgRef<'_, RGB<u8>> = ImgRef::new(&orig_rgb_vec, w, h);
    let recon_img: ImgRef<'_, RGB<u8>> = ImgRef::new(&recon_rgb_vec, w, h);

    for m in metrics {
        match m {
            MetricKind::Ssim2 => {
                let orig_arr: Vec<[u8; 3]> = orig_rgb_vec.iter().map(|p| [p.r, p.g, p.b]).collect();
                let recon_arr: Vec<[u8; 3]> = recon_rgb_vec.iter().map(|p| [p.r, p.g, p.b]).collect();
                let o = ImgVec::new(orig_arr, w, h);
                let r = ImgVec::new(recon_arr, w, h);
                let score = fast_ssim2::compute_ssimulacra2(o.as_ref(), r.as_ref()).unwrap_or(0.0);
                // Distortion convention: smaller is better. SSIM2 is higher-better.
                out.insert(MetricKind::Ssim2, 100.0 - score);
            }
            MetricKind::Bbs => {
                let bbs = zenjpeg::metrics::bbs::bbs_rgb8(recon_img, orig_img);
                out.insert(MetricKind::Bbs, bbs.total);
            }
            MetricKind::Butteraugli => {
                // Only wire up Butteraugli when it's available from
                // zenjpeg-bench-utils. Gated behind a run-time check so
                // the example builds without it.
                #[allow(unused)]
                {
                    // Intentionally left unimplemented in the public
                    // example — the butteraugli crate's API across
                    // versions isn't stable enough to hard-code here.
                    // The closure interface in sweep.rs means adding
                    // Butteraugli is a two-line patch for a project
                    // that has it wired up.
                    eprintln!(
                        "warning: butteraugli metric is not wired up in this example; \
                         skipping for this sample"
                    );
                }
            }
            MetricKind::Dssim => {
                eprintln!(
                    "warning: dssim metric is not wired up in this example; skipping"
                );
            }
            MetricKind::Custom(_) => {
                // Silently skip custom metrics in this example.
            }
        }
    }
    out
}

fn main() {
    let args = parse_args();
    let images = collect_corpus(&args.corpus_specs, args.max_corpus);
    if images.is_empty() {
        eprintln!("no corpus images loaded, aborting");
        std::process::exit(1);
    }
    eprintln!(
        "loaded {} images, running {} × {} configs × qualities {:?}, metrics {:?}",
        images.len(),
        args.qualities.len(),
        if args.baseline == args.candidate { 1 } else { 2 },
        args.qualities,
        args.metrics.iter().map(|m| m.slug()).collect::<Vec<_>>()
    );

    let config_names = if args.baseline == args.candidate {
        vec![args.baseline.clone()]
    } else {
        vec![args.baseline.clone(), args.candidate.clone()]
    };

    let metrics = args.metrics.clone();
    let encoder = |img: &CorpusImage, cfg_name: &str, q: u8| -> Option<SampleOutput> {
        let config = build_config(cfg_name, q)?;
        let start = Instant::now();
        let jpeg = encode_jpeg(config, img)?;
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
        let recon = decode_jpeg_rgb(&jpeg, img.width, img.height)?;
        let distortions = compute_distortions(&metrics, &img.rgb8, &recon, img.width, img.height);
        Some(SampleOutput {
            bytes: jpeg.len(),
            distortions,
            encode_ms,
        })
    };

    let total = images.len() * config_names.len() * args.qualities.len();
    eprintln!("sweep: {} encodes total", total);
    let t0 = Instant::now();
    let result = sweep::run_sweep(
        &images,
        &config_names,
        &args.qualities,
        &encoder,
        &mut |done, total| {
            if done % 10 == 0 || done == total {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed.max(1e-6);
                let eta = if rate > 0.0 {
                    (total - done) as f64 / rate
                } else {
                    0.0
                };
                eprintln!(
                    "  [{}/{}]  {:.1}/s  elapsed={:.1}s  eta={:.1}s",
                    done, total, rate, elapsed, eta
                );
            }
        },
    );

    let run_dir = args.output_dir.join(&args.run_id);
    fs::create_dir_all(&run_dir).expect("mkdir -p output_dir");
    write_curves_csv(&run_dir, &result, &args);
    let comparisons = write_per_image_report(&run_dir, &result, &args);
    write_class_aggregate(&run_dir, &comparisons, &args);
    print_summary(&comparisons, &args);
    eprintln!("wrote {}", run_dir.display());
}

fn write_curves_csv(run_dir: &Path, result: &SweepResult, args: &Args) {
    let path = run_dir.join("curves.csv");
    let mut f = fs::File::create(&path).expect("create curves.csv");
    write!(
        f,
        "image,class,config,quality,bytes,bpp,encode_ms"
    )
    .unwrap();
    for m in &args.metrics {
        write!(f, ",dist_{}", m.slug()).unwrap();
    }
    writeln!(f).unwrap();
    for (label, class) in result.images() {
        for cfg in result.configs() {
            if let Some(points) = result
                .results
                .get(label)
                .and_then(|per_cfg| per_cfg.get(&cfg))
            {
                for p in points {
                    write!(
                        f,
                        "{},{},{},{},{},{:.6},{:.3}",
                        label,
                        class.slug(),
                        cfg,
                        p.quality,
                        p.bytes,
                        p.bpp,
                        p.encode_ms,
                    )
                    .unwrap();
                    for m in &args.metrics {
                        let v = p.distortions.get(m).copied().unwrap_or(f64::NAN);
                        write!(f, ",{:.6}", v).unwrap();
                    }
                    writeln!(f).unwrap();
                }
            }
        }
    }
    eprintln!("wrote {}", path.display());
}

#[derive(Debug, Clone)]
struct PerImageComparison {
    image: String,
    class: ImageClass,
    metric: MetricKind,
    baseline_points: usize,
    candidate_points: usize,
    bd_rate: Option<f64>,
    mean_distance: f64,
    win_rate: f64,
}

fn write_per_image_report(
    run_dir: &Path,
    result: &SweepResult,
    args: &Args,
) -> Vec<PerImageComparison> {
    let mut comparisons = Vec::new();
    for (label, class) in result.images() {
        for metric in &args.metrics {
            let base = result.rd_curve(label, &args.baseline, metric);
            let cand = result.rd_curve(label, &args.candidate, metric);
            if base.is_empty() || cand.is_empty() {
                continue;
            }
            let cmp = rd::compare(&base, &cand);
            comparisons.push(PerImageComparison {
                image: label.to_owned(),
                class,
                metric: metric.clone(),
                baseline_points: base.len(),
                candidate_points: cand.len(),
                bd_rate: cmp.bd_rate,
                mean_distance: cmp.mean_distance,
                win_rate: cmp.win_rate,
            });
        }
    }
    let path = run_dir.join("per_image.csv");
    let mut f = fs::File::create(&path).expect("create per_image.csv");
    writeln!(
        f,
        "image,class,metric,baseline,candidate,baseline_points,candidate_points,\
         bd_rate_pct,mean_distance,win_rate"
    )
    .unwrap();
    for c in &comparisons {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{:.6},{:.4}",
            c.image,
            c.class.slug(),
            c.metric.slug(),
            args.baseline,
            args.candidate,
            c.baseline_points,
            c.candidate_points,
            format_opt(c.bd_rate, 4),
            c.mean_distance,
            c.win_rate,
        )
        .unwrap();
    }
    eprintln!("wrote {}", path.display());
    comparisons
}

fn format_opt(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) => format!("{:.*}", prec, x),
        None => String::from("NA"),
    }
}

fn write_class_aggregate(run_dir: &Path, comparisons: &[PerImageComparison], args: &Args) {
    // Group by (class, metric), summarise.
    let mut groups: BTreeMap<(String, String), Vec<&PerImageComparison>> = BTreeMap::new();
    for c in comparisons {
        groups
            .entry((c.class.slug().to_owned(), c.metric.slug().to_owned()))
            .or_default()
            .push(c);
    }
    let path = run_dir.join("by_class.csv");
    let mut f = fs::File::create(&path).expect("create by_class.csv");
    writeln!(
        f,
        "class,metric,baseline,candidate,n,bd_rate_mean,bd_rate_stdev,\
         mean_distance_mean,win_rate_mean"
    )
    .unwrap();
    for ((class, metric), rows) in &groups {
        let bds: Vec<f64> = rows.iter().filter_map(|r| r.bd_rate).collect();
        let dists: Vec<f64> = rows.iter().map(|r| r.mean_distance).collect();
        let wins: Vec<f64> = rows.iter().map(|r| r.win_rate).collect();
        let (bd_mean, bd_sd) = mean_stdev(&bds);
        let (dist_mean, _) = mean_stdev(&dists);
        let (win_mean, _) = mean_stdev(&wins);
        writeln!(
            f,
            "{},{},{},{},{},{},{},{:.6},{:.4}",
            class,
            metric,
            args.baseline,
            args.candidate,
            rows.len(),
            format_opt(Some(bd_mean), 4),
            format_opt(Some(bd_sd), 4),
            dist_mean,
            win_mean,
        )
        .unwrap();
    }
    eprintln!("wrote {}", path.display());
}

fn mean_stdev(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    (mean, var.sqrt())
}

fn print_summary(comparisons: &[PerImageComparison], args: &Args) {
    println!();
    println!("Summary: {} vs {}", args.baseline, args.candidate);
    println!(
        "{:<22} {:<10} {:<12} {:>12} {:>14} {:>9}",
        "image", "class", "metric", "BD-rate %", "mean_distance", "win_rate"
    );
    for c in comparisons {
        println!(
            "{:<22} {:<10} {:<12} {:>12} {:>14.4} {:>9.2}",
            truncate(&c.image, 22),
            c.class.slug(),
            c.metric.slug(),
            format_opt(c.bd_rate, 3),
            c.mean_distance,
            c.win_rate,
        );
    }
    // Means by metric.
    println!();
    let metrics: Vec<MetricKind> = args.metrics.clone();
    println!("Per-metric aggregate:");
    for m in &metrics {
        let bds: Vec<f64> = comparisons
            .iter()
            .filter(|c| &c.metric == m)
            .filter_map(|c| c.bd_rate)
            .collect();
        let (mean, sd) = mean_stdev(&bds);
        let dists: Vec<f64> = comparisons
            .iter()
            .filter(|c| &c.metric == m)
            .map(|c| c.mean_distance)
            .collect();
        let (dist_mean, _) = mean_stdev(&dists);
        println!(
            "  {:<12} n={} BD-rate mean={} stdev={} mean_distance={}",
            m.slug(),
            bds.len(),
            format_opt(Some(mean), 3),
            format_opt(Some(sd), 3),
            format_opt(Some(dist_mean), 4),
        );
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[s.len() - max..]
    }
}
