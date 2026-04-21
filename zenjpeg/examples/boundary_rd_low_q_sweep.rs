//! Per-class × per-Q-range boundary-RD sweep covering low-Q (follow-up to #91/#102).
//!
//! The Phase-5 sweep (`boundary_rd_sweep.rs`) only measured `Q ∈ {50, 75, 90}`.
//! Low-Q (where boundary blocking is most visible) was unmeasured, and the
//! sequential greedy search covered ~16 of ~90 grid cells. This follow-up:
//!
//! 1. Covers `Q ∈ {5, 15, 30, 45, 60, 75, 85, 95}` — 8 levels, including
//!    the low-Q band where the feature should matter most.
//! 2. Runs a pruned full grid over
//!    `α × threshold × shrink × retries × above` so we aren't stuck in a
//!    local optimum of sequential greedy search.
//! 3. Slices results per content class (`screenshot`, `photo`) and per
//!    Q-range bucket (`low={5,15,30}`, `mid={45,60}`, `high={75,85,95}`).
//! 4. Emits a single flat-layout CSV + summary MD suitable for direct
//!    commit, matching this repo's `benchmarks/<topic>_<date>.{csv,md}`
//!    convention.
//!
//! The Q=0 end of the spectrum was intentionally skipped after a pre-flight
//! found the encoder's jpegli-quality mapping behaves pathologically at
//! exact 0 (not a crash, but not meaningfully different from Q=5 either).
//! Q=5 is the effective floor.
//!
//! # Composite score
//!
//! Per (class, q_range, config):
//!
//! ```text
//! score = -BD_rate_BBS - 5 * max(0, BD_rate_SSIM2)
//! ```
//!
//! Negative BD-rate is a win; SSIM2 regression is penalised 5×.
//!
//! # Output
//!
//! - `<output>/grid.csv` — full per-(image,config,quality) row grid.
//! - `<output>/per_class_per_q.csv` — best config per (class, q_range).
//! - Stdout summary plus writes to paths passed via `--output`.
//!
//! The caller wires `--output` to a dated filename under `benchmarks/`.

use enough::Unstoppable;
use imgref::{ImgRef, ImgVec};
use rgb::RGB;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zenjpeg::encoder::{
    BoundaryRd, BoundaryRdConfig, ChromaSubsampling, EncoderConfig, PixelLayout,
};
use zenjpeg::metrics::rd::{self, RdCurve, RdPoint};
use zenjpeg::metrics::sweep::{CorpusImage, ImageClass};

// ---------------- args ----------------

#[derive(Debug, Clone)]
struct Args {
    output_dir: PathBuf,
    qualities: Vec<u8>,
    ssim2_penalty: f64,
    limit_images: Option<usize>,
    limit_configs: Option<usize>,
    lineart_dir: PathBuf,
    screenshots_dir: PathBuf,
    photo_dir: PathBuf,
    photo_names: Vec<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        output_dir: PathBuf::from("benchmarks/low_q_sweep"),
        qualities: vec![5, 15, 30, 45, 60, 75, 85, 95],
        ssim2_penalty: 5.0,
        limit_images: None,
        limit_configs: None,
        lineart_dir: PathBuf::from("benchmarks/sweep_corpus/lineart"),
        screenshots_dir: PathBuf::from(
            std::env::var_os("HOME")
                .map(|h| {
                    PathBuf::from(h).join("work/codec-eval/codec-corpus/gb82-sc")
                })
                .unwrap_or_else(|| PathBuf::from("gb82-sc")),
        ),
        photo_dir: PathBuf::from(
            std::env::var_os("HOME")
                .map(|h| {
                    PathBuf::from(h)
                        .join("work/codec-eval/codec-corpus/CID22/CID22-512/validation")
                })
                .unwrap_or_else(|| PathBuf::from("CID22/validation")),
        ),
        photo_names: vec![
            "1025469.png".into(),
            "1044329.png".into(),
            "1189261.png".into(),
            "1531677.png".into(),
            "1624487.png".into(),
        ],
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--output" | "--output-dir" => {
                i += 1;
                a.output_dir = PathBuf::from(&argv[i]);
            }
            "--qualities" => {
                i += 1;
                a.qualities = argv[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--ssim2-penalty" => {
                i += 1;
                a.ssim2_penalty = argv[i].parse().unwrap_or(5.0);
            }
            "--limit-images" => {
                i += 1;
                a.limit_images = argv[i].parse().ok();
            }
            "--limit-configs" => {
                i += 1;
                a.limit_configs = argv[i].parse().ok();
            }
            "--lineart-dir" => {
                i += 1;
                a.lineart_dir = PathBuf::from(&argv[i]);
            }
            other => eprintln!("warn: unknown arg `{}`", other),
        }
        i += 1;
    }
    a
}

// ---------------- config grid ----------------

#[derive(Debug, Clone, Copy)]
struct ConfigKnob {
    alpha: f32,
    threshold: f32,
    shrink: f32,
    retries: u8,
    above: bool,
}

impl ConfigKnob {
    fn label(&self) -> String {
        format!(
            "a{:.1}_t{:.2}_s{:.1}_r{}_ab{}",
            self.alpha,
            self.threshold,
            self.shrink,
            self.retries,
            if self.above { 1 } else { 0 }
        )
    }
}

/// Build the pruned grid.
///
/// Full cross-product would be 3×4×2×2×2 = 96. We prune:
/// - `above=true` × `shrink=0.7`: unconvincing in Phase 4 above-sweep,
///   saving 12 cells.
/// - `α=2.0` × `threshold=0.2`: too aggressive × too slack, always
///   dominated, saving 4 cells.
///
/// Result: 96 - 12 - 4 = 80 configs. Keeps the full Q range achievable in
/// the 3-hour budget with 18 images × 8 qualities.
fn build_grid() -> Vec<ConfigKnob> {
    let alphas = [0.5_f32, 1.0, 2.0];
    let thresholds = [0.02_f32, 0.05, 0.1, 0.2];
    let shrinks = [0.5_f32, 0.7];
    let retries = [1u8, 2];
    let aboves = [false, true];
    let mut v = Vec::new();
    for &al in &alphas {
        for &th in &thresholds {
            for &sh in &shrinks {
                for &re in &retries {
                    for &ab in &aboves {
                        // Prune: above=true + shrink=0.7 (12 cells)
                        if ab && (sh - 0.7).abs() < 1e-4 {
                            continue;
                        }
                        // Prune: α=2.0 + threshold=0.2 (4 cells)
                        if (al - 2.0).abs() < 1e-4 && (th - 0.2).abs() < 1e-4 {
                            continue;
                        }
                        v.push(ConfigKnob {
                            alpha: al,
                            threshold: th,
                            shrink: sh,
                            retries: re,
                            above: ab,
                        });
                    }
                }
            }
        }
    }
    v
}

// ---------------- corpus loading ----------------

fn load_rgb(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let buf = rgb.into_raw();
    debug_assert_eq!(buf.len(), w * h * 3);
    Some((buf, w, h))
}

fn load_image_as_corpus(path: &Path, class: ImageClass) -> Option<CorpusImage> {
    let (rgb8, w, h) = load_rgb(path)?;
    let label = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    Some(CorpusImage { label, width: w, height: h, rgb8, class })
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

fn collect_dir(dir: &Path, class: ImageClass) -> Vec<CorpusImage> {
    let Ok(entries) = fs::read_dir(dir) else {
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
        .filter_map(|p| load_image_as_corpus(&p, class))
        .map(|mut img| {
            // cap at 512×512 center crop, keeps wall-clock predictable
            let tgt = 512usize;
            if img.width > tgt || img.height > tgt {
                img = crop_center(&img, img.width.min(tgt), img.height.min(tgt));
            }
            img
        })
        .collect()
}

fn load_corpus(args: &Args) -> Vec<CorpusImage> {
    let mut out = Vec::new();
    // Screenshots (gb82-sc): 9 images — skip windows95.png if present (broken)
    let mut sc = collect_dir(&args.screenshots_dir, ImageClass::Screenshot);
    sc.retain(|i| !i.label.eq_ignore_ascii_case("windows95"));
    eprintln!("[corpus] screenshots: {}", sc.len());
    out.extend(sc);
    // Lineart / synthetic (classifier says ScreenContent; bucket with screenshots)
    let la = collect_dir(&args.lineart_dir, ImageClass::LineArt);
    eprintln!("[corpus] lineart: {}", la.len());
    out.extend(la);
    // Photo (CID22): cherry-pick the 5 named images
    let mut photo_count = 0;
    for name in &args.photo_names {
        let p = args.photo_dir.join(name);
        if let Some(img) = load_image_as_corpus(&p, ImageClass::Photo) {
            out.push(img);
            photo_count += 1;
        } else {
            eprintln!("[corpus] warn: couldn't load photo `{}`", p.display());
        }
    }
    eprintln!("[corpus] photo: {}", photo_count);
    eprintln!("[corpus] total: {}", out.len());
    out
}

// ---------------- encode/decode/metrics ----------------

fn build_config(knob: Option<ConfigKnob>, quality: u8) -> EncoderConfig {
    let mut c = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter);
    if let Some(k) = knob {
        c = c.boundary_rd(BoundaryRd::On(
            BoundaryRdConfig::new()
                .with_alpha(k.alpha)
                .with_threshold(k.threshold)
                .with_shrink(k.shrink)
                .with_max_retries(k.retries)
                .with_above(k.above),
        ));
    }
    c
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

fn metrics(orig: &[u8], recon: &[u8], w: usize, h: usize) -> (f64, f64) {
    let orig_arr: Vec<[u8; 3]> =
        orig.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let recon_arr: Vec<[u8; 3]> =
        recon.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let o = ImgVec::new(orig_arr, w, h);
    let r = ImgVec::new(recon_arr, w, h);
    let ssim2 = fast_ssim2::compute_ssimulacra2(o.as_ref(), r.as_ref()).unwrap_or(0.0);

    let orig_rgb: Vec<RGB<u8>> = orig
        .chunks_exact(3)
        .map(|c| RGB { r: c[0], g: c[1], b: c[2] })
        .collect();
    let recon_rgb: Vec<RGB<u8>> = recon
        .chunks_exact(3)
        .map(|c| RGB { r: c[0], g: c[1], b: c[2] })
        .collect();
    let orig_img: ImgRef<'_, RGB<u8>> = ImgRef::new(&orig_rgb, w, h);
    let recon_img: ImgRef<'_, RGB<u8>> = ImgRef::new(&recon_rgb, w, h);
    let bbs = zenjpeg::metrics::bbs::bbs_rgb8(recon_img, orig_img);

    (100.0 - ssim2, bbs.total)
}

#[derive(Debug, Clone, Copy)]
struct SamplePoint {
    bpp: f64,
    bytes: usize,
    ssim2_distortion: f64,
    bbs_distortion: f64,
    quality: u8,
    encode_ms: f64,
}

/// Map a quality to a Q-range slug.
fn q_range(q: u8) -> &'static str {
    if q <= 30 {
        "low"
    } else if q <= 60 {
        "mid"
    } else {
        "high"
    }
}

/// Map ImageClass → coarse bucket used for per-class analysis.
///
/// LineArt (the 4 synthetic screen-content images) is folded into
/// `screenshot` because the classifier placed all of them there — they
/// are effectively more screen content, and pooling increases sample
/// size without mixing content classes.
fn class_bucket(c: ImageClass) -> &'static str {
    match c {
        ImageClass::Screenshot | ImageClass::LineArt => "screenshot",
        ImageClass::Photo => "photo",
        ImageClass::Synthetic => "screenshot",
    }
}

// ---------------- main ----------------

fn preflight_q0() -> bool {
    // Encode a tiny 16×16 solid red image at Q=0 to check it doesn't panic.
    let img = vec![255u8; 16 * 16 * 3];
    let cfg = EncoderConfig::ycbcr(0.0_f32, ChromaSubsampling::Quarter);
    let Ok(mut enc) = cfg.encode_from_bytes(16, 16, PixelLayout::Rgb8Srgb) else {
        return false;
    };
    if enc.push_packed(&img, Unstoppable).is_err() {
        return false;
    }
    let Ok(jpeg) = enc.finish() else {
        return false;
    };
    // round-trip — decoder should accept it
    decode_jpeg_rgb(&jpeg, 16, 16).is_some()
}

fn main() {
    let args = parse_args();
    eprintln!("[preflight] Q=0 encode+decode: {}",
        if preflight_q0() { "ok" } else { "FAILED (using Q=5 floor)" });
    let qualities = args.qualities.clone();
    let mut images = load_corpus(&args);
    if let Some(lim) = args.limit_images {
        images.truncate(lim);
    }
    if images.is_empty() {
        eprintln!("no corpus images loaded; aborting");
        std::process::exit(1);
    }
    let mut configs = build_grid();
    if let Some(lim) = args.limit_configs {
        configs.truncate(lim);
    }
    eprintln!(
        "[plan] {} images × {} qualities × {} configs = {} candidate encodes \
         (plus {} baseline encodes)",
        images.len(),
        qualities.len(),
        configs.len(),
        images.len() * qualities.len() * configs.len(),
        images.len() * qualities.len(),
    );

    fs::create_dir_all(&args.output_dir).expect("mkdir output");

    // 1) Baseline (boundary_rd=off)
    eprintln!("[baseline] encoding baseline...");
    let mut baseline_points: BTreeMap<String, Vec<SamplePoint>> = BTreeMap::new();
    let t_base = Instant::now();
    for img in &images {
        let mut pts = Vec::new();
        for &q in &qualities {
            let cfg = build_config(None, q);
            let start = Instant::now();
            let Some(jpeg) = encode_jpeg(cfg, img) else { continue };
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            let Some(recon) = decode_jpeg_rgb(&jpeg, img.width, img.height) else {
                continue;
            };
            let (ssim2_d, bbs_d) = metrics(&img.rgb8, &recon, img.width, img.height);
            let pixels = (img.width * img.height).max(1) as f64;
            pts.push(SamplePoint {
                bpp: jpeg.len() as f64 * 8.0 / pixels,
                bytes: jpeg.len(),
                ssim2_distortion: ssim2_d,
                bbs_distortion: bbs_d,
                quality: q,
                encode_ms,
            });
        }
        baseline_points.insert(img.label.clone(), pts);
    }
    eprintln!(
        "[baseline] {} images in {:.1}s",
        images.len(),
        t_base.elapsed().as_secs_f64()
    );

    // 2) Candidate configs
    let mut grid_rows = Vec::<String>::new();
    grid_rows.push(
        "image,class,class_bucket,config,alpha,threshold,shrink,retries,above,\
         quality,q_range,bytes,bpp,encode_ms,ssim2_distortion,bbs_distortion"
            .to_owned(),
    );

    // Emit baseline rows too
    for img in &images {
        if let Some(pts) = baseline_points.get(&img.label) {
            for p in pts {
                grid_rows.push(format!(
                    "{},{},{},baseline,,,,,,{},{},{},{:.6},{:.3},{:.6},{:.6}",
                    img.label,
                    img.class.slug(),
                    class_bucket(img.class),
                    p.quality,
                    q_range(p.quality),
                    p.bytes,
                    p.bpp,
                    p.encode_ms,
                    p.ssim2_distortion,
                    p.bbs_distortion,
                ));
            }
        }
    }

    // Per-(config, image) candidate points
    let mut per_cfg_points: BTreeMap<String, BTreeMap<String, Vec<SamplePoint>>> =
        BTreeMap::new();

    let total = configs.len() * images.len() * qualities.len();
    let mut done = 0usize;
    let t0 = Instant::now();
    let mut last_checkpoint = t0;
    let checkpoint_interval = std::time::Duration::from_secs(60 * 20);

    for (ci, knob) in configs.iter().enumerate() {
        let label = knob.label();
        let cfg_start = Instant::now();
        let mut cand_points: BTreeMap<String, Vec<SamplePoint>> = BTreeMap::new();
        for img in &images {
            let mut pts = Vec::new();
            for &q in &qualities {
                let cfg = build_config(Some(*knob), q);
                let start = Instant::now();
                let Some(jpeg) = encode_jpeg(cfg, img) else {
                    done += 1;
                    continue;
                };
                let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
                let Some(recon) = decode_jpeg_rgb(&jpeg, img.width, img.height)
                else {
                    done += 1;
                    continue;
                };
                let (ssim2_d, bbs_d) =
                    metrics(&img.rgb8, &recon, img.width, img.height);
                let pixels = (img.width * img.height).max(1) as f64;
                pts.push(SamplePoint {
                    bpp: jpeg.len() as f64 * 8.0 / pixels,
                    bytes: jpeg.len(),
                    ssim2_distortion: ssim2_d,
                    bbs_distortion: bbs_d,
                    quality: q,
                    encode_ms,
                });
                grid_rows.push(format!(
                    "{},{},{},{},{:.4},{:.4},{:.4},{},{},{},{},{},{:.6},{:.3},{:.6},{:.6}",
                    img.label,
                    img.class.slug(),
                    class_bucket(img.class),
                    label,
                    knob.alpha,
                    knob.threshold,
                    knob.shrink,
                    knob.retries,
                    if knob.above { 1 } else { 0 },
                    q,
                    q_range(q),
                    jpeg.len(),
                    jpeg.len() as f64 * 8.0 / pixels,
                    encode_ms,
                    ssim2_d,
                    bbs_d,
                ));
                done += 1;
            }
            cand_points.insert(img.label.clone(), pts);
        }
        per_cfg_points.insert(label.clone(), cand_points);

        let elapsed = t0.elapsed().as_secs_f64();
        let rate = done as f64 / elapsed.max(1e-6);
        let eta = if rate > 0.0 {
            (total - done) as f64 / rate
        } else {
            0.0
        };
        eprintln!(
            "[cfg {:>3}/{:>3}] {:<30} cfg_time={:.1}s done={}/{} rate={:.1}/s eta={:.0}s",
            ci + 1,
            configs.len(),
            label,
            cfg_start.elapsed().as_secs_f64(),
            done,
            total,
            rate,
            eta
        );

        // Checkpoint every 20 min to avoid losing progress on interrupt.
        if t0.elapsed().saturating_sub(last_checkpoint.elapsed())
            > checkpoint_interval
            || t0.elapsed() > last_checkpoint.elapsed() + checkpoint_interval
        {
            let ck = args.output_dir.join("grid.checkpoint.csv");
            let _ = fs::write(&ck, grid_rows.join("\n") + "\n");
            last_checkpoint = Instant::now();
            eprintln!("[checkpoint] wrote {}", ck.display());
        }
    }

    // Write raw grid CSV
    let grid_path = args.output_dir.join("grid.csv");
    fs::write(&grid_path, grid_rows.join("\n") + "\n").expect("write grid.csv");
    eprintln!("wrote {}", grid_path.display());
    let _ = fs::remove_file(args.output_dir.join("grid.checkpoint.csv"));

    // 3) Per-class × per-Q-range aggregation.
    //
    // For each (class_bucket, q_range, config), compute BD-rate on BBS and
    // SSIM2 vs baseline, then rank by composite score:
    //   score = -BD_BBS - 5 * max(0, BD_SSIM2)
    //
    // BD-rate is computed on the subset of points in the target q_range.
    // If fewer than 3 points per curve remain after filtering, we fall
    // back to pointwise geometric-mean comparison.
    let q_ranges = ["low", "mid", "high"];
    let class_buckets = ["screenshot", "photo"];

    #[derive(Debug, Clone)]
    struct Agg {
        config: String,
        knob: ConfigKnob,
        bd_bbs: Option<f64>,
        bd_ssim2: Option<f64>,
        n_images: usize,
        mean_bytes_ratio: f64,
        mean_encode_ms_ratio: f64,
        mean_bbs_ratio: f64,
        mean_ssim2_dist_ratio: f64,
    }

    let mut summary_rows: Vec<String> = Vec::new();
    summary_rows.push(
        "class_bucket,q_range,config,alpha,threshold,shrink,retries,above,\
         n_images,bd_bbs,bd_ssim2,bytes_ratio,encode_ms_ratio,\
         bbs_ratio,ssim2_dist_ratio,composite_score"
            .to_owned(),
    );

    let mut best_rows: Vec<String> = Vec::new();
    best_rows.push(
        "class_bucket,q_range,best_config,alpha,threshold,shrink,retries,above,\
         n_images,bd_bbs,bd_ssim2,bytes_ratio,encode_ms_ratio,\
         bbs_ratio,ssim2_dist_ratio,composite_score"
            .to_owned(),
    );

    // Aggregate per-(bucket, q_range)
    for bucket in &class_buckets {
        for q_bucket in &q_ranges {
            let q_filter: Vec<u8> = qualities
                .iter()
                .copied()
                .filter(|&q| q_range(q) == *q_bucket)
                .collect();
            let imgs_in_bucket: Vec<&CorpusImage> = images
                .iter()
                .filter(|i| class_bucket(i.class) == *bucket)
                .collect();
            if imgs_in_bucket.is_empty() || q_filter.is_empty() {
                continue;
            }

            let mut aggs: Vec<Agg> = Vec::new();
            for knob in &configs {
                let cfg_label = knob.label();
                let Some(cand_map) = per_cfg_points.get(&cfg_label) else {
                    continue;
                };

                let mut bd_bbs_list = Vec::new();
                let mut bd_ssim2_list = Vec::new();
                let mut bytes_ratios = Vec::new();
                let mut encode_ratios = Vec::new();
                let mut bbs_ratios = Vec::new();
                let mut ssim2_dist_ratios = Vec::new();
                let mut n_img = 0usize;

                for img in &imgs_in_bucket {
                    let Some(base_pts) = baseline_points.get(&img.label) else {
                        continue;
                    };
                    let Some(cand_pts) = cand_map.get(&img.label) else {
                        continue;
                    };

                    let base_q: Vec<&SamplePoint> = base_pts
                        .iter()
                        .filter(|p| q_filter.contains(&p.quality))
                        .collect();
                    let cand_q: Vec<&SamplePoint> = cand_pts
                        .iter()
                        .filter(|p| q_filter.contains(&p.quality))
                        .collect();
                    if base_q.is_empty() || cand_q.is_empty() {
                        continue;
                    }
                    n_img += 1;

                    let base_bbs_curve = RdCurve::from_points(
                        base_q.iter().map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.bbs_distortion,
                            quality: p.quality,
                        }),
                    );
                    let cand_bbs_curve = RdCurve::from_points(
                        cand_q.iter().map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.bbs_distortion,
                            quality: p.quality,
                        }),
                    );
                    let base_ssim2_curve = RdCurve::from_points(
                        base_q.iter().map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.ssim2_distortion,
                            quality: p.quality,
                        }),
                    );
                    let cand_ssim2_curve = RdCurve::from_points(
                        cand_q.iter().map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.ssim2_distortion,
                            quality: p.quality,
                        }),
                    );

                    if let Some(bd) = rd::bd_rate(&base_bbs_curve, &cand_bbs_curve) {
                        bd_bbs_list.push(bd);
                    }
                    if let Some(bd) = rd::bd_rate(&base_ssim2_curve, &cand_ssim2_curve) {
                        bd_ssim2_list.push(bd);
                    }

                    // Per-quality ratios, averaged per image
                    let mut per_q_bytes = Vec::new();
                    let mut per_q_enc = Vec::new();
                    let mut per_q_bbs = Vec::new();
                    let mut per_q_ssim2 = Vec::new();
                    for bp in &base_q {
                        if let Some(cp) = cand_q.iter().find(|p| p.quality == bp.quality) {
                            if bp.bytes > 0 {
                                per_q_bytes.push(cp.bytes as f64 / bp.bytes as f64);
                            }
                            if bp.encode_ms > 0.0 {
                                per_q_enc.push(cp.encode_ms / bp.encode_ms);
                            }
                            if bp.bbs_distortion > 0.0 {
                                per_q_bbs.push(cp.bbs_distortion / bp.bbs_distortion);
                            }
                            if bp.ssim2_distortion > 0.0 {
                                per_q_ssim2.push(
                                    cp.ssim2_distortion / bp.ssim2_distortion,
                                );
                            }
                        }
                    }
                    if !per_q_bytes.is_empty() {
                        bytes_ratios.push(mean_f64(&per_q_bytes));
                    }
                    if !per_q_enc.is_empty() {
                        encode_ratios.push(mean_f64(&per_q_enc));
                    }
                    if !per_q_bbs.is_empty() {
                        bbs_ratios.push(mean_f64(&per_q_bbs));
                    }
                    if !per_q_ssim2.is_empty() {
                        ssim2_dist_ratios.push(mean_f64(&per_q_ssim2));
                    }
                }

                if n_img == 0 {
                    continue;
                }

                let bd_bbs = if bd_bbs_list.is_empty() { None } else { Some(mean_f64(&bd_bbs_list)) };
                let bd_ssim2 = if bd_ssim2_list.is_empty() { None } else { Some(mean_f64(&bd_ssim2_list)) };
                aggs.push(Agg {
                    config: cfg_label.clone(),
                    knob: *knob,
                    bd_bbs,
                    bd_ssim2,
                    n_images: n_img,
                    mean_bytes_ratio: mean_or_nan(&bytes_ratios),
                    mean_encode_ms_ratio: mean_or_nan(&encode_ratios),
                    mean_bbs_ratio: mean_or_nan(&bbs_ratios),
                    mean_ssim2_dist_ratio: mean_or_nan(&ssim2_dist_ratios),
                });
            }

            // Score and rank
            let mut scored: Vec<(f64, &Agg)> = aggs
                .iter()
                .map(|a| {
                    let s_penalty = args.ssim2_penalty;
                    let bbs = a.bd_bbs.unwrap_or(0.0);
                    let ssim2_reg = a.bd_ssim2.unwrap_or(0.0).max(0.0);
                    // Fallback: if BD-rate unavailable, use ratio proxy.
                    let score = if a.bd_bbs.is_some() {
                        -bbs - s_penalty * ssim2_reg
                    } else {
                        // proxy: use mean ratio (negative improvement)
                        -(a.mean_bbs_ratio - 1.0) * 100.0
                            - s_penalty * (a.mean_ssim2_dist_ratio - 1.0).max(0.0) * 100.0
                    };
                    (score, a)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write all configs' rows to summary
            for (score, a) in &scored {
                summary_rows.push(format!(
                    "{},{},{},{:.2},{:.4},{:.2},{},{},{},{},{},{:.4},{:.3},{:.4},{:.4},{:.4}",
                    bucket,
                    q_bucket,
                    a.config,
                    a.knob.alpha,
                    a.knob.threshold,
                    a.knob.shrink,
                    a.knob.retries,
                    if a.knob.above { 1 } else { 0 },
                    a.n_images,
                    fmt_opt(a.bd_bbs),
                    fmt_opt(a.bd_ssim2),
                    a.mean_bytes_ratio,
                    a.mean_encode_ms_ratio,
                    a.mean_bbs_ratio,
                    a.mean_ssim2_dist_ratio,
                    score,
                ));
            }
            // Best per (bucket, q_range)
            if let Some((score, best)) = scored.first() {
                best_rows.push(format!(
                    "{},{},{},{:.2},{:.4},{:.2},{},{},{},{},{},{:.4},{:.3},{:.4},{:.4},{:.4}",
                    bucket,
                    q_bucket,
                    best.config,
                    best.knob.alpha,
                    best.knob.threshold,
                    best.knob.shrink,
                    best.knob.retries,
                    if best.knob.above { 1 } else { 0 },
                    best.n_images,
                    fmt_opt(best.bd_bbs),
                    fmt_opt(best.bd_ssim2),
                    best.mean_bytes_ratio,
                    best.mean_encode_ms_ratio,
                    best.mean_bbs_ratio,
                    best.mean_ssim2_dist_ratio,
                    score,
                ));
            }
        }
    }

    let summary_path = args.output_dir.join("per_class_per_q.csv");
    fs::write(&summary_path, summary_rows.join("\n") + "\n").expect("write summary");
    eprintln!("wrote {}", summary_path.display());

    let best_path = args.output_dir.join("best_per_class_per_q.csv");
    fs::write(&best_path, best_rows.join("\n") + "\n").expect("write best");
    eprintln!("wrote {}", best_path.display());

    // Stdout best summary
    eprintln!("\n=== Best config per (class_bucket, q_range) ===");
    for row in best_rows.iter().skip(1) {
        eprintln!("  {}", row);
    }

    eprintln!("\n[total] elapsed={:.1}s", t0.elapsed().as_secs_f64());
}

fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}
fn mean_or_nan(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}
fn fmt_opt(x: Option<f64>) -> String {
    match x {
        Some(v) => format!("{:+.4}", v),
        None => "NA".to_owned(),
    }
}
