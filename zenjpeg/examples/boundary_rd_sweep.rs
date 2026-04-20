//! Parameter sweep for the boundary-RD (non-trellis) refinement knobs.
//!
//! Phase 5 of issue #91 — takes a list of (α, threshold, shrink, retries)
//! quadruples, encodes every (image, quality) under every combination,
//! and ranks configs against a baseline (boundary_rd off) by BD-rate.
//!
//! # Composite score
//!
//! Each config gets a composite score based on its per-class BD-rate
//! measurements:
//!
//! ```text
//! score = -BD_rate_BBS + ssim2_penalty * max(0, BD_rate_SSIM2)
//! ```
//!
//! Negative BD-rate means "smaller file at the same quality" (i.e. a
//! win). We reward BBS reduction (block-seam quality gain) and
//! heavily penalize SSIM2 regression. The penalty multiplier defaults
//! to `5.0`; tune with `--ssim2-penalty`.
//!
//! # Usage
//!
//! ```bash
//! # Stage A — sweep α on a fixed (threshold, shrink, retries):
//! cargo run --release -p zenjpeg --features "decoder trellis" \
//!   --example boundary_rd_sweep -- \
//!   --stage alpha \
//!   --corpus cid22:2,screenshots:1,synthetic:2 \
//!   --qualities 50,75,90 \
//!   --output-dir benchmarks/rd_compare/2026-04-20-phase5/stage_alpha
//!
//! # Stage B — sweep threshold/shrink with best α:
//! cargo run --release -p zenjpeg --features "decoder trellis" \
//!   --example boundary_rd_sweep -- \
//!   --stage thresh_shrink --alpha 1.0 \
//!   --corpus cid22:2,screenshots:1,synthetic:2 \
//!   --qualities 50,75,90 \
//!   --output-dir benchmarks/rd_compare/2026-04-20-phase5/stage_thresh_shrink
//!
//! # Stage C — sweep retries with best (α, threshold, shrink):
//! cargo run --release -p zenjpeg --features "decoder trellis" \
//!   --example boundary_rd_sweep -- \
//!   --stage retries --alpha 1.0 --threshold 0.1 --shrink 0.7 \
//!   --output-dir benchmarks/rd_compare/2026-04-20-phase5/stage_retries
//! ```
//!
//! Each run emits a CSV grid + a stdout summary. The CSV columns are:
//!
//! `image,class,config,alpha,threshold,shrink,retries,quality,bytes,bpp,\
//!  encode_ms,ssim2_distortion,bbs_distortion`
//!
//! Per-config aggregates (BD-rate per metric per class + composite)
//! land in `summary.csv`.

use enough::Unstoppable;
use imgref::{ImgRef, ImgVec};
use rgb::RGB;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::metrics::rd::{self, RdCurve, RdPoint};
use zenjpeg::metrics::sweep::{CorpusImage, ImageClass};

#[derive(Debug, Clone)]
struct Args {
    stage: String,
    alpha: Option<f32>,
    threshold: Option<f32>,
    shrink: Option<f32>,
    retries: Option<u8>,
    corpus_specs: Vec<CorpusSpec>,
    qualities: Vec<u8>,
    output_dir: PathBuf,
    ssim2_penalty: f64,
    // Optional explicit list of configs via repeated
    // `--config alpha,threshold,shrink,retries` args. If empty, the
    // stage drives the grid.
    explicit_configs: Vec<ConfigKnob>,
}

#[derive(Debug, Clone, Copy)]
struct ConfigKnob {
    alpha: f32,
    threshold: f32,
    shrink: f32,
    retries: u8,
}

impl ConfigKnob {
    fn label(&self) -> String {
        format!(
            "a{:.2}_t{:.2}_s{:.2}_r{}",
            self.alpha, self.threshold, self.shrink, self.retries
        )
    }
}

#[derive(Debug, Clone)]
struct CorpusSpec {
    kind: String,
    count: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        stage: "alpha".to_owned(),
        alpha: None,
        threshold: None,
        shrink: None,
        retries: None,
        corpus_specs: vec![
            CorpusSpec { kind: "cid22".into(), count: 2 },
            CorpusSpec { kind: "screenshots".into(), count: 1 },
            CorpusSpec { kind: "synthetic".into(), count: 2 },
        ],
        qualities: vec![50, 75, 90],
        output_dir: PathBuf::from("benchmarks/rd_compare/phase5_sweep"),
        ssim2_penalty: 5.0,
        explicit_configs: Vec::new(),
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--stage" => { i += 1; a.stage = argv[i].clone(); }
            "--alpha" => { i += 1; a.alpha = argv[i].parse().ok(); }
            "--threshold" => { i += 1; a.threshold = argv[i].parse().ok(); }
            "--shrink" => { i += 1; a.shrink = argv[i].parse().ok(); }
            "--retries" => { i += 1; a.retries = argv[i].parse().ok(); }
            "--corpus" => {
                i += 1;
                a.corpus_specs = argv[i]
                    .split(',')
                    .filter_map(|item| {
                        let (k, n) = item.split_once(':')?;
                        Some(CorpusSpec { kind: k.to_owned(), count: n.parse().ok()? })
                    })
                    .collect();
            }
            "--qualities" => {
                i += 1;
                a.qualities = argv[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--output-dir" => { i += 1; a.output_dir = PathBuf::from(&argv[i]); }
            "--ssim2-penalty" => { i += 1; a.ssim2_penalty = argv[i].parse().unwrap_or(5.0); }
            "--config" => {
                i += 1;
                let parts: Vec<&str> = argv[i].split(',').collect();
                if parts.len() == 4 {
                    if let (Ok(al), Ok(t), Ok(s), Ok(r)) = (
                        parts[0].parse::<f32>(),
                        parts[1].parse::<f32>(),
                        parts[2].parse::<f32>(),
                        parts[3].parse::<u8>(),
                    ) {
                        a.explicit_configs.push(ConfigKnob {
                            alpha: al, threshold: t, shrink: s, retries: r,
                        });
                    }
                }
            }
            other => {
                eprintln!("warning: unrecognised arg `{}`", other);
            }
        }
        i += 1;
    }
    a
}

fn stage_grid(args: &Args) -> Vec<ConfigKnob> {
    if !args.explicit_configs.is_empty() {
        return args.explicit_configs.clone();
    }
    // Default anchor values — overridden by the per-stage grid.
    let a = args.alpha.unwrap_or(1.0);
    let t = args.threshold.unwrap_or(0.1);
    let s = args.shrink.unwrap_or(0.7);
    let r = args.retries.unwrap_or(1);
    match args.stage.as_str() {
        "alpha" => [0.25_f32, 0.5, 1.0, 2.0, 4.0]
            .iter()
            .map(|&al| ConfigKnob { alpha: al, threshold: t, shrink: s, retries: r })
            .collect(),
        "thresh_shrink" => {
            let mut v = Vec::new();
            for &th in &[0.05_f32, 0.1, 0.2] {
                for &sh in &[0.5_f32, 0.7, 0.85] {
                    v.push(ConfigKnob { alpha: a, threshold: th, shrink: sh, retries: r });
                }
            }
            v
        }
        "retries" => [1u8, 2]
            .iter()
            .map(|&rr| ConfigKnob { alpha: a, threshold: t, shrink: s, retries: rr })
            .collect(),
        "validate" => vec![ConfigKnob { alpha: a, threshold: t, shrink: s, retries: r }],
        other => {
            eprintln!("unknown stage `{}`; defaulting to alpha sweep", other);
            [0.25_f32, 0.5, 1.0, 2.0, 4.0]
                .iter()
                .map(|&al| ConfigKnob { alpha: al, threshold: t, shrink: s, retries: r })
                .collect()
        }
    }
}

// ------- Corpus loading (shared shape with rd_compare.rs) -------

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

fn collect_corpus(specs: &[CorpusSpec]) -> Vec<CorpusImage> {
    let mut out = Vec::new();
    for s in specs {
        let images = match s.kind.as_str() {
            "cid22" => collect_cid22(s.count),
            "screenshots" => collect_screenshots(s.count),
            "synthetic" => generate_synthetic(s.count),
            _ => Vec::new(),
        };
        out.extend(images);
    }
    out
}

fn collect_cid22(n: usize) -> Vec<CorpusImage> {
    let root = home().join("work/codec-eval/codec-corpus/CID22/CID22-512/validation");
    dir_pngs(&root, n, ImageClass::Photo)
}

fn collect_screenshots(n: usize) -> Vec<CorpusImage> {
    let root = home().join("work/codec-eval/codec-corpus/gb82-sc");
    let Ok(entries) = fs::read_dir(&root) else { return Vec::new(); };
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
    let mut out = Vec::new();
    for p in paths.into_iter().take(n) {
        if let Some(mut img) = load_image_as_corpus(&p, ImageClass::Screenshot) {
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
    let generators: Vec<(&str, ImageClass, fn(usize, usize) -> Vec<u8>)> = vec![
        ("synth_grid", ImageClass::LineArt, gen_grid),
        ("synth_stripes", ImageClass::LineArt, gen_stripes),
        ("synth_noise_edges", ImageClass::Synthetic, gen_noise_edges),
    ];
    let size = 384usize;
    generators
        .into_iter()
        .take(n)
        .map(|(label, class, g)| CorpusImage {
            label: label.to_owned(),
            width: size,
            height: size,
            rgb8: g(size, size),
            class,
        })
        .collect()
}

fn gen_stripes(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![255u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
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

fn load_image_as_corpus(path: &Path, class: ImageClass) -> Option<CorpusImage> {
    let (rgb8, w, h) = load_rgb(path)?;
    let label = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    Some(CorpusImage { label, width: w, height: h, rgb8, class })
}

fn dir_pngs(root: &Path, n: usize, class: ImageClass) -> Vec<CorpusImage> {
    let Ok(entries) = fs::read_dir(root) else { return Vec::new(); };
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
        .filter_map(|p| load_image_as_corpus(&p, class))
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

// ------- Encode / decode / metrics -------

fn build_config(knob: Option<ConfigKnob>, quality: u8) -> EncoderConfig {
    let mut c = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter);
    if let Some(k) = knob {
        use zenjpeg::encoder::{BoundaryRd, BoundaryRdConfig};
        c = c.boundary_rd(BoundaryRd::Manual(BoundaryRdConfig {
            alpha: k.alpha,
            threshold: k.threshold,
            shrink: k.shrink,
            max_retries: k.retries,
            above: false,
        }));
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
    // Returns (ssim2_distortion, bbs_total).
    let orig_arr: Vec<[u8; 3]> = orig.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let recon_arr: Vec<[u8; 3]> = recon.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let o = ImgVec::new(orig_arr, w, h);
    let r = ImgVec::new(recon_arr, w, h);
    let ssim2 = fast_ssim2::compute_ssimulacra2(o.as_ref(), r.as_ref()).unwrap_or(0.0);

    let orig_rgb: Vec<RGB<u8>> = orig.chunks_exact(3).map(|c| RGB { r: c[0], g: c[1], b: c[2] }).collect();
    let recon_rgb: Vec<RGB<u8>> = recon.chunks_exact(3).map(|c| RGB { r: c[0], g: c[1], b: c[2] }).collect();
    let orig_img: ImgRef<'_, RGB<u8>> = ImgRef::new(&orig_rgb, w, h);
    let recon_img: ImgRef<'_, RGB<u8>> = ImgRef::new(&recon_rgb, w, h);
    let bbs = zenjpeg::metrics::bbs::bbs_rgb8(recon_img, orig_img);

    // Distortion convention: smaller is better.
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

fn main() {
    let args = parse_args();
    let images = collect_corpus(&args.corpus_specs);
    if images.is_empty() {
        eprintln!("no corpus images loaded; aborting");
        std::process::exit(1);
    }
    let configs = stage_grid(&args);
    eprintln!(
        "loaded {} images, stage={}, {} configs, qualities={:?}",
        images.len(),
        args.stage,
        configs.len(),
        args.qualities
    );

    fs::create_dir_all(&args.output_dir).expect("mkdir output_dir");

    // 1) Encode the baseline (boundary_rd off) first — shared across configs.
    eprintln!("[baseline] encoding boundary_rd=off...");
    let mut baseline_points: BTreeMap<String, Vec<SamplePoint>> = BTreeMap::new();
    let t_base = Instant::now();
    for img in &images {
        let mut pts = Vec::new();
        for &q in &args.qualities {
            let cfg = build_config(None, q);
            let start = Instant::now();
            let Some(jpeg) = encode_jpeg(cfg, img) else { continue };
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            let Some(recon) = decode_jpeg_rgb(&jpeg, img.width, img.height) else { continue };
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
    eprintln!("[baseline] {} images encoded in {:.1}s",
        images.len(),
        t_base.elapsed().as_secs_f64()
    );

    // 2) For each candidate config, encode all (image, quality), record samples.
    let mut grid_rows = Vec::<String>::new();
    grid_rows.push(
        "image,class,config,alpha,threshold,shrink,retries,quality,\
         bytes,bpp,encode_ms,ssim2_distortion,bbs_distortion"
            .to_owned(),
    );

    // baseline rows too, so the CSV is self-contained for per-image RD plotting
    for img in &images {
        if let Some(pts) = baseline_points.get(&img.label) {
            for p in pts {
                grid_rows.push(format!(
                    "{},{},baseline,,,,,{},{},{:.6},{:.3},{:.6},{:.6}",
                    img.label,
                    img.class.slug(),
                    p.quality,
                    p.bytes,
                    p.bpp,
                    p.encode_ms,
                    p.ssim2_distortion,
                    p.bbs_distortion,
                ));
            }
        }
    }

    #[derive(Debug, Clone)]
    struct ConfigSummary {
        knob: ConfigKnob,
        bd_bbs_by_class: BTreeMap<String, Vec<f64>>,
        bd_ssim2_by_class: BTreeMap<String, Vec<f64>>,
        mean_encode_ms_ratio: f64,
    }

    let mut summaries: Vec<ConfigSummary> = Vec::new();

    let total = configs.len() * images.len() * args.qualities.len();
    let mut done = 0;
    let t0 = Instant::now();

    for knob in &configs {
        let label = knob.label();
        eprintln!("[cfg] {}", label);
        let mut cand_points: BTreeMap<String, Vec<SamplePoint>> = BTreeMap::new();
        let mut encode_ms_cand_total = 0.0;
        let mut encode_ms_base_total = 0.0;
        for img in &images {
            let mut pts = Vec::new();
            for &q in &args.qualities {
                let cfg = build_config(Some(*knob), q);
                let start = Instant::now();
                let Some(jpeg) = encode_jpeg(cfg, img) else {
                    done += 1; continue;
                };
                let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
                encode_ms_cand_total += encode_ms;
                // matching baseline encode_ms for ratio
                if let Some(base_pts) = baseline_points.get(&img.label)
                    && let Some(bp) = base_pts.iter().find(|p| p.quality == q)
                {
                    encode_ms_base_total += bp.encode_ms;
                }
                let Some(recon) = decode_jpeg_rgb(&jpeg, img.width, img.height) else {
                    done += 1; continue;
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
                grid_rows.push(format!(
                    "{},{},{},{:.4},{:.4},{:.4},{},{},{},{:.6},{:.3},{:.6},{:.6}",
                    img.label,
                    img.class.slug(),
                    label,
                    knob.alpha,
                    knob.threshold,
                    knob.shrink,
                    knob.retries,
                    q,
                    jpeg.len(),
                    jpeg.len() as f64 * 8.0 / pixels,
                    encode_ms,
                    ssim2_d,
                    bbs_d,
                ));
                done += 1;
                if done % 10 == 0 {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let rate = done as f64 / elapsed.max(1e-6);
                    let eta = if rate > 0.0 { (total - done) as f64 / rate } else { 0.0 };
                    eprintln!("  [{}/{}] {:.1}/s elapsed={:.1}s eta={:.1}s",
                        done, total, rate, elapsed, eta);
                }
            }
            cand_points.insert(img.label.clone(), pts);
        }

        // Compute per-image BD-rate, aggregated by class.
        let mut bd_bbs_by_class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut bd_ssim2_by_class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for img in &images {
            let (Some(base_pts), Some(cand_pts)) =
                (baseline_points.get(&img.label), cand_points.get(&img.label))
            else { continue };
            let base_bbs = RdCurve::from_points(base_pts.iter().map(|p| RdPoint {
                rate_bpp: p.bpp,
                distortion: p.bbs_distortion,
                quality: p.quality,
            }));
            let cand_bbs = RdCurve::from_points(cand_pts.iter().map(|p| RdPoint {
                rate_bpp: p.bpp,
                distortion: p.bbs_distortion,
                quality: p.quality,
            }));
            let base_ssim2 = RdCurve::from_points(base_pts.iter().map(|p| RdPoint {
                rate_bpp: p.bpp,
                distortion: p.ssim2_distortion,
                quality: p.quality,
            }));
            let cand_ssim2 = RdCurve::from_points(cand_pts.iter().map(|p| RdPoint {
                rate_bpp: p.bpp,
                distortion: p.ssim2_distortion,
                quality: p.quality,
            }));
            let class_slug = img.class.slug().to_owned();
            if let Some(bd) = rd::bd_rate(&base_bbs, &cand_bbs) {
                bd_bbs_by_class.entry(class_slug.clone()).or_default().push(bd);
            }
            if let Some(bd) = rd::bd_rate(&base_ssim2, &cand_ssim2) {
                bd_ssim2_by_class.entry(class_slug).or_default().push(bd);
            }
        }
        let ratio = if encode_ms_base_total > 0.0 {
            encode_ms_cand_total / encode_ms_base_total
        } else { 1.0 };
        summaries.push(ConfigSummary {
            knob: *knob,
            bd_bbs_by_class,
            bd_ssim2_by_class,
            mean_encode_ms_ratio: ratio,
        });
    }

    // Write raw grid CSV.
    let grid_path = args.output_dir.join("grid.csv");
    fs::write(&grid_path, grid_rows.join("\n") + "\n").expect("write grid.csv");
    eprintln!("wrote {}", grid_path.display());

    // Summary CSV.
    let summary_path = args.output_dir.join("summary.csv");
    let mut s = String::new();
    s.push_str("config,alpha,threshold,shrink,retries,\
                bbs_bd_photo,bbs_bd_screenshot,bbs_bd_lineart,bbs_bd_synthetic,bbs_bd_overall,\
                ssim2_bd_photo,ssim2_bd_screenshot,ssim2_bd_lineart,ssim2_bd_synthetic,ssim2_bd_overall,\
                composite_score,encode_ms_ratio\n");
    let mut ranked: Vec<(f64, String)> = Vec::new();
    for sm in &summaries {
        let bbs_photo = mean(sm.bd_bbs_by_class.get("photo"));
        let bbs_screen = mean(sm.bd_bbs_by_class.get("screenshot"));
        let bbs_line = mean(sm.bd_bbs_by_class.get("lineart"));
        let bbs_synth = mean(sm.bd_bbs_by_class.get("synthetic"));
        let bbs_overall = mean_all(&sm.bd_bbs_by_class);
        let ss_photo = mean(sm.bd_ssim2_by_class.get("photo"));
        let ss_screen = mean(sm.bd_ssim2_by_class.get("screenshot"));
        let ss_line = mean(sm.bd_ssim2_by_class.get("lineart"));
        let ss_synth = mean(sm.bd_ssim2_by_class.get("synthetic"));
        let ss_overall = mean_all(&sm.bd_ssim2_by_class);

        // Composite: reward BBS reduction, penalize SSIM2 regression.
        // Both BD-rates use "negative = better" convention.
        let ssim2_penalty = args.ssim2_penalty;
        let ssim2_regression = ss_overall.unwrap_or(0.0).max(0.0);
        let score = -bbs_overall.unwrap_or(0.0) - ssim2_penalty * ssim2_regression;

        let line = format!(
            "{},{:.4},{:.4},{:.4},{},\
             {},{},{},{},{},\
             {},{},{},{},{},\
             {:.4},{:.3}\n",
            sm.knob.label(),
            sm.knob.alpha,
            sm.knob.threshold,
            sm.knob.shrink,
            sm.knob.retries,
            fmt_opt(bbs_photo), fmt_opt(bbs_screen), fmt_opt(bbs_line), fmt_opt(bbs_synth), fmt_opt(bbs_overall),
            fmt_opt(ss_photo), fmt_opt(ss_screen), fmt_opt(ss_line), fmt_opt(ss_synth), fmt_opt(ss_overall),
            score,
            sm.mean_encode_ms_ratio,
        );
        s.push_str(&line);
        ranked.push((score, sm.knob.label()));
    }
    fs::write(&summary_path, &s).expect("write summary.csv");
    eprintln!("wrote {}", summary_path.display());

    // Stdout ranking.
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\n=== Top configs by composite score ===");
    for (score, label) in ranked.iter().take(10) {
        eprintln!("  {:>24}   score={:+.4}", label, score);
    }
    eprintln!();
    print!("{}", s);
}

fn mean(v: Option<&Vec<f64>>) -> Option<f64> {
    v.and_then(|x| if x.is_empty() { None } else {
        Some(x.iter().sum::<f64>() / x.len() as f64)
    })
}

fn mean_all(m: &BTreeMap<String, Vec<f64>>) -> Option<f64> {
    let (sum, n) = m
        .values()
        .flat_map(|v| v.iter().copied())
        .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { None } else { Some(sum / n as f64) }
}

fn fmt_opt(x: Option<f64>) -> String {
    match x {
        Some(v) => format!("{:+.4}", v),
        None => "NA".to_owned(),
    }
}
