//! Zero-bias-shrink sweep for boundary-RD (Task 6 of #102 rollup).
//!
//! Explores the new `zero_bias_shrink` knob added alongside the existing
//! `aq_shrink`. Both control the retry-path quantize in boundary-RD:
//!
//! - `aq_shrink ∈ (0, 1]` — multiplies the AQ strength per retry.
//! - `zero_bias_shrink ∈ (0, 1]` — scales the zero-bias threshold
//!   (`offset + mul * aq`) by a constant factor.
//!
//! Empirical question: does `zero_bias_shrink` add signal over `aq_shrink`
//! alone, or are they collinear?
//!
//! # Grid
//!
//! ```text
//! α                ∈ {1.0, 2.0}
//! threshold        ∈ {0.02, 0.10}
//! aq_shrink        ∈ {0.5, 1.0}     // 1.0 = AQ-shrink disabled
//! zero_bias_shrink ∈ {0.3, 0.5, 1.0} // 1.0 = ZB-shrink disabled
//! max_retries      ∈ {1, 2}
//! above            = true           // phase-5 winner, not re-swept
//! ```
//!
//! Cells where BOTH `aq_shrink == 1.0 AND zero_bias_shrink == 1.0` are
//! skipped — they are equivalent to `BoundaryRd::Off` (no retry effect).
//! Total configs: `2×2×(2×3 - 1)×2 = 40`.
//!
//! # Corpus
//!
//! Reads `benchmarks/boundary_rd/sweep_corpus/manifest.tsv` — a
//! `class\tpath\tsha256` TSV referencing 20-30 line-art / screen-content
//! / doc-text / palette-rich images on local block storage. The four
//! committed synthetic PNGs in `sweep_corpus/lineart/` are included via
//! relative paths; external corpus images via absolute paths.
//!
//! Images over 512×512 are center-cropped to 512 to keep wall-clock
//! predictable — the feature matters most at low-to-mid image sizes
//! anyway (block-boundary artifacts are relative to iMCU geometry).
//!
//! # Output
//!
//! Written to `benchmarks/boundary_rd/zero_bias_sweep/`:
//!
//! - `grid.csv` — every (image, config, quality) row.
//! - `per_class_per_q.csv` — BD-rate aggregated per
//!   (class_bucket, q_range) × config.
//! - `best_per_class_per_q.csv` — top config per
//!   (class_bucket, q_range) by composite score.
//! - `README.md` — written by the caller, not this harness; the
//!   CSVs are the authoritative data.
//!
//! # Score
//!
//! Composite: `-BD_BBS - 5 * max(0, BD_SSIM2)` — same weighting used
//! in the Phase-5 low-Q sweep. Reduces to "minimize bytes at equal
//! BBS; never take a positive SSIM2 regression without paying 5×".

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
use zenjpeg_bench_utils::rd::{self, RdCurve, RdPoint};

// ------------------------- config grid -------------------------

#[derive(Debug, Clone, Copy)]
struct Knob {
    alpha: f32,
    threshold: f32,
    aq_shrink: f32,
    zb_shrink: f32,
    retries: u8,
}

impl Knob {
    fn label(&self) -> String {
        format!(
            "a{:.1}_t{:.2}_aq{:.1}_zb{:.1}_r{}",
            self.alpha, self.threshold, self.aq_shrink, self.zb_shrink, self.retries
        )
    }
}

fn build_grid() -> Vec<Knob> {
    let mut v = Vec::new();
    for &alpha in &[1.0_f32, 2.0] {
        for &threshold in &[0.02_f32, 0.10] {
            for &aq in &[0.5_f32, 1.0] {
                for &zb in &[0.3_f32, 0.5, 1.0] {
                    // Skip cells where both shrinks are 1.0 — equivalent
                    // to BoundaryRd::Off with retries > 0 (no effect).
                    if (aq - 1.0).abs() < 1e-6 && (zb - 1.0).abs() < 1e-6 {
                        continue;
                    }
                    for &r in &[1u8, 2] {
                        v.push(Knob {
                            alpha,
                            threshold,
                            aq_shrink: aq,
                            zb_shrink: zb,
                            retries: r,
                        });
                    }
                }
            }
        }
    }
    v
}

// ------------------------- corpus loader -------------------------

#[derive(Debug)]
struct CorpusImage {
    label: String,
    class: String,
    width: usize,
    height: usize,
    rgb8: Vec<u8>,
}

fn load_rgb(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    // `image` crate auto-detects PNG / TIFF / JPEG / etc. from extension.
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    Some((rgb.into_raw(), w, h))
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
        class: img.class.clone(),
        width: new_w,
        height: new_h,
        rgb8: out,
    }
}

fn load_manifest(manifest: &Path, corpus_root: &Path) -> Vec<CorpusImage> {
    let mut out = Vec::new();
    let Ok(content) = fs::read_to_string(manifest) else {
        eprintln!("[corpus] cannot read manifest {}", manifest.display());
        return out;
    };
    for (i, line) in content.lines().enumerate() {
        if i == 0 && line.starts_with("class\t") {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            continue;
        }
        let class = parts[0].to_string();
        let raw_path = parts[1];
        // Resolve committed synthetics via repo-relative path under sweep_corpus.
        let path = if raw_path.starts_with('/') {
            PathBuf::from(raw_path)
        } else {
            corpus_root.join(raw_path)
        };
        if !path.exists() {
            eprintln!("[corpus] missing {} (skipped)", path.display());
            continue;
        }
        let Some((rgb8, w, h)) = load_rgb(&path) else {
            eprintln!("[corpus] failed to decode {}", path.display());
            continue;
        };
        let label = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let mut img = CorpusImage {
            label,
            class,
            width: w,
            height: h,
            rgb8,
        };
        // Cap at 512×512 center crop.
        if img.width > 512 || img.height > 512 {
            img = crop_center(&img, img.width.min(512), img.height.min(512));
        }
        out.push(img);
    }
    out
}

// ------------------------- encode / metrics -------------------------

fn build_config(knob: Option<Knob>, quality: u8) -> EncoderConfig {
    let mut c = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter);
    if let Some(k) = knob {
        c = c.boundary_rd(BoundaryRd::On(
            BoundaryRdConfig::new()
                .with_alpha(k.alpha)
                .with_threshold(k.threshold)
                .with_aq_shrink(k.aq_shrink)
                .with_zero_bias_shrink(k.zb_shrink)
                .with_max_retries(k.retries)
                .with_above(true),
        ));
    }
    c
}

fn encode_jpeg(cfg: EncoderConfig, img: &CorpusImage) -> Option<Vec<u8>> {
    let mut enc = cfg
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
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = dec.decode().ok()?;
    if pixels.len() != w * h * 3 {
        return None;
    }
    Some(pixels)
}

fn metrics(orig: &[u8], recon: &[u8], w: usize, h: usize) -> (f64, f64) {
    let orig_arr: Vec<[u8; 3]> = orig.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let recon_arr: Vec<[u8; 3]> = recon.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
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
    let o_rgb: ImgRef<'_, RGB<u8>> = ImgRef::new(&orig_rgb, w, h);
    let r_rgb: ImgRef<'_, RGB<u8>> = ImgRef::new(&recon_rgb, w, h);
    let bbs = zenjpeg_bench_utils::bbs::bbs_rgb8(r_rgb, o_rgb);

    (100.0 - ssim2, bbs.total)
}

#[derive(Debug, Clone, Copy)]
struct SamplePoint {
    bpp: f64,
    bytes: usize,
    ssim2_d: f64,
    bbs_d: f64,
    quality: u8,
    encode_ms: f64,
}

fn q_range(q: u8) -> &'static str {
    if q <= 30 {
        "low"
    } else if q <= 60 {
        "mid"
    } else {
        "high"
    }
}

fn class_bucket(class: &str) -> &str {
    // We keep per-class detail but also bucket a couple of related
    // classes together for the small-sample aggregation.
    match class {
        "bilevel" | "screencontent" | "doc-text" | "map-detail"
        | "palette-rich" | "mixed-vector" => class,
        _ => "other",
    }
}

// ------------------------- main -------------------------

struct Args {
    output_dir: PathBuf,
    manifest: PathBuf,
    corpus_root: PathBuf,
    qualities: Vec<u8>,
    limit_images: Option<usize>,
    limit_configs: Option<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        output_dir: PathBuf::from("benchmarks/boundary_rd/zero_bias_sweep"),
        manifest: PathBuf::from("benchmarks/boundary_rd/sweep_corpus/manifest.tsv"),
        corpus_root: PathBuf::from("benchmarks/boundary_rd/sweep_corpus"),
        qualities: vec![5, 15, 30, 45, 60, 75, 85, 95],
        limit_images: None,
        limit_configs: None,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--output" | "--output-dir" => {
                i += 1;
                a.output_dir = PathBuf::from(&argv[i]);
            }
            "--manifest" => {
                i += 1;
                a.manifest = PathBuf::from(&argv[i]);
            }
            "--corpus-root" => {
                i += 1;
                a.corpus_root = PathBuf::from(&argv[i]);
            }
            "--qualities" => {
                i += 1;
                a.qualities = argv[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--limit-images" => {
                i += 1;
                a.limit_images = argv[i].parse().ok();
            }
            "--limit-configs" => {
                i += 1;
                a.limit_configs = argv[i].parse().ok();
            }
            other => eprintln!("warn: unknown arg `{}`", other),
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();
    let mut images = load_manifest(&args.manifest, &args.corpus_root);
    if let Some(lim) = args.limit_images {
        images.truncate(lim);
    }
    if images.is_empty() {
        eprintln!("no corpus images; aborting");
        std::process::exit(1);
    }
    let mut configs = build_grid();
    if let Some(lim) = args.limit_configs {
        configs.truncate(lim);
    }
    eprintln!(
        "[plan] {} images × {} qualities × {} configs = {} candidate encodes (+{} baseline)",
        images.len(),
        args.qualities.len(),
        configs.len(),
        images.len() * args.qualities.len() * configs.len(),
        images.len() * args.qualities.len(),
    );

    fs::create_dir_all(&args.output_dir).expect("mkdir output");

    // Baseline (boundary-RD off).
    eprintln!("[baseline] encoding...");
    let t_base = Instant::now();
    let mut baseline_points: BTreeMap<String, Vec<SamplePoint>> = BTreeMap::new();
    for img in &images {
        let mut pts = Vec::new();
        for &q in &args.qualities {
            let cfg = build_config(None, q);
            let start = Instant::now();
            let Some(jpeg) = encode_jpeg(cfg, img) else {
                continue;
            };
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            let Some(recon) = decode_jpeg_rgb(&jpeg, img.width, img.height) else {
                continue;
            };
            let (ssim2_d, bbs_d) = metrics(&img.rgb8, &recon, img.width, img.height);
            let pixels = (img.width * img.height).max(1) as f64;
            pts.push(SamplePoint {
                bpp: jpeg.len() as f64 * 8.0 / pixels,
                bytes: jpeg.len(),
                ssim2_d,
                bbs_d,
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

    // Grid CSV rows.
    let mut grid_rows = Vec::<String>::new();
    grid_rows.push(
        "image,class,class_bucket,config,alpha,threshold,aq_shrink,zb_shrink,retries,\
         quality,q_range,bytes,bpp,encode_ms,ssim2_distortion,bbs_distortion"
            .into(),
    );
    for img in &images {
        if let Some(pts) = baseline_points.get(&img.label) {
            for p in pts {
                grid_rows.push(format!(
                    "{},{},{},baseline,,,,,,{},{},{},{:.6},{:.3},{:.6},{:.6}",
                    img.label,
                    img.class,
                    class_bucket(&img.class),
                    p.quality,
                    q_range(p.quality),
                    p.bytes,
                    p.bpp,
                    p.encode_ms,
                    p.ssim2_d,
                    p.bbs_d,
                ));
            }
        }
    }

    // Candidate encodes.
    let total = configs.len() * images.len() * args.qualities.len();
    let mut done = 0usize;
    let t0 = Instant::now();
    let mut per_cfg_points: BTreeMap<String, BTreeMap<String, Vec<SamplePoint>>> =
        BTreeMap::new();
    for (ci, k) in configs.iter().enumerate() {
        let label = k.label();
        let cfg_start = Instant::now();
        let mut cand: BTreeMap<String, Vec<SamplePoint>> = BTreeMap::new();
        for img in &images {
            let mut pts = Vec::new();
            for &q in &args.qualities {
                let cfg = build_config(Some(*k), q);
                let start = Instant::now();
                let Some(jpeg) = encode_jpeg(cfg, img) else {
                    done += 1;
                    continue;
                };
                let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
                let Some(recon) = decode_jpeg_rgb(&jpeg, img.width, img.height) else {
                    done += 1;
                    continue;
                };
                let (ssim2_d, bbs_d) = metrics(&img.rgb8, &recon, img.width, img.height);
                let pixels = (img.width * img.height).max(1) as f64;
                pts.push(SamplePoint {
                    bpp: jpeg.len() as f64 * 8.0 / pixels,
                    bytes: jpeg.len(),
                    ssim2_d,
                    bbs_d,
                    quality: q,
                    encode_ms,
                });
                grid_rows.push(format!(
                    "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{:.6},{:.3},{:.6},{:.6}",
                    img.label,
                    img.class,
                    class_bucket(&img.class),
                    label,
                    k.alpha,
                    k.threshold,
                    k.aq_shrink,
                    k.zb_shrink,
                    k.retries,
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
            cand.insert(img.label.clone(), pts);
        }
        per_cfg_points.insert(label.clone(), cand);

        let elapsed = t0.elapsed().as_secs_f64();
        let rate = done as f64 / elapsed.max(1e-6);
        let eta = if rate > 0.0 { (total - done) as f64 / rate } else { 0.0 };
        eprintln!(
            "[cfg {:>3}/{:>3}] {:<30} t={:.1}s done={}/{} rate={:.1}/s eta={:.0}s",
            ci + 1,
            configs.len(),
            label,
            cfg_start.elapsed().as_secs_f64(),
            done,
            total,
            rate,
            eta,
        );

        // Periodic checkpoint.
        if (ci + 1) % 8 == 0 {
            let ck = args.output_dir.join("grid.checkpoint.csv");
            let _ = fs::write(&ck, grid_rows.join("\n") + "\n");
        }
    }

    // Write grid.csv.
    let grid_path = args.output_dir.join("grid.csv");
    fs::write(&grid_path, grid_rows.join("\n") + "\n").expect("write grid.csv");
    eprintln!("wrote {}", grid_path.display());
    let _ = fs::remove_file(args.output_dir.join("grid.checkpoint.csv"));

    // Aggregate per-(class_bucket, q_range).
    let q_ranges = ["low", "mid", "high"];
    // All classes present in the manifest get an aggregation row.
    let mut class_set = std::collections::BTreeSet::<String>::new();
    for img in &images {
        class_set.insert(class_bucket(&img.class).to_string());
    }
    let class_buckets: Vec<String> = class_set.into_iter().collect();

    #[derive(Debug, Clone)]
    struct Agg {
        config: String,
        knob: Knob,
        bd_bbs: Option<f64>,
        bd_ssim2: Option<f64>,
        n_images: usize,
        bytes_ratio: f64,
        encode_ratio: f64,
        bbs_ratio: f64,
        ssim2_ratio: f64,
    }

    let mut per_rows: Vec<String> = Vec::new();
    per_rows.push(
        "class_bucket,q_range,config,alpha,threshold,aq_shrink,zb_shrink,retries,\
         n_images,bd_bbs,bd_ssim2,bytes_ratio,encode_ms_ratio,\
         bbs_ratio,ssim2_dist_ratio,composite_score"
            .into(),
    );
    let mut best_rows: Vec<String> = Vec::new();
    best_rows.push(
        "class_bucket,q_range,best_config,alpha,threshold,aq_shrink,zb_shrink,retries,\
         n_images,bd_bbs,bd_ssim2,bytes_ratio,encode_ms_ratio,\
         bbs_ratio,ssim2_dist_ratio,composite_score"
            .into(),
    );

    for bucket in &class_buckets {
        for qr in &q_ranges {
            let q_filter: Vec<u8> = args
                .qualities
                .iter()
                .copied()
                .filter(|&q| q_range(q) == *qr)
                .collect();
            let imgs_in: Vec<&CorpusImage> = images
                .iter()
                .filter(|i| class_bucket(&i.class) == bucket.as_str())
                .collect();
            if imgs_in.is_empty() || q_filter.is_empty() {
                continue;
            }

            let mut aggs: Vec<Agg> = Vec::new();
            for knob in &configs {
                let cfg_label = knob.label();
                let Some(cand_map) = per_cfg_points.get(&cfg_label) else { continue };

                let mut bd_bbs_list = Vec::new();
                let mut bd_ssim2_list = Vec::new();
                let mut bytes_ratios = Vec::new();
                let mut enc_ratios = Vec::new();
                let mut bbs_ratios = Vec::new();
                let mut ssim2_ratios = Vec::new();
                let mut n_img = 0usize;

                for img in &imgs_in {
                    let Some(base_pts) = baseline_points.get(&img.label) else { continue };
                    let Some(cand_pts) = cand_map.get(&img.label) else { continue };

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

                    let base_bbs_curve: Vec<RdPoint> = base_q
                        .iter()
                        .map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.bbs_d,
                            quality: p.quality,
                        })
                        .collect();
                    let cand_bbs_curve: Vec<RdPoint> = cand_q
                        .iter()
                        .map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.bbs_d,
                            quality: p.quality,
                        })
                        .collect();
                    let base_ssim_curve: Vec<RdPoint> = base_q
                        .iter()
                        .map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.ssim2_d,
                            quality: p.quality,
                        })
                        .collect();
                    let cand_ssim_curve: Vec<RdPoint> = cand_q
                        .iter()
                        .map(|p| RdPoint {
                            rate_bpp: p.bpp,
                            distortion: p.ssim2_d,
                            quality: p.quality,
                        })
                        .collect();

                    let base_rd = RdCurve::from_points(base_bbs_curve);
                    let cand_rd = RdCurve::from_points(cand_bbs_curve);
                    if let Some(bd) = rd::bd_rate(&base_rd, &cand_rd) {
                        bd_bbs_list.push(bd);
                    }
                    let base_rd = RdCurve::from_points(base_ssim_curve);
                    let cand_rd = RdCurve::from_points(cand_ssim_curve);
                    if let Some(bd) = rd::bd_rate(&base_rd, &cand_rd) {
                        bd_ssim2_list.push(bd);
                    }

                    // Mean pointwise ratios for the per-q band.
                    let mut base_bytes = 0u64;
                    let mut cand_bytes = 0u64;
                    let mut base_enc = 0.0;
                    let mut cand_enc = 0.0;
                    let mut base_bbs = 0.0;
                    let mut cand_bbs = 0.0;
                    let mut base_ssim = 0.0;
                    let mut cand_ssim = 0.0;
                    let n = base_q.len().min(cand_q.len()).max(1);
                    for i in 0..n {
                        base_bytes += base_q[i].bytes as u64;
                        cand_bytes += cand_q[i].bytes as u64;
                        base_enc += base_q[i].encode_ms;
                        cand_enc += cand_q[i].encode_ms;
                        base_bbs += base_q[i].bbs_d;
                        cand_bbs += cand_q[i].bbs_d;
                        base_ssim += base_q[i].ssim2_d;
                        cand_ssim += cand_q[i].ssim2_d;
                    }
                    if base_bytes > 0 {
                        bytes_ratios.push(cand_bytes as f64 / base_bytes as f64);
                    }
                    if base_enc > 0.0 {
                        enc_ratios.push(cand_enc / base_enc);
                    }
                    if base_bbs > 0.0 {
                        bbs_ratios.push(cand_bbs / base_bbs);
                    }
                    if base_ssim > 0.0 {
                        ssim2_ratios.push(cand_ssim / base_ssim);
                    }
                    n_img += 1;
                }

                let mean = |v: &[f64]| {
                    if v.is_empty() {
                        f64::NAN
                    } else {
                        v.iter().copied().sum::<f64>() / v.len() as f64
                    }
                };
                let mean_opt = |v: &[f64]| {
                    if v.is_empty() {
                        None
                    } else {
                        Some(mean(v))
                    }
                };

                aggs.push(Agg {
                    config: cfg_label.clone(),
                    knob: *knob,
                    bd_bbs: mean_opt(&bd_bbs_list),
                    bd_ssim2: mean_opt(&bd_ssim2_list),
                    n_images: n_img,
                    bytes_ratio: mean(&bytes_ratios),
                    encode_ratio: mean(&enc_ratios),
                    bbs_ratio: mean(&bbs_ratios),
                    ssim2_ratio: mean(&ssim2_ratios),
                });
            }

            // Rank by composite score = -bd_bbs - 5*max(0,bd_ssim2).
            // NaN-safe by treating missing as 0 (conservative; zero signal).
            let score = |a: &Agg| -> f64 {
                let bd_bbs = a.bd_bbs.unwrap_or(0.0);
                let bd_ssim = a.bd_ssim2.unwrap_or(0.0);
                -bd_bbs - 5.0 * bd_ssim.max(0.0)
            };
            aggs.sort_by(|a, b| {
                score(b)
                    .partial_cmp(&score(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Emit all rows for this (bucket, qr).
            for a in &aggs {
                per_rows.push(format!(
                    "{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    bucket,
                    qr,
                    a.config,
                    a.knob.alpha,
                    a.knob.threshold,
                    a.knob.aq_shrink,
                    a.knob.zb_shrink,
                    a.knob.retries,
                    a.n_images,
                    a.bd_bbs
                        .map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "NA".into()),
                    a.bd_ssim2
                        .map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "NA".into()),
                    a.bytes_ratio,
                    a.encode_ratio,
                    a.bbs_ratio,
                    a.ssim2_ratio,
                    score(a),
                ));
            }
            if let Some(best) = aggs.first() {
                best_rows.push(format!(
                    "{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    bucket,
                    qr,
                    best.config,
                    best.knob.alpha,
                    best.knob.threshold,
                    best.knob.aq_shrink,
                    best.knob.zb_shrink,
                    best.knob.retries,
                    best.n_images,
                    best.bd_bbs
                        .map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "NA".into()),
                    best.bd_ssim2
                        .map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "NA".into()),
                    best.bytes_ratio,
                    best.encode_ratio,
                    best.bbs_ratio,
                    best.ssim2_ratio,
                    score(best),
                ));
            }
        }
    }

    let per_path = args.output_dir.join("per_class_per_q.csv");
    fs::write(&per_path, per_rows.join("\n") + "\n").expect("write per_class_per_q.csv");
    eprintln!("wrote {}", per_path.display());
    let best_path = args.output_dir.join("best_per_class_per_q.csv");
    fs::write(&best_path, best_rows.join("\n") + "\n").expect("write best_per_class_per_q.csv");
    eprintln!("wrote {}", best_path.display());

    eprintln!("[done] total runtime {:.1}s", t0.elapsed().as_secs_f64());
}
