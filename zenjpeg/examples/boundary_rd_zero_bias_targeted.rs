//! Targeted zero-bias-shrink validation sweep (Task 7 of #102 rollup).
//!
//! Answers ONE specific question:
//!
//!   Does `zero_bias_shrink < 1.0` provide consistent per-image Pareto
//!   improvement over `zero_bias_shrink = 1.0` (off) in any
//!   (content class, Q-range) cell?
//!
//! The broad 40-config sweep (`boundary_rd_zero_bias_sweep.rs`) found
//! that BD-rate math on 4-6 images per cell flattered interpolated
//! curves — doc-text/high at zb=0.5 reported BD_BBS -23% while the
//! per-image inspection showed one Gutenberg page was strictly losing
//! at every Q level. This harness addresses that by (a) larger cell
//! sample size (10 per category vs 4-6), (b) per-image Pareto-win
//! accounting instead of BD-rate curve fits, and (c) a tight grid that
//! compares only the zb knob against a fixed anchor.
//!
//! # Grid (6 configs)
//!
//! Candidate 1 (current shipped config — no zb):
//!   α=2.0, threshold=0.02, aq_shrink=0.5, zb_shrink=1.0, r=2, above=true
//! Candidates 2-5: vary only zb_shrink ∈ {0.7, 0.5, 0.3, 0.15}.
//! Candidate 6 (pure zb, no aq-shrink):
//!   α=2.0, threshold=0.02, aq_shrink=1.0, zb_shrink=0.3, r=2, above=true
//!
//! Anchor for BD-rate: `BoundaryRd::Off` (no retry).
//!
//! # Q levels: {5, 15, 30, 45, 60, 75, 85, 95} — full range.
//!
//! # Corpus: 40 images, 10 per GPT category.
//!
//! # Total: 6×8×40 = 1920 candidate encodes + 8×40 = 320 baselines = 2240.
//!
//! # Output: benchmarks/boundary_rd/zero_bias_targeted/
//!
//! - grid.csv — per (image, Q, config) raw
//! - per_image_per_cell.csv — per (image, cell, candidate) bytes/distortion
//!   deltas vs Candidate 1 (the no-zb baseline).
//! - per_cell_stats.csv — per (cell, candidate) Pareto-win-fraction
//!   and mean SSIM2 improvement.
//! - README.md is written by the caller, not this harness.
//!
//! # Decision rule
//!
//! For each candidate C ∈ {2, 3, 4, 5, 6} and each (class, Q-range) cell,
//! Pareto win (against Candidate 1) is:
//!   (bytes_C ≤ 1.02 * bytes_C1 AND ssim2_dist_C ≤ 0.98 * ssim2_dist_C1)
//! OR (bytes_C ≤ 0.98 * bytes_C1 AND ssim2_dist_C ≤ 1.02 * ssim2_dist_C1).
//!
//! Keep the knob if ANY cell × candidate pair has win_fraction ≥ 0.70
//! AND mean ssim2 distortion improvement ≥ 0.01. Otherwise drop.

use enough::Unstoppable;
use imgref::{ImgRef, ImgVec};
use rgb::RGB;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zenjpeg::encoder::{BoundaryRd, BoundaryRdConfig, ChromaSubsampling, EncoderConfig, PixelLayout};

// ------------------------- config grid -------------------------

#[derive(Debug, Clone, Copy)]
struct Knob {
    tag: &'static str,     // short label like "C1", "C2", ...
    alpha: f32,
    threshold: f32,
    aq_shrink: f32,
    zb_shrink: f32,
    retries: u8,
    above: bool,
}

impl Knob {
    fn label(&self) -> String {
        format!(
            "{}_a{:.1}_t{:.2}_aq{:.2}_zb{:.2}_r{}",
            self.tag, self.alpha, self.threshold, self.aq_shrink, self.zb_shrink, self.retries
        )
    }
}

fn build_grid() -> Vec<Knob> {
    vec![
        // C1 — current shipped (no zb)
        Knob {
            tag: "C1",
            alpha: 2.0,
            threshold: 0.02,
            aq_shrink: 0.5,
            zb_shrink: 1.0,
            retries: 2,
            above: true,
        },
        Knob {
            tag: "C2",
            alpha: 2.0,
            threshold: 0.02,
            aq_shrink: 0.5,
            zb_shrink: 0.7,
            retries: 2,
            above: true,
        },
        Knob {
            tag: "C3",
            alpha: 2.0,
            threshold: 0.02,
            aq_shrink: 0.5,
            zb_shrink: 0.5,
            retries: 2,
            above: true,
        },
        Knob {
            tag: "C4",
            alpha: 2.0,
            threshold: 0.02,
            aq_shrink: 0.5,
            zb_shrink: 0.3,
            retries: 2,
            above: true,
        },
        Knob {
            tag: "C5",
            alpha: 2.0,
            threshold: 0.02,
            aq_shrink: 0.5,
            zb_shrink: 0.15,
            retries: 2,
            above: true,
        },
        Knob {
            tag: "C6",
            alpha: 2.0,
            threshold: 0.02,
            aq_shrink: 1.0,
            zb_shrink: 0.3,
            retries: 2,
            above: true,
        },
    ]
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

fn load_manifest(manifest: &Path) -> Vec<CorpusImage> {
    let mut out = Vec::new();
    let Ok(content) = fs::read_to_string(manifest) else {
        eprintln!("[corpus] cannot read manifest {}", manifest.display());
        return out;
    };
    for (i, line) in content.lines().enumerate() {
        if i == 0 && line.starts_with("gpt_category\t") {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            continue;
        }
        let class = parts[0].to_string();
        let raw_path = parts[1];
        let path = PathBuf::from(raw_path);
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
        // Cap at 512×512 center crop — matches broad sweep conventions.
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
                .with_above(k.above),
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
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let recon_rgb: Vec<RGB<u8>> = recon
        .chunks_exact(3)
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let o_rgb: ImgRef<'_, RGB<u8>> = ImgRef::new(&orig_rgb, w, h);
    let r_rgb: ImgRef<'_, RGB<u8>> = ImgRef::new(&recon_rgb, w, h);
    let bbs = zenjpeg_bench_utils::bbs::bbs_rgb8(r_rgb, o_rgb);

    (100.0 - ssim2, bbs.total)
}

#[derive(Debug, Clone, Copy)]
struct Point {
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

// ------------------------- main -------------------------

struct Args {
    output_dir: PathBuf,
    manifest: PathBuf,
    qualities: Vec<u8>,
}

fn parse_args() -> Args {
    let mut a = Args {
        output_dir: PathBuf::from("benchmarks/boundary_rd/zero_bias_targeted"),
        manifest: PathBuf::from(
            "benchmarks/boundary_rd/zero_bias_targeted/corpus_manifest.tsv",
        ),
        qualities: vec![5, 15, 30, 45, 60, 75, 85, 95],
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
            "--qualities" => {
                i += 1;
                a.qualities = argv[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            other => eprintln!("warn: unknown arg `{}`", other),
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();
    let images = load_manifest(&args.manifest);
    if images.is_empty() {
        eprintln!("no corpus images; aborting");
        std::process::exit(1);
    }
    let configs = build_grid();
    eprintln!(
        "[plan] {} images × {} qualities × ({} configs + 1 baseline) = {} encodes",
        images.len(),
        args.qualities.len(),
        configs.len(),
        images.len() * args.qualities.len() * (configs.len() + 1),
    );

    fs::create_dir_all(&args.output_dir).expect("mkdir output");

    // Map: (image_label, config_tag or "OFF", quality) -> Point
    let mut data: BTreeMap<(String, String, u8), Point> = BTreeMap::new();
    // Image metadata keyed by label.
    let mut img_meta: BTreeMap<String, (String, usize, usize)> = BTreeMap::new();

    let t0 = Instant::now();

    // Baseline pass (OFF).
    for (idx, img) in images.iter().enumerate() {
        img_meta.insert(
            img.label.clone(),
            (img.class.clone(), img.width, img.height),
        );
        for &q in &args.qualities {
            let cfg = build_config(None, q);
            let start = Instant::now();
            let Some(jpeg) = encode_jpeg(cfg, img) else {
                eprintln!("[baseline] encode fail {} q={}", img.label, q);
                continue;
            };
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            let Some(recon) = decode_jpeg_rgb(&jpeg, img.width, img.height) else {
                eprintln!("[baseline] decode fail {} q={}", img.label, q);
                continue;
            };
            let (ssim2_d, bbs_d) = metrics(&img.rgb8, &recon, img.width, img.height);
            data.insert(
                (img.label.clone(), "OFF".into(), q),
                Point {
                    bytes: jpeg.len(),
                    ssim2_d,
                    bbs_d,
                    quality: q,
                    encode_ms,
                },
            );
        }
        if (idx + 1) % 5 == 0 {
            eprintln!(
                "[baseline] {}/{} images done, elapsed {:.1}s",
                idx + 1,
                images.len(),
                t0.elapsed().as_secs_f64(),
            );
        }
    }
    eprintln!("[baseline] done in {:.1}s", t0.elapsed().as_secs_f64());

    // Candidate pass.
    let total_cand = configs.len() * images.len() * args.qualities.len();
    let mut done = 0usize;
    let t_cand = Instant::now();
    for (ci, k) in configs.iter().enumerate() {
        let cfg_start = Instant::now();
        for img in &images {
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
                data.insert(
                    (img.label.clone(), k.tag.to_string(), q),
                    Point {
                        bytes: jpeg.len(),
                        ssim2_d,
                        bbs_d,
                        quality: q,
                        encode_ms,
                    },
                );
                done += 1;
            }
        }
        let elapsed = t_cand.elapsed().as_secs_f64();
        let rate = done as f64 / elapsed.max(1e-6);
        let eta = if rate > 0.0 {
            (total_cand - done) as f64 / rate
        } else {
            0.0
        };
        eprintln!(
            "[cfg {}/{}] {:<40} cfg_time={:.1}s done={}/{} rate={:.1}/s eta={:.0}s",
            ci + 1,
            configs.len(),
            k.label(),
            cfg_start.elapsed().as_secs_f64(),
            done,
            total_cand,
            rate,
            eta,
        );
    }
    eprintln!(
        "[candidates] done in {:.1}s (total {:.1}s)",
        t_cand.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64()
    );

    // -------- grid.csv --------
    let mut grid_rows: Vec<String> = Vec::new();
    grid_rows.push(
        "image,class,config,alpha,threshold,aq_shrink,zb_shrink,retries,\
         quality,q_range,width,height,bytes,bpp,encode_ms,ssim2_distortion,bbs_distortion"
            .into(),
    );
    // helper: lookup image width/height by label
    let describe = |tag: &str| -> (f32, f32, f32, f32, u8) {
        if tag == "OFF" {
            return (0.0, 0.0, 0.0, 0.0, 0);
        }
        for k in configs.iter() {
            if k.tag == tag {
                return (k.alpha, k.threshold, k.aq_shrink, k.zb_shrink, k.retries);
            }
        }
        (0.0, 0.0, 0.0, 0.0, 0)
    };

    for ((label, tag, q), p) in &data {
        let Some((class, w, h)) = img_meta.get(label).cloned() else {
            continue;
        };
        let pix = (w * h).max(1) as f64;
        let (alpha, threshold, aq, zb, r) = describe(tag);
        grid_rows.push(format!(
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{:.6},{:.3},{:.6},{:.6}",
            label,
            class,
            tag,
            alpha,
            threshold,
            aq,
            zb,
            r,
            q,
            q_range(*q),
            w,
            h,
            p.bytes,
            p.bytes as f64 * 8.0 / pix,
            p.encode_ms,
            p.ssim2_d,
            p.bbs_d,
        ));
    }
    let grid_path = args.output_dir.join("grid.csv");
    fs::write(&grid_path, grid_rows.join("\n") + "\n").expect("write grid.csv");
    eprintln!("wrote {}", grid_path.display());

    // -------- per_image_per_cell.csv --------
    // For each image, for each cell (class, q_range), for each candidate Cx:
    //   compare against C1 (baseline of the comparison per the question).
    //   Report bytes_ratio, ssim2_dist_ratio, bbs_ratio (mean over Q in the band).
    //   Also mark pareto_win (vs C1) using the decision rule:
    //     (bytes_C ≤ 1.02*bytes_C1 AND ssim2_C ≤ 0.98*ssim2_C1)
    //  OR (bytes_C ≤ 0.98*bytes_C1 AND ssim2_C ≤ 1.02*ssim2_C1)
    //   And report vs OFF reference too (bytes_vs_off, ssim2_vs_off, bbs_vs_off).

    let q_ranges: Vec<(&str, Vec<u8>)> = vec![
        (
            "low",
            args.qualities
                .iter()
                .copied()
                .filter(|&q| q_range(q) == "low")
                .collect(),
        ),
        (
            "mid",
            args.qualities
                .iter()
                .copied()
                .filter(|&q| q_range(q) == "mid")
                .collect(),
        ),
        (
            "high",
            args.qualities
                .iter()
                .copied()
                .filter(|&q| q_range(q) == "high")
                .collect(),
        ),
    ];

    let mut pic_rows: Vec<String> = Vec::new();
    pic_rows.push(
        "image,class,q_range,candidate,mean_bytes_c,mean_bytes_c1,bytes_ratio,\
         mean_ssim2_c,mean_ssim2_c1,ssim2_ratio,mean_bbs_c,mean_bbs_c1,bbs_ratio,\
         pareto_win_vs_c1,bytes_ratio_vs_off,ssim2_ratio_vs_off,bbs_ratio_vs_off"
            .into(),
    );

    // tuple: (class, q_range, candidate_tag) -> Vec<win_bool>, Vec<ssim2_improvement>
    type CellKey = (String, String, String);
    let mut cell_wins: BTreeMap<CellKey, Vec<bool>> = BTreeMap::new();
    let mut cell_ssim_improve: BTreeMap<CellKey, Vec<f64>> = BTreeMap::new();
    let mut cell_bytes_ratio: BTreeMap<CellKey, Vec<f64>> = BTreeMap::new();
    let mut cell_bbs_ratio: BTreeMap<CellKey, Vec<f64>> = BTreeMap::new();
    let mut cell_n: BTreeMap<CellKey, usize> = BTreeMap::new();

    for (label, meta) in &img_meta {
        let class = &meta.0;
        for (qr_name, qr_qs) in &q_ranges {
            if qr_qs.is_empty() {
                continue;
            }
            // Gather C1 reference
            let mean_of = |tag: &str, band: &[u8]| -> Option<(f64, f64, f64)> {
                let mut b = 0.0;
                let mut s = 0.0;
                let mut bbs = 0.0;
                let mut n = 0usize;
                for &q in band {
                    if let Some(p) = data.get(&(label.clone(), tag.to_string(), q)) {
                        b += p.bytes as f64;
                        s += p.ssim2_d;
                        bbs += p.bbs_d;
                        n += 1;
                    }
                }
                if n == 0 {
                    None
                } else {
                    Some((b / n as f64, s / n as f64, bbs / n as f64))
                }
            };

            let Some((b1, s1, bbs1)) = mean_of("C1", qr_qs) else {
                continue;
            };
            let Some((boff, soff, bbsoff)) = mean_of("OFF", qr_qs) else {
                continue;
            };

            for k in &configs {
                if k.tag == "C1" {
                    continue; // C1 is the reference
                }
                let Some((bc, sc, bbsc)) = mean_of(k.tag, qr_qs) else {
                    continue;
                };

                let bytes_ratio = if b1 > 0.0 { bc / b1 } else { f64::NAN };
                let ssim2_ratio = if s1 > 0.0 { sc / s1 } else { f64::NAN };
                let bbs_ratio = if bbs1 > 0.0 { bbsc / bbs1 } else { f64::NAN };

                // Pareto rule vs C1: strict improvement on one axis + not-worse on the other
                // with 2% slop for noise.
                let win1 = (bytes_ratio <= 1.02) && (ssim2_ratio <= 0.98);
                let win2 = (bytes_ratio <= 0.98) && (ssim2_ratio <= 1.02);
                let win = win1 || win2;

                let bytes_vs_off = if boff > 0.0 { bc / boff } else { f64::NAN };
                let ssim2_vs_off = if soff > 0.0 { sc / soff } else { f64::NAN };
                let bbs_vs_off = if bbsoff > 0.0 { bbsc / bbsoff } else { f64::NAN };

                pic_rows.push(format!(
                    "{},{},{},{},{:.2},{:.2},{:.4},{:.6},{:.6},{:.4},{:.6},{:.6},{:.4},{},{:.4},{:.4},{:.4}",
                    label,
                    class,
                    qr_name,
                    k.tag,
                    bc,
                    b1,
                    bytes_ratio,
                    sc,
                    s1,
                    ssim2_ratio,
                    bbsc,
                    bbs1,
                    bbs_ratio,
                    if win { 1 } else { 0 },
                    bytes_vs_off,
                    ssim2_vs_off,
                    bbs_vs_off,
                ));

                let ck: CellKey = (class.clone(), qr_name.to_string(), k.tag.to_string());
                cell_wins.entry(ck.clone()).or_default().push(win);
                cell_ssim_improve
                    .entry(ck.clone())
                    .or_default()
                    .push(s1 - sc); // positive = improvement (smaller distortion)
                cell_bytes_ratio
                    .entry(ck.clone())
                    .or_default()
                    .push(bytes_ratio);
                cell_bbs_ratio.entry(ck.clone()).or_default().push(bbs_ratio);
                *cell_n.entry(ck).or_default() += 1;
            }
        }
    }

    let pic_path = args.output_dir.join("per_image_per_cell.csv");
    fs::write(&pic_path, pic_rows.join("\n") + "\n").expect("write per_image_per_cell.csv");
    eprintln!("wrote {}", pic_path.display());

    // -------- per_cell_stats.csv --------
    let mut cell_rows: Vec<String> = Vec::new();
    cell_rows.push(
        "class,q_range,candidate,n_images,pareto_win_fraction,\
         mean_ssim2_improvement_abs,mean_bytes_ratio,mean_bbs_ratio,decision_keep"
            .into(),
    );

    // Decision: candidate passes if win_fraction ≥ 0.70 AND mean_ssim2_improvement ≥ 0.01
    let mut any_pass = false;

    for (ck, wins) in &cell_wins {
        let n = wins.len();
        let won = wins.iter().filter(|&&b| b).count();
        let wf = if n > 0 {
            won as f64 / n as f64
        } else {
            0.0
        };
        let ssim_imp = cell_ssim_improve
            .get(ck)
            .map(|v| {
                if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            })
            .unwrap_or(0.0);
        let br_mean = cell_bytes_ratio
            .get(ck)
            .map(|v| {
                if v.is_empty() {
                    f64::NAN
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            })
            .unwrap_or(f64::NAN);
        let bbs_mean = cell_bbs_ratio
            .get(ck)
            .map(|v| {
                if v.is_empty() {
                    f64::NAN
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            })
            .unwrap_or(f64::NAN);
        let pass = wf >= 0.70 && ssim_imp >= 0.01;
        if pass {
            any_pass = true;
        }
        cell_rows.push(format!(
            "{},{},{},{},{:.4},{:.6},{:.4},{:.4},{}",
            ck.0, // class
            ck.1, // q_range
            ck.2, // candidate
            n,
            wf,
            ssim_imp,
            br_mean,
            bbs_mean,
            if pass { "yes" } else { "no" },
        ));
    }

    let cell_path = args.output_dir.join("per_cell_stats.csv");
    fs::write(&cell_path, cell_rows.join("\n") + "\n").expect("write per_cell_stats.csv");
    eprintln!("wrote {}", cell_path.display());

    eprintln!(
        "[done] total {:.1}s. decision={}",
        t0.elapsed().as_secs_f64(),
        if any_pass {
            "KEEP (some cell passed)"
        } else {
            "DROP (no cell passed)"
        }
    );
}
