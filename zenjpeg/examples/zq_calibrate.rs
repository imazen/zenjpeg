//! Calibrate per-bucket starting-q values for `Quality::Zq` (#113 PR-D).
//!
//! For each (content_bucket, target_zq) combination, finds the smallest
//! jpegli quality that lands at-or-above target zq when encoded with
//! default streaming AQ (no controller). Prints the values in the format
//! that [`zenjpeg::encode::zq::zq_to_starting_jpegli_q_for_bucket`] expects.
//!
//! The output of this binary is meant to be hand-pasted into the anchor
//! tables in `src/encode/zq.rs`. Re-run when:
//!  - The streaming AQ pipeline changes (e.g. new SIMD path, new perceptual
//!    weighting).
//!  - The zensim profile is replaced (calibration is profile-relative).
//!  - The corpus shifts substantially (new content types).
//!
//! Usage:
//!   cargo run --release -p zenjpeg --features target-zq \
//!     --example zq_calibrate -- --corpus /path/to/corpus
//!
//! Default corpus is CID22 validation (40 photo images). For a richer fit
//! pass `--corpus /path/to/mixed-content/dir` covering screen content +
//! illustration too.

#![cfg(feature = "target-zq")]

use enough::Unstoppable;
use std::path::PathBuf;
use zenjpeg::analyze::analyze_rgb8;
use zenjpeg::decode::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

const ZQ_TARGETS: &[f32] = &[40.0, 60.0, 75.0, 80.0, 85.0, 90.0, 95.0];
const Q_GRID: &[u8] = &[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
];

struct Args {
    corpus: PathBuf,
    max_images: usize,
}

fn parse_args() -> Args {
    let mut corpus =
        PathBuf::from("/home/lilith/work/codec-eval/codec-corpus/CID22/CID22-512/validation");
    let mut max_images = 64;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--corpus" => corpus = PathBuf::from(it.next().unwrap()),
            "--max-images" => max_images = it.next().unwrap().parse().expect("max-images uint"),
            other => panic!("unknown arg: {other}"),
        }
    }
    Args { corpus, max_images }
}

fn load_png(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img =
        image::open(path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
}

fn encode_and_score(
    z: &Zensim,
    pre: &zensim::PrecomputedReference,
    rgb: &[u8],
    w: u32,
    h: u32,
    q: u8,
) -> f32 {
    let cfg = EncoderConfig::ycbcr(Quality::ApproxJpegli(q as f32), ChromaSubsampling::Quarter);
    let jpeg = cfg
        .encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode");
    let dec = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .expect("u8 pixels");
    let chunks: &[[u8; 3]] = dec.as_chunks::<3>().0;
    let dec_slice = RgbSlice::new(chunks, w as usize, h as usize);
    z.compute_with_ref_and_diffmap(pre, &dec_slice, DiffmapWeighting::Trained)
        .expect("zensim")
        .score() as f32
}

/// For one image, find smallest q with score >= target_zq for each
/// target. Returns Vec of (target_zq, smallest_q) tuples.
fn fit_image(
    z: &Zensim,
    pre: &zensim::PrecomputedReference,
    rgb: &[u8],
    w: u32,
    h: u32,
) -> Vec<(f32, u8)> {
    // Encode at every q in the grid once; record score → use these to
    // pick smallest q meeting each target.
    let mut q_score: Vec<(u8, f32)> = Q_GRID
        .iter()
        .map(|&q| (q, encode_and_score(z, pre, rgb, w, h, q)))
        .collect();
    q_score.sort_by_key(|&(q, _)| q);

    ZQ_TARGETS
        .iter()
        .map(|&zq| {
            let q = q_score
                .iter()
                .find(|&&(_, score)| score >= zq)
                .map(|&(q, _)| q)
                .unwrap_or(100);
            (zq, q)
        })
        .collect()
}

fn main() {
    let args = parse_args();
    let z = Zensim::new(ZensimProfile::latest());

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.corpus)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", args.corpus.display()))
        .filter_map(|r| r.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    paths.sort();
    paths.truncate(args.max_images);
    eprintln!(
        "[zq_calibrate] {} image(s), {} q values × {} targets",
        paths.len(),
        Q_GRID.len(),
        ZQ_TARGETS.len()
    );

    // Per-bucket tabulation. For each bucket, we'll accumulate the
    // smallest q per target across all images in that bucket, then take
    // the median (a single typical anchor that hits target on most
    // bucket-members at-or-above target).
    use std::collections::HashMap;
    let mut by_bucket: HashMap<&'static str, Vec<Vec<(f32, u8)>>> = HashMap::new();

    for (i, path) in paths.iter().enumerate() {
        let (rgb, w, h) = load_png(path);
        let features = analyze_rgb8(&rgb, w, h);
        let bucket_label: &'static str = match (
            features.text_likelihood,
            features.screen_content_likelihood,
            features.uniformity,
            features.flat_color_block_ratio,
            features.high_freq_energy_ratio,
            features.edge_density,
            features.cb_peak_sharpness,
            features.cr_peak_sharpness,
            features.chroma_complexity,
        ) {
            (txt, _, _, _, _, _, cb_p, _, c) if txt > 0.55 && (c > 0.04 || cb_p > 5.0) => {
                "Illustration"
            }
            (txt, _, _, _, _, _, _, _, _) if txt > 0.55 => "ScreenContent",
            (_, scr, _, _, _, _, _, _, _) if scr > 0.5 => "ScreenContent",
            (_, _, u, f, _, _, _, _, _) if u > 0.55 || f > 0.25 => "PhotoFlat",
            (_, _, _, _, hf, ed, cb_p, cr_p, _)
                if hf > 0.30 || ed > 0.18 || cb_p > 8.0 || cr_p > 8.0 =>
            {
                "PhotoDetailed"
            }
            _ => "PhotoNatural",
        };

        let src_chunks: &[[u8; 3]] = rgb.as_chunks::<3>().0;
        let src_slice = RgbSlice::new(src_chunks, w as usize, h as usize);
        let pre = z.precompute_reference(&src_slice).expect("precompute");
        let fits = fit_image(&z, &pre, &rgb, w, h);

        by_bucket.entry(bucket_label).or_default().push(fits);

        if (i + 1) % 5 == 0 {
            eprintln!("  progress: {}/{}", i + 1, paths.len());
        }
    }

    // Median per (bucket, target) → final anchor.
    println!("\n=== Calibration anchors per bucket ===\n");
    let bucket_order = [
        "PhotoNatural",
        "PhotoDetailed",
        "PhotoFlat",
        "Illustration",
        "ScreenContent",
    ];
    for &bucket in &bucket_order {
        match by_bucket.get(bucket) {
            None => {
                println!("// {}: NO IMAGES IN CORPUS", bucket);
                continue;
            }
            Some(rows) => {
                let n = rows.len();
                println!("// {} (n={})", bucket, n);
                println!("const {}: &[(f32, f32)] = &[", bucket.to_uppercase());
                for (i, &target) in ZQ_TARGETS.iter().enumerate() {
                    let mut qs: Vec<u8> = rows.iter().map(|r| r[i].1).collect();
                    qs.sort();
                    let median = qs[n / 2];
                    println!(
                        "    ({:.1}, {:.1}),  // n={n}, p25={} p50={median} p75={}",
                        target,
                        median as f32,
                        qs[n / 4],
                        qs[(3 * n) / 4],
                    );
                    let _ = target;
                }
                println!("];\n");
            }
        }
    }
}
