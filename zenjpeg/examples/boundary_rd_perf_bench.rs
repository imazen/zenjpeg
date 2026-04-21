//! Focused wall-clock perf bench for `BoundaryRd::On` vs `BoundaryRd::Off`.
//!
//! Loads a stratified 50-image sample from the 1,375-image zenjpeg tuning
//! corpus (4 GPT categories: screen_ui, screen_chart, screen_document,
//! illustration), encodes each at Q=75 under both configs, and reports
//! per-category median wall-clock overhead.
//!
//! The sample is selected deterministically from a fixed RNG seed so
//! repeated runs measure the SAME image set.
//!
//! Usage:
//!   cargo run --release -p zenjpeg --features "trellis" --example boundary_rd_perf_bench -- \
//!     [--seed N] [--sample N] [--quality Q] [--iters N] [--output-dir path] [--max-side N]
//!
//! Default: seed=42, sample=50, quality=75, iters=5, output-dir=benchmarks/boundary_rd,
//!          max-side=1024 (center-crop-and-scale to this max dimension for bench consistency).

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};
use zenjpeg::encoder::{BoundaryRd, BoundaryRdConfig};

const CORPUS_PATH: &str =
    "/home/lilith/work/coefficient/scripts/selector_corpus/lineart/zenjpeg_tuning_corpus_gpt.txt";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Category {
    ScreenUi,
    ScreenChart,
    ScreenDocument,
    Illustration,
}

impl Category {
    fn slug(&self) -> &'static str {
        match self {
            Category::ScreenUi => "screen_ui",
            Category::ScreenChart => "screen_chart",
            Category::ScreenDocument => "screen_document",
            Category::Illustration => "illustration",
        }
    }
    fn all() -> [Category; 4] {
        [
            Category::ScreenUi,
            Category::ScreenChart,
            Category::ScreenDocument,
            Category::Illustration,
        ]
    }
}

#[derive(Debug, Clone)]
struct CorpusEntry {
    path: PathBuf,
    category: Category,
}

fn parse_corpus_list() -> Vec<CorpusEntry> {
    let text = fs::read_to_string(CORPUS_PATH)
        .unwrap_or_else(|e| panic!("failed to read {CORPUS_PATH}: {e}"));
    let mut out = Vec::new();
    let mut current: Option<Category> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("# === ") {
            let name = rest.split_whitespace().next().unwrap_or("");
            current = match name {
                "screen_ui" => Some(Category::ScreenUi),
                "screen_chart" => Some(Category::ScreenChart),
                "screen_document" => Some(Category::ScreenDocument),
                "illustration" => Some(Category::Illustration),
                _ => None,
            };
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(cat) = current {
            out.push(CorpusEntry {
                path: PathBuf::from(trimmed),
                category: cat,
            });
        }
    }
    out
}

/// Small, deterministic PRNG so the sample is fully reproducible across runs
/// (no `rand` dependency pull-in on the encode crate).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn usize_bound(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.usize_bound(i + 1);
            v.swap(i, j);
        }
    }
}

fn stratified_sample(entries: &[CorpusEntry], total: usize, seed: u64) -> Vec<CorpusEntry> {
    let mut per_cat: std::collections::HashMap<Category, Vec<usize>> = Default::default();
    for (i, e) in entries.iter().enumerate() {
        per_cat.entry(e.category).or_default().push(i);
    }

    let cats = Category::all();
    let per = (total + cats.len() - 1) / cats.len();

    let mut rng = SplitMix64::new(seed);
    let mut picked = Vec::new();

    for cat in cats.iter() {
        let mut indices = per_cat.get(cat).cloned().unwrap_or_default();
        rng.shuffle(&mut indices);
        for &idx in indices.iter().take(per) {
            picked.push(entries[idx].clone());
        }
    }

    picked.truncate(total);
    picked
}

fn load_and_downscale(path: &Path, max_side: u32) -> Option<(Vec<u8>, usize, usize)> {
    let img = match image::open(path) {
        Ok(i) => i,
        Err(_) => return None,
    };
    let (w, h) = (img.width(), img.height());
    let scaled = if w.max(h) > max_side {
        let (tw, th) = if w >= h {
            (
                max_side,
                (h as u64 * max_side as u64 / w as u64).max(1) as u32,
            )
        } else {
            (
                (w as u64 * max_side as u64 / h as u64).max(1) as u32,
                max_side,
            )
        };
        img.resize_exact(tw, th, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    // Round dims down to MCU-aligned when possible for fair measurements.
    let w = (scaled.width() as usize) & !7;
    let h = (scaled.height() as usize) & !7;
    if w < 64 || h < 64 {
        return None;
    }
    let rgb = scaled.to_rgb8();
    // If downscale caused non-MCU dims, crop to top-left w×h.
    let orig_w = rgb.width() as usize;
    let mut buf = Vec::with_capacity(w * h * 3);
    let raw = rgb.as_raw();
    for y in 0..h {
        let row_start = y * orig_w * 3;
        buf.extend_from_slice(&raw[row_start..row_start + w * 3]);
    }
    Some((buf, w, h))
}

fn encode_once(
    config: EncoderConfig,
    rgb: &[u8],
    w: usize,
    h: usize,
) -> Option<std::time::Duration> {
    let t0 = Instant::now();
    let result = config.encode_bytes(rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb);
    let elapsed = t0.elapsed();
    if result.is_ok() { Some(elapsed) } else { None }
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    xs[idx.min(n - 1)]
}

fn main() {
    let mut seed: u64 = 42;
    let mut sample: usize = 50;
    let mut quality: f32 = 75.0;
    let mut iters: usize = 5;
    let mut output_dir = PathBuf::from("benchmarks/boundary_rd");
    let mut max_side: u32 = 1024;
    let mut tag: String = String::from("default");

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--seed" => {
                i += 1;
                seed = argv[i].parse().unwrap();
            }
            "--sample" => {
                i += 1;
                sample = argv[i].parse().unwrap();
            }
            "--quality" => {
                i += 1;
                quality = argv[i].parse().unwrap();
            }
            "--iters" => {
                i += 1;
                iters = argv[i].parse().unwrap();
            }
            "--output-dir" => {
                i += 1;
                output_dir = PathBuf::from(&argv[i]);
            }
            "--max-side" => {
                i += 1;
                max_side = argv[i].parse().unwrap();
            }
            "--tag" => {
                i += 1;
                tag = argv[i].clone();
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let entries = parse_corpus_list();
    eprintln!("corpus total: {} entries", entries.len());

    let picked = stratified_sample(&entries, sample, seed);
    eprintln!("stratified sample: {} images", picked.len());

    fs::create_dir_all(&output_dir).unwrap();
    let csv_path = output_dir.join(format!("perf_{tag}_seed{seed}_q{}.csv", quality as u32));
    let mut csv = fs::File::create(&csv_path).unwrap();
    writeln!(
        csv,
        "label,category,width,height,off_median_ms,on_median_ms,overhead_pct"
    )
    .unwrap();

    let mut per_cat: std::collections::HashMap<Category, Vec<f64>> = Default::default();
    let mut all_overhead: Vec<f64> = Vec::new();
    let mut loaded_count = 0usize;
    let mut skipped = 0usize;

    for (idx, entry) in picked.iter().enumerate() {
        let Some((rgb, w, h)) = load_and_downscale(&entry.path, max_side) else {
            skipped += 1;
            continue;
        };
        loaded_count += 1;

        // Warmup
        let cfg_off =
            EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::Off);
        let _ = encode_once(cfg_off, &rgb, w, h);
        let cfg_on = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
            .boundary_rd(BoundaryRd::On(BoundaryRdConfig::default()));
        let _ = encode_once(cfg_on, &rgb, w, h);

        // Measure, interleaved
        let mut off_times = Vec::new();
        let mut on_times = Vec::new();
        for _ in 0..iters {
            let cfg_off = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                .boundary_rd(BoundaryRd::Off);
            if let Some(d) = encode_once(cfg_off, &rgb, w, h) {
                off_times.push(d.as_secs_f64() * 1000.0);
            }
            let cfg_on = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                .boundary_rd(BoundaryRd::On(BoundaryRdConfig::default()));
            if let Some(d) = encode_once(cfg_on, &rgb, w, h) {
                on_times.push(d.as_secs_f64() * 1000.0);
            }
        }
        if off_times.is_empty() || on_times.is_empty() {
            skipped += 1;
            continue;
        }
        let off_med = median(off_times);
        let on_med = median(on_times);
        let overhead = (on_med - off_med) / off_med * 100.0;

        let label: String = entry
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .chars()
            .take(40)
            .collect();
        writeln!(
            csv,
            "{label},{cat},{w},{h},{off_med:.3},{on_med:.3},{overhead:.2}",
            cat = entry.category.slug(),
        )
        .unwrap();

        per_cat.entry(entry.category).or_default().push(overhead);
        all_overhead.push(overhead);

        if idx % 5 == 0 {
            eprintln!(
                "[{}/{}] {} cat={} w={w} h={h} off={off_med:.2}ms on={on_med:.2}ms overhead={overhead:+.1}%",
                idx + 1,
                picked.len(),
                label,
                entry.category.slug(),
            );
        }
    }

    eprintln!(
        "\nloaded {loaded_count} of {} ({} skipped)",
        picked.len(),
        skipped
    );

    // Aggregate
    let summary_path = output_dir.join(format!("perf_{tag}_seed{seed}_q{}.md", quality as u32));
    let mut out = fs::File::create(&summary_path).unwrap();
    writeln!(
        out,
        "# boundary_rd perf bench — {} — Q={}",
        tag, quality as u32
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Sample seed: {seed}. Iterations per config per image: {iters}.\n"
    )
    .unwrap();
    writeln!(out, "| Category | N | Median overhead | p95 overhead |").unwrap();
    writeln!(out, "|---|---:|---:|---:|").unwrap();
    for cat in Category::all() {
        let xs = per_cat.get(&cat).cloned().unwrap_or_default();
        let n = xs.len();
        let med = median(xs.clone());
        let p95 = percentile(xs, 95.0);
        writeln!(out, "| {} | {n} | {med:+.2}% | {p95:+.2}% |", cat.slug()).unwrap();
    }
    let overall_med = median(all_overhead.clone());
    let overall_p95 = percentile(all_overhead, 95.0);
    writeln!(out).unwrap();
    writeln!(
        out,
        "**Overall median: {overall_med:+.2}% | p95: {overall_p95:+.2}%**"
    )
    .unwrap();
    writeln!(out, "\nCSV: `{}`", csv_path.display()).unwrap();

    println!("\n==============================================");
    println!("overall median overhead: {overall_med:+.2}%");
    println!("overall p95 overhead: {overall_p95:+.2}%");
    for cat in Category::all() {
        if let Some(xs) = per_cat.get(&cat) {
            let m = median(xs.clone());
            println!("  {}: median={:+.2}% (N={})", cat.slug(), m, xs.len());
        }
    }
    println!("summary -> {}", summary_path.display());
    println!("csv -> {}", csv_path.display());
}
