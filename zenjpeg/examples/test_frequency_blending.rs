//! Test frequency blending for streaming JPEG transition.
//!
//! Compares different strategies when transitioning from buffered to streaming:
//! - Partial: Use only frequencies from rows seen so far
//! - Trained: Use pre-trained corpus frequencies (ignore partial data)
//! - Blended: Combine partial + trained prior for rare symbols (threshold-based)
//! - Proportional: Add small fraction of prior to all symbols
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example test_frequency_blending

use std::fs::{self, File};
use std::path::PathBuf;

use zenjpeg::encode::{Quality, StreamingEncoder};
use zenjpeg::huffman::optimize::FrequencyCounter;
use zenjpeg::types::Subsampling;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const QUALITY: u8 = 85;
const COVERAGE_LEVELS: &[f64] = &[0.25, 0.40, 0.50, 0.60, 0.75];
const MIN_SAMPLES: i64 = 50;
const PRIOR_WEIGHTS: &[f64] = &[0.001, 0.005, 0.01, 0.02, 0.05];

fn main() -> Result<()> {
    println!("=== Frequency Blending Test ===\n");

    // Load trained frequencies
    let trained = load_trained_frequencies(QUALITY)?;
    println!(
        "Loaded trained frequencies for Q{}: {:.0}M AC luma symbols\n",
        QUALITY,
        trained.ac_luma.total() as f64 / 1_000_000.0
    );

    // Find test images
    let test_dir = find_test_images()?;
    let images: Vec<_> = fs::read_dir(&test_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .take(10)
        .collect();

    if images.is_empty() {
        println!("No test images found in {:?}", test_dir);
        return Ok(());
    }

    println!("Testing {} images at Q{}\n", images.len(), QUALITY);

    // ============================================================
    // Part 1: Compare Partial vs Trained vs Threshold-Blended
    // ============================================================
    println!("=== Part 1: Threshold-Based Blending (min_samples={}) ===\n", MIN_SAMPLES);

    let mut partial_totals = vec![0.0f64; COVERAGE_LEVELS.len()];
    let mut trained_totals = vec![0.0f64; COVERAGE_LEVELS.len()];
    let mut blended_totals = vec![0.0f64; COVERAGE_LEVELS.len()];
    let mut count = 0;

    for entry in &images {
        let path = entry.path();
        let (w, h, pixels) = load_png(&path)?;
        let optimal = get_frequencies(w, h, &pixels, h)?;

        for (i, &coverage) in COVERAGE_LEVELS.iter().enumerate() {
            let rows_seen = (h as f64 * coverage) as u32;
            let partial = get_frequencies(w, h, &pixels, rows_seen)?;

            let partial_overhead = compute_overhead(&partial.ac_luma, &optimal.ac_luma);
            let trained_overhead = compute_overhead(&trained.ac_luma, &optimal.ac_luma);
            let blended = partial.ac_luma.blend_with_prior(&trained.ac_luma, MIN_SAMPLES);
            let blended_overhead = compute_overhead(&blended, &optimal.ac_luma);

            partial_totals[i] += partial_overhead;
            trained_totals[i] += trained_overhead;
            blended_totals[i] += blended_overhead;
        }
        count += 1;
    }

    println!(
        "{:>6}   {:>10} {:>10} {:>10}   {:>8}",
        "Cover", "Partial", "Trained", "Blended", "Best"
    );
    println!("{}", "-".repeat(55));

    for (i, &coverage) in COVERAGE_LEVELS.iter().enumerate() {
        let partial_avg = partial_totals[i] / count as f64;
        let trained_avg = trained_totals[i] / count as f64;
        let blended_avg = blended_totals[i] / count as f64;

        let best = if partial_avg <= trained_avg && partial_avg <= blended_avg {
            "Partial"
        } else if trained_avg <= blended_avg {
            "Trained"
        } else {
            "Blended"
        };

        println!(
            "{:>5.0}%   {:>+9.1}% {:>+9.1}% {:>+9.1}%   {:>8}",
            coverage * 100.0, partial_avg, trained_avg, blended_avg, best
        );
    }

    // ============================================================
    // Part 2: Proportional Prior Blending
    // ============================================================
    println!("\n=== Part 2: Proportional Prior Blending (at 25% coverage) ===\n");
    println!("Prior weight = fraction of observed total added from trained prior\n");

    let mut prop_totals: Vec<f64> = vec![0.0; PRIOR_WEIGHTS.len()];

    for entry in &images {
        let path = entry.path();
        let (w, h, pixels) = load_png(&path)?;
        let optimal = get_frequencies(w, h, &pixels, h)?;

        let rows_seen = (h as f64 * 0.25) as u32; // 25% coverage
        let partial = get_frequencies(w, h, &pixels, rows_seen)?;

        for (i, &weight) in PRIOR_WEIGHTS.iter().enumerate() {
            let blended = partial.ac_luma.add_prior_proportional(&trained.ac_luma, weight);
            let overhead = compute_overhead(&blended, &optimal.ac_luma);
            prop_totals[i] += overhead;
        }
    }

    let partial_25 = partial_totals[0] / count as f64;
    println!("Partial only (baseline): {:+.2}%\n", partial_25);

    println!("{:>12}   {:>10}   {:>10}", "Weight", "Overhead", "vs Partial");
    println!("{}", "-".repeat(40));

    for (i, &weight) in PRIOR_WEIGHTS.iter().enumerate() {
        let avg = prop_totals[i] / count as f64;
        let diff = avg - partial_25;
        println!(
            "{:>11.1}%   {:>+9.2}%   {:>+9.2}%",
            weight * 100.0, avg, diff
        );
    }

    // ============================================================
    // Conclusions
    // ============================================================
    println!("\n=== Conclusions ===\n");

    let partial_25 = partial_totals[0] / count as f64;
    let best_prop_idx = prop_totals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_prop = prop_totals[best_prop_idx] / count as f64;
    let best_weight = PRIOR_WEIGHTS[best_prop_idx];

    if best_prop < partial_25 - 0.05 {
        println!(
            "Proportional blending at {:.1}% weight improves over partial: {:+.2}% vs {:+.2}%",
            best_weight * 100.0, best_prop, partial_25
        );
    } else if (best_prop - partial_25).abs() < 0.1 {
        println!("Proportional blending shows no significant improvement over partial");
        println!("Recommendation: Just use partial frequencies (simpler)");
    } else {
        println!("Partial frequencies win - no blending needed");
    }

    let good_coverage_idx = COVERAGE_LEVELS
        .iter()
        .enumerate()
        .find(|(i, _)| partial_totals[*i] / (count as f64) < 1.0)
        .map(|(i, _)| i);

    if let Some(idx) = good_coverage_idx {
        println!(
            "\nPartial reaches <1% overhead at {}% coverage",
            (COVERAGE_LEVELS[idx] * 100.0) as u32
        );
    }

    Ok(())
}

struct Frequencies {
    #[allow(dead_code)]
    dc_luma: FrequencyCounter,
    ac_luma: FrequencyCounter,
    #[allow(dead_code)]
    dc_chroma: FrequencyCounter,
    #[allow(dead_code)]
    ac_chroma: FrequencyCounter,
}

fn get_frequencies(w: u32, h: u32, pixels: &[u8], rows_to_encode: u32) -> Result<Frequencies> {
    let mut encoder = StreamingEncoder::new(w, h)
        .quality(Quality::ApproxJpegli(QUALITY as f32))
        .subsampling(Subsampling::S420)
        .start()?;

    let stride = w as usize * 3;
    for y in 0..rows_to_encode.min(h) {
        let row_start = y as usize * stride;
        let row_end = row_start + stride;
        encoder.push_rows(&pixels[row_start..row_end], 1)?;
    }

    let counters = encoder.frequency_counters();
    Ok(Frequencies {
        dc_luma: counters.0.clone(),
        ac_luma: counters.1.clone(),
        dc_chroma: counters.2.clone(),
        ac_chroma: counters.3.clone(),
    })
}

fn compute_overhead(table_freq: &FrequencyCounter, actual_freq: &FrequencyCounter) -> f64 {
    let table_lengths = match table_freq.generate_lengths() {
        Ok(l) => l,
        Err(_) => return f64::MAX,
    };

    let optimal_lengths = match actual_freq.generate_lengths() {
        Ok(l) => l,
        Err(_) => return f64::MAX,
    };

    let mut table_cost: u64 = 0;
    let mut optimal_cost: u64 = 0;
    for i in 0..256 {
        let count = actual_freq.get_count(i as u8) as u64;
        table_cost += count * table_lengths[i] as u64;
        optimal_cost += count * optimal_lengths[i] as u64;
    }

    if optimal_cost == 0 {
        return 0.0;
    }

    (table_cost as f64 / optimal_cost as f64 - 1.0) * 100.0
}

fn load_trained_frequencies(quality: u8) -> Result<Frequencies> {
    let json_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/trained_tables/raw_frequencies.json");
    let json_str = fs::read_to_string(&json_path)?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;

    let quality_key = format!("q{}", quality);
    let tier = data
        .get(&quality_key)
        .ok_or_else(|| format!("No data for {}", quality_key))?;

    fn parse_counter(arr: &serde_json::Value) -> FrequencyCounter {
        if let Some(arr) = arr.as_array() {
            let counts: Vec<i64> = arr.iter().map(|v| v.as_i64().unwrap_or(0)).collect();
            FrequencyCounter::from_counts(&counts)
        } else {
            FrequencyCounter::new()
        }
    }

    Ok(Frequencies {
        dc_luma: parse_counter(&tier["dc_luma"]),
        ac_luma: parse_counter(&tier["ac_luma"]),
        dc_chroma: parse_counter(&tier["dc_chroma"]),
        ac_chroma: parse_counter(&tier["ac_chroma"]),
    })
}

fn load_png(path: &PathBuf) -> Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    let (w, h) = (info.width, info.height);
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return Err(format!("Unsupported: {:?}", info.color_type).into()),
    };

    Ok((w, h, pixels))
}

fn find_test_images() -> Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".to_string());
    let candidates = [
        PathBuf::from(&home).join("work/codec-corpus/clic2025/final-test"),
        PathBuf::from(&home).join("work/codec-eval/codec-corpus/clic2025/final-test"),
        PathBuf::from(&home).join("work/codec-corpus/CID22/CID22-512/validation"),
        PathBuf::from("internal/jpegli-cpp/testdata"),
    ];

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    Err("No test image directory found".into())
}
