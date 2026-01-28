//! Test frequency blending for streaming JPEG transition.
//!
//! Simulates the scenario where we've encoded part of an image and need to
//! transition to fixed Huffman tables. Compares:
//! - Optimal: Full-image frequencies (theoretical best)
//! - Partial: Only frequencies from seen rows (can be poor for rare symbols)
//! - Corpus: Pre-trained corpus tables (good general case)
//! - Blended: Partial + corpus prior (hopefully best of both)
//!
//! Run with: cargo run --release --example test_frequency_blending

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use zenjpeg::encode::streaming::StreamingEncoder;
use zenjpeg::huffman::optimize::FrequencyCounter;
use zenjpeg::types::Subsampling;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const QUALITY: u8 = 85;
const COVERAGE_LEVELS: &[f64] = &[0.25, 0.50, 0.75];
const MIN_SAMPLES: i64 = 50;

fn main() -> Result<()> {
    println!("=== Frequency Blending Test ===\n");

    // Load corpus frequencies for Q85
    let corpus = load_corpus_frequencies(QUALITY)?;
    println!(
        "Loaded corpus frequencies for Q{}: {:.0} total AC luma symbols\n",
        QUALITY,
        corpus.ac_luma.total() as f64
    );

    // Find test images
    let test_dir = find_test_images()?;
    let images: Vec<_> = fs::read_dir(&test_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "png" || ext == "jpg")
        })
        .take(5) // Just test a few
        .collect();

    if images.is_empty() {
        println!("No test images found in {:?}", test_dir);
        return Ok(());
    }

    println!("Testing {} images at Q{}\n", images.len(), QUALITY);
    println!(
        "{:<30} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Image", "Coverage", "Optimal", "Partial", "Corpus", "Blended"
    );
    println!("{}", "-".repeat(88));

    let mut totals: HashMap<String, (f64, usize)> = HashMap::new();

    for entry in &images {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy();

        // Load image
        let img = image::open(&path)?.to_rgb8();
        let (width, height) = img.dimensions();

        // Get optimal (full image) frequencies
        let optimal = get_image_frequencies(&img, width, height, 0, height)?;
        let optimal_cost = optimal.ac_luma.estimate_encoding_cost();

        for &coverage in COVERAGE_LEVELS {
            let rows_seen = (height as f64 * coverage) as u32;

            // Partial frequencies (simulated transition point)
            let partial = get_image_frequencies(&img, width, height, 0, rows_seen)?;
            let partial_cost = partial.ac_luma.estimate_encoding_cost();

            // Corpus-only cost
            let corpus_for_image = scale_corpus_to_image(&corpus, &optimal);
            let corpus_cost = corpus_for_image.ac_luma.estimate_encoding_cost();

            // Blended: partial + corpus prior
            let blended_ac = partial.ac_luma.blend_with_prior(&corpus.ac_luma, MIN_SAMPLES);
            // Scale blended to match optimal total for fair comparison
            let blended_scaled = scale_counter_to_total(&blended_ac, optimal.ac_luma.total());
            let blended_cost = blended_scaled.estimate_encoding_cost();

            // Compute overhead percentages vs optimal
            let partial_overhead = (partial_cost / optimal_cost - 1.0) * 100.0;
            let corpus_overhead = (corpus_cost / optimal_cost - 1.0) * 100.0;
            let blended_overhead = (blended_cost / optimal_cost - 1.0) * 100.0;

            println!(
                "{:<30} {:>7.0}% {:>9.0} {:>+9.1}% {:>+9.1}% {:>+9.1}%",
                if coverage == COVERAGE_LEVELS[0] {
                    name.to_string()
                } else {
                    String::new()
                },
                coverage * 100.0,
                optimal_cost,
                partial_overhead,
                corpus_overhead,
                blended_overhead
            );

            // Accumulate totals
            let key = format!("{:.0}%", coverage * 100.0);
            let entry = totals.entry(key).or_insert((0.0, 0));
            entry.0 += blended_overhead - partial_overhead; // improvement from blending
            entry.1 += 1;
        }
        println!();
    }

    // Summary
    println!("\n=== Summary ===\n");
    println!("Blending improvement over partial-only:");
    for coverage in COVERAGE_LEVELS {
        let key = format!("{:.0}%", coverage * 100.0);
        if let Some((total_improvement, count)) = totals.get(&key) {
            let avg = total_improvement / *count as f64;
            println!("  {:>3}% coverage: {:+.2}% average", key, avg);
        }
    }

    Ok(())
}

struct Frequencies {
    dc_luma: FrequencyCounter,
    ac_luma: FrequencyCounter,
    dc_chroma: FrequencyCounter,
    ac_chroma: FrequencyCounter,
}

fn get_image_frequencies(
    img: &image::RgbImage,
    width: u32,
    height: u32,
    start_row: u32,
    end_row: u32,
) -> Result<Frequencies> {
    // Create a streaming encoder to collect frequencies
    let mut encoder = StreamingEncoder::new(width, height)
        .quality(QUALITY)
        .subsampling(Subsampling::S420)
        .start()?;

    // Push rows up to end_row
    let pixels: Vec<_> = img.pixels().cloned().collect();
    let stride = width as usize;

    for y in start_row..end_row.min(height) {
        let row_start = y as usize * stride;
        let row_end = row_start + stride;
        let row_data: Vec<[u8; 3]> = pixels[row_start..row_end]
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();

        encoder.push_rows(&row_data, 1)?;
    }

    // Extract frequency counters
    let counters = encoder.frequency_counters();
    Ok(Frequencies {
        dc_luma: counters.0.clone(),
        ac_luma: counters.1.clone(),
        dc_chroma: counters.2.clone(),
        ac_chroma: counters.3.clone(),
    })
}

fn load_corpus_frequencies(quality: u8) -> Result<Frequencies> {
    let json_path = PathBuf::from("/mnt/v/output/zenjpeg/corpus_tables/frequency_counts.json");
    let json_str = fs::read_to_string(&json_path)?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;

    let quality_key = format!("Q{}", quality);
    let tier = data
        .get(&quality_key)
        .ok_or_else(|| format!("No data for {}", quality_key))?;

    fn parse_counter(arr: &serde_json::Value) -> FrequencyCounter {
        let mut counter = FrequencyCounter::new();
        if let Some(arr) = arr.as_array() {
            for (i, v) in arr.iter().enumerate() {
                if i < 256 {
                    let count = v.as_i64().unwrap_or(0);
                    for _ in 0..count {
                        counter.count(i as u8);
                    }
                }
            }
        }
        counter
    }

    Ok(Frequencies {
        dc_luma: parse_counter(&tier["dc_luma"]),
        ac_luma: parse_counter(&tier["ac_luma"]),
        dc_chroma: parse_counter(&tier["dc_chroma"]),
        ac_chroma: parse_counter(&tier["ac_chroma"]),
    })
}

fn scale_corpus_to_image(corpus: &Frequencies, image: &Frequencies) -> Frequencies {
    Frequencies {
        dc_luma: scale_counter_to_total(&corpus.dc_luma, image.dc_luma.total()),
        ac_luma: scale_counter_to_total(&corpus.ac_luma, image.ac_luma.total()),
        dc_chroma: scale_counter_to_total(&corpus.dc_chroma, image.dc_chroma.total()),
        ac_chroma: scale_counter_to_total(&corpus.ac_chroma, image.ac_chroma.total()),
    }
}

fn scale_counter_to_total(counter: &FrequencyCounter, target_total: i64) -> FrequencyCounter {
    let current_total = counter.total();
    if current_total == 0 || target_total == 0 {
        return counter.clone();
    }

    let scale = target_total as f64 / current_total as f64;
    let mut result = FrequencyCounter::new();

    for i in 0..256 {
        let count = counter.get_count(i as u8);
        let scaled = (count as f64 * scale).round() as i64;
        for _ in 0..scaled {
            result.count(i as u8);
        }
    }

    result
}

fn find_test_images() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from(env!("HOME")).join("work/codec-eval/codec-corpus/clic2025/final-test"),
        PathBuf::from(env!("HOME")).join("work/codec-eval/codec-corpus/CID22/CID22-512/validation"),
        PathBuf::from("internal/jpegli-cpp/testdata"),
    ];

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    Err("No test image directory found".into())
}
