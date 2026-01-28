//! Test how partial image frequencies converge to optimal.
//!
//! Shows the overhead at different coverage levels when transitioning
//! from buffered to streaming mode.
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example test_frequency_blending

use std::fs::{self, File};
use std::path::PathBuf;

use zenjpeg::encode::{Quality, StreamingEncoder};
use zenjpeg::huffman::optimize::FrequencyCounter;
use zenjpeg::types::Subsampling;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const QUALITY: u8 = 85;
const COVERAGE_LEVELS: &[f64] = &[0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90];

fn main() -> Result<()> {
    println!("=== Partial Frequency Convergence Test ===\n");
    println!("Shows overhead when using partial-image frequencies vs full-image optimal.\n");

    // Find test images
    let test_dir = find_test_images()?;
    let images: Vec<_> = fs::read_dir(&test_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .take(8)
        .collect();

    if images.is_empty() {
        println!("No test images found in {:?}", test_dir);
        return Ok(());
    }

    println!("Testing {} images at Q{}\n", images.len(), QUALITY);

    // Print header
    print!("{:<25}", "Image");
    for &coverage in COVERAGE_LEVELS {
        print!(" {:>6.0}%", coverage * 100.0);
    }
    println!();
    println!("{}", "-".repeat(25 + COVERAGE_LEVELS.len() * 8));

    // Track averages
    let mut coverage_totals: Vec<f64> = vec![0.0; COVERAGE_LEVELS.len()];
    let mut count = 0;

    for entry in &images {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy();
        let short_name: String = name.chars().take(24).collect();

        // Load image
        let (w, h, pixels) = load_png(&path)?;

        // Get optimal (full image) frequencies
        let optimal = get_frequencies(w, h, &pixels, h)?;

        print!("{:<25}", short_name);

        for (i, &coverage) in COVERAGE_LEVELS.iter().enumerate() {
            let rows_seen = (h as f64 * coverage) as u32;

            // Partial frequencies
            let partial = get_frequencies(w, h, &pixels, rows_seen)?;

            // Overhead if we use partial frequencies to encode full image
            let overhead = compute_overhead(&partial.ac_luma, &optimal.ac_luma);
            coverage_totals[i] += overhead;

            print!(" {:>+6.1}%", overhead);
        }
        println!();
        count += 1;
    }

    // Print averages
    println!("{}", "-".repeat(25 + COVERAGE_LEVELS.len() * 8));
    print!("{:<25}", "AVERAGE");
    for total in &coverage_totals {
        print!(" {:>+6.1}%", total / count as f64);
    }
    println!("\n");

    // Summary
    println!("=== Implications for Streaming ===\n");
    println!("When transitioning to fixed tables at X% coverage:");
    for (i, &coverage) in COVERAGE_LEVELS.iter().enumerate() {
        let avg = coverage_totals[i] / count as f64;
        let verdict = if avg < 1.0 {
            "excellent"
        } else if avg < 2.0 {
            "good"
        } else if avg < 4.0 {
            "acceptable"
        } else {
            "poor"
        };
        println!(
            "  {:>3.0}%: {:>+5.1}% overhead ({})",
            coverage * 100.0,
            avg,
            verdict
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

    // Push rows
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

/// Compute % overhead if we use `table_freq` distribution to encode `actual_freq` data.
fn compute_overhead(table_freq: &FrequencyCounter, actual_freq: &FrequencyCounter) -> f64 {
    // Generate code lengths from the table we'd use
    let table_lengths = match table_freq.generate_lengths() {
        Ok(l) => l,
        Err(_) => return f64::MAX,
    };

    // Generate optimal code lengths
    let optimal_lengths = match actual_freq.generate_lengths() {
        Ok(l) => l,
        Err(_) => return f64::MAX,
    };

    // Cost with our table
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
