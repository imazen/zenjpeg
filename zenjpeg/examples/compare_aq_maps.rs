//! Compare AQ maps between C++ and Rust jpegli.
//!
//! Usage:
//! ```bash
//! cargo run --release --example compare_aq_maps -- /tmp/cpp_aq.bin /tmp/rust_aq.bin
//! ```

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn load_aq_map(path: &Path) -> Option<(u32, u32, Vec<f32>)> {
    let mut file = File::open(path).ok()?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).ok()?;

    if data.len() < 8 {
        return None;
    }

    let w = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let h = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    let expected_size = 8 + (w as usize * h as usize * 4);
    if data.len() != expected_size {
        eprintln!(
            "Warning: file size mismatch. Expected {} bytes, got {}",
            expected_size,
            data.len()
        );
    }

    let values: Vec<f32> = data[8..]
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Some((w, h, values))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <cpp_aq.bin> <rust_aq.bin>", args[0]);
        std::process::exit(1);
    }

    let cpp_path = Path::new(&args[1]);
    let rust_path = Path::new(&args[2]);

    let (cpp_w, cpp_h, cpp_vals) = load_aq_map(cpp_path).expect("Failed to load C++ AQ map");
    let (rust_w, rust_h, rust_vals) = load_aq_map(rust_path).expect("Failed to load Rust AQ map");

    println!("=== AQ Map Comparison ===\n");
    println!(
        "C++ AQ:  {}x{} blocks ({} values)",
        cpp_w,
        cpp_h,
        cpp_vals.len()
    );
    println!(
        "Rust AQ: {}x{} blocks ({} values)",
        rust_w,
        rust_h,
        rust_vals.len()
    );

    if cpp_w != rust_w || cpp_h != rust_h {
        println!("\nERROR: Dimension mismatch!");
        return;
    }

    if cpp_vals.len() != rust_vals.len() {
        println!("\nERROR: Value count mismatch!");
        return;
    }

    // Statistics
    let mut diffs: Vec<f32> = cpp_vals
        .iter()
        .zip(rust_vals.iter())
        .map(|(c, r)| r - c)
        .collect();

    let sum_diff: f32 = diffs.iter().sum();
    let mean_diff = sum_diff / diffs.len() as f32;

    let sum_abs_diff: f32 = diffs.iter().map(|d| d.abs()).sum();
    let mean_abs_diff = sum_abs_diff / diffs.len() as f32;

    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_diff = diffs[0];
    let max_diff = diffs[diffs.len() - 1];
    let median_diff = diffs[diffs.len() / 2];

    // Relative differences
    let rel_diffs: Vec<f32> = cpp_vals
        .iter()
        .zip(rust_vals.iter())
        .filter(|(c, _)| c.abs() > 1e-6)
        .map(|(c, r)| (r - c) / c)
        .collect();

    let mean_rel_diff = if !rel_diffs.is_empty() {
        rel_diffs.iter().sum::<f32>() / rel_diffs.len() as f32
    } else {
        0.0
    };

    println!("\n=== Difference Statistics (Rust - C++) ===");
    println!("Mean difference:     {:+.6}", mean_diff);
    println!("Mean |difference|:   {:.6}", mean_abs_diff);
    println!("Median difference:   {:+.6}", median_diff);
    println!("Min difference:      {:+.6}", min_diff);
    println!("Max difference:      {:+.6}", max_diff);
    println!("Mean relative diff:  {:+.2}%", mean_rel_diff * 100.0);

    // Value range statistics
    let cpp_min = cpp_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let cpp_max = cpp_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpp_mean: f32 = cpp_vals.iter().sum::<f32>() / cpp_vals.len() as f32;

    let rust_min = rust_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let rust_max = rust_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let rust_mean: f32 = rust_vals.iter().sum::<f32>() / rust_vals.len() as f32;

    println!("\n=== Value Range Statistics ===");
    println!(
        "C++:  min={:.4}, max={:.4}, mean={:.4}",
        cpp_min, cpp_max, cpp_mean
    );
    println!(
        "Rust: min={:.4}, max={:.4}, mean={:.4}",
        rust_min, rust_max, rust_mean
    );

    // First few values comparison
    println!("\n=== First 10 Values ===");
    println!("{:>8} {:>10} {:>10} {:>10}", "Index", "C++", "Rust", "Diff");
    for i in 0..10.min(cpp_vals.len()) {
        let diff = rust_vals[i] - cpp_vals[i];
        println!(
            "{:>8} {:>10.6} {:>10.6} {:>+10.6}",
            i, cpp_vals[i], rust_vals[i], diff
        );
    }

    // Find largest differences
    let mut indexed_diffs: Vec<(usize, f32)> = cpp_vals
        .iter()
        .zip(rust_vals.iter())
        .enumerate()
        .map(|(i, (c, r))| (i, (r - c).abs()))
        .collect();
    indexed_diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n=== Largest Absolute Differences ===");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>6} {:>6}",
        "Index", "C++", "Rust", "Diff", "Row", "Col"
    );
    for (i, _diff) in indexed_diffs.iter().take(10) {
        let row = i / cpp_w as usize;
        let col = i % cpp_w as usize;
        println!(
            "{:>8} {:>10.6} {:>10.6} {:>+10.6} {:>6} {:>6}",
            i,
            cpp_vals[*i],
            rust_vals[*i],
            rust_vals[*i] - cpp_vals[*i],
            row,
            col
        );
    }

    // Histogram of differences
    println!("\n=== Difference Histogram ===");
    let mut hist = [0usize; 10];
    let bins = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
    for d in &diffs {
        let abs_d = d.abs();
        let bin = bins.iter().position(|&b| abs_d < b).unwrap_or(9);
        hist[bin] += 1;
    }
    let total = diffs.len();
    println!(
        "  <0.001:   {:>6} ({:>5.1}%)",
        hist[0],
        hist[0] as f32 / total as f32 * 100.0
    );
    println!(
        "  <0.005:   {:>6} ({:>5.1}%)",
        hist[1],
        hist[1] as f32 / total as f32 * 100.0
    );
    println!(
        "  <0.01:    {:>6} ({:>5.1}%)",
        hist[2],
        hist[2] as f32 / total as f32 * 100.0
    );
    println!(
        "  <0.02:    {:>6} ({:>5.1}%)",
        hist[3],
        hist[3] as f32 / total as f32 * 100.0
    );
    println!(
        "  <0.05:    {:>6} ({:>5.1}%)",
        hist[4],
        hist[4] as f32 / total as f32 * 100.0
    );
    println!(
        "  <0.1:     {:>6} ({:>5.1}%)",
        hist[5],
        hist[5] as f32 / total as f32 * 100.0
    );
    println!(
        "  <0.2:     {:>6} ({:>5.1}%)",
        hist[6],
        hist[6] as f32 / total as f32 * 100.0
    );
    println!(
        "  <0.5:     {:>6} ({:>5.1}%)",
        hist[7],
        hist[7] as f32 / total as f32 * 100.0
    );
    println!(
        "  <1.0:     {:>6} ({:>5.1}%)",
        hist[8],
        hist[8] as f32 / total as f32 * 100.0
    );
    println!(
        "  >=1.0:    {:>6} ({:>5.1}%)",
        hist[9],
        hist[9] as f32 / total as f32 * 100.0
    );
}
