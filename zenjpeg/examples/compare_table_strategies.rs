//! Compare overhead: optimized tables vs standard tables at different thresholds.
//!
//! Tests the "fallback to standard tables" strategy for pathological images.
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example compare_table_strategies

use zenjpeg::encode::encoder_types::Quality;
use zenjpeg::encode::streaming::StreamingEncoder;
use zenjpeg::types::Subsampling;

fn main() {
    let clic_dir = "/home/lilith/work/codec-corpus/clic2025/validation";

    let test_images: Vec<_> = std::fs::read_dir(clic_dir)
        .expect("Failed to read CLIC validation directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "png") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    eprintln!("Testing {} images\n", test_images.len());

    // Test thresholds
    let thresholds = [15, 20, 25];
    let memory_limit = 1024 * 1024;

    // Collect results for each strategy at each threshold
    // Strategy 1: Always use optimized tables from partial data
    // Strategy 2: Use standard tables (no optimization)
    // Strategy 3: Hybrid - optimized if heuristics pass, standard otherwise

    eprintln!("=== Strategy Comparison ===\n");

    for &thresh in &thresholds {
        eprintln!("--- Threshold: {}% ---", thresh);

        let mut optimized_overheads = Vec::new();
        let mut standard_overheads = Vec::new();
        let mut hybrid_overheads = Vec::new();
        let mut hybrid_choices: Vec<&str> = Vec::new();

        for img_path in &test_images {
            let (width, height, pixels) = load_png(img_path.to_str().unwrap());
            let row_size = width as usize * 3;

            // Baseline (full optimization, no streaming limit)
            let baseline_size = encode_baseline(width, height, &pixels);

            // Strategy 1: Optimized tables from partial data
            let opt_size =
                encode_with_threshold(width, height, &pixels, thresh, memory_limit, true);
            let opt_overhead =
                100.0 * (opt_size as f64 - baseline_size as f64) / baseline_size as f64;
            optimized_overheads.push(opt_overhead);

            // Strategy 2: Standard tables (no optimization)
            let std_size =
                encode_with_threshold(width, height, &pixels, thresh, memory_limit, false);
            let std_overhead =
                100.0 * (std_size as f64 - baseline_size as f64) / baseline_size as f64;
            standard_overheads.push(std_overhead);

            // Strategy 3: Hybrid - check heuristics at transition point
            let (hybrid_size, used_optimized) =
                encode_hybrid(width, height, &pixels, thresh, memory_limit);
            let hybrid_overhead =
                100.0 * (hybrid_size as f64 - baseline_size as f64) / baseline_size as f64;
            hybrid_overheads.push(hybrid_overhead);
            hybrid_choices.push(if used_optimized { "opt" } else { "std" });
        }

        // Statistics
        let opt_mean = mean(&optimized_overheads);
        let opt_max = max(&optimized_overheads);
        let opt_fail = optimized_overheads.iter().filter(|&&o| o > 4.0).count();

        let std_mean = mean(&standard_overheads);
        let std_max = max(&standard_overheads);
        let std_fail = standard_overheads.iter().filter(|&&o| o > 4.0).count();

        let hyb_mean = mean(&hybrid_overheads);
        let hyb_max = max(&hybrid_overheads);
        let hyb_fail = hybrid_overheads.iter().filter(|&&o| o > 4.0).count();
        let hyb_used_std = hybrid_choices.iter().filter(|&&c| c == "std").count();

        eprintln!(
            "{:>20} {:>10} {:>10} {:>10}",
            "Strategy", "Mean", "Max", "Fail >4%"
        );
        eprintln!("{}", "-".repeat(55));
        eprintln!(
            "{:>20} {:>9.2}% {:>9.2}% {:>6}/{}",
            "Optimized",
            opt_mean,
            opt_max,
            opt_fail,
            test_images.len()
        );
        eprintln!(
            "{:>20} {:>9.2}% {:>9.2}% {:>6}/{}",
            "Standard",
            std_mean,
            std_max,
            std_fail,
            test_images.len()
        );
        eprintln!(
            "{:>20} {:>9.2}% {:>9.2}% {:>6}/{} ({} used std)",
            "Hybrid",
            hyb_mean,
            hyb_max,
            hyb_fail,
            test_images.len(),
            hyb_used_std
        );
        eprintln!();
    }

    // Also test with relaxed threshold (e.g., 10% acceptable overhead)
    eprintln!("=== With relaxed 10% threshold ===\n");

    for &thresh in &thresholds {
        let mut optimized_overheads = Vec::new();

        for img_path in &test_images {
            let (width, height, pixels) = load_png(img_path.to_str().unwrap());
            let baseline_size = encode_baseline(width, height, &pixels);
            let opt_size = encode_with_threshold(width, height, &pixels, thresh, 1024 * 1024, true);
            let opt_overhead =
                100.0 * (opt_size as f64 - baseline_size as f64) / baseline_size as f64;
            optimized_overheads.push(opt_overhead);
        }

        let fail_10 = optimized_overheads.iter().filter(|&&o| o > 10.0).count();
        let fail_15 = optimized_overheads.iter().filter(|&&o| o > 15.0).count();

        eprintln!("{}%: {} fail >10%, {} fail >15%", thresh, fail_10, fail_15);
    }
}

fn load_png(path: &str) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::fs::File::open(path).expect("Failed to open file"));
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}

fn encode_baseline(width: u32, height: u32, pixels: &[u8]) -> usize {
    let mut encoder = StreamingEncoder::new(width, height)
        .quality(Quality::ApproxJpegli(85.0))
        .subsampling(Subsampling::S420)
        .progressive(false)
        .start()
        .unwrap();

    let row_size = width as usize * 3;
    for y in 0..height as usize {
        let start = y * row_size;
        let end = start + row_size;
        encoder.push_row(&pixels[start..end]).unwrap();
    }
    encoder.finish().unwrap().len()
}

fn encode_with_threshold(
    width: u32,
    height: u32,
    pixels: &[u8],
    threshold_percent: usize,
    _memory_limit: usize,
    use_standard_tables: bool,
) -> usize {
    let threshold_rows = (height as usize * threshold_percent) / 100;

    let mut builder = StreamingEncoder::new(width, height)
        .quality(Quality::ApproxJpegli(85.0))
        .subsampling(Subsampling::S420)
        .progressive(false)
        .transition_after_rows(threshold_rows);

    if use_standard_tables {
        builder = builder.use_standard_huffman_tables(true);
    }

    let mut encoder = builder.start().unwrap();

    let row_size = width as usize * 3;
    for y in 0..height as usize {
        let start = y * row_size;
        let end = start + row_size;
        encoder.push_row(&pixels[start..end]).unwrap();
    }
    encoder.finish().unwrap().len()
}

fn encode_hybrid(
    width: u32,
    height: u32,
    pixels: &[u8],
    threshold_percent: usize,
    memory_limit: usize,
) -> (usize, bool) {
    // First, probe the heuristics at the threshold point
    let threshold_rows = (height as usize * threshold_percent) / 100;

    let mut probe = StreamingEncoder::new(width, height)
        .quality(Quality::ApproxJpegli(85.0))
        .subsampling(Subsampling::S420)
        .progressive(false)
        .start()
        .unwrap();

    let row_size = width as usize * 3;
    for y in 0..threshold_rows {
        let start = y * row_size;
        let end = start + row_size;
        probe.push_row(&pixels[start..end]).unwrap();
    }

    let (cov, ent, _, _) = probe.frequency_heuristics();
    let heuristics_pass = ent >= 4.0 && cov >= 0.30; // 30% coverage

    // Now encode with the chosen strategy
    let use_optimized = heuristics_pass;
    let size = encode_with_threshold(
        width,
        height,
        pixels,
        threshold_percent,
        memory_limit,
        use_optimized,
    );

    (size, use_optimized)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn max(values: &[f64]) -> f64 {
    values.iter().cloned().fold(f64::MIN, f64::max)
}
