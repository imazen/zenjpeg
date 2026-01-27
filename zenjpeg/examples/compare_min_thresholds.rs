//! Compare different min_transition_percent values with heuristics.
//!
//! Tests 25%, 30%, 35%, and 40% minimums to find the sweet spot.
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example compare_min_thresholds

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

    eprintln!(
        "Testing {} images with different min_percent values\n",
        test_images.len()
    );

    let min_percents = [25, 30, 35, 40, 50];
    let memory_limit = 1024 * 1024; // 1MB

    // Results: (min_pct, mean_overhead, max_overhead, over_4_count, mean_trans_pct)
    let mut results: Vec<(usize, f64, f64, usize, f64)> = Vec::new();

    for &min_pct in &min_percents {
        let mut overheads = Vec::new();
        let mut trans_pcts = Vec::new();

        for img_path in &test_images {
            let (width, height, pixels) = load_png(img_path.to_str().unwrap());

            // Baseline (no streaming)
            let baseline_size = {
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
            };

            // With heuristics at this min_percent
            let mut encoder = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(85.0))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .memory_limit(memory_limit)
                .min_transition_percent(min_pct)
                .min_entropy(4.0)
                .min_coverage(30.0)
                .start()
                .unwrap();

            let row_size = width as usize * 3;
            for y in 0..height as usize {
                let start = y * row_size;
                let end = start + row_size;
                encoder.push_row(&pixels[start..end]).unwrap();
            }

            let trans_pct = encoder.transition_percent().unwrap_or(100.0);
            let result = encoder.finish().unwrap();
            let overhead =
                100.0 * (result.len() as f64 - baseline_size as f64) / baseline_size as f64;

            overheads.push(overhead);
            trans_pcts.push(trans_pct);
        }

        let mean_overhead = overheads.iter().sum::<f64>() / overheads.len() as f64;
        let max_overhead = overheads.iter().cloned().fold(f64::MIN, f64::max);
        let over_4_count = overheads.iter().filter(|&&o| o > 4.0).count();
        let mean_trans = trans_pcts.iter().sum::<f64>() / trans_pcts.len() as f64;

        results.push((
            min_pct,
            mean_overhead,
            max_overhead,
            over_4_count,
            mean_trans,
        ));
    }

    // Print comparison
    eprintln!(
        "{:>10} {:>12} {:>12} {:>12} {:>15}",
        "Min %", "Mean OH", "Max OH", "Fail >4%", "Mean Trans%"
    );
    eprintln!("{}", "-".repeat(65));

    for (min_pct, mean_oh, max_oh, fail_count, mean_trans) in &results {
        eprintln!(
            "{:>9}% {:>11.2}% {:>11.2}% {:>8}/{:<4} {:>14.1}%",
            min_pct,
            mean_oh,
            max_oh,
            fail_count,
            test_images.len(),
            mean_trans
        );
    }

    eprintln!("\nNotes:");
    eprintln!("  - Higher min% = better worst-case but slower transition");
    eprintln!("  - 'Fail >4%' = images with overhead above 4% threshold");
    eprintln!("  - 'Mean Trans%' = average % of image processed before streaming");
}

fn load_png(path: &str) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::fs::File::open(path).expect("Failed to open file"));
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}
