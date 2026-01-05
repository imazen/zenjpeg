//! Analyze AQ complexity and predict hybrid trellis effectiveness.
//!
//! Run with:
//! ```
//! cargo run --release --example aq_analysis --features experimental-hybrid-trellis -- /path/to/images
//! ```

#[cfg(not(feature = "experimental-hybrid-trellis"))]
fn main() {
    eprintln!("This example requires the 'experimental-hybrid-trellis' feature.");
    eprintln!("Run with: cargo run --release --example aq_analysis --features experimental-hybrid-trellis");
}

#[cfg(feature = "experimental-hybrid-trellis")]
fn main() {
    use jpegli::adaptive_quant::compute_aq_strength_map;
    use jpegli::hybrid_config::{
        estimate_hybrid_improvement, should_use_hybrid, AQ_MEAN_THRESHOLD,
    };
    use std::env;
    use std::path::PathBuf;

    struct AQStats {
        min: f32,
        max: f32,
        mean: f32,
        std: f32,
    }

    fn analyze_image(path: &PathBuf) -> Option<AQStats> {
        let file = std::fs::File::open(path).ok()?;
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).ok()?;

        if info.color_type != png::ColorType::Rgb {
            return None;
        }

        let width = info.width as usize;
        let height = info.height as usize;
        let pixels = &buf[..info.buffer_size()];

        // Extract Y plane
        let y_plane: Vec<f32> = pixels
            .chunks(3)
            .map(|c| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32)
            .collect();

        // Use y_quant_01 = 8 (typical for Q75)
        let aq_map = compute_aq_strength_map(&y_plane, width, height, 8);
        let (min, max, mean, std) = aq_map.stats();

        Some(AQStats {
            min,
            max,
            mean,
            std,
        })
    }

    let args: Vec<String> = env::args().collect();
    let image_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("/home/lilith/work/codec-eval/codec-corpus/kodak")
    };

    println!("=== AQ Complexity Analysis ===");
    println!("Threshold for hybrid: aq_mean > {:.2}\n", AQ_MEAN_THRESHOLD);
    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8} {:>12} {:>12}",
        "Image", "AQ Min", "AQ Max", "AQ Mean", "AQ Std", "Use Hybrid?", "Est. Impr%"
    );
    println!("{}", "-".repeat(85));

    let mut files: Vec<_> = std::fs::read_dir(&image_dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();

    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    files.truncate(max_files);

    let mut total_use = 0;
    let mut total_skip = 0;

    for path in &files {
        if let Some(stats) = analyze_image(&path) {
            let use_hybrid = should_use_hybrid(stats.mean);
            let est_improvement = estimate_hybrid_improvement(stats.mean);

            if use_hybrid {
                total_use += 1;
            } else {
                total_skip += 1;
            }

            let decision = if use_hybrid { "YES" } else { "no" };
            println!(
                "{:<15} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>12} {:>11.1}%",
                path.file_name().unwrap().to_string_lossy(),
                stats.min,
                stats.max,
                stats.mean,
                stats.std,
                decision,
                est_improvement
            );
        }
    }

    println!("\n=== Summary ===");
    println!("Images where hybrid recommended: {}", total_use);
    println!("Images where hybrid skipped:     {}", total_skip);
}
