//! Hybrid vs non-hybrid comparison for sharpened images
//!
//! Tests rate-distortion efficiency across quality levels.
//!
//! Run with:
//! cargo run --release --example hybrid_sharpened_test --features hybrid-trellis -- /path/to/images

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[cfg(not(feature = "hybrid-trellis"))]
fn main() {
    eprintln!("Requires: --features hybrid-trellis");
}

#[cfg(feature = "hybrid-trellis")]
fn main() {
    use dssim::Dssim;

    let args: Vec<String> = env::args().collect();
    let corpus_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/lilith/work/codec-eval/corpus/sharpened-800px")
    });

    let output_csv = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        corpus_dir.join("hybrid_comparison.csv")
    });

    // Collect PNG files (skip non-image PNGs)
    let mut files: Vec<PathBuf> = fs::read_dir(&corpus_dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "png") &&
            !p.file_name().unwrap().to_string_lossy().starts_with("bpp_")
        })
        .collect();
    files.sort();

    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    files.truncate(max_files);

    println!("Testing {} images from {:?}", files.len(), corpus_dir);

    // Quality levels to test (jpegli distance values)
    let distances = [0.5f32, 1.0, 1.5, 2.0, 3.0];

    let mut csv = fs::File::create(&output_csv).expect("create csv");
    writeln!(csv, "image,distance,mode,file_size,bpp,dssim").unwrap();

    let attr = Dssim::new();

    for (idx, file) in files.iter().enumerate() {
        let filename = file.file_name().unwrap().to_string_lossy();
        println!("\n[{}/{}] {}", idx + 1, files.len(), filename);

        // Load image
        let Ok(f) = fs::File::open(file) else { continue };
        let decoder = png::Decoder::new(f);
        let Ok(mut reader) = decoder.read_info() else { continue };
        let mut buf = vec![0; reader.output_buffer_size()];
        let Ok(info) = reader.next_frame(&mut buf) else { continue };

        if info.color_type != png::ColorType::Rgb {
            println!("  Skipping (not RGB)");
            continue;
        }

        let pixels = &buf[..info.buffer_size()];
        let width = info.width as usize;
        let height = info.height as usize;
        let total_pixels = width * height;

        // Create original for DSSIM
        let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
            .chunks(3)
            .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
            .collect();
        let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

        for &distance in &distances {
            // Standard jpegli (no hybrid)
            let std_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .encode(pixels)
                .unwrap();

            // Hybrid jpegli (AQ + trellis)
            let hybrid_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .hybrid_trellis(true)
                .encode(pixels)
                .unwrap();

            // Measure DSSIM
            let std_dssim = compute_dssim(&attr, &orig_img, &std_result, width, height);
            let hybrid_dssim = compute_dssim(&attr, &orig_img, &hybrid_result, width, height);

            let std_bpp = (std_result.len() * 8) as f64 / total_pixels as f64;
            let hybrid_bpp = (hybrid_result.len() * 8) as f64 / total_pixels as f64;

            // RD efficiency
            let std_rd = std_dssim * std_bpp;
            let hybrid_rd = hybrid_dssim * hybrid_bpp;

            let better = if hybrid_rd < std_rd { "hybrid" } else { "std" };

            println!(
                "  d={:.1}: std={:.2}bpp/{:.5}dssim  hybrid={:.2}bpp/{:.5}dssim  winner={}",
                distance, std_bpp, std_dssim, hybrid_bpp, hybrid_dssim, better
            );

            // Write to CSV
            writeln!(csv, "{},{},{},{},{:.4},{:.6}",
                filename, distance, "standard", std_result.len(), std_bpp, std_dssim).unwrap();
            writeln!(csv, "{},{},{},{},{:.4},{:.6}",
                filename, distance, "hybrid", hybrid_result.len(), hybrid_bpp, hybrid_dssim).unwrap();
        }
    }

    println!("\n=== Results written to {:?} ===", output_csv);

    // Quick summary
    summarize_csv(&output_csv);
}

#[cfg(feature = "hybrid-trellis")]
fn compute_dssim(
    attr: &dssim::Dssim,
    orig: &dssim::DssimImage<f32>,
    jpeg_data: &[u8],
    width: usize,
    height: usize,
) -> f64 {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr.create_image_rgba(&decoded_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(orig, decoded_img);
    dssim.into()
}

#[cfg(feature = "hybrid-trellis")]
fn summarize_csv(csv_path: &PathBuf) {
    use std::collections::HashMap;

    let content = fs::read_to_string(csv_path).unwrap();
    let mut std_stats: HashMap<String, (f64, f64, usize)> = HashMap::new(); // bpp_sum, dssim_sum, count
    let mut hybrid_stats: HashMap<String, (f64, f64, usize)> = HashMap::new();

    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 { continue; }
        let distance = parts[1];
        let mode = parts[2];
        let bpp: f64 = parts[4].parse().unwrap_or(0.0);
        let dssim: f64 = parts[5].parse().unwrap_or(0.0);

        let stats = if mode == "standard" { &mut std_stats } else { &mut hybrid_stats };
        let entry = stats.entry(distance.to_string()).or_insert((0.0, 0.0, 0));
        entry.0 += bpp;
        entry.1 += dssim;
        entry.2 += 1;
    }

    println!("\n=== Summary by Distance ===");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "distance", "std_bpp", "std_dssim", "hyb_bpp", "hyb_dssim", "std_RD", "hyb_RD", "winner");

    let mut distances: Vec<&str> = std_stats.keys().map(|s| s.as_str()).collect();
    distances.sort_by(|a, b| a.parse::<f64>().unwrap().partial_cmp(&b.parse::<f64>().unwrap()).unwrap());

    let mut std_rd_total = 0.0;
    let mut hybrid_rd_total = 0.0;

    for d in distances {
        let std = std_stats.get(d).unwrap();
        let hyb = hybrid_stats.get(d).unwrap();

        let std_bpp = std.0 / std.2 as f64;
        let std_dssim = std.1 / std.2 as f64;
        let hyb_bpp = hyb.0 / hyb.2 as f64;
        let hyb_dssim = hyb.1 / hyb.2 as f64;

        let std_rd = std_bpp * std_dssim;
        let hyb_rd = hyb_bpp * hyb_dssim;

        std_rd_total += std_rd;
        hybrid_rd_total += hyb_rd;

        let winner = if hyb_rd < std_rd { "hybrid" } else { "standard" };

        println!("{:>8} {:>10.3} {:>10.6} {:>10.3} {:>10.6} {:>12.6} {:>12.6} {:>8}",
            d, std_bpp, std_dssim, hyb_bpp, hyb_dssim, std_rd, hyb_rd, winner);
    }

    println!("\nOverall RD: standard={:.6} hybrid={:.6}", std_rd_total, hybrid_rd_total);
    let pct = (1.0 - hybrid_rd_total / std_rd_total) * 100.0;
    println!("Hybrid is {:.1}% {} in rate-distortion efficiency",
        pct.abs(), if pct > 0.0 { "better" } else { "worse" });
}
