//! Combined test: hybrid trellis + AQ scale 0.25 for sharpened images
//!
//! Tests if combining both optimizations is beneficial.

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
    use jpegli::adaptive_quant::compute_aq_strength_map;

    let args: Vec<String> = env::args().collect();
    let corpus_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/lilith/work/codec-eval/corpus/sharpened-800px")
    });

    let output_csv = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        corpus_dir.join("hybrid_aq_combined.csv")
    });

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

    println!("Testing {} images - comparing 4 modes:", files.len());
    println!("  1. standard (default AQ)");
    println!("  2. aq025 (AQ scale 0.25)");
    println!("  3. hybrid (default AQ + trellis)");
    println!("  4. hybrid_aq025 (AQ 0.25 + trellis)\n");

    let distances = [1.0f32, 2.0];
    let attr = Dssim::new();

    let mut csv = fs::File::create(&output_csv).expect("create csv");
    writeln!(csv, "image,distance,mode,file_size,bpp,dssim,rd").unwrap();

    #[derive(Default)]
    struct Stats { bpp_sum: f64, dssim_sum: f64, rd_sum: f64, count: usize }
    let mut stats: std::collections::HashMap<(String, String), Stats> = std::collections::HashMap::new();

    for (idx, file) in files.iter().enumerate() {
        let filename = file.file_name().unwrap().to_string_lossy();
        println!("[{}/{}] {}", idx + 1, files.len(), filename);

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

        let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
            .chunks(3)
            .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
            .collect();
        let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

        // Compute Y plane for AQ
        let y_plane: Vec<f32> = pixels
            .chunks(3)
            .map(|rgb| 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
            .collect();

        for &distance in &distances {
            let y_quant_01 = (distance * 8.0).max(1.0) as u16;

            // Mode 1: standard (default AQ)
            let std_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .encode(pixels)
                .unwrap();

            // Mode 2: AQ scale 0.25
            let mut aq_map = compute_aq_strength_map(&y_plane, width, height, y_quant_01);
            aq_map.scale(0.25);
            let aq025_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .aq_map(aq_map)
                .encode(pixels)
                .unwrap();

            // Mode 3: hybrid (default AQ + trellis)
            let hybrid_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .hybrid_trellis(true)
                .encode(pixels)
                .unwrap();

            // Mode 4: hybrid + AQ 0.25
            let mut aq_map2 = compute_aq_strength_map(&y_plane, width, height, y_quant_01);
            aq_map2.scale(0.25);
            let hybrid_aq025_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .aq_map(aq_map2)
                .hybrid_trellis(true)
                .encode(pixels)
                .unwrap();

            let modes = [
                ("standard", &std_result),
                ("aq025", &aq025_result),
                ("hybrid", &hybrid_result),
                ("hybrid_aq025", &hybrid_aq025_result),
            ];

            print!("  d={:.1}: ", distance);
            for (name, data) in &modes {
                let bpp = (data.len() * 8) as f64 / total_pixels as f64;
                let dssim = compute_dssim(&attr, &orig_img, data, width, height);
                let rd = bpp * dssim;

                print!("{}={:.2}/{:.5} ", name, bpp, dssim);

                writeln!(csv, "{},{},{},{},{:.4},{:.6},{:.6}",
                    filename, distance, name, data.len(), bpp, dssim, rd).unwrap();

                let entry = stats.entry((format!("{}", distance), name.to_string())).or_default();
                entry.bpp_sum += bpp;
                entry.dssim_sum += dssim;
                entry.rd_sum += rd;
                entry.count += 1;
            }
            println!();
        }
    }

    println!("\n=== Summary ===");
    println!("{:>8} {:>12} {:>10} {:>10} {:>10}",
        "dist", "mode", "avg_bpp", "avg_dssim", "avg_RD");

    let mut keys: Vec<_> = stats.keys().collect();
    keys.sort();

    for (dist, mode) in keys {
        let s = stats.get(&(dist.clone(), mode.clone())).unwrap();
        let n = s.count as f64;
        println!("{:>8} {:>12} {:>10.3} {:>10.6} {:>10.6}",
            dist, mode, s.bpp_sum / n, s.dssim_sum / n, s.rd_sum / n);
    }

    // Find best mode per distance
    println!("\n=== Best RD per distance ===");
    for dist in ["1", "2"] {
        let mut best_mode = "";
        let mut best_rd = f64::MAX;
        for mode in ["standard", "aq025", "hybrid", "hybrid_aq025"] {
            if let Some(s) = stats.get(&(dist.to_string(), mode.to_string())) {
                let rd = s.rd_sum / s.count as f64;
                if rd < best_rd {
                    best_rd = rd;
                    best_mode = mode;
                }
            }
        }
        println!("  distance {}: {} (RD={:.6})", dist, best_mode, best_rd);
    }

    println!("\nResults saved to {:?}", output_csv);
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
