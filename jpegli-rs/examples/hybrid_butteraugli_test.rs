//! Hybrid vs standard comparison with butteraugli metric for sharpened images
//!
//! Run with:
//! cargo run --release --example hybrid_butteraugli_test --features hybrid-trellis

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use butteraugli::{compute_butteraugli, ButteraugliParams};
use dssim::Dssim;

#[cfg(not(feature = "hybrid-trellis"))]
fn main() {
    eprintln!("Requires: --features hybrid-trellis");
}

#[cfg(feature = "hybrid-trellis")]
fn main() {
    let args: Vec<String> = env::args().collect();
    let corpus_dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/lilith/work/codec-eval/corpus/sharpened-800px")
    });

    let output_csv = corpus_dir.join("hybrid_butteraugli.csv");

    let mut files: Vec<PathBuf> = fs::read_dir(&corpus_dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "png") &&
            p.file_name().unwrap().to_string_lossy().starts_with("clic_")
        })
        .collect();
    files.sort();

    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    files.truncate(max_files);

    println!("Testing {} images with butteraugli + dssim\n", files.len());

    let distances = [1.0f32, 2.0];
    let attr = Dssim::new();

    let mut csv = fs::File::create(&output_csv).expect("create csv");
    writeln!(csv, "image,distance,mode,file_size,bpp,dssim,butteraugli").unwrap();

    #[derive(Default, Clone)]
    struct Stats { bpp: f64, dssim: f64, butter: f64, count: usize }
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

        // For DSSIM
        let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
            .chunks(3)
            .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
            .collect();
        let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

        for &distance in &distances {
            // Standard jpegli
            let std_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .encode(pixels)
                .unwrap();

            // Hybrid jpegli
            let hybrid_result = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .quality(jpegli::quant::Quality::from_distance(distance))
                .hybrid_trellis(true)
                .encode(pixels)
                .unwrap();

            let modes = [
                ("standard", &std_result),
                ("hybrid", &hybrid_result),
            ];

            print!("  d={:.1}: ", distance);

            for (name, data) in &modes {
                let bpp = (data.len() * 8) as f64 / total_pixels as f64;
                let dssim = compute_dssim(&attr, &orig_img, data, width, height);
                let butter = compute_butter(pixels, data, width, height);

                print!("{}={:.2}bpp/{:.4}dssim/{:.2}ba ", name, bpp, dssim, butter);

                writeln!(csv, "{},{},{},{},{:.4},{:.6},{:.4}",
                    filename, distance, name, data.len(), bpp, dssim, butter).unwrap();

                let entry = stats.entry((format!("{}", distance), name.to_string())).or_default();
                entry.bpp += bpp;
                entry.dssim += dssim;
                entry.butter += butter;
                entry.count += 1;
            }
            println!();
        }
    }

    println!("\n=== Summary ===");
    println!("{:>8} {:>10} {:>10} {:>10} {:>12}",
        "dist", "mode", "avg_bpp", "avg_dssim", "avg_butter");

    let mut keys: Vec<_> = stats.keys().collect();
    keys.sort();

    for (dist, mode) in keys {
        let s = stats.get(&(dist.clone(), mode.clone())).unwrap();
        let n = s.count as f64;
        println!("{:>8} {:>10} {:>10.3} {:>10.6} {:>12.3}",
            dist, mode, s.bpp / n, s.dssim / n, s.butter / n);
    }

    // Compare RD efficiency using butteraugli
    println!("\n=== Butteraugli RD Efficiency ===");
    for dist in ["1", "2"] {
        let std = stats.get(&(dist.to_string(), "standard".to_string()));
        let hyb = stats.get(&(dist.to_string(), "hybrid".to_string()));

        if let (Some(s), Some(h)) = (std, hyb) {
            let std_rd = (s.bpp / s.count as f64) * (s.butter / s.count as f64);
            let hyb_rd = (h.bpp / h.count as f64) * (h.butter / h.count as f64);
            let improvement = (1.0 - hyb_rd / std_rd) * 100.0;
            println!("  d={}: std_RD={:.4} hybrid_RD={:.4} → hybrid is {:.1}% better",
                dist, std_rd, hyb_rd, improvement);
        }
    }

    println!("\nResults saved to {:?}", output_csv);
}

#[cfg(feature = "hybrid-trellis")]
fn compute_dssim(
    attr: &Dssim,
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
fn compute_butter(orig: &[u8], jpeg_data: &[u8], width: usize, height: usize) -> f64 {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");

    let params = ButteraugliParams::default();
    compute_butteraugli(orig, &decoded, width, height, &params)
        .map(|r| r.score)
        .unwrap_or(999.0)
}
