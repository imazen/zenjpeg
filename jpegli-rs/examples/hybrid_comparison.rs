//! Comparison benchmark: jpegli vs hybrid (AQ+trellis) vs mozjpeg
//!
//! Run with:
//! cargo run --release --example hybrid_comparison --features hybrid-trellis
//!
//! Or with a specific directory:
//! cargo run --release --example hybrid_comparison --features hybrid-trellis -- /path/to/images

use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(not(feature = "hybrid-trellis"))]
fn main() {
    eprintln!("This example requires the hybrid-trellis feature.");
    eprintln!("Run with: cargo run --release --example hybrid_comparison --features hybrid-trellis");
}

#[cfg(feature = "hybrid-trellis")]
fn main() {
    // Find images to test
    let args: Vec<String> = env::args().collect();
    let image_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        // Try to find codec-corpus
        let paths = [
            PathBuf::from("/home/lilith/work/codec-eval/codec-corpus/kodak"),
            PathBuf::from("../codec-eval/codec-corpus/kodak"),
            PathBuf::from("/mnt/v/work/corpus/CID22-512"),
        ];
        paths.into_iter().find(|p| p.exists()).unwrap_or_else(|| {
            eprintln!("No image directory found. Please provide a path as argument.");
            std::process::exit(1);
        })
    };

    println!("=== Hybrid Encoder Comparison ===\n");
    println!("Image directory: {}\n", image_dir.display());

    // Collect PNG files
    let mut files: Vec<PathBuf> = std::fs::read_dir(&image_dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();

    // Limit to first N files for quick testing
    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    files.truncate(max_files);

    if files.is_empty() {
        eprintln!("No PNG files found in {}", image_dir.display());
        return;
    }

    println!("Testing {} images at quality 75\n", files.len());

    // Results accumulators
    let mut results = Vec::new();
    let quality = 75u8;

    for file in &files {
        let filename = file.file_name().unwrap().to_string_lossy();

        // Load image
        let decoder = png::Decoder::new(std::fs::File::open(file).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let pixels = &buf[..info.buffer_size()];

        if info.color_type != png::ColorType::Rgb {
            eprintln!("Skipping {} (not RGB)", filename);
            continue;
        }

        let width = info.width as usize;
        let height = info.height as usize;

        // Encode with each method
        let result = encode_and_compare(pixels, width, height, quality, &filename);
        results.push(result);
    }

    // Print summary table
    println!("\n{}", "=".repeat(100));
    println!("{:30} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "Image", "jpegli", "hybrid", "mozjpeg", "j_dssim", "h_dssim", "m_dssim");
    println!("{}", "-".repeat(100));

    let mut total_jpegli = 0usize;
    let mut total_hybrid = 0usize;
    let mut total_mozjpeg = 0usize;
    let mut sum_jpegli_dssim = 0.0f64;
    let mut sum_hybrid_dssim = 0.0f64;
    let mut sum_mozjpeg_dssim = 0.0f64;

    for r in &results {
        println!("{:30} {:>12} {:>12} {:>12} {:>10.6} {:>10.6} {:>10.6}",
            r.name, r.jpegli_size, r.hybrid_size, r.mozjpeg_size,
            r.jpegli_dssim, r.hybrid_dssim, r.mozjpeg_dssim);

        total_jpegli += r.jpegli_size;
        total_hybrid += r.hybrid_size;
        total_mozjpeg += r.mozjpeg_size;
        sum_jpegli_dssim += r.jpegli_dssim;
        sum_hybrid_dssim += r.hybrid_dssim;
        sum_mozjpeg_dssim += r.mozjpeg_dssim;
    }

    let n = results.len() as f64;
    println!("{}", "-".repeat(100));
    println!("{:30} {:>12} {:>12} {:>12} {:>10.6} {:>10.6} {:>10.6}",
        "TOTAL/MEAN",
        total_jpegli, total_hybrid, total_mozjpeg,
        sum_jpegli_dssim / n, sum_hybrid_dssim / n, sum_mozjpeg_dssim / n);

    // Summary analysis
    println!("\n=== Summary ===\n");

    let hybrid_vs_jpegli_size = (total_hybrid as f64 / total_jpegli as f64 - 1.0) * 100.0;
    let mozjpeg_vs_jpegli_size = (total_mozjpeg as f64 / total_jpegli as f64 - 1.0) * 100.0;

    let hybrid_vs_jpegli_dssim = sum_hybrid_dssim / sum_jpegli_dssim;
    let mozjpeg_vs_jpegli_dssim = sum_mozjpeg_dssim / sum_jpegli_dssim;

    println!("Hybrid vs jpegli:");
    println!("  File size: {:+.2}%", hybrid_vs_jpegli_size);
    println!("  DSSIM ratio: {:.3}x (lower = better quality)", hybrid_vs_jpegli_dssim);

    println!("\nmozjpeg vs jpegli:");
    println!("  File size: {:+.2}%", mozjpeg_vs_jpegli_size);
    println!("  DSSIM ratio: {:.3}x (lower = better quality)", mozjpeg_vs_jpegli_dssim);

    // Count wins
    let mut hybrid_wins_size = 0;
    let mut hybrid_wins_dssim = 0;
    let mut mozjpeg_wins_size = 0;
    let mut mozjpeg_wins_dssim = 0;

    for r in &results {
        if r.hybrid_size < r.jpegli_size { hybrid_wins_size += 1; }
        if r.hybrid_dssim < r.jpegli_dssim { hybrid_wins_dssim += 1; }
        if r.mozjpeg_size < r.jpegli_size { mozjpeg_wins_size += 1; }
        if r.mozjpeg_dssim < r.jpegli_dssim { mozjpeg_wins_dssim += 1; }
    }

    println!("\nWins vs jpegli (out of {}):", results.len());
    println!("  Hybrid: {} smaller, {} better quality", hybrid_wins_size, hybrid_wins_dssim);
    println!("  mozjpeg: {} smaller, {} better quality", mozjpeg_wins_size, mozjpeg_wins_dssim);
}

#[cfg(feature = "hybrid-trellis")]
struct CompareResult {
    name: String,
    jpegli_size: usize,
    hybrid_size: usize,
    mozjpeg_size: usize,
    jpegli_dssim: f64,
    hybrid_dssim: f64,
    mozjpeg_dssim: f64,
}

#[cfg(feature = "hybrid-trellis")]
fn encode_and_compare(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    name: &str,
) -> CompareResult {
    use dssim::Dssim;

    print!("{:30} ", name);

    // Create original image for DSSIM comparison
    let attr = Dssim::new();
    let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = attr
        .create_image_rgba(&orig_rgba, width, height)
        .expect("create orig image");

    // 1. Encode with standard jpegli (AQ only)
    let start = Instant::now();
    let jpegli_result = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(jpegli::quant::Quality::from_quality(quality as f32))
        .encode(pixels)
        .unwrap();
    let _jpegli_time = start.elapsed();

    // 2. Encode with hybrid jpegli (AQ + trellis)
    let start = Instant::now();
    let hybrid_result = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(jpegli::quant::Quality::from_quality(quality as f32))
        .hybrid_trellis(true)
        .encode(pixels)
        .unwrap();
    let _hybrid_time = start.elapsed();

    // 3. Encode with mozjpeg
    let start = Instant::now();
    let mozjpeg_result = encode_mozjpeg(pixels, width, height, quality);
    let _mozjpeg_time = start.elapsed();

    // Decode and compute DSSIM
    let jpegli_dssim = compute_dssim(&attr, &orig_img, &jpegli_result, width, height);
    let hybrid_dssim = compute_dssim(&attr, &orig_img, &hybrid_result, width, height);
    let mozjpeg_dssim = compute_dssim(&attr, &orig_img, &mozjpeg_result, width, height);

    println!(
        "j={:>6} h={:>6} m={:>6}  DSSIM: j={:.5} h={:.5} m={:.5}",
        jpegli_result.len(),
        hybrid_result.len(),
        mozjpeg_result.len(),
        jpegli_dssim,
        hybrid_dssim,
        mozjpeg_dssim
    );

    CompareResult {
        name: name.to_string(),
        jpegli_size: jpegli_result.len(),
        hybrid_size: hybrid_result.len(),
        mozjpeg_size: mozjpeg_result.len(),
        jpegli_dssim,
        hybrid_dssim,
        mozjpeg_dssim,
    }
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
    let decoded = decoder.decode().expect("decode JPEG");

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr
        .create_image_rgba(&decoded_rgba, width, height)
        .expect("create decoded image");

    let (dssim, _) = attr.compare(orig, decoded_img);
    dssim.into()
}

#[cfg(feature = "hybrid-trellis")]
fn encode_mozjpeg(pixels: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    std::panic::catch_unwind(|| {
        use mozjpeg::{ColorSpace, Compress};

        let mut comp = Compress::new(ColorSpace::JCS_RGB);
        comp.set_size(width, height);
        comp.set_quality(quality as f32);
        // Use 4:4:4 subsampling to match jpegli defaults
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));

        let mut started = comp
            .start_compress(Vec::new())
            .expect("mozjpeg start error");

        let row_stride = width * 3;
        for y in 0..height {
            let row_start = y * row_stride;
            let row = &pixels[row_start..row_start + row_stride];
            let _ = started.write_scanlines(row);
        }

        started.finish().expect("mozjpeg finish error")
    })
    .unwrap_or_else(|_| {
        eprintln!("mozjpeg panicked");
        Vec::new()
    })
}
