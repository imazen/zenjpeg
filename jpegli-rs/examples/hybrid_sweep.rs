//! Parameter sweep for hybrid AQ+trellis optimization.
//!
//! Systematically tests different parameter combinations and outputs results
//! in CSV format for analysis.
//!
//! Run with:
//! ```
//! # Quick sweep (few params, fast)
//! cargo run --release --example hybrid_sweep --features hybrid-trellis
//!
//! # Comprehensive sweep (many params, slower)
//! SWEEP=comprehensive cargo run --release --example hybrid_sweep --features hybrid-trellis
//!
//! # Custom image directory
//! cargo run --release --example hybrid_sweep --features hybrid-trellis -- /path/to/images
//!
//! # Limit images for faster testing
//! MAX_FILES=5 cargo run --release --example hybrid_sweep --features hybrid-trellis
//! ```
//!
//! Output is CSV to stdout, can be redirected:
//! ```
//! cargo run --release --example hybrid_sweep --features hybrid-trellis > results.csv
//! ```

use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(not(feature = "hybrid-trellis"))]
fn main() {
    eprintln!("This example requires the hybrid-trellis feature.");
    eprintln!("Run with: cargo run --release --example hybrid_sweep --features hybrid-trellis");
}

#[cfg(feature = "hybrid-trellis")]
fn main() {
    use jpegli::hybrid_config::SweepConfig;

    // Determine sweep type
    let sweep_config = match env::var("SWEEP").as_deref() {
        Ok("comprehensive") => {
            eprintln!("Using comprehensive sweep configuration");
            SweepConfig::comprehensive()
        }
        Ok("quick") | Err(_) => {
            eprintln!("Using quick sweep configuration (set SWEEP=comprehensive for full sweep)");
            SweepConfig::quick()
        }
        Ok(other) => {
            eprintln!("Unknown sweep type '{}', using quick", other);
            SweepConfig::quick()
        }
    };

    // Find images
    let args: Vec<String> = env::args().collect();
    let image_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
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

    // Collect PNG files
    let mut files: Vec<PathBuf> = std::fs::read_dir(&image_dir)
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

    if files.is_empty() {
        eprintln!("No PNG files found in {}", image_dir.display());
        return;
    }

    // Load images into memory
    eprintln!("Loading {} images from {}", files.len(), image_dir.display());
    let images: Vec<ImageData> = files
        .iter()
        .filter_map(|f| load_image(f))
        .collect();
    eprintln!("Loaded {} images", images.len());

    // Generate configs
    let configs = sweep_config.generate_configs();
    let total = configs.len() * sweep_config.quality_levels.len() * images.len();
    eprintln!(
        "Testing {} configs × {} qualities × {} images = {} combinations",
        configs.len(),
        sweep_config.quality_levels.len(),
        images.len(),
        total
    );

    // Print CSV header
    println!(
        "config_id,aq_lambda_scale,base_scale1,dc_enabled,aq_exponent,quality,image,\
         jpegli_size,hybrid_size,jpegli_dssim,hybrid_dssim,\
         size_ratio,dssim_ratio,encode_time_ms"
    );

    let mut completed = 0;
    let start_total = Instant::now();

    for quality in &sweep_config.quality_levels {
        // Encode baseline jpegli once per quality level
        let jpegli_results: Vec<(usize, f64)> = images
            .iter()
            .map(|img| encode_jpegli(img, *quality))
            .collect();

        for config in &configs {
            for (img_idx, img) in images.iter().enumerate() {
                let (jpegli_size, jpegli_dssim) = jpegli_results[img_idx];

                // Encode with hybrid config
                let start = Instant::now();
                let (hybrid_size, hybrid_dssim) = encode_hybrid(img, *quality, config);
                let encode_time = start.elapsed().as_secs_f64() * 1000.0;

                let size_ratio = hybrid_size as f64 / jpegli_size as f64;
                let dssim_ratio = hybrid_dssim / jpegli_dssim;

                println!(
                    "{},{:.2},{:.2},{},{:.2},{},{},{},{},{:.6},{:.6},{:.4},{:.4},{:.2}",
                    config.id(),
                    config.aq_lambda_scale,
                    config.base_lambda_scale1,
                    config.dc_enabled as u8,
                    config.aq_exponent,
                    quality,
                    img.name,
                    jpegli_size,
                    hybrid_size,
                    jpegli_dssim,
                    hybrid_dssim,
                    size_ratio,
                    dssim_ratio,
                    encode_time
                );

                // Flush to see progress in real-time
                io::stdout().flush().ok();

                completed += 1;
                if completed % 10 == 0 {
                    let elapsed = start_total.elapsed().as_secs_f64();
                    let rate = completed as f64 / elapsed;
                    let remaining = (total - completed) as f64 / rate;
                    eprintln!(
                        "Progress: {}/{} ({:.1}%), ETA: {:.0}s",
                        completed,
                        total,
                        100.0 * completed as f64 / total as f64,
                        remaining
                    );
                }
            }
        }
    }

    eprintln!(
        "Completed {} combinations in {:.1}s",
        total,
        start_total.elapsed().as_secs_f64()
    );
}

#[cfg(feature = "hybrid-trellis")]
struct ImageData {
    name: String,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

#[cfg(feature = "hybrid-trellis")]
fn load_image(path: &PathBuf) -> Option<ImageData> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    if info.color_type != png::ColorType::Rgb {
        return None;
    }

    Some(ImageData {
        name: path.file_name()?.to_string_lossy().to_string(),
        pixels: buf[..info.buffer_size()].to_vec(),
        width: info.width as usize,
        height: info.height as usize,
    })
}

#[cfg(feature = "hybrid-trellis")]
fn encode_jpegli(img: &ImageData, quality: u8) -> (usize, f64) {
    let result = jpegli::Encoder::new()
        .width(img.width as u32)
        .height(img.height as u32)
        .quality(jpegli::quant::Quality::from_quality(quality as f32))
        .encode(&img.pixels)
        .expect("jpegli encode");

    let dssim = compute_dssim(&img.pixels, img.width, img.height, &result);

    (result.len(), dssim)
}

#[cfg(feature = "hybrid-trellis")]
fn encode_hybrid(
    img: &ImageData,
    quality: u8,
    config: &jpegli::hybrid_config::HybridConfig,
) -> (usize, f64) {
    // Use the full HybridConfig
    let result = jpegli::Encoder::new()
        .width(img.width as u32)
        .height(img.height as u32)
        .quality(jpegli::quant::Quality::from_quality(quality as f32))
        .hybrid_config(*config)
        .encode(&img.pixels)
        .expect("hybrid encode");

    let dssim = compute_dssim(&img.pixels, img.width, img.height, &result);

    (result.len(), dssim)
}

#[cfg(feature = "hybrid-trellis")]
fn compute_dssim(original: &[u8], width: usize, height: usize, jpeg_data: &[u8]) -> f64 {
    use dssim::Dssim;

    let attr = Dssim::new();

    // Original
    let orig_rgba: Vec<rgb::RGBA<u8>> = original
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = attr
        .create_image_rgba(&orig_rgba, width, height)
        .expect("create orig");

    // Decoded
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");
    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr
        .create_image_rgba(&decoded_rgba, width, height)
        .expect("create decoded");

    let (dssim, _) = attr.compare(&orig_img, decoded_img);
    dssim.into()
}
