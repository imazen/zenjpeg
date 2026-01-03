//! Parameter sweep for hybrid AQ+trellis optimization.
//!
//! Systematically tests different parameter combinations and outputs results
//! in CSV format with multiple quality metrics for Pareto analysis.
//!
//! Metrics:
//! - DSSIM: Structural dissimilarity (lower = better)
//! - SSIMULACRA2: Perceptual quality 0-100 (higher = better)
//! - Butteraugli: Psychovisual distance (lower = better, <1.0 good, >2.0 bad)
//!
//! Run with:
//! ```
//! # Quick sweep (few params, fast)
//! cargo run --release --example hybrid_sweep --features experimental-hybrid-trellis
//!
//! # Comprehensive sweep (many params, slower)
//! SWEEP=comprehensive cargo run --release --example hybrid_sweep --features experimental-hybrid-trellis
//!
//! # Save results with timestamp
//! cargo run --release --example hybrid_sweep --features experimental-hybrid-trellis \
//!   > sweep_$(date +%Y%m%d_%H%M%S).csv
//! ```

use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(not(feature = "experimental-hybrid-trellis"))]
fn main() {
    eprintln!("This example requires the experimental-hybrid-trellis feature.");
    eprintln!("Run with: cargo run --release --example hybrid_sweep --features experimental-hybrid-trellis");
}

#[cfg(feature = "experimental-hybrid-trellis")]
fn main() {
    use jpegli::hybrid_config::SweepConfig;

    // Determine sweep type
    let sweep_config = match env::var("SWEEP").as_deref() {
        Ok("comprehensive") => {
            eprintln!("Using comprehensive sweep configuration");
            SweepConfig::comprehensive()
        }
        Ok("pareto") => {
            eprintln!("Using Pareto-focused sweep (size vs quality)");
            pareto_sweep_config()
        }
        Ok("multiq") => {
            eprintln!("Using multi-quality sweep (focused params, many Q levels)");
            multiq_sweep_config()
        }
        Ok("quick") | Err(_) => {
            eprintln!("Using quick sweep configuration (set SWEEP=comprehensive or SWEEP=pareto)");
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
    eprintln!(
        "Loading {} images from {}",
        files.len(),
        image_dir.display()
    );
    let images: Vec<ImageData> = files.iter().filter_map(|f| load_image(f)).collect();
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

    // Print CSV header with all metrics
    println!(
        "config_id,aq_lambda_scale,base_scale1,dc_enabled,aq_exponent,quality,image,\
         width,height,pixels,\
         jpegli_bytes,hybrid_bytes,\
         jpegli_bpp,hybrid_bpp,\
         jpegli_dssim,hybrid_dssim,\
         jpegli_ssim2,hybrid_ssim2,\
         jpegli_butteraugli,hybrid_butteraugli,\
         size_ratio,dssim_ratio,ssim2_diff,butteraugli_ratio,\
         encode_time_ms"
    );

    let mut completed = 0;
    let start_total = Instant::now();

    for quality in &sweep_config.quality_levels {
        // Encode baseline jpegli once per quality level per image
        let jpegli_results: Vec<EncodingResult> = images
            .iter()
            .map(|img| encode_and_measure(img, *quality, None))
            .collect();

        for config in &configs {
            for (img_idx, img) in images.iter().enumerate() {
                let jpegli = &jpegli_results[img_idx];

                // Encode with hybrid config
                let start = Instant::now();
                let hybrid = encode_and_measure(img, *quality, Some(config));
                let encode_time = start.elapsed().as_secs_f64() * 1000.0;

                let pixels = img.width * img.height;
                let jpegli_bpp = 8.0 * jpegli.bytes as f64 / pixels as f64;
                let hybrid_bpp = 8.0 * hybrid.bytes as f64 / pixels as f64;

                let size_ratio = hybrid.bytes as f64 / jpegli.bytes as f64;
                let dssim_ratio = hybrid.dssim / jpegli.dssim;
                let ssim2_diff = hybrid.ssim2 - jpegli.ssim2; // positive = hybrid better
                let butteraugli_ratio = hybrid.butteraugli / jpegli.butteraugli;

                println!(
                    "{},{:.2},{:.2},{},{:.2},{},{},{},{},{},{},{},{:.4},{:.4},{:.6},{:.6},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2}",
                    config.id(),
                    config.aq_lambda_scale,
                    config.base_lambda_scale1,
                    config.dc_enabled as u8,
                    config.aq_exponent,
                    quality,
                    img.name,
                    img.width,
                    img.height,
                    pixels,
                    jpegli.bytes,
                    hybrid.bytes,
                    jpegli_bpp,
                    hybrid_bpp,
                    jpegli.dssim,
                    hybrid.dssim,
                    jpegli.ssim2,
                    hybrid.ssim2,
                    jpegli.butteraugli,
                    hybrid.butteraugli,
                    size_ratio,
                    dssim_ratio,
                    ssim2_diff,
                    butteraugli_ratio,
                    encode_time
                );

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

    let total_time = start_total.elapsed().as_secs_f64();
    eprintln!(
        "\nCompleted {} combinations in {:.1}s ({:.1} per second)",
        total,
        total_time,
        total as f64 / total_time
    );

    // Print summary statistics
    eprintln!("\nResults saved. Analyze with:");
    eprintln!("  python3 -c \"import pandas as pd; df = pd.read_csv('results.csv'); print(df.groupby('config_id')[['size_ratio','dssim_ratio','ssim2_diff','butteraugli_ratio']].mean().sort_values('dssim_ratio'))\"");
}

/// Pareto-focused sweep: vary quality to get size variation, test key params
#[cfg(feature = "experimental-hybrid-trellis")]
fn pareto_sweep_config() -> jpegli::hybrid_config::SweepConfig {
    jpegli::hybrid_config::SweepConfig {
        aq_lambda_scales: vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        base_scale1_values: vec![14.0, 14.75, 15.5],
        dc_enabled_values: vec![false, true],
        aq_exponents: vec![1.0, 2.0],
        quality_levels: vec![50, 60, 70, 75, 80, 85, 90, 95],
    }
}

/// Multi-quality sweep: focused params, many quality levels for Pareto curves
#[cfg(feature = "experimental-hybrid-trellis")]
fn multiq_sweep_config() -> jpegli::hybrid_config::SweepConfig {
    jpegli::hybrid_config::SweepConfig {
        // Key aq_lambda_scale values: 0 (no AQ), 2 (default), 4 (aggressive)
        aq_lambda_scales: vec![0.0, 2.0, 4.0],
        // Focus on default base_scale1 for now
        base_scale1_values: vec![14.75],
        dc_enabled_values: vec![false],
        aq_exponents: vec![1.0],
        // Many quality levels for Pareto curves
        quality_levels: vec![30, 40, 50, 60, 70, 75, 80, 85, 90, 95],
    }
}

#[cfg(feature = "experimental-hybrid-trellis")]
struct ImageData {
    name: String,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

#[cfg(feature = "experimental-hybrid-trellis")]
struct EncodingResult {
    bytes: usize,
    dssim: f64,
    ssim2: f64,
    butteraugli: f64,
}

#[cfg(feature = "experimental-hybrid-trellis")]
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

#[cfg(feature = "experimental-hybrid-trellis")]
fn encode_and_measure(
    img: &ImageData,
    quality: u8,
    config: Option<&jpegli::hybrid_config::HybridConfig>,
) -> EncodingResult {
    // Encode
    let mut encoder = jpegli::Encoder::new()
        .width(img.width as u32)
        .height(img.height as u32)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32));

    if let Some(cfg) = config {
        encoder = encoder.hybrid_config(*cfg);
    }

    let jpeg_data = encoder.encode(&img.pixels).expect("encode");

    // Decode
    let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
    let decoded = decoder.decode().expect("decode");

    // Compute all metrics
    let dssim = compute_dssim(&img.pixels, &decoded, img.width, img.height);
    let ssim2 = compute_ssim2(&img.pixels, &decoded, img.width, img.height);
    let butteraugli = compute_butteraugli_score(&img.pixels, &decoded, img.width, img.height);

    EncodingResult {
        bytes: jpeg_data.len(),
        dssim,
        ssim2,
        butteraugli,
    }
}

#[cfg(feature = "experimental-hybrid-trellis")]
fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;

    let attr = Dssim::new();

    let orig_rgba: Vec<rgb::RGBA<u8>> = original
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr
        .create_image_rgba(&decoded_rgba, width, height)
        .unwrap();

    let (dssim, _) = attr.compare(&orig_img, decoded_img);
    dssim.into()
}

#[cfg(feature = "experimental-hybrid-trellis")]
fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

    let orig_rgb = Rgb::new(
        original
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let decoded_rgb = Rgb::new(
        decoded
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_rgb, decoded_rgb).unwrap_or(0.0)
}

#[cfg(feature = "experimental-hybrid-trellis")]
fn compute_butteraugli_score(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use butteraugli::{compute_butteraugli, ButteraugliParams};

    let params = ButteraugliParams::default();
    match compute_butteraugli(original, decoded, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => 99.0,
    }
}
