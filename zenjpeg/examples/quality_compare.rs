//! Unified quality comparison tool for JPEG encoders.
//!
//! Compares encoders across quality levels with multiple metrics.
//! Supports single-image analysis, Pareto curves, and same-size comparisons.
//!
//! Usage:
//!   cargo run --release --example quality_compare -- [OPTIONS] <image.png>
//!
//! Options:
//!   --encoder <name>    Encoder to test. Can specify multiple times. Default: all
//!                       Available: zenjpeg-ycbcr, zenjpeg-ycbcr-hybrid, zenjpeg-xyb,
//!                                  zenjpeg-ycbcr-444, zenjpeg-ycbcr-hybrid-444
//!   --quality <n>       Single quality level (0-100). Default: sweep 10-95
//!   --metric <name>     Metric to use (dssim, ssim2, butteraugli, all). Default: all
//!   --output <file>     Output CSV file
//!   --pareto            Generate Pareto curve data
//!   --same-size         Compare at same file size (find matching quality)
//!   --quick             Quick mode: fewer quality levels (10, 30, 50, 70, 90)
//!
//! Examples:
//!   # Quick YCbCr vs hybrid comparison
//!   cargo run --release --example quality_compare --features experimental-hybrid-trellis -- \
//!     --encoder zenjpeg-ycbcr --encoder hybrid --quick image.png
//!
//!   # Full Pareto curve to CSV
//!   cargo run --release --example quality_compare -- --pareto --output results.csv image.png
//!
//!   # Single quality point
//!   cargo run --release --example quality_compare -- --quality 75 image.png

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use zenjpeg_bench_utils::{
    decode_jpeg_to_rgb, decode_jpeg_with_icc, ChromaSubsampling, ColorMode, EncoderConfig,
    EncoderImpl, ImageData, QualityMetrics,
};

#[derive(Debug, Clone)]
#[allow(dead_code)] // pareto/same_size are parsed but not yet implemented
struct Config {
    encoders: Vec<EncoderConfig>,
    qualities: Vec<u8>,
    metrics: Vec<Metric>,
    output_csv: Option<String>,
    pareto: bool,
    same_size: bool,
    image_path: String,
}

#[derive(Debug, Clone, Copy)]
enum Metric {
    Dssim,
    Ssim2,
    Butteraugli,
}

impl Metric {
    fn name(&self) -> &'static str {
        match self {
            Metric::Dssim => "DSSIM",
            Metric::Ssim2 => "SSIM2",
            Metric::Butteraugli => "Butteraugli",
        }
    }
}

#[derive(Debug)]
struct Result {
    encoder: String,
    quality: u8,
    bytes: usize,
    bpp: f64,
    dssim: Option<f64>,
    ssim2: Option<f64>,
    butteraugli: Option<f64>,
    encode_ms: f64,
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();

    let mut encoders = Vec::new();
    let mut qualities = Vec::new();
    let mut metrics = Vec::new();
    let mut output_csv = None;
    let mut pareto = false;
    let mut same_size = false;
    let mut quick = false;
    let mut image_path = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--encoder" => {
                i += 1;
                if i < args.len() {
                    encoders.push(parse_encoder(&args[i]));
                }
            }
            "--quality" => {
                i += 1;
                if i < args.len() {
                    qualities.push(args[i].parse().expect("invalid quality"));
                }
            }
            "--metric" => {
                i += 1;
                if i < args.len() {
                    match args[i].as_str() {
                        "dssim" => metrics.push(Metric::Dssim),
                        "ssim2" => metrics.push(Metric::Ssim2),
                        "butteraugli" => metrics.push(Metric::Butteraugli),
                        "all" => {
                            metrics.extend([Metric::Dssim, Metric::Ssim2, Metric::Butteraugli])
                        }
                        _ => eprintln!("Unknown metric: {}", args[i]),
                    }
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_csv = Some(args[i].clone());
                }
            }
            "--pareto" => pareto = true,
            "--same-size" => same_size = true,
            "--quick" => quick = true,
            arg if !arg.starts_with('-') => {
                image_path = Some(arg.to_string());
            }
            _ => eprintln!("Unknown argument: {}", args[i]),
        }
        i += 1;
    }

    // Defaults
    if encoders.is_empty() {
        encoders.push(EncoderConfig::new(EncoderImpl::JpegliRs).color(ColorMode::YCbCr));
        encoders.push(EncoderConfig::new(EncoderImpl::JpegliRs).color(ColorMode::Xyb));
        encoders.push(EncoderConfig::new(EncoderImpl::CJpegli).color(ColorMode::YCbCr));
    }

    if qualities.is_empty() {
        if quick {
            qualities = vec![10, 30, 50, 70, 90];
        } else if pareto {
            qualities = (10..=95).step_by(5).collect();
        } else {
            qualities = vec![75];
        }
    }

    if metrics.is_empty() {
        metrics = vec![Metric::Dssim, Metric::Ssim2, Metric::Butteraugli];
    }

    let image_path = image_path.unwrap_or_else(|| {
        eprintln!("Usage: quality_compare [OPTIONS] <image.png>");
        eprintln!("  --encoder <name>   zenjpeg-ycbcr, zenjpeg-xyb, cjpegli");
        eprintln!("  --quality <n>      Quality level 0-100");
        eprintln!("  --metric <name>    dssim, ssim2, butteraugli, all");
        eprintln!("  --output <file>    CSV output file");
        eprintln!("  --pareto           Sweep quality levels for Pareto curve");
        eprintln!("  --quick            Quick mode (5 quality levels)");
        std::process::exit(1);
    });

    Config {
        encoders,
        qualities,
        metrics,
        output_csv,
        pareto,
        same_size,
        image_path,
    }
}

fn parse_encoder(name: &str) -> EncoderConfig {
    match name {
        "zenjpeg-ycbcr" | "jpegli-ycbcr" | "jpegli" => {
            EncoderConfig::new(EncoderImpl::JpegliRs).color(ColorMode::YCbCr)
        }
        "zenjpeg-ycbcr-hybrid" | "jpegli-ycbcr-hybrid" | "jpegli-hybrid" | "hybrid" => {
            EncoderConfig::new(EncoderImpl::JpegliRs)
                .color(ColorMode::YCbCr)
                .hybrid(true)
        }
        "zenjpeg-xyb" | "jpegli-xyb" | "xyb" => {
            EncoderConfig::new(EncoderImpl::JpegliRs).color(ColorMode::Xyb)
        }
        "cjpegli" => EncoderConfig::new(EncoderImpl::CJpegli).color(ColorMode::YCbCr),
        "zenjpeg-ycbcr-444" => EncoderConfig::new(EncoderImpl::JpegliRs)
            .color(ColorMode::YCbCr)
            .subsampling(ChromaSubsampling::S444),
        "zenjpeg-ycbcr-hybrid-444" => EncoderConfig::new(EncoderImpl::JpegliRs)
            .color(ColorMode::YCbCr)
            .subsampling(ChromaSubsampling::S444)
            .hybrid(true),
        "zenjpeg-xyb-444" => EncoderConfig::new(EncoderImpl::JpegliRs)
            .color(ColorMode::Xyb)
            .subsampling(ChromaSubsampling::S444),
        _ => {
            eprintln!("Unknown encoder: {}. Using zenjpeg-ycbcr", name);
            EncoderConfig::new(EncoderImpl::JpegliRs).color(ColorMode::YCbCr)
        }
    }
}

fn run_comparison(config: &Config) -> Vec<Result> {
    let img = ImageData::from_path(Path::new(&config.image_path)).expect("Failed to load image");

    let orig_rgb = img.as_rgb_image();
    let pixels = img.pixel_count();

    println!(
        "Image: {} ({}x{}, {:.2} MP)",
        config.image_path,
        img.width,
        img.height,
        pixels as f64 / 1_000_000.0
    );
    println!();

    // Print header
    print!("{:<25} {:>5} {:>10} {:>8}", "Encoder", "Q", "Size", "BPP");
    for metric in &config.metrics {
        print!(" {:>12}", metric.name());
    }
    println!(" {:>8}", "Time");
    println!("{}", "-".repeat(80));

    let mut results = Vec::new();

    for encoder in &config.encoders {
        for &quality in &config.qualities {
            let mut enc = encoder.clone();
            enc.quality = quality;

            let start = Instant::now();
            let jpeg_data: Vec<u8> = match enc.encode(&img) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("  {} q{}: {}", enc.short_name(), quality, e);
                    continue;
                }
            };
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

            let bpp = jpeg_data.len() as f64 * 8.0 / pixels as f64;

            // Decode and compute metrics
            // XYB JPEGs require ICC-aware decoding to get correct colors
            let decoded = if enc.color == ColorMode::Xyb {
                decode_jpeg_with_icc(&jpeg_data).expect("Failed to decode XYB JPEG")
            } else {
                decode_jpeg_to_rgb(&jpeg_data).expect("Failed to decode JPEG")
            };

            let mut dssim = None;
            let mut ssim2 = None;
            let mut butteraugli = None;

            for metric in &config.metrics {
                match metric {
                    Metric::Dssim => {
                        dssim = Some(QualityMetrics::dssim(orig_rgb.as_ref(), decoded.as_ref()));
                    }
                    Metric::Ssim2 => {
                        ssim2 = Some(QualityMetrics::ssimulacra2(
                            orig_rgb.as_ref(),
                            decoded.as_ref(),
                        ));
                    }
                    Metric::Butteraugli => {
                        butteraugli = Some(QualityMetrics::butteraugli(
                            orig_rgb.as_ref(),
                            decoded.as_ref(),
                        ));
                    }
                }
            }

            // Print row
            print!(
                "{:<25} {:>5} {:>10} {:>8.3}",
                enc.short_name(),
                quality,
                format_size(jpeg_data.len()),
                bpp
            );

            for metric in &config.metrics {
                let value = match metric {
                    Metric::Dssim => dssim.map(|v| format!("{:.6}", v)),
                    Metric::Ssim2 => ssim2.map(|v| format!("{:.2}", v)),
                    Metric::Butteraugli => butteraugli.map(|v| format!("{:.4}", v)),
                };
                print!(" {:>12}", value.unwrap_or_default());
            }
            println!(" {:>7.1}ms", encode_ms);

            results.push(Result {
                encoder: enc.short_name(),
                quality,
                bytes: jpeg_data.len(),
                bpp,
                dssim,
                ssim2,
                butteraugli,
                encode_ms,
            });
        }
    }

    results
}

fn write_csv(results: &[Result], path: &str, config: &Config) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    // Header
    write!(f, "encoder,quality,bytes,bpp")?;
    for metric in &config.metrics {
        write!(f, ",{}", metric.name().to_lowercase())?;
    }
    writeln!(f, ",encode_ms")?;

    // Data
    for r in results {
        write!(f, "{},{},{},{:.4}", r.encoder, r.quality, r.bytes, r.bpp)?;
        for metric in &config.metrics {
            let value = match metric {
                Metric::Dssim => r.dssim,
                Metric::Ssim2 => r.ssim2,
                Metric::Butteraugli => r.butteraugli,
            };
            if let Some(v) = value {
                write!(f, ",{:.6}", v)?;
            } else {
                write!(f, ",")?;
            }
        }
        writeln!(f, ",{:.2}", r.encode_ms)?;
    }

    Ok(())
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn main() {
    let config = parse_args();
    let results = run_comparison(&config);

    if let Some(csv_path) = &config.output_csv {
        if let Err(e) = write_csv(&results, csv_path, &config) {
            eprintln!("Failed to write CSV: {}", e);
        } else {
            println!("\nResults written to {}", csv_path);
        }
    }

    // Summary
    if results.len() > 1 {
        println!("\n--- Summary ---");

        // Group by encoder and find best quality at each size
        let encoders: Vec<String> = config
            .encoders
            .iter()
            .map(|e: &EncoderConfig| e.short_name())
            .collect();

        for encoder in &encoders {
            let encoder_results: Vec<_> =
                results.iter().filter(|r| &r.encoder == encoder).collect();

            if encoder_results.is_empty() {
                continue;
            }

            let avg_bpp: f64 =
                encoder_results.iter().map(|r| r.bpp).sum::<f64>() / encoder_results.len() as f64;

            let avg_ssim2 = if config.metrics.iter().any(|m| matches!(m, Metric::Ssim2)) {
                let sum: f64 = encoder_results.iter().filter_map(|r| r.ssim2).sum();
                let count = encoder_results.iter().filter(|r| r.ssim2.is_some()).count();
                if count > 0 {
                    Some(sum / count as f64)
                } else {
                    None
                }
            } else {
                None
            };

            print!("{}: avg bpp={:.3}", encoder, avg_bpp);
            if let Some(s) = avg_ssim2 {
                print!(", avg SSIM2={:.2}", s);
            }
            println!();
        }
    }
}
