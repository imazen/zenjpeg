//! czenjpeg-profile: CLI profiling tool matching cjpegli defaults.
//!
//! Loads image once, compresses N times for accurate timing.
//! Matches cjpegli defaults: [YUV d1.000 AQ p2 OPT]
//!
//! Usage: cargo run --release --example cjpegli_rs_profile -- INPUT [OUTPUT] [OPTIONS]
//!
//! Options:
//!   -d, --distance N     Butteraugli distance (default: 1.0, visually lossless)
//!   -q, --quality N      Quality 1-100 (alternative to distance)
//!   -p, --progressive N  Progressive level 0-2 (0=sequential, 2=default)
//!   --chroma_subsampling 444|440|422|420  Chroma subsampling (default: 444)
//!   --num_reps N         Number of iterations (default: 50)
//!   --disable_output     Don't write output file (for benchmarking)
//!   --quiet              Suppress informative output
//!
//! Examples:
//!   cargo run --release --example cjpegli_rs_profile -- image.png
//!   cargo run --release --example cjpegli_rs_profile -- image.png -d 1.0 -p 2 --num_reps 500
//!   cargo run --release --example cjpegli_rs_profile -- image.png out.jpg -q 90
//!   cargo run --release --example cjpegli_rs_profile -- image.png --disable_output --num_reps 500

use enough::Unstoppable;
use std::time::Instant;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

struct Args {
    input: String,
    output: Option<String>,
    distance: Option<f32>,
    quality: Option<u8>,
    progressive_level: u8,
    chroma_subsampling: ChromaSubsampling,
    num_reps: usize,
    disable_output: bool,
    quiet: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut input = None;
    let mut output = None;
    let mut distance = None;
    let mut quality = None;
    let mut progressive_level = 2u8;
    let mut chroma_subsampling = ChromaSubsampling::None; // 444 = cjpegli default
    let mut num_reps = 50;
    let mut disable_output = false;
    let mut quiet = false;
    let mut positional_count = 0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--distance" => {
                i += 1;
                if i < args.len() {
                    distance = Some(args[i].parse().expect("Invalid --distance value"));
                }
            }
            "-q" | "--quality" => {
                i += 1;
                if i < args.len() {
                    quality = Some(args[i].parse().expect("Invalid --quality value"));
                }
            }
            "-p" | "--progressive_level" => {
                i += 1;
                if i < args.len() {
                    progressive_level = args[i].parse().expect("Invalid --progressive_level value");
                }
            }
            "--chroma_subsampling" => {
                i += 1;
                if i < args.len() {
                    chroma_subsampling = match args[i].as_str() {
                        "444" => ChromaSubsampling::None,
                        "440" => ChromaSubsampling::HalfVertical,
                        "422" => ChromaSubsampling::HalfHorizontal,
                        "420" => ChromaSubsampling::Quarter,
                        _ => panic!("Invalid --chroma_subsampling: use 444, 440, 422, or 420"),
                    };
                }
            }
            "--num_reps" => {
                i += 1;
                if i < args.len() {
                    num_reps = args[i].parse().expect("Invalid --num_reps value");
                }
            }
            "--disable_output" => {
                disable_output = true;
            }
            "--quiet" => {
                quiet = true;
            }
            "-h" | "--help" => {
                eprintln!("Usage: cjpegli_rs_profile INPUT [OUTPUT] [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -d, --distance N     Butteraugli distance (default: 1.0)");
                eprintln!("  -q, --quality N      Quality 1-100 (alternative to distance)");
                eprintln!("  -p, --progressive N  Progressive level 0-2 (default: 2)");
                eprintln!("  --chroma_subsampling 444|440|422|420 (default: 444)");
                eprintln!("  --num_reps N         Number of iterations (default: 50)");
                eprintln!("  --disable_output     Don't write output file");
                eprintln!("  --quiet              Suppress informative output");
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') => {
                if positional_count == 0 {
                    input = Some(arg.to_string());
                } else if positional_count == 1 {
                    output = Some(arg.to_string());
                }
                positional_count += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let input = input.unwrap_or_else(|| {
        eprintln!("Error: INPUT file required");
        eprintln!("Usage: cjpegli_rs_profile INPUT [OUTPUT] [OPTIONS]");
        eprintln!("Try --help for more options");
        std::process::exit(1);
    });

    Args {
        input,
        output,
        distance,
        quality,
        progressive_level,
        chroma_subsampling,
        num_reps,
        disable_output,
        quiet,
    }
}

fn main() {
    let args = parse_args();

    if !args.quiet {
        eprintln!("Loading image: {}", args.input);
    }
    let load_start = Instant::now();

    // Load and decode image once
    let loaded = zenjpeg_bench_utils::load_png(std::path::Path::new(&args.input))
        .expect("Failed to load PNG");
    let width = loaded.width();
    let height = loaded.height();
    let color_type_str = "RGB";
    let pixels: Vec<u8> = loaded.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let layout = PixelLayout::Rgb8Srgb;

    let load_time = load_start.elapsed();
    let megapixels = (width * height) as f64 / 1_000_000.0;

    if !args.quiet {
        eprintln!(
            "Loaded {}x{} ({:.2} MP) {} in {:.1}ms",
            width,
            height,
            megapixels,
            color_type_str,
            load_time.as_secs_f64() * 1000.0
        );
    }

    // Build quality setting
    let quality: Quality = if let Some(d) = args.distance {
        Quality::ApproxButteraugli(d)
    } else if let Some(q) = args.quality {
        Quality::ApproxJpegli(q as f32)
    } else {
        Quality::ApproxButteraugli(1.0) // default: d1.0
    };

    // Build config matching cjpegli style
    let config = EncoderConfig::ycbcr(quality, args.chroma_subsampling)
        .progressive(args.progressive_level > 0);

    // Print encoding params like cjpegli: [YUV d1.000 AQ p2 OPT]
    if !args.quiet {
        // Only show subsampling if not default (444)
        let subsampling_str = match args.chroma_subsampling {
            ChromaSubsampling::None => "",
            ChromaSubsampling::HalfVertical => "440",
            ChromaSubsampling::HalfHorizontal => "422",
            ChromaSubsampling::Quarter => "420",
            _ => "???",
        };
        let dist_str = if let Some(d) = args.distance {
            format!("d{:.3}", d)
        } else if let Some(q) = args.quality {
            format!("q{}", q)
        } else {
            "d1.000".to_string()
        };
        eprintln!(
            "Encoding [YUV{} {} AQ p{} OPT]",
            subsampling_str, dist_str, args.progressive_level
        );
    }

    let iterations = args.num_reps;

    // Warmup (3 iterations)
    if !args.quiet {
        eprintln!("Warming up...");
    }
    for _ in 0..3 {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, layout)
            .unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        let _ = std::hint::black_box(enc.finish().unwrap());
    }

    // Timed runs
    if !args.quiet {
        eprintln!("Running {} iterations...", iterations);
    }
    let start = Instant::now();
    let mut jpeg_bytes = Vec::new();

    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, layout)
            .unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        jpeg_bytes = enc.finish().unwrap();
        std::hint::black_box(&jpeg_bytes);
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;
    let mp_per_sec = megapixels / avg_ms * 1000.0;
    let bpp = jpeg_bytes.len() as f64 * 8.0 / (width * height) as f64;

    // Write output file if specified
    if let Some(ref output_path) = args.output
        && !args.disable_output {
            std::fs::write(output_path, &jpeg_bytes).expect("Failed to write output file");
            if !args.quiet {
                eprintln!("Wrote {} bytes to {}", jpeg_bytes.len(), output_path);
            }
        }

    if !args.quiet {
        eprintln!();
    }
    println!("Results:");
    println!("  Image:       {}x{} ({:.2} MP)", width, height, megapixels);
    println!("  Iterations:  {}", iterations);
    println!("  Total time:  {:.1}ms", total_ms);
    println!("  Average:     {:.2}ms per encode", avg_ms);
    println!("  Throughput:  {:.1} MP/s", mp_per_sec);
    println!("  Output size: {} bytes ({:.2} bpp)", jpeg_bytes.len(), bpp);
}
