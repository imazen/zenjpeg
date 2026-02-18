//! Minimal layout pipeline CLI tool.
//!
//! Usage:
//!   cargo run --release --features layout --example layout_cli -- [OPTIONS] INPUT OUTPUT
//!
//! Options:
//!   --quality Q        Output quality 0-100 (default: 85)
//!   --auto-orient      Apply EXIF orientation correction
//!   --rotate DEGREES   Rotate (90, 180, 270)
//!   --flip DIR         Flip (h or v)
//!   --fit WxH          Fit within dimensions (may upscale)
//!   --within WxH       Fit within dimensions (no upscale)
//!   --fit-crop WxH     Fill and crop to exact dimensions
//!   --no-progressive   Use baseline instead of progressive
//!   --no-optimize      Disable auto_optimize (hybrid trellis)
//!
//! Examples:
//!   # Auto-orient and fit within 800x600
//!   cargo run --release --features layout --example layout_cli -- \
//!     --auto-orient --fit 800x600 input.jpg output.jpg
//!
//!   # Rotate 90 degrees (lossless if MCU-aligned)
//!   cargo run --release --features layout --example layout_cli -- \
//!     --rotate 90 input.jpg output.jpg

use std::env;
use std::fs;
use std::process;

use enough::Unstoppable;
use zenjpeg::layout::LayoutConfig;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        usage(&args[0]);
        process::exit(1);
    }

    let mut quality: f32 = 85.0;
    let mut progressive = true;
    let mut auto_optimize = true;
    let mut auto_orient = false;
    let mut rotate: Option<u16> = None;
    let mut flip: Option<&str> = None;
    let mut fit: Option<(u32, u32)> = None;
    let mut within: Option<(u32, u32)> = None;
    let mut fit_crop: Option<(u32, u32)> = None;

    let mut positional = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--quality" | "-q" => {
                i += 1;
                quality = parse_or_exit(&args, i, "quality");
            }
            "--auto-orient" => {
                auto_orient = true;
            }
            "--rotate" => {
                i += 1;
                rotate = Some(parse_or_exit(&args, i, "rotate degrees"));
            }
            "--flip" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --flip requires h or v");
                    process::exit(1);
                }
                flip = Some(leak_string(&args[i]));
            }
            "--fit" => {
                i += 1;
                fit = Some(parse_dimensions(&args, i, "fit"));
            }
            "--within" => {
                i += 1;
                within = Some(parse_dimensions(&args, i, "within"));
            }
            "--fit-crop" => {
                i += 1;
                fit_crop = Some(parse_dimensions(&args, i, "fit-crop"));
            }
            "--no-progressive" => {
                progressive = false;
            }
            "--no-optimize" => {
                auto_optimize = false;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                process::exit(0);
            }
            s if s.starts_with('-') => {
                eprintln!("Unknown option: {s}");
                process::exit(1);
            }
            _ => {
                positional.push(args[i].clone());
            }
        }
        i += 1;
    }

    if positional.len() != 2 {
        eprintln!("Error: expected INPUT and OUTPUT paths");
        usage(&args[0]);
        process::exit(1);
    }

    let input_path = &positional[0];
    let output_path = &positional[1];

    // Read input
    let jpeg_data = match fs::read(input_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading {input_path}: {e}");
            process::exit(1);
        }
    };

    // Parse and show input info
    let info = match zenjpeg::decoder::DecodeConfig::new().read_info(&jpeg_data) {
        Ok(info) => info,
        Err(e) => {
            eprintln!("Error parsing JPEG: {e}");
            process::exit(1);
        }
    };

    eprintln!(
        "Input:  {}x{} {:?} {:?} {:?} ({} bytes)",
        info.dimensions.width,
        info.dimensions.height,
        info.mode,
        info.subsampling,
        info.color_space,
        jpeg_data.len(),
    );

    // Build config
    let config = LayoutConfig::new(quality)
        .with_progressive(progressive)
        .with_auto_optimize(auto_optimize);

    // Build request
    let mut request = config.request(&jpeg_data);

    if auto_orient {
        // 0 means "read from EXIF"
        request = request.auto_orient(0);
    }

    if let Some(deg) = rotate {
        request = match deg {
            90 => request.rotate_90(),
            180 => request.rotate_180(),
            270 => request.rotate_270(),
            _ => {
                eprintln!("Error: --rotate must be 90, 180, or 270");
                process::exit(1);
            }
        };
    }

    if let Some(dir) = flip {
        request = match dir {
            "h" | "horizontal" => request.flip_h(),
            "v" | "vertical" => request.flip_v(),
            _ => {
                eprintln!("Error: --flip must be h or v");
                process::exit(1);
            }
        };
    }

    if let Some((w, h)) = fit {
        request = request.fit(w, h);
    }
    if let Some((w, h)) = within {
        request = request.within(w, h);
    }
    if let Some((w, h)) = fit_crop {
        request = request.fit_crop(w, h);
    }

    // Execute
    let result = match request.execute(&Unstoppable) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    };

    // Write output
    if let Err(e) = fs::write(output_path, &result.data) {
        eprintln!("Error writing {output_path}: {e}");
        process::exit(1);
    }

    let path_label = if result.lossless { "lossless" } else { "lossy" };
    eprintln!(
        "Output: {}x{} {} ({} bytes, {})",
        result.width,
        result.height,
        path_label,
        result.data.len(),
        output_path,
    );
}

fn usage(program: &str) {
    eprintln!("Usage: {program} [OPTIONS] INPUT OUTPUT");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --quality Q        Output quality 0-100 (default: 85)");
    eprintln!("  --auto-orient      Apply EXIF orientation correction");
    eprintln!("  --rotate DEGREES   Rotate (90, 180, 270)");
    eprintln!("  --flip DIR         Flip (h or v)");
    eprintln!("  --fit WxH          Fit within dimensions (may upscale)");
    eprintln!("  --within WxH       Fit within dimensions (no upscale)");
    eprintln!("  --fit-crop WxH     Fill and crop to exact dimensions");
    eprintln!("  --no-progressive   Use baseline instead of progressive");
    eprintln!("  --no-optimize      Disable auto_optimize (hybrid trellis)");
}

fn parse_or_exit<T: std::str::FromStr>(args: &[String], idx: usize, name: &str) -> T {
    if idx >= args.len() {
        eprintln!("Error: --{name} requires a value");
        process::exit(1);
    }
    args[idx].parse().unwrap_or_else(|_| {
        eprintln!("Error: invalid {name}: {}", args[idx]);
        process::exit(1);
    })
}

fn parse_dimensions(args: &[String], idx: usize, name: &str) -> (u32, u32) {
    if idx >= args.len() {
        eprintln!("Error: --{name} requires WxH");
        process::exit(1);
    }
    let s = &args[idx];
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        eprintln!("Error: --{name} format is WxH (e.g., 800x600), got: {s}");
        process::exit(1);
    }
    let w: u32 = parts[0].parse().unwrap_or_else(|_| {
        eprintln!("Error: invalid width in --{name}: {}", parts[0]);
        process::exit(1);
    });
    let h: u32 = parts[1].parse().unwrap_or_else(|_| {
        eprintln!("Error: invalid height in --{name}: {}", parts[1]);
        process::exit(1);
    });
    (w, h)
}

/// Leak a string to get a `&'static str`. Used for the flip direction
/// which needs to outlive the args parsing loop.
fn leak_string(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}
