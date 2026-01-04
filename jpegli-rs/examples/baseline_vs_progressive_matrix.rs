//! Quick comparison: When is Progressive smaller than Baseline?
//!
//! Tests the key question: Under what conditions does Progressive produce
//! smaller files than Baseline?

use jpegli::{Encoder, PixelFormat};
use std::fs;
use std::process::Command;

fn create_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = ((x * 255) / width.max(1)) as u8;
            rgb[idx + 1] = ((y * 255) / height.max(1)) as u8;
            rgb[idx + 2] = 128;
        }
    }
    rgb
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

fn test_config(
    rgb: &[u8],
    width: usize,
    height: usize,
    use_xyb: bool,
    optimize: bool,
    quality: u8,
    label: &str,
) {
    // Rust baseline
    let rust_baseline = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(use_xyb)
        .optimize_huffman(optimize)
        .encode(rgb)
        .unwrap();

    // Rust progressive
    let rust_progressive = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(use_xyb)
        .optimize_huffman(optimize)
        .encode(rgb)
        .unwrap();

    let rust_base_size = rust_baseline.len();
    let rust_prog_size = rust_progressive.len();
    let rust_diff = rust_prog_size as i32 - rust_base_size as i32;
    let rust_pct = 100.0 * rust_diff as f64 / rust_base_size as f64;

    // C++ baseline
    let ppm_path = format!("/tmp/test_{}x{}.ppm", width, height);
    write_ppm(&ppm_path, rgb, width, height).ok();

    let cjpegli = match jpegli::test_utils::find_cjpegli() {
        Some(p) => p,
        None => {
            println!(
                "{:20} | {}x{} | Rust: Base={:7} Prog={:7} ({:+6.1}%) | C++: UNAVAILABLE",
                label, width, height, rust_base_size, rust_prog_size, rust_pct
            );
            return;
        }
    };

    let mut base_args = vec![
        ppm_path.clone(),
        "/tmp/cpp_base.jpg".to_string(),
        "-p".to_string(),
        "0".to_string(),
        "-q".to_string(),
        quality.to_string(),
    ];
    if use_xyb {
        base_args.push("--xyb".to_string());
    }
    if !optimize {
        base_args.push("--fixed_code".to_string());
    }

    Command::new(&cjpegli).args(&base_args).output().ok();

    let mut prog_args = vec![
        ppm_path.clone(),
        "/tmp/cpp_prog.jpg".to_string(),
        "-p".to_string(),
        "2".to_string(),
        "-q".to_string(),
        quality.to_string(),
    ];
    if use_xyb {
        prog_args.push("--xyb".to_string());
    }
    if !optimize {
        prog_args.push("--fixed_code".to_string());
    }

    Command::new(&cjpegli).args(&prog_args).output().ok();

    let cpp_base_size = fs::metadata("/tmp/cpp_base.jpg").map(|m| m.len() as usize).ok();
    let cpp_prog_size = fs::metadata("/tmp/cpp_prog.jpg").map(|m| m.len() as usize).ok();

    if let (Some(cpp_base), Some(cpp_prog)) = (cpp_base_size, cpp_prog_size) {
        let cpp_diff = cpp_prog as i32 - cpp_base as i32;
        let cpp_pct = 100.0 * cpp_diff as f64 / cpp_base as f64;

        let rust_winner = if rust_diff < 0 {
            "✓ Prog"
        } else {
            "✗ Base"
        };
        let cpp_winner = if cpp_diff < 0 { "✓ Prog" } else { "✗ Base" };

        println!(
            "{:20} | {}x{:4} | Rust: B={:7} P={:7} {:+6.1}% {} | C++: B={:7} P={:7} {:+6.1}% {}",
            label,
            width,
            height,
            rust_base_size,
            rust_prog_size,
            rust_pct,
            rust_winner,
            cpp_base,
            cpp_prog,
            cpp_pct,
            cpp_winner
        );
    } else {
        println!(
            "{:20} | {}x{} | Rust: Base={:7} Prog={:7} ({:+6.1}%) | C++: FAILED",
            label, width, height, rust_base_size, rust_prog_size, rust_pct
        );
    }
}

fn main() {
    println!("\n{}", "=".repeat(120));
    println!(" BASELINE vs PROGRESSIVE: When is Progressive Smaller?");
    println!("{}\n", "=".repeat(120));

    println!(
        "{:20} | {:9} | {:80}",
        "Config", "Size", "Results (B=Baseline, P=Progressive)"
    );
    println!("{}", "-".repeat(120));

    let sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048)];

    for (width, height) in &sizes {
        let rgb = create_gradient(*width, *height);

        println!("\n--- {}x{} Gradient ---", width, height);
        test_config(&rgb, *width, *height, false, false, 90, "YCbCr/Fixed");
        test_config(&rgb, *width, *height, false, true, 90, "YCbCr/Optimized");
        test_config(&rgb, *width, *height, true, true, 90, "XYB/Optimized");
    }

    println!("\n{}", "=".repeat(120));
    println!("INTERPRETATION:");
    println!("  ✓ Prog = Progressive is SMALLER (good!)");
    println!("  ✗ Base = Baseline is SMALLER (progressive overhead not worth it)");
    println!("\nKEY QUESTION: At what image size does progressive become smaller?");
    println!("{}\n", "=".repeat(120));
}
