use jpegli::{Encoder, PixelFormat};

fn main() {
    // Simple 64x64 gradient
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = 128;
        }
    }

    println!("=== Baseline vs Progressive File Size Comparison ===\n");

    // Test YCbCr mode
    println!("YCbCr Mode (Q90, 4:4:4, Huffman optimized):");

    let ycbcr_baseline = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(false)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    let ycbcr_progressive = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(false)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    println!("  Baseline:    {} bytes", ycbcr_baseline.len());
    println!("  Progressive: {} bytes", ycbcr_progressive.len());
    let diff = ycbcr_progressive.len() as i32 - ycbcr_baseline.len() as i32;
    let pct = 100.0 * diff as f64 / ycbcr_baseline.len() as f64;
    println!("  Difference:  {:+} bytes ({:+.1}%)", diff, pct);
    if diff < 0 {
        println!("  ✓ Progressive is SMALLER");
    } else {
        println!("  ✗ Progressive is LARGER");
    }

    println!();

    // Test XYB mode
    println!("XYB Mode (Q90, Huffman optimized):");

    let xyb_baseline = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    let xyb_progressive = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    println!("  Baseline:    {} bytes", xyb_baseline.len());
    println!("  Progressive: {} bytes", xyb_progressive.len());
    let diff = xyb_progressive.len() as i32 - xyb_baseline.len() as i32;
    let pct = 100.0 * diff as f64 / xyb_baseline.len() as f64;
    println!("  Difference:  {:+} bytes ({:+.1}%)", diff, pct);
    if diff < 0 {
        println!("  ✓ Progressive is SMALLER");
    } else {
        println!("  ✗ Progressive is LARGER");
    }

    println!();

    // Also test C++ cjpegli defaults
    println!("Testing C++ cjpegli defaults...");

    // Write test input
    std::fs::write("/tmp/test_baseline_prog.ppm", format!(
        "P6\n{} {}\n255\n",
        width, height
    ).as_bytes()).unwrap();
    std::fs::write("/tmp/test_baseline_prog.ppm",
        format!("P6\n{} {}\n255\n", width, height).into_bytes()
        .into_iter()
        .chain(data.iter().copied())
        .collect::<Vec<u8>>()
    ).unwrap();

    // Test C++ baseline (progressive level 0)
    std::process::Command::new("internal/jpegli-cpp/build/tools/cjpegli")
        .args(&[
            "/tmp/test_baseline_prog.ppm",
            "/tmp/cpp_baseline.jpg",
            "-p", "0",  // Baseline
            "-q", "90",
        ])
        .output()
        .ok();

    // Test C++ progressive (progressive level 2 - the default)
    std::process::Command::new("internal/jpegli-cpp/build/tools/cjpegli")
        .args(&[
            "/tmp/test_baseline_prog.ppm",
            "/tmp/cpp_progressive.jpg",
            "-p", "2",  // Progressive level 2 (default)
            "-q", "90",
        ])
        .output()
        .ok();

    if let (Ok(baseline_meta), Ok(prog_meta)) = (
        std::fs::metadata("/tmp/cpp_baseline.jpg"),
        std::fs::metadata("/tmp/cpp_progressive.jpg")
    ) {
        let baseline_size = baseline_meta.len();
        let prog_size = prog_meta.len();
        println!("C++ cjpegli (YCbCr, Q90):");
        println!("  Baseline (p=0):    {} bytes", baseline_size);
        println!("  Progressive (p=2): {} bytes", prog_size);
        let diff = prog_size as i64 - baseline_size as i64;
        let pct = 100.0 * diff as f64 / baseline_size as f64;
        println!("  Difference:        {:+} bytes ({:+.1}%)", diff, pct);
        if diff < 0 {
            println!("  ✓ C++ progressive is SMALLER");
        } else {
            println!("  ✗ C++ progressive is LARGER");
        }
    }
}
