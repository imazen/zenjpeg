//! Compare DCT coefficients between C++ and Rust encoded JPEGs

use std::fs;
use std::process::Command;

fn main() {
    // Create a small test pattern (16x16 gradient)
    let width = 16usize;
    let height = 16usize;
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = (x * 16) as u8;     // R gradient
            rgb[idx + 1] = (y * 16) as u8; // G gradient
            rgb[idx + 2] = 128;            // B constant
        }
    }

    // Write PPM for C++
    let ppm_path = "/tmp/gradient16.ppm";
    {
        use std::io::Write;
        let mut f = fs::File::create(ppm_path).unwrap();
        writeln!(f, "P6").unwrap();
        writeln!(f, "{} {}", width, height).unwrap();
        writeln!(f, "255").unwrap();
        f.write_all(&rgb).unwrap();
    }

    // Encode with C++
    let cpp_path = "/tmp/cpp_gradient16.jpg";
    Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p", "0",
            "--fixed_code",
            ppm_path,
            cpp_path,
            "-q", "90",
        ])
        .output()
        .expect("cjpegli failed");

    // Encode with Rust
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&rgb)
        .expect("Rust encoding failed");

    let cpp_jpeg = fs::read(cpp_path).unwrap();
    fs::write("/tmp/rust_gradient16.jpg", &rust_jpeg).unwrap();

    println!("=== 16x16 Gradient Test ===");
    println!("C++ size: {} bytes", cpp_jpeg.len());
    println!("Rust size: {} bytes", rust_jpeg.len());

    // Decode both and compare coefficients using djpeg with -dct int
    println!("\n=== Decoding with djpeg to compare ===");

    // Use jpegtran to dump coefficients (if available)
    let cpp_coef = Command::new("djpeg")
        .args(["-ppm", "-dct", "int", cpp_path])
        .output();

    let rust_coef = Command::new("djpeg")
        .args(["-ppm", "-dct", "int", "/tmp/rust_gradient16.jpg"])
        .output();

    match (cpp_coef, rust_coef) {
        (Ok(cpp), Ok(rust)) if cpp.status.success() && rust.status.success() => {
            println!("C++ decoded size: {} bytes", cpp.stdout.len());
            println!("Rust decoded size: {} bytes", rust.stdout.len());

            // Compare decoded pixels
            let mut diff_count = 0;
            let mut max_diff = 0i32;
            for i in 0..cpp.stdout.len().min(rust.stdout.len()) {
                let d = (cpp.stdout[i] as i32 - rust.stdout[i] as i32).abs();
                if d > 0 {
                    diff_count += 1;
                    max_diff = max_diff.max(d);
                }
            }
            println!("Pixel differences: {} (max diff: {})", diff_count, max_diff);

            if diff_count == 0 {
                println!("✓ Decoded images are identical!");
                println!("  (file size diff must be from entropy coding efficiency)");
            } else {
                println!("✗ Decoded images differ - DCT/quantization mismatch");
            }
        }
        _ => {
            println!("djpeg not available, using internal decoder");
        }
    }

    // Analyze the coefficient distribution in scan data
    println!("\n=== Entropy Analysis ===");
    analyze_scan(&cpp_jpeg, "C++");
    analyze_scan(&rust_jpeg, "Rust");
}

fn analyze_scan(jpeg: &[u8], name: &str) {
    // Find SOS marker and scan data
    let mut i = 0;
    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
            // Found SOS
            let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let scan_start = i + 2 + len;
            let mut scan_end = scan_start;

            // Find end of scan (next marker)
            while scan_end < jpeg.len() - 1 {
                if jpeg[scan_end] == 0xFF && jpeg[scan_end + 1] != 0x00 && jpeg[scan_end + 1] != 0xFF {
                    break;
                }
                scan_end += 1;
            }

            let scan_data = &jpeg[scan_start..scan_end];

            // Count stuffed bytes (0xFF 0x00)
            let mut stuffed = 0;
            for j in 0..scan_data.len().saturating_sub(1) {
                if scan_data[j] == 0xFF && scan_data[j + 1] == 0x00 {
                    stuffed += 1;
                }
            }

            // Count high-value bytes (indicator of less efficient coding)
            let high_bytes = scan_data.iter().filter(|&&b| b >= 0xF0).count();

            println!("{}: scan={} bytes, stuffed={}, high_bytes={}",
                     name, scan_data.len(), stuffed, high_bytes);
            return;
        }
        i += 1;
    }
}
