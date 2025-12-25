//! Unified JPEG debugging tool for jpegli development.
//!
//! Replaces 27+ individual debug scripts with a single CLI tool.
//!
//! # Usage
//!
//! ```bash
//! # Trace encoding pipeline for an image
//! cargo run --release --example jpegli_debug -- trace image.png
//!
//! # Dump coefficients/tables for a JPEG
//! cargo run --release --example jpegli_debug -- dump image.jpg
//!
//! # Compare Rust vs C++ encoding
//! cargo run --release --example jpegli_debug -- compare image.png
//!
//! # Analyze quality metrics
//! cargo run --release --example jpegli_debug -- analyze original.png encoded.jpg
//! ```

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "trace" => cmd_trace(&args[2..]),
        "dump" => cmd_dump(&args[2..]),
        "compare" => cmd_compare(&args[2..]),
        "analyze" => cmd_analyze(&args[2..]),
        "block" => cmd_block(&args[2..]),
        "quant" => cmd_quant(&args[2..]),
        "help" | "--help" | "-h" => print_help(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_help();
        }
    }
}

fn print_help() {
    println!(
        r#"jpegli-debug: Unified JPEG debugging tool

USAGE:
    cargo run --example jpegli_debug -- <COMMAND> [OPTIONS]

COMMANDS:
    trace <image>           Trace encoding pipeline step by step
    dump <jpeg>             Dump JPEG structure (coefficients, tables, markers)
    compare <image> [Q]     Compare Rust vs C++ encoding (default Q=90)
    analyze <orig> <jpeg>   Analyze quality metrics (DSSIM, file size)
    block [pattern]         Encode/analyze a single 8x8 block
    quant [Q]               Show quantization tables at quality level

OPTIONS:
    -v, --verbose           Show more details
    -q, --quality <N>       Quality level (1-100, default 90)
    --xyb                   Use XYB color space instead of YCbCr

EXAMPLES:
    # Trace full encoding pipeline
    cargo run --example jpegli_debug -- trace photo.png

    # Compare Rust vs C++ at Q100
    cargo run --example jpegli_debug -- compare photo.png 100

    # Dump coefficient statistics
    cargo run --example jpegli_debug -- dump output.jpg

    # Analyze quality
    cargo run --example jpegli_debug -- analyze original.png compressed.jpg
"#
    );
}

// ============================================================================
// TRACE: Step through encoding pipeline
// ============================================================================

fn cmd_trace(args: &[String]) {
    if args.is_empty() {
        println!("Usage: trace <image.png>");
        println!("\nTraces the encoding pipeline step by step:");
        println!("  1. RGB input");
        println!("  2. RGB → YCbCr conversion");
        println!("  3. Level shift (−128)");
        println!("  4. DCT transform");
        println!("  5. Quantization");
        println!("  6. Entropy encoding");
        return;
    }

    let path = &args[0];
    let quality: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(90.0);

    println!("=== JPEGLI ENCODING TRACE ===");
    println!("Input: {}", path);
    println!("Quality: {}", quality);
    println!();

    // Load image
    let (rgb, width, height) = match load_png(path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);
    println!("Total pixels: {}", width * height);
    println!("Total blocks: {}x{} = {}",
        (width + 7) / 8, (height + 7) / 8,
        ((width + 7) / 8) * ((height + 7) / 8));
    println!();

    // Trace first block
    trace_block(&rgb, width, quality, 0, 0);

    // Also trace a mid-image block if image is large enough
    if width >= 64 && height >= 64 {
        println!("\n--- Block at (4, 4) ---");
        trace_block(&rgb, width, quality, 4, 4);
    }
}

fn trace_block(rgb: &[u8], width: usize, quality: f32, bx: usize, by: usize) {
    use jpegli::color::rgb_to_ycbcr_f32;
    use jpegli::dct::forward_dct_8x8;
    use jpegli::quant::{generate_quant_table, Quality};
    use jpegli::types::ColorSpace;

    println!("=== Block ({}, {}) ===", bx, by);

    // Extract 8x8 block
    let mut block_rgb = [[0u8; 3]; 64];
    for dy in 0..8 {
        for dx in 0..8 {
            let x = bx * 8 + dx;
            let y = by * 8 + dy;
            if x < width {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb.len() {
                    block_rgb[dy * 8 + dx] = [rgb[idx], rgb[idx + 1], rgb[idx + 2]];
                }
            }
        }
    }

    // Step 1: Show RGB
    println!("\n1. RGB (first row):");
    for i in 0..8 {
        let [r, g, b] = block_rgb[i];
        print!("  ({:3},{:3},{:3})", r, g, b);
    }
    println!();

    // Step 2: RGB → YCbCr
    let mut y_block = [0.0f32; 64];
    let mut cb_block = [0.0f32; 64];
    let mut cr_block = [0.0f32; 64];

    for i in 0..64 {
        let [r, g, b] = block_rgb[i];
        let (y, cb, cr) = rgb_to_ycbcr_f32(r as f32, g as f32, b as f32);
        y_block[i] = y;
        cb_block[i] = cb;
        cr_block[i] = cr;
    }

    println!("\n2. YCbCr (first row):");
    print!("   Y: ");
    for i in 0..8 {
        print!("{:6.1}", y_block[i]);
    }
    println!();
    print!("  Cb: ");
    for i in 0..8 {
        print!("{:6.1}", cb_block[i]);
    }
    println!();
    print!("  Cr: ");
    for i in 0..8 {
        print!("{:6.1}", cr_block[i]);
    }
    println!();

    // Step 3: Level shift
    let mut y_shifted = y_block;
    for v in y_shifted.iter_mut() {
        *v -= 128.0;
    }

    println!("\n3. Level-shifted Y (first row):");
    print!("     ");
    for i in 0..8 {
        print!("{:6.1}", y_shifted[i]);
    }
    println!();

    // Step 4: DCT
    let y_dct = forward_dct_8x8(&y_shifted);

    println!("\n4. DCT coefficients (Y, zigzag order first 16):");
    print!("     ");
    for i in 0..16 {
        print!("{:7.1}", y_dct[i]);
    }
    println!();
    println!("   DC = {:.1}, AC[1] = {:.1}", y_dct[0], y_dct[1]);

    // Step 5: Quantization
    let q = Quality::from_quality(quality);
    let y_quant = generate_quant_table(q, 0, ColorSpace::YCbCr, false);

    println!("\n5. Quantization table (Y, first 16):");
    print!("     ");
    for i in 0..16 {
        print!("{:4}", y_quant.values[i]);
    }
    println!();

    // Quantized coefficients
    let mut y_quantized = [0i16; 64];
    for i in 0..64 {
        y_quantized[i] = (y_dct[i] / y_quant.values[i] as f32).round() as i16;
    }

    println!("\n6. Quantized coefficients (first 16):");
    print!("     ");
    for i in 0..16 {
        print!("{:4}", y_quantized[i]);
    }
    println!();

    // Count zeros
    let zeros = y_quantized.iter().filter(|&&x| x == 0).count();
    let nonzeros = 64 - zeros;
    println!("\n   Zeros: {}/64 ({:.1}%), Non-zeros: {}",
        zeros, zeros as f32 / 64.0 * 100.0, nonzeros);
}

// ============================================================================
// DUMP: Inspect JPEG structure
// ============================================================================

fn cmd_dump(args: &[String]) {
    if args.is_empty() {
        println!("Usage: dump <image.jpg>");
        println!("\nDumps JPEG structure:");
        println!("  - Markers and segments");
        println!("  - Quantization tables");
        println!("  - Huffman tables");
        println!("  - Coefficient statistics");
        return;
    }

    let path = &args[0];
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read file: {}", e);
            return;
        }
    };

    println!("=== JPEG DUMP: {} ===", path);
    println!("File size: {} bytes ({:.1} KB)", data.len(), data.len() as f32 / 1024.0);
    println!();

    // Parse markers
    dump_markers(&data);

    // Decode and show coefficient stats
    println!("\n=== COEFFICIENT STATISTICS ===");
    dump_coefficient_stats(path);
}

fn dump_markers(data: &[u8]) {
    println!("=== MARKERS ===");
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];
            let name = marker_name(marker);

            if marker == 0xD8 || marker == 0xD9 {
                // SOI/EOI - no length
                println!("  {:04X}: FF {:02X} {}", i, marker, name);
                i += 2;
            } else if marker >= 0xD0 && marker <= 0xD7 {
                // RST markers - no length
                println!("  {:04X}: FF {:02X} {}", i, marker, name);
                i += 2;
            } else if i + 3 < data.len() {
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                println!("  {:04X}: FF {:02X} {} (len={})", i, marker, name, len);

                // Show quant table details
                if marker == 0xDB && len > 2 {
                    dump_dqt(&data[i + 4..i + 2 + len]);
                }

                i += 2 + len;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

fn marker_name(marker: u8) -> &'static str {
    match marker {
        0xD8 => "SOI (Start of Image)",
        0xD9 => "EOI (End of Image)",
        0xC0 => "SOF0 (Baseline DCT)",
        0xC2 => "SOF2 (Progressive DCT)",
        0xC4 => "DHT (Define Huffman Table)",
        0xDB => "DQT (Define Quantization Table)",
        0xDD => "DRI (Define Restart Interval)",
        0xDA => "SOS (Start of Scan)",
        0xE0 => "APP0 (JFIF)",
        0xE1 => "APP1 (EXIF)",
        0xE2 => "APP2 (ICC Profile)",
        0xFE => "COM (Comment)",
        _ if marker >= 0xD0 && marker <= 0xD7 => "RST (Restart)",
        _ => "Unknown",
    }
}

fn dump_dqt(data: &[u8]) {
    if data.is_empty() {
        return;
    }
    let pq = (data[0] >> 4) & 0x0F;
    let tq = data[0] & 0x0F;
    println!("       Table {}: precision={} ({})",
        tq, pq, if pq == 0 { "8-bit" } else { "16-bit" });

    if pq == 0 && data.len() >= 65 {
        print!("       Values: ");
        for i in 0..8 {
            print!("{:3} ", data[1 + i]);
        }
        println!("...");
    }
}

fn dump_coefficient_stats(path: &str) {
    // Use jpeg-decoder to get coefficients
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => return,
    };

    match jpeg_decoder::Decoder::new(&data[..]).decode() {
        Ok(pixels) => {
            println!("Decoded successfully: {} bytes of pixel data", pixels.len());
        }
        Err(e) => {
            println!("Decode error: {}", e);
        }
    }
}

// ============================================================================
// COMPARE: Rust vs C++ encoding
// ============================================================================

fn cmd_compare(args: &[String]) {
    if args.is_empty() {
        println!("Usage: compare <image.png> [quality]");
        println!("\nCompares Rust jpegli vs C++ cjpegli encoding:");
        println!("  - File sizes");
        println!("  - Quality metrics (DSSIM)");
        println!("  - Coefficient differences");
        return;
    }

    let path = &args[0];
    let quality: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(90);

    println!("=== RUST vs C++ COMPARISON ===");
    println!("Input: {}", path);
    println!("Quality: {}", quality);
    println!();

    // Check if input exists
    if !Path::new(path).exists() {
        eprintln!("File not found: {}", path);
        return;
    }

    // Load image
    let (rgb, width, height) = match load_png(path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            return;
        }
    };

    // Encode with Rust
    println!("Encoding with Rust jpegli...");
    let rust_start = std::time::Instant::now();
    let rust_jpeg = match encode_rust(&rgb, width, height, quality) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Rust encoding failed: {}", e);
            return;
        }
    };
    let rust_time = rust_start.elapsed();

    // Encode with C++
    println!("Encoding with C++ cjpegli...");
    let cpp_start = std::time::Instant::now();
    let cpp_jpeg = match encode_cpp(path, quality) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("C++ encoding failed: {}", e);
            println!("(Is cjpegli installed? Try: ../build/tools/cjpegli)");
            return;
        }
    };
    let cpp_time = cpp_start.elapsed();

    // Compare sizes
    println!("\n=== FILE SIZE ===");
    println!("  Rust: {:>8} bytes ({:.1} KB)", rust_jpeg.len(), rust_jpeg.len() as f32 / 1024.0);
    println!("  C++:  {:>8} bytes ({:.1} KB)", cpp_jpeg.len(), cpp_jpeg.len() as f32 / 1024.0);
    let diff = rust_jpeg.len() as f64 / cpp_jpeg.len() as f64 - 1.0;
    println!("  Diff: {:+.2}%", diff * 100.0);

    // Compare times
    println!("\n=== ENCODE TIME ===");
    println!("  Rust: {:?}", rust_time);
    println!("  C++:  {:?}", cpp_time);

    // Decode both and compare quality
    println!("\n=== QUALITY (vs original) ===");
    compare_quality(&rgb, width, height, &rust_jpeg, &cpp_jpeg);

    // Save for inspection
    let rust_out = "/tmp/jpegli_debug_rust.jpg";
    let cpp_out = "/tmp/jpegli_debug_cpp.jpg";
    fs::write(rust_out, &rust_jpeg).ok();
    fs::write(cpp_out, &cpp_jpeg).ok();
    println!("\nSaved: {} and {}", rust_out, cpp_out);
}

fn encode_rust(rgb: &[u8], width: usize, height: usize, quality: u8) -> Result<Vec<u8>, String> {
    use jpegli::{Encoder, Quality};
    use jpegli::types::PixelFormat;

    Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality as f32))
        .optimize_huffman(true)
        .encode(rgb)
        .map_err(|e| e.to_string())
}

fn encode_cpp(input_path: &str, quality: u8) -> Result<Vec<u8>, String> {
    let output = "/tmp/jpegli_debug_cpp_temp.jpg";

    // Try multiple possible locations for cjpegli
    let cjpegli_paths = [
        "../build/tools/cjpegli",
        "../../build/tools/cjpegli",
        "/usr/local/bin/cjpegli",
        "cjpegli",
    ];

    let mut cmd_result = None;
    for cjpegli in &cjpegli_paths {
        let result = Command::new(cjpegli)
            .args(["-q", &quality.to_string(), input_path, output])
            .output();

        if result.is_ok() {
            cmd_result = Some(result);
            break;
        }
    }

    let cmd_output = cmd_result
        .ok_or("cjpegli not found")?
        .map_err(|e| e.to_string())?;

    if !cmd_output.status.success() {
        return Err(format!("cjpegli failed: {}", String::from_utf8_lossy(&cmd_output.stderr)));
    }

    fs::read(output).map_err(|e| e.to_string())
}

fn compare_quality(original: &[u8], width: usize, height: usize, rust_jpeg: &[u8], cpp_jpeg: &[u8]) {
    // Decode both JPEGs
    let rust_decoded = decode_jpeg(rust_jpeg);
    let cpp_decoded = decode_jpeg(cpp_jpeg);

    if let (Ok(rust_pixels), Ok(cpp_pixels)) = (&rust_decoded, &cpp_decoded) {
        // Calculate max pixel difference
        let rust_diff = max_pixel_diff(original, rust_pixels);
        let cpp_diff = max_pixel_diff(original, cpp_pixels);

        println!("  Max pixel diff (Rust vs orig): {}", rust_diff);
        println!("  Max pixel diff (C++ vs orig):  {}", cpp_diff);

        // Compare decoded outputs
        let rust_cpp_diff = max_pixel_diff(rust_pixels, cpp_pixels);
        println!("  Max pixel diff (Rust vs C++):  {}", rust_cpp_diff);

        // DSSIM if available
        if let (Ok(rust_dssim), Ok(cpp_dssim)) = (
            compute_dssim(original, width, height, rust_pixels),
            compute_dssim(original, width, height, cpp_pixels),
        ) {
            println!("  DSSIM (Rust): {:.6}", rust_dssim);
            println!("  DSSIM (C++):  {:.6}", cpp_dssim);
        }
    }
}

fn decode_jpeg(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().map_err(|e| e.to_string())
}

fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn compute_dssim(original: &[u8], width: usize, height: usize, decoded: &[u8]) -> Result<f64, String> {
    use dssim::Dssim;
    use rgb::RGBA8;

    let attr = Dssim::new();

    // Convert to RGBA for dssim
    let orig_rgba: Vec<RGBA8> = original.chunks(3)
        .map(|rgb| RGBA8::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let dec_rgba: Vec<RGBA8> = decoded.chunks(3)
        .map(|rgb| RGBA8::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height)
        .ok_or("Failed to create original image")?;
    let dec_img = attr.create_image_rgba(&dec_rgba, width, height)
        .ok_or("Failed to create decoded image")?;

    let (dssim, _) = attr.compare(&orig_img, dec_img);
    Ok(dssim.into())
}

// ============================================================================
// ANALYZE: Quality metrics
// ============================================================================

fn cmd_analyze(args: &[String]) {
    if args.len() < 2 {
        println!("Usage: analyze <original.png> <encoded.jpg>");
        println!("\nAnalyzes quality metrics:");
        println!("  - DSSIM (structural dissimilarity)");
        println!("  - Max/mean pixel difference");
        println!("  - Compression ratio");
        return;
    }

    let orig_path = &args[0];
    let jpeg_path = &args[1];

    println!("=== QUALITY ANALYSIS ===");
    println!("Original: {}", orig_path);
    println!("Encoded:  {}", jpeg_path);
    println!();

    // Load original
    let (original, width, height) = match load_png(orig_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load original: {}", e);
            return;
        }
    };

    // Load encoded
    let jpeg_data = match fs::read(jpeg_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load JPEG: {}", e);
            return;
        }
    };

    let decoded = match decode_jpeg(&jpeg_data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to decode JPEG: {}", e);
            return;
        }
    };

    // Size analysis
    let orig_size = original.len();
    let jpeg_size = jpeg_data.len();
    let ratio = orig_size as f64 / jpeg_size as f64;

    println!("=== SIZE ===");
    println!("  Original: {} bytes ({:.1} KB)", orig_size, orig_size as f32 / 1024.0);
    println!("  JPEG:     {} bytes ({:.1} KB)", jpeg_size, jpeg_size as f32 / 1024.0);
    println!("  Ratio:    {:.1}:1", ratio);
    println!("  BPP:      {:.3}", jpeg_size as f64 * 8.0 / (width * height) as f64);

    // Quality analysis
    println!("\n=== QUALITY ===");
    let max_diff = max_pixel_diff(&original, &decoded);
    let mean_diff: f64 = original.iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).abs() as f64)
        .sum::<f64>() / original.len() as f64;

    println!("  Max pixel diff:  {}", max_diff);
    println!("  Mean pixel diff: {:.3}", mean_diff);

    if let Ok(dssim) = compute_dssim(&original, width, height, &decoded) {
        println!("  DSSIM:           {:.6}", dssim);
    }
}

// ============================================================================
// BLOCK: Single block analysis
// ============================================================================

fn cmd_block(args: &[String]) {
    let pattern = args.get(0).map(|s| s.as_str()).unwrap_or("gradient");

    println!("=== SINGLE BLOCK ANALYSIS ===");
    println!("Pattern: {}", pattern);
    println!();

    // Generate test block
    let mut rgb = [0u8; 64 * 3];
    match pattern {
        "gradient" => {
            for i in 0..64 {
                let x = (i % 8) as u8;
                let y = (i / 8) as u8;
                rgb[i * 3] = x * 32;
                rgb[i * 3 + 1] = y * 32;
                rgb[i * 3 + 2] = 128;
            }
        }
        "uniform" => {
            for i in 0..64 {
                rgb[i * 3] = 128;
                rgb[i * 3 + 1] = 128;
                rgb[i * 3 + 2] = 128;
            }
        }
        "checkerboard" => {
            for i in 0..64 {
                let x = i % 8;
                let y = i / 8;
                let val = if (x + y) % 2 == 0 { 200u8 } else { 50u8 };
                rgb[i * 3] = val;
                rgb[i * 3 + 1] = val;
                rgb[i * 3 + 2] = val;
            }
        }
        _ => {
            println!("Unknown pattern. Available: gradient, uniform, checkerboard");
            return;
        }
    }

    // Trace this block at multiple quality levels
    for q in [100, 90, 75, 50] {
        println!("\n--- Quality {} ---", q);
        trace_block(&rgb, 8, q as f32, 0, 0);
    }
}

// ============================================================================
// QUANT: Show quantization tables
// ============================================================================

fn cmd_quant(args: &[String]) {
    use jpegli::quant::{generate_quant_table, Quality};
    use jpegli::types::ColorSpace;

    let quality: f32 = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(90.0);

    println!("=== QUANTIZATION TABLES ===");
    println!("Quality: {}", quality);
    println!();

    let q = Quality::from_quality(quality);
    let y_table = generate_quant_table(q, 0, ColorSpace::YCbCr, false);
    let c_table = generate_quant_table(q, 1, ColorSpace::YCbCr, false);

    println!("Y (Luminance) Table:");
    print_quant_table(&y_table.values);

    println!("\nCbCr (Chrominance) Table:");
    print_quant_table(&c_table.values);

    // Show statistics
    let y_sum: u32 = y_table.values.iter().map(|&x| x as u32).sum();
    let c_sum: u32 = c_table.values.iter().map(|&x| x as u32).sum();

    println!("\nStatistics:");
    println!("  Y table sum:  {:5} (mean: {:.1})", y_sum, y_sum as f32 / 64.0);
    println!("  C table sum:  {:5} (mean: {:.1})", c_sum, c_sum as f32 / 64.0);
    println!("  Y DC quant:   {}", y_table.values[0]);
    println!("  Y AC[1]:      {}", y_table.values[1]);
}

fn print_quant_table(table: &[u16; 64]) {
    // Print in 8x8 grid (natural order, not zigzag)
    let zigzag_to_natural = [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ];

    let mut natural = [0u16; 64];
    for (z, &n) in zigzag_to_natural.iter().enumerate() {
        natural[n] = table[z];
    }

    for y in 0..8 {
        print!("  ");
        for x in 0..8 {
            print!("{:4}", natural[y * 8 + x]);
        }
        println!();
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn load_png(path: &str) -> Result<(Vec<u8>, usize, usize), String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;
    let decoder = png::Decoder::new(&data[..]);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type)),
    };

    Ok((rgb, info.width as usize, info.height as usize))
}
