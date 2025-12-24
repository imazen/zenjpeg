//! Deep pixel analysis between C++ and Rust encoded JPEGs
//! Checks for systematic biases, per-channel errors, and spatial distribution

use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    let png_path = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";

    println!("=== Deep Analysis: C++ vs Rust JPEG Encoding ===\n");

    let (original, width, height) = load_png(png_path).expect("Failed to load PNG");

    // Write PPM for C++
    let ppm_path = "/tmp/deep_compare.ppm";
    write_ppm(ppm_path, &original, width as usize, height as usize).expect("Failed to write PPM");

    // Encode with C++
    let cpp_jpg = "/tmp/deep_cpp.jpg";
    Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p",
            "0",
            "--fixed_code",
            ppm_path,
            cpp_jpg,
            "-q",
            "90",
        ])
        .output()
        .expect("Failed to run cjpegli");

    // Encode with Rust
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&original)
        .expect("Failed to encode");

    // Save Rust JPEG
    let rust_jpg = "/tmp/deep_rust.jpg";
    fs::write(rust_jpg, &rust_jpeg).expect("Failed to write Rust JPEG");

    let cpp_decoded = decode_jpeg(&fs::read(cpp_jpg).expect("Failed to read C++ JPEG"))
        .expect("Failed to decode C++ JPEG");
    let rust_decoded = decode_jpeg(&rust_jpeg).expect("Failed to decode Rust JPEG");

    let total = original.len();
    let num_pixels = total / 3;

    println!(
        "Image: {}x{} = {} pixels ({} bytes)",
        width, height, num_pixels, total
    );
    println!("C++ JPEG: {} bytes", fs::metadata(cpp_jpg).unwrap().len());
    println!("Rust JPEG: {} bytes", rust_jpeg.len());
    println!();

    // Per-channel analysis
    println!("=== Per-Channel Analysis (C++ vs Rust decoded) ===\n");

    for (ch_name, ch_offset) in [("R", 0), ("G", 1), ("B", 2)] {
        let mut sum_cpp = 0i64;
        let mut sum_rust = 0i64;
        let mut sum_diff = 0i64;
        let mut sum_abs_diff = 0u64;
        let mut max_diff: i16 = 0;
        let mut min_diff: i16 = 0;
        let mut exact = 0usize;

        for i in 0..num_pixels {
            let cpp_val = cpp_decoded[i * 3 + ch_offset] as i16;
            let rust_val = rust_decoded[i * 3 + ch_offset] as i16;
            let diff = rust_val - cpp_val; // positive = Rust brighter

            sum_cpp += cpp_val as i64;
            sum_rust += rust_val as i64;
            sum_diff += diff as i64;
            sum_abs_diff += diff.unsigned_abs() as u64;
            max_diff = max_diff.max(diff);
            min_diff = min_diff.min(diff);
            if diff == 0 {
                exact += 1;
            }
        }

        let mean_cpp = sum_cpp as f64 / num_pixels as f64;
        let mean_rust = sum_rust as f64 / num_pixels as f64;
        let mean_diff = sum_diff as f64 / num_pixels as f64;
        let mean_abs_diff = sum_abs_diff as f64 / num_pixels as f64;

        println!("{} channel:", ch_name);
        println!("  Mean C++: {:.2}, Mean Rust: {:.2}", mean_cpp, mean_rust);
        println!("  Mean diff: {:+.4} (Rust - C++)", mean_diff);
        println!("  Mean |diff|: {:.4}", mean_abs_diff);
        println!("  Range: {} to {}", min_diff, max_diff);
        println!(
            "  Exact match: {} ({:.1}%)",
            exact,
            100.0 * exact as f64 / num_pixels as f64
        );
        println!();
    }

    // Compare both to original
    println!("=== Quality vs Original (per channel) ===\n");

    for (ch_name, ch_offset) in [("R", 0), ("G", 1), ("B", 2)] {
        let mut cpp_mse = 0.0f64;
        let mut rust_mse = 0.0f64;
        let mut cpp_better = 0usize;
        let mut rust_better = 0usize;
        let mut same = 0usize;

        for i in 0..num_pixels {
            let orig = original[i * 3 + ch_offset] as i16;
            let cpp_val = cpp_decoded[i * 3 + ch_offset] as i16;
            let rust_val = rust_decoded[i * 3 + ch_offset] as i16;

            let cpp_err = (orig - cpp_val).abs();
            let rust_err = (orig - rust_val).abs();

            cpp_mse += (cpp_err as f64).powi(2);
            rust_mse += (rust_err as f64).powi(2);

            if cpp_err < rust_err {
                cpp_better += 1;
            } else if rust_err < cpp_err {
                rust_better += 1;
            } else {
                same += 1;
            }
        }

        cpp_mse /= num_pixels as f64;
        rust_mse /= num_pixels as f64;

        let cpp_psnr = 10.0 * (255.0_f64.powi(2) / cpp_mse).log10();
        let rust_psnr = 10.0 * (255.0_f64.powi(2) / rust_mse).log10();

        println!("{} channel:", ch_name);
        println!(
            "  C++ PSNR: {:.2} dB, Rust PSNR: {:.2} dB (diff: {:+.2} dB)",
            cpp_psnr,
            rust_psnr,
            rust_psnr - cpp_psnr
        );
        println!(
            "  C++ better: {} ({:.1}%), Rust better: {} ({:.1}%), Same: {} ({:.1}%)",
            cpp_better,
            100.0 * cpp_better as f64 / num_pixels as f64,
            rust_better,
            100.0 * rust_better as f64 / num_pixels as f64,
            same,
            100.0 * same as f64 / num_pixels as f64
        );
        println!();
    }

    // Difference histogram with sign
    println!("=== Signed Difference Histogram (Rust - C++) ===\n");

    let mut signed_hist: std::collections::BTreeMap<i16, usize> = std::collections::BTreeMap::new();
    for i in 0..total {
        let diff = rust_decoded[i] as i16 - cpp_decoded[i] as i16;
        *signed_hist.entry(diff).or_insert(0) += 1;
    }

    println!("Diff  Count       Pct");
    for (&diff, &count) in &signed_hist {
        let pct = 100.0 * count as f64 / total as f64;
        if pct > 0.1 {
            // Only show diffs with >0.1%
            let bar_len = (pct * 2.0).min(40.0) as usize;
            let bar = "█".repeat(bar_len);
            println!("{:>4}: {:>8} {:>5.1}%  {}", diff, count, pct, bar);
        }
    }

    // Sum of positive vs negative differences (bias check)
    let positive: i64 = signed_hist
        .iter()
        .filter(|(&d, _)| d > 0)
        .map(|(&d, &c)| d as i64 * c as i64)
        .sum();
    let negative: i64 = signed_hist
        .iter()
        .filter(|(&d, _)| d < 0)
        .map(|(&d, &c)| d as i64 * c as i64)
        .sum();

    println!("\nBias check:");
    println!("  Sum of positive diffs: +{}", positive);
    println!("  Sum of negative diffs: {}", negative);
    println!("  Net bias: {:+}", positive + negative);
    println!(
        "  Net bias per byte: {:+.4}",
        (positive + negative) as f64 / total as f64
    );

    // Spatial analysis: where are the worst differences?
    println!("\n=== Worst Differences (top 10 locations) ===\n");

    let mut diffs_with_loc: Vec<(usize, usize, i16, &str)> = Vec::new();
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            for (ch_offset, ch_name) in [(0, "R"), (1, "G"), (2, "B")] {
                let diff =
                    rust_decoded[idx + ch_offset] as i16 - cpp_decoded[idx + ch_offset] as i16;
                if diff.abs() >= 10 {
                    diffs_with_loc.push((x, y, diff, ch_name));
                }
            }
        }
    }

    diffs_with_loc.sort_by_key(|&(_, _, d, _)| std::cmp::Reverse(d.abs()));

    println!("  (x, y) Channel  Diff  Original  C++  Rust");
    for (x, y, diff, ch) in diffs_with_loc.iter().take(20) {
        let idx = (*y * width as usize + *x) * 3;
        let ch_off = match *ch {
            "R" => 0,
            "G" => 1,
            _ => 2,
        };
        let orig = original[idx + ch_off];
        let cpp = cpp_decoded[idx + ch_off];
        let rust = rust_decoded[idx + ch_off];
        println!(
            "  ({:3}, {:3}) {:>5}  {:>+4}     {:>3}     {:>3}  {:>3}",
            x, y, ch, diff, orig, cpp, rust
        );
    }

    // Compare file sizes
    println!("\n=== File Size Comparison ===\n");
    let cpp_size = fs::metadata(cpp_jpg).unwrap().len();
    let rust_size = rust_jpeg.len() as u64;
    println!("C++ JPEG: {} bytes", cpp_size);
    println!("Rust JPEG: {} bytes", rust_size);
    println!(
        "Difference: {:+} bytes ({:+.2}%)",
        rust_size as i64 - cpp_size as i64,
        100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64
    );
}

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let decoder = png::Decoder::new(fs::File::open(path).ok()?);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6\n{} {}\n255", width, height)?;
    file.write_all(rgb)
}

fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Cursor;
    jpeg_decoder::Decoder::new(Cursor::new(data)).decode().ok()
}
