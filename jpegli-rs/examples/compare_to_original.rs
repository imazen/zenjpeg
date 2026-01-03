//! Compare decoded outputs to original image

use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    // Build test image list dynamically
    let flower_path = jpegli::test_utils::require_flower_small_path();
    let flower_str = flower_path.to_string_lossy().to_string();

    let mut test_images: Vec<(String, &str)> = vec![(flower_str.clone(), "flower")];

    // Add optional corpus images if they exist
    for (path, name) in [
        ("/mnt/v/work/corpus/CID22-512/1459534.png", "cid22_large"),
        (
            "/mnt/v/work/corpus/CID22-512/nicubunu_Game_baddie_Policeman.png",
            "cid22_small",
        ),
    ] {
        if std::path::Path::new(path).exists() {
            test_images.push((path.to_string(), name));
        }
    }

    println!("=== Quality vs Original (Q90) ===\n");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12}",
        "Image", "C++ PSNR", "Rust PSNR", "C++ MaxErr", "Rust MaxErr"
    );
    println!("{}", "-".repeat(70));

    for (png_path, name) in &test_images {
        if let Some(result) = compare_to_original(png_path, name) {
            println!(
                "{:<20} {:>12.2} {:>12.2} {:>12} {:>12}",
                name, result.cpp_psnr, result.rust_psnr, result.cpp_max_err, result.rust_max_err
            );
        }
    }

    println!("\n=== Detailed breakdown for flower ===\n");
    if let Some(result) = compare_to_original(&flower_str, "flower") {
        println!(
            "C++  vs Original: PSNR={:.2} dB, MSE={:.2}, MaxErr={}",
            result.cpp_psnr, result.cpp_mse, result.cpp_max_err
        );
        println!(
            "Rust vs Original: PSNR={:.2} dB, MSE={:.2}, MaxErr={}",
            result.rust_psnr, result.rust_mse, result.rust_max_err
        );
        println!(
            "C++  vs Rust:     PSNR={:.2} dB, MSE={:.2}, MaxErr={}",
            result.cross_psnr, result.cross_mse, result.cross_max_err
        );

        println!(
            "\nC++ better on {} pixels ({:.1}%)",
            result.cpp_better,
            100.0 * result.cpp_better as f64 / result.total as f64
        );
        println!(
            "Rust better on {} pixels ({:.1}%)",
            result.rust_better,
            100.0 * result.rust_better as f64 / result.total as f64
        );
        println!(
            "Same distance: {} pixels ({:.1}%)",
            result.same,
            100.0 * result.same as f64 / result.total as f64
        );
    }
}

struct CompareResult {
    cpp_psnr: f64,
    rust_psnr: f64,
    cpp_mse: f64,
    rust_mse: f64,
    cpp_max_err: u8,
    rust_max_err: u8,
    cross_psnr: f64,
    cross_mse: f64,
    cross_max_err: u8,
    cpp_better: usize,
    rust_better: usize,
    same: usize,
    total: usize,
}

fn compare_to_original(png_path: &str, name: &str) -> Option<CompareResult> {
    let (original, width, height) = load_png(png_path)?;

    let ppm_path = format!("/tmp/{}_compare.ppm", name);
    write_ppm(&ppm_path, &original, width as usize, height as usize).ok()?;

    // Encode with C++
    let cpp_jpg_path = format!("/tmp/{}_cpp.jpg", name);
    let cjpegli_path = jpegli::test_utils::find_cjpegli()?;
    Command::new(&cjpegli_path)
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p",
            "0",
            "--fixed_code",
            &ppm_path,
            &cpp_jpg_path,
            "-q",
            "90",
        ])
        .output()
        .ok()?;

    // Encode with Rust
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&original)
        .ok()?;

    let cpp_decoded = decode_jpeg(&fs::read(&cpp_jpg_path).ok()?)?;
    let rust_decoded = decode_jpeg(&rust_jpeg)?;

    let total = original.len();
    let mut cpp_mse = 0.0f64;
    let mut rust_mse = 0.0f64;
    let mut cross_mse = 0.0f64;
    let mut cpp_max_err = 0u8;
    let mut rust_max_err = 0u8;
    let mut cross_max_err = 0u8;
    let mut cpp_better = 0usize;
    let mut rust_better = 0usize;
    let mut same = 0usize;

    for i in 0..total {
        let orig = original[i] as i16;
        let cpp = cpp_decoded[i] as i16;
        let rust = rust_decoded[i] as i16;

        let cpp_err = (orig - cpp).unsigned_abs() as u8;
        let rust_err = (orig - rust).unsigned_abs() as u8;
        let cross_err = (cpp - rust).unsigned_abs() as u8;

        cpp_mse += (cpp_err as f64).powi(2);
        rust_mse += (rust_err as f64).powi(2);
        cross_mse += (cross_err as f64).powi(2);

        cpp_max_err = cpp_max_err.max(cpp_err);
        rust_max_err = rust_max_err.max(rust_err);
        cross_max_err = cross_max_err.max(cross_err);

        if cpp_err < rust_err {
            cpp_better += 1;
        } else if rust_err < cpp_err {
            rust_better += 1;
        } else {
            same += 1;
        }
    }

    cpp_mse /= total as f64;
    rust_mse /= total as f64;
    cross_mse /= total as f64;

    let cpp_psnr = if cpp_mse > 0.0 {
        10.0 * (255.0_f64.powi(2) / cpp_mse).log10()
    } else {
        100.0
    };
    let rust_psnr = if rust_mse > 0.0 {
        10.0 * (255.0_f64.powi(2) / rust_mse).log10()
    } else {
        100.0
    };
    let cross_psnr = if cross_mse > 0.0 {
        10.0 * (255.0_f64.powi(2) / cross_mse).log10()
    } else {
        100.0
    };

    Some(CompareResult {
        cpp_psnr,
        rust_psnr,
        cpp_mse,
        rust_mse,
        cpp_max_err,
        rust_max_err,
        cross_psnr,
        cross_mse,
        cross_max_err,
        cpp_better,
        rust_better,
        same,
        total,
    })
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
