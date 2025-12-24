//! Compare decoded pixels between C++ and Rust encoded JPEGs

use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    let test_images = [
        (
            "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png",
            "flower",
        ),
        ("/mnt/v/work/corpus/CID22-512/1459534.png", "cid22_large"),
        (
            "/mnt/v/work/corpus/CID22-512/2504911.png",
            "cid22_medium_large",
        ),
        ("/mnt/v/work/corpus/CID22-512/3616956.png", "cid22_medium"),
        (
            "/mnt/v/work/corpus/CID22-512/nicubunu_Game_baddie_Policeman.png",
            "cid22_small",
        ),
    ];

    println!("=== Decoded Pixel Comparison (C++ vs Rust, Q90) ===\n");
    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "Image", "Exact%", "±1", "±2", "±3+", "MaxDiff", "PSNR"
    );
    println!("{}", "-".repeat(80));

    for (png_path, name) in test_images {
        if let Some(result) = compare_image(png_path, name) {
            let total = result.total_pixels as f64;
            println!(
                "{:<20} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>8} {:>6.1}",
                name,
                100.0 * result.exact_match as f64 / total,
                100.0 * result.off_by_1 as f64 / total,
                100.0 * result.off_by_2 as f64 / total,
                100.0 * result.off_by_3_plus as f64 / total,
                result.max_diff,
                result.psnr
            );
        } else {
            println!("{:<20} SKIPPED", name);
        }
    }

    println!("\n=== Histogram of differences ===\n");

    // Just do flower for detailed histogram
    if let Some((hist, max_diff)) = detailed_histogram(
        "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png",
        "flower",
    ) {
        println!("Flower image difference histogram:");
        for (diff, count) in hist.iter().enumerate().take(max_diff as usize + 1) {
            if *count > 0 {
                let pct = 100.0 * *count as f64 / hist.iter().sum::<usize>() as f64;
                let bar = "█".repeat((pct * 2.0) as usize);
                println!("  {:>2}: {:>8} ({:>5.2}%) {}", diff, count, pct, bar);
            }
        }
    }
}

struct CompareResult {
    total_pixels: usize,
    exact_match: usize,
    off_by_1: usize,
    off_by_2: usize,
    off_by_3_plus: usize,
    max_diff: u8,
    psnr: f64,
}

fn compare_image(png_path: &str, name: &str) -> Option<CompareResult> {
    let (rgb, width, height) = load_png(png_path)?;

    let ppm_path = format!("/tmp/{}_compare.ppm", name);
    write_ppm(&ppm_path, &rgb, width as usize, height as usize).ok()?;

    let cpp_jpg_path = format!("/tmp/{}_cpp.jpg", name);
    let output = Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
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

    if !output.status.success() {
        return None;
    }

    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&rgb)
        .ok()?;

    let cpp_decoded = decode_jpeg(&fs::read(&cpp_jpg_path).ok()?)?;
    let rust_decoded = decode_jpeg(&rust_jpeg)?;

    if cpp_decoded.len() != rust_decoded.len() {
        return None;
    }

    let total_pixels = cpp_decoded.len();
    let mut exact_match = 0usize;
    let mut off_by_1 = 0usize;
    let mut off_by_2 = 0usize;
    let mut off_by_3_plus = 0usize;
    let mut max_diff = 0u8;
    let mut mse_sum = 0.0f64;

    for i in 0..total_pixels {
        let diff = (cpp_decoded[i] as i16 - rust_decoded[i] as i16).unsigned_abs() as u8;
        mse_sum += (diff as f64).powi(2);
        max_diff = max_diff.max(diff);

        match diff {
            0 => exact_match += 1,
            1 => off_by_1 += 1,
            2 => off_by_2 += 1,
            _ => off_by_3_plus += 1,
        }
    }

    let mse = mse_sum / total_pixels as f64;
    let psnr = if mse > 0.0 {
        10.0 * (255.0_f64.powi(2) / mse).log10()
    } else {
        100.0
    };

    Some(CompareResult {
        total_pixels,
        exact_match,
        off_by_1,
        off_by_2,
        off_by_3_plus,
        max_diff,
        psnr,
    })
}

fn detailed_histogram(png_path: &str, name: &str) -> Option<(Vec<usize>, u8)> {
    let (rgb, width, height) = load_png(png_path)?;

    let ppm_path = format!("/tmp/{}_compare.ppm", name);
    write_ppm(&ppm_path, &rgb, width as usize, height as usize).ok()?;

    let cpp_jpg_path = format!("/tmp/{}_cpp.jpg", name);
    Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
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

    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&rgb)
        .ok()?;

    let cpp_decoded = decode_jpeg(&fs::read(&cpp_jpg_path).ok()?)?;
    let rust_decoded = decode_jpeg(&rust_jpeg)?;

    let mut histogram = vec![0usize; 256];
    let mut max_diff = 0u8;

    for i in 0..cpp_decoded.len() {
        let diff = (cpp_decoded[i] as i16 - rust_decoded[i] as i16).unsigned_abs() as u8;
        histogram[diff as usize] += 1;
        max_diff = max_diff.max(diff);
    }

    Some((histogram, max_diff))
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
