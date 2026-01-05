//! Simpler test - low frequency content that JPEG should handle well
//! but with odd dimensions to stress MCU boundaries

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;

fn generate_blocks(width: usize, height: usize) -> Vec<u8> {
    // 8x8 blocks of solid colors - should be nearly lossless
    let mut data = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let block_x = x / 8;
            let block_y = y / 8;

            // Different solid colors per 8x8 block
            let color = ((block_x * 37 + block_y * 53) % 8) as u8;
            let (r, g, b) = match color {
                0 => (200, 50, 50),
                1 => (50, 200, 50),
                2 => (50, 50, 200),
                3 => (200, 200, 50),
                4 => (200, 50, 200),
                5 => (50, 200, 200),
                6 => (150, 150, 150),
                _ => (100, 100, 100),
            };

            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
        }
    }
    data
}

fn decode_mozjpeg(data: &[u8]) -> Vec<u8> {
    let d = mozjpeg::Decompress::new_mem(data).unwrap();
    let mut dec = d.rgb().unwrap();
    let mut buf = vec![0u8; dec.width() * dec.height() * 3];
    #[allow(deprecated)]
    let _ = dec.read_scanlines_into::<u8>(&mut buf);
    buf
}

fn main() {
    // Odd dimension that's not a multiple of 8 or 16
    let width = 67usize;
    let height = 51usize;

    let original = generate_blocks(width, height);

    println!("=== Simple Block Test {}x{} ===\n", width, height);
    println!("{:<20} {:>8} {:>10} {:>10}", "Config", "Size", "MaxDiff", "AvgDiff");
    println!("{:-<55}", "");

    let configs = [
        ("Baseline 444", JpegMode::Baseline, Subsampling::S444),
        ("Baseline 420", JpegMode::Baseline, Subsampling::S420),
        ("Progressive 444", JpegMode::Progressive, Subsampling::S444),
        ("Progressive 420", JpegMode::Progressive, Subsampling::S420),
    ];

    // Save PNG for C++ comparison
    let png_path = "/tmp/simple_block.png";
    {
        let file = fs::File::create(png_path).unwrap();
        let mut encoder = png::Encoder::new(file, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&original).unwrap();
    }

    for (name, mode, sub) in &configs {
        let jpeg = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .mode(*mode)
            .subsampling(*sub)
            .optimize_huffman(true)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&original)
            .unwrap();

        let decoded = decode_mozjpeg(&jpeg);

        let mut max_diff = 0u8;
        let mut sum_diff = 0u64;
        for (&o, &d) in original.iter().zip(decoded.iter()) {
            let diff = (o as i16 - d as i16).unsigned_abs() as u8;
            max_diff = max_diff.max(diff);
            sum_diff += diff as u64;
        }
        let avg_diff = sum_diff as f64 / original.len() as f64;

        let status = if max_diff > 20 { " <-- BAD" } else { "" };
        println!("{:<20} {:>8} {:>10} {:>10.2}{}", name, jpeg.len(), max_diff, avg_diff, status);

        // Save JPEGs for inspection
        let safe_name = name.replace(' ', "_").to_lowercase();
        fs::write(format!("/tmp/simple_{}.jpg", safe_name), &jpeg).unwrap();
    }

    // C++ comparison
    println!("\n=== C++ Reference ===\n");
    let cjpegli = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli";

    for (sub, prog) in [("444", "0"), ("420", "0"), ("444", "2"), ("420", "2")] {
        let name = format!("{} p{}", sub, prog);
        let out = format!("/tmp/simple_cpp_{}_{}.jpg", sub, prog);
        let _ = std::process::Command::new(cjpegli)
            .args([png_path, &out, "-q", "90",
                   &format!("--chroma_subsampling={}", sub),
                   &format!("--progressive_level={}", prog)])
            .output();

        if let Ok(jpeg) = fs::read(&out) {
            let decoded = decode_mozjpeg(&jpeg);
            let mut max_diff = 0u8;
            let mut sum_diff = 0u64;
            for (&o, &d) in original.iter().zip(decoded.iter()) {
                let diff = (o as i16 - d as i16).unsigned_abs() as u8;
                max_diff = max_diff.max(diff);
                sum_diff += diff as u64;
            }
            let avg_diff = sum_diff as f64 / original.len() as f64;
            println!("C++ {:<16} {:>8} {:>10} {:>10.2}", name, jpeg.len(), max_diff, avg_diff);
        }
    }
}
