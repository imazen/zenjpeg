//! Minimal test for progressive + subsampling

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;

fn main() {
    let width = 64u32;
    let height = 64u32;

    // Simple gradient
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 4) as u8);
            rgb.push((y * 4) as u8);
            rgb.push(128u8);
        }
    }

    let configs: Vec<(&str, JpegMode, Subsampling)> = vec![
        ("base_444", JpegMode::Baseline, Subsampling::S444),
        ("base_422", JpegMode::Baseline, Subsampling::S422),
        ("base_420", JpegMode::Baseline, Subsampling::S420),
        ("base_440", JpegMode::Baseline, Subsampling::S440),
        ("prog_444", JpegMode::Progressive, Subsampling::S444),
        ("prog_422", JpegMode::Progressive, Subsampling::S422),
        ("prog_420", JpegMode::Progressive, Subsampling::S420),
        ("prog_440", JpegMode::Progressive, Subsampling::S440),
    ];

    println!(
        "{:<12} {:>8} {:>10} {:>10} {:>10}",
        "Config", "Size", "mozjpeg", "jpegli", "zune"
    );
    println!("{:-<55}", "");

    for (name, mode, sub) in &configs {
        let result = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .mode(*mode)
            .subsampling(*sub)
            .optimize_huffman(true)
            .jpegli_quality(Quality::from_quality(85.0))
            .encode(&rgb);

        match result {
            Ok(jpeg) => {
                let path = format!("/tmp/min_{}.jpg", name);
                fs::write(&path, &jpeg).unwrap();

                // Test decoders
                let moz = test_mozjpeg(&jpeg);
                let jpegli_dec = test_jpegli(&jpeg);
                let zune = test_zune(&jpeg);

                println!(
                    "{:<12} {:>8} {:>10} {:>10} {:>10}",
                    name,
                    jpeg.len(),
                    if moz { "OK" } else { "FAIL" },
                    if jpegli_dec { "OK" } else { "FAIL" },
                    if zune { "OK" } else { "FAIL" }
                );

                // If any fail, show details
                if !moz || !jpegli_dec || !zune {
                    println!("  Saved to: {}", path);
                }
            }
            Err(e) => {
                println!("{:<12} ENCODE ERROR: {}", name, e);
            }
        }
    }

    // Now check against C++ jpegli
    println!("\n=== C++ cjpegli comparison ===\n");

    // Save a PNG for cjpegli input
    let png_path = "/tmp/min_input.png";
    {
        let file = fs::File::create(png_path).unwrap();
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&rgb).unwrap();
    }

    let cjpegli = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli";

    let cpp_configs = [
        (
            "cpp_prog_444",
            "--chroma_subsampling=444",
            "--progressive_level=2",
        ),
        (
            "cpp_prog_422",
            "--chroma_subsampling=422",
            "--progressive_level=2",
        ),
        (
            "cpp_prog_420",
            "--chroma_subsampling=420",
            "--progressive_level=2",
        ),
        (
            "cpp_prog_440",
            "--chroma_subsampling=440",
            "--progressive_level=2",
        ),
    ];

    for (name, sub, prog) in &cpp_configs {
        let out_path = format!("/tmp/{}.jpg", name);
        let status = std::process::Command::new(cjpegli)
            .args([png_path, &out_path, "-q", "85", sub, prog])
            .output();

        match status {
            Ok(output) => {
                if output.status.success() {
                    let cpp_jpeg = fs::read(&out_path).unwrap();
                    println!("{:<15} {:>8} bytes", name, cpp_jpeg.len());
                } else {
                    println!(
                        "{:<15} FAILED: {:?}",
                        name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }
            Err(e) => println!("{:<15} ERROR: {}", name, e),
        }
    }
}

fn test_mozjpeg(data: &[u8]) -> bool {
    match mozjpeg::Decompress::new_mem(data) {
        Ok(d) => match d.rgb() {
            Ok(mut dec) => {
                let mut buf = vec![0u8; dec.width() * dec.height() * 3];
                #[allow(deprecated)]
                dec.read_scanlines_into::<u8>(&mut buf).is_ok()
            }
            Err(_) => false,
        },
        Err(_) => false,
    }
}

fn test_jpegli(data: &[u8]) -> bool {
    jpegli::Decoder::new().decode(data).is_ok()
}

fn test_zune(data: &[u8]) -> bool {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    JpegDecoder::new(ZCursor::new(data)).decode().is_ok()
}
