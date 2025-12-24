//! Visual encoding test - produces human-inspectable JPEG outputs.
//!
//! This test encodes jpegli/tests/images/1.png at various quality levels
//! and writes the results to jpegli/tests/outputs/ for visual inspection.

use jpegli::{Encoder, PixelFormat, Quality};
use std::fs;
use std::path::PathBuf;

fn get_test_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("images")
        .join("1.png")
}

fn get_output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("outputs");
    fs::create_dir_all(&dir).ok();
    dir
}

/// Load PNG and convert to RGB bytes
fn load_png_as_rgb(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let png_data = fs::read(path).ok()?;
    let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
    let mut reader = decoder.read_info().ok()?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let width = info.width;
    let height = info.height;

    let rgb_data: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..(width * height * 3) as usize].to_vec(),
        png::ColorType::Rgba => buf[..(width * height * 4) as usize]
            .chunks(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..(width * height) as usize]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..(width * height * 2) as usize]
            .chunks(2)
            .flat_map(|ga| [ga[0], ga[0], ga[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb_data, width, height))
}

#[test]
fn test_visual_encoding_quality_levels() {
    let image_path = get_test_image_path();
    if !image_path.exists() {
        eprintln!("Test image not found: {:?}", image_path);
        return;
    }

    let (rgb_data, width, height) = match load_png_as_rgb(&image_path) {
        Some(data) => data,
        None => {
            eprintln!("Failed to load PNG");
            return;
        }
    };

    let output_dir = get_output_dir();

    println!("Encoding {}x{} image at various quality levels...", width, height);

    // Encode at various quality levels
    let quality_levels = [30, 50, 70, 85, 95];

    for &q in &quality_levels {
        let encoder = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .quality(Quality::from_quality(q as f32));

        match encoder.encode(&rgb_data) {
            Ok(jpeg_data) => {
                let output_path = output_dir.join(format!("1_q{}.jpg", q));
                fs::write(&output_path, &jpeg_data).expect("Failed to write JPEG");

                let file_size = jpeg_data.len();
                let bpp = (file_size as f64 * 8.0) / (width * height) as f64;

                println!(
                    "Q{:2}: {:6} bytes ({:.3} bpp) -> {:?}",
                    q, file_size, bpp, output_path
                );
            }
            Err(e) => {
                eprintln!("Failed to encode at Q{}: {:?}", q, e);
            }
        }
    }

    println!("\nOutput files written to: {:?}", output_dir);
}

#[test]
fn test_visual_encoding_with_aq() {
    let image_path = get_test_image_path();
    if !image_path.exists() {
        eprintln!("Test image not found: {:?}", image_path);
        return;
    }

    let (rgb_data, width, height) = match load_png_as_rgb(&image_path) {
        Some(data) => data,
        None => {
            eprintln!("Failed to load PNG");
            return;
        }
    };

    let output_dir = get_output_dir();

    // Encode at Q70 (good balance of quality and size)
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(70.0));

    match encoder.encode(&rgb_data) {
        Ok(jpeg_data) => {
            let output_path = output_dir.join("1_q70_default.jpg");
            fs::write(&output_path, &jpeg_data).expect("Failed to write JPEG");
            println!(
                "Default encoding: {} bytes -> {:?}",
                jpeg_data.len(),
                output_path
            );
        }
        Err(e) => {
            eprintln!("Failed to encode: {:?}", e);
        }
    }

    println!("\nVisual comparison files written to: {:?}", output_dir);
    println!("Inspect these files to verify encoding quality.");
}
