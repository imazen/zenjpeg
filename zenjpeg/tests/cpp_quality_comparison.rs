//! Compare Rust vs C++ progressive encoding quality and size.

use std::fs;
use std::process::Command;
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

fn encode_rgb_progressive(width: u32, height: u32, data: &[u8], quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .quality(quality)
        .progressive(true)
        .optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push data");
    enc.finish().expect("finish")
}

/// Compare Rust vs C++ at various quality levels.
#[test]
fn test_cpp_quality_comparison() {
    let width = 128u32;
    let height = 128u32;

    // Photo-like content
    let data: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let r = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8;
                let g = ((x.wrapping_mul(13) ^ y.wrapping_mul(23)) % 256) as u8;
                let b = ((x.wrapping_mul(11) ^ y.wrapping_mul(19)) % 256) as u8;
                [r, g, b]
            })
        })
        .collect();

    // Save as PNG for C++ input
    let tmp_dir = std::env::temp_dir();
    let png_path = tmp_dir.join("zenjpeg_test_input.png");
    {
        let file = fs::File::create(&png_path).unwrap();
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&data).unwrap();
    }

    // Check cjpegli exists
    let cjpegli_path = match zenjpeg::test_utils::find_cjpegli() {
        Some(p) => p,
        None => {
            println!("Skipping: cjpegli not found. Set CJPEGLI_PATH env var.");
            return;
        }
    };

    let dssim = dssim_core::Dssim::new();
    let orig_pixels: Vec<rgb::RGB<u8>> = data
        .chunks(3)
        .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
        .collect();
    let orig_img = dssim
        .create_image_rgb(&orig_pixels, width as usize, height as usize)
        .unwrap();

    println!();
    println!(
        "{:>5} {:>12} {:>12} {:>10} {:>10} {:>8}",
        "Q", "Rust Size", "C++ Size", "Rust DSSIM", "C++ DSSIM", "Size Δ"
    );
    println!(
        "{:-<5} {:-<12} {:-<12} {:-<10} {:-<10} {:-<8}",
        "", "", "", "", "", ""
    );

    for q in [3, 10, 30, 50, 70, 80, 85, 95] {
        // Rust encoding (progressive + optimized)
        let rust_jpeg = encode_rgb_progressive(width, height, &data, q as f32);
        let rust_size = rust_jpeg.len();

        // Decode Rust JPEG for DSSIM
        let rust_decoded = decode_zune(&rust_jpeg[..]).unwrap();
        let rust_pixels: Vec<rgb::RGB<u8>> = rust_decoded
            .chunks(3)
            .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
            .collect();
        let rust_img = dssim
            .create_image_rgb(&rust_pixels, width as usize, height as usize)
            .unwrap();
        let (rust_dssim, _) = dssim.compare(&orig_img, rust_img);

        // C++ encoding with progressive level 2
        let cpp_out = tmp_dir.join(format!("zenjpeg_cpp_q{}.jpg", q));
        let status = Command::new(&cjpegli_path)
            .args([
                png_path.to_str().unwrap(),
                cpp_out.to_str().unwrap(),
                "-q",
                &q.to_string(),
                "--progressive_level=2",
            ])
            .output()
            .expect("Failed to run cjpegli");

        if !status.status.success() {
            println!(
                "Q{}: C++ encoding failed: {}",
                q,
                String::from_utf8_lossy(&status.stderr)
            );
            continue;
        }

        let cpp_jpeg = fs::read(cpp_out).unwrap();
        let cpp_size = cpp_jpeg.len();

        // Decode C++ JPEG for DSSIM
        let cpp_decoded = decode_zune(&cpp_jpeg[..]).unwrap();
        let cpp_pixels: Vec<rgb::RGB<u8>> = cpp_decoded
            .chunks(3)
            .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
            .collect();
        let cpp_img = dssim
            .create_image_rgb(&cpp_pixels, width as usize, height as usize)
            .unwrap();
        let (cpp_dssim, _) = dssim.compare(&orig_img, cpp_img);

        let size_diff = (rust_size as f64 - cpp_size as f64) / cpp_size as f64 * 100.0;

        println!(
            "{:>5} {:>12} {:>12} {:>10.6} {:>10.6} {:>+7.1}%",
            q, rust_size, cpp_size, rust_dssim, cpp_dssim, size_diff
        );
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
