//! Compare decoded pixels from Rust vs C++ progressive + subsampling

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;

fn main() {
    let width = 64u32;
    let height = 64u32;

    // Simple gradient
    let mut original = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            original.push((x * 4) as u8);
            original.push((y * 4) as u8);
            original.push(128u8);
        }
    }

    // Test progressive 420
    let rust_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .mode(JpegMode::Progressive)
        .subsampling(Subsampling::S420)
        .optimize_huffman(true)
        .jpegli_quality(Quality::from_quality(85.0))
        .encode(&original)
        .expect("encode");

    println!("Rust progressive 420: {} bytes", rust_jpeg.len());

    // Decode with different decoders
    let moz_decoded = decode_mozjpeg(&rust_jpeg);
    let jpegli_decoded = decode_jpegli(&rust_jpeg);
    let zune_decoded = decode_zune(&rust_jpeg);

    println!("\nDecoded sizes:");
    println!("  Original: {} pixels", original.len() / 3);
    println!("  mozjpeg:  {} pixels", moz_decoded.len() / 3);
    println!("  jpegli:   {} pixels", jpegli_decoded.len() / 3);
    println!("  zune:     {} pixels", zune_decoded.len() / 3);

    // Check if decoded matches original reasonably
    let check_decode = |name: &str, decoded: &[u8]| {
        if decoded.len() != original.len() {
            println!(
                "{}: SIZE MISMATCH {} vs {}",
                name,
                decoded.len(),
                original.len()
            );
            return;
        }

        let mut max_diff = 0u8;
        let mut sum_diff = 0u64;
        for (i, (&o, &d)) in original.iter().zip(decoded.iter()).enumerate() {
            let diff = (o as i16 - d as i16).unsigned_abs() as u8;
            if diff > max_diff {
                max_diff = diff;
            }
            sum_diff += diff as u64;
        }
        let avg_diff = sum_diff as f64 / original.len() as f64;
        println!("{}: max_diff={}, avg_diff={:.2}", name, max_diff, avg_diff);

        // Sample some pixels
        println!("  Sample pixels (original vs decoded):");
        for i in [0, 100, 500, 1000, 2000, 3000].iter() {
            if *i < original.len() / 3 {
                let idx = i * 3;
                println!(
                    "    [{:4}] ({},{},{}) vs ({},{},{})",
                    i,
                    original[idx],
                    original[idx + 1],
                    original[idx + 2],
                    decoded[idx],
                    decoded[idx + 1],
                    decoded[idx + 2]
                );
            }
        }
    };

    println!("\nDecode quality check:");
    check_decode("mozjpeg", &moz_decoded);
    check_decode("jpegli", &jpegli_decoded);
    check_decode("zune", &zune_decoded);

    // Now compare with C++ jpegli
    println!("\n=== C++ comparison ===");

    let png_path = "/tmp/decode_test_input.png";
    {
        let file = fs::File::create(png_path).unwrap();
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&original).unwrap();
    }

    let cpp_path = "/tmp/decode_test_cpp.jpg";
    let cjpegli = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli";

    let _ = std::process::Command::new(cjpegli)
        .args([
            png_path,
            cpp_path,
            "-q",
            "85",
            "--chroma_subsampling=420",
            "--progressive_level=2",
        ])
        .output();

    if let Ok(cpp_jpeg) = fs::read(cpp_path) {
        println!("C++ progressive 420: {} bytes", cpp_jpeg.len());
        let cpp_decoded = decode_mozjpeg(&cpp_jpeg);
        check_decode("C++ (mozjpeg)", &cpp_decoded);
    }
}

fn decode_mozjpeg(data: &[u8]) -> Vec<u8> {
    let d = mozjpeg::Decompress::new_mem(data).unwrap();
    let mut dec = d.rgb().unwrap();
    let mut buf = vec![0u8; dec.width() * dec.height() * 3];
    #[allow(deprecated)]
    let _ = dec.read_scanlines_into::<u8>(&mut buf);
    buf
}

fn decode_jpegli(data: &[u8]) -> Vec<u8> {
    jpegli::Decoder::new().decode(data).unwrap().data
}

fn decode_zune(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    JpegDecoder::new(ZCursor::new(data)).decode().unwrap()
}
