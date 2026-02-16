//! Single-decode test for valgrind profiling (callgrind/cachegrind)
//!
//! Usage:
//!   cargo build --release --example valgrind_decode
//!   valgrind --tool=callgrind ./target/release/examples/valgrind_decode jpegli 2048
//!   valgrind --tool=callgrind ./target/release/examples/valgrind_decode jpegli 2048 progressive
//!   valgrind --tool=callgrind ./target/release/examples/valgrind_decode zune 2048
//!   kcachegrind callgrind.out.*  # To visualize

use enough::Unstoppable;
use std::env;
use std::hint::black_box;
use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

fn create_test_jpeg(width: u32, height: u32, progressive: bool, no_dri: bool) -> Vec<u8> {
    // Deterministic noise+patches pattern matching decode_compare benchmark.
    // 4 block types: textured, gradient, sharp edges, high-frequency noise.
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let noise = (h >> 24) as u8;
            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / width as usize) as u8;
                    data[idx + 1] = ((y * 255) / height as usize) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255 - edge;
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }
    let mut config =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(progressive);
    if no_dri {
        config = config.restart_mcu_rows(0);
    }
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn decode_jpegli(jpeg_data: &[u8]) {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder
        .decode(jpeg_data, Unstoppable)
        .expect("jpegli decode failed");
    black_box(result);
}

fn decode_zune(jpeg_data: &[u8]) {
    let options = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);
    let cursor = ZCursor::new(jpeg_data);
    let mut decoder = JpegDecoder::new_with_options(cursor, options);
    let result = decoder.decode().expect("zune decode failed");
    black_box(result);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <jpegli|zune|save> [size|file.jpg] [progressive] [save_path]",
            args[0]
        );
        eprintln!("  size: 512 (default), 1024, or 2048");
        eprintln!("  file.jpg: read JPEG from file instead of generating");
        eprintln!("  progressive: add 'progressive' or 'prog' for progressive JPEG");
        eprintln!("  save: write generated JPEG to file (default /tmp/test.jpg)");
        std::process::exit(1);
    }

    let decoder_type = &args[1];
    let arg2 = args.get(2).map(|s| s.as_str()).unwrap_or("512");

    let extra: Vec<&str> = args.iter().skip(3).map(|s| s.as_str()).collect();

    let jpeg_data = if arg2.ends_with(".jpg") || arg2.ends_with(".jpeg") {
        eprintln!("Reading JPEG from {}...", arg2);
        std::fs::read(arg2).expect("failed to read JPEG file")
    } else {
        let size: u32 = arg2.parse().unwrap_or(512);
        let progressive = extra.iter().any(|s| s.starts_with("prog"));
        let no_dri = extra.iter().any(|s| *s == "nodri");
        eprintln!(
            "Creating {}x{} {}{} test JPEG...",
            size,
            size,
            if progressive {
                "progressive"
            } else {
                "baseline"
            },
            if no_dri { " (no DRI)" } else { "" }
        );
        create_test_jpeg(size, size, progressive, no_dri)
    };
    eprintln!("JPEG size: {} bytes", jpeg_data.len());

    // save mode: write JPEG to /tmp and exit (for creating test data)
    if decoder_type == "save" {
        let path = extra
            .iter()
            .find(|s| s.ends_with(".jpg") || s.ends_with(".jpeg"))
            .copied()
            .unwrap_or("/tmp/test.jpg");
        std::fs::write(path, &jpeg_data).unwrap();
        eprintln!("Saved to {}", path);
        return;
    }

    // compare mode: decode two files and compare pixel output
    if decoder_type == "compare" {
        let file_a = args.get(2).expect("need two file paths");
        let file_b = args.get(3).expect("need two file paths");
        let decoder_name = args.get(4).map(|s| s.as_str()).unwrap_or("zune");
        let data_a = std::fs::read(file_a).unwrap();
        let data_b = std::fs::read(file_b).unwrap();
        let (out_a, out_b) = match decoder_name {
            "zune" => {
                let opts = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);
                let opts2 = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);
                let mut da = JpegDecoder::new_with_options(ZCursor::new(&data_a), opts);
                let mut db = JpegDecoder::new_with_options(ZCursor::new(&data_b), opts2);
                (da.decode().unwrap(), db.decode().unwrap())
            }
            "jpegli" => {
                let dec = Decoder::new().output_format(PixelFormat::Rgb);
                let ra = dec.decode(&data_a, Unstoppable).unwrap();
                let rb = dec.decode(&data_b, Unstoppable).unwrap();
                (
                    ra.pixels_u8().unwrap().to_vec(),
                    rb.pixels_u8().unwrap().to_vec(),
                )
            }
            _ => panic!("unknown decoder"),
        };
        eprintln!("A: {} bytes, B: {} bytes", out_a.len(), out_b.len());
        if out_a == out_b {
            eprintln!("IDENTICAL output");
        } else {
            let mut max_diff: u8 = 0;
            let mut diff_count = 0u64;
            let mut sum_abs: u64 = 0;
            for (a, b) in out_a.iter().zip(out_b.iter()) {
                let d = (*a as i16 - *b as i16).unsigned_abs() as u8;
                if d > 0 {
                    diff_count += 1;
                }
                if d > max_diff {
                    max_diff = d;
                }
                sum_abs += d as u64;
            }
            eprintln!(
                "DIFFERENT! max_diff={}, diff_count={}/{} ({:.1}%), mean_abs={:.4}",
                max_diff,
                diff_count,
                out_a.len(),
                diff_count as f64 * 100.0 / out_a.len() as f64,
                sum_abs as f64 / out_a.len() as f64
            );
        }
        return;
    }

    // cross-decoder comparison: compare zune vs jpegli output for same file
    if decoder_type == "crosscheck" {
        let file = args.get(2).expect("need file path");
        let data = std::fs::read(file).unwrap();
        let zune_opts = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut zune_dec = JpegDecoder::new_with_options(ZCursor::new(&data), zune_opts);
        let zune_out = zune_dec.decode().unwrap();
        let dec = Decoder::new().output_format(PixelFormat::Rgb);
        let jpegli_out = dec
            .decode(&data, Unstoppable)
            .unwrap()
            .pixels_u8()
            .unwrap()
            .to_vec();
        eprintln!(
            "zune: {} bytes, jpegli: {} bytes",
            zune_out.len(),
            jpegli_out.len()
        );
        if zune_out == jpegli_out {
            eprintln!("IDENTICAL output between zune and jpegli");
        } else {
            let mut max_diff: u8 = 0;
            let mut diff_count = 0u64;
            let mut sum_abs: u64 = 0;
            for (a, b) in zune_out.iter().zip(jpegli_out.iter()) {
                let d = (*a as i16 - *b as i16).unsigned_abs() as u8;
                if d > 0 {
                    diff_count += 1;
                }
                if d > max_diff {
                    max_diff = d;
                }
                sum_abs += d as u64;
            }
            eprintln!(
                "DIFFERENT! max_diff={}, diff_count={}/{} ({:.1}%), mean_abs={:.4}",
                max_diff,
                diff_count,
                zune_out.len(),
                diff_count as f64 * 100.0 / zune_out.len() as f64,
                sum_abs as f64 / zune_out.len() as f64
            );
        }
        return;
    }

    eprintln!("Decoding with {}...", decoder_type);

    match decoder_type.as_str() {
        "jpegli" => decode_jpegli(&jpeg_data),
        "zune" => decode_zune(&jpeg_data),
        _ => {
            eprintln!("Unknown decoder: {}. Use 'jpegli' or 'zune'", decoder_type);
            std::process::exit(1);
        }
    }

    eprintln!("Done.");
}
