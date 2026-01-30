//! Profile encoding pipeline for comparison with C++ jpegli.
//!
//! Encodes a PPM image N times using the streaming encoder.
//! Designed for use with callgrind, cachegrind, and heaptrack.
//!
//! Usage:
//!   cargo build --release -p zenjpeg --features test-utils --example profile_pipeline
//!   valgrind --tool=callgrind ./target/release/examples/profile_pipeline /tmp/test_profile.ppm 5
//!   valgrind --tool=cachegrind ./target/release/examples/profile_pipeline /tmp/test_profile.ppm 5
//!   heaptrack ./target/release/examples/profile_pipeline /tmp/test_profile.ppm 5

use std::env;
use std::fs;

use zenjpeg::encode::{Quality, StreamingEncoder};
use zenjpeg::types::Subsampling;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.ppm> [iterations]", args[0]);
        std::process::exit(1);
    }

    let ppm_path = &args[1];
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let (w, h, pixels) = load_ppm(ppm_path).expect("Failed to load PPM");
    eprintln!("Loaded {}x{} image ({} bytes)", w, h, pixels.len());
    eprintln!("Encoding {} iterations, Q85, 4:2:0, optimize_huffman=true", iterations);

    let stride = w as usize * 3;

    for i in 0..iterations {
        let mut encoder = StreamingEncoder::new(w, h)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .optimize_huffman(true)
            .start()
            .expect("Failed to start encoder");

        for y in 0..h {
            let row_start = y as usize * stride;
            let row_end = row_start + stride;
            encoder
                .push_rows(&pixels[row_start..row_end], 1)
                .expect("Failed to push row");
        }

        let result = encoder.finish().expect("Failed to finish");
        if i == 0 {
            eprintln!("Output size: {} bytes", result.len());
        }
        std::hint::black_box(&result);
    }

    eprintln!("Done.");
}

/// Load a binary PPM (P6) file.
fn load_ppm(path: &str) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let mut pos = 0;

    // Parse header
    if data.len() < 3 || data[0] != b'P' || data[1] != b'6' {
        return Err("Not a P6 PPM file".into());
    }
    pos = 2;

    // Skip whitespace and comments
    let skip_ws = |data: &[u8], mut p: usize| -> usize {
        loop {
            while p < data.len() && data[p].is_ascii_whitespace() {
                p += 1;
            }
            if p < data.len() && data[p] == b'#' {
                while p < data.len() && data[p] != b'\n' {
                    p += 1;
                }
            } else {
                break;
            }
        }
        p
    };

    let read_int = |data: &[u8], mut p: usize| -> (usize, usize) {
        let start = p;
        while p < data.len() && data[p].is_ascii_digit() {
            p += 1;
        }
        let val: usize = std::str::from_utf8(&data[start..p])
            .unwrap()
            .parse()
            .unwrap();
        (val, p)
    };

    pos = skip_ws(&data, pos);
    let (width, p) = read_int(&data, pos);
    pos = skip_ws(&data, p);
    let (height, p) = read_int(&data, pos);
    pos = skip_ws(&data, p);
    let (maxval, p) = read_int(&data, pos);
    pos = p;

    if maxval != 255 {
        return Err(format!("Unsupported maxval: {}", maxval).into());
    }

    // Skip exactly one whitespace character after maxval
    pos += 1;

    let pixel_data = &data[pos..];
    let expected = width * height * 3;
    if pixel_data.len() < expected {
        return Err(format!(
            "Not enough pixel data: {} < {}",
            pixel_data.len(),
            expected
        )
        .into());
    }

    Ok((width as u32, height as u32, pixel_data[..expected].to_vec()))
}
