//! Decode both JPEGs and compare coefficient statistics

use std::fs;
use std::io::Cursor;

fn main() {
    let cpp_data = fs::read("/tmp/coeff_cpp.jpg").expect("Run compare_coefficients first");
    let rust_data = fs::read("/tmp/coeff_rust.jpg").expect("Run compare_coefficients first");

    println!("=== Coefficient Statistics ===\n");

    // Use the decoder to get MCU data
    // For now, let's decode and compare YCbCr values before color conversion

    let cpp_decoded = decode_to_ycbcr(&cpp_data);
    let rust_decoded = decode_to_ycbcr(&rust_data);

    if cpp_decoded.is_none() || rust_decoded.is_none() {
        println!("Could not decode to YCbCr. Using RGB comparison instead.");

        let cpp_rgb = decode_to_rgb(&cpp_data).unwrap();
        let rust_rgb = decode_to_rgb(&rust_data).unwrap();

        // Compare per-channel histograms
        for (name, offset) in [("R", 0), ("G", 1), ("B", 2)] {
            let mut cpp_hist = [0usize; 256];
            let mut rust_hist = [0usize; 256];

            for i in (offset..cpp_rgb.len()).step_by(3) {
                cpp_hist[cpp_rgb[i] as usize] += 1;
                rust_hist[rust_rgb[i] as usize] += 1;
            }

            // Find biggest histogram differences
            let mut diffs: Vec<(i32, usize)> = (0..256)
                .map(|v| (cpp_hist[v] as i32 - rust_hist[v] as i32, v))
                .filter(|(d, _)| d.abs() > 100)
                .collect();
            diffs.sort_by_key(|(d, _)| -d.abs());

            println!("{} channel histogram diffs (>100 pixels):", name);
            for (diff, val) in diffs.iter().take(10) {
                let sign = if *diff > 0 {
                    "C++ has more"
                } else {
                    "Rust has more"
                };
                println!("  Value {}: {:+} ({})", val, diff, sign);
            }
            println!();
        }
        return;
    }
}

fn decode_to_ycbcr(_data: &[u8]) -> Option<Vec<u8>> {
    // jpeg-decoder doesn't expose YCbCr directly
    None
}

fn decode_to_rgb(data: &[u8]) -> Option<Vec<u8>> {
    jpeg_decoder::Decoder::new(Cursor::new(data)).decode().ok()
}
