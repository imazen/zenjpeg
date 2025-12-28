use jpegli::{decode::Decoder, PixelFormat};
use std::fs;

fn main() {
    // Read the JPEG that was encoded by cjpegli
    let jpeg = fs::read("/tmp/test_decode.jpg").expect("read jpeg");

    // Decode with our decoder
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let decoded = decoder.decode(&jpeg).expect("decode");

    // Read djpegli output
    let djpegli_pixels = fs::read("/tmp/decoder_test/djpegli_pixels.bin").expect("read djpegli");

    println!(
        "Our decoder: {}x{}, {} bytes",
        decoded.width,
        decoded.height,
        decoded.data.len()
    );
    println!(
        "djpegli:     {}x{}, {} bytes",
        512,
        512,
        djpegli_pixels.len()
    );

    // Compare
    let mut diffs = 0;
    let mut max_diff = 0i32;
    let mut sum_diff = 0i64;

    for (&a, &b) in decoded.data.iter().zip(djpegli_pixels.iter()) {
        let diff = (a as i32 - b as i32).abs();
        if diff > 0 {
            diffs += 1;
            sum_diff += diff as i64;
        }
        max_diff = max_diff.max(diff);
    }

    println!("\nComparison with djpegli:");
    println!(
        "Pixels with differences: {} / {} ({:.2}%)",
        diffs,
        decoded.data.len(),
        100.0 * diffs as f64 / decoded.data.len() as f64
    );
    println!("Max difference: {}", max_diff);
    if diffs > 0 {
        println!(
            "Avg difference (of non-zero): {:.2}",
            sum_diff as f64 / diffs as f64
        );
    }

    // Show first few differing pixels
    println!("\nFirst 5 differing pixel components:");
    let mut shown = 0;
    for (i, (&a, &b)) in decoded.data.iter().zip(djpegli_pixels.iter()).enumerate() {
        if a != b && shown < 5 {
            let px = i / 3;
            let ch = i % 3;
            let channel = ["R", "G", "B"][ch];
            println!(
                "  Pixel {} {}: ours={}, djpegli={}, diff={}",
                px,
                channel,
                a,
                b,
                a as i32 - b as i32
            );
            shown += 1;
        }
    }
}
