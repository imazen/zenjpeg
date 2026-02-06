//! Compare first block pixels.
use enough::Unstoppable;

use std::process::Command;
use zenjpeg::decode::Decoder;

const TESTIMGARI_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../internal/jpegli-cpp/third_party/libjpeg-turbo/testimages/testimgari.jpg"
);

#[test]
fn compare_first_block_pixels() {
    // Decode with our decoder
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read file");
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&data, Unstoppable)
        .expect("failed to decode");

    let width = 227usize;
    let stride = width * 3;

    // Show first 8x8 block pixels (top-left corner)
    println!("Our decoder - first 8x8 block (RGB):");
    for y in 0..8 {
        print!("  row {}: ", y);
        for x in 0..8 {
            let idx = y * stride + x * 3;
            let r = decoded.data[idx];
            let g = decoded.data[idx + 1];
            let b = decoded.data[idx + 2];
            print!("({:3},{:3},{:3}) ", r, g, b);
        }
        println!();
    }

    // Decode with djpeg
    let output = Command::new("djpeg")
        .args(["-pnm", TESTIMGARI_PATH])
        .output()
        .expect("failed to run djpeg");

    if output.status.success() {
        let ppm = output.stdout;
        // Find RGB data start
        let mut newlines = 0;
        let mut rgb_start = 0;
        for (i, &b) in ppm.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    rgb_start = i + 1;
                    break;
                }
            }
        }
        let ref_rgb = &ppm[rgb_start..];

        println!("\ndjpeg reference - first 8x8 block (RGB):");
        for y in 0..8 {
            print!("  row {}: ", y);
            for x in 0..8 {
                let idx = y * stride + x * 3;
                let r = ref_rgb[idx];
                let g = ref_rgb[idx + 1];
                let b = ref_rgb[idx + 2];
                print!("({:3},{:3},{:3}) ", r, g, b);
            }
            println!();
        }

        // Calculate average pixel values for first block
        let mut our_avg = [0u64; 3];
        let mut ref_avg = [0u64; 3];
        for y in 0..8 {
            for x in 0..8 {
                let idx = y * stride + x * 3;
                for c in 0..3 {
                    our_avg[c] += decoded.data[idx + c] as u64;
                    ref_avg[c] += ref_rgb[idx + c] as u64;
                }
            }
        }
        println!("\nAverage RGB for first 8x8 block:");
        println!(
            "  Ours: R={:.1}, G={:.1}, B={:.1}",
            our_avg[0] as f64 / 64.0,
            our_avg[1] as f64 / 64.0,
            our_avg[2] as f64 / 64.0
        );
        println!(
            "  Ref:  R={:.1}, G={:.1}, B={:.1}",
            ref_avg[0] as f64 / 64.0,
            ref_avg[1] as f64 / 64.0,
            ref_avg[2] as f64 / 64.0
        );
    }
}
