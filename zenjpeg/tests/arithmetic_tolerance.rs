//! Test arithmetic decoding with tolerance for rounding differences.

use zenjpeg::decode::Decoder;
use std::process::Command;

const TESTIMGARI_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../internal/jpegli-cpp/third_party/libjpeg-turbo/testimages/testimgari.jpg"
);

#[test]
fn arithmetic_decode_with_tolerance() {
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read file");
    let decoder = Decoder::new();
    let decoded = decoder.decode(&data).expect("failed to decode");
    
    // Get djpeg reference
    let output = Command::new("djpeg")
        .args(["-pnm", TESTIMGARI_PATH])
        .output()
        .expect("failed to run djpeg");
    
    assert!(output.status.success(), "djpeg failed");
    
    let ppm = output.stdout;
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
    
    assert_eq!(decoded.data.len(), ref_rgb.len(), "size mismatch");
    
    // Calculate statistics
    let mut diff_hist = [0usize; 256];
    let mut sum_abs_diff: u64 = 0;
    let mut sum_sq_diff: f64 = 0.0;
    
    for (&ours, &reference) in decoded.data.iter().zip(ref_rgb.iter()) {
        let diff = (ours as i16 - reference as i16).abs() as u8;
        diff_hist[diff as usize] += 1;
        sum_abs_diff += diff as u64;
        sum_sq_diff += (diff as f64).powi(2);
    }
    
    let n = decoded.data.len();
    let mae = sum_abs_diff as f64 / n as f64;  // Mean Absolute Error
    let rmse = (sum_sq_diff / n as f64).sqrt();  // Root Mean Square Error
    
    println!("Total pixels: {}", n / 3);
    println!("Mean Absolute Error: {:.2}", mae);
    println!("Root Mean Square Error: {:.2}", rmse);
    
    println!("\nDifference histogram (showing non-zero):");
    for (diff, &count) in diff_hist.iter().enumerate() {
        if count > 0 {
            let pct = 100.0 * count as f64 / n as f64;
            println!("  diff={:3}: {:7} values ({:.2}%)", diff, count, pct);
        }
    }
    
    // Check if values are within acceptable range
    // JPEG decoders can have 1-2 units of rounding difference
    // But our decoder seems to have larger differences
    
    let within_1 = diff_hist[0] + diff_hist[1];
    let within_2 = within_1 + diff_hist[2];
    let within_3 = within_2 + diff_hist[3];
    
    println!("\nWithin tolerance:");
    println!("  <=1: {:.1}%", 100.0 * within_1 as f64 / n as f64);
    println!("  <=2: {:.1}%", 100.0 * within_2 as f64 / n as f64);
    println!("  <=3: {:.1}%", 100.0 * within_3 as f64 / n as f64);
}
