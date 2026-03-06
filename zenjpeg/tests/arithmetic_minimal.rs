//! Minimal arithmetic decode test with reference comparison.
use enough::Unstoppable;

use std::process::Command;

const TESTIMGARI_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../internal/jpegli-cpp/third_party/libjpeg-turbo/testimages/testimgari.jpg"
);

#[test]
fn check_ac_in_first_blocks() {
    // Use jpegtran to get coefficient information
    let output = Command::new("djpeg").args(["-v", TESTIMGARI_PATH]).output();

    if let Ok(out) = output {
        println!("djpeg verbose output:");
        println!("{}", String::from_utf8_lossy(&out.stderr));
    }

    // Check with our decoder what AC coefficients we get
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read file");
    let decoder = zenjpeg::decode::Decoder::new();
    let coeffs = decoder
        .decode_coefficients(&data, Unstoppable)
        .expect("failed to decode");

    // First 5 Y blocks - show all non-zero coefficients
    println!("\nOur decoder - First 5 Y blocks:");
    for blk in 0..5 {
        let block = &coeffs.components[0].coeffs[blk * 64..(blk + 1) * 64];
        let nonzero: Vec<(usize, i16)> = block
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, &v)| (i, v))
            .collect();
        println!("  Block {}: {:?}", blk, nonzero);
    }

    // Check Cb/Cr blocks too
    println!("\nFirst 3 Cb blocks:");
    for blk in 0..3 {
        let block = &coeffs.components[1].coeffs[blk * 64..(blk + 1) * 64];
        let nonzero: Vec<(usize, i16)> = block
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, &v)| (i, v))
            .collect();
        println!("  Block {}: {:?}", blk, nonzero);
    }

    println!("\nFirst 3 Cr blocks:");
    for blk in 0..3 {
        let block = &coeffs.components[2].coeffs[blk * 64..(blk + 1) * 64];
        let nonzero: Vec<(usize, i16)> = block
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, &v)| (i, v))
            .collect();
        println!("  Block {}: {:?}", blk, nonzero);
    }
}
