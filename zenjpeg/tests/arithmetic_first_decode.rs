#![cfg(feature = "ffi-tests")]
//! Trace the first few arithmetic decode operations.
use enough::Unstoppable;

use zenjpeg::decode::Decoder;

const TESTIMGARI_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../internal/jpegli-cpp/third_party/libjpeg-turbo/testimages/testimgari.jpg"
);

#[test]
fn trace_first_decode() {
    // Read the JPEG file and extract scan data
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read file");

    // Find SOS marker (ffda) and get scan data offset
    let mut sos_pos = None;
    for i in 0..data.len() - 1 {
        if data[i] == 0xff && data[i + 1] == 0xda {
            sos_pos = Some(i);
            break;
        }
    }
    let sos_pos = sos_pos.expect("SOS marker not found");
    println!("SOS marker at offset {:#x}", sos_pos);

    // SOS length is at sos_pos + 2
    let sos_len = ((data[sos_pos + 2] as usize) << 8) | (data[sos_pos + 3] as usize);
    println!("SOS length: {}", sos_len);

    // Scan data starts after SOS marker + length
    let scan_start = sos_pos + 2 + sos_len;
    println!("Scan data starts at offset {:#x}", scan_start);

    // Show first 20 bytes of scan data
    println!("First 20 bytes of scan data:");
    for i in 0..20.min(data.len() - scan_start) {
        print!("{:02x} ", data[scan_start + i]);
    }
    println!();

    // Now decode the whole image and show first DC values
    let decoder = Decoder::new();
    let coeffs = decoder
        .decode_coefficients(&data, Unstoppable)
        .expect("failed to decode");

    // First 10 Y DC values
    println!("\nFirst 10 Y DC values:");
    for i in 0..10 {
        let dc = coeffs.components[0].coeffs[i * 64];
        println!("  Block {}: DC = {}", i, dc);
    }

    // Also show a few AC coefficients from first block to see if those look reasonable
    println!("\nFirst block AC coefficients (showing first 10 after DC):");
    for i in 1..11 {
        let ac = coeffs.components[0].coeffs[i];
        if ac != 0 {
            println!("  coeff[{}] = {}", i, ac);
        }
    }

    // Check if there's any non-zero AC in first block
    let first_block = &coeffs.components[0].coeffs[0..64];
    let nonzero_count = first_block.iter().filter(|&&x| x != 0).count();
    println!(
        "\nFirst Y block has {} non-zero coefficients",
        nonzero_count
    );
}
