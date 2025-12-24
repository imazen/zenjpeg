//! Test DCT precision against known reference values

use jpegli::dct;
use jpegli::idct;

fn main() {
    // Standard JPEG test block: all 128 (after level shift = all 0)
    let uniform_block = [0.0f32; 64];
    let uniform_dct = dct::forward_dct_blocks(&[uniform_block])[0];
    println!("Uniform (0) block DCT[0] = {} (expected 0)", uniform_dct[0]);

    // All 1 block - DC should be 8 (sum of 64 ones scaled)
    let ones_block = [1.0f32; 64];
    let ones_dct = dct::forward_dct_blocks(&[ones_block])[0];
    println!("Ones block DCT[0] = {:.4} (expected 8.0)", ones_dct[0]);

    // Simple horizontal gradient (tests AC coefficients)
    let mut gradient_block = [0.0f32; 64];
    for i in 0..64 {
        gradient_block[i] = (i % 8) as f32 - 3.5; // -3.5 to 3.5
    }
    let gradient_dct = dct::forward_dct_blocks(&[gradient_block])[0];
    println!(
        "H-gradient DCT (first row): {:?}",
        gradient_dct[..8]
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );

    // The key test: round-trip should preserve coefficients exactly
    let test_block: [f32; 64] = [
        -76.0, -73.0, -67.0, -62.0, -58.0, -67.0, -64.0, -55.0, -65.0, -69.0, -73.0, -38.0, -19.0,
        -43.0, -59.0, -56.0, -66.0, -69.0, -60.0, -15.0, 16.0, -24.0, -62.0, -55.0, -65.0, -70.0,
        -57.0, -6.0, 26.0, -22.0, -58.0, -59.0, -61.0, -67.0, -60.0, -24.0, -2.0, -40.0, -60.0,
        -58.0, -49.0, -63.0, -68.0, -58.0, -51.0, -60.0, -70.0, -53.0, -43.0, -57.0, -64.0, -69.0,
        -73.0, -67.0, -63.0, -45.0, -41.0, -49.0, -59.0, -60.0, -63.0, -52.0, -50.0, -34.0,
    ];

    let test_dct = dct::forward_dct_blocks(&[test_block])[0];
    println!("\nTest block DCT (first row):");
    for i in 0..8 {
        print!("{:8.2} ", test_dct[i]);
    }
    println!();

    // IDCT and back
    let reconstructed = idct::inverse_dct_blocks(&[test_dct])[0];

    let mut max_error = 0.0f32;
    for i in 0..64 {
        let error = (reconstructed[i] - test_block[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }
    println!("\nRound-trip max error: {:.6}", max_error);

    // Compare with known JPEG DCT values (from libjpeg reference)
    // For the test block above, the expected DC is approximately -415
    println!("\nExpected DC ≈ -415.0, got {:.2}", test_dct[0]);

    // Check DCT scaling factor
    // JPEG DCT uses 1/8 scaling for forward, 1 for inverse
    let one_in_corner = [0.0f32; 64]
        .iter()
        .enumerate()
        .map(|(i, _)| if i == 0 { 8.0 } else { 0.0 })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let one_corner_dct: [f32; 64] = dct::forward_dct_blocks(&[one_in_corner])[0];
    println!(
        "\n8.0 in position [0,0] gives DC = {:.4}",
        one_corner_dct[0]
    );
}
