//! Quick benchmark: AVX-512 dual-block vs AVX2 single-block DCT

use std::time::Instant;

fn main() {
    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    {
        use archmage::{Avx512fToken, Desktop64, SimdToken};
        use jpegli::encode::mage_simd::{mage_forward_dct_8x8, mage_forward_dct_8x8_dual};

        const BLOCKS: usize = 100_000;
        const ITERS: usize = 10;

        // Test data
        let input: Vec<[f32; 64]> = (0..BLOCKS * 2)
            .map(|i| core::array::from_fn(|j| ((i * 64 + j) % 256) as f32))
            .collect();

        // AVX2 single-block
        if let Some(token) = Desktop64::summon() {
            let mut outputs: Vec<[f32; 64]> = vec![[0.0; 64]; BLOCKS * 2];

            // Warmup
            for i in 0..BLOCKS * 2 {
                mage_forward_dct_8x8(token, &input[i], &mut outputs[i]);
            }

            let start = Instant::now();
            for _ in 0..ITERS {
                for i in 0..BLOCKS * 2 {
                    mage_forward_dct_8x8(token, &input[i], &mut outputs[i]);
                }
            }
            let elapsed = start.elapsed();
            let blocks_per_sec = (BLOCKS * 2 * ITERS) as f64 / elapsed.as_secs_f64();
            println!(
                "AVX2 single-block: {:.2}M blocks/sec ({:.2}ms for {}K blocks)",
                blocks_per_sec / 1e6,
                elapsed.as_secs_f64() * 1000.0 / ITERS as f64,
                BLOCKS * 2 / 1000
            );
        }

        // AVX-512 dual-block
        if let Some(token) = Avx512fToken::try_new() {
            let mut outputs: Vec<[f32; 64]> = vec![[0.0; 64]; BLOCKS * 2];

            // Warmup
            for i in (0..BLOCKS * 2).step_by(2) {
                let (left, right) = outputs.split_at_mut(i + 1);
                mage_forward_dct_8x8_dual(
                    token,
                    &input[i],
                    &input[i + 1],
                    &mut left[i],
                    &mut right[0],
                );
            }

            let start = Instant::now();
            for _ in 0..ITERS {
                for i in (0..BLOCKS * 2).step_by(2) {
                    let (left, right) = outputs.split_at_mut(i + 1);
                    mage_forward_dct_8x8_dual(
                        token,
                        &input[i],
                        &input[i + 1],
                        &mut left[i],
                        &mut right[0],
                    );
                }
            }
            let elapsed = start.elapsed();
            let blocks_per_sec = (BLOCKS * 2 * ITERS) as f64 / elapsed.as_secs_f64();
            println!(
                "AVX-512 dual-block: {:.2}M blocks/sec ({:.2}ms for {}K blocks)",
                blocks_per_sec / 1e6,
                elapsed.as_secs_f64() * 1000.0 / ITERS as f64,
                BLOCKS * 2 / 1000
            );
        } else {
            println!("AVX-512 not available on this CPU");
        }
    }

    #[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
    println!("Requires archmage-simd feature on x86_64");
}
