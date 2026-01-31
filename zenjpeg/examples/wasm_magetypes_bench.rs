//! Benchmark comparing wide crate vs magetypes on WASM SIMD128
//!
//! Run with:
//! ```sh
//! CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --wasm simd" \
//! RUSTFLAGS="-C target-feature=+simd128" \
//! cargo run --release -p zenjpeg --example wasm_magetypes_bench \
//!     --target wasm32-wasip1 --no-default-features --features "std,archmage-simd"
//! ```

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod bench {
    use archmage::SimdToken; // Required for try_new()
    use std::time::{Duration, Instant};

    // Get archmage token for WASM
    fn get_token() -> archmage::Simd128Token {
        archmage::Simd128Token::try_new().expect("SIMD128 not available")
    }

    /// Benchmark using wide crate f32x4
    pub fn bench_wide_arithmetic(iterations: usize) -> Duration {
        use wide::f32x4;

        let a = f32x4::new([1.0, 2.0, 3.0, 4.0]);
        let b = f32x4::new([5.0, 6.0, 7.0, 8.0]);
        let c = f32x4::new([0.5, 0.5, 0.5, 0.5]);

        let start = Instant::now();
        for _ in 0..iterations {
            // Typical DCT-like operations: add, sub, mul
            let t1 = a + b;
            let t2 = a - b;
            let t3 = t1 * c;
            let t4 = t2 * c;
            let result = t3 + t4;
            std::hint::black_box(&result);
        }
        start.elapsed()
    }

    /// Benchmark using magetypes f32x4
    pub fn bench_magetypes_arithmetic(iterations: usize) -> Duration {
        use magetypes::simd::f32x4;

        let token = get_token();
        let a = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
        let b = f32x4::from_array(token, [5.0, 6.0, 7.0, 8.0]);
        let c = f32x4::from_array(token, [0.5, 0.5, 0.5, 0.5]);

        let start = Instant::now();
        for _ in 0..iterations {
            // Same operations as wide
            let t1 = a + b;
            let t2 = a - b;
            let t3 = t1 * c;
            let t4 = t2 * c;
            let result = t3 + t4;
            std::hint::black_box(&result);
        }
        start.elapsed()
    }

    /// Benchmark wide 8x8 transpose
    pub fn bench_wide_transpose(iterations: usize) -> Duration {
        use wide::f32x8;

        let rows = [
            f32x8::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            f32x8::new([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            f32x8::new([16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]),
            f32x8::new([24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]),
            f32x8::new([32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]),
            f32x8::new([40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0]),
            f32x8::new([48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]),
            f32x8::new([56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0]),
        ];

        let start = Instant::now();
        for _ in 0..iterations {
            let transposed = f32x8::transpose(rows);
            std::hint::black_box(&transposed);
        }
        start.elapsed()
    }

    /// Benchmark magetypes 4x4 transpose (WASM native)
    pub fn bench_magetypes_transpose_4x4(iterations: usize) -> Duration {
        use core::arch::wasm32::*;
        use magetypes::simd::f32x4;

        let token = get_token();
        let r0 = f32x4::from_array(token, [0.0, 1.0, 2.0, 3.0]);
        let r1 = f32x4::from_array(token, [4.0, 5.0, 6.0, 7.0]);
        let r2 = f32x4::from_array(token, [8.0, 9.0, 10.0, 11.0]);
        let r3 = f32x4::from_array(token, [12.0, 13.0, 14.0, 15.0]);

        let start = Instant::now();
        for _ in 0..iterations {
            // 4x4 transpose using i32x4_shuffle
            let a0 = unsafe { f32x4::from_raw(i32x4_shuffle::<0, 4, 1, 5>(r0.raw(), r1.raw())) };
            let a1 = unsafe { f32x4::from_raw(i32x4_shuffle::<2, 6, 3, 7>(r0.raw(), r1.raw())) };
            let a2 = unsafe { f32x4::from_raw(i32x4_shuffle::<0, 4, 1, 5>(r2.raw(), r3.raw())) };
            let a3 = unsafe { f32x4::from_raw(i32x4_shuffle::<2, 6, 3, 7>(r2.raw(), r3.raw())) };

            let t0 = unsafe { f32x4::from_raw(i32x4_shuffle::<0, 1, 4, 5>(a0.raw(), a2.raw())) };
            let t1 = unsafe { f32x4::from_raw(i32x4_shuffle::<2, 3, 6, 7>(a0.raw(), a2.raw())) };
            let t2 = unsafe { f32x4::from_raw(i32x4_shuffle::<0, 1, 4, 5>(a1.raw(), a3.raw())) };
            let t3 = unsafe { f32x4::from_raw(i32x4_shuffle::<2, 3, 6, 7>(a1.raw(), a3.raw())) };

            std::hint::black_box(&(t0, t1, t2, t3));
        }
        start.elapsed()
    }

    /// Benchmark log2 (transcendental - where magetypes shines)
    pub fn bench_wide_log2(iterations: usize) -> Duration {
        use wide::f32x4;

        let a = f32x4::new([1.0, 2.0, 4.0, 8.0]);

        let start = Instant::now();
        for _ in 0..iterations {
            // wide doesn't have log2, use ln and convert
            let result = a.ln() * f32x4::splat(core::f32::consts::LOG2_E);
            std::hint::black_box(&result);
        }
        start.elapsed()
    }

    /// Benchmark magetypes log2 (native implementation)
    pub fn bench_magetypes_log2(iterations: usize) -> Duration {
        use magetypes::simd::f32x4;

        let token = get_token();
        let a = f32x4::from_array(token, [1.0, 2.0, 4.0, 8.0]);

        let start = Instant::now();
        for _ in 0..iterations {
            let result = a.log2_lowp();
            std::hint::black_box(&result);
        }
        start.elapsed()
    }
}

fn main() {
    println!("WASM SIMD128: wide vs magetypes Benchmark");
    println!("==========================================\n");

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        println!("This benchmark requires WASM SIMD128.");
        println!("Build with: RUSTFLAGS=\"-C target-feature=+simd128\"");
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let iterations = 1_000_000;

        println!("Iterations: {}\n", iterations);

        // Arithmetic benchmark
        println!("--- Arithmetic (add, sub, mul chain) ---");
        let wide_time = bench::bench_wide_arithmetic(iterations);
        let mage_time = bench::bench_magetypes_arithmetic(iterations);
        println!(
            "wide:      {:?} ({:.2} ns/op)",
            wide_time,
            wide_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "magetypes: {:?} ({:.2} ns/op)",
            mage_time,
            mage_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "Ratio: {:.2}x ({})",
            wide_time.as_nanos() as f64 / mage_time.as_nanos() as f64,
            if wide_time > mage_time {
                "magetypes faster"
            } else {
                "wide faster"
            }
        );

        // Transpose benchmark
        println!("\n--- Transpose ---");
        let wide_trans = bench::bench_wide_transpose(iterations);
        let mage_trans = bench::bench_magetypes_transpose_4x4(iterations);
        println!(
            "wide 8x8:       {:?} ({:.2} ns/op)",
            wide_trans,
            wide_trans.as_nanos() as f64 / iterations as f64
        );
        println!(
            "magetypes 4x4:  {:?} ({:.2} ns/op)",
            mage_trans,
            mage_trans.as_nanos() as f64 / iterations as f64
        );
        // Note: 8x8 vs 4x4 isn't directly comparable, but shows native SIMD shuffle

        // Log2 benchmark
        println!("\n--- Transcendental (log2) ---");
        let wide_log = bench::bench_wide_log2(iterations);
        let mage_log = bench::bench_magetypes_log2(iterations);
        println!(
            "wide (ln * LOG2_E): {:?} ({:.2} ns/op)",
            wide_log,
            wide_log.as_nanos() as f64 / iterations as f64
        );
        println!(
            "magetypes log2:     {:?} ({:.2} ns/op)",
            mage_log,
            mage_log.as_nanos() as f64 / iterations as f64
        );
        println!(
            "Ratio: {:.2}x ({})",
            wide_log.as_nanos() as f64 / mage_log.as_nanos() as f64,
            if wide_log > mage_log {
                "magetypes faster"
            } else {
                "wide faster"
            }
        );
    }
}
