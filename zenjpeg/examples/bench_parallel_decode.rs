//! Benchmark parallel output pass for JPEG decode.
//!
//! Compares serial vs parallel output (IDCT + upsample + color convert) at
//! different image sizes and subsampling modes.
//!
//! Run with: cargo run --release --features parallel --example bench_parallel_decode

#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("ERROR: Run with --features parallel");
    eprintln!("  cargo run --release --features parallel --example bench_parallel_decode");
}

#[cfg(feature = "parallel")]
fn main() {
    bench::run();
}

#[cfg(feature = "parallel")]
mod bench {
    use enough::Unstoppable;
    use std::time::Instant;
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig};

    fn create_test_jpeg(width: u32, height: u32, subsampling: ChromaSubsampling) -> Vec<u8> {
        let pixels: Vec<rgb::RGB<u8>> = (0..width as usize * height as usize)
            .map(|i| {
                let x = i % width as usize;
                let y = i / width as usize;
                rgb::RGB {
                    r: ((x * 7 + y * 3) % 256) as u8,
                    g: ((x * 11 + y * 5) % 256) as u8,
                    b: ((x * 13 + y * 9) % 256) as u8,
                }
            })
            .collect();

        let config = EncoderConfig::ycbcr(90.0, subsampling).progressive(false);
        config.encode(&pixels, width, height).unwrap()
    }

    fn bench_decode(data: &[u8], pool: &rayon::ThreadPool, iterations: usize) -> f64 {
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);

        // Warmup
        for _ in 0..3 {
            pool.install(|| decoder.decode(data, Unstoppable).unwrap());
        }

        let start = Instant::now();
        for _ in 0..iterations {
            pool.install(|| {
                std::hint::black_box(decoder.decode(data, Unstoppable).unwrap());
            });
        }
        start.elapsed().as_secs_f64() / iterations as f64
    }

    pub fn run() {
        let num_threads = rayon::current_num_threads();
        eprintln!("Parallel output pass benchmark");
        eprintln!("Threads: {}", num_threads);

        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        eprintln!("\nGenerating test images...");

        struct TestCase {
            label: &'static str,
            width: u32,
            height: u32,
            subsampling: ChromaSubsampling,
            iterations: usize,
        }

        let cases = [
            TestCase {
                label: "2048x2048 4:4:4",
                width: 2048,
                height: 2048,
                subsampling: ChromaSubsampling::None,
                iterations: 20,
            },
            TestCase {
                label: "2048x2048 4:2:0",
                width: 2048,
                height: 2048,
                subsampling: ChromaSubsampling::Quarter,
                iterations: 20,
            },
            TestCase {
                label: "4096x2160 4:4:4",
                width: 4096,
                height: 2160,
                subsampling: ChromaSubsampling::None,
                iterations: 10,
            },
            TestCase {
                label: "4096x2160 4:2:0",
                width: 4096,
                height: 2160,
                subsampling: ChromaSubsampling::Quarter,
                iterations: 10,
            },
            TestCase {
                label: "7680x4320 4:4:4",
                width: 7680,
                height: 4320,
                subsampling: ChromaSubsampling::None,
                iterations: 5,
            },
            TestCase {
                label: "7680x4320 4:2:0",
                width: 7680,
                height: 4320,
                subsampling: ChromaSubsampling::Quarter,
                iterations: 5,
            },
        ];

        // Pre-encode all test images
        let images: Vec<(&TestCase, Vec<u8>)> = cases
            .iter()
            .map(|tc| {
                let jpeg = create_test_jpeg(tc.width, tc.height, tc.subsampling);
                eprintln!("  {}: {} bytes", tc.label, jpeg.len());
                (tc, jpeg)
            })
            .collect();

        eprintln!(
            "\n{:>24} {:>10} {:>10} {:>8}",
            "Image", "Serial", "Parallel", "Speedup"
        );
        eprintln!("{}", "-".repeat(56));

        for (tc, jpeg) in &images {
            let t_serial = bench_decode(jpeg, &serial_pool, tc.iterations);
            let t_parallel = bench_decode(jpeg, &parallel_pool, tc.iterations);
            let speedup = t_serial / t_parallel;

            eprintln!(
                "{:>24} {:>8.2}ms {:>8.2}ms {:>7.2}x",
                tc.label,
                t_serial * 1000.0,
                t_parallel * 1000.0,
                speedup,
            );
        }

        eprintln!("\nNotes:");
        eprintln!("- No restart markers: entropy decode is always serial");
        eprintln!("- Parallel speedup is from output pass only (IDCT + upsample + color convert)");
        eprintln!("- Threshold: >= 8M pixels and >= 8 MCU rows (falls through to serial otherwise)");
    }
}
