//! Benchmark parallel decode pipeline (entropy + output pass).
//!
//! Compares 1-thread (serial) vs N-thread (parallel) decode for images
//! with restart markers, measuring the combined speedup from both
//! parallel entropy decode and parallel output pass.
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

    fn create_test_jpeg(
        width: u32,
        height: u32,
        subsampling: ChromaSubsampling,
        restart_interval: u16,
    ) -> Vec<u8> {
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

        // Must use progressive(false) — progressive doesn't emit RST markers
        let config = EncoderConfig::ycbcr(90.0, subsampling)
            .progressive(false)
            .restart_interval(restart_interval);
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
        eprintln!("Parallel decode benchmark (entropy + output pass)");
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
            dri: u16,
            iterations: usize,
        }

        let cases = [
            // 4:4:4 — parallel output path activates at >=8M pixels
            TestCase {
                label: "2048x2048 4:4:4 DRI=20",
                width: 2048,
                height: 2048,
                subsampling: ChromaSubsampling::None,
                dri: 20,
                iterations: 20,
            },
            TestCase {
                label: "4096x2160 4:4:4 DRI=20",
                width: 4096,
                height: 2160,
                subsampling: ChromaSubsampling::None,
                dri: 20,
                iterations: 10,
            },
            TestCase {
                label: "7680x4320 4:4:4 DRI=20",
                width: 7680,
                height: 4320,
                subsampling: ChromaSubsampling::None,
                dri: 20,
                iterations: 5,
            },
            // 4:2:0 — both entropy and output parallelism
            TestCase {
                label: "2048x2048 4:2:0 DRI=20",
                width: 2048,
                height: 2048,
                subsampling: ChromaSubsampling::Quarter,
                dri: 20,
                iterations: 20,
            },
            TestCase {
                label: "4096x2160 4:2:0 DRI=20",
                width: 4096,
                height: 2160,
                subsampling: ChromaSubsampling::Quarter,
                dri: 20,
                iterations: 10,
            },
            TestCase {
                label: "7680x4320 4:2:0 DRI=20",
                width: 7680,
                height: 4320,
                subsampling: ChromaSubsampling::Quarter,
                dri: 20,
                iterations: 5,
            },
            // 8K with larger restart interval (fewer, bigger segments)
            TestCase {
                label: "7680x4320 4:2:0 DRI=100",
                width: 7680,
                height: 4320,
                subsampling: ChromaSubsampling::Quarter,
                dri: 100,
                iterations: 5,
            },
        ];

        // Pre-encode all test images
        let images: Vec<(&TestCase, Vec<u8>)> = cases
            .iter()
            .map(|tc| {
                let jpeg = create_test_jpeg(tc.width, tc.height, tc.subsampling, tc.dri);
                let rst_count = jpeg
                    .windows(2)
                    .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
                    .count();
                eprintln!(
                    "  {}: {} bytes, {} RST markers",
                    tc.label,
                    jpeg.len(),
                    rst_count
                );
                (tc, jpeg)
            })
            .collect();

        // Also create no-DRI versions to isolate output-pass parallelism
        eprintln!("\nGenerating no-DRI baseline images...");
        let nodri_cases = [
            ("4096x2160 4:2:0 noDRI", 4096u32, 2160u32, ChromaSubsampling::Quarter, 10usize),
            ("7680x4320 4:2:0 noDRI", 7680, 4320, ChromaSubsampling::Quarter, 5),
            ("7680x4320 4:4:4 noDRI", 7680, 4320, ChromaSubsampling::None, 5),
        ];
        let nodri_images: Vec<(&str, Vec<u8>, usize)> = nodri_cases
            .iter()
            .map(|(label, w, h, ss, iters)| {
                let jpeg = create_test_jpeg(*w, *h, *ss, 0);
                eprintln!("  {}: {} bytes", label, jpeg.len());
                (*label, jpeg, *iters)
            })
            .collect();

        eprintln!(
            "\n{:>30} {:>10} {:>10} {:>8}",
            "Image", "Serial", "Parallel", "Speedup"
        );
        eprintln!("{}", "-".repeat(62));

        // DRI images (parallel entropy + parallel output)
        eprintln!("--- DRI images (parallel entropy + output) ---");
        for (tc, jpeg) in &images {
            let t_serial = bench_decode(jpeg, &serial_pool, tc.iterations);
            let t_parallel = bench_decode(jpeg, &parallel_pool, tc.iterations);
            let speedup = t_serial / t_parallel;

            eprintln!(
                "{:>30} {:>8.2}ms {:>8.2}ms {:>7.2}x",
                tc.label,
                t_serial * 1000.0,
                t_parallel * 1000.0,
                speedup,
            );
        }

        // No-DRI images (parallel output only)
        eprintln!("--- No-DRI images (parallel output only) ---");
        for (label, jpeg, iters) in &nodri_images {
            let t_serial = bench_decode(jpeg, &serial_pool, *iters);
            let t_parallel = bench_decode(jpeg, &parallel_pool, *iters);
            let speedup = t_serial / t_parallel;

            eprintln!(
                "{:>30} {:>8.2}ms {:>8.2}ms {:>7.2}x",
                label,
                t_serial * 1000.0,
                t_parallel * 1000.0,
                speedup,
            );
        }

        eprintln!("\nNotes:");
        eprintln!("- Serial = 1-thread rayon pool, Parallel = {}-thread pool", num_threads);
        eprintln!("- DRI: parallel entropy decode + parallel output pass");
        eprintln!("- No-DRI: parallel output pass only (entropy is always serial)");
        eprintln!("- Parallel output activates at >= 8M pixels");
    }
}
