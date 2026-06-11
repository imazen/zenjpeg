//! Cache pressure experiment: is the >2048 parallel scaling cliff from output writes?
//!
//! Measures:
//! 1. Raw parallel write bandwidth (regular stores vs non-temporal stores)
//! 2. Actual decode time at 1T vs 8T
//! 3. Per-pixel cost breakdown to identify where time goes
//!
//! The hypothesis: at 4096+, the 48MB+ RGB output buffer exceeds L3 (32MB on Zen 4),
//! and 8 threads writing scattered stripes saturate memory bandwidth. NT stores
//! bypass cache, potentially reducing L3 pollution and improving parallel scaling.
//!
//! Run: cargo run --release --features parallel,decoder --example bench_cache_experiment
//!
//! Expected runtime: ~2 minutes

#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("ERROR: Run with --features parallel,decoder");
}

#[cfg(feature = "parallel")]
fn main() {
    bench::run();
}

#[cfg(feature = "parallel")]
mod bench {
    use enough::Unstoppable;
    use std::path::Path;
    use std::time::Instant;
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig};

    fn load_png(path: &Path) -> (Vec<rgb::RGB<u8>>, u32, u32) {
        let img = zenjpeg_bench_utils::load_png(path).expect("Failed to load PNG");
        (img.buf().to_vec(), img.width() as u32, img.height() as u32)
    }

    fn tile_to_size(
        pixels: &[rgb::RGB<u8>],
        src_w: u32,
        src_h: u32,
        target: u32,
    ) -> Vec<rgb::RGB<u8>> {
        let tw = target as usize;
        let th = target as usize;
        let sw = src_w as usize;
        let sh = src_h as usize;
        let mut out = vec![rgb::RGB { r: 0, g: 0, b: 0 }; tw * th];
        for y in 0..th {
            for x in 0..tw {
                out[y * tw + x] = pixels[(y % sh) * sw + (x % sw)];
            }
        }
        out
    }

    fn median_of(times: &mut [f64]) -> f64 {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    }

    // --- Raw memory write bandwidth tests ---

    /// Regular stores via fill (memset): what the compiler generates for bulk writes
    fn bench_write_regular(size: usize, pool: &rayon::ThreadPool, iters: usize) -> f64 {
        use rayon::prelude::*;
        let num_threads = pool.current_num_threads();
        let mut buf = vec![0u8; size];
        let stripe = size / num_threads;

        for _ in 0..3 {
            pool.install(|| {
                buf.par_chunks_mut(stripe)
                    .for_each(|chunk| chunk.fill(0x42));
            });
        }

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            pool.install(|| {
                buf.par_chunks_mut(stripe)
                    .for_each(|chunk| chunk.fill(0x42));
            });
            times.push(start.elapsed().as_secs_f64());
        }
        median_of(&mut times)
    }

    /// Non-temporal stores: bypass cache hierarchy
    #[cfg(target_arch = "x86_64")]
    fn bench_write_nt(size: usize, pool: &rayon::ThreadPool, iters: usize) -> f64 {
        use rayon::prelude::*;
        let num_threads = pool.current_num_threads();
        let aligned_size = (size + 31) & !31;
        // Use aligned allocation so NT stores don't fault
        let mut buf = alloc_aligned(aligned_size);
        // Stripe size must also be 32-byte aligned for NT store alignment
        let stripe = ((aligned_size / num_threads) + 31) & !31;

        for _ in 0..3 {
            pool.install(|| {
                buf.par_chunks_mut(stripe).for_each(|chunk| {
                    nt_fill(chunk);
                });
            });
        }

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            pool.install(|| {
                buf.par_chunks_mut(stripe).for_each(|chunk| {
                    nt_fill(chunk);
                });
            });
            times.push(start.elapsed().as_secs_f64());
        }
        median_of(&mut times)
    }

    /// Allocate a 32-byte aligned buffer
    fn alloc_aligned(size: usize) -> Vec<u8> {
        // Round up to 32 bytes
        let aligned = (size + 31) & !31;
        let layout = std::alloc::Layout::from_size_align(aligned, 32).unwrap();
        unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Vec::from_raw_parts(ptr, aligned, aligned)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn nt_fill(dst: &mut [u8]) {
        use std::arch::x86_64::*;
        let len = dst.len();
        let ptr = dst.as_mut_ptr();
        // Find first 32-byte aligned offset
        let align_off = ptr.align_offset(32);
        // Fill unaligned prefix with regular stores
        let prefix_end = align_off.min(len);
        for i in 0..prefix_end {
            dst[i] = 0x42;
        }
        if prefix_end >= len {
            return;
        }
        let remaining_len = len - prefix_end;
        let chunks = remaining_len / 32;
        let aligned_ptr = unsafe { ptr.add(prefix_end) };

        unsafe {
            let val = _mm256_set1_epi8(0x42);
            for i in 0..chunks {
                _mm256_stream_si256(aligned_ptr.add(i * 32) as *mut __m256i, val);
            }
            _mm_sfence();
        }
        // Handle remainder with regular stores
        let done = prefix_end + chunks * 32;
        for i in done..len {
            dst[i] = 0x42;
        }
    }

    /// Single-threaded write bandwidth baseline
    fn bench_write_1t(size: usize, iters: usize) -> f64 {
        let mut buf = vec![0u8; size];
        for _ in 0..3 {
            buf.fill(0x42);
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            buf.fill(0x42);
            times.push(start.elapsed().as_secs_f64());
        }
        median_of(&mut times)
    }

    #[cfg(target_arch = "x86_64")]
    fn bench_write_nt_1t(size: usize, iters: usize) -> f64 {
        let mut buf = alloc_aligned((size + 31) & !31);
        for _ in 0..3 {
            nt_fill(&mut buf);
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            nt_fill(&mut buf);
            times.push(start.elapsed().as_secs_f64());
        }
        median_of(&mut times)
    }

    pub fn run() {
        let corpus_base = std::path::PathBuf::from(
            std::env::var("CORPUS_PATH")
                .unwrap_or_else(|_| String::from("/home/lilith/work/codec-eval/codec-corpus")),
        );

        let clic_dir = corpus_base.join("clic2025/final-test");
        if !clic_dir.exists() {
            eprintln!("ERROR: CLIC corpus not found at {:?}", clic_dir);
            return;
        }

        // Load just 2 images
        let mut clic_files: Vec<_> = std::fs::read_dir(&clic_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
            .collect();
        clic_files.sort_by_key(|e| e.file_name());

        let selected: Vec<_> = clic_files.iter().step_by(10).take(2).collect();
        eprintln!("Loading {} images...", selected.len());

        let mut source_images = Vec::new();
        for entry in &selected {
            let path = entry.path();
            let name = path.file_stem().unwrap().to_string_lossy();
            let (pixels, w, h) = load_png(&path);
            eprintln!("  {}... ({}x{})", &name[..8], w, h);
            source_images.push((pixels, w, h));
        }

        let sizes: &[u32] = &[512, 1024, 2048, 4096];
        let thread_counts = [1, 2, 4, 8];

        let pools: Vec<(usize, rayon::ThreadPool)> = thread_counts
            .iter()
            .map(|&n| {
                (
                    n,
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n)
                        .build()
                        .unwrap(),
                )
            })
            .collect();

        // ============================================================
        // Part 1: Raw write bandwidth at each buffer size
        // ============================================================
        eprintln!("\n=== Part 1: Raw write bandwidth (MB/s) ===");
        eprintln!("Buffer = width² × 3 (RGB output size)\n");

        eprintln!(
            "{:>7} {:>8} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}",
            "Size", "buf MB", "1T fill", "8T fill", "1T NT", "8T NT", "NT/fill", "time ms"
        );
        eprintln!("{}", "-".repeat(90));

        let pool_8t = &pools.iter().find(|(n, _)| *n == 8).unwrap().1;

        for &size in sizes {
            let buf_size = (size as usize) * (size as usize) * 3;
            let buf_mb = buf_size as f64 / (1024.0 * 1024.0);
            let write_iters = if size <= 1024 { 50 } else { 20 };

            let t_1t = bench_write_1t(buf_size, write_iters);
            let bw_1t = buf_mb / t_1t;

            let t_8t = bench_write_regular(buf_size, pool_8t, write_iters);
            let bw_8t = buf_mb / t_8t;

            #[cfg(target_arch = "x86_64")]
            {
                let t_nt_1t = bench_write_nt_1t(buf_size, write_iters);
                let bw_nt_1t = buf_mb / t_nt_1t;

                let t_nt_8t = bench_write_nt(buf_size, pool_8t, write_iters);
                let bw_nt_8t = buf_mb / t_nt_8t;

                let nt_ratio = bw_nt_8t / bw_8t;

                eprintln!(
                    "{:>5}² {:>6.1}MB {:>8.0}MB/s {:>8.0}MB/s {:>8.0}MB/s {:>8.0}MB/s {:>7.2}x {:>7.2}ms",
                    size,
                    buf_mb,
                    bw_1t,
                    bw_8t,
                    bw_nt_1t,
                    bw_nt_8t,
                    nt_ratio,
                    t_8t * 1000.0
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                eprintln!(
                    "{:>5}² {:>6.1}MB {:>8.0}MB/s {:>8.0}MB/s {:>10} {:>10} {:>9} {:>7.2}ms",
                    size,
                    buf_mb,
                    bw_1t,
                    bw_8t,
                    "n/a",
                    "n/a",
                    "n/a",
                    t_8t * 1000.0
                );
            }
        }

        // ============================================================
        // Part 2: Decode times
        // ============================================================
        eprintln!(
            "\n=== Part 2: Actual decode times (bl-420 Q85, avg of {} images) ===\n",
            source_images.len()
        );

        // Encode test JPEGs
        eprintln!("Encoding test JPEGs...");
        let mut encoded: Vec<(u32, Vec<u8>)> = Vec::new();
        for &size in sizes {
            let mut jpegs = Vec::new();
            for (pixels, w, h) in &source_images {
                let square = tile_to_size(pixels, *w, *h, size);
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                    .progressive(false)
                    .restart_mcu_rows(4);
                jpegs.push(config.encode(&square, size, size).unwrap());
            }
            // We'll decode each and average
            for jpeg in jpegs {
                encoded.push((size, jpeg));
            }
        }
        eprintln!("Done.\n");

        // Header
        eprint!("{:>7} {:>10}", "Size", "buf MB");
        for &(n, _) in &pools {
            eprint!(" {:>9}", format!("zen-{}T", n));
        }
        eprint!(" {:>10} {:>10}", "8T/1T", "write%");
        eprintln!();
        eprintln!("{}", "-".repeat(100));

        let decode_warmup = 5;
        let decode_iters = 15;

        for &size in sizes {
            let buf_mb = (size as f64 * size as f64 * 3.0) / (1024.0 * 1024.0);

            let this_jpegs: Vec<&[u8]> = encoded
                .iter()
                .filter(|(s, _)| *s == size)
                .map(|(_, j)| j.as_slice())
                .collect();

            let mut times_per_thread: Vec<f64> = Vec::new();

            for (_n, pool) in &pools {
                let mut avg = 0.0;
                for jpeg in &this_jpegs {
                    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                    let mut times = Vec::with_capacity(decode_iters);

                    // Warmup
                    for _ in 0..decode_warmup {
                        pool.install(|| {
                            std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                        });
                    }

                    for _ in 0..decode_iters {
                        let start = Instant::now();
                        pool.install(|| {
                            std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                        });
                        times.push(start.elapsed().as_secs_f64());
                    }
                    avg += median_of(&mut times);
                }
                avg /= this_jpegs.len() as f64;
                times_per_thread.push(avg);
            }

            let t_1t = times_per_thread[0];
            let t_8t = times_per_thread[3]; // 8T is index 3

            // Raw write time for comparison
            let buf_size = (size as usize) * (size as usize) * 3;
            let write_iters = if size <= 1024 { 30 } else { 15 };
            let write_8t = bench_write_regular(buf_size, pool_8t, write_iters);

            eprint!("{:>5}² {:>8.1}MB", size, buf_mb);
            for &t in &times_per_thread {
                eprint!(" {:>9}", format_time(t));
            }
            let speedup = t_1t / t_8t;
            let write_pct = (write_8t / t_8t) * 100.0;
            eprint!(" {:>9.2}x {:>9.1}%", speedup, write_pct);
            eprintln!();
        }

        // ============================================================
        // Part 3: Per-pixel cost analysis
        // ============================================================
        eprintln!("\n=== Part 3: Per-pixel cost (ns/pixel) ===\n");
        eprintln!(
            "{:>7} {:>9} {:>9} {:>9} {:>9} {:>12}",
            "Size", "zen-1T", "zen-8T", "write-8T", "decode-only", "Amdahl limit"
        );
        eprintln!("{}", "-".repeat(70));

        for &size in sizes {
            let pixels = size as f64 * size as f64;
            let buf_size = (size as usize) * (size as usize) * 3;

            let this_jpegs: Vec<&[u8]> = encoded
                .iter()
                .filter(|(s, _)| *s == size)
                .map(|(_, j)| j.as_slice())
                .collect();

            // 1T decode
            let pool_1t = &pools[0].1;
            let mut t_1t_avg = 0.0;
            for jpeg in &this_jpegs {
                let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                let mut times = Vec::with_capacity(decode_iters);
                for _ in 0..decode_warmup {
                    pool_1t.install(|| {
                        std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                    });
                }
                for _ in 0..decode_iters {
                    let start = Instant::now();
                    pool_1t.install(|| {
                        std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                    });
                    times.push(start.elapsed().as_secs_f64());
                }
                t_1t_avg += median_of(&mut times);
            }
            t_1t_avg /= this_jpegs.len() as f64;

            // 8T decode
            let mut t_8t_avg = 0.0;
            for jpeg in &this_jpegs {
                let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                let mut times = Vec::with_capacity(decode_iters);
                for _ in 0..decode_warmup {
                    pool_8t.install(|| {
                        std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                    });
                }
                for _ in 0..decode_iters {
                    let start = Instant::now();
                    pool_8t.install(|| {
                        std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                    });
                    times.push(start.elapsed().as_secs_f64());
                }
                t_8t_avg += median_of(&mut times);
            }
            t_8t_avg /= this_jpegs.len() as f64;

            let w_iters = if size <= 1024 { 30 } else { 15 };
            let write_8t = bench_write_regular(buf_size, pool_8t, w_iters);

            let ns_1t = t_1t_avg / pixels * 1e9;
            let ns_8t = t_8t_avg / pixels * 1e9;
            let ns_write = write_8t / pixels * 1e9;
            let ns_decode_only = ns_8t - ns_write; // approx compute-only portion

            // Amdahl: if write_fraction is serial (memory-bound), max speedup =
            // 1 / (serial_frac + parallel_frac/N)
            let serial_frac = write_8t / t_1t_avg; // fraction of 1T time that's write-bound
            let amdahl_8t = 1.0 / (serial_frac + (1.0 - serial_frac) / 8.0);

            eprintln!(
                "{:>5}² {:>7.1}ns {:>7.1}ns {:>7.1}ns {:>7.1}ns {:>10.1}x",
                size, ns_1t, ns_8t, ns_write, ns_decode_only, amdahl_8t,
            );
        }

        eprintln!("\nNotes:");
        eprintln!(
            "  - 'write%' = raw 8T write time / decode-8T time (how much of decode is just writes)"
        );
        eprintln!(
            "  - 'Amdahl limit' = max theoretical 8T speedup if writes are serial bottleneck"
        );
        eprintln!("  - L2 = 1MB/core, L3 = 32MB shared (AMD 7950X)");
        eprintln!(
            "  - 512²×3 = 0.75MB (fits L2), 2048²×3 = 12MB (fits L3), 4096²×3 = 48MB (exceeds L3)"
        );

        // ============================================================
        // Part 4: DRI sweep — optimal restart interval per image size
        // ============================================================
        eprintln!(
            "\n=== Part 4: DRI sweep (bl-420 Q85, avg of {} images) ===\n",
            source_images.len()
        );
        eprintln!("Encoding with different restart_mcu_rows values...");

        let dri_values: &[u16] = &[0, 1, 2, 4, 8, 16];
        let dri_iters = 15;
        let dri_warmup = 5;

        // Header
        eprint!("{:>7} {:>4}", "Size", "DRI");
        eprint!(" {:>7} {:>9} {:>9} {:>9}", "segs", "1T", "4T", "8T");
        eprint!(" {:>7} {:>7}", "8T/1T", "vs DRI4");
        eprintln!();
        eprintln!("{}", "-".repeat(80));

        for &size in sizes {
            // Encode with each DRI
            let mut dri_jpegs: Vec<(u16, Vec<Vec<u8>>)> = Vec::new();
            for &dri in dri_values {
                let mut jpegs = Vec::new();
                for (pixels, w, h) in &source_images {
                    let square = tile_to_size(pixels, *w, *h, size);
                    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                        .progressive(false)
                        .restart_mcu_rows(dri);
                    jpegs.push(config.encode(&square, size, size).unwrap());
                }
                dri_jpegs.push((dri, jpegs));
            }

            let mut dri4_8t = 0.0f64; // baseline for comparison

            for (dri, jpegs) in &dri_jpegs {
                // Count restart segments by checking file
                let decoder_probe = Decoder::new().output_format(PixelFormat::Rgb);
                let _probe = decoder_probe.decode(&jpegs[0], Unstoppable).unwrap();

                // Decode with 1T, 4T, 8T
                let thread_counts = [1usize, 4, 8];
                let mut times_by_tc: Vec<f64> = Vec::new();

                for &tc in &thread_counts {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(tc)
                        .build()
                        .unwrap();

                    let mut avg = 0.0;
                    for jpeg in jpegs {
                        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
                        let mut times = Vec::with_capacity(dri_iters);

                        for _ in 0..dri_warmup {
                            pool.install(|| {
                                std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                            });
                        }
                        for _ in 0..dri_iters {
                            let start = Instant::now();
                            pool.install(|| {
                                std::hint::black_box(decoder.decode(jpeg, Unstoppable).unwrap());
                            });
                            times.push(start.elapsed().as_secs_f64());
                        }
                        avg += median_of(&mut times);
                    }
                    avg /= jpegs.len() as f64;
                    times_by_tc.push(avg);
                }

                let t_1t = times_by_tc[0];
                let t_4t = times_by_tc[1];
                let t_8t = times_by_tc[2];
                let ratio_8t = t_1t / t_8t;

                if *dri == 4 {
                    dri4_8t = t_8t;
                }

                // Estimate segment count
                let _mcu_cols = (size as usize + 15) / 16;
                let mcu_rows = (size as usize + 15) / 16;
                let segs = if *dri == 0 {
                    1
                } else {
                    mcu_rows / (*dri as usize)
                };

                let vs_dri4 = if dri4_8t > 0.0 && *dri != 4 {
                    format!("{:+.1}%", (t_8t / dri4_8t - 1.0) * 100.0)
                } else if *dri == 4 {
                    "  base".to_string()
                } else {
                    "  n/a".to_string()
                };

                eprintln!(
                    "{:>5}² {:>4} {:>7} {:>9} {:>9} {:>9} {:>7.2}x {:>7}",
                    size,
                    dri,
                    segs,
                    format_time(t_1t),
                    format_time(t_4t),
                    format_time(t_8t),
                    ratio_8t,
                    vs_dri4,
                );
            }
            eprintln!(); // blank line between sizes
        }
    }

    fn format_time(secs: f64) -> String {
        let ms = secs * 1000.0;
        if ms < 1.0 {
            format!("{:.0}us", secs * 1_000_000.0)
        } else if ms < 10.0 {
            format!("{:.2}ms", ms)
        } else if ms < 100.0 {
            format!("{:.1}ms", ms)
        } else {
            format!("{:.0}ms", ms)
        }
    }
}
