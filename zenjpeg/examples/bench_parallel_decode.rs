//! Benchmark: encode-for-fast-decode strategies.
//!
//! Compares baseline+DRI vs progressive (no DRI) on real corpus images.
//! Shows file size cost and decode speedup at 1-4 threads.
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
    use std::path::Path;
    use std::time::Instant;
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig};

    fn load_png(path: &Path) -> (Vec<rgb::RGB<u8>>, u32, u32) {
        let img = zenjpeg_bench_utils::load_png(path).expect("Failed to load PNG");
        (img.buf().to_vec(), img.width() as u32, img.height() as u32)
    }

    fn encode_jpeg(
        pixels: &[rgb::RGB<u8>],
        width: u32,
        height: u32,
        subsampling: ChromaSubsampling,
        progressive: bool,
        restart_rows: u16,
    ) -> Vec<u8> {
        let config = EncoderConfig::ycbcr(90.0, subsampling)
            .progressive(progressive)
            .restart_mcu_rows(restart_rows);
        config.encode(pixels, width, height).unwrap()
    }

    fn count_rst_markers(data: &[u8]) -> usize {
        data.windows(2)
            .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
            .count()
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
        let corpus_base = std::path::PathBuf::from(
            std::env::var("CORPUS_PATH")
                .unwrap_or_else(|_| String::from("/home/lilith/work/codec-eval/codec-corpus")),
        );

        if !corpus_base.exists() {
            eprintln!("ERROR: corpus not found at {:?}", corpus_base);
            return;
        }

        let thread_counts = [1, 2, 3, 4];

        eprintln!("Encode-for-fast-decode benchmark");
        eprintln!("Comparing: progressive (no DRI) vs baseline + row-aligned DRI");
        eprintln!("Thread counts: {:?}\n", thread_counts);

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

        // Collect source images
        let mut source_images: Vec<(String, Vec<rgb::RGB<u8>>, u32, u32)> = Vec::new();

        // CLIC2025 photos
        let clic_dir = corpus_base.join("clic2025/final-test");
        if clic_dir.exists() {
            let mut clic_files: Vec<_> = std::fs::read_dir(&clic_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
                .collect();
            clic_files.sort_by_key(|e| e.file_name());
            for entry in clic_files.iter().step_by(5).take(6) {
                let path = entry.path();
                let (pixels, w, h) = load_png(&path);
                let name = path.file_stem().unwrap().to_string_lossy();
                source_images.push((format!("clic_{}", &name[..8]), pixels, w, h));
            }
        }

        // gb82-sc screenshots
        let sc_dir = corpus_base.join("gb82-sc");
        if sc_dir.exists() {
            for name in &["imac_dark", "codec_wiki", "windows"] {
                let path = sc_dir.join(format!("{}.png", name));
                if path.exists() {
                    let (pixels, w, h) = load_png(&path);
                    source_images.push((format!("sc_{}", name), pixels, w, h));
                }
            }
        }

        // gb82 photos
        let gb82_dir = corpus_base.join("gb82");
        if gb82_dir.exists() {
            for name in &["baby", "city", "flowers"] {
                let path = gb82_dir.join(format!("{}-lossless.png", name));
                if path.exists() {
                    let (pixels, w, h) = load_png(&path);
                    source_images.push((format!("gb82_{}", name), pixels, w, h));
                }
            }
        }

        if source_images.is_empty() {
            eprintln!("ERROR: no source images found");
            return;
        }

        // Sort by pixel count descending
        source_images.sort_by_key(|e| std::cmp::Reverse(e.2 as u64 * e.3 as u64));

        // Encode each image in multiple modes
        struct EncodedSet {
            name: String,
            width: u32,
            height: u32,
            progressive: Vec<u8>,
            baseline_nodri: Vec<u8>,
            baseline_dri_1row: Vec<u8>,
            baseline_dri_4row: Vec<u8>,
            rst_count_1row: usize,
            rst_count_4row: usize,
            iterations: usize,
        }

        eprintln!(
            "Encoding {} images in 4 modes (Q90, 4:2:0)...\n",
            source_images.len()
        );

        let mut images: Vec<EncodedSet> = Vec::new();

        for (name, pixels, w, h) in &source_images {
            let mpix = (*w as f64 * *h as f64) / 1e6;

            let progressive = encode_jpeg(pixels, *w, *h, ChromaSubsampling::Quarter, true, 0);
            let baseline_nodri = encode_jpeg(pixels, *w, *h, ChromaSubsampling::Quarter, false, 0);
            let baseline_dri_1row =
                encode_jpeg(pixels, *w, *h, ChromaSubsampling::Quarter, false, 1);
            let baseline_dri_4row =
                encode_jpeg(pixels, *w, *h, ChromaSubsampling::Quarter, false, 4);

            let rst_1row = count_rst_markers(&baseline_dri_1row);
            let rst_4row = count_rst_markers(&baseline_dri_4row);
            let iters = if mpix > 3.0 { 10 } else { 20 };

            images.push(EncodedSet {
                name: name.clone(),
                width: *w,
                height: *h,
                progressive,
                baseline_nodri,
                baseline_dri_1row,
                baseline_dri_4row,
                rst_count_1row: rst_1row,
                rst_count_4row: rst_4row,
                iterations: iters,
            });
        }

        // === File size comparison ===
        eprintln!("=== File sizes (bytes) ===");
        eprintln!(
            "{:>16} {:>9} {:>10} {:>10} {:>10} {:>10} {:>6} {:>6}",
            "Image", "Size", "Prog", "BL noDRI", "BL DRI/1r", "BL DRI/4r", "RST/1", "RST/4"
        );
        eprintln!("{}", "-".repeat(90));

        let mut total_prog = 0usize;
        let mut total_bl = 0usize;
        let mut total_dri1 = 0usize;
        let mut total_dri4 = 0usize;

        for img in &images {
            total_prog += img.progressive.len();
            total_bl += img.baseline_nodri.len();
            total_dri1 += img.baseline_dri_1row.len();
            total_dri4 += img.baseline_dri_4row.len();

            eprintln!(
                "{:>16} {:>4}x{:<4} {:>10} {:>10} {:>10} {:>10} {:>6} {:>6}",
                img.name,
                img.width,
                img.height,
                img.progressive.len(),
                img.baseline_nodri.len(),
                img.baseline_dri_1row.len(),
                img.baseline_dri_4row.len(),
                img.rst_count_1row,
                img.rst_count_4row,
            );
        }

        eprintln!("{}", "-".repeat(90));
        eprintln!(
            "{:>16} {:>9} {:>10} {:>10} {:>10} {:>10}",
            "TOTAL", "", total_prog, total_bl, total_dri1, total_dri4
        );
        let dri1_vs_prog = (total_dri1 as f64 / total_prog as f64 - 1.0) * 100.0;
        let dri4_vs_prog = (total_dri4 as f64 / total_prog as f64 - 1.0) * 100.0;
        let dri1_vs_bl = (total_dri1 as f64 / total_bl as f64 - 1.0) * 100.0;
        let dri4_vs_bl = (total_dri4 as f64 / total_bl as f64 - 1.0) * 100.0;
        let bl_vs_prog = (total_bl as f64 / total_prog as f64 - 1.0) * 100.0;
        eprintln!("\nBL noDRI vs Prog:  +{:.1}%", bl_vs_prog);
        eprintln!("BL DRI/1row vs Prog: +{:.1}%", dri1_vs_prog);
        eprintln!("BL DRI/4row vs Prog: +{:.1}%", dri4_vs_prog);
        eprintln!("DRI/1row vs BL noDRI: +{:.2}%", dri1_vs_bl);
        eprintln!("DRI/4row vs BL noDRI: +{:.2}%", dri4_vs_bl);

        // === Decode speed: Progressive (1T baseline) ===
        eprintln!("\n=== Decode: Progressive (no parallel entropy possible) ===");
        eprint!("{:>16} {:>9}", "Image", "Size");
        for &(n, _) in &pools {
            eprint!(" {:>8}", format!("{}T", n));
        }
        eprintln!();
        eprintln!("{}", "-".repeat(16 + 10 + pools.len() * 9));

        let mut prog_1t_total = 0.0f64;
        for img in &images {
            let times: Vec<f64> = pools
                .iter()
                .map(|(_, pool)| bench_decode(&img.progressive, pool, img.iterations))
                .collect();

            prog_1t_total += times[0];
            eprint!("{:>16} {:>4}x{:<4}", img.name, img.width, img.height);
            for t in &times {
                eprint!(" {:>6.1}ms", t * 1000.0);
            }
            eprintln!();
        }

        // === Decode speed: Baseline + DRI/1row ===
        eprintln!("\n=== Decode: Baseline + DRI/1row (parallel entropy + output) ===");
        eprint!("{:>16} {:>9}", "Image", "Size");
        for &(n, _) in &pools {
            eprint!(" {:>8}", format!("{}T", n));
        }
        for &(n, _) in &pools[1..] {
            eprint!(" {:>7}", format!("{}T/1T", n));
        }
        eprintln!();
        eprintln!(
            "{}",
            "-".repeat(16 + 10 + pools.len() * 9 + (pools.len() - 1) * 8)
        );

        let mut dri1_1t_total = 0.0f64;
        let mut dri1_4t_total = 0.0f64;
        for img in &images {
            let times: Vec<f64> = pools
                .iter()
                .map(|(_, pool)| bench_decode(&img.baseline_dri_1row, pool, img.iterations))
                .collect();

            dri1_1t_total += times[0];
            dri1_4t_total += times[times.len() - 1];
            eprint!("{:>16} {:>4}x{:<4}", img.name, img.width, img.height);
            for t in &times {
                eprint!(" {:>6.1}ms", t * 1000.0);
            }
            let t1 = times[0];
            for t in &times[1..] {
                eprint!(" {:>6.2}x", t1 / t);
            }
            eprintln!();
        }

        // === Decode speed: Baseline + DRI/4row ===
        eprintln!("\n=== Decode: Baseline + DRI/4row (parallel entropy + output) ===");
        eprint!("{:>16} {:>9}", "Image", "Size");
        for &(n, _) in &pools {
            eprint!(" {:>8}", format!("{}T", n));
        }
        for &(n, _) in &pools[1..] {
            eprint!(" {:>7}", format!("{}T/1T", n));
        }
        eprintln!();
        eprintln!(
            "{}",
            "-".repeat(16 + 10 + pools.len() * 9 + (pools.len() - 1) * 8)
        );

        let mut dri4_4t_total = 0.0f64;
        for img in &images {
            let times: Vec<f64> = pools
                .iter()
                .map(|(_, pool)| bench_decode(&img.baseline_dri_4row, pool, img.iterations))
                .collect();

            dri4_4t_total += times[times.len() - 1];
            eprint!("{:>16} {:>4}x{:<4}", img.name, img.width, img.height);
            for t in &times {
                eprint!(" {:>6.1}ms", t * 1000.0);
            }
            let t1 = times[0];
            for t in &times[1..] {
                eprint!(" {:>6.2}x", t1 / t);
            }
            eprintln!();
        }

        // === Summary ===
        eprintln!(
            "\n=== Summary (total decode time across {} images) ===",
            images.len()
        );
        eprintln!("Progressive 1T:         {:>7.1}ms", prog_1t_total * 1000.0);
        eprintln!("BL+DRI/1row 1T:         {:>7.1}ms", dri1_1t_total * 1000.0);
        eprintln!(
            "BL+DRI/1row 4T:         {:>7.1}ms  ({:.2}x vs prog 1T)",
            dri1_4t_total * 1000.0,
            prog_1t_total / dri1_4t_total
        );
        eprintln!(
            "BL+DRI/4row 4T:         {:>7.1}ms  ({:.2}x vs prog 1T)",
            dri4_4t_total * 1000.0,
            prog_1t_total / dri4_4t_total
        );
        eprintln!("File size cost (DRI/1row vs prog): +{:.1}%", dri1_vs_prog);
    }
}
