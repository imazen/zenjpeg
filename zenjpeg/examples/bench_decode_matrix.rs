//! Comprehensive decode speed benchmark matrix.
//!
//! Tests all meaningful decode permutations using real CLIC 2025 photos:
//! - 5 source images (every 6th from 30 CLIC final-test)
//! - 6 target sizes (256, 512, 1024, 2048, 4096, 8192)
//! - 4 encoding modes (baseline 4:2:0, baseline 4:4:4, progressive 4:2:0, XYB BQuarter)
//! - 3 quality levels (Q50, Q85, Q95)
//! - 7 decoder variants (mozjpeg, zune, zenjpeg ×4 thread counts, box, scanline)
//!
//! **WARNING**: This benchmark is slow to run (~30+ minutes). The 8192 size in
//! particular takes a long time both for encoding and decoding. Consider commenting
//! out 8192 for quick iteration.
//!
//! **TODO**: Add per-size multiplier columns (e.g. zen-4T/mozjpeg, zen-8T/zen-1T)
//! to make it easier to spot scaling patterns without mental arithmetic.
//!
//! Run with:
//! ```sh
//! cargo run --release --features parallel,decoder --example bench_decode_matrix
//! ```

#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("ERROR: Run with --features parallel,decoder");
    eprintln!("  cargo run --release --features parallel,decoder --example bench_decode_matrix");
}

#[cfg(feature = "parallel")]
fn main() {
    bench::run();
}

#[cfg(feature = "parallel")]
mod bench {
    use enough::Unstoppable;
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::Path;
    use std::time::Instant;
    use zenjpeg::decode::{ChromaUpsampling, Decoder};
    use zenjpeg::decoder::PixelFormat;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, XybSubsampling};

    // --- Image loading and preparation ---

    fn load_png(path: &Path) -> (Vec<rgb::RGB<u8>>, u32, u32) {
        let img = zenjpeg_bench_utils::load_png(path).expect("Failed to load PNG");
        (img.buf().to_vec(), img.width() as u32, img.height() as u32)
    }

    fn center_crop(
        pixels: &[rgb::RGB<u8>],
        src_w: u32,
        src_h: u32,
        target: u32,
    ) -> Vec<rgb::RGB<u8>> {
        let tw = target as usize;
        let th = target as usize;
        let sw = src_w as usize;
        let sh = src_h as usize;

        // Crop to min(src, target) in each dimension, centered
        let crop_w = tw.min(sw);
        let crop_h = th.min(sh);
        let x_off = (sw - crop_w) / 2;
        let y_off = (sh - crop_h) / 2;

        let mut out = vec![rgb::RGB { r: 0, g: 0, b: 0 }; tw * th];
        for y in 0..crop_h {
            let src_row = &pixels[(y_off + y) * sw + x_off..][..crop_w];
            out[y * tw..y * tw + crop_w].copy_from_slice(src_row);
        }
        out
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
            let sy = y % sh;
            for x in 0..tw {
                let sx = x % sw;
                out[y * tw + x] = pixels[sy * sw + sx];
            }
        }
        out
    }

    fn prepare_square(
        pixels: &[rgb::RGB<u8>],
        src_w: u32,
        src_h: u32,
        target: u32,
    ) -> Vec<rgb::RGB<u8>> {
        let min_dim = src_w.min(src_h);
        if target <= min_dim {
            center_crop(pixels, src_w, src_h, target)
        } else {
            tile_to_size(pixels, src_w, src_h, target)
        }
    }

    // --- Encoding ---

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum EncodeMode {
        Baseline420,
        Baseline444,
        Progressive420,
        XybBQuarter,
    }

    impl EncodeMode {
        fn label(self) -> &'static str {
            match self {
                Self::Baseline420 => "bl-420",
                Self::Baseline444 => "bl-444",
                Self::Progressive420 => "prog-420",
                Self::XybBQuarter => "xyb-bq",
            }
        }

        fn is_xyb(self) -> bool {
            matches!(self, Self::XybBQuarter)
        }
    }

    fn encode_jpeg(
        pixels: &[rgb::RGB<u8>],
        w: u32,
        h: u32,
        mode: EncodeMode,
        quality: f32,
    ) -> Vec<u8> {
        let config = match mode {
            EncodeMode::Baseline420 => EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                .progressive(false)
                .restart_mcu_rows(4),
            EncodeMode::Baseline444 => EncoderConfig::ycbcr(quality, ChromaSubsampling::None)
                .progressive(false)
                .restart_mcu_rows(4),
            EncodeMode::Progressive420 => {
                EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).progressive(true)
            }
            EncodeMode::XybBQuarter => EncoderConfig::xyb(quality, XybSubsampling::BQuarter)
                .progressive(false)
                .restart_mcu_rows(4),
        };
        config.encode(pixels, w, h).unwrap()
    }

    // --- Decoders ---

    /// Decode JPEG data using mozjpeg (libjpeg-turbo with NASM SIMD).
    unsafe fn decode_with_mozjpeg(data: &[u8]) -> Vec<u8> {
        use mozjpeg_sys::*;
        use std::mem;

        unsafe {
            let mut err: jpeg_error_mgr = mem::zeroed();
            jpeg_std_error(&mut err);

            let mut cinfo: jpeg_decompress_struct = mem::zeroed();
            cinfo.common.err = &mut err;
            jpeg_create_decompress(&mut cinfo);

            jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);
            jpeg_read_header(&mut cinfo, true as boolean);
            cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
            jpeg_start_decompress(&mut cinfo);

            let width = cinfo.output_width as usize;
            let height = cinfo.output_height as usize;
            let components = cinfo.output_components as usize;
            let row_stride = width * components;

            let mut output = vec![0u8; height * row_stride];

            while (cinfo.output_scanline as usize) < height {
                let offset = cinfo.output_scanline as usize * row_stride;
                let mut row_ptr = output[offset..].as_mut_ptr();
                jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
            }

            jpeg_finish_decompress(&mut cinfo);
            jpeg_destroy_decompress(&mut cinfo);

            output
        }
    }

    fn decode_with_zune(data: &[u8]) -> Vec<u8> {
        use std::io::Cursor;
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::colorspace::ColorSpace;
        use zune_jpeg::zune_core::options::DecoderOptions;

        let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let cursor = Cursor::new(data);
        let mut decoder = JpegDecoder::new_with_options(cursor, options);
        decoder.decode().expect("zune decode failed")
    }

    fn decode_with_zenjpeg(data: &[u8], pool: &rayon::ThreadPool) -> Vec<u8> {
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        pool.install(|| {
            decoder
                .decode(data, Unstoppable)
                .unwrap()
                .into_pixels_u8()
                .unwrap()
        })
    }

    fn decode_with_zenjpeg_box(data: &[u8], pool: &rayon::ThreadPool) -> Vec<u8> {
        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .chroma_upsampling(ChromaUpsampling::NearestNeighbor);
        pool.install(|| {
            decoder
                .decode(data, Unstoppable)
                .unwrap()
                .into_pixels_u8()
                .unwrap()
        })
    }

    fn decode_with_zenjpeg_scanline(data: &[u8]) -> Vec<u8> {
        use imgref::ImgRefMut;
        let decoder = Decoder::new();
        let mut reader = decoder
            .scanline_reader(data)
            .expect("scanline_reader failed");
        let w = reader.width() as usize;
        let h = reader.height() as usize;
        let mut pixels = vec![0u8; w * h * 3];
        let mut rows_read = 0;
        while rows_read < h {
            let remaining = h - rows_read;
            let output = ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
            rows_read += reader.read_rows_rgb8(output).expect("read failed");
        }
        pixels
    }

    // --- Timing ---

    fn bench_median<F: FnMut() -> Vec<u8>>(mut f: F, warmup: usize, iters: usize) -> f64 {
        // Warmup
        for _ in 0..warmup {
            std::hint::black_box(f());
        }

        // Timed iterations
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            std::hint::black_box(f());
            times.push(start.elapsed().as_secs_f64());
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    }

    fn iteration_counts(size: u32) -> (usize, usize) {
        match size {
            0..=512 => (50, 100),
            1024 => (20, 50),
            2048 => (10, 20),
            4096 => (5, 10),
            _ => (3, 5),
        }
    }

    // --- CSV record ---

    struct CsvRecord {
        image: String,
        size: u32,
        mode: EncodeMode,
        quality: u32,
        decoder: String,
        threads: u32,
        upsampler: String,
        time_ms: f64,
        mpix_per_sec: f64,
    }

    // --- Main ---

    pub fn run() {
        let corpus_base = std::path::PathBuf::from(
            std::env::var("CORPUS_PATH")
                .unwrap_or_else(|_| String::from("/home/lilith/work/codec-eval/codec-corpus")),
        );

        let clic_dir = corpus_base.join("clic2025/final-test");
        if !clic_dir.exists() {
            eprintln!("ERROR: CLIC corpus not found at {:?}", clic_dir);
            eprintln!("Set CORPUS_PATH env var or ensure corpus exists");
            return;
        }

        // Load 5 images: every 6th from 30 sorted files
        let mut clic_files: Vec<_> = std::fs::read_dir(&clic_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
            .collect();
        clic_files.sort_by_key(|e| e.file_name());

        let selected: Vec<_> = clic_files.iter().step_by(6).take(5).collect();
        if selected.len() < 5 {
            eprintln!("WARNING: only found {} images (expected 5)", selected.len());
        }

        eprintln!("Loading {} CLIC 2025 images...", selected.len());
        let mut source_images: Vec<(String, Vec<rgb::RGB<u8>>, u32, u32)> = Vec::new();
        for entry in &selected {
            let path = entry.path();
            let name = path.file_stem().unwrap().to_string_lossy();
            let short_name = format!("clic_{}", &name[..8]);
            let (pixels, w, h) = load_png(&path);
            eprintln!("  {} ({}x{})", short_name, w, h);
            source_images.push((short_name, pixels, w, h));
        }

        let target_sizes: &[u32] = &[256, 512, 1024, 2048, 4096, 8192];
        let qualities: &[f32] = &[50.0, 85.0, 95.0];
        let modes = [
            EncodeMode::Baseline420,
            EncodeMode::Baseline444,
            EncodeMode::Progressive420,
            EncodeMode::XybBQuarter,
        ];

        // Build thread pools
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

        // Prepare all square images at all sizes (shared across modes/qualities)
        eprintln!(
            "\nPreparing square images at {} sizes...",
            target_sizes.len()
        );
        // Key: (image_idx, size) -> pixels
        let mut prepared: HashMap<(usize, u32), Vec<rgb::RGB<u8>>> = HashMap::new();
        for (idx, (_name, pixels, w, h)) in source_images.iter().enumerate() {
            for &size in target_sizes {
                let square = prepare_square(pixels, *w, *h, size);
                prepared.insert((idx, size), square);
            }
        }
        eprintln!("Done. {} prepared images.", prepared.len());

        // Encode all combinations
        // Key: (image_idx, size, mode, quality) -> jpeg bytes
        eprintln!("\nEncoding all combinations...");
        let mut encoded: HashMap<(usize, u32, EncodeMode, u32), Vec<u8>> = HashMap::new();
        let total_encodes =
            source_images.len() * target_sizes.len() * modes.len() * qualities.len();
        let mut count = 0;
        for (idx, (name, ..)) in source_images.iter().enumerate() {
            for &size in target_sizes {
                let pixels = &prepared[&(idx, size)];
                for &mode in &modes {
                    for &q in qualities {
                        let qi = q as u32;
                        let jpeg = encode_jpeg(pixels, size, size, mode, q);
                        count += 1;
                        if count % 20 == 0 || count == total_encodes {
                            eprint!(
                                "\r  {}/{} encoded ({} {}x{} {} Q{}  {} bytes)    ",
                                count,
                                total_encodes,
                                name,
                                size,
                                size,
                                mode.label(),
                                qi,
                                jpeg.len()
                            );
                        }
                        encoded.insert((idx, size, mode, qi), jpeg);
                    }
                }
            }
        }
        eprintln!();

        // Free prepared pixel data — we only need the encoded JPEGs now
        drop(prepared);

        // Define decoder variants
        #[derive(Debug, Clone)]
        struct DecoderVariant {
            name: String,
            threads: u32,
            upsampler: String,
            /// Can decode non-XYB only
            xyb_capable: bool,
        }

        let mut variants: Vec<DecoderVariant> = Vec::new();
        // mozjpeg (1T only, no XYB)
        variants.push(DecoderVariant {
            name: "mozjpeg".into(),
            threads: 1,
            upsampler: "libjpeg".into(),
            xyb_capable: false,
        });
        // zune (1T only, no XYB)
        variants.push(DecoderVariant {
            name: "zune".into(),
            threads: 1,
            upsampler: "zune".into(),
            xyb_capable: false,
        });
        // zenjpeg triangle at 1, 2, 4, 8 threads
        for &t in &thread_counts {
            variants.push(DecoderVariant {
                name: format!("zen-{}T", t),
                threads: t as u32,
                upsampler: "triangle".into(),
                xyb_capable: true,
            });
        }
        // zenjpeg box at 1, 4 threads
        for &t in &[1, 4] {
            variants.push(DecoderVariant {
                name: format!("box-{}T", t),
                threads: t as u32,
                upsampler: "box".into(),
                xyb_capable: true,
            });
        }
        // zenjpeg scanline (1T only)
        variants.push(DecoderVariant {
            name: "scanline".into(),
            threads: 1,
            upsampler: "scanline".into(),
            xyb_capable: true,
        });

        // Run benchmarks
        let mut csv_records: Vec<CsvRecord> = Vec::new();

        for &mode in &modes {
            for &q in qualities {
                let qi = q as u32;
                eprintln!("\n{}", "=".repeat(60));
                eprintln!("=== {}, Q{} ===", mode.label(), qi);
                eprintln!("{}", "=".repeat(60));

                // Filter variants for this mode
                let active_variants: Vec<&DecoderVariant> = variants
                    .iter()
                    .filter(|v| !mode.is_xyb() || v.xyb_capable)
                    .collect();

                // Print header
                eprint!("{:>7}", "Size");
                for v in &active_variants {
                    eprint!(" {:>9}", v.name);
                }
                eprintln!();
                eprint!("{:>7}", "");
                for _ in &active_variants {
                    eprint!(" {:>9}", "---------");
                }
                eprintln!();

                for &size in target_sizes {
                    let (warmup, iters) = iteration_counts(size);
                    let mpix = (size as f64 * size as f64) / 1e6;

                    // Average across images
                    let mut variant_times: Vec<f64> = vec![0.0; active_variants.len()];
                    let mut image_count = 0;

                    for (idx, _) in source_images.iter().enumerate() {
                        let key = (idx, size, mode, qi);
                        let jpeg = match encoded.get(&key) {
                            Some(j) => j,
                            None => continue,
                        };

                        image_count += 1;

                        for (vi, v) in active_variants.iter().enumerate() {
                            let time = match (v.upsampler.as_str(), v.name.as_str()) {
                                ("libjpeg", _) => {
                                    let data = jpeg.clone();
                                    bench_median(
                                        || unsafe { decode_with_mozjpeg(&data) },
                                        warmup,
                                        iters,
                                    )
                                }
                                ("zune", _) => {
                                    let data = jpeg.clone();
                                    bench_median(|| decode_with_zune(&data), warmup, iters)
                                }
                                ("triangle", _) => {
                                    let pool_idx = thread_counts
                                        .iter()
                                        .position(|&t| t == v.threads as usize)
                                        .unwrap();
                                    let pool = &pools[pool_idx].1;
                                    let data = jpeg.clone();
                                    bench_median(|| decode_with_zenjpeg(&data, pool), warmup, iters)
                                }
                                ("box", _) => {
                                    let pool_idx = thread_counts
                                        .iter()
                                        .position(|&t| t == v.threads as usize)
                                        .unwrap();
                                    let pool = &pools[pool_idx].1;
                                    let data = jpeg.clone();
                                    bench_median(
                                        || decode_with_zenjpeg_box(&data, pool),
                                        warmup,
                                        iters,
                                    )
                                }
                                ("scanline", _) => {
                                    let data = jpeg.clone();
                                    bench_median(
                                        || decode_with_zenjpeg_scanline(&data),
                                        warmup,
                                        iters,
                                    )
                                }
                                _ => unreachable!(),
                            };
                            variant_times[vi] += time;
                        }
                    }

                    // Average
                    if image_count > 0 {
                        for t in &mut variant_times {
                            *t /= image_count as f64;
                        }
                    }

                    // Print row
                    eprint!("{:>7}", size);
                    for &t in &variant_times {
                        eprint!(" {:>9}", format_time(t));
                    }
                    eprintln!();

                    // Store CSV records
                    for (vi, v) in active_variants.iter().enumerate() {
                        let t = variant_times[vi];
                        csv_records.push(CsvRecord {
                            image: "avg".into(),
                            size,
                            mode,
                            quality: qi,
                            decoder: v.name.clone(),
                            threads: v.threads,
                            upsampler: v.upsampler.clone(),
                            time_ms: t * 1000.0,
                            mpix_per_sec: mpix / t,
                        });
                    }
                }

                // Print speedup row (zen-1T = baseline)
                eprintln!();
                let zen1_idx = active_variants.iter().position(|v| v.name == "zen-1T");
                if let Some(_z1) = zen1_idx {
                    eprint!("{:>7}", "vs z1T");
                    for (vi, _) in active_variants.iter().enumerate() {
                        // Find corresponding records
                        let this_records: Vec<_> = csv_records
                            .iter()
                            .filter(|r| {
                                r.mode == mode
                                    && r.quality == qi
                                    && r.decoder == active_variants[vi].name
                            })
                            .collect();
                        let base_records: Vec<_> = csv_records
                            .iter()
                            .filter(|r| r.mode == mode && r.quality == qi && r.decoder == "zen-1T")
                            .collect();

                        if this_records.len() == base_records.len() && !this_records.is_empty() {
                            let ratio: f64 = base_records
                                .iter()
                                .zip(this_records.iter())
                                .map(|(b, t)| b.time_ms / t.time_ms)
                                .sum::<f64>()
                                / this_records.len() as f64;
                            eprint!(" {:>8.2}x", ratio);
                        } else {
                            eprint!(" {:>9}", "");
                        }
                    }
                    eprintln!();
                }
            }
        }

        // Write CSV
        let csv_dir = zenjpeg_bench_utils::zenjpeg_output_dir().join("decode-matrix");
        if let Err(e) = std::fs::create_dir_all(&csv_dir) {
            eprintln!("\nWARNING: Could not create CSV dir {:?}: {}", csv_dir, e);
            eprintln!("Writing CSV to current directory instead.");
            write_csv(Path::new("decode-matrix-results.csv"), &csv_records);
        } else {
            write_csv(&csv_dir.join("results.csv"), &csv_records);
        }

        eprintln!("\nDone! {} total measurements.", csv_records.len());
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

    fn write_csv(path: &Path, records: &[CsvRecord]) {
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(
            f,
            "image,size,mode,quality,decoder,threads,upsampler,time_ms,mpix_per_sec"
        )
        .unwrap();
        for r in records {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:.3},{:.1}",
                r.image,
                r.size,
                r.mode.label(),
                r.quality,
                r.decoder,
                r.threads,
                r.upsampler,
                r.time_ms,
                r.mpix_per_sec,
            )
            .unwrap();
        }
        eprintln!("\nCSV written to {:?} ({} records)", path, records.len());
    }
}
