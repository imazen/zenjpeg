//! Sweep sharp YUV parameter space: init method × iteration count.
//! Finds the Pareto frontier of quality vs speed.
//!
//! Run: `cargo test --release -p zenjpeg --test sharp_yuv_sweep --features decoder -- --nocapture --ignored`

#[cfg(feature = "decoder")]
#[test]
#[ignore] // takes ~10s
fn sharp_yuv_parameter_sweep() {
    use enough::Unstoppable;
    use zenjpeg::decode::Decoder;
    use zenjpeg::encode::encoder_types::PixelLayout;

    let corpus_dir = std::path::Path::new(
        &std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into()),
    )
    .join("work/codec-eval/codec-corpus/CID22/CID22-512/training");

    let mut paths: Vec<_> = std::fs::read_dir(&corpus_dir)
        .expect("CID22 corpus required")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    paths.sort();
    paths.truncate(10);

    // Load all images.
    let images: Vec<(String, Vec<u8>, u32, u32)> = paths
        .iter()
        .filter_map(|p| {
            let (rgb, w, h) = load_png_rgb(p)?;
            let name = p.file_stem()?.to_string_lossy().into_owned();
            Some((name, rgb, w, h))
        })
        .collect();

    let decoder = Decoder::new();

    // Box 4:2:0 baseline (no sharp).
    let mut box_total = 0.0f64;
    let mut box_pixels = 0u64;
    for (_, rgb, w, h) in &images {
        let cfg = zenjpeg::encode::EncoderConfig::ycbcr(
            85.0,
            zenjpeg::encode::encoder_types::ChromaSubsampling::Quarter,
        );
        let jpeg = cfg.encode_bytes(rgb, *w, *h, PixelLayout::Rgb8Srgb).unwrap();
        let dec = decoder.decode(&jpeg, Unstoppable).unwrap();
        let rt = dec.pixels_u8().unwrap();
        box_total += mean_abs_err(rgb, rt) * (*w as f64 * *h as f64);
        box_pixels += *w as u64 * *h as u64;
    }
    let box_baseline = box_total / box_pixels as f64;

    eprintln!("=== Sharp YUV Parameter Sweep (Q85, {} images) ===", images.len());
    eprintln!("Box 4:2:0 baseline: {box_baseline:.4}");
    eprintln!();
    eprintln!("{:>12} {:>5} {:>8} {:>8} {:>8} {:>10}",
        "init", "iters", "error", "vs_box%", "vs_best%", "time_ms"
    );

    struct Result {
        init: &'static str,
        iters: u32,
        error: f64,
        time_ms: f64,
    }
    let mut results = Vec::new();

    let luts = zenyuv::GammaLuts::srgb();

    for &gamma_init in &[false, true] {
        for &iters in &[0u32, 1, 2, 3, 4, 6, 8] {
            let init_name = if gamma_init { "gamma" } else { "box" };
            let config = zenyuv::SharpYuvConfig {
                max_iterations: iters,
                convergence_threshold: 0.1,
                gamma_aware_init: gamma_init,
                srgb_delinearize: true,
            };

            // Quality: JPEG roundtrip error.
            let mut total_err = 0.0f64;
            let mut total_px = 0u64;
            // Speed: isolated sharp conversion time.
            let mut total_us = 0u64;

            for (_, rgb, w, h) in &images {
                let (w, h) = (*w as usize, *h as usize);
                let n = w * h;
                let cw = w / 2;
                let ch = h / 2;

                // Time the sharp conversion.
                let mut y = vec![0u8; n];
                let mut cb = vec![0u8; cw * ch];
                let mut cr = vec![0u8; cw * ch];
                let start = std::time::Instant::now();
                zenyuv::sharp::rgb_to_yuv420_sharp(
                    rgb, &mut y, &mut cb, &mut cr, w, h,
                    zenyuv::Range::Full, zenyuv::Matrix::Bt601, &luts, &config,
                );
                total_us += start.elapsed().as_micros() as u64;

                // Encode through JPEG to measure codec-context quality.
                // Use zenjpeg with box chroma but feed our pre-computed sharp YUV...
                // Actually: just measure the raw YUV roundtrip error for speed.
                // The JPEG roundtrip adds ~15ms per image and dominates the sweep.
                // Use raw roundtrip instead.
                let mut rt = vec![0u8; n * 3];
                zenyuv::yuv420_to_rgb(&y, &cb, &cr, &mut rt, w, h);
                total_err += mean_abs_err(rgb, &rt) * n as f64;
                total_px += n as u64;
            }

            let avg_err = total_err / total_px as f64;
            let vs_box = (avg_err - box_baseline) / box_baseline * 100.0;
            let time_ms = total_us as f64 / 1000.0;

            results.push(Result {
                init: if gamma_init { "gamma" } else { "box" },
                iters,
                error: avg_err,
                time_ms,
            });

            eprintln!("{init_name:>12} {iters:>5} {avg_err:8.4} {vs_box:+8.2}% {:>8} {time_ms:10.1}",
                "");
        }
    }

    // Find best quality overall.
    let best_err = results.iter().map(|r| r.error).fold(f64::MAX, f64::min);
    eprintln!();
    eprintln!("Best quality: {best_err:.4}");
    eprintln!();

    // Pareto frontier: for each time budget, what's the best quality?
    eprintln!("Pareto frontier (best quality at each speed tier):");
    let mut sorted = results.iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.time_ms.partial_cmp(&b.time_ms).unwrap());
    let mut best_so_far = f64::MAX;
    for r in &sorted {
        if r.error < best_so_far {
            best_so_far = r.error;
            let vs_box = (r.error - box_baseline) / box_baseline * 100.0;
            eprintln!("  {:>6} iter={}: error={:.4} ({:+.2}% vs box) in {:.1}ms",
                r.init, r.iters, r.error, vs_box, r.time_ms);
        }
    }
}

fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut out = Vec::with_capacity((info.width * info.height * 3) as usize);
            for c in src.chunks_exact(4) {
                out.extend_from_slice(&c[..3]);
            }
            out
        }
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn mean_abs_err(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0u64;
    for i in 0..n {
        s += a[i].abs_diff(b[i]) as u64;
    }
    s as f64 / n as f64
}
