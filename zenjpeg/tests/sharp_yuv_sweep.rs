//! Sweep sharp YUV iteration count on real images.
//!
//! Run: `cargo test --release -p zenjpeg --test sharp_yuv_sweep --features decoder -- --nocapture --ignored`

#[test]
#[ignore] // takes ~10s
fn sharp_yuv_iteration_sweep() {
    // Speed sweep only — no decoder needed.

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

    let images: Vec<(String, Vec<u8>, u32, u32)> = paths
        .iter()
        .filter_map(|p| {
            let (rgb, w, h) = load_png_rgb(p)?;
            let name = p.file_stem()?.to_string_lossy().into_owned();
            Some((name, rgb, w, h))
        })
        .collect();

    eprintln!("=== Sharp YUV Iteration Sweep (Q85, {} images) ===", images.len());
    eprintln!("{:>6} {:>8} {:>10}", "iters", "error", "time_ms");

    let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);

    for &iters in &[0u32, 1, 2, 3, 4, 6, 8] {
        let config = zenyuv::SharpYuvConfig {
            max_iterations: iters,
            ..Default::default()
        };

        let mut total_err = 0.0f64;
        let mut total_px = 0u64;
        let mut total_us = 0u64;

        for (_, rgb, w, h) in &images {
            let (w, h) = (*w as usize, *h as usize);
            let n = w * h;
            let cw = w / 2;
            let ch = h / 2;

            let mut y = vec![0u8; n];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            let start = std::time::Instant::now();
            ctx.encode_sharp_420_u8(rgb, &mut y, &mut cb, &mut cr, w, h, &config);
            total_us += start.elapsed().as_micros() as u64;

            // Skip quality measurement — this is a speed sweep only.
            // Quality is verified in sharp_yuv_jpeg_roundtrip.rs.
            total_err += 0.0;
            total_px += n as u64;
        }

        let avg_err = total_err / total_px as f64;
        let time_ms = total_us as f64 / 1000.0;
        eprintln!("{iters:>6} {avg_err:8.4} {time_ms:10.1}");
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
