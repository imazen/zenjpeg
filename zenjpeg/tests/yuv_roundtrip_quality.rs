//! True YUV encoder quality test: RGB → YUV 4:2:0 → libjpeg-turbo h2v2_fancy
//! upsample → RGB.
//!
//! This is what actually matters — encoder choice is invisible without seeing
//! how the subsampled chroma reconstructs through a real fancy upsampler.
//! Sharp YUV exists specifically to win this metric.
//!
//! Uses zenjpeg's `upsample_h2v2_i16_libjpeg`, which is a bit-exact port of
//! libjpeg-turbo's `jdsample.c` h2v2_fancy_upsample.
//!
//! Run: `cargo test --release -p zenjpeg --features __test-utils --test
//! yuv_roundtrip_quality -- --nocapture --ignored`
//!
//! Results format (per encoder, averaged over corpus):
//!   RMSE / MAE / max_err — error vs original RGB after roundtrip
//!   per-channel RMSE — separate R, G, B (chroma loss affects channels differently)
//!   time_us/MP — encode cost per megapixel (decode cost is constant)

#![cfg(feature = "__test-utils")]

use zenjpeg::decode::upsample::upsample_h2v2_i16_libjpeg;
use zenyuv::{Matrix, Range, SharpYuvConfig, YuvContext};

// ── Decode pipeline (zenjpeg's real h2v2_fancy + BT.601 inverse) ─────────────

fn fancy_decode_420_to_rgb(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let n = width * height;

    // Convert chroma planes to i16 (zenjpeg's upsampler signature).
    let cb_i16: Vec<i16> = cb_plane.iter().map(|&v| v as i16).collect();
    let cr_i16: Vec<i16> = cr_plane.iter().map(|&v| v as i16).collect();

    let mut cb_full_i16 = vec![0i16; n];
    let mut cr_full_i16 = vec![0i16; n];
    upsample_h2v2_i16_libjpeg(&cb_i16, cw, ch, &mut cb_full_i16, width, height);
    upsample_h2v2_i16_libjpeg(&cr_i16, cw, ch, &mut cr_full_i16, width, height);

    // YCbCr → RGB (BT.601 full-range, f32 reference math).
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        let y = y_plane[i] as f32;
        let cb = cb_full_i16[i].clamp(0, 255) as f32 - 128.0;
        let cr = cr_full_i16[i].clamp(0, 255) as f32 - 128.0;
        let r = y + 1.402 * cr;
        let g = y - 0.344_136 * cb - 0.714_136 * cr;
        let b = y + 1.772 * cb;
        rgb[i * 3] = r.round().clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 1] = g.round().clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 2] = b.round().clamp(0.0, 255.0) as u8;
    }
    rgb
}

// ── Metrics ──────────────────────────────────────────────────────────────────

fn sum_sq(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum()
}

fn sum_abs(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.abs_diff(*y) as u64)
        .sum()
}

fn max_err(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// Per-channel sum of squared errors for packed RGB.
fn sum_sq_per_channel(a: &[u8], b: &[u8]) -> [f64; 3] {
    let mut s = [0.0f64; 3];
    for (chunk_a, chunk_b) in a.chunks_exact(3).zip(b.chunks_exact(3)) {
        for c in 0..3 {
            let d = chunk_a[c] as f64 - chunk_b[c] as f64;
            s[c] += d * d;
        }
    }
    s
}

// ── PNG loader (crops to even dimensions for 4:2:0) ──────────────────────────

fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let raw_w = info.width as usize;
    let raw_h = info.height as usize;
    let raw: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut out = Vec::with_capacity(raw_w * raw_h * 3);
            for c in src.chunks_exact(4) {
                out.extend_from_slice(&c[..3]);
            }
            out
        }
        _ => return None,
    };
    let w = raw_w & !1;
    let h = raw_h & !1;
    if w == raw_w && h == raw_h {
        return Some((raw, w as u32, h as u32));
    }
    let mut cropped = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        let src_off = row * raw_w * 3;
        cropped.extend_from_slice(&raw[src_off..src_off + w * 3]);
    }
    Some((cropped, w as u32, h as u32))
}

// ── Encoders under test ──────────────────────────────────────────────────────

type EncodeFn = Box<dyn Fn(&[u8], usize, usize) -> (Vec<u8>, Vec<u8>, Vec<u8>)>;

fn encode_zenyuv_box(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
    ctx.encode_420_u8(rgb, &mut y, &mut cb, &mut cr, w, h);
    (y, cb, cr)
}

fn encode_zenyuv_sharp(rgb: &[u8], w: usize, h: usize, iters: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
    let config = SharpYuvConfig {
        max_iterations: iters,
        ..Default::default()
    };
    ctx.encode_sharp_420_u8(rgb, &mut y, &mut cb, &mut cr, w, h, &config);
    (y, cb, cr)
}

// ── Test ─────────────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct Accum {
    name: &'static str,
    sum_sq: f64,
    sum_sq_per_ch: [f64; 3],
    sum_abs: u64,
    worst_max: u8,
    total_px: u64,
    total_us: u64,
}

#[test]
#[ignore]
fn yuv_roundtrip_quality_cid22() {
    let corpus_dir =
        std::path::Path::new(&std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into()))
            .join("work/codec-eval/codec-corpus/CID22/CID22-512/training");

    let n_images: usize = std::env::var("N_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

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
    paths.truncate(n_images);

    let images: Vec<(String, Vec<u8>, usize, usize)> = paths
        .iter()
        .filter_map(|p| {
            let (rgb, w, h) = load_png_rgb(p)?;
            let name = p.file_stem()?.to_string_lossy().into_owned();
            Some((name, rgb, w as usize, h as usize))
        })
        .collect();

    eprintln!();
    eprintln!("=== YUV Roundtrip Quality: 4:2:0 + libjpeg-turbo h2v2_fancy upsample ===");
    eprintln!(
        "{} images from CID22-512/training (override with N_IMAGES env var)",
        images.len()
    );
    eprintln!();
    eprintln!(
        "{:>22} | {:>8} {:>8} | {:>7} {:>7} {:>7} | {:>5} | {:>10}",
        "encoder", "RMSE", "MAE", "RMSE_R", "RMSE_G", "RMSE_B", "max", "us/MP"
    );
    eprintln!("{}", "-".repeat(94));

    let encoders: &[(&'static str, EncodeFn)] = &[
        ("zenyuv box", Box::new(encode_zenyuv_box)),
        (
            "zenyuv sharp iters=1",
            Box::new(|r, w, h| encode_zenyuv_sharp(r, w, h, 1)),
        ),
        (
            "zenyuv sharp iters=2",
            Box::new(|r, w, h| encode_zenyuv_sharp(r, w, h, 2)),
        ),
        (
            "zenyuv sharp iters=4",
            Box::new(|r, w, h| encode_zenyuv_sharp(r, w, h, 4)),
        ),
        (
            "zenyuv sharp iters=8",
            Box::new(|r, w, h| encode_zenyuv_sharp(r, w, h, 8)),
        ),
    ];

    let mut results: Vec<Accum> = encoders
        .iter()
        .map(|(name, _)| Accum {
            name,
            ..Default::default()
        })
        .collect();

    for (_name, rgb, w, h) in &images {
        let px = (w * h) as u64;
        for (idx, (_, encode)) in encoders.iter().enumerate() {
            let start = std::time::Instant::now();
            let (y, cb, cr) = encode(rgb, *w, *h);
            let elapsed_us = start.elapsed().as_micros() as u64;
            let decoded = fancy_decode_420_to_rgb(&y, &cb, &cr, *w, *h);

            let acc = &mut results[idx];
            acc.sum_sq += sum_sq(rgb, &decoded);
            let per_ch = sum_sq_per_channel(rgb, &decoded);
            for c in 0..3 {
                acc.sum_sq_per_ch[c] += per_ch[c];
            }
            acc.sum_abs += sum_abs(rgb, &decoded);
            acc.worst_max = acc.worst_max.max(max_err(rgb, &decoded));
            acc.total_px += px;
            acc.total_us += elapsed_us;
        }
    }

    let mut rmse_box = 0.0;
    let mut rmse_sharp2 = 0.0;
    for acc in &results {
        let total_samples = acc.total_px * 3;
        let rmse = (acc.sum_sq / total_samples as f64).sqrt();
        let mae = acc.sum_abs as f64 / total_samples as f64;
        let rmse_r = (acc.sum_sq_per_ch[0] / acc.total_px as f64).sqrt();
        let rmse_g = (acc.sum_sq_per_ch[1] / acc.total_px as f64).sqrt();
        let rmse_b = (acc.sum_sq_per_ch[2] / acc.total_px as f64).sqrt();
        let us_per_mp = acc.total_us as f64 * 1_000_000.0 / acc.total_px as f64;
        eprintln!(
            "{:>22} | {:>8.4} {:>8.4} | {:>7.4} {:>7.4} {:>7.4} | {:>5} | {:>10.1}",
            acc.name, rmse, mae, rmse_r, rmse_g, rmse_b, acc.worst_max, us_per_mp
        );
        if acc.name.starts_with("zenyuv box") {
            rmse_box = rmse;
        }
        if acc.name.contains("iters=2") {
            rmse_sharp2 = rmse;
        }
    }
    eprintln!();

    // Regression gates:
    //   1. zenyuv box must match (or beat) yuv crate within 0.5% RMSE.
    //   2. Sharp iters=2 must improve RMSE by at least 1.5% over box average.
    let yuv_crate_rmse = {
        let acc = &results[1];
        (acc.sum_sq / (acc.total_px * 3) as f64).sqrt()
    };
    let box_vs_yuv_pct = (rmse_box - yuv_crate_rmse) / yuv_crate_rmse * 100.0;
    let sharp_gain_pct = (rmse_box - rmse_sharp2) / rmse_box * 100.0;
    eprintln!("zenyuv box vs yuv crate: {box_vs_yuv_pct:+.3}% RMSE (must be < +0.5%)");
    eprintln!("sharp iters=2 vs box: {sharp_gain_pct:+.3}% RMSE improvement (must be > 1.5%)");
    eprintln!();

    assert!(
        box_vs_yuv_pct < 0.5,
        "zenyuv box regressed vs yuv crate by {box_vs_yuv_pct:.3}% RMSE"
    );
    assert!(
        sharp_gain_pct > 1.5,
        "Sharp YUV iters=2 only improved by {sharp_gain_pct:.3}% RMSE (expected > 1.5%)"
    );
}
