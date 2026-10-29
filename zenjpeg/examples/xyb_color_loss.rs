//! Brute-force diagnostic: how much of sRGB space survives an XYB roundtrip?
//!
//! Constructs a synthetic image that contains a coarse 3D grid of sRGB
//! samples (256 / step values per channel), encodes via XYB, decodes, and
//! reports per-pixel ΔE-like deltas plus a coverage map of which sRGB
//! values are reachable in the decoded output.
//!
//! Two encode configurations are compared back-to-back:
//!   - `xyb_full_q100`: XYB Full at Q100 (the highest-quality XYB recipe
//!     this codec offers)
//!   - `ycbcr_444_q100`: YCbCr 4:4:4 progressive at Q100 (a control to see
//!     where YCbCr is exact and XYB is lossy)
//!
//! Use the output to identify "sparse zones" — sRGB regions where the
//! decoded output collapses many input values onto few outputs (saturated
//! primaries, shadow detail, etc.).
//!
//! Run:
//!   cargo run --release -p zenjpeg --example xyb_color_loss \
//!     --features "trellis decoder moxcms" -- [step]
//!
//! `step` controls the sRGB grid resolution. Default 16 produces a
//! 16³=4096-pixel grid (one row per channel triple, image 4096×1).
//! Smaller values (e.g. 8 → 32768 pixels) give finer resolution but bigger
//! decode error since the encoder may smear isolated grid points across
//! 8×8 DCT blocks.

use enough::Unstoppable;
use zenjpeg::color::icc::TargetColorSpace;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout, XybSubsampling};

/// Build an N×N sRGB grid image: one block of 8×8 px per (R, G, B) triple
/// so DCT block boundaries don't smear adjacent grid samples together.
fn build_grid_image(step: u32) -> (u32, u32, Vec<u8>, Vec<(u8, u8, u8)>) {
    assert!(step > 0 && step <= 256);
    let n = 256 / step.max(1); // samples per channel
    let total = n * n * n;
    let block = 8u32; // one 8×8 px block per sample so JPEG sees a uniform color
    let cols_per_row = 256u32; // wrap into a roughly square image
    let cols = cols_per_row.min(total);
    let rows = (total + cols - 1) / cols;
    let w = cols * block;
    let h = rows * block;
    let mut rgb = vec![0u8; (w * h) as usize * 3];
    let mut samples = Vec::with_capacity(total as usize);

    for sample_idx in 0..total {
        let cell_x = sample_idx % cols;
        let cell_y = sample_idx / cols;
        // R outer loop, G middle, B inner: (sample_idx) → (r,g,b) on uniform grid.
        let bi = sample_idx % n;
        let gi = (sample_idx / n) % n;
        let ri = (sample_idx / (n * n)) % n;
        // Center of the bin so we don't sit on a quantization boundary.
        let half = step / 2;
        let r = (ri * step + half).min(255) as u8;
        let g = (gi * step + half).min(255) as u8;
        let b = (bi * step + half).min(255) as u8;
        samples.push((r, g, b));

        // Fill the 8×8 block.
        for dy in 0..block {
            for dx in 0..block {
                let px = cell_x * block + dx;
                let py = cell_y * block + dy;
                let i = (py * w + px) as usize * 3;
                rgb[i] = r;
                rgb[i + 1] = g;
                rgb[i + 2] = b;
            }
        }
    }
    (w, h, rgb, samples)
}

fn read_sample_pixel(decoded: &[u8], w: u32, sample_idx: u32, cols: u32) -> (u8, u8, u8) {
    let cell_x = sample_idx % cols;
    let cell_y = sample_idx / cols;
    let block = 8u32;
    // Sample the center of the block (avoids any block-edge ringing).
    let px = cell_x * block + block / 2;
    let py = cell_y * block + block / 2;
    let i = (py * w + px) as usize * 3;
    (decoded[i], decoded[i + 1], decoded[i + 2])
}

#[derive(Default)]
struct Stats {
    n: u64,
    sum_dr: i64,
    sum_dg: i64,
    sum_db: i64,
    sum_abs_dr: u64,
    sum_abs_dg: u64,
    sum_abs_db: u64,
    sum_sq_d: u64, // ΔE² (Euclidean RGB)
    max_d: u32,
    n_within_1: u64,
    n_within_4: u64,
    n_within_16: u64,
    distinct_outputs: std::collections::HashSet<(u8, u8, u8)>,
}

impl Stats {
    fn observe(&mut self, src: (u8, u8, u8), dst: (u8, u8, u8)) {
        let dr = dst.0 as i32 - src.0 as i32;
        let dg = dst.1 as i32 - src.1 as i32;
        let db = dst.2 as i32 - src.2 as i32;
        let d2 = (dr * dr + dg * dg + db * db) as u32;
        let d = (d2 as f32).sqrt() as u32;
        self.n += 1;
        self.sum_dr += dr as i64;
        self.sum_dg += dg as i64;
        self.sum_db += db as i64;
        self.sum_abs_dr += dr.unsigned_abs() as u64;
        self.sum_abs_dg += dg.unsigned_abs() as u64;
        self.sum_abs_db += db.unsigned_abs() as u64;
        self.sum_sq_d += d2 as u64;
        self.max_d = self.max_d.max(d);
        if d <= 1 {
            self.n_within_1 += 1;
        }
        if d <= 4 {
            self.n_within_4 += 1;
        }
        if d <= 16 {
            self.n_within_16 += 1;
        }
        self.distinct_outputs.insert(dst);
    }

    fn report(&self, label: &str, n_inputs: usize) {
        let n = self.n.max(1) as f64;
        let rmse = (self.sum_sq_d as f64 / n).sqrt();
        let mae_r = self.sum_abs_dr as f64 / n;
        let mae_g = self.sum_abs_dg as f64 / n;
        let mae_b = self.sum_abs_db as f64 / n;
        let bias_r = self.sum_dr as f64 / n;
        let bias_g = self.sum_dg as f64 / n;
        let bias_b = self.sum_db as f64 / n;
        let coverage = self.distinct_outputs.len() as f64 / n_inputs as f64 * 100.0;
        println!("\n== {label} ==");
        println!("  samples:           {}", self.n);
        println!(
            "  distinct outputs:  {} ({coverage:.1}% of inputs)",
            self.distinct_outputs.len()
        );
        println!("  per-pixel RMSE:    {rmse:.2}");
        println!("  per-channel MAE:   R={mae_r:.2} G={mae_g:.2} B={mae_b:.2}");
        println!("  per-channel bias:  R={bias_r:+.2} G={bias_g:+.2} B={bias_b:+.2}");
        println!("  max ΔE:            {}", self.max_d);
        println!(
            "  within ΔE≤1:       {} ({:.1}%)",
            self.n_within_1,
            self.n_within_1 as f64 / n * 100.0
        );
        println!(
            "  within ΔE≤4:       {} ({:.1}%)",
            self.n_within_4,
            self.n_within_4 as f64 / n * 100.0
        );
        println!(
            "  within ΔE≤16:      {} ({:.1}%)",
            self.n_within_16,
            self.n_within_16 as f64 / n * 100.0
        );
    }
}

fn run_one(
    label: &str,
    jpeg: &[u8],
    samples: &[(u8, u8, u8)],
    w: u32,
    cols: u32,
    with_correct_color: bool,
) {
    let mut decoder = Decoder::new();
    if with_correct_color {
        decoder = decoder.correct_color(Some(TargetColorSpace::Srgb));
    }
    let decoded = decoder.decode(jpeg, Unstoppable).expect("decode");
    let pixels = decoded.pixels_u8().unwrap();
    let mut stats = Stats::default();
    for (i, &src) in samples.iter().enumerate() {
        let dst = read_sample_pixel(pixels, w, i as u32, cols);
        stats.observe(src, dst);
    }
    stats.report(label, samples.len());
}

fn worst_zones(jpeg: &[u8], samples: &[(u8, u8, u8)], w: u32, cols: u32, k: usize, label: &str) {
    let decoded = Decoder::new()
        .correct_color(Some(TargetColorSpace::Srgb))
        .decode(jpeg, Unstoppable)
        .expect("decode");
    let pixels = decoded.pixels_u8().unwrap();
    let mut deltas: Vec<((u8, u8, u8), (u8, u8, u8), u32)> = samples
        .iter()
        .enumerate()
        .map(|(i, &src)| {
            let dst = read_sample_pixel(pixels, w, i as u32, cols);
            let d2 = (src.0 as i32 - dst.0 as i32).pow(2)
                + (src.1 as i32 - dst.1 as i32).pow(2)
                + (src.2 as i32 - dst.2 as i32).pow(2);
            (src, dst, d2 as u32)
        })
        .collect();
    deltas.sort_by_key(|&(_, _, d2)| std::cmp::Reverse(d2));
    println!("\n== {label}: worst {k} sRGB samples by ΔE² ==");
    for (src, dst, d2) in deltas.iter().take(k) {
        let d = (*d2 as f32).sqrt();
        println!(
            "  src=({:>3},{:>3},{:>3}) -> dst=({:>3},{:>3},{:>3})  ΔE={d:5.1}",
            src.0, src.1, src.2, dst.0, dst.1, dst.2
        );
    }
}

fn main() {
    let step: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    println!("Building sRGB grid: step={step} ({}³ samples)", 256 / step);
    let (w, h, rgb, samples) = build_grid_image(step);
    let cols = (256u32).min(samples.len() as u32);
    println!(
        "  image: {w}×{h} ({} px), {} sRGB samples\n",
        w * h,
        samples.len()
    );

    println!("Encoding XYB Full Q100 progressive…");
    let xyb = EncoderConfig::xyb(100.0, XybSubsampling::Full)
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("xyb encode");
    println!("  bytes={}", xyb.len());

    println!("Encoding YCbCr 4:4:4 progressive Q100…");
    let ycbcr = EncoderConfig::ycbcr(100.0, ChromaSubsampling::None)
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("ycbcr encode");
    println!("  bytes={}\n", ycbcr.len());

    // Roundtrip via different decode paths.
    run_one(
        "XYB Full Q100, default decode (no correct_color)",
        &xyb,
        &samples,
        w,
        cols,
        false,
    );
    run_one(
        "XYB Full Q100, correct_color(Srgb)",
        &xyb,
        &samples,
        w,
        cols,
        true,
    );
    run_one(
        "YCbCr 4:4:4 Q100 (control)",
        &ycbcr,
        &samples,
        w,
        cols,
        false,
    );

    // Identify worst sRGB zones for the recommended XYB path.
    worst_zones(
        &xyb,
        &samples,
        w,
        cols,
        12,
        "XYB Full Q100 + correct_color(Srgb)",
    );
    worst_zones(&ycbcr, &samples, w, cols, 12, "YCbCr 4:4:4 Q100");
}
