//! Encode-time overhead measurement for Phase 2 boundary-RD (#91).
//!
//! Run with `--ignored` to include (kept off the default run to avoid
//! variance-sensitive timing in CI).

use enough::Unstoppable;
use std::time::Instant;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

#[test]
#[ignore]
fn boundary_rd_encode_overhead_at_q85() {
    let (w, h) = (512usize, 512usize);
    let mut rgb = vec![0u8; w * h * 3];
    let mut s: u64 = 0xDEAD_BEEF_0123_4567;
    for y in 0..h {
        for x in 0..w {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let n = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let noise = ((n >> 32) & 0xFF) as u8;
            let patch = if (x / 32 % 2) == (y / 32 % 2) { 200u32 } else { 40u32 };
            let v = (patch / 2) as u8 + noise / 2;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }

    for _ in 0..3 {
        let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter);
        let mut enc = cfg
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let _ = enc.finish().unwrap();
    }

    let iters = 15usize;
    let mut off_total = 0u128;
    let mut on_total = 0u128;
    let mut size_off = 0usize;
    let mut size_on = 0usize;
    for _ in 0..iters {
        let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter);
        let t = Instant::now();
        let mut enc = cfg
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let out = enc.finish().unwrap();
        off_total += t.elapsed().as_nanos();
        size_off = out.len();
    }
    for _ in 0..iters {
        let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter).boundary_rd(true);
        let t = Instant::now();
        let mut enc = cfg
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let out = enc.finish().unwrap();
        on_total += t.elapsed().as_nanos();
        size_on = out.len();
    }
    let mean_off = off_total as f64 / iters as f64 / 1e6;
    let mean_on = on_total as f64 / iters as f64 / 1e6;
    let overhead = (mean_on / mean_off - 1.0) * 100.0;
    let size_delta = (size_on as f64 / size_off as f64 - 1.0) * 100.0;
    eprintln!(
        "\n512x512 Q85 noise+patches, {} iters\n  off: {:.2}ms  size={}\n  on:  {:.2}ms  size={}\n  overhead: {:+.1}%   size delta: {:+.2}%\n",
        iters, mean_off, size_off, mean_on, size_on, overhead, size_delta
    );

    // Informational only; do not gate CI on timing. Budget in spec: ≤ +20%.
    // This variance-tolerant upper bound fires only on catastrophic regressions.
    assert!(
        overhead < 80.0,
        "encode overhead {:+.1}% is catastrophically high vs +20% budget",
        overhead
    );
}

#[test]
#[ignore]
fn boundary_rd_left_above_encode_overhead_at_q85() {
    let (w, h) = (512usize, 512usize);
    let mut rgb = vec![0u8; w * h * 3];
    let mut s: u64 = 0xDEAD_BEEF_0123_4567;
    for y in 0..h {
        for x in 0..w {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let n = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let noise = ((n >> 32) & 0xFF) as u8;
            let patch = if (x / 32 % 2) == (y / 32 % 2) {
                200u32
            } else {
                40u32
            };
            let v = (patch / 2) as u8 + noise / 2;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }

    for _ in 0..3 {
        let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter).boundary_rd(true);
        let mut enc = cfg
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let _ = enc.finish().unwrap();
    }

    let iters = 15usize;
    let mut left_total = 0u128;
    let mut above_total = 0u128;
    let mut size_left = 0usize;
    let mut size_above = 0usize;
    for _ in 0..iters {
        let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter).boundary_rd(true);
        let t = Instant::now();
        let mut enc = cfg
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let out = enc.finish().unwrap();
        left_total += t.elapsed().as_nanos();
        size_left = out.len();
    }
    for _ in 0..iters {
        let cfg = EncoderConfig::ycbcr(85f32, ChromaSubsampling::Quarter)
            .boundary_rd(true)
            .boundary_rd_above(true);
        let t = Instant::now();
        let mut enc = cfg
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        let out = enc.finish().unwrap();
        above_total += t.elapsed().as_nanos();
        size_above = out.len();
    }
    let mean_left = left_total as f64 / iters as f64 / 1e6;
    let mean_above = above_total as f64 / iters as f64 / 1e6;
    let overhead = (mean_above / mean_left - 1.0) * 100.0;
    let size_delta = (size_above as f64 / size_left as f64 - 1.0) * 100.0;
    eprintln!(
        "\n512x512 Q85 noise+patches, {} iters (left-only vs left+above)\n  left-only:  {:.2}ms  size={}\n  left+above: {:.2}ms  size={}\n  overhead: {:+.1}%   size delta: {:+.2}%\n",
        iters, mean_left, size_left, mean_above, size_above, overhead, size_delta
    );
    // Informational only — we're measuring the above-neighbor increment
    // on top of left-only, which should be very small (one extra SSD per
    // candidate plus persistent per-column bottom-edge buffer updates).
    assert!(
        overhead < 80.0,
        "left+above extra overhead {:+.1}% is catastrophically high",
        overhead
    );
}
