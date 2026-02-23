//! Quick benchmark for deblocking filter throughput on 2K and 4K planes.
//!
//! Usage: cargo run --release -p zenjpeg --features decoder,parallel --example deblock_bench

use std::time::Instant;

use zenjpeg::deblock::boundary::{BoundaryStrength, filter_plane_boundary_4tap};
use zenjpeg::deblock::knusperli;
use zenjpeg::foundation::consts::JPEG_ZIGZAG_ORDER;

fn make_noisy_coeffs(blocks_wide: usize, blocks_high: usize) -> (Vec<i16>, [u16; 64]) {
    let num_blocks = blocks_wide * blocks_high;
    let mut coeffs = vec![0i16; num_blocks * 64];

    // Simulate a typical Q50 turbo JPEG: moderate DC variation, sparse AC
    let mut seed: u64 = 0xDEAD_BEEF;
    for bi in 0..num_blocks {
        let base = bi * 64;
        // DC: smooth gradient + noise
        let bx = bi % blocks_wide;
        let by = bi / blocks_wide;
        let dc =
            (bx as f32 / blocks_wide as f32 * 30.0 + by as f32 / blocks_high as f32 * 20.0) as i16;
        coeffs[base] = dc;

        // Sparse AC: ~30% non-zero in low frequencies
        for k in 1..20 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (seed >> 32) % 3 == 0 {
                let val = ((seed >> 40) % 7) as i16 - 3;
                coeffs[base + k] = val;
            }
        }
    }

    // Typical Q50 turbo quant table (luma-ish)
    #[rustfmt::skip]
    let quant: [u16; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68,109,103, 77,
        24, 35, 55, 64, 81,104,113, 92,
        49, 64, 78, 87,103,121,120,101,
        72, 92, 95, 98,112,100,103, 99,
    ];

    (coeffs, quant)
}

fn make_pixel_plane(width: usize, height: usize) -> Vec<f32> {
    let mut plane = vec![0.0f32; width * height];
    let mut seed: u64 = 0xCAFE_BABE;
    for by in 0..(height / 8) {
        for bx in 0..(width / 8) {
            // Each block gets a base value with some variation
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let base = 60.0 + (seed >> 48) as f32 / 65536.0 * 140.0;
            for row in 0..8 {
                for col in 0..8 {
                    let px = bx * 8 + col;
                    let py = by * 8 + row;
                    if px < width && py < height {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let noise = (seed >> 48) as f32 / 65536.0 * 10.0 - 5.0;
                        plane[py * width + px] = (base + noise).clamp(0.0, 255.0);
                    }
                }
            }
        }
    }
    plane
}

fn bench_boundary(label: &str, width: usize, height: usize, iters: usize) {
    let strength = BoundaryStrength::from_dc_quant(16); // typical Q50
    let template = make_pixel_plane(width, height);

    // Warmup
    let mut plane = template.clone();
    filter_plane_boundary_4tap(&mut plane, width, height, strength);

    let start = Instant::now();
    for _ in 0..iters {
        plane.copy_from_slice(&template);
        filter_plane_boundary_4tap(&mut plane, width, height, strength);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    let mpix = (width * height) as f64 / 1e6;
    let mpix_per_sec = mpix / per_iter.as_secs_f64();

    println!("  {label:<30} {per_iter:>8.1?}   ({mpix:.1} MP, {mpix_per_sec:.0} MP/s)");
}

fn bench_knusperli(label: &str, blocks_wide: usize, blocks_high: usize, iters: usize) {
    let (coeffs, quant) = make_noisy_coeffs(blocks_wide, blocks_high);

    // Warmup
    let _ = knusperli::process_component(&coeffs, blocks_wide, blocks_high, &quant);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = knusperli::process_component(&coeffs, blocks_wide, blocks_high, &quant);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    let mpix = (blocks_wide * blocks_high * 64) as f64 / 1e6;
    let mpix_per_sec = mpix / per_iter.as_secs_f64();

    println!("  {label:<30} {per_iter:>8.1?}   ({mpix:.1} MP, {mpix_per_sec:.0} MP/s)");
}

fn bench_boundary_3comp(label: &str, width: usize, height: usize, iters: usize) {
    let strength_y = BoundaryStrength::from_dc_quant(16);
    let strength_c = BoundaryStrength::from_dc_quant(17);

    let template_y = make_pixel_plane(width, height);
    // 4:2:0 chroma
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let template_cb = make_pixel_plane(cw, ch);
    let template_cr = make_pixel_plane(cw, ch);

    let mut y = template_y.clone();
    let mut cb = template_cb.clone();
    let mut cr = template_cr.clone();

    // Sequential
    let start = Instant::now();
    for _ in 0..iters {
        y.copy_from_slice(&template_y);
        cb.copy_from_slice(&template_cb);
        cr.copy_from_slice(&template_cr);
        filter_plane_boundary_4tap(&mut y, width, height, strength_y);
        filter_plane_boundary_4tap(&mut cb, cw, ch, strength_c);
        filter_plane_boundary_4tap(&mut cr, cw, ch, strength_c);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    let total_mpix = (width * height + cw * ch * 2) as f64 / 1e6;
    println!(
        "  {label} seq          {per_iter:>8.1?}   ({total_mpix:.1} MP, {:.0} MP/s)",
        total_mpix / per_iter.as_secs_f64()
    );

    // Parallel (rayon)
    use rayon::prelude::*;
    let start = Instant::now();
    for _ in 0..iters {
        y.copy_from_slice(&template_y);
        cb.copy_from_slice(&template_cb);
        cr.copy_from_slice(&template_cr);
        rayon::join(
            || filter_plane_boundary_4tap(&mut y, width, height, strength_y),
            || {
                rayon::join(
                    || filter_plane_boundary_4tap(&mut cb, cw, ch, strength_c),
                    || filter_plane_boundary_4tap(&mut cr, cw, ch, strength_c),
                )
            },
        );
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    println!(
        "  {label} par          {per_iter:>8.1?}   ({total_mpix:.1} MP, {:.0} MP/s)",
        total_mpix / per_iter.as_secs_f64()
    );
}

fn bench_knusperli_3comp(label: &str, luma_bw: usize, luma_bh: usize, iters: usize) {
    let (coeffs_y, quant_y) = make_noisy_coeffs(luma_bw, luma_bh);
    let cbw = (luma_bw + 1) / 2;
    let cbh = (luma_bh + 1) / 2;
    let (coeffs_cb, quant_cb) = make_noisy_coeffs(cbw, cbh);
    let (coeffs_cr, _) = make_noisy_coeffs(cbw, cbh);

    let total_mpix = (luma_bw * luma_bh * 64 + cbw * cbh * 64 * 2) as f64 / 1e6;

    // Sequential
    let start = Instant::now();
    for _ in 0..iters {
        let _ = knusperli::process_component(&coeffs_y, luma_bw, luma_bh, &quant_y);
        let _ = knusperli::process_component(&coeffs_cb, cbw, cbh, &quant_cb);
        let _ = knusperli::process_component(&coeffs_cr, cbw, cbh, &quant_cb);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    println!(
        "  {label} seq          {per_iter:>8.1?}   ({total_mpix:.1} MP, {:.0} MP/s)",
        total_mpix / per_iter.as_secs_f64()
    );

    // Parallel
    use rayon::prelude::*;
    let start = Instant::now();
    for _ in 0..iters {
        rayon::join(
            || knusperli::process_component(&coeffs_y, luma_bw, luma_bh, &quant_y),
            || {
                rayon::join(
                    || knusperli::process_component(&coeffs_cb, cbw, cbh, &quant_cb),
                    || knusperli::process_component(&coeffs_cr, cbw, cbh, &quant_cb),
                )
            },
        );
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    println!(
        "  {label} par          {per_iter:>8.1?}   ({total_mpix:.1} MP, {:.0} MP/s)",
        total_mpix / per_iter.as_secs_f64()
    );
}

fn main() {
    // Sizes: 2K = 1920x1080, 4K = 3840x2160
    let sizes = [
        ("2K (1920x1080)", 1920usize, 1080usize, 240, 135),
        ("4K (3840x2160)", 3840usize, 2160usize, 480, 270),
    ];

    println!("=== Boundary 4-Tap (single luma plane) ===");
    for &(label, w, h, _, _) in &sizes {
        bench_boundary(label, w, h, 50);
    }

    println!();
    println!("=== Knusperli (single luma component) ===");
    for &(label, _, _, bw, bh) in &sizes {
        bench_knusperli(label, bw, bh, 50);
    }

    println!();
    println!("=== Boundary 4-Tap (YCbCr 4:2:0, seq vs par) ===");
    for &(label, w, h, _, _) in &sizes {
        bench_boundary_3comp(label, w, h, 50);
    }

    println!();
    println!("=== Knusperli (YCbCr 4:2:0, seq vs par) ===");
    for &(label, _, _, bw, bh) in &sizes {
        bench_knusperli_3comp(label, bw, bh, 50);
    }
}
