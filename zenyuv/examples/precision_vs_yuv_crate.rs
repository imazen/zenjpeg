//! Precision comparison: zenyuv's fixed-point BT.601 (Full range) vs
//! an inline f32 reference, and against the `yuv` crate's Professional
//! (15-bit) and Balanced (13-bit) modes.
//!
//! Demonstrates that zenyuv's precision matches yuv-crate-Professional to
//! within ±1 level on u8 output — well below JPEG quantization noise.
//!
//! Run: `cargo run --release -p zenyuv --example precision_vs_yuv_crate`

use yuv::{
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
    rgb_to_yuv420, rgb_to_yuv444,
};
use zenyuv::{Matrix, Range, YuvContext};

/// Synthetic 1024×1024 with gradients + noise — stresses precision across the RGB cube.
fn synthetic(width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = ((x * 255 / width) as u8).wrapping_add((y.wrapping_mul(17)) as u8);
            let g = ((y * 255 / height) as u8).wrapping_add((x.wrapping_mul(23)) as u8);
            let b = (((x + y) * 127 / (width + height)) as u8).wrapping_add(64);
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
        }
    }
    (data, width, height)
}

/// Inline BT.601 Full-range f32 reference.
/// Y = 0.299R + 0.587G + 0.114B; Cb/Cr from ITU-R BT.601.
fn rgb_to_ycbcr_f32(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168_735_89 * r - 0.331_264_1 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418_687_6 * g - 0.081_312_4 * b + 128.0;
    (y, cb, cr)
}

fn compare_plane_u8(ours: &[u8], theirs: &[u8], label: &str) {
    let n = ours.len() as f64;
    let mut max = 0i32;
    let mut sum = 0i64;
    let mut sum_sq = 0i64;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        max = max.max(d);
        sum += d as i64;
        sum_sq += (d * d) as i64;
    }
    let avg = sum as f64 / n;
    let rmse = (sum_sq as f64 / n).sqrt();
    println!(
        "  {:<22} max={:>3}  avg={:>6.3}  rmse={:>6.3}",
        label, max, avg, rmse
    );
}

fn encode_zenyuv_444(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = w * h;
    let mut y = vec![0u8; n];
    let mut u = vec![0u8; n];
    let mut v = vec![0u8; n];
    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
    ctx.encode_444_u8(rgb, &mut y, &mut u, &mut v, w, h);
    (y, u, v)
}

fn encode_yuv_crate_444(rgb: &[u8], w: usize, h: usize, mode: YuvConversionMode) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut img = YuvPlanarImageMut::alloc(w as u32, h as u32, YuvChromaSubsampling::Yuv444);
    rgb_to_yuv444(
        &mut img,
        rgb,
        (w * 3) as u32,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        mode,
    )
    .unwrap();
    (
        img.y_plane.borrow().to_vec(),
        img.u_plane.borrow().to_vec(),
        img.v_plane.borrow().to_vec(),
    )
}

fn encode_yuv_crate_420(rgb: &[u8], w: usize, h: usize, mode: YuvConversionMode) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut img = YuvPlanarImageMut::alloc(w as u32, h as u32, YuvChromaSubsampling::Yuv420);
    rgb_to_yuv420(
        &mut img,
        rgb,
        (w * 3) as u32,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        mode,
    )
    .unwrap();
    (
        img.y_plane.borrow().to_vec(),
        img.u_plane.borrow().to_vec(),
        img.v_plane.borrow().to_vec(),
    )
}

fn encode_zenyuv_420(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
    ctx.encode_420_u8(rgb, &mut y, &mut u, &mut v, w, h);
    (y, u, v)
}

fn f32_reference_444(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = w * h;
    let mut y = vec![0u8; n];
    let mut u = vec![0u8; n];
    let mut v = vec![0u8; n];
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let (yf, cb, cr) = rgb_to_ycbcr_f32(r, g, b);
        y[i] = yf.round().clamp(0.0, 255.0) as u8;
        u[i] = cb.round().clamp(0.0, 255.0) as u8;
        v[i] = cr.round().clamp(0.0, 255.0) as u8;
    }
    (y, u, v)
}

fn coefficient_error_analysis() {
    println!("\n=== BT.601 coefficient error vs fixed-point precision ===\n");
    const YR: f32 = 0.299;
    const YG: f32 = 0.587;
    const YB: f32 = 0.114;
    for &precision in &[13u32, 15] {
        let scale = (1 << precision) as f32;
        let yr = (YR * scale).round() as i32;
        let yg = (YG * scale).round() as i32;
        let yb = (YB * scale).round() as i32;
        let label = if precision == 13 {
            "13-bit (Balanced)"
        } else {
            "15-bit (Professional / zenyuv)"
        };
        println!("{} — coefficient error at worst RGB=(255,255,255):", label);
        println!(
            "  YR err: {:+.8}  YG err: {:+.8}  YB err: {:+.8}",
            (yr as f32 / scale) - YR,
            (yg as f32 / scale) - YG,
            (yb as f32 / scale) - YB,
        );
    }
}

fn main() {
    println!("zenyuv precision vs yuv crate");
    println!("=============================");
    coefficient_error_analysis();

    let (rgb, w, h) = synthetic(1024, 1024);
    println!("\nSynthetic test image: {w}×{h} (gradient + noise)\n");

    let (yref, uref, vref) = f32_reference_444(&rgb, w, h);
    let (yzy, uzy, vzy) = encode_zenyuv_444(&rgb, w, h);
    let (ypro, upro, vpro) = encode_yuv_crate_444(&rgb, w, h, YuvConversionMode::Professional);
    let (ybal, ubal, vbal) = encode_yuv_crate_444(&rgb, w, h, YuvConversionMode::Balanced);

    println!("=== 4:4:4 ===\n");
    println!("vs f32 reference:");
    compare_plane_u8(&yzy, &yref, "zenyuv Y");
    compare_plane_u8(&uzy, &uref, "zenyuv Cb");
    compare_plane_u8(&vzy, &vref, "zenyuv Cr");
    println!();
    compare_plane_u8(&ypro, &yref, "yuv-Pro Y");
    compare_plane_u8(&upro, &uref, "yuv-Pro Cb");
    compare_plane_u8(&vpro, &vref, "yuv-Pro Cr");
    println!();
    compare_plane_u8(&ybal, &yref, "yuv-Bal Y");
    compare_plane_u8(&ubal, &uref, "yuv-Bal Cb");
    compare_plane_u8(&vbal, &vref, "yuv-Bal Cr");
    println!("\nzenyuv vs yuv-Pro (both 15-bit):");
    compare_plane_u8(&yzy, &ypro, "zenyuv-Pro Y");
    compare_plane_u8(&uzy, &upro, "zenyuv-Pro Cb");
    compare_plane_u8(&vzy, &vpro, "zenyuv-Pro Cr");

    let (yzy2, uzy2, vzy2) = encode_zenyuv_420(&rgb, w, h);
    let (ypro2, upro2, vpro2) = encode_yuv_crate_420(&rgb, w, h, YuvConversionMode::Professional);

    println!("\n=== 4:2:0 ===\n");
    println!("zenyuv vs yuv-Pro (both 15-bit, box-average chroma):");
    compare_plane_u8(&yzy2, &ypro2, "Y");
    compare_plane_u8(&uzy2, &upro2, "Cb");
    compare_plane_u8(&vzy2, &vpro2, "Cr");

    println!("\nzenyuv should match yuv-Pro within ±1 level on u8 output.");
    println!("Both run ~15-bit fixed-point; small differences are integer rounding.");
}
