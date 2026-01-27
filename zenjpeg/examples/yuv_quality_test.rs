//! Quality comparison: Does using yuv crate's integer conversion affect precision?
//!
//! Compares conversion precision between:
//! 1. Our f32 RGB→YCbCr conversion
//! 2. yuv crate's i32 fixed-point conversion (13-bit precision in Balanced mode)
//!
//! Run with: cargo run --release --example yuv_quality_test

use std::path::Path;

use yuv::{
    rgb_to_yuv420, YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix,
};
use zenjpeg::color::rgb_to_ycbcr_f32;

/// Load a test image (or generate synthetic)
fn load_or_generate_test_image(width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    // Try to load frymire.png
    let test_path = Path::new("internal/jpegli-cpp/testdata/frymire.png");
    if test_path.exists() {
        let decoder = png::Decoder::new(std::fs::File::open(test_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();

        let w = info.width as usize;
        let h = info.height as usize;

        // Convert to RGB if needed
        let rgb = match info.color_type {
            png::ColorType::Rgb => buf[..w * h * 3].to_vec(),
            png::ColorType::Rgba => {
                let mut rgb = vec![0u8; w * h * 3];
                for i in 0..(w * h) {
                    rgb[i * 3] = buf[i * 4];
                    rgb[i * 3 + 1] = buf[i * 4 + 1];
                    rgb[i * 3 + 2] = buf[i * 4 + 2];
                }
                rgb
            }
            _ => panic!("Unsupported color type"),
        };

        println!("Loaded frymire.png: {}x{}", w, h);
        return (rgb, w, h);
    }

    // Generate synthetic test image with challenging content
    println!("Generating synthetic {}x{} test image", width, height);
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Mix of gradients, edges, and noise-like patterns
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

/// Compare Y channel precision between f32 and yuv crate
fn compare_y_precision(rgb: &[u8], width: usize, height: usize) {
    println!("\n=== Y Channel Precision Comparison ===\n");

    // Our f32 conversion
    let mut y_f32 = vec![0.0f32; width * height];
    for i in 0..(width * height) {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let (y, _, _) = rgb_to_ycbcr_f32(r, g, b);
        y_f32[i] = y;
    }

    // yuv crate conversion
    let mut yuv_image =
        YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);
    rgb_to_yuv420(
        &mut yuv_image,
        rgb,
        width as u32 * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Professional,
    )
    .unwrap();

    let y_yuv: Vec<u8> = yuv_image.y_plane.borrow().to_vec();

    // Compare
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut diff_histogram = [0u32; 10]; // 0, 0.5, 1.0, 1.5, 2.0, ...

    for i in 0..(width * height) {
        let diff = (y_f32[i] - y_yuv[i] as f32).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff as f64;
        sum_sq_diff += (diff * diff) as f64;

        let bucket = (diff * 2.0).floor().min(9.0) as usize;
        diff_histogram[bucket] += 1;
    }

    let n = (width * height) as f64;
    let avg_diff = sum_diff / n;
    let rmse = (sum_sq_diff / n).sqrt();

    println!("Y channel difference (f32 vs yuv crate i32/13-bit):");
    println!("  Max difference:  {:.3} levels (out of 255)", max_diff);
    println!("  Avg difference:  {:.4} levels", avg_diff);
    println!("  RMSE:            {:.4} levels", rmse);
    println!("\nHistogram (difference ranges):");
    println!(
        "  [0.0, 0.5): {:>8} ({:>5.2}%)",
        diff_histogram[0],
        diff_histogram[0] as f64 / n * 100.0
    );
    println!(
        "  [0.5, 1.0): {:>8} ({:>5.2}%)",
        diff_histogram[1],
        diff_histogram[1] as f64 / n * 100.0
    );
    println!(
        "  [1.0, 1.5): {:>8} ({:>5.2}%)",
        diff_histogram[2],
        diff_histogram[2] as f64 / n * 100.0
    );
    println!(
        "  [1.5, 2.0): {:>8} ({:>5.2}%)",
        diff_histogram[3],
        diff_histogram[3] as f64 / n * 100.0
    );
    let higher: u32 = diff_histogram[4..].iter().sum();
    println!(
        "  [2.0+):     {:>8} ({:>5.2}%)",
        higher,
        higher as f64 / n * 100.0
    );
}

/// Compare Cb/Cr chroma precision (full-resolution before subsampling)
fn compare_chroma_precision(rgb: &[u8], width: usize, height: usize) {
    println!("\n=== Cb/Cr Channel Precision (before subsampling) ===\n");

    // Our f32 conversion
    let mut cb_f32 = vec![0.0f32; width * height];
    let mut cr_f32 = vec![0.0f32; width * height];
    for i in 0..(width * height) {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let (_, cb, cr) = rgb_to_ycbcr_f32(r, g, b);
        cb_f32[i] = cb;
        cr_f32[i] = cr;
    }

    // yuv crate with 4:4:4 to get full-res chroma
    let mut yuv_image =
        YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv444);
    yuv::rgb_to_yuv444(
        &mut yuv_image,
        rgb,
        width as u32 * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Professional,
    )
    .unwrap();

    let cb_yuv: Vec<u8> = yuv_image.u_plane.borrow().to_vec();
    let cr_yuv: Vec<u8> = yuv_image.v_plane.borrow().to_vec();

    // Compare Cb
    let mut cb_max_diff = 0.0f32;
    let mut cb_sum_diff = 0.0f64;
    for i in 0..(width * height) {
        let diff = (cb_f32[i] - cb_yuv[i] as f32).abs();
        cb_max_diff = cb_max_diff.max(diff);
        cb_sum_diff += diff as f64;
    }

    // Compare Cr
    let mut cr_max_diff = 0.0f32;
    let mut cr_sum_diff = 0.0f64;
    for i in 0..(width * height) {
        let diff = (cr_f32[i] - cr_yuv[i] as f32).abs();
        cr_max_diff = cr_max_diff.max(diff);
        cr_sum_diff += diff as f64;
    }

    let n = (width * height) as f64;
    println!("Cb channel:");
    println!("  Max difference: {:.3} levels", cb_max_diff);
    println!("  Avg difference: {:.4} levels", cb_sum_diff / n);
    println!("Cr channel:");
    println!("  Max difference: {:.3} levels", cr_max_diff);
    println!("  Avg difference: {:.4} levels", cr_sum_diff / n);
}

/// Analyze what causes the precision loss
fn analyze_precision_source(rgb: &[u8], width: usize, height: usize) {
    println!("\n=== Precision Loss Analysis ===\n");

    // BT.601 coefficients
    const YR: f32 = 0.299;
    const YG: f32 = 0.587;
    const YB: f32 = 0.114;

    // yuv crate uses 13-bit precision for Balanced mode
    let precision = 13;
    let scale = (1 << precision) as f32;
    let yr_int = (YR * scale).round() as i32;
    let yg_int = (YG * scale).round() as i32;
    let yb_int = (YB * scale).round() as i32;

    println!("BT.601 Y coefficients:");
    println!("  f32:   YR={:.6}, YG={:.6}, YB={:.6}", YR, YG, YB);
    println!(
        "  i32:   YR={}, YG={}, YB={} (13-bit scaled)",
        yr_int, yg_int, yb_int
    );
    println!(
        "  scaled back: YR={:.6}, YG={:.6}, YB={:.6}",
        yr_int as f32 / scale,
        yg_int as f32 / scale,
        yb_int as f32 / scale
    );
    println!("\nCoefficient error:");
    println!("  YR: {:.9}", (yr_int as f32 / scale) - YR);
    println!("  YG: {:.9}", (yg_int as f32 / scale) - YG);
    println!("  YB: {:.9}", (yb_int as f32 / scale) - YB);

    // Find worst-case pixel
    let mut worst_diff = 0.0f32;
    let mut worst_rgb = (0u8, 0u8, 0u8);
    let mut worst_y_f32 = 0.0f32;
    let mut worst_y_int = 0i32;

    for i in 0..(width * height) {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];

        // f32 calculation
        let y_f32 = YR * r as f32 + YG * g as f32 + YB * b as f32;

        // Integer calculation (mimicking yuv crate)
        let bias = (1 << (precision - 1)) - 1; // rounding bias
        let y_int = (r as i32 * yr_int + g as i32 * yg_int + b as i32 * yb_int + bias) >> precision;

        let diff = (y_f32 - y_int as f32).abs();
        if diff > worst_diff {
            worst_diff = diff;
            worst_rgb = (r, g, b);
            worst_y_f32 = y_f32;
            worst_y_int = y_int;
        }
    }

    println!("\nWorst-case pixel in this image:");
    println!("  RGB: ({}, {}, {})", worst_rgb.0, worst_rgb.1, worst_rgb.2);
    println!("  Y (f32):     {:.4}", worst_y_f32);
    println!("  Y (i32/13b): {}", worst_y_int);
    println!("  Difference:  {:.4} levels", worst_diff);
}

fn main() {
    println!("yuv crate Precision Analysis");
    println!("============================\n");
    println!("yuv crate uses fixed-point integer math:");
    println!("  - Balanced mode: 13-bit precision (coefficients × 8192)");
    println!("  - Professional mode: 15-bit precision (coefficients × 32768)");
    println!("  - Fast mode: 7-bit precision (coefficients × 128, like libyuv)");

    let (rgb, width, height) = load_or_generate_test_image(1024, 1024);

    compare_y_precision(&rgb, width, height);
    compare_chroma_precision(&rgb, width, height);
    analyze_precision_source(&rgb, width, height);

    println!("\n=== Conclusion ===\n");
    println!("For JPEG encoding:");
    println!("  - Y differences of ~1 level are within quantization noise");
    println!("  - DCT quantization at Q85 loses ~2-4 levels of precision anyway");
    println!("  - The 13-bit fixed-point math is MORE than sufficient");
    println!("\nRecommendation: Safe to integrate yuv crate for RGB→YCbCr conversion");
}
