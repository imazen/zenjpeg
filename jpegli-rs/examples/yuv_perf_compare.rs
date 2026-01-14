//! Performance comparison: yuv crate vs jpegli-rs RGB→YCbCr conversion
//!
//! Run with: cargo run --release --example yuv_perf_compare
//!
//! Tests different image sizes and measures throughput in megapixels/second.

use std::time::Instant;

use jpegli::color::rgb_to_ycbcr_f32;
use yuv::{
    rgb_to_yuv420, YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix,
};

/// Generate a test image with varied content
fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create a gradient pattern with some variation
            data[idx] = ((x * 255 / width) as u8).wrapping_add((y * 17) as u8);
            data[idx + 1] = ((y * 255 / height) as u8).wrapping_add((x * 23) as u8);
            data[idx + 2] = (((x + y) * 127 / (width + height)) as u8).wrapping_add(64);
        }
    }
    data
}

/// Our current implementation: f32-based conversion with box-filter downsampling
fn convert_jpegli_f32(data: &[u8], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;

    // Full-res YCbCr
    let mut y_full = vec![0.0f32; width * height];
    let mut cb_full = vec![0.0f32; width * height];
    let mut cr_full = vec![0.0f32; width * height];

    for i in 0..(width * height) {
        let r = data[i * 3] as f32;
        let g = data[i * 3 + 1] as f32;
        let b = data[i * 3 + 2] as f32;
        let (y, cb, cr) = rgb_to_ycbcr_f32(r, g, b);
        y_full[i] = y;
        cb_full[i] = cb;
        cr_full[i] = cr;
    }

    // Box-filter downsample chroma for 4:2:0
    let mut cb_down = vec![0.0f32; c_width * c_height];
    let mut cr_down = vec![0.0f32; c_width * c_height];

    for cy in 0..c_height {
        for cx in 0..c_width {
            let x0 = cx * 2;
            let y0 = cy * 2;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let cb_sum = cb_full[y0 * width + x0]
                + cb_full[y0 * width + x1]
                + cb_full[y1 * width + x0]
                + cb_full[y1 * width + x1];
            let cr_sum = cr_full[y0 * width + x0]
                + cr_full[y0 * width + x1]
                + cr_full[y1 * width + x0]
                + cr_full[y1 * width + x1];

            cb_down[cy * c_width + cx] = cb_sum / 4.0;
            cr_down[cy * c_width + cx] = cr_sum / 4.0;
        }
    }

    (y_full, cb_down, cr_down)
}

/// yuv crate implementation
fn convert_yuv_crate(data: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let c_width = (width + 1) / 2;
    let c_height = (height + 1) / 2;

    let mut yuv_image =
        YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

    rgb_to_yuv420(
        &mut yuv_image,
        data,
        width as u32 * 3,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Balanced,
    )
    .expect("rgb_to_yuv420 failed");

    // Extract planes (clone to own the data)
    let y: Vec<u8> = yuv_image
        .y_plane
        .borrow()
        .iter()
        .take(width * height)
        .copied()
        .collect();
    let cb: Vec<u8> = yuv_image
        .u_plane
        .borrow()
        .iter()
        .take(c_width * c_height)
        .copied()
        .collect();
    let cr: Vec<u8> = yuv_image
        .v_plane
        .borrow()
        .iter()
        .take(c_width * c_height)
        .copied()
        .collect();

    (y, cb, cr)
}

/// Measure throughput for a given conversion function
fn benchmark<F, R>(name: &str, width: usize, height: usize, iterations: usize, f: F) -> f64
where
    F: Fn(&[u8], usize, usize) -> R,
{
    let data = generate_test_image(width, height);
    let pixels = width * height;

    // Warm up
    for _ in 0..3 {
        let _ = f(&data, width, height);
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = f(&data, width, height);
    }
    let elapsed = start.elapsed();

    let total_pixels = pixels * iterations;
    let megapixels_per_sec = (total_pixels as f64 / 1_000_000.0) / elapsed.as_secs_f64();

    println!(
        "  {}: {:.1} MP/s ({} iterations, {:.2}ms total)",
        name,
        megapixels_per_sec,
        iterations,
        elapsed.as_secs_f64() * 1000.0
    );

    megapixels_per_sec
}

fn main() {
    println!("RGB→YCbCr 4:2:0 Performance Comparison\n");
    println!("Comparing jpegli-rs (f32) vs yuv crate (SIMD-optimized integer)\n");

    let test_sizes = [
        (512, 512, 100),
        (1024, 1024, 50),
        (1920, 1080, 30),
        (2048, 2048, 20),
        (4096, 4096, 5),
    ];

    println!(
        "{:>12} {:>15} {:>15} {:>10}",
        "Size", "jpegli (f32)", "yuv crate", "Speedup"
    );
    println!("{:-<56}", "");

    for (width, height, iterations) in test_sizes {
        println!("\n{}x{}:", width, height);

        let jpegli_mps = benchmark("jpegli-rs", width, height, iterations, convert_jpegli_f32);
        let yuv_mps = benchmark("yuv crate", width, height, iterations, convert_yuv_crate);

        let speedup = yuv_mps / jpegli_mps;
        println!(
            "\n{:>12} {:>12.1} MP/s {:>12.1} MP/s {:>9.2}x",
            format!("{}x{}", width, height),
            jpegli_mps,
            yuv_mps,
            speedup
        );
    }

    println!("\n\nNotes:");
    println!("- jpegli-rs uses f32 math throughout (matches jpegli C++ precision)");
    println!("- yuv crate uses integer SIMD (AVX2/SSE/NEON) with fixed-point math");
    println!("- Both use BT.601 full-range coefficients");
    println!("- Both output 4:2:0 subsampled planes");
}
