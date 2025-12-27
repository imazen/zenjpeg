//! Compare XYB constants between Rust and C++.
//!
//! Usage: cargo run --release --example compare_xyb_constants

use jpegli::consts::{XYB_OPSIN_ABSORBANCE_BIAS, XYB_OPSIN_ABSORBANCE_MATRIX};
use jpegli::xyb::{SCALED_XYB_OFFSET, SCALED_XYB_SCALE};

fn main() {
    println!("=== XYB Constants Comparison ===\n");

    // Get C++ constants
    let (cpp_opsin_matrix, cpp_opsin_bias, cpp_scaled_offset, cpp_scaled_scale) =
        jpegli_internals_sys::cpp_get_xyb_constants();

    // Rust constants
    let rust_opsin_matrix = &XYB_OPSIN_ABSORBANCE_MATRIX;
    let rust_opsin_bias = &XYB_OPSIN_ABSORBANCE_BIAS;
    let rust_scaled_offset = SCALED_XYB_OFFSET;
    let rust_scaled_scale = SCALED_XYB_SCALE;

    println!("Opsin Absorbance Matrix (9 values):");
    println!("  C++:  {:?}", cpp_opsin_matrix);
    println!("  Rust: {:?}", rust_opsin_matrix);
    let matrix_match = cpp_opsin_matrix
        .iter()
        .zip(rust_opsin_matrix.iter())
        .all(|(c, r)| (c - r).abs() < 1e-6);
    println!("  Match: {}\n", matrix_match);

    println!("Opsin Absorbance Bias (3 values):");
    println!("  C++:  {:?}", cpp_opsin_bias);
    println!("  Rust: {:?}", rust_opsin_bias);
    let bias_match = cpp_opsin_bias
        .iter()
        .zip(rust_opsin_bias.iter())
        .all(|(c, r)| (c - r).abs() < 1e-6);
    println!("  Match: {}\n", bias_match);

    println!("Scaled XYB Offset (3 values):");
    println!("  C++:  {:?}", cpp_scaled_offset);
    println!("  Rust: {:?}", rust_scaled_offset);
    let offset_match = cpp_scaled_offset
        .iter()
        .zip(rust_scaled_offset.iter())
        .all(|(c, r)| (c - r).abs() < 1e-6);
    println!("  Match: {}\n", offset_match);

    println!("Scaled XYB Scale (3 values):");
    println!("  C++:  {:?}", cpp_scaled_scale);
    println!("  Rust: {:?}", rust_scaled_scale);
    let scale_match = cpp_scaled_scale
        .iter()
        .zip(rust_scaled_scale.iter())
        .all(|(c, r)| (c - r).abs() < 1e-6);
    println!("  Match: {}\n", scale_match);

    // Test actual XYB conversion
    println!("=== XYB Conversion Test ===\n");

    // Test with a known color
    let test_colors: [(u8, u8, u8, &str); 5] = [
        (255, 0, 0, "Red"),
        (0, 255, 0, "Green"),
        (0, 0, 255, "Blue"),
        (128, 128, 128, "Gray"),
        (255, 255, 255, "White"),
    ];

    for (r, g, b, name) in &test_colors {
        // Rust conversion
        let (rust_x, rust_y, rust_b) = jpegli::xyb::srgb_to_scaled_xyb(*r, *g, *b);

        // C++ conversion
        let srgb = vec![*r, *g, *b];
        let cpp_xyb = jpegli_internals_sys::cpp_srgb_to_scaled_xyb(&srgb, 1, 1, 255.0);

        let diff_x = (rust_x - cpp_xyb[0]).abs();
        let diff_y = (rust_y - cpp_xyb[1]).abs();
        let diff_b = (rust_b - cpp_xyb[2]).abs();
        let max_diff = diff_x.max(diff_y).max(diff_b);

        println!("{} ({}, {}, {}):", name, r, g, b);
        println!("  Rust: X={:.6}, Y={:.6}, B={:.6}", rust_x, rust_y, rust_b);
        println!(
            "  C++:  X={:.6}, Y={:.6}, B={:.6}",
            cpp_xyb[0], cpp_xyb[1], cpp_xyb[2]
        );
        println!(
            "  Diff: X={:.6}, Y={:.6}, B={:.6} (max={:.6})\n",
            diff_x, diff_y, diff_b, max_diff
        );
    }

    // Test with real image pixels
    println!("=== Image Conversion Test ===\n");

    let image_path =
        "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";
    let file = std::fs::File::open(image_path).unwrap();
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let (width, height) = (info.width as usize, info.height as usize);

    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unexpected color type"),
    };

    // Convert entire image with both implementations
    let mut rust_xyb = vec![0.0f32; width * height * 3];
    for i in 0..width * height {
        let (x, y, b) = jpegli::xyb::srgb_to_scaled_xyb(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        rust_xyb[i * 3] = x;
        rust_xyb[i * 3 + 1] = y;
        rust_xyb[i * 3 + 2] = b;
    }

    let cpp_xyb = jpegli_internals_sys::cpp_srgb_to_scaled_xyb(&rgb, width, height, 255.0);

    // Calculate statistics
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut count = 0;

    for i in 0..width * height * 3 {
        let diff = (rust_xyb[i] - cpp_xyb[i]).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff as f64;
        count += 1;
    }

    let avg_diff = sum_diff / count as f64;

    println!(
        "Full image conversion ({}x{} = {} pixels):",
        width,
        height,
        width * height
    );
    println!("  Max difference: {:.9}", max_diff);
    println!("  Avg difference: {:.9}", avg_diff);

    if max_diff > 0.0001 {
        // Find the worst pixel
        let mut worst_idx = 0;
        let mut worst_diff = 0.0f32;
        for i in 0..width * height * 3 {
            let diff = (rust_xyb[i] - cpp_xyb[i]).abs();
            if diff > worst_diff {
                worst_diff = diff;
                worst_idx = i;
            }
        }

        let pixel_idx = worst_idx / 3;
        let channel = worst_idx % 3;
        let channel_name = ["X", "Y", "B"][channel];

        println!(
            "\n  Worst pixel at index {} (channel {}):",
            pixel_idx, channel_name
        );
        println!(
            "    RGB: ({}, {}, {})",
            rgb[pixel_idx * 3],
            rgb[pixel_idx * 3 + 1],
            rgb[pixel_idx * 3 + 2]
        );
        println!(
            "    Rust XYB: ({:.6}, {:.6}, {:.6})",
            rust_xyb[pixel_idx * 3],
            rust_xyb[pixel_idx * 3 + 1],
            rust_xyb[pixel_idx * 3 + 2]
        );
        println!(
            "    C++  XYB: ({:.6}, {:.6}, {:.6})",
            cpp_xyb[pixel_idx * 3],
            cpp_xyb[pixel_idx * 3 + 1],
            cpp_xyb[pixel_idx * 3 + 2]
        );
    }
}
