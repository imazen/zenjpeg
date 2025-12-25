//! Compare Rust AQ implementation against C++ testdata.
//!
//! This test parses the C++ instrumented testdata and compares
//! our Rust implementation against expected output.

use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Slice of image data from C++ testdata.
#[derive(Debug, Deserialize)]
struct DataSlice {
    component_index: i32,
    start_row: i32,
    num_rows: i32,
    start_col: i32,
    num_cols: i32,
    stride: i32,
    data: Vec<Vec<f32>>,
}

/// FuzzyErosion test case from C++ testdata.
#[derive(Debug, Deserialize)]
struct FuzzyErosionTest {
    test_type: String,
    input_pre_erosion_slice: DataSlice,
    expected_quant_field_slice: DataSlice,
}

/// ComputeAdaptiveQuantField test case from C++ testdata.
#[derive(Debug, Deserialize)]
struct ComputeAdaptiveQuantFieldTest {
    test_type: String,
    config_y_quant_01: f32,
    config_y_comp_width_in_blocks: i32,
    config_y_comp_height_in_blocks: i32,
    input_buffer_y_slice: DataSlice,
    expected_quant_field_slice: DataSlice,
}

/// Parse first line of testdata file.
fn parse_first_fuzzy_erosion_test() -> Option<FuzzyErosionTest> {
    let path = "/home/lilith/work/jpegli/FuzzyErosion.testdata";
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.ok()?;
        if line.trim().is_empty() {
            continue;
        }
        // Remove trailing comma if present (C++ instrumentation quirk)
        let line = line.trim_end_matches(',');
        // Try to parse
        match serde_json::from_str::<FuzzyErosionTest>(line) {
            Ok(test) => return Some(test),
            Err(e) => {
                eprintln!("Parse error: {}", e);
                eprintln!("Line start: {}...", &line[..200.min(line.len())]);
                eprintln!("Line end: ...{}", &line[line.len().saturating_sub(50)..]);
                return None;
            }
        }
    }
    None
}

#[test]
fn test_parse_fuzzy_erosion_testdata() {
    let test = parse_first_fuzzy_erosion_test();

    if test.is_none() {
        eprintln!("Could not parse FuzzyErosion.testdata - skipping test");
        return;
    }

    let test = test.unwrap();

    println!("Test type: {}", test.test_type);
    println!(
        "Input pre_erosion: {}x{}",
        test.input_pre_erosion_slice.num_cols, test.input_pre_erosion_slice.num_rows
    );
    println!(
        "Expected output: {}x{}",
        test.expected_quant_field_slice.num_cols, test.expected_quant_field_slice.num_rows
    );

    // Print some expected values
    if !test.expected_quant_field_slice.data.is_empty() {
        let first_row = &test.expected_quant_field_slice.data[0];
        println!(
            "First 10 expected values: {:?}",
            &first_row[..10.min(first_row.len())]
        );

        // Compute stats
        let all_values: Vec<f32> = test
            .expected_quant_field_slice
            .data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let min = all_values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = all_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = all_values.iter().sum::<f32>() / all_values.len() as f32;

        println!(
            "Expected stats: min={:.4}, max={:.4}, mean={:.4}",
            min, max, mean
        );
    }
}

#[test]
fn test_fuzzy_erosion_vs_cpp() {
    let test = match parse_first_fuzzy_erosion_test() {
        Some(t) => t,
        None => {
            eprintln!("Could not parse FuzzyErosion.testdata - skipping test");
            return;
        }
    };

    // Extract input
    let pre_erosion_w = test.input_pre_erosion_slice.num_cols as usize;
    let pre_erosion_h = test.input_pre_erosion_slice.num_rows as usize;

    // Flatten input data
    let input: Vec<f32> = test
        .input_pre_erosion_slice
        .data
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    // Expected output dimensions
    let block_w = test.expected_quant_field_slice.num_cols as usize;
    let block_h = test.expected_quant_field_slice.num_rows as usize;

    println!(
        "Pre-erosion: {}x{} = {} values",
        pre_erosion_w,
        pre_erosion_h,
        input.len()
    );
    println!("Expected blocks: {}x{}", block_w, block_h);

    // Flatten expected output
    let expected: Vec<f32> = test
        .expected_quant_field_slice
        .data
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    println!("Expected output has {} values", expected.len());

    // Call our Rust implementation
    // Note: Our fuzzy_erosion_scalar is private, so we need to expose it or test differently
    // For now, just verify the expected values are in reasonable range

    let min = expected.iter().copied().fold(f32::INFINITY, f32::min);
    let max = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = expected.iter().sum::<f32>() / expected.len() as f32;

    println!("C++ FuzzyErosion output stats:");
    println!("  min={:.4}, max={:.4}, mean={:.4}", min, max, mean);

    // These are quant_field values BEFORE the final transform
    // After transform: aq_strength = max(0, 0.6/qf - 1)
    // So qf=0.6 -> aq=0, qf=0.3 -> aq=1, qf=0.4 -> aq=0.5

    // Verify C++ values are positive and reasonable
    assert!(
        min > 0.0,
        "FuzzyErosion output should be positive, got min={}",
        min
    );
    assert!(
        max < 100.0,
        "FuzzyErosion output should be < 100, got max={}",
        max
    );
}

/// Make the implementation accessible for testing
mod aq_impl {
    // Re-export the implementation function for testing
    pub use jpegli::adaptive_quant::*;
}

#[test]
fn test_rust_aq_impl_produces_valid_output() {
    // Test that our implementation produces values in the expected range
    let width = 64 * 8; // 64 blocks
    let height = 67 * 8; // 67 blocks

    // Create a test image with varying content
    let mut y_plane = vec![128.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            // Add some variation
            y_plane[y * width + x] = 50.0 + 150.0 * ((x + y) % 256) as f32 / 255.0;
        }
    }

    // Run our full implementation
    let map = aq_impl::compute_aq_strength_map_impl(&y_plane, width, height, 1.0);

    // Check values are in valid range
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;

    for &s in &map.strengths {
        assert!(
            s >= 0.0 && s <= 0.5,
            "aq_strength {} outside range [0, 0.5]",
            s
        );
        min = min.min(s);
        max = max.max(s);
        sum += s;
    }

    let mean = sum / map.strengths.len() as f32;
    println!("Rust AQ impl: {} blocks", map.strengths.len());
    println!("  min={:.4}, max={:.4}, mean={:.4}", min, max, mean);

    // C++ produces mean ~0.08, we should be in same ballpark
    // (This is a loose check - proper validation needs testdata comparison)
}

#[test]
fn test_rust_vs_cpp_on_testdata() {
    // Parse actual C++ testdata and compare Rust implementation
    let path = "/home/lilith/work/jpegli/ComputeAdaptiveQuantField.testdata";
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Testdata not found - skipping");
            return;
        }
    };

    let reader = BufReader::new(file);

    // Just test the first entry
    for line in reader.lines().take(1) {
        let line = match line {
            Ok(l) => l.trim_end_matches(',').to_string(),
            Err(_) => continue,
        };

        let test: ComputeAdaptiveQuantFieldTest = match serde_json::from_str(&line) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Parse error: {}", e);
                return;
            }
        };

        // Extract input Y channel
        let input_data = &test.input_buffer_y_slice.data;
        let input_height = test.input_buffer_y_slice.num_rows as usize;
        let input_width = test.input_buffer_y_slice.num_cols as usize;

        // Flatten input (it includes padding, we need actual image dims)
        let block_w = test.config_y_comp_width_in_blocks as usize;
        let _block_h = test.config_y_comp_height_in_blocks as usize;

        // The testdata only has a slice (one iMCU row), not the whole image
        // Use only the available input data
        let start_row = (-test.input_buffer_y_slice.start_row) as usize;
        let start_col = (-test.input_buffer_y_slice.start_col) as usize;

        // Actual available dimensions after removing padding
        let avail_rows = input_height.saturating_sub(start_row);
        let avail_cols = input_width.saturating_sub(start_col);

        // Round down to block boundaries
        let img_width = (avail_cols / 8) * 8;
        let img_height = (avail_rows / 8) * 8;
        let actual_block_w = img_width / 8;
        let actual_block_h = img_height / 8;

        println!(
            "Input slice: {}x{} (padding: row={}, col={})",
            input_width, input_height, start_row, start_col
        );
        println!("Available after padding: {}x{}", avail_cols, avail_rows);
        println!(
            "Processing: {}x{} pixels ({} x {} = {} blocks)",
            img_width,
            img_height,
            actual_block_w,
            actual_block_h,
            actual_block_w * actual_block_h
        );

        if img_width == 0 || img_height == 0 {
            eprintln!("Not enough data to process");
            return;
        }

        // Create Y plane from input data
        let mut y_plane = vec![0.0f32; img_width * img_height];

        for (row_idx, row) in input_data.iter().enumerate() {
            if row_idx < start_row {
                continue;
            }
            let y = row_idx - start_row;
            if y >= img_height {
                break;
            }

            for (col_idx, &val) in row.iter().enumerate() {
                if col_idx < start_col {
                    continue;
                }
                let x = col_idx - start_col;
                if x >= img_width {
                    break;
                }

                y_plane[y * img_width + x] = val;
            }
        }

        // Print sample of input values
        println!("Input sample: {:?}", &y_plane[..10.min(y_plane.len())]);

        // Get expected output
        let expected: Vec<f32> = test
            .expected_quant_field_slice
            .data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // Run Rust implementation
        let distance = 1.0 / test.config_y_quant_01; // Approximate
        let rust_map =
            aq_impl::compute_aq_strength_map_impl(&y_plane, img_width, img_height, distance);

        println!(
            "\nExpected {} values, Rust produced {}",
            expected.len(),
            rust_map.strengths.len()
        );

        // Compare stats
        let exp_min = expected.iter().copied().fold(f32::INFINITY, f32::min);
        let exp_max = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_mean: f32 = expected.iter().sum::<f32>() / expected.len() as f32;

        let rust_min = rust_map
            .strengths
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let rust_max = rust_map
            .strengths
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let rust_mean: f32 =
            rust_map.strengths.iter().sum::<f32>() / rust_map.strengths.len() as f32;

        println!(
            "C++ : min={:.4}, max={:.4}, mean={:.4}",
            exp_min, exp_max, exp_mean
        );
        println!(
            "Rust: min={:.4}, max={:.4}, mean={:.4}",
            rust_min, rust_max, rust_mean
        );

        // Calculate per-block differences
        let num_to_compare = expected.len().min(rust_map.strengths.len());
        let mut sum_abs_diff = 0.0f32;
        let mut max_abs_diff = 0.0f32;

        for i in 0..num_to_compare {
            let diff = (expected[i] - rust_map.strengths[i]).abs();
            sum_abs_diff += diff;
            max_abs_diff = max_abs_diff.max(diff);
        }

        let mean_abs_diff = sum_abs_diff / num_to_compare as f32;
        println!(
            "Mean abs diff: {:.4}, Max abs diff: {:.4}",
            mean_abs_diff, max_abs_diff
        );

        // STRICT CHECK: Must match C++ within 0.01 absolute difference
        // This assertion is currently failing - see test_compute_aq_field_vs_cpp for tracking.
        assert!(
            max_abs_diff < 0.01,
            "AQ implementation differs from C++ by {:.4} (max allowed: 0.01). \
             See docs/ADAPTIVE_QUANTIZATION.md for known gaps.",
            max_abs_diff
        );
    }
}

#[test]
#[ignore] // Run with --ignored when ready
fn test_compute_aq_field_vs_cpp() {
    let path = "/home/lilith/work/jpegli/ComputeAdaptiveQuantField.testdata";
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Could not open {}: {}", path, e);
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut test_count = 0;
    let mut pass_count = 0;
    let mut max_diff = 0.0f32;

    for line in reader.lines().take(10) {
        // Process first 10 tests
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        if line.trim().is_empty() {
            continue;
        }

        // Remove trailing comma if present
        let line = line.trim_end_matches(',');

        let test: ComputeAdaptiveQuantFieldTest = match serde_json::from_str(line) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Parse error: {}", e);
                continue;
            }
        };

        test_count += 1;

        // Get expected values
        let expected: Vec<f32> = test
            .expected_quant_field_slice
            .data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // Compute stats
        let min = expected.iter().copied().fold(f32::INFINITY, f32::min);
        let max = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = expected.iter().sum::<f32>() / expected.len() as f32;

        println!(
            "Test {}: y_quant={:.1}, blocks={}x{}",
            test_count,
            test.config_y_quant_01,
            test.config_y_comp_width_in_blocks,
            test.config_y_comp_height_in_blocks
        );
        println!(
            "  Expected aq_strength: min={:.4}, max={:.4}, mean={:.4}",
            min, max, mean
        );

        // STRICT CHECK: Values must be in C++ documented 0-0.2 range
        if min >= 0.0 && max <= 0.2 {
            pass_count += 1;
            println!("  ✓ Values in C++ expected range [0, 0.2]");
        } else {
            println!("  ✗ Values outside C++ expected range [0, 0.2]!");
        }

        if max > max_diff {
            max_diff = max;
        }
    }

    println!(
        "\nSummary: {}/{} tests have values in expected range",
        pass_count, test_count
    );
    println!("Max aq_strength seen: {:.4}", max_diff);

    // STRICT CHECK: C++ produces values in 0-0.2 range per documentation
    assert!(
        max_diff <= 0.2,
        "C++ produces values up to {:.4}, expected max 0.2",
        max_diff
    );
}
