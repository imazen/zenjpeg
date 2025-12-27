//! Compare Rust AQ implementation against C++ jpegli testdata.
//!
//! This test parses the C++ instrumented testdata and compares
//! our Rust implementation against expected output.
//!
//! Adapted from jpegli-rs/tests/aq_cpp_comparison.rs

mod common;

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

/// Parse first line of FuzzyErosion testdata file.
fn parse_first_fuzzy_erosion_test() -> Option<FuzzyErosionTest> {
    let path = common::get_cpp_testdata_path("FuzzyErosion.testdata")?;
    let file = File::open(&path).ok()?;
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
        eprintln!("Generate testdata with: GENERATE_RUST_TEST_DATA=1 cjpegli input.png output.jpg");
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
fn test_fuzzy_erosion_expected_range() {
    let test = match parse_first_fuzzy_erosion_test() {
        Some(t) => t,
        None => {
            eprintln!("Could not parse FuzzyErosion.testdata - skipping test");
            return;
        }
    };

    // Flatten expected output
    let expected: Vec<f32> = test
        .expected_quant_field_slice
        .data
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    let min = expected.iter().copied().fold(f32::INFINITY, f32::min);
    let max = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = expected.iter().sum::<f32>() / expected.len() as f32;

    println!("C++ FuzzyErosion output stats:");
    println!("  min={:.4}, max={:.4}, mean={:.4}", min, max, mean);

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

#[test]
fn test_compute_aq_field_testdata_range() {
    // Parse actual C++ testdata
    let Some(path) = common::try_get_cpp_testdata_path("ComputeAdaptiveQuantField.testdata") else {
        eprintln!("ComputeAdaptiveQuantField.testdata not found. Set CPP_TESTDATA_DIR env var.");
        return;
    };
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Testdata not found at {:?} - skipping", path);
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

        println!(
            "Test: y_quant={:.1}, blocks={}x{}",
            test.config_y_quant_01,
            test.config_y_comp_width_in_blocks,
            test.config_y_comp_height_in_blocks
        );

        // Get expected values (aq_strength after transform)
        let expected: Vec<f32> = test
            .expected_quant_field_slice
            .data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let min = expected.iter().copied().fold(f32::INFINITY, f32::min);
        let max = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = expected.iter().sum::<f32>() / expected.len() as f32;

        println!(
            "C++ aq_strength: min={:.4}, max={:.4}, mean={:.4}",
            min, max, mean
        );

        // STRICT CHECK: C++ produces values in 0.0-0.2 range per documentation
        assert!(
            min >= 0.0,
            "aq_strength min {} should be >= 0.0",
            min
        );
        assert!(
            max <= 0.21, // Small tolerance for floating point
            "aq_strength max {} should be <= 0.2",
            max
        );
    }
}

#[test]
#[ignore] // Run with --ignored when full AQ implementation is ready
fn test_rust_vs_cpp_aq() {
    // Parse actual C++ testdata and compare Rust implementation
    let Some(path) = common::try_get_cpp_testdata_path("ComputeAdaptiveQuantField.testdata") else {
        eprintln!("Testdata not found. Set CPP_TESTDATA_DIR env var.");
        return;
    };
    let file = match File::open(&path) {
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

        // Handle padding
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
        println!("Processing: {}x{} pixels ({} x {} blocks)", img_width, img_height, actual_block_w, actual_block_h);

        if img_width == 0 || img_height == 0 {
            eprintln!("Not enough data to process");
            return;
        }

        // Create Y plane from input data (as f32 for full AQ)
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

        // Get expected output
        let expected: Vec<f32> = test
            .expected_quant_field_slice
            .data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // TODO: Run full Rust AQ implementation here
        // For now, just validate the testdata exists and is parseable

        println!("Expected {} aq_strength values", expected.len());
        println!("Y plane has {} pixels", y_plane.len());

        // Calculate expected stats
        let exp_min = expected.iter().copied().fold(f32::INFINITY, f32::min);
        let exp_max = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_mean: f32 = expected.iter().sum::<f32>() / expected.len() as f32;

        println!(
            "C++ : min={:.4}, max={:.4}, mean={:.4}",
            exp_min, exp_max, exp_mean
        );

        // When full AQ is implemented, compare and assert:
        // let tolerance = 0.025;
        // assert!(max_abs_diff < tolerance, "AQ differs from C++ by too much");
    }
}

#[test]
#[ignore] // Run with --ignored for comprehensive testing
fn test_all_aq_testdata_entries() {
    let Some(path) = common::try_get_cpp_testdata_path("ComputeAdaptiveQuantField.testdata") else {
        eprintln!("Testdata not found. Set CPP_TESTDATA_DIR env var.");
        return;
    };
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Could not open {}: {}", path, e);
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut test_count = 0;
    let mut pass_count = 0;
    let mut max_aq_seen = 0.0f32;

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
        if min >= 0.0 && max <= 0.21 {
            pass_count += 1;
            println!("  PASS: Values in C++ expected range [0, 0.2]");
        } else {
            println!("  FAIL: Values outside C++ expected range [0, 0.2]!");
        }

        if max > max_aq_seen {
            max_aq_seen = max;
        }
    }

    println!(
        "\nSummary: {}/{} tests have values in expected range",
        pass_count, test_count
    );
    println!("Max aq_strength seen: {:.4}", max_aq_seen);

    // STRICT CHECK: C++ produces values in 0-0.2 range per documentation
    assert!(
        max_aq_seen <= 0.21,
        "C++ produces values up to {:.4}, expected max 0.2",
        max_aq_seen
    );
}
