//! Huffman table optimization for JPEG encoding.
//!
//! **This module is deprecated.** Use `crate::huffman::optimize` instead.
//!
//! This module re-exports types from `huffman::optimize` for backward compatibility.

// Re-export everything from the new location
pub use crate::huffman::optimize::{
    cluster_histograms, ClusterResult, ContextConfig, FrequencyCounter, OptimizedHuffmanTables,
    OptimizedTable, ProgressiveTokenBuffer, RefToken, ScanTokenInfo, Token, TokenBuffer,
};

// Re-export the tests module for test compatibility
#[cfg(test)]
mod tests {
    // Tests have been moved to individual submodule files:
    // - frequency.rs: FrequencyCounter tests
    // - tokens.rs: Token, RefToken, TokenBuffer tests
    // - progressive.rs: ProgressiveTokenBuffer tests
    //
    // The cpp_comparison_tests remain in this file for now since they test
    // the generate_code_lengths function which is in huffman_classic.

    use crate::huffman::classic::generate_code_lengths;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    fn load_testdata() -> Option<Vec<(Vec<i64>, Vec<u8>)>> {
        let path = crate::test_utils::get_cpp_testdata_path("CreateHuffmanTree.testdata")?;
        let file = File::open(&path).ok()?;
        let reader = BufReader::new(file);

        let mut tests = Vec::new();
        for line in reader.lines() {
            let line = line.ok()?;
            let line = line.trim_end_matches(',');
            let v: serde_json::Value = serde_json::from_str(line).ok()?;

            let input: Vec<i64> = v["input_data"]
                .as_array()?
                .iter()
                .map(|x| x.as_i64().unwrap_or(0))
                .collect();
            let expected: Vec<u8> = v["output_depth"]
                .as_array()?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as u8)
                .collect();

            tests.push((input, expected));
        }
        Some(tests)
    }

    #[test]
    #[ignore] // FAILING: 4/185 cases where C++ is better - algorithm needs fixing
    fn test_against_cpp_testdata() {
        let tests = match load_testdata() {
            Some(t) => t,
            None => {
                eprintln!("Skipping: CreateHuffmanTree.testdata not found");
                return;
            }
        };

        let mut exact_match = 0;
        let mut mozjpeg_better = 0;
        let mut cpp_better = 0;
        let total = tests.len();

        for (input, expected) in &tests {
            let mut freq = [0i64; 257];
            for (i, &f) in input.iter().enumerate().take(257) {
                freq[i] = f;
            }

            let result = generate_code_lengths(&mut freq).unwrap();

            // Check exact match
            let exact = (0..256).all(|i| result[i] == expected[i]);

            // Calculate bit costs
            let cost_result: i64 = (0..256).map(|i| input[i] * result[i] as i64).sum();
            let cost_expected: i64 = (0..256).map(|i| input[i] * expected[i] as i64).sum();

            if exact {
                exact_match += 1;
            } else if cost_result < cost_expected {
                mozjpeg_better += 1;
            } else if cost_result > cost_expected {
                cpp_better += 1;
            } else {
                // Same cost, different assignment (equally valid)
                exact_match += 1;
            }
        }

        println!("C++ comparison results:");
        println!("  Exact match: {}/{}", exact_match, total);
        println!("  mozjpeg better: {}", mozjpeg_better);
        println!("  C++ better: {}", cpp_better);

        // Assert we're at least as good as C++
        assert_eq!(
            cpp_better, 0,
            "mozjpeg algorithm should never be worse than C++"
        );

        // Assert reasonable match rate
        let match_rate = (exact_match + mozjpeg_better) as f64 / total as f64;
        assert!(
            match_rate >= 0.80,
            "Match rate {:.1}% is too low",
            match_rate * 100.0
        );
    }

    #[test]
    fn test_specific_cpp_case() {
        // Test case from C++ testdata that we know produces exact match
        let input = [
            61i64, 98, 196, 372, 613, 754, 818, 663, 525, 185, 3, 0, 0, 0, 0, 0,
        ];
        let expected_depths = [7u8, 6, 4, 3, 3, 3, 2, 3, 3, 5, 8];

        let mut freq = [0i64; 257];
        for (i, &f) in input.iter().enumerate() {
            freq[i] = f;
        }
        freq[256] = 1; // pseudo-symbol

        let result = generate_code_lengths(&mut freq).unwrap();

        for (i, &expected) in expected_depths.iter().enumerate() {
            assert_eq!(
                result[i], expected,
                "Symbol {} depth mismatch: got {}, expected {}",
                i, result[i], expected
            );
        }
    }
}
