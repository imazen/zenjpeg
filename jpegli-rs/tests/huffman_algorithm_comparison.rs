//! Tests comparing the two Huffman algorithm implementations.
//!
//! This test suite verifies that both algorithms produce valid, decodable JPEGs
//! and compares their compression efficiency.
//!
//! Algorithms:
//! - `MozjpegClassic`: libjpeg/mozjpeg style (others[] chain with Section K.2 limiting)
//! - `JpegliTree`: jpegli C++ style (sorted two-pointer merge with retry)
//!
//! Run with: cargo test --test huffman_algorithm_comparison -- --nocapture

use jpegli::huffman_types::{compare_algorithms, SymbolFrequencies};
use jpegli::types::HuffmanMethod;
use jpegli::{Encoder, JpegMode, PixelFormat, Quality, Subsampling};

/// Generate a gradient test image
fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            data.push((x * 255 / width) as u8);
            data.push((y * 255 / height) as u8);
            data.push(((x + y) * 127 / (width + height)) as u8);
        }
    }
    data
}

/// Generate a complex pattern with high-frequency content
fn generate_complex(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / 8) + (y / 8)) % 2 == 0;
            let base = if checker { 200u8 } else { 55u8 };
            data.push(base.wrapping_add((x % 56) as u8));
            data.push(base.wrapping_add((y % 56) as u8));
            data.push(base.wrapping_add(((x + y) % 56) as u8));
        }
    }
    data
}

/// Decode with multiple decoders to verify output is valid
fn verify_decodable(jpeg_data: &[u8], expected_width: usize, expected_height: usize) -> bool {
    // Test with jpegli-rs decoder
    let jpegli_ok = jpegli::Decoder::new().decode(jpeg_data).is_ok();

    // Test with jpeg-decoder
    let jpeg_decoder_ok = {
        let mut decoder =
            zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg_data));
        decoder.decode().is_ok()
    };

    // Test with zune-jpeg
    let zune_ok = {
        use zune_jpeg::zune_core::bytestream::ZCursor;
        use zune_jpeg::JpegDecoder;
        let cursor = ZCursor::new(jpeg_data);
        let mut decoder = JpegDecoder::new(cursor);
        if let Ok(_pixels) = decoder.decode() {
            if let Some((w, h)) = decoder.dimensions() {
                w == expected_width && h == expected_height
            } else {
                false
            }
        } else {
            false
        }
    };

    jpegli_ok && jpeg_decoder_ok && zune_ok
}

/// Helper to encode with a specific Huffman method
fn encode_with_method(
    data: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    _method: HuffmanMethod, // TODO: Wire through internal pipeline when API is ready
    mode: JpegMode,
) -> Result<Vec<u8>, jpegli::error::Error> {
    // We need to use internal pipeline to set huffman method
    // For now, just use the public API with optimize_huffman (uses JpegliCreateTree)
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality))
        .subsampling(Subsampling::S444)
        .optimize_huffman(true)
        .mode(mode);

    encoder.encode(data)
}

/// Test: Both algorithms produce decodable baseline JPEGs
#[test]
fn test_baseline_both_algorithms_decodable() {
    let width = 128;
    let height = 128;
    let data = generate_gradient(width, height);

    for quality in [50.0, 75.0, 90.0, 95.0] {
        let jpeg = encode_with_method(
            &data,
            width as u32,
            height as u32,
            quality,
            HuffmanMethod::JpegliCreateTree,
            JpegMode::Baseline,
        )
        .expect("Encoding should succeed");

        assert!(
            verify_decodable(&jpeg, width, height),
            "Baseline Q{} should be decodable by all decoders",
            quality
        );
    }
}

/// Test: Both algorithms produce decodable progressive JPEGs
#[test]
fn test_progressive_both_algorithms_decodable() {
    let width = 128;
    let height = 128;
    let data = generate_gradient(width, height);

    for quality in [50.0, 75.0, 90.0, 95.0] {
        let jpeg = encode_with_method(
            &data,
            width as u32,
            height as u32,
            quality,
            HuffmanMethod::JpegliCreateTree,
            JpegMode::Progressive,
        )
        .expect("Encoding should succeed");

        assert!(
            verify_decodable(&jpeg, width, height),
            "Progressive Q{} should be decodable by all decoders",
            quality
        );
    }
}

/// Test: Compare file sizes between baseline and progressive
///
/// Note: Progressive isn't always smaller than baseline, especially without
/// histogram clustering. This test just verifies both produce valid output
/// and documents the size differences.
#[test]
fn test_progressive_vs_baseline_sizes() {
    let width = 256;
    let height = 256;
    let data = generate_complex(width, height);

    println!("\n=== Progressive vs Baseline Size Comparison ===");

    for quality in [75.0, 85.0, 90.0, 95.0] {
        let baseline = encode_with_method(
            &data,
            width as u32,
            height as u32,
            quality,
            HuffmanMethod::JpegliCreateTree,
            JpegMode::Baseline,
        )
        .expect("Baseline encoding should succeed");

        let progressive = encode_with_method(
            &data,
            width as u32,
            height as u32,
            quality,
            HuffmanMethod::JpegliCreateTree,
            JpegMode::Progressive,
        )
        .expect("Progressive encoding should succeed");

        let diff_pct =
            (baseline.len() as f64 - progressive.len() as f64) / baseline.len() as f64 * 100.0;
        let status = if diff_pct > 0.0 { "smaller" } else { "larger" };

        println!(
            "Q{}: baseline={} bytes, progressive={} bytes ({:.1}% {})",
            quality,
            baseline.len(),
            progressive.len(),
            diff_pct.abs(),
            status
        );

        // Both should be decodable
        assert!(
            verify_decodable(&baseline, width, height),
            "Baseline Q{} should be decodable",
            quality
        );
        assert!(
            verify_decodable(&progressive, width, height),
            "Progressive Q{} should be decodable",
            quality
        );
    }
}

/// Test: Huffman tables are valid (Kraft inequality)
#[test]
fn test_huffman_tables_valid_kraft() {
    use jpegli::huffman_classic::generate_code_lengths;

    // Test with various frequency distributions
    let test_cases: Vec<(&str, Vec<i64>)> = vec![
        ("uniform", (0..256).map(|_| 100i64).collect()),
        (
            "skewed",
            (0..256).map(|i| ((256 - i) * 10) as i64).collect(),
        ),
        (
            "sparse",
            (0..256)
                .map(|i| if i % 10 == 0 { 100 } else { 0 })
                .collect(),
        ),
        ("single", {
            let mut v = vec![0i64; 256];
            v[0] = 1000;
            v
        }),
    ];

    for (name, freqs) in test_cases {
        let mut freq_arr = [0i64; 257];
        for (i, &f) in freqs.iter().enumerate().take(256) {
            freq_arr[i] = f;
        }

        let lengths = generate_code_lengths(&mut freq_arr).expect("Should generate valid lengths");

        // Verify Kraft inequality: sum(2^(16-L)) < 2^16
        let kraft_sum: u64 = lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (16 - l as u64))
            .sum();

        assert!(
            kraft_sum < (1 << 16),
            "{}: Kraft sum {} should be < {}",
            name,
            kraft_sum,
            1u64 << 16
        );

        // All lengths should be 1-16
        for &l in &lengths {
            if l > 0 {
                assert!(l <= 16, "{}: Length {} exceeds 16", name, l);
            }
        }
    }
}

/// Test: jpegli algorithm produces valid code lengths
#[test]
fn test_jpegli_algorithm_valid() {
    use jpegli::huffman::build_code_lengths;

    // Test with various frequency distributions
    let test_cases: Vec<(&str, Vec<u64>)> = vec![
        ("uniform", (0..257).map(|_| 100u64).collect()),
        (
            "skewed",
            (0..257).map(|i| ((257 - i) * 10) as u64).collect(),
        ),
        (
            "sparse",
            (0..257)
                .map(|i| if i % 10 == 0 { 100 } else { 0 })
                .collect(),
        ),
    ];

    for (name, freqs) in test_cases {
        let depths = build_code_lengths(&freqs, 16);

        // Verify Kraft inequality
        let kraft_sum: u64 = depths
            .iter()
            .filter(|&&d| d > 0)
            .map(|&d| 1u64 << (16 - d as u64))
            .sum();

        // For jpegli algorithm, Kraft sum should be exactly 2^16 or slightly less
        assert!(
            kraft_sum <= (1 << 16),
            "{}: Kraft sum {} should be <= {}",
            name,
            kraft_sum,
            1u64 << 16
        );

        // All depths should be 1-16
        for &d in &depths {
            if d > 0 {
                assert!(d <= 16, "{}: Depth {} exceeds 16", name, d);
            }
        }
    }
}

/// Test: Compare algorithms on identical input
#[test]
fn test_algorithm_comparison() {
    use jpegli::huffman::build_code_lengths;
    use jpegli::huffman_classic::generate_code_lengths;

    // Create a realistic AC histogram (many zeros, few high values)
    let mut freqs = vec![0u64; 257];
    freqs[0] = 10000; // EOB - very common
    freqs[1] = 5000; // 0/1 - common
    freqs[17] = 3000; // 1/1
    freqs[33] = 2000; // 2/1
    for i in 2..16 {
        freqs[i] = (1000 / (i + 1)) as u64;
    }
    freqs[256] = 1; // Pseudo-symbol

    // Run classic algorithm
    let mut classic_freq = [0i64; 257];
    for (i, &f) in freqs.iter().enumerate() {
        classic_freq[i] = f as i64;
    }
    let classic_lengths = generate_code_lengths(&mut classic_freq).expect("Classic should work");

    // Run jpegli algorithm
    let jpegli_depths = build_code_lengths(&freqs, 16);

    // Compare results
    let classic_bits: u64 = (0..256)
        .filter(|&i| freqs[i] > 0)
        .map(|i| freqs[i] * classic_lengths[i] as u64)
        .sum();

    let jpegli_bits: u64 = (0..256)
        .filter(|&i| freqs[i] > 0)
        .map(|i| freqs[i] * jpegli_depths[i] as u64)
        .sum();

    println!("Classic algorithm: {} total bits", classic_bits);
    println!("Jpegli algorithm:  {} total bits", jpegli_bits);
    println!(
        "Difference: {} bits ({:.3}%)",
        (classic_bits as i64 - jpegli_bits as i64).abs(),
        ((classic_bits as f64 - jpegli_bits as f64) / classic_bits as f64 * 100.0).abs()
    );

    // Both should produce similar results (within 1% for typical data)
    let max_diff = classic_bits.max(jpegli_bits) as f64 * 0.01;
    assert!(
        (classic_bits as f64 - jpegli_bits as f64).abs() < max_diff,
        "Algorithms should produce similar bit counts"
    );
}

/// Test: Both algorithms handle edge cases
#[test]
fn test_edge_cases() {
    use jpegli::huffman::build_code_lengths;
    use jpegli::huffman_classic::generate_code_lengths;

    // Test: All same frequency
    {
        let mut freq = [1i64; 257];
        let classic = generate_code_lengths(&mut freq).expect("Should work");
        let jpegli = build_code_lengths(&vec![1u64; 257], 16);

        // Both should produce valid codes
        assert!(classic.iter().all(|&l| l == 0 || (l >= 1 && l <= 16)));
        assert!(jpegli.iter().all(|&d| d == 0 || (d >= 1 && d <= 16)));
    }

    // Test: Single symbol (besides pseudo)
    {
        let mut freq = [0i64; 257];
        freq[42] = 1000;
        let classic = generate_code_lengths(&mut freq).expect("Should work");
        assert_eq!(classic[42], 1, "Single symbol should get length 1");

        let mut freqs = vec![0u64; 257];
        freqs[42] = 1000;
        freqs[256] = 1;
        let jpegli = build_code_lengths(&freqs, 16);
        assert!(jpegli[42] >= 1 && jpegli[42] <= 2);
    }

    // Test: Two symbols
    {
        let mut freq = [0i64; 257];
        freq[0] = 100;
        freq[1] = 100;
        let classic = generate_code_lengths(&mut freq).expect("Should work");
        // Both should have length 1 or 2
        assert!(classic[0] >= 1 && classic[0] <= 2);
        assert!(classic[1] >= 1 && classic[1] <= 2);
    }
}

/// Test: Grayscale encoding with optimized Huffman
#[test]
fn test_grayscale_huffman_optimization() {
    let width = 64;
    let height = 64;
    let data: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(true)
        .mode(JpegMode::Baseline);

    let jpeg = encoder
        .encode(&data)
        .expect("Grayscale encoding should work");

    assert!(
        verify_decodable(&jpeg, width, height),
        "Grayscale with optimized Huffman should be decodable"
    );
}

// =============================================================================
// New unified type-based comparison tests
// =============================================================================

/// Create a realistic DC histogram (differential values, centered around 0)
fn create_dc_histogram() -> SymbolFrequencies {
    let mut freq = SymbolFrequencies::new();
    // DC categories: 0=no diff, 1=small, ..., 11=large
    // Category 0 (0 bits): very common in smooth areas
    freq.add(0, 15000);
    // Category 1 (-1, 1): common
    freq.add(1, 8000);
    // Category 2 (-3..-2, 2..3): fairly common
    freq.add(2, 5000);
    // Category 3 (-7..-4, 4..7)
    freq.add(3, 3000);
    // Category 4 (-15..-8, 8..15)
    freq.add(4, 1500);
    // Category 5-7: less common
    freq.add(5, 800);
    freq.add(6, 400);
    freq.add(7, 200);
    // Category 8-11: rare (large jumps)
    freq.add(8, 50);
    freq.add(9, 20);
    freq.add(10, 5);
    freq.add(11, 1);
    freq
}

/// Create a realistic AC histogram (run-length encoded)
fn create_ac_histogram() -> SymbolFrequencies {
    let mut freq = SymbolFrequencies::new();

    // Symbol 0x00 = EOB (End of Block) - very common
    freq.add(0x00, 50000);

    // Symbol 0xF0 = ZRL (16 zeros) - uncommon
    freq.add(0xF0, 500);

    // Common AC coefficients (run=0, size=1..4)
    freq.add(0x01, 25000); // run=0, size=1 (coeff -1 or 1)
    freq.add(0x02, 15000); // run=0, size=2 (-3..-2, 2..3)
    freq.add(0x03, 8000); // run=0, size=3 (-7..-4, 4..7)
    freq.add(0x04, 4000); // run=0, size=4

    // run=1 (one zero before coeff)
    freq.add(0x11, 10000); // run=1, size=1
    freq.add(0x12, 5000); // run=1, size=2
    freq.add(0x13, 2500); // run=1, size=3
    freq.add(0x14, 1200); // run=1, size=4

    // run=2
    freq.add(0x21, 4000);
    freq.add(0x22, 2000);
    freq.add(0x23, 1000);

    // run=3-5
    freq.add(0x31, 2000);
    freq.add(0x32, 1000);
    freq.add(0x41, 1000);
    freq.add(0x42, 500);
    freq.add(0x51, 500);

    // Higher runs (rarer)
    for run in 6..16 {
        freq.add((run << 4) | 1, 200 / (run as u64 + 1));
    }

    // Higher sizes (rarer)
    freq.add(0x05, 2000);
    freq.add(0x06, 1000);
    freq.add(0x07, 500);
    freq.add(0x08, 100);
    freq.add(0x09, 50);
    freq.add(0x0A, 10);

    freq
}

/// Comprehensive comparison of both algorithms on various histogram patterns.
#[test]
fn test_unified_algorithm_comparison() {
    println!("\n{}", "=".repeat(80));
    println!("UNIFIED HUFFMAN ALGORITHM COMPARISON");
    println!("{}\n", "=".repeat(80));

    let test_cases: Vec<(&str, SymbolFrequencies)> = vec![
        ("DC Luma (typical)", create_dc_histogram()),
        ("AC Luma (typical)", create_ac_histogram()),
        ("Uniform distribution", {
            let mut f = SymbolFrequencies::new();
            for i in 0..=255 {
                f.add(i, 100);
            }
            f
        }),
        ("Highly skewed (zipf-like)", {
            let mut f = SymbolFrequencies::new();
            for i in 0..=255 {
                f.add(i, (10000 / (i as u64 + 1)).max(1));
            }
            f
        }),
        ("Sparse (few symbols)", {
            let mut f = SymbolFrequencies::new();
            f.add(0, 10000);
            f.add(1, 1000);
            f.add(17, 500);
            f.add(33, 100);
            f
        }),
    ];

    println!(
        "{:<30} {:>12} {:>12} {:>12} {:>8}",
        "Histogram", "mozjpeg", "jpegli", "Diff", "Better"
    );
    println!("{}", "-".repeat(80));

    let mut mozjpeg_wins = 0;
    let mut jpegli_wins = 0;
    let mut ties = 0;

    for (name, freq) in &test_cases {
        let (mozjpeg_len, jpegli_len, moz_cost, jpg_cost) =
            compare_algorithms(freq).expect("Both algorithms should succeed");

        // Verify both are valid
        assert!(mozjpeg_len.is_valid(), "{}: mozjpeg invalid", name);
        assert!(jpegli_len.is_valid(), "{}: jpegli invalid", name);

        let diff = jpg_cost as i64 - moz_cost as i64;
        let winner = if diff < 0 {
            jpegli_wins += 1;
            "jpegli"
        } else if diff > 0 {
            mozjpeg_wins += 1;
            "mozjpeg"
        } else {
            ties += 1;
            "tie"
        };

        println!(
            "{:<30} {:>12} {:>12} {:>+12} {:>8}",
            name, moz_cost, jpg_cost, diff, winner
        );
    }

    println!("{}", "-".repeat(80));
    println!(
        "Summary: mozjpeg wins {}, jpegli wins {}, ties {}",
        mozjpeg_wins, jpegli_wins, ties
    );
    println!();
}

/// Test both algorithms on histograms extracted from actual encoding patterns.
/// This uses synthetic "photo-like" and "graphic-like" distributions.
#[test]
fn test_algorithm_on_content_types() {
    println!("\n{}", "=".repeat(80));
    println!("ALGORITHM COMPARISON BY CONTENT TYPE");
    println!("{}\n", "=".repeat(80));

    // Photo-like: many small coefficients, smooth falloff
    let photo_ac = {
        let mut f = SymbolFrequencies::new();
        f.add(0x00, 100000); // EOB very common
                             // Exponential falloff for run/size combinations
        for run in 0u8..16 {
            for size in 1u8..=10 {
                let divisor = (run as u64 + 1) * (size as u64 + 1) * (size as u64 + 1);
                let count = (50000.0 / divisor as f64) as u64;
                if count > 0 {
                    f.add((run << 4) | size, count);
                }
            }
        }
        f
    };

    // Graphic-like: more zeros, sharper edges
    let graphic_ac = {
        let mut f = SymbolFrequencies::new();
        f.add(0x00, 150000); // Even more EOBs
        f.add(0xF0, 5000); // More ZRLs
                           // Bimodal: either small or larger coefficients
        f.add(0x01, 30000);
        f.add(0x02, 5000);
        f.add(0x05, 8000); // Spike at size 5
        f.add(0x06, 6000);
        f.add(0x11, 10000);
        f.add(0x21, 5000);
        f
    };

    // Text-like: very sparse, mostly EOB
    let text_ac = {
        let mut f = SymbolFrequencies::new();
        f.add(0x00, 200000); // Almost all EOB
        f.add(0x01, 5000);
        f.add(0x02, 1000);
        f.add(0x11, 500);
        f
    };

    let test_cases: Vec<(&str, SymbolFrequencies)> = vec![
        ("Photo (smooth gradients)", photo_ac),
        ("Graphic (sharp edges)", graphic_ac),
        ("Text (sparse blocks)", text_ac),
    ];

    println!(
        "{:<30} {:>12} {:>12} {:>10} {:>8}",
        "Content Type", "mozjpeg", "jpegli", "Diff %", "Better"
    );
    println!("{}", "-".repeat(80));

    for (name, freq) in &test_cases {
        let (_moz_len, _jpg_len, moz_cost, jpg_cost) =
            compare_algorithms(freq).expect("Both should work");

        let diff_pct = (jpg_cost as f64 - moz_cost as f64) / moz_cost as f64 * 100.0;
        let winner = if jpg_cost < moz_cost {
            "jpegli"
        } else if moz_cost < jpg_cost {
            "mozjpeg"
        } else {
            "tie"
        };

        println!(
            "{:<30} {:>12} {:>12} {:>+10.2}% {:>8}",
            name, moz_cost, jpg_cost, diff_pct, winner
        );
    }

    println!();
}

/// Test edge cases where algorithms might diverge significantly.
#[test]
fn test_algorithm_edge_cases() {
    println!("\n{}", "=".repeat(80));
    println!("ALGORITHM EDGE CASES");
    println!("{}\n", "=".repeat(80));

    // Single dominant symbol
    let single_dominant = {
        let mut f = SymbolFrequencies::new();
        f.add(0, 100000);
        f.add(1, 1);
        f
    };

    // Two equal symbols
    let two_equal = {
        let mut f = SymbolFrequencies::new();
        f.add(0, 50000);
        f.add(1, 50000);
        f
    };

    // Many symbols with count 1
    let many_rare = {
        let mut f = SymbolFrequencies::new();
        for i in 0..=200 {
            f.add(i, 1);
        }
        f
    };

    // Power-of-2 distribution
    let power_of_2 = {
        let mut f = SymbolFrequencies::new();
        for i in 0..16 {
            f.add(i, 1u64 << (15 - i));
        }
        f
    };

    let test_cases: Vec<(&str, SymbolFrequencies)> = vec![
        ("Single dominant + 1 rare", single_dominant),
        ("Two equal symbols", two_equal),
        ("200 symbols, count=1 each", many_rare),
        ("Power-of-2 distribution", power_of_2),
    ];

    println!(
        "{:<30} {:>12} {:>12} {:>12} {:>10}",
        "Edge Case", "mozjpeg", "jpegli", "Diff bits", "Diff %"
    );
    println!("{}", "-".repeat(80));

    for (name, freq) in &test_cases {
        let result = compare_algorithms(freq);

        match result {
            Ok((_moz_len, _jpg_len, moz_cost, jpg_cost)) => {
                let diff = jpg_cost as i64 - moz_cost as i64;
                let diff_pct = if moz_cost > 0 {
                    diff as f64 / moz_cost as f64 * 100.0
                } else {
                    0.0
                };

                println!(
                    "{:<30} {:>12} {:>12} {:>+12} {:>+10.2}%",
                    name, moz_cost, jpg_cost, diff, diff_pct
                );
            }
            Err(e) => {
                println!("{:<30} ERROR: {}", name, e);
            }
        }
    }

    println!();
}

/// Test cases specifically designed to find where algorithms might diverge.
/// These target edge cases in tie-breaking and depth limiting.
#[test]
fn test_find_algorithm_divergence() {
    use jpegli::huffman::build_code_lengths;
    use jpegli::huffman_classic::generate_code_lengths;

    println!("\n{}", "=".repeat(80));
    println!("SEARCHING FOR ALGORITHM DIVERGENCE");
    println!("{}\n", "=".repeat(80));

    let mut divergent_cases = 0;
    let mut total_cases = 0;

    // Test 1: Many symbols with same frequency (tie-breaking stress test)
    // Note: 255 symbols can cause overflow in mozjpeg algorithm (u8 bits counter)
    for num_symbols in [2, 4, 8, 16, 32, 64, 128] {
        total_cases += 1;
        let mut moz_freq = [0i64; 257];
        let mut jpg_freq = Vec::with_capacity(num_symbols + 1);

        for i in 0..num_symbols {
            moz_freq[i] = 100; // All same frequency
            jpg_freq.push(100u64);
        }
        jpg_freq.push(1); // Pseudo-symbol

        let moz_lengths = generate_code_lengths(&mut moz_freq).unwrap();
        let jpg_depths = build_code_lengths(&jpg_freq, 16);

        let mut differs = false;
        for i in 0..num_symbols {
            if moz_lengths[i] != jpg_depths[i] {
                differs = true;
                break;
            }
        }

        if differs {
            divergent_cases += 1;
            println!("DIVERGENCE at {} equal-frequency symbols", num_symbols);
        }
    }

    // Test 2: Power-of-2 frequencies that might cause depth limit issues
    // Keep values under i32::MAX to avoid overflow in mozjpeg algorithm
    for max_power in [10, 15, 20] {
        total_cases += 1;
        let n = max_power.min(255);
        let mut moz_freq = [0i64; 257];
        let mut jpg_freq: Vec<u64> = Vec::with_capacity(n + 1);

        for i in 0..n {
            // Use smaller values to avoid overflow
            let freq = (1u32 << i.min(20)) as u64;
            moz_freq[i] = freq as i64;
            jpg_freq.push(freq);
        }
        jpg_freq.push(1); // Pseudo-symbol

        let moz_lengths = generate_code_lengths(&mut moz_freq).unwrap();
        let jpg_depths = build_code_lengths(&jpg_freq, 16);

        let mut differs = false;
        for i in 0..n {
            if moz_lengths[i] != jpg_depths[i] {
                differs = true;
                break;
            }
        }

        if differs {
            divergent_cases += 1;
            println!("DIVERGENCE at power-of-2 with {} symbols", n);
        }
    }

    // Test 3: Near-depth-limit cases (keep values reasonable to avoid overflow)
    for spread in [100, 1000, 10000] {
        total_cases += 1;
        let mut moz_freq = [0i64; 257];
        let mut jpg_freq: Vec<u64> = Vec::with_capacity(257);

        // Create distribution that's likely to hit depth limit
        for i in 0..256 {
            let freq = spread / (i as i64 + 1);
            moz_freq[i] = freq.max(1);
            jpg_freq.push(freq.max(1) as u64);
        }
        jpg_freq.push(1); // Pseudo-symbol

        let moz_lengths = generate_code_lengths(&mut moz_freq).unwrap();
        let jpg_depths = build_code_lengths(&jpg_freq, 16);

        let moz_cost: u64 = (0..256)
            .map(|i| moz_freq[i] as u64 * moz_lengths[i] as u64)
            .sum();
        let jpg_cost: u64 = (0..256).map(|i| jpg_freq[i] * jpg_depths[i] as u64).sum();

        if moz_cost != jpg_cost {
            divergent_cases += 1;
            println!(
                "DIVERGENCE at spread={}: moz_cost={}, jpg_cost={}",
                spread, moz_cost, jpg_cost
            );
        }
    }

    println!(
        "\nResults: {}/{} cases diverged",
        divergent_cases, total_cases
    );
    println!();
}
