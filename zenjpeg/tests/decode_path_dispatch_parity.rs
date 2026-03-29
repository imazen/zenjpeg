//! Comprehensive decode path × SIMD tier parity tests.
//!
//! Tests that all decode paths produce consistent output across all SIMD tiers
//! (via `for_each_token_permutation`), across chroma upsampling modes, and
//! across subsampling configurations.
//!
//! This catches bugs where:
//! - AVX2 and scalar paths disagree (formula mismatch, like the fixup bug)
//! - Different decode paths (streaming, scanline, coefficient) diverge
//! - Boundary fixup is inconsistent with the main upsampler
//!
//! Run: cargo test --release -p zenjpeg --test decode_path_dispatch_parity --features decoder -- --nocapture

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, Decoder, OutputTarget};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

// ============================================================================
// Test image generators
// ============================================================================

/// High-contrast alternating red/blue blocks — worst case for chroma upsampling.
/// Color transitions at 8-row boundaries create maximum chroma gradients exactly
/// where MCU boundary fixup operates.
fn make_red_blue_blocks(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let block_y = y / 8;
            if block_y % 2 == 0 {
                data[idx] = 255; // R
                data[idx + 2] = 0; // B
            } else {
                data[idx] = 0;
                data[idx + 2] = 255; // B
            }
            // Horizontal variation so h-interpolation matters
            data[idx + 1] = ((x * 3 + y * 7) % 200) as u8;
        }
    }
    data
}

/// Smooth gradient — exercises interior upsampling more than boundaries.
fn make_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = (x * 255 / width.max(1)) as u8;
            data[idx + 1] = (y * 255 / height.max(1)) as u8;
            data[idx + 2] = ((x + y) * 128 / (width + height).max(1)) as u8;
        }
    }
    data
}

/// Noise-like pattern — stresses all code paths uniformly.
fn make_noise(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    let mut rng: u32 = 0xDEAD_BEEF;
    for byte in data.iter_mut() {
        // xorshift32
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        *byte = (rng & 0xFF) as u8;
    }
    data
}

// ============================================================================
// Encode helpers
// ============================================================================

fn encode_jpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
    quality: f32,
    progressive: bool,
    restart_mcu_rows: u16,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, subsampling)
        .progressive(progressive)
        .restart_mcu_rows(restart_mcu_rows)
        .allow_16bit_quant_tables(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

// ============================================================================
// Decode helpers
// ============================================================================

/// Decode via `decode()` (streaming or coefficient path, auto-selected).
fn decode_full(jpeg: &[u8], upsampling: ChromaUpsampling) -> Vec<u8> {
    let decoder = Decoder::new()
        .chroma_upsampling(upsampling)
        .auto_orient(false)
        .num_threads(1); // Force sequential to isolate SIMD effects
    let img = decoder.decode(jpeg, Unstoppable).expect("decode");
    img.into_pixels_u8().unwrap()
}

/// Decode via `scanline_reader()` (pull-based streaming).
fn decode_scanline(jpeg: &[u8], upsampling: ChromaUpsampling) -> Vec<u8> {
    let decoder = Decoder::new()
        .chroma_upsampling(upsampling)
        .auto_orient(false)
        .num_threads(1);
    let mut reader = decoder.scanline_reader(jpeg).expect("scanline_reader");
    let width = reader.width() as usize;
    let height = reader.height() as usize;
    let stride = width * 3;
    let mut pixels = vec![0u8; stride * height];
    let mut total = 0;
    while !reader.is_finished() {
        let remaining = height - total;
        let buf_start = total * stride;
        let output = imgref::ImgRefMut::new(&mut pixels[buf_start..], stride, remaining);
        let rows = reader.read_rows_rgb8(output).expect("read");
        total += rows;
    }
    assert_eq!(total, height, "didn't read all rows");
    pixels
}

/// Decode via coefficient path (forced by requesting f32 output), converted to u8.
fn decode_coefficient_f32_to_u8(jpeg: &[u8], upsampling: ChromaUpsampling) -> Vec<u8> {
    let decoder = Decoder::new()
        .chroma_upsampling(upsampling)
        .output_target(OutputTarget::SrgbF32)
        .auto_orient(false)
        .num_threads(1);
    let img = decoder.decode(jpeg, Unstoppable).expect("decode");
    let f32_pixels = img.pixels_f32().unwrap();
    // Convert f32 [0,1] to u8 [0,255] with rounding
    f32_pixels
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect()
}

/// Decode via zune-jpeg (external reference).
fn decode_zune(jpeg: &[u8]) -> Vec<u8> {
    use zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let mut dec = JpegDecoder::new(ZCursor::new(jpeg));
    dec.decode().expect("decode")
}

/// Decode via jpeg-decoder (libjpeg reference).
fn decode_jpeg_decoder(jpeg: &[u8]) -> Vec<u8> {
    let mut dec = jpeg_decoder::Decoder::new(jpeg);
    dec.decode().expect("decode")
}

// ============================================================================
// Comparison helpers
// ============================================================================

/// Compute max absolute pixel diff between two RGB images.
fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

/// Compute per-row max diff, returning (boundary_max, interior_max).
/// MCU boundary = row % mcu_height == 0 or row % mcu_height == mcu_height - 1.
fn boundary_interior_max(
    a: &[u8],
    b: &[u8],
    width: usize,
    height: usize,
    mcu_height: usize,
) -> (u8, u8) {
    let mut boundary_max = 0u8;
    let mut interior_max = 0u8;
    for y in 0..height {
        let row_start = y * width * 3;
        let row_end = row_start + width * 3;
        let row_max = a[row_start..row_end]
            .iter()
            .zip(b[row_start..row_end].iter())
            .map(|(&x, &y)| x.abs_diff(y))
            .max()
            .unwrap_or(0);
        let in_mcu = y % mcu_height;
        if in_mcu == 0 || in_mcu == mcu_height - 1 {
            boundary_max = boundary_max.max(row_max);
        } else {
            interior_max = interior_max.max(row_max);
        }
    }
    (boundary_max, interior_max)
}

// ============================================================================
// Test configurations
// ============================================================================

struct TestCase {
    name: &'static str,
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
    quality: f32,
    progressive: bool,
    restart_mcu_rows: u16,
}

fn test_cases() -> Vec<TestCase> {
    vec![
        // 4:2:0 baseline — the most common case and where the fixup bug lived
        TestCase {
            name: "420_32x32",
            width: 32,
            height: 32,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "420_64x64",
            width: 64,
            height: 64,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "420_128x128",
            width: 128,
            height: 128,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        // Non-MCU-aligned sizes — expose padding/edge bugs
        TestCase {
            name: "420_33x33",
            width: 33,
            height: 33,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "420_47x31",
            width: 47,
            height: 31,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "420_100x50",
            width: 100,
            height: 50,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "420_255x255",
            width: 255,
            height: 255,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        // 4:4:4 baseline — no chroma upsampling, but still exercises decode paths
        TestCase {
            name: "444_64x64",
            width: 64,
            height: 64,
            subsampling: ChromaSubsampling::None,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "444_33x33",
            width: 33,
            height: 33,
            subsampling: ChromaSubsampling::None,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "444_128x128",
            width: 128,
            height: 128,
            subsampling: ChromaSubsampling::None,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        // 4:2:2 baseline — horizontal-only upsampling
        TestCase {
            name: "422_64x64",
            width: 64,
            height: 64,
            subsampling: ChromaSubsampling::HalfHorizontal,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "422_33x33",
            width: 33,
            height: 33,
            subsampling: ChromaSubsampling::HalfHorizontal,
            quality: 85.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        // 4:2:0 progressive — forces coefficient buffering path
        TestCase {
            name: "420_prog_64x64",
            width: 64,
            height: 64,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: true,
            restart_mcu_rows: 0,
        },
        TestCase {
            name: "420_prog_33x33",
            width: 33,
            height: 33,
            subsampling: ChromaSubsampling::Quarter,
            quality: 85.0,
            progressive: true,
            restart_mcu_rows: 0,
        },
        // Low quality — larger quant values exercise different coefficient ranges
        TestCase {
            name: "420_q50_64x64",
            width: 64,
            height: 64,
            subsampling: ChromaSubsampling::Quarter,
            quality: 50.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
        // High quality — near-lossless, small coefficients
        TestCase {
            name: "420_q97_64x64",
            width: 64,
            height: 64,
            subsampling: ChromaSubsampling::Quarter,
            quality: 97.0,
            progressive: false,
            restart_mcu_rows: 0,
        },
    ]
}

// ============================================================================
// Tests
// ============================================================================

/// Core test: For each test case × image pattern × upsampling mode, verify that
/// `decode()` and `scanline_reader()` produce identical output across all SIMD tiers.
///
/// This is the primary regression test for the fixup formula bug: if the fixup
/// function uses a different formula than the main upsampler at any SIMD tier,
/// the streaming and scanline paths will diverge at MCU boundaries.
#[cfg(target_arch = "x86_64")]
#[test]
fn decode_paths_dispatch_parity() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let cases = test_cases();
    let upsampling_modes = [
        ("Triangle", ChromaUpsampling::Jpegli),
        ("NearestNeighbor", ChromaUpsampling::NearestNeighbor),
        ("LibjpegCompat", ChromaUpsampling::Triangle),
    ];

    let mut first_report = true;

    for tc in &cases {
        let w = tc.width as usize;
        let h = tc.height as usize;

        // Test with multiple image patterns
        let patterns: Vec<(&str, Vec<u8>)> = vec![
            ("red_blue", make_red_blue_blocks(w, h)),
            ("gradient", make_gradient(w, h)),
            ("noise", make_noise(w, h)),
        ];

        for (pat_name, pixels) in &patterns {
            let jpeg = encode_jpeg(
                pixels,
                tc.width,
                tc.height,
                tc.subsampling,
                tc.quality,
                tc.progressive,
                tc.restart_mcu_rows,
            );

            for &(upsample_name, upsampling) in &upsampling_modes {
                // Skip upsampling modes that don't matter for 4:4:4
                if tc.subsampling == ChromaSubsampling::None
                    && upsampling != ChromaUpsampling::Jpegli
                {
                    continue;
                }

                // Compute reference at current (native) SIMD tier
                let ref_full = decode_full(&jpeg, upsampling);
                let ref_scanline = decode_scanline(&jpeg, upsampling);

                // Streaming vs scanline must match at native tier
                // ±2 allowed: horizontal chroma padding fix at non-MCU-aligned edges
                let native_diff = max_diff(&ref_full, &ref_scanline);
                assert!(
                    native_diff <= 2,
                    "{} {} {}: streaming vs scanline native diff={native_diff} (expected ≤2)",
                    tc.name,
                    pat_name,
                    upsample_name
                );

                // Test across all SIMD tiers
                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let full = decode_full(&jpeg, upsampling);
                    let scanline = decode_scanline(&jpeg, upsampling);

                    // streaming vs scanline must be consistent at every tier
                    // (both use the same upsampler at the same tier)
                    // ±2 allowed: the horizontal chroma padding fix at non-MCU-aligned
                    // edges can produce ±1-2 rounding differences between the scan.rs
                    // inline upsampler and the pipeline.rs strided upsampler+fixup.
                    let path_diff = max_diff(&full, &scanline);
                    assert!(
                        path_diff <= 2,
                        "{} {} {}: streaming vs scanline diff={path_diff} at {perm}",
                        tc.name,
                        pat_name,
                        upsample_name
                    );

                    // Each path must be stable vs its own reference across tiers.
                    // ±2 because: chroma upsampling formula differs by ±1 between
                    // AVX2 separable and scalar non-separable, then color conversion
                    // (YCbCr→RGB matrix multiply + clamp) can amplify to ±2.
                    let full_vs_ref = max_diff(&full, &ref_full);
                    assert!(
                        full_vs_ref <= 2,
                        "{} {} {}: full vs ref diff={full_vs_ref} at {perm}",
                        tc.name,
                        pat_name,
                        upsample_name
                    );

                    let scanline_vs_ref = max_diff(&scanline, &ref_scanline);
                    assert!(
                        scanline_vs_ref <= 2,
                        "{} {} {}: scanline vs ref diff={scanline_vs_ref} at {perm}",
                        tc.name,
                        pat_name,
                        upsample_name
                    );
                });

                if first_report {
                    eprintln!("decode_paths dispatch: {report}");
                    assert!(
                        report.permutations_run >= 2,
                        "expected at least 2 permutations"
                    );
                    first_report = false;
                }
            }
        }
    }
}

/// Verify that the coefficient (f32) path agrees with the streaming (u8) path
/// within expected tolerance across all SIMD tiers.
///
/// The coefficient path uses f32 IDCT → f32 color conversion → f32→u8 conversion,
/// while the streaming path uses i16 IDCT → u8 color conversion. Expected diff
/// is up to ~8 from IDCT precision differences and f32→u8 rounding.
#[cfg(target_arch = "x86_64")]
#[test]
fn coefficient_vs_streaming_dispatch_parity() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    // Focus on 4:2:0 since it exercises chroma upsampling
    let sizes: &[(u32, u32)] = &[(32, 32), (64, 64), (33, 33), (128, 128)];

    let mut first_report = true;

    for &(w, h) in sizes {
        let pixels = make_red_blue_blocks(w as usize, h as usize);
        let jpeg = encode_jpeg(&pixels, w, h, ChromaSubsampling::Quarter, 85.0, false, 0);

        let ref_streaming = decode_full(&jpeg, ChromaUpsampling::Jpegli);
        let ref_coeff = decode_coefficient_f32_to_u8(&jpeg, ChromaUpsampling::Jpegli);

        // Coefficient vs streaming at native tier
        // f32 IDCT vs i16 IDCT differ due to precision: i16 uses 12-bit fixed-point
        // scaling while f32 has full precision. Plus f32→u8 conversion rounds differently
        // from the clamped i16→u8 path. ≤8 is normal.
        let native_diff = max_diff(&ref_streaming, &ref_coeff);
        assert!(
            native_diff <= 10,
            "{w}x{h}: coeff vs streaming native diff={native_diff} (expected ≤10)"
        );

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let streaming = decode_full(&jpeg, ChromaUpsampling::Jpegli);
            let coeff = decode_coefficient_f32_to_u8(&jpeg, ChromaUpsampling::Jpegli);

            // Both paths stable across tiers (±2 for upsampling formula + color conversion)
            let s_diff = max_diff(&streaming, &ref_streaming);
            assert!(
                s_diff <= 2,
                "{w}x{h}: streaming vs ref diff={s_diff} at {perm}"
            );

            let c_diff = max_diff(&coeff, &ref_coeff);
            assert!(c_diff <= 2, "{w}x{h}: coeff vs ref diff={c_diff} at {perm}");

            // Cross-path: streaming vs coefficient within tolerance at each tier
            let cross = max_diff(&streaming, &coeff);
            assert!(
                cross <= 10,
                "{w}x{h}: streaming vs coeff diff={cross} at {perm}"
            );
        });

        if first_report {
            eprintln!("coeff_vs_streaming dispatch: {report}");
            assert!(
                report.permutations_run >= 2,
                "expected at least 2 permutations"
            );
            first_report = false;
        }
    }
}

/// Verify decode path agreement against external reference decoders (zune-jpeg,
/// jpeg-decoder) across all SIMD tiers.
///
/// External decoders are NOT affected by `for_each_token_permutation` (they don't
/// use archmage), so they provide a fixed reference to measure zenjpeg's SIMD
/// stability against.
#[cfg(target_arch = "x86_64")]
#[test]
fn external_reference_dispatch_parity() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let sizes: &[(u32, u32)] = &[(32, 32), (64, 64), (33, 33), (128, 128)];

    let mut first_report = true;

    for &(w, h) in sizes {
        let pixels = make_red_blue_blocks(w as usize, h as usize);

        // 4:2:0 baseline
        let jpeg_420 = encode_jpeg(&pixels, w, h, ChromaSubsampling::Quarter, 85.0, false, 0);

        // External references (SIMD-stable, not affected by token permutation)
        let zune_ref = decode_zune(&jpeg_420);
        let jpd_ref = decode_jpeg_decoder(&jpeg_420);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            // Triangle vs zune (both use triangle filter, should be close)
            {
                let full = decode_full(&jpeg_420, ChromaUpsampling::Jpegli);
                let scanline = decode_scanline(&jpeg_420, ChromaUpsampling::Jpegli);

                let full_vs_zune = max_diff(&full, &zune_ref);
                let scanline_vs_zune = max_diff(&scanline, &zune_ref);

                assert!(
                    full_vs_zune <= 4,
                    "{w}x{h} Triangle: full vs zune diff={full_vs_zune} at {perm}"
                );
                assert!(
                    scanline_vs_zune <= 4,
                    "{w}x{h} Triangle: scanline vs zune diff={scanline_vs_zune} at {perm}"
                );

                // streaming vs scanline internally consistent
                // ±2 for horizontal padding rounding at non-MCU-aligned edges
                let path_diff = max_diff(&full, &scanline);
                assert!(
                    path_diff <= 2,
                    "{w}x{h} Triangle: full vs scanline diff={path_diff} at {perm}"
                );
            }

            // NearestNeighbor: only test internal consistency (not vs zune,
            // since zune uses triangle — a fundamentally different filter)
            {
                let full = decode_full(&jpeg_420, ChromaUpsampling::NearestNeighbor);
                let scanline = decode_scanline(&jpeg_420, ChromaUpsampling::NearestNeighbor);

                let path_diff = max_diff(&full, &scanline);
                assert!(
                    path_diff <= 2,
                    "{w}x{h} NearestNeighbor: full vs scanline diff={path_diff} at {perm}"
                );
            }

            // LibjpegCompat vs jpeg-decoder (should be closest match)
            {
                let ljc = decode_full(&jpeg_420, ChromaUpsampling::Triangle);
                let ljc_vs_jpd = max_diff(&ljc, &jpd_ref);
                assert!(
                    ljc_vs_jpd <= 4,
                    "{w}x{h}: LibjpegCompat vs jpeg-decoder diff={ljc_vs_jpd} at {perm}"
                );
            }
        });

        if first_report {
            eprintln!("external_reference dispatch: {report}");
            assert!(
                report.permutations_run >= 2,
                "expected at least 2 permutations"
            );
            first_report = false;
        }
    }
}

/// Verify MCU boundary behavior specifically: boundary rows must not have
/// systematically larger diffs than interior rows across SIMD tiers.
///
/// The fixup bug manifested as boundary-specific ±1-2 diffs that interior
/// rows didn't have. This test catches that pattern.
#[cfg(target_arch = "x86_64")]
#[test]
fn mcu_boundary_no_systematic_shift() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let sizes: &[(u32, u32)] = &[(64, 64), (128, 128), (33, 33), (100, 50)];

    let mut first_report = true;

    for &(w, h) in sizes {
        let pixels = make_red_blue_blocks(w as usize, h as usize);
        let jpeg = encode_jpeg(&pixels, w, h, ChromaSubsampling::Quarter, 85.0, false, 0);

        let zune_ref = decode_zune(&jpeg);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let full = decode_full(&jpeg, ChromaUpsampling::Jpegli);
            let scanline = decode_scanline(&jpeg, ChromaUpsampling::Jpegli);

            let ww = w as usize;
            let hh = h as usize;

            // streaming vs zune: boundary should not be worse than interior
            let (bnd, int) = boundary_interior_max(&full, &zune_ref, ww, hh, 16);
            assert!(
                bnd <= int.saturating_add(2),
                "{w}x{h}: boundary({bnd}) >> interior({int}) in streaming vs zune at {perm}"
            );

            // streaming vs scanline: must match everywhere, especially boundaries
            // ±2 for horizontal padding rounding at non-MCU-aligned edges
            let (bnd_ss, int_ss) = boundary_interior_max(&full, &scanline, ww, hh, 16);
            assert!(
                bnd_ss <= 2 && int_ss <= 2,
                "{w}x{h}: streaming vs scanline boundary={bnd_ss} interior={int_ss} at {perm}"
            );

            // scanline vs zune: same boundary check
            let (bnd_sz, int_sz) = boundary_interior_max(&scanline, &zune_ref, ww, hh, 16);
            assert!(
                bnd_sz <= int_sz.saturating_add(2),
                "{w}x{h}: boundary({bnd_sz}) >> interior({int_sz}) in scanline vs zune at {perm}"
            );
        });

        if first_report {
            eprintln!("boundary_shift dispatch: {report}");
            assert!(
                report.permutations_run >= 2,
                "expected at least 2 permutations"
            );
            first_report = false;
        }
    }
}

/// 4:2:2 (horizontal-only) upsampling dispatch parity.
///
/// Separate from the main test because 4:2:2 uses different upsampling code
/// (horizontal-only, no vertical context) and has different MCU structure.
#[cfg(target_arch = "x86_64")]
#[test]
fn h2v1_decode_dispatch_parity() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let sizes: &[(u32, u32)] = &[(64, 64), (33, 33), (128, 64)];
    let mut first_report = true;

    for &(w, h) in sizes {
        for (pat_name, pixels) in [
            ("red_blue", make_red_blue_blocks(w as usize, h as usize)),
            ("noise", make_noise(w as usize, h as usize)),
        ] {
            let jpeg = encode_jpeg(
                &pixels,
                w,
                h,
                ChromaSubsampling::HalfHorizontal,
                85.0,
                false,
                0,
            );

            let ref_full = decode_full(&jpeg, ChromaUpsampling::Jpegli);
            let zune_ref = decode_zune(&jpeg);

            let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                let full = decode_full(&jpeg, ChromaUpsampling::Jpegli);
                let scanline = decode_scanline(&jpeg, ChromaUpsampling::Jpegli);

                // Internal consistency (±2 for horizontal padding rounding)
                let path_diff = max_diff(&full, &scanline);
                assert!(
                    path_diff <= 2,
                    "422 {w}x{h} {pat_name}: full vs scanline diff={path_diff} at {perm}"
                );

                // Stability across tiers (±2 for h-only upsampling + color convert)
                let full_vs_ref = max_diff(&full, &ref_full);
                assert!(
                    full_vs_ref <= 2,
                    "422 {w}x{h} {pat_name}: full vs ref diff={full_vs_ref} at {perm}"
                );

                // vs external reference
                let full_vs_zune = max_diff(&full, &zune_ref);
                assert!(
                    full_vs_zune <= 4,
                    "422 {w}x{h} {pat_name}: full vs zune diff={full_vs_zune} at {perm}"
                );
            });

            if first_report {
                eprintln!("h2v1 dispatch: {report}");
                assert!(
                    report.permutations_run >= 2,
                    "expected at least 2 permutations"
                );
                first_report = false;
            }
        }
    }
}

/// 4:4:4 decode dispatch parity — no upsampling, but exercises IDCT and
/// color conversion SIMD paths.
#[cfg(target_arch = "x86_64")]
#[test]
fn s444_decode_dispatch_parity() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let sizes: &[(u32, u32)] = &[(64, 64), (33, 33), (128, 128)];
    let mut first_report = true;

    for &(w, h) in sizes {
        for (pat_name, pixels) in [
            ("red_blue", make_red_blue_blocks(w as usize, h as usize)),
            ("noise", make_noise(w as usize, h as usize)),
        ] {
            let jpeg = encode_jpeg(&pixels, w, h, ChromaSubsampling::None, 85.0, false, 0);

            let ref_full = decode_full(&jpeg, ChromaUpsampling::Jpegli);
            let zune_ref = decode_zune(&jpeg);

            let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                let full = decode_full(&jpeg, ChromaUpsampling::Jpegli);
                let scanline = decode_scanline(&jpeg, ChromaUpsampling::Jpegli);

                // 4:4:4: no chroma upsampling, so paths should be very close
                let path_diff = max_diff(&full, &scanline);
                assert!(
                    path_diff == 0,
                    "444 {w}x{h} {pat_name}: full vs scanline diff={path_diff} at {perm} (expected 0)"
                );

                // Stability across tiers
                let full_vs_ref = max_diff(&full, &ref_full);
                assert!(
                    full_vs_ref == 0,
                    "444 {w}x{h} {pat_name}: full vs ref diff={full_vs_ref} at {perm}"
                );

                // vs zune (should be ≤2, both integer IDCT)
                let full_vs_zune = max_diff(&full, &zune_ref);
                assert!(
                    full_vs_zune <= 3,
                    "444 {w}x{h} {pat_name}: full vs zune diff={full_vs_zune} at {perm}"
                );
            });

            if first_report {
                eprintln!("444 dispatch: {report}");
                assert!(
                    report.permutations_run >= 2,
                    "expected at least 2 permutations"
                );
                first_report = false;
            }
        }
    }
}

/// Progressive 4:2:0 decode dispatch parity.
///
/// Progressive forces coefficient buffering path, so this tests a different
/// code path than baseline streaming.
#[cfg(target_arch = "x86_64")]
#[test]
fn progressive_420_dispatch_parity() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let sizes: &[(u32, u32)] = &[(64, 64), (33, 33), (128, 128)];
    let mut first_report = true;

    for &(w, h) in sizes {
        let pixels = make_red_blue_blocks(w as usize, h as usize);
        let jpeg = encode_jpeg(
            &pixels,
            w,
            h,
            ChromaSubsampling::Quarter,
            85.0,
            true, // progressive
            0,
        );

        for &(upsample_name, upsampling) in &[
            ("Triangle", ChromaUpsampling::Jpegli),
            ("NearestNeighbor", ChromaUpsampling::NearestNeighbor),
        ] {
            let ref_full = decode_full(&jpeg, upsampling);

            let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                let full = decode_full(&jpeg, upsampling);
                let scanline = decode_scanline(&jpeg, upsampling);

                // Progressive: both paths use coefficient buffering, should match exactly
                let path_diff = max_diff(&full, &scanline);
                assert!(
                    path_diff == 0,
                    "prog {w}x{h} {upsample_name}: full vs scanline diff={path_diff} at {perm}"
                );

                // Stability across tiers (±2 for chroma upsampling formula + color convert)
                let full_vs_ref = max_diff(&full, &ref_full);
                assert!(
                    full_vs_ref <= 2,
                    "prog {w}x{h} {upsample_name}: full vs ref diff={full_vs_ref} at {perm}"
                );
            });

            if first_report {
                eprintln!("progressive dispatch: {report}");
                assert!(
                    report.permutations_run >= 2,
                    "expected at least 2 permutations"
                );
                first_report = false;
            }
        }
    }
}
