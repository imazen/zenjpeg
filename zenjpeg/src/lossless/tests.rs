//! Tests for lossless JPEG transforms.

use super::coeff_transform::*;
use crate::foundation::consts::JPEG_NATURAL_ORDER;

/// Create a test block where coefficient at position (row, col) in the 8×8 matrix
/// has value `row * 8 + col + 1` (so DC = 1, and values increase left-to-right, top-to-bottom).
///
/// The block is stored in zigzag order.
fn make_test_block() -> [i16; 64] {
    let mut block = [0i16; 64];
    for z in 0..64 {
        let linear = JPEG_NATURAL_ORDER[z] as usize;
        // Value encodes the 8×8 position: row * 8 + col + 1
        block[z] = (linear + 1) as i16;
    }
    block
}

/// Read the coefficient at 8×8 position (row, col) from a zigzag-ordered block.
fn read_at(block: &[i16; 64], row: usize, col: usize) -> i16 {
    let linear = row * 8 + col;
    let z = super::coeff_transform::ZIGZAG_FROM_LINEAR[linear] as usize;
    block[z]
}

#[test]
fn test_zigzag_roundtrip() {
    // Verify our zigzag ↔ linear conversion is consistent
    for z in 0..64usize {
        let linear = JPEG_NATURAL_ORDER[z] as usize;
        let back = super::coeff_transform::ZIGZAG_FROM_LINEAR[linear];
        assert_eq!(back as usize, z, "roundtrip failed for zigzag index {z}");
    }
}

#[test]
fn test_identity_transform() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::None);
    let result = bt.apply(&block);
    assert_eq!(block, result, "identity transform should not change coefficients");
}

#[test]
fn test_flip_horizontal_negates_odd_cols() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::FlipHorizontal);
    let result = bt.apply(&block);

    for row in 0..8 {
        for col in 0..8 {
            let src_val = read_at(&block, row, col);
            let dst_val = read_at(&result, row, col);
            if col % 2 == 1 {
                assert_eq!(dst_val, -src_val,
                    "H-flip should negate at ({row},{col}): expected {}, got {dst_val}",
                    -src_val);
            } else {
                assert_eq!(dst_val, src_val,
                    "H-flip should preserve at ({row},{col}): expected {src_val}, got {dst_val}");
            }
        }
    }
}

#[test]
fn test_flip_vertical_negates_odd_rows() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::FlipVertical);
    let result = bt.apply(&block);

    for row in 0..8 {
        for col in 0..8 {
            let src_val = read_at(&block, row, col);
            let dst_val = read_at(&result, row, col);
            if row % 2 == 1 {
                assert_eq!(dst_val, -src_val,
                    "V-flip should negate at ({row},{col})");
            } else {
                assert_eq!(dst_val, src_val,
                    "V-flip should preserve at ({row},{col})");
            }
        }
    }
}

#[test]
fn test_transpose_swaps_row_col() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::Transpose);
    let result = bt.apply(&block);

    for row in 0..8 {
        for col in 0..8 {
            let src_val = read_at(&block, row, col);
            // After transpose, value at (row, col) in source appears at (col, row) in dest
            let dst_val = read_at(&result, col, row);
            assert_eq!(dst_val, src_val,
                "Transpose: src({row},{col})={src_val} should appear at dst({col},{row}), got {dst_val}");
        }
    }
}

#[test]
fn test_rotate90_is_transpose_plus_hflip() {
    let block = make_test_block();

    // Rotate90 should equal Transpose followed by FlipHorizontal
    let bt_rot90 = BlockTransform::for_transform(LosslessTransform::Rotate90);
    let result_rot90 = bt_rot90.apply(&block);

    let bt_transpose = BlockTransform::for_transform(LosslessTransform::Transpose);
    let bt_hflip = BlockTransform::for_transform(LosslessTransform::FlipHorizontal);
    let intermediate = bt_transpose.apply(&block);
    let result_composed = bt_hflip.apply(&intermediate);

    assert_eq!(result_rot90, result_composed,
        "Rotate90 should equal Transpose + FlipHorizontal");
}

#[test]
fn test_rotate180_is_hflip_plus_vflip() {
    let block = make_test_block();

    let bt_rot180 = BlockTransform::for_transform(LosslessTransform::Rotate180);
    let result_rot180 = bt_rot180.apply(&block);

    let bt_hflip = BlockTransform::for_transform(LosslessTransform::FlipHorizontal);
    let bt_vflip = BlockTransform::for_transform(LosslessTransform::FlipVertical);
    let intermediate = bt_hflip.apply(&block);
    let result_composed = bt_vflip.apply(&intermediate);

    assert_eq!(result_rot180, result_composed,
        "Rotate180 should equal FlipHorizontal + FlipVertical");
}

#[test]
fn test_rotate270_is_transpose_plus_vflip() {
    let block = make_test_block();

    let bt_rot270 = BlockTransform::for_transform(LosslessTransform::Rotate270);
    let result_rot270 = bt_rot270.apply(&block);

    let bt_transpose = BlockTransform::for_transform(LosslessTransform::Transpose);
    let bt_vflip = BlockTransform::for_transform(LosslessTransform::FlipVertical);
    let intermediate = bt_transpose.apply(&block);
    let result_composed = bt_vflip.apply(&intermediate);

    assert_eq!(result_rot270, result_composed,
        "Rotate270 should equal Transpose + FlipVertical");
}

#[test]
fn test_transverse_is_transpose_plus_rot180() {
    let block = make_test_block();

    let bt_transverse = BlockTransform::for_transform(LosslessTransform::Transverse);
    let result_transverse = bt_transverse.apply(&block);

    let bt_transpose = BlockTransform::for_transform(LosslessTransform::Transpose);
    let bt_rot180 = BlockTransform::for_transform(LosslessTransform::Rotate180);
    let intermediate = bt_transpose.apply(&block);
    let result_composed = bt_rot180.apply(&intermediate);

    assert_eq!(result_transverse, result_composed,
        "Transverse should equal Transpose + Rotate180");
}

#[test]
fn test_dc_coefficient_never_negated() {
    // The DC coefficient (position 0,0) has even row and even col,
    // so it should never be negated by any transform.
    let mut block = [0i16; 64];
    block[0] = 1000; // DC is at zigzag index 0

    for transform in [
        LosslessTransform::None,
        LosslessTransform::FlipHorizontal,
        LosslessTransform::FlipVertical,
        LosslessTransform::Transpose,
        LosslessTransform::Rotate90,
        LosslessTransform::Rotate180,
        LosslessTransform::Rotate270,
        LosslessTransform::Transverse,
    ] {
        let bt = BlockTransform::for_transform(transform);
        let result = bt.apply(&block);
        // DC is always at (0,0) → zigzag 0, and should never be negated
        assert_eq!(read_at(&result, 0, 0), 1000,
            "DC should not be negated by {:?}", transform);
    }
}

#[test]
fn test_four_rotations_is_identity() {
    // Applying Rotate90 four times should give back the original
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::Rotate90);

    let r1 = bt.apply(&block);
    let r2 = bt.apply(&r1);
    let r3 = bt.apply(&r2);
    let r4 = bt.apply(&r3);

    assert_eq!(block, r4, "four 90° rotations should be identity");
}

#[test]
fn test_double_flip_h_is_identity() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::FlipHorizontal);
    let r1 = bt.apply(&block);
    let r2 = bt.apply(&r1);
    assert_eq!(block, r2, "double horizontal flip should be identity");
}

#[test]
fn test_double_flip_v_is_identity() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::FlipVertical);
    let r1 = bt.apply(&block);
    let r2 = bt.apply(&r1);
    assert_eq!(block, r2, "double vertical flip should be identity");
}

#[test]
fn test_double_transpose_is_identity() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::Transpose);
    let r1 = bt.apply(&block);
    let r2 = bt.apply(&r1);
    assert_eq!(block, r2, "double transpose should be identity");
}

#[test]
fn test_double_rot180_is_identity() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::Rotate180);
    let r1 = bt.apply(&block);
    let r2 = bt.apply(&r1);
    assert_eq!(block, r2, "double 180° rotation should be identity");
}

#[test]
fn test_rot90_plus_rot270_is_identity() {
    let block = make_test_block();
    let bt90 = BlockTransform::for_transform(LosslessTransform::Rotate90);
    let bt270 = BlockTransform::for_transform(LosslessTransform::Rotate270);
    let r1 = bt90.apply(&block);
    let r2 = bt270.apply(&r1);
    assert_eq!(block, r2, "90° + 270° should be identity");
}

#[test]
fn test_double_transverse_is_identity() {
    let block = make_test_block();
    let bt = BlockTransform::for_transform(LosslessTransform::Transverse);
    let r1 = bt.apply(&block);
    let r2 = bt.apply(&r1);
    assert_eq!(block, r2, "double transverse should be identity");
}

#[test]
fn test_swaps_dimensions() {
    assert!(!LosslessTransform::None.swaps_dimensions());
    assert!(!LosslessTransform::FlipHorizontal.swaps_dimensions());
    assert!(!LosslessTransform::FlipVertical.swaps_dimensions());
    assert!(LosslessTransform::Transpose.swaps_dimensions());
    assert!(LosslessTransform::Rotate90.swaps_dimensions());
    assert!(!LosslessTransform::Rotate180.swaps_dimensions());
    assert!(LosslessTransform::Rotate270.swaps_dimensions());
    assert!(LosslessTransform::Transverse.swaps_dimensions());
}

// ===== Block grid remap tests =====

#[test]
fn test_block_remap_identity() {
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::None), (2, 3));
}

#[test]
fn test_block_remap_hflip() {
    // In a 10-wide grid, block at x=2 should go to x=7
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::FlipHorizontal), (7, 3));
}

#[test]
fn test_block_remap_vflip() {
    // In an 8-high grid, block at y=3 should go to y=4
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::FlipVertical), (2, 4));
}

#[test]
fn test_block_remap_transpose() {
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::Transpose), (3, 2));
}

#[test]
fn test_block_remap_rot90() {
    // Rotate90: (bx, by) → (bh-1-by, bx)
    // (2, 3) in 10×8 → (8-1-3, 2) = (4, 2) in 8×10 grid
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::Rotate90), (4, 2));
}

#[test]
fn test_block_remap_rot180() {
    // (2, 3) in 10×8 → (7, 4)
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::Rotate180), (7, 4));
}

#[test]
fn test_block_remap_rot270() {
    // Rotate270: (bx, by) → (by, bw-1-bx)
    // (2, 3) → (3, 10-1-2) = (3, 7) in 8×10 grid
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::Rotate270), (3, 7));
}

#[test]
fn test_block_remap_transverse() {
    // Transverse: (bx, by) → (bh-1-by, bw-1-bx)
    // (2, 3) → (8-1-3, 10-1-2) = (4, 7) in 8×10 grid
    assert_eq!(remap_block(2, 3, 10, 8, LosslessTransform::Transverse), (4, 7));
}

// ===== Full coefficient transform tests =====

use crate::decode::{ComponentCoefficients, DecodedCoefficients};

/// Create a simple 16×16 test image with 1 component (4 blocks in a 2×2 grid).
/// Each block has a distinct DC coefficient for tracking.
fn make_test_coefficients() -> DecodedCoefficients {
    let mut coeffs = vec![0i16; 4 * 64]; // 4 blocks, 64 coefficients each

    // Set DC values to identify each block
    // Block (0,0) DC=10, Block (1,0) DC=20, Block (0,1) DC=30, Block (1,1) DC=40
    coeffs[0 * 64] = 10;  // block 0 = (bx=0, by=0)
    coeffs[1 * 64] = 20;  // block 1 = (bx=1, by=0)
    coeffs[2 * 64] = 30;  // block 2 = (bx=0, by=1)
    coeffs[3 * 64] = 40;  // block 3 = (bx=1, by=1)

    DecodedCoefficients {
        width: 16,
        height: 16,
        components: vec![ComponentCoefficients {
            id: 1,
            coeffs,
            blocks_wide: 2,
            blocks_high: 2,
            h_samp: 1,
            v_samp: 1,
            quant_table_idx: 0,
        }],
        quant_tables: vec![Some([1u16; 64])],
    }
}

fn get_dc(result: &TransformedCoefficients, comp: usize, bx: usize, by: usize) -> i16 {
    let c = &result.components[comp];
    let idx = by * c.blocks_wide + bx;
    c.coeffs[idx * 64]
}

#[test]
fn test_transform_identity() {
    let coeffs = make_test_coefficients();
    let config = TransformConfig {
        transform: LosslessTransform::None,
        edge_handling: EdgeHandling::Trim,
    };
    let result = transform_coefficients(&coeffs, &config).unwrap();

    assert_eq!(result.width, 16);
    assert_eq!(result.height, 16);
    assert_eq!(get_dc(&result, 0, 0, 0), 10);
    assert_eq!(get_dc(&result, 0, 1, 0), 20);
    assert_eq!(get_dc(&result, 0, 0, 1), 30);
    assert_eq!(get_dc(&result, 0, 1, 1), 40);
}

#[test]
fn test_transform_hflip_block_positions() {
    let coeffs = make_test_coefficients();
    let config = TransformConfig {
        transform: LosslessTransform::FlipHorizontal,
        edge_handling: EdgeHandling::Trim,
    };
    let result = transform_coefficients(&coeffs, &config).unwrap();

    assert_eq!(result.width, 16);
    assert_eq!(result.height, 16);
    // Columns are mirrored: block (0,y) ↔ block (1,y)
    assert_eq!(get_dc(&result, 0, 0, 0), 20); // was at (1,0)
    assert_eq!(get_dc(&result, 0, 1, 0), 10); // was at (0,0)
    assert_eq!(get_dc(&result, 0, 0, 1), 40); // was at (1,1)
    assert_eq!(get_dc(&result, 0, 1, 1), 30); // was at (0,1)
}

#[test]
fn test_transform_vflip_block_positions() {
    let coeffs = make_test_coefficients();
    let config = TransformConfig {
        transform: LosslessTransform::FlipVertical,
        edge_handling: EdgeHandling::Trim,
    };
    let result = transform_coefficients(&coeffs, &config).unwrap();

    // Rows are mirrored: block (x,0) ↔ block (x,1)
    assert_eq!(get_dc(&result, 0, 0, 0), 30); // was at (0,1)
    assert_eq!(get_dc(&result, 0, 1, 0), 40); // was at (1,1)
    assert_eq!(get_dc(&result, 0, 0, 1), 10); // was at (0,0)
    assert_eq!(get_dc(&result, 0, 1, 1), 20); // was at (1,0)
}

#[test]
fn test_transform_transpose_swaps_dims() {
    let coeffs = make_test_coefficients();
    let config = TransformConfig {
        transform: LosslessTransform::Transpose,
        edge_handling: EdgeHandling::Trim,
    };
    let result = transform_coefficients(&coeffs, &config).unwrap();

    assert_eq!(result.width, 16);  // square, so same
    assert_eq!(result.height, 16);

    // (0,0)→(0,0), (1,0)→(0,1), (0,1)→(1,0), (1,1)→(1,1)
    assert_eq!(get_dc(&result, 0, 0, 0), 10);
    assert_eq!(get_dc(&result, 0, 1, 0), 30); // was (0,1) → now at (1,0)
    assert_eq!(get_dc(&result, 0, 0, 1), 20); // was (1,0) → now at (0,1)
    assert_eq!(get_dc(&result, 0, 1, 1), 40);
}

#[test]
fn test_transform_rot90_block_positions() {
    let coeffs = make_test_coefficients();
    let config = TransformConfig {
        transform: LosslessTransform::Rotate90,
        edge_handling: EdgeHandling::Trim,
    };
    let result = transform_coefficients(&coeffs, &config).unwrap();

    // Rotate90: (bx, by) → (bh-1-by, bx)
    // (0,0)→(1,0), (1,0)→(1,1), (0,1)→(0,0), (1,1)→(0,1)
    assert_eq!(get_dc(&result, 0, 0, 0), 30); // from (0,1)
    assert_eq!(get_dc(&result, 0, 1, 0), 10); // from (0,0)
    assert_eq!(get_dc(&result, 0, 0, 1), 40); // from (1,1)
    assert_eq!(get_dc(&result, 0, 1, 1), 20); // from (1,0)
}

#[test]
fn test_transform_rot180_block_positions() {
    let coeffs = make_test_coefficients();
    let config = TransformConfig {
        transform: LosslessTransform::Rotate180,
        edge_handling: EdgeHandling::Trim,
    };
    let result = transform_coefficients(&coeffs, &config).unwrap();

    // (0,0)→(1,1), (1,0)→(0,1), (0,1)→(1,0), (1,1)→(0,0)
    assert_eq!(get_dc(&result, 0, 0, 0), 40); // from (1,1)
    assert_eq!(get_dc(&result, 0, 1, 0), 30); // from (0,1)
    assert_eq!(get_dc(&result, 0, 0, 1), 20); // from (1,0)
    assert_eq!(get_dc(&result, 0, 1, 1), 10); // from (0,0)
}

#[test]
fn test_transform_nonsquare_transpose() {
    // 24×16 image (3×2 blocks) → transpose to 16×24 (2×3 blocks)
    let mut coeffs_data = vec![0i16; 6 * 64];
    for i in 0..6 {
        coeffs_data[i * 64] = (i as i16 + 1) * 10; // DC = 10, 20, 30, 40, 50, 60
    }

    let coeffs = DecodedCoefficients {
        width: 24,
        height: 16,
        components: vec![ComponentCoefficients {
            id: 1,
            coeffs: coeffs_data,
            blocks_wide: 3,
            blocks_high: 2,
            h_samp: 1,
            v_samp: 1,
            quant_table_idx: 0,
        }],
        quant_tables: vec![Some([1u16; 64])],
    };

    let config = TransformConfig {
        transform: LosslessTransform::Transpose,
        edge_handling: EdgeHandling::Trim,
    };
    let result = transform_coefficients(&coeffs, &config).unwrap();

    assert_eq!(result.width, 16);
    assert_eq!(result.height, 24);
    assert_eq!(result.components[0].blocks_wide, 2);
    assert_eq!(result.components[0].blocks_high, 3);

    // Source layout (blocks_wide=3, blocks_high=2):
    //   (0,0)=10  (1,0)=20  (2,0)=30
    //   (0,1)=40  (1,1)=50  (2,1)=60
    // After transpose (blocks_wide=2, blocks_high=3):
    //   (0,0)=10  (1,0)=40
    //   (0,1)=20  (1,1)=50
    //   (0,2)=30  (1,2)=60
    assert_eq!(get_dc(&result, 0, 0, 0), 10);
    assert_eq!(get_dc(&result, 0, 1, 0), 40);
    assert_eq!(get_dc(&result, 0, 0, 1), 20);
    assert_eq!(get_dc(&result, 0, 1, 1), 50);
    assert_eq!(get_dc(&result, 0, 0, 2), 30);
    assert_eq!(get_dc(&result, 0, 1, 2), 60);
}

// ===== End-to-end pipeline tests =====

mod pipeline_tests {
    use crate::lossless::{transform, LosslessTransform, TransformConfig, EdgeHandling};
    use crate::decode::DecodeConfig;
    use enough::Unstoppable;

    /// Create a test JPEG from a known pixel pattern using zenjpeg encoder.
    fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
        use crate::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout};

        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                // Create a pattern that's visually distinct when rotated
                pixels.push(((x * 255 / width) & 0xFF) as u8);  // R: gradient L→R
                pixels.push(((y * 255 / height) & 0xFF) as u8); // G: gradient T→B
                pixels.push(128u8);                               // B: constant
            }
        }

        let config = EncoderConfig::ycbcr(90, ChromaSubsampling::None);
        let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    /// Create a test JPEG with 4:2:0 subsampling.
    fn create_test_jpeg_420(width: u32, height: u32) -> Vec<u8> {
        use crate::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout};

        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 255 / width) & 0xFF) as u8);
                pixels.push(((y * 255 / height) & 0xFF) as u8);
                pixels.push(128u8);
            }
        }

        let config = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter);
        let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn test_roundtrip_identity() {
        // Transform with None should produce a valid JPEG that decodes to
        // the same pixels (modulo Huffman table differences)
        let jpeg = create_test_jpeg(64, 64);

        let result = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::None,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        // Result should be a valid JPEG
        assert!(result.len() > 100, "output too small");
        assert_eq!(result[0], 0xFF);
        assert_eq!(result[1], 0xD8); // SOI

        // Decode both and compare
        let decoder = DecodeConfig::new();
        let orig = decoder.decode(&jpeg, Unstoppable).unwrap();
        let transformed = decoder.decode(&result, Unstoppable).unwrap();

        assert_eq!(orig.width(), transformed.width());
        assert_eq!(orig.height(), transformed.height());

        // Pixels should be identical (lossless round-trip of coefficients)
        let orig_px = orig.pixels_u8().unwrap();
        let trans_px = transformed.pixels_u8().unwrap();
        assert_eq!(orig_px.len(), trans_px.len());

        let mut max_diff = 0u8;
        for (a, b) in orig_px.iter().zip(trans_px.iter()) {
            let diff = (*a as i16 - *b as i16).unsigned_abs() as u8;
            max_diff = max_diff.max(diff);
        }
        assert_eq!(max_diff, 0, "identity transform should produce identical pixels");
    }

    #[test]
    fn test_rotate90_dimensions() {
        let jpeg = create_test_jpeg(64, 48);

        let result = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::Rotate90,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let decoder = DecodeConfig::new();
        let decoded = decoder.decode(&result, Unstoppable).unwrap();

        // 64×48 rotated 90° → 48×64
        assert_eq!(decoded.width(), 48);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn test_rotate180_same_dimensions() {
        let jpeg = create_test_jpeg(64, 48);

        let result = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::Rotate180,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let decoder = DecodeConfig::new();
        let decoded = decoder.decode(&result, Unstoppable).unwrap();

        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 48);
    }

    #[test]
    fn test_double_rotate90_equals_rotate180() {
        let jpeg = create_test_jpeg(64, 64);

        let rot90_1 = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::Rotate90,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let rot90_2 = transform(&rot90_1, &TransformConfig {
            transform: LosslessTransform::Rotate90,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let rot180 = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::Rotate180,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        // Both should decode to the same pixels
        let decoder = DecodeConfig::new();
        let px_2x90 = decoder.decode(&rot90_2, Unstoppable).unwrap();
        let px_180 = decoder.decode(&rot180, Unstoppable).unwrap();

        let px_a = px_2x90.pixels_u8().unwrap();
        let px_b = px_180.pixels_u8().unwrap();
        assert_eq!(px_a.len(), px_b.len());

        let mut max_diff = 0u8;
        for (a, b) in px_a.iter().zip(px_b.iter()) {
            let diff = (*a as i16 - *b as i16).unsigned_abs() as u8;
            max_diff = max_diff.max(diff);
        }
        assert_eq!(max_diff, 0, "2×rot90 should equal rot180");
    }

    #[test]
    fn test_four_rotations_roundtrip() {
        let jpeg = create_test_jpeg(64, 64);

        let mut current = jpeg.clone();
        for _ in 0..4 {
            current = transform(&current, &TransformConfig {
                transform: LosslessTransform::Rotate90,
                edge_handling: EdgeHandling::Trim,
            }, Unstoppable).unwrap();
        }

        // After 4 rotations, should be back to original pixels
        let decoder = DecodeConfig::new();
        let orig = decoder.decode(&jpeg, Unstoppable).unwrap();
        let roundtrip = decoder.decode(&current, Unstoppable).unwrap();

        assert_eq!(orig.width(), roundtrip.width());
        assert_eq!(orig.height(), roundtrip.height());

        let px_orig = orig.pixels_u8().unwrap();
        let px_rt = roundtrip.pixels_u8().unwrap();

        let mut max_diff = 0u8;
        for (a, b) in px_orig.iter().zip(px_rt.iter()) {
            let diff = (*a as i16 - *b as i16).unsigned_abs() as u8;
            max_diff = max_diff.max(diff);
        }
        assert_eq!(max_diff, 0, "4×rot90 should equal identity");
    }

    #[test]
    fn test_hflip_roundtrip() {
        let jpeg = create_test_jpeg(64, 48);

        let flipped = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::FlipHorizontal,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let double_flipped = transform(&flipped, &TransformConfig {
            transform: LosslessTransform::FlipHorizontal,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let decoder = DecodeConfig::new();
        let orig = decoder.decode(&jpeg, Unstoppable).unwrap();
        let roundtrip = decoder.decode(&double_flipped, Unstoppable).unwrap();

        let px_orig = orig.pixels_u8().unwrap();
        let px_rt = roundtrip.pixels_u8().unwrap();

        let mut max_diff = 0u8;
        for (a, b) in px_orig.iter().zip(px_rt.iter()) {
            let diff = (*a as i16 - *b as i16).unsigned_abs() as u8;
            max_diff = max_diff.max(diff);
        }
        assert_eq!(max_diff, 0, "2×hflip should equal identity");
    }

    #[test]
    fn test_420_rotate90() {
        // Test with 4:2:0 subsampling (MCU size 16×16)
        let jpeg = create_test_jpeg_420(64, 48);

        let result = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::Rotate90,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let decoder = DecodeConfig::new();
        let decoded = decoder.decode(&result, Unstoppable).unwrap();

        // Should produce a valid, decodable JPEG with swapped dimensions
        assert_eq!(decoded.width(), 48);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn test_all_transforms_produce_valid_jpeg() {
        let jpeg = create_test_jpeg(64, 64);
        let decoder = DecodeConfig::new();

        for xform in [
            LosslessTransform::None,
            LosslessTransform::FlipHorizontal,
            LosslessTransform::FlipVertical,
            LosslessTransform::Transpose,
            LosslessTransform::Rotate90,
            LosslessTransform::Rotate180,
            LosslessTransform::Rotate270,
            LosslessTransform::Transverse,
        ] {
            let result = transform(&jpeg, &TransformConfig {
                transform: xform,
                edge_handling: EdgeHandling::Trim,
            }, Unstoppable).unwrap();

            // Should be a valid JPEG
            let decoded = decoder.decode(&result, Unstoppable);
            assert!(decoded.is_ok(),
                "transform {:?} should produce valid JPEG, got error: {:?}",
                xform, decoded.err());
        }
    }
}

mod coefficient_roundtrip_tests {
    use crate::lossless::{transform, LosslessTransform, TransformConfig, EdgeHandling};
    use crate::decode::DecodeConfig;
    use enough::Unstoppable;

    /// Create a test JPEG
    fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
        use crate::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout};
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 255 / width) & 0xFF) as u8);
                pixels.push(((y * 255 / height) & 0xFF) as u8);
                pixels.push(128u8);
            }
        }
        let config = EncoderConfig::ycbcr(90, ChromaSubsampling::None);
        let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn test_coefficient_identity_roundtrip() {
        // Compare coefficients before and after identity transform
        let jpeg = create_test_jpeg(64, 64);
        let decoder = DecodeConfig::new();

        let orig_coeffs = decoder.decode_coefficients(&jpeg, Unstoppable).unwrap();

        let result = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::None,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let new_coeffs = decoder.decode_coefficients(&result, Unstoppable).unwrap();

        // Compare coefficient by coefficient
        for (comp_idx, (c1, c2)) in orig_coeffs.components.iter().zip(&new_coeffs.components).enumerate() {
            assert_eq!(c1.blocks_wide, c2.blocks_wide, "blocks_wide mismatch comp {comp_idx}");
            assert_eq!(c1.blocks_high, c2.blocks_high, "blocks_high mismatch comp {comp_idx}");
            assert_eq!(c1.h_samp, c2.h_samp, "h_samp mismatch comp {comp_idx}");
            assert_eq!(c1.v_samp, c2.v_samp, "v_samp mismatch comp {comp_idx}");

            let num_blocks = c1.num_blocks().min(c2.num_blocks());
            let mut diffs = 0;
            let mut max_diff = 0i16;
            for block_idx in 0..num_blocks {
                let b1 = c1.block(block_idx);
                let b2 = c2.block(block_idx);
                for i in 0..64 {
                    let d = (b1[i] as i32 - b2[i] as i32).abs() as i16;
                    if d != 0 {
                        diffs += 1;
                        max_diff = max_diff.max(d);
                    }
                }
            }
            if diffs > 0 {
                eprintln!("Component {comp_idx}: {diffs} differing coefficients, max_diff={max_diff}");
            }
            assert_eq!(diffs, 0, "Component {comp_idx}: coefficients should be identical after identity transform");
        }
    }

    #[test]
    fn test_quant_tables_preserved() {
        let jpeg = create_test_jpeg(64, 64);
        let decoder = DecodeConfig::new();

        let orig_coeffs = decoder.decode_coefficients(&jpeg, Unstoppable).unwrap();

        let result = transform(&jpeg, &TransformConfig {
            transform: LosslessTransform::None,
            edge_handling: EdgeHandling::Trim,
        }, Unstoppable).unwrap();

        let new_coeffs = decoder.decode_coefficients(&result, Unstoppable).unwrap();

        // Quant tables should be preserved
        for (idx, (qt1, qt2)) in orig_coeffs.quant_tables.iter().zip(&new_coeffs.quant_tables).enumerate() {
            match (qt1, qt2) {
                (Some(t1), Some(t2)) => {
                    assert_eq!(t1, t2, "quant table {idx} should be preserved");
                }
                (None, None) => {}
                _ => panic!("quant table {idx} presence mismatch"),
            }
        }
    }
}

#[cfg(test)]
mod debug_tests {
    use crate::decode::DecodeConfig;
    use enough::Unstoppable;

    fn create_test_jpeg_444(width: u32, height: u32) -> Vec<u8> {
        use crate::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout};
        let pixels = vec![128u8; (width * height * 3) as usize];
        let config = EncoderConfig::ycbcr(90, ChromaSubsampling::None);
        let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn debug_quant_table_indices() {
        let jpeg = create_test_jpeg_444(64, 64);
        let decoder = DecodeConfig::new();
        let coeffs = decoder.decode_coefficients(&jpeg, Unstoppable).unwrap();

        eprintln!("Number of quant table slots: {}", coeffs.quant_tables.len());
        for (i, qt) in coeffs.quant_tables.iter().enumerate() {
            eprintln!("  Table {}: {}", i, if qt.is_some() { "present" } else { "absent" });
        }

        eprintln!("Number of components: {}", coeffs.components.len());
        for (i, comp) in coeffs.components.iter().enumerate() {
            eprintln!("  Component {}: id={}, h_samp={}, v_samp={}, blocks={}x{}",
                i, comp.id, comp.h_samp, comp.v_samp, comp.blocks_wide, comp.blocks_high);
        }
    }
}
