//! Decoder error handling tests matching C++ jpegli.
//!
//! These tests validate that the decoder correctly rejects malformed JPEG data.
//! Test cases are taken directly from lib/jpegli/error_handling_test.cc.
//!
//! The kCompressed0 test data is a minimal valid 1x1 grayscale JPEG.

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;

// ============================================================================
// Test Data (from C++ error_handling_test.cc lines 1005-1054)
// ============================================================================

/// Minimal valid 1x1 grayscale JPEG for mutation testing.
/// This is the exact kCompressed0 from C++ error_handling_test.cc.
#[rustfmt::skip]
const COMPRESSED_0: &[u8] = &[
    // SOI
    0xff, 0xd8,
    // SOF (offset 2)
    0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
    0x01, 0x11, 0x00,
    // DQT (offset 15)
    0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02,
    0x02, 0x03, 0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05,
    0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07, 0x07, 0x06,
    0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d,
    0x0e, 0x12, 0x10, 0x0d, 0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10,
    0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
    0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14,
    // DHT (offset 84)
    0xff, 0xc4, 0x00, 0xd2, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02,
    0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31,
    0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
    0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
    0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94,
    0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
    0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
    0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
    // SOS (offset 296)
    0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00,
    // entropy coded data
    0xfc, 0xaa, 0xaf,
    // EOI
    0xff, 0xd9,
];

/// Marker offsets in COMPRESSED_0 (from C++ error_handling_test.cc)
const SOF_OFFSET: usize = 2;
const DQT_OFFSET: usize = 15;
const DHT_OFFSET: usize = 84;
const SOS_OFFSET: usize = 296;

/// Helper to parse compressed data and return success/failure.
fn parse_compressed(data: &[u8]) -> bool {
    let decoder = Decoder::new();
    decoder.decode(data, Unstoppable).is_ok()
}

// ============================================================================
// Minimal Success Test
// ============================================================================

/// Verify that the unmodified test data decodes successfully.
/// C++ test: DecoderErrorHandlingTest.MinimalSuccess
#[test]
fn test_minimal_success() {
    // Verify marker offsets are correct
    assert_eq!(COMPRESSED_0[SOF_OFFSET], 0xff, "SOF marker byte");
    assert_eq!(COMPRESSED_0[DQT_OFFSET], 0xff, "DQT marker byte");
    assert_eq!(COMPRESSED_0[DHT_OFFSET], 0xff, "DHT marker byte");
    assert_eq!(COMPRESSED_0[SOS_OFFSET], 0xff, "SOS marker byte");

    // The base data should decode successfully
    assert!(
        parse_compressed(COMPRESSED_0),
        "Unmodified COMPRESSED_0 should decode successfully"
    );
}

// ============================================================================
// NoSOI Tests
// ============================================================================

/// Test corrupted SOI marker.
/// C++ test: DecoderErrorHandlingTest.NoSOI
#[test]
fn test_no_soi_pos0() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[0] = 0x00;
    assert!(
        !parse_compressed(&compressed),
        "Should reject corrupted SOI (pos 0)"
    );
}

#[test]
fn test_no_soi_pos1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[1] = 0x00;
    assert!(
        !parse_compressed(&compressed),
        "Should reject corrupted SOI (pos 1)"
    );
}

// ============================================================================
// InvalidDQT Tests
// ============================================================================

/// Test DQT with bad marker length.
/// C++ test: DecoderErrorHandlingTest.InvalidDQT (bad marker length)
#[test]
fn test_invalid_dqt_length_minus2() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 3] = compressed[DQT_OFFSET + 3].wrapping_sub(2);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with length -2"
    );
}

#[test]
fn test_invalid_dqt_length_minus1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 3] = compressed[DQT_OFFSET + 3].wrapping_sub(1);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with length -1"
    );
}

#[test]
fn test_invalid_dqt_length_plus1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 3] = compressed[DQT_OFFSET + 3].wrapping_add(1);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with length +1"
    );
}

#[test]
fn test_invalid_dqt_length_plus2() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 3] = compressed[DQT_OFFSET + 3].wrapping_add(2);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with length +2"
    );
}

/// Test DQT with invalid table index / precision.
/// C++ test: DecoderErrorHandlingTest.InvalidDQT (invalid table index / precision)
#[test]
fn test_invalid_dqt_table_index_0x20() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 4] = 0x20;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with table index 0x20"
    );
}

#[test]
fn test_invalid_dqt_table_index_0x05() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 4] = 0x05;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with table index 0x05"
    );
}

/// Test DQT with zero quant value.
/// C++ test: DecoderErrorHandlingTest.InvalidDQT (zero quant value)
#[test]
fn test_invalid_dqt_zero_quant_k0() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 5] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with zero quant at k=0"
    );
}

#[test]
fn test_invalid_dqt_zero_quant_k1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 5 + 1] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with zero quant at k=1"
    );
}

#[test]
fn test_invalid_dqt_zero_quant_k17() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 5 + 17] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with zero quant at k=17"
    );
}

#[test]
fn test_invalid_dqt_zero_quant_k63() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DQT_OFFSET + 5 + 63] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DQT with zero quant at k=63"
    );
}

// ============================================================================
// InvalidSOF Tests
// ============================================================================

/// Test SOF with bad marker length.
/// C++ test: DecoderErrorHandlingTest.InvalidSOF (bad marker length)
#[test]
fn test_invalid_sof_length_minus2() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 3] = compressed[SOF_OFFSET + 3].wrapping_sub(2);
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with length -2"
    );
}

#[test]
fn test_invalid_sof_length_minus1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 3] = compressed[SOF_OFFSET + 3].wrapping_sub(1);
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with length -1"
    );
}

#[test]
fn test_invalid_sof_length_plus1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 3] = compressed[SOF_OFFSET + 3].wrapping_add(1);
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with length +1"
    );
}

#[test]
fn test_invalid_sof_length_plus2() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 3] = compressed[SOF_OFFSET + 3].wrapping_add(2);
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with length +2"
    );
}

/// Test SOF with zero width, height, or num_components.
/// C++ test: DecoderErrorHandlingTest.InvalidSOF (zero width, height or num_components)
#[test]
fn test_invalid_sof_zero_height() {
    // Position 6 in SOF is the low byte of height (SOF_OFFSET + 6)
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 6] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with zero height"
    );
}

#[test]
fn test_invalid_sof_zero_width() {
    // Position 8 in SOF is the low byte of width (SOF_OFFSET + 8)
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 8] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with zero width"
    );
}

#[test]
fn test_invalid_sof_zero_num_components() {
    // Position 9 in SOF is num_components (SOF_OFFSET + 9)
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 9] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with zero num_components"
    );
}

/// Test SOF with invalid data precision.
/// C++ test: DecoderErrorHandlingTest.InvalidSOF (invalid data precision)
#[test]
fn test_invalid_sof_precision_0() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 4] = 0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with precision 0"
    );
}

#[test]
fn test_invalid_sof_precision_1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 4] = 1;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with precision 1"
    );
}

#[test]
fn test_invalid_sof_precision_127() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 4] = 127;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with precision 127"
    );
}

/// Test SOF with too many num_components.
/// C++ test: DecoderErrorHandlingTest.InvalidSOF (too many num_components)
#[test]
fn test_invalid_sof_num_components_5() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 9] = 5;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with 5 components"
    );
}

#[test]
fn test_invalid_sof_num_components_255() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 9] = 255;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with 255 components"
    );
}

/// Test SOF with invalid sampling factors.
/// C++ test: DecoderErrorHandlingTest.InvalidSOF (invalid sampling factors)
#[test]
fn test_invalid_sof_sampling_0x00() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 11] = 0x00;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with sampling factor 0x00"
    );
}

#[test]
fn test_invalid_sof_sampling_0x01() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 11] = 0x01;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with sampling factor 0x01"
    );
}

#[test]
fn test_invalid_sof_sampling_0x10() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 11] = 0x10;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with sampling factor 0x10"
    );
}

#[test]
fn test_invalid_sof_sampling_0x15() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 11] = 0x15;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with sampling factor 0x15"
    );
}

#[test]
fn test_invalid_sof_sampling_0x51() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 11] = 0x51;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with sampling factor 0x51"
    );
}

/// Test SOF with invalid quant table index.
/// C++ test: DecoderErrorHandlingTest.InvalidSOF (invalid quant table index)
#[test]
fn test_invalid_sof_quant_table_5() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 12] = 5;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with quant table index 5"
    );
}

#[test]
fn test_invalid_sof_quant_table_17() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOF_OFFSET + 12] = 17;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOF with quant table index 17"
    );
}

// ============================================================================
// InvalidDHT Tests
// ============================================================================

/// Test DHT with bad marker length.
/// C++ test: DecoderErrorHandlingTest.InvalidDHT (bad marker length)
#[test]
fn test_invalid_dht_length_minus2() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 3] = compressed[DHT_OFFSET + 3].wrapping_sub(2);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with length -2"
    );
}

#[test]
fn test_invalid_dht_length_minus1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 3] = compressed[DHT_OFFSET + 3].wrapping_sub(1);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with length -1"
    );
}

#[test]
fn test_invalid_dht_length_plus1() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 3] = compressed[DHT_OFFSET + 3].wrapping_add(1);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with length +1"
    );
}

#[test]
fn test_invalid_dht_length_plus2() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 3] = compressed[DHT_OFFSET + 3].wrapping_add(2);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with length +2"
    );
}

/// Test DHT with invalid high byte of length (causes overflow).
/// C++ test: DecoderErrorHandlingTest.InvalidDHT (length high byte +17)
#[test]
fn test_invalid_dht_length_high_byte() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 2] = compressed[DHT_OFFSET + 2].wrapping_add(17);
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with length high byte +17"
    );
}

/// Test DHT with invalid table slot_id.
/// C++ test: DecoderErrorHandlingTest.InvalidDHT (invalid table slot_id)
#[test]
fn test_invalid_dht_slot_id_0x05() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 4] = 0x05;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with slot_id 0x05"
    );
}

#[test]
fn test_invalid_dht_slot_id_0x15() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 4] = 0x15;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with slot_id 0x15"
    );
}

#[test]
fn test_invalid_dht_slot_id_0x20() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[DHT_OFFSET + 4] = 0x20;
    assert!(
        !parse_compressed(&compressed),
        "Should reject DHT with slot_id 0x20"
    );
}

// ============================================================================
// InvalidSOS Tests
// ============================================================================

/// Test SOS with invalid comps_in_scan.
/// C++ test: DecoderErrorHandlingTest.InvalidSOS (invalid comps_in_scan)
#[test]
fn test_invalid_sos_comps_in_scan_2() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 4] = 2;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with comps_in_scan=2"
    );
}

#[test]
fn test_invalid_sos_comps_in_scan_5() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 4] = 5;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with comps_in_scan=5"
    );
}

#[test]
fn test_invalid_sos_comps_in_scan_17() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 4] = 17;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with comps_in_scan=17"
    );
}

/// Test SOS with invalid Huffman table indexes.
/// C++ test: DecoderErrorHandlingTest.InvalidSOS (invalid Huffman table indexes)
#[test]
fn test_invalid_sos_huffman_0x05() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 6] = 0x05;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Huffman index 0x05"
    );
}

#[test]
fn test_invalid_sos_huffman_0x50() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 6] = 0x50;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Huffman index 0x50"
    );
}

#[test]
fn test_invalid_sos_huffman_0x15() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 6] = 0x15;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Huffman index 0x15"
    );
}

#[test]
fn test_invalid_sos_huffman_0x51() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 6] = 0x51;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Huffman index 0x51"
    );
}

/// Test SOS with invalid Ss/Se (spectral selection).
/// C++ test: DecoderErrorHandlingTest.InvalidSOS (invalid Ss/Se)
#[test]
fn test_invalid_sos_ss_64() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 7] = 64;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Ss=64"
    );
}

#[test]
fn test_invalid_sos_se_64() {
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 8] = 64;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Se=64"
    );
}

// ============================================================================
// SOS Ss > Se validation
// ============================================================================

/// Regression test: SOS with Ss > Se should be rejected.
/// This is invalid per JPEG spec (spectral start must be <= spectral end).
#[test]
fn test_invalid_sos_ss_greater_than_se() {
    let mut compressed = COMPRESSED_0.to_vec();
    // Set Ss=10, Se=5 (invalid: start > end)
    compressed[SOS_OFFSET + 7] = 10; // Ss
    compressed[SOS_OFFSET + 8] = 5; // Se
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Ss > Se"
    );
}

/// Regression test: SOS with Ss == Se should be accepted (single coefficient scan).
#[test]
fn test_valid_sos_ss_equals_se() {
    // Ss==Se is valid for progressive scans (single coefficient).
    // For baseline this doesn't matter since Ss=0, Se=63 is required,
    // but the parser shouldn't reject it at the SOS level.
    // This tests the boundary: Ss=0, Se=0 (DC-only scan in progressive).
    let mut compressed = COMPRESSED_0.to_vec();
    compressed[SOS_OFFSET + 7] = 0; // Ss
    compressed[SOS_OFFSET + 8] = 0; // Se
    // This may or may not decode successfully (depends on baseline mode
    // rejecting non-standard Ss/Se), but it should NOT crash.
    let _ = parse_compressed(&compressed);
}

/// Regression test: SOS with Ah > 13 should be rejected.
/// Valid range is 0-13 per JPEG spec (successive approximation bit position).
#[test]
fn test_invalid_sos_ah_out_of_range() {
    let mut compressed = COMPRESSED_0.to_vec();
    // Ah=14, Al=0 → ah_al = 0xE0
    compressed[SOS_OFFSET + 9] = 0xE0;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Ah=14 (out of range 0-13)"
    );
}

/// Regression test: SOS with Al > 13 should be rejected.
#[test]
fn test_invalid_sos_al_out_of_range() {
    let mut compressed = COMPRESSED_0.to_vec();
    // Ah=0, Al=14 → ah_al = 0x0E
    compressed[SOS_OFFSET + 9] = 0x0E;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Al=14 (out of range 0-13)"
    );
}

/// Regression test: SOS with Ah=15, Al=15 (both max invalid).
#[test]
fn test_invalid_sos_ah_al_both_out_of_range() {
    let mut compressed = COMPRESSED_0.to_vec();
    // Ah=15, Al=15 → ah_al = 0xFF
    compressed[SOS_OFFSET + 9] = 0xFF;
    assert!(
        !parse_compressed(&compressed),
        "Should reject SOS with Ah=15, Al=15"
    );
}

// ============================================================================
// MutateSingleBytes Test
// ============================================================================

/// Test that mutating single bytes doesn't cause crashes.
/// C++ test: DecoderErrorHandlingTest.MutateSingleBytes
///
/// Note: This test doesn't assert success/failure - it just verifies
/// that the decoder doesn't crash on arbitrary mutations.
#[test]
fn test_mutate_single_bytes() {
    let values = [0x00u8, 0x0f, 0xf0, 0xff];

    for pos in 0..COMPRESSED_0.len() {
        for &val in &values {
            let mut compressed = COMPRESSED_0.to_vec();
            compressed[pos] = val;
            // Just call parse - don't assert result, just verify no crash
            let _ = parse_compressed(&compressed);
        }
    }
}

// ============================================================================
// Additional robustness tests
// ============================================================================

/// Test with completely zeroed data of same length.
#[test]
fn test_all_zeros() {
    let zeros = vec![0u8; COMPRESSED_0.len()];
    assert!(!parse_compressed(&zeros), "Should reject all-zeros data");
}

/// Test with all 0xFF bytes.
#[test]
fn test_all_0xff() {
    let ones = vec![0xffu8; COMPRESSED_0.len()];
    assert!(!parse_compressed(&ones), "Should reject all-0xFF data");
}

// ============================================================================
// Scanline reader crash regressions
// ============================================================================

/// Regression test for fuzz-found index OOB in scanline decoder.
///
/// The crash file has a valid SOF0 (1x1 grayscale, sampling 1x1, quant 0)
/// but a malformed SOS with Huffman table indices dc=4, ac=15, both
/// exceeding MAX_HUFFMAN_TABLES (4). The scanline parser's `find_scan_info()`
/// lacked bounds checks on these indices, causing a panic at
/// `self.dc_tables[dc_idx]` in `decode_mcu_row`.
///
/// Crash artifact: zencodecs fuzz_push_decode crash-d387982cfd9793c47f9cd3411e37d01832524791
#[test]
fn test_scanline_huffman_table_oob() {
    let data = std::fs::read("tests/crash_repro/crash_scanline_huffman_table_oob.jpg")
        .expect("crash repro file should exist");

    // Full decode path (uses scan.rs parser, already validated)
    let decoder = Decoder::new();
    assert!(
        decoder.decode(&data, Unstoppable).is_err(),
        "Full decode should return error, not panic"
    );

    // Scanline reader path (uses find_scan_info in parser/mod.rs — this was the crash site)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decoder.scanline_reader(&data)
    }));
    match result {
        Ok(Err(_)) => {} // Expected: graceful error
        Ok(Ok(_)) => panic!("scanline_reader should reject malformed SOS"),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            panic!("scanline_reader panicked instead of returning error: {msg}");
        }
    }
}

// ============================================================================
// Parser Hardening Tests
// ============================================================================

/// Minimal valid 3-component (YCbCr) 1x1 JPEG for multi-component mutation tests.
///
/// Structure: SOI, SOF0 (3 components), DQT, DHT (DC0+AC0), SOS (3 comps), data, EOI.
#[rustfmt::skip]
fn make_3comp_jpeg() -> Vec<u8> {
    use enough::Unstoppable;
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let config = EncoderConfig::ycbcr(90, ChromaSubsampling::None);
    let mut enc = config.encode_from_bytes(2, 2, PixelLayout::Rgb8Srgb).unwrap();
    enc.push_packed(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0], Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// Duplicate component ID in SOS must be rejected.
#[test]
fn test_reject_duplicate_component_in_sos() {
    let jpeg = make_3comp_jpeg();

    // Find SOS marker (0xFF 0xDA)
    let sos_pos = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("SOS marker not found");

    // SOS layout after marker: length(2), num_components(1), then per-component: id(1), tables(1)
    // For 3 components: length=0x000C, Ns=3, [id1,tbl1, id2,tbl2, id3,tbl3], Ss, Se, AhAl
    let ns_offset = sos_pos + 4; // byte after the 2-byte length
    let first_id_offset = ns_offset + 1; // first component ID
    let third_id_offset = first_id_offset + 4; // third component ID (skip 2 bytes per component)

    // Mutate: set third component ID = first component ID (duplicate)
    let mut mutated = jpeg.clone();
    mutated[third_id_offset] = mutated[first_id_offset];

    let decoder = Decoder::new();
    let result = decoder.decode(&mutated, Unstoppable);
    assert!(result.is_err(), "should reject duplicate component in SOS");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("duplicate component"),
        "error should mention duplicate component, got: {err_msg}"
    );
}

/// DHT with symbol count > 256 must be rejected before allocation.
#[test]
fn test_reject_dht_symbol_count_over_256() {
    // Build a malformed DHT where the bits array sums to > 256.
    // Start with a valid JPEG, find DHT, mutate bits array.
    let mut data = Vec::new();
    data.extend_from_slice(&[0xFF, 0xD8]); // SOI

    // SOF0: 1x1 grayscale, 1 component
    data.extend_from_slice(&[
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
    ]);

    // DQT: table 0, all ones
    data.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    data.extend_from_slice(&[1u8; 64]);

    // Malformed DHT: table class 0, index 0, bits summing to 257
    // bits[0..16]: 16 entries each = 17 would be 272 symbols, but let's use
    // 16 × 16 = 256 is valid, 16 × 16 + 1 = 257 is invalid.
    // Set bits = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17]
    // Sum = 15*16 + 17 = 257
    let num_values: usize = 15 * 16 + 17; // 257
    let length: u16 = 2 + 1 + 16 + num_values as u16;
    data.extend_from_slice(&[0xFF, 0xC4]);
    data.extend_from_slice(&length.to_be_bytes());
    data.push(0x00); // DC table 0
    for i in 0..16u8 {
        data.push(if i == 15 { 17 } else { 16 });
    }
    // Values (257 bytes — we won't even get here, but fill anyway)
    data.extend(core::iter::repeat_n(0u8, num_values));

    // SOS + EOI (won't reach these)
    data.extend_from_slice(&[
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x00, 0xFF, 0xD9,
    ]);

    let decoder = Decoder::new();
    let result = decoder.decode(&data, Unstoppable);
    assert!(result.is_err(), "should reject DHT with >256 symbols");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("256") || err_msg.contains("symbol"),
        "error should mention symbol count, got: {err_msg}"
    );
}

/// Extraneous bytes between markers should produce a warning in Balanced mode
/// and an error in Strict mode.
#[test]
fn test_extraneous_inter_marker_bytes_strict() {
    // Insert garbage bytes between DQT and DHT markers
    let jpeg = COMPRESSED_0.to_vec();

    // Find DHT marker position
    let dht_pos = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC4])
        .expect("DHT marker not found");

    // Insert 5 garbage bytes before DHT
    let mut mutated = Vec::with_capacity(jpeg.len() + 5);
    mutated.extend_from_slice(&jpeg[..dht_pos]);
    mutated.extend_from_slice(&[0x42, 0x43, 0x44, 0x45, 0x46]); // garbage
    mutated.extend_from_slice(&jpeg[dht_pos..]);

    // Strict mode should reject
    let decoder = Decoder::new().strict();
    let result = decoder.decode(&mutated, Unstoppable);
    assert!(
        result.is_err(),
        "strict mode should reject extraneous inter-marker bytes"
    );
}

#[test]
fn test_extraneous_inter_marker_bytes_balanced() {
    // Insert garbage bytes between DQT and DHT markers
    let jpeg = COMPRESSED_0.to_vec();

    let dht_pos = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC4])
        .expect("DHT marker not found");

    let mut mutated = Vec::with_capacity(jpeg.len() + 5);
    mutated.extend_from_slice(&jpeg[..dht_pos]);
    mutated.extend_from_slice(&[0x42, 0x43, 0x44, 0x45, 0x46]); // garbage
    mutated.extend_from_slice(&jpeg[dht_pos..]);

    // Balanced mode should succeed with warning
    let decoder = Decoder::new();
    let result = decoder.decode(&mutated, Unstoppable);
    assert!(
        result.is_ok(),
        "balanced mode should accept extraneous bytes with warning"
    );
    let info = result.unwrap();
    let warnings = info.warnings();
    assert!(
        warnings.iter().any(|w| {
            let s = format!("{w}");
            s.contains("extraneous")
        }),
        "should have ExtraneousBytesSkipped warning, got: {:?}",
        warnings
    );
}

// ============================================================================
// Non-standard IJG libjpeg v9+ block size rejection
// ============================================================================

/// IJG libjpeg v9+ with `-block N` (N > 8) produces non-standard JPEGs where
/// Se (spectral selection end) exceeds 63. These require fundamentally different
/// DCT sizes and are not decodable by standard JPEG decoders.
///
/// The decoder should return an `UnsupportedFeature` error with a clear message
/// mentioning the non-standard block size, not a confusing "out of range" error.
#[test]
fn sos_se_out_of_range_ijg_block_size() {
    let data = include_bytes!("testdata/all_the_images/sos_se_outofrange_libjpeg9b.jpg");
    let decoder = Decoder::new();
    let result = decoder.decode(data, Unstoppable);
    assert!(
        result.is_err(),
        "should reject non-standard block size JPEG"
    );

    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("block size") || err_msg.contains("Se > 63"),
        "error should mention non-standard block size or Se > 63, got: {err_msg}"
    );
}
