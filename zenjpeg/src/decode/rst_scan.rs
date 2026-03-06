//! SIMD-accelerated restart marker scanner.
//!
//! Scans raw JPEG entropy-coded data for RST markers (0xFF 0xD0-0xD7) using
//! AVX2 vectorized 0xFF detection, then validates marker bytes at each hit.
//!
//! This enables parallel decode by finding all restart boundaries upfront
//! without sequential entropy decoding.

/// A restart marker location in the entropy-coded data.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct RstMarkerPos {
    /// Byte offset of the 0xFF byte within the scan data slice.
    pub offset: usize,
    /// RST marker number (0-7), extracted from the second byte (0xD0-0xD7).
    pub rst_num: u8,
}

/// Result of scanning entropy-coded data for RST markers.
pub(crate) struct RstScanResult {
    /// All RST marker positions found, in order.
    pub markers: Vec<RstMarkerPos>,
    /// Byte offset where entropy-coded data ends (position of first non-RST
    /// marker like EOI, or end of data if no terminating marker found).
    pub entropy_end: usize,
}

/// Scan entropy-coded data for all RST markers.
///
/// Returns marker positions and the entropy data boundary. The scan data
/// should be the slice starting after the SOS header (entropy-coded segment).
///
/// `capacity_hint` pre-allocates the markers vector when the expected count
/// is known from DRI (restart interval) and total MCU count.
///
/// This function handles JPEG byte stuffing: 0xFF 0x00 is a stuffed byte
/// (not a marker), while 0xFF 0xD0-0xD7 are restart markers.
pub(crate) fn scan_rst_markers(scan_data: &[u8], capacity_hint: usize) -> RstScanResult {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return scan_rst_markers_avx2(token, scan_data, capacity_hint);
        }
    }

    scan_rst_markers_scalar(scan_data, capacity_hint)
}

/// Scalar fallback: simple byte-by-byte scan for 0xFF markers.
fn scan_rst_markers_scalar(data: &[u8], capacity_hint: usize) -> RstScanResult {
    let mut markers = Vec::with_capacity(capacity_hint);
    let mut i = 0;
    let len = data.len();

    while i + 1 < len {
        if data[i] == 0xFF {
            let next = data[i + 1];
            if (0xD0..=0xD7).contains(&next) {
                markers.push(RstMarkerPos {
                    offset: i,
                    rst_num: next - 0xD0,
                });
                i += 2;
                continue;
            }
            // 0xFF 0x00 = stuffed byte, 0xFF 0xFF = fill byte — skip
            if next == 0x00 || next == 0xFF {
                i += 2;
                continue;
            }
            // Any other marker (EOI, SOS, etc.) — stop scanning
            // RST markers only appear within entropy-coded segments
            break;
        }
        i += 1;
    }

    RstScanResult {
        markers,
        entropy_end: i,
    }
}

/// AVX2 vectorized RST marker scanner.
///
/// Uses `_mm256_cmpeq_epi8` to find all 0xFF bytes in 32-byte chunks,
/// then checks the byte after each 0xFF for RST marker range 0xD0-0xD7.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn scan_rst_markers_avx2(
    _token: archmage::X64V3Token,
    data: &[u8],
    capacity_hint: usize,
) -> RstScanResult {
    use core::arch::x86_64::*;
    use safe_unaligned_simd::x86_64 as safe_simd;

    let mut markers = Vec::with_capacity(capacity_hint);
    let len = data.len();
    if len < 2 {
        return RstScanResult {
            markers,
            entropy_end: 0,
        };
    }

    let chunk_size = 32;
    let simd_end = len.saturating_sub(chunk_size);
    let mut i = 0;

    // Broadcast 0xFF for comparison
    let ff_vec = _mm256_set1_epi8(-1i8); // 0xFF as i8

    while i < simd_end {
        // Load 32 bytes and reinterpret as 16 i16s for _mm256_loadu_si256.
        // Copy to stack first: bytemuck::cast on owned values handles alignment,
        // while cast_ref would panic on unaligned source data.
        let bytes: [u8; 32] = data[i..i + 32].try_into().unwrap();
        let bytes_as_i16: [i16; 16] = bytemuck::cast(bytes);
        let chunk = safe_simd::_mm256_loadu_si256(&bytes_as_i16);

        // Compare each byte to 0xFF
        let cmp = _mm256_cmpeq_epi8(chunk, ff_vec);
        let mut mask = _mm256_movemask_epi8(cmp) as u32;

        // Process each 0xFF position
        while mask != 0 {
            let bit_pos = mask.trailing_zeros() as usize;
            let ff_offset = i + bit_pos;

            // Check the byte after 0xFF
            if ff_offset + 1 < len {
                let next = data[ff_offset + 1];
                if (0xD0..=0xD7).contains(&next) {
                    markers.push(RstMarkerPos {
                        offset: ff_offset,
                        rst_num: next - 0xD0,
                    });
                } else if next != 0x00 && next != 0xFF {
                    // Non-RST, non-stuffing marker — end of entropy data
                    return RstScanResult {
                        markers,
                        entropy_end: ff_offset,
                    };
                }
            }

            // Clear the lowest set bit
            mask &= mask - 1;
        }

        i += chunk_size;
    }

    // Handle remaining bytes with scalar
    while i + 1 < len {
        if data[i] == 0xFF {
            let next = data[i + 1];
            if (0xD0..=0xD7).contains(&next) {
                markers.push(RstMarkerPos {
                    offset: i,
                    rst_num: next - 0xD0,
                });
                i += 2;
                continue;
            }
            if next == 0x00 || next == 0xFF {
                i += 2;
                continue;
            }
            // Other marker — stop
            break;
        }
        i += 1;
    }

    RstScanResult {
        markers,
        entropy_end: i,
    }
}

/// Given RST marker positions, compute segment byte ranges for parallel decode.
///
/// Each segment is a byte range `[start, end)` within the scan data.
/// The first segment starts at offset 0.
/// Each subsequent segment starts at the byte after the RST marker (offset + 2).
/// The last segment ends at `scan_data_len`.
///
/// Returns `(segment_starts, segment_ends)` where both have `markers.len() + 1` entries.
pub(crate) fn compute_segments(
    markers: &[RstMarkerPos],
    scan_data_len: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = markers.len() + 1;
    let mut starts = Vec::with_capacity(n);
    let mut ends = Vec::with_capacity(n);

    // First segment: from start of scan data to first RST marker
    starts.push(0);

    for m in markers {
        // Current segment ends at the 0xFF of the RST marker
        ends.push(m.offset);
        // Next segment starts after the 2-byte RST marker
        starts.push(m.offset + 2);
    }

    // Last segment ends at end of scan data
    ends.push(scan_data_len);

    (starts, ends)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_markers() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let result = scan_rst_markers(&data, 0);
        assert!(result.markers.is_empty());
        // Scanner needs i+1 < len, so stops at i=3 (can't check 2-byte marker)
        assert_eq!(result.entropy_end, 3);
    }

    #[test]
    fn test_single_rst0() {
        let data = [0x12, 0x34, 0xFF, 0xD0, 0x56, 0x78];
        let result = scan_rst_markers(&data, 0);
        assert_eq!(result.markers.len(), 1);
        assert_eq!(result.markers[0].offset, 2);
        assert_eq!(result.markers[0].rst_num, 0);
        // After RST at offset 2, scans 0x56, 0x78 — stops at i=5 (i+1=6 not < 6)
        assert_eq!(result.entropy_end, 5);
    }

    #[test]
    fn test_multiple_rst_markers() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0x11, 0x22, 0x33]);
        data.extend_from_slice(&[0xFF, 0xD0]);
        data.extend_from_slice(&[0x44, 0x55]);
        data.extend_from_slice(&[0xFF, 0xD1]);
        data.extend_from_slice(&[0x66]);

        let result = scan_rst_markers(&data, 0);
        assert_eq!(result.markers.len(), 2);
        assert_eq!(result.markers[0].offset, 3);
        assert_eq!(result.markers[0].rst_num, 0);
        assert_eq!(result.markers[1].offset, 7);
        assert_eq!(result.markers[1].rst_num, 1);
    }

    #[test]
    fn test_stuffed_bytes_ignored() {
        let data = [0xFF, 0x00, 0x12, 0xFF, 0xD0, 0x34];
        let result = scan_rst_markers(&data, 0);
        assert_eq!(result.markers.len(), 1);
        assert_eq!(result.markers[0].offset, 3);
        assert_eq!(result.markers[0].rst_num, 0);
    }

    #[test]
    fn test_stops_at_non_rst_marker() {
        let data = [0x11, 0xFF, 0xD0, 0x22, 0xFF, 0xD9];
        let result = scan_rst_markers(&data, 0);
        assert_eq!(result.markers.len(), 1);
        assert_eq!(result.markers[0].offset, 1);
        // entropy_end should be at the 0xFF of EOI marker
        assert_eq!(result.entropy_end, 4);
    }

    #[test]
    fn test_compute_segments() {
        let markers = vec![
            RstMarkerPos {
                offset: 100,
                rst_num: 0,
            },
            RstMarkerPos {
                offset: 250,
                rst_num: 1,
            },
        ];
        let (starts, ends) = compute_segments(&markers, 400);
        assert_eq!(starts, vec![0, 102, 252]);
        assert_eq!(ends, vec![100, 250, 400]);
    }

    #[test]
    fn test_rst_cycling() {
        let mut data = Vec::new();
        for i in 0..10u8 {
            data.push(0x11);
            data.extend_from_slice(&[0xFF, 0xD0 + (i & 7)]);
        }
        let result = scan_rst_markers(&data, 0);
        assert_eq!(result.markers.len(), 10);
        for (i, m) in result.markers.iter().enumerate() {
            assert_eq!(m.rst_num, (i as u8) & 7);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar() {
        // Build data with markers at various positions including crossing
        // 32-byte boundaries
        let mut data = vec![0u8; 200];
        data[10] = 0xFF;
        data[11] = 0xD0;
        data[31] = 0xFF;
        data[32] = 0xD1;
        data[63] = 0xFF;
        data[64] = 0xD2;
        data[100] = 0xFF;
        data[101] = 0x00; // stuffed byte
        data[150] = 0xFF;
        data[151] = 0xD3;

        let scalar = scan_rst_markers_scalar(&data, 0);
        let simd = scan_rst_markers(&data, 0);

        assert_eq!(scalar.markers.len(), simd.markers.len());
        assert_eq!(scalar.entropy_end, simd.entropy_end);
        for (s, v) in scalar.markers.iter().zip(simd.markers.iter()) {
            assert_eq!(s.offset, v.offset);
            assert_eq!(s.rst_num, v.rst_num);
        }
    }
}
