//! Gain map preservation for UltraHDR JPEGs.
//!
//! Detects, extracts, and reattaches gain map secondary images through
//! layout transforms (lossless or lossy). The gain map maintains spatial
//! lock with the primary image by applying proportional transforms.

use alloc::vec::Vec;

use crate::encode::extras::{EncoderSegments, MpfImageType, inject_encoder_segments};

/// Check if XMP indicates UltraHDR content (gain map metadata present).
pub(crate) fn is_ultrahdr_xmp(xmp: &str) -> bool {
    xmp.contains("hdrgm:Version") || xmp.contains("hdrgm:GainMapMax")
}

/// Find the secondary JPEG (gain map) in a multi-picture byte stream.
///
/// Scans for the second SOI→EOI boundary after the primary JPEG's EOI.
/// Returns a copy of the gain map JPEG bytes, or `None` if no secondary
/// JPEG is found.
pub(crate) fn find_secondary_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    const SOI: [u8; 2] = [0xFF, 0xD8];
    const EOI: [u8; 2] = [0xFF, 0xD9];

    // Find the primary JPEG's EOI by scanning for marker boundaries.
    let primary_end = find_primary_eoi(data)?;

    // Look for a second SOI after the primary EOI.
    let remaining = &data[primary_end..];
    if remaining.len() < 4 {
        return None;
    }

    // Find SOI in remaining data
    for i in 0..remaining.len().saturating_sub(1) {
        if remaining[i] == SOI[0] && remaining[i + 1] == SOI[1] {
            let gm_start = primary_end + i;

            // Find the corresponding EOI
            for j in (i + 2)..remaining.len().saturating_sub(1) {
                if remaining[j] == EOI[0] && remaining[j + 1] == EOI[1] {
                    let gm_end = primary_end + j + 2;
                    return Some(data[gm_start..gm_end].to_vec());
                }
            }

            // No EOI found — take everything from SOI to end
            return Some(data[gm_start..].to_vec());
        }
    }

    None
}

/// Find the end position of the primary JPEG (position after EOI marker).
///
/// Properly skips entropy-coded segments by parsing marker structure.
fn find_primary_eoi(data: &[u8]) -> Option<usize> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None; // Not a JPEG
    }

    let mut pos = 2; // Skip SOI

    while pos < data.len().saturating_sub(1) {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = data[pos + 1];

        match marker {
            0xD9 => {
                // EOI found
                return Some(pos + 2);
            }
            0x00 => {
                // Byte stuffing, skip
                pos += 2;
            }
            0xFF => {
                // Fill byte
                pos += 1;
            }
            0xDA => {
                // SOS — skip the marker segment, then scan entropy data
                if pos + 4 > data.len() {
                    return None;
                }
                let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + len;

                // Scan entropy-coded data for the next marker
                while pos < data.len().saturating_sub(1) {
                    if data[pos] == 0xFF {
                        let next = data[pos + 1];
                        if next == 0x00 {
                            // Byte stuffing
                            pos += 2;
                        } else if next == 0xFF {
                            // Fill byte
                            pos += 1;
                        } else if (0xD0..=0xD7).contains(&next) {
                            // RST marker (no payload)
                            pos += 2;
                        } else {
                            // Real marker — break out of entropy scan
                            break;
                        }
                    } else {
                        pos += 1;
                    }
                }
            }
            m if (0xD0..=0xD7).contains(&m) => {
                // RST marker (no payload)
                pos += 2;
            }
            _ => {
                // Marker with length field
                if pos + 4 > data.len() {
                    return None;
                }
                let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + len;
            }
        }
    }

    None
}

/// Assemble a primary JPEG + gain map JPEG into an UltraHDR JPEG with MPF.
///
/// Injects an MPF APP2 directory into the primary JPEG and appends the
/// gain map JPEG after the primary's EOI.
pub(crate) fn assemble_ultrahdr(primary: Vec<u8>, gain_map: Vec<u8>) -> Vec<u8> {
    let segments = EncoderSegments::new().add_mpf_image(gain_map, MpfImageType::Undefined);
    inject_encoder_segments(primary, &segments)
}

/// Compute proportional gain map target dimensions.
///
/// Scales the gain map dimensions proportionally to the primary resize.
pub(crate) fn compute_gainmap_target(
    primary_src_w: u32,
    primary_src_h: u32,
    primary_dst_w: u32,
    primary_dst_h: u32,
    gm_src_w: u32,
    gm_src_h: u32,
) -> (u32, u32) {
    let scale_x = primary_dst_w as f64 / primary_src_w as f64;
    let scale_y = primary_dst_h as f64 / primary_src_h as f64;
    let gm_dst_w = (gm_src_w as f64 * scale_x).round().max(1.0) as u32;
    let gm_dst_h = (gm_src_h as f64 * scale_y).round().max(1.0) as u32;
    (gm_dst_w, gm_dst_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_ultrahdr_xmp() {
        assert!(is_ultrahdr_xmp(r#"<rdf:Description hdrgm:Version="1.0"/>"#));
        assert!(is_ultrahdr_xmp(
            r#"<rdf:Description hdrgm:GainMapMax="4.0"/>"#
        ));
        assert!(!is_ultrahdr_xmp(r#"<rdf:Description dc:creator="Test"/>"#));
        assert!(!is_ultrahdr_xmp(""));
    }

    #[test]
    fn find_secondary_in_multi_jpeg() {
        // Build a fake two-JPEG stream:
        // Primary: SOI + minimal JPEG + EOI
        // Secondary: SOI + minimal JPEG + EOI
        let primary = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9];
        let secondary = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9];

        let mut combined = primary.clone();
        combined.extend_from_slice(&secondary);

        let found = find_secondary_jpeg(&combined);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), secondary);
    }

    #[test]
    fn no_secondary_in_single_jpeg() {
        let single = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9];
        assert!(find_secondary_jpeg(&single).is_none());
    }

    #[test]
    fn gainmap_target_proportional() {
        // Primary: 2048→512 (4x downscale)
        // Gain map: 512 should become 128
        let (w, h) = compute_gainmap_target(2048, 2048, 512, 512, 512, 512);
        assert_eq!(w, 128);
        assert_eq!(h, 128);
    }

    #[test]
    fn gainmap_target_asymmetric() {
        // Primary: 1000x500 → 500x250
        // Gain map: 250x125 → 125x63 (rounded)
        let (w, h) = compute_gainmap_target(1000, 500, 500, 250, 250, 125);
        assert_eq!(w, 125);
        assert_eq!(h, 63);
    }

    #[test]
    fn gainmap_target_minimum() {
        // Very small gain map should not go below 1x1
        let (w, h) = compute_gainmap_target(1000, 1000, 1, 1, 2, 2);
        assert!(w >= 1);
        assert!(h >= 1);
    }
}
