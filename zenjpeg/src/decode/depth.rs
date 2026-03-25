//! Depth map extraction from JPEG files.
//!
//! Supports two depth map embedding formats found in smartphone photos:
//!
//! - **iPhone (MPF Disparity)**: Depth stored as a secondary JPEG image in
//!   the MPF (Multi-Picture Format) directory with type code 0x04 (Disparity).
//!   The [`PreserveConfig`] must have `mpf_depth` enabled for this to be extracted.
//!
//! - **Android GDepth (XMP)**: Depth stored as base64-encoded image data in
//!   the `GDepth` XMP namespace. May use Extended XMP if the data exceeds 64KB.
//!   The XMP string must be preserved (`PreserveConfig.xmp = true`).
//!
//! - **Android Dynamic Depth Format (DDF)**: A newer Google format where depth
//!   images are appended after the primary JPEG, with a container directory in
//!   XMP describing item offsets and sizes.
//!
//! # Usage
//!
//! ```ignore
//! use zenjpeg::decode::{Decoder, PreserveConfig};
//!
//! let config = Decoder::new()
//!     .preserve(PreserveConfig::all());
//!
//! let result = config.decode(&jpeg_data, enough::Unstoppable)?;
//! let extras = result.extras().unwrap();
//!
//! if let Some(depth) = extras.extract_depth_map(Some(&jpeg_data)) {
//!     println!("Depth source: {:?}", depth.source);
//!     println!("Image bytes: {}", depth.data.len());
//! }
//! ```

use alloc::string::String;
use alloc::vec::Vec;

use super::extras::{DecodedExtras, MpfImageTypeExt};

// ============================================================================
// Public types
// ============================================================================

/// Unified depth map data from any extraction source.
#[derive(Clone, Debug)]
pub struct DepthMapData {
    /// How the depth map was found.
    pub source: DepthSource,
    /// Raw image bytes (JPEG or PNG) of the depth map.
    pub data: Vec<u8>,
    /// MIME type of the depth image data.
    pub mime: String,
    /// Optional GDepth metadata (only present for GDepth/DDF sources).
    pub metadata: Option<GDepthMetadata>,
    /// Optional confidence map bytes.
    pub confidence: Option<Vec<u8>>,
    /// MIME type of the confidence map (if present).
    pub confidence_mime: Option<String>,
}

/// Where the depth map was extracted from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthSource {
    /// iPhone MPF secondary image with Disparity type code (0x04).
    MpfDisparity,
    /// Android GDepth XMP namespace (standard or extended XMP).
    GDepthXmp,
    /// Android Dynamic Depth Format (DDF) container directory.
    DynamicDepth,
}

/// Metadata parsed from GDepth/DDF XMP.
#[derive(Clone, Debug)]
pub struct GDepthMetadata {
    /// Depth map format.
    pub format: GDepthFormat,
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance.
    pub far: f32,
    /// Distance units.
    pub units: GDepthUnits,
    /// Measurement type.
    pub measure_type: GDepthMeasureType,
}

/// Depth map value encoding format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GDepthFormat {
    /// Linear depth: pixel_value = (depth - near) / (far - near).
    #[default]
    RangeLinear,
    /// Inverse depth: pixel_value = (1/depth - 1/far) / (1/near - 1/far).
    RangeInverse,
}

/// Units for near/far plane distances.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GDepthUnits {
    /// Meters (default per Google spec).
    #[default]
    Meters,
    /// Diopters (1/meters).
    Diopters,
}

/// How depth is measured from the camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GDepthMeasureType {
    /// Depth along the optical axis (perpendicular to sensor plane).
    #[default]
    OpticalAxis,
    /// Depth along the optic ray (Euclidean distance from camera center).
    OpticRay,
}

// ============================================================================
// GDepth XMP parsing
// ============================================================================

/// Parse GDepth metadata and data from an XMP string.
///
/// Handles the Google Depth Map XMP namespace (`http://ns.google.com/photos/1.0/depthmap/`).
/// The depth image data is base64-encoded within the XMP.
///
/// Returns `None` if no GDepth namespace is found.
pub(crate) fn parse_gdepth_xmp(xmp: &str) -> Option<DepthMapData> {
    // Check for GDepth namespace
    if !xmp.contains("GDepth:") && !xmp.contains("gdepth:") {
        return None;
    }

    // Extract metadata fields
    let format = extract_xmp_attr(xmp, "GDepth:Format")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:Format"))
        .and_then(|v| match v.as_str() {
            "RangeLinear" => Some(GDepthFormat::RangeLinear),
            "RangeInverse" => Some(GDepthFormat::RangeInverse),
            _ => None,
        })
        .unwrap_or_default();

    let near = extract_xmp_attr(xmp, "GDepth:Near")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:Near"))
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);

    let far = extract_xmp_attr(xmp, "GDepth:Far")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:Far"))
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);

    let mime = extract_xmp_attr(xmp, "GDepth:Mime")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:Mime"))
        .unwrap_or_else(|| String::from("image/jpeg"));

    let units = extract_xmp_attr(xmp, "GDepth:Units")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:Units"))
        .and_then(|v| match v.as_str() {
            "Meters" | "meters" => Some(GDepthUnits::Meters),
            "Diopters" | "diopters" => Some(GDepthUnits::Diopters),
            _ => None,
        })
        .unwrap_or_default();

    let measure_type = extract_xmp_attr(xmp, "GDepth:MeasureType")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:MeasureType"))
        .and_then(|v| match v.as_str() {
            "OpticalAxis" => Some(GDepthMeasureType::OpticalAxis),
            "OpticRay" => Some(GDepthMeasureType::OpticRay),
            _ => None,
        })
        .unwrap_or_default();

    // Extract and decode base64 depth data
    let data_b64 = extract_xmp_attr(xmp, "GDepth:Data")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:Data"))
        .or_else(|| extract_xmp_element(xmp, "GDepth:Data"))
        .or_else(|| extract_xmp_element(xmp, "gdepth:Data"))?;

    let data = base64_decode(&data_b64)?;

    // Extract optional confidence map
    let confidence_b64 = extract_xmp_attr(xmp, "GDepth:Confidence")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:Confidence"))
        .or_else(|| extract_xmp_element(xmp, "GDepth:Confidence"))
        .or_else(|| extract_xmp_element(xmp, "gdepth:Confidence"));
    let confidence = confidence_b64.and_then(|b64| base64_decode(&b64));

    let confidence_mime = extract_xmp_attr(xmp, "GDepth:ConfidenceMime")
        .or_else(|| extract_xmp_attr(xmp, "gdepth:ConfidenceMime"));

    Some(DepthMapData {
        source: DepthSource::GDepthXmp,
        data,
        mime,
        metadata: Some(GDepthMetadata {
            format,
            near,
            far,
            units,
            measure_type,
        }),
        confidence,
        confidence_mime,
    })
}

// ============================================================================
// Dynamic Depth Format (DDF) parsing
// ============================================================================

/// Parse Dynamic Depth Format container directory from XMP and extract depth.
///
/// DDF stores metadata in the `http://ns.google.com/photos/dd/1.0/` namespace.
/// Images are appended after the primary JPEG, with lengths and types in XMP.
///
/// `file_data` is the complete JPEG file (including appended images).
pub(crate) fn parse_ddf(xmp: &str, file_data: &[u8]) -> Option<DepthMapData> {
    use ultrahdr_core::metadata::container::{ItemSemantic, parse_container_items};

    // Check for Dynamic Depth namespace
    if !xmp.contains("Container:Directory") && !xmp.contains("http://ns.google.com/photos/dd/1.0/")
    {
        return None;
    }

    // Parse container items from XMP (using shared ultrahdr-core parser)
    let items = parse_container_items(xmp);
    if items.is_empty() {
        return None;
    }

    // Parse DepthMap metadata from XMP (separate from container directory)
    let format = extract_xmp_attr(xmp, "GDepth:Format")
        .or_else(|| extract_xmp_attr(xmp, "DepthMap:Format"))
        .and_then(|v| match v.as_str() {
            "RangeLinear" => Some(GDepthFormat::RangeLinear),
            "RangeInverse" => Some(GDepthFormat::RangeInverse),
            _ => None,
        })
        .unwrap_or_default();

    let near = extract_xmp_attr(xmp, "GDepth:Near")
        .or_else(|| extract_xmp_attr(xmp, "DepthMap:Near"))
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);

    let far = extract_xmp_attr(xmp, "GDepth:Far")
        .or_else(|| extract_xmp_attr(xmp, "DepthMap:Far"))
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);

    let units = extract_xmp_attr(xmp, "GDepth:Units")
        .or_else(|| extract_xmp_attr(xmp, "DepthMap:Units"))
        .and_then(|v| match v.as_str() {
            "Meters" | "meters" => Some(GDepthUnits::Meters),
            "Diopters" | "diopters" => Some(GDepthUnits::Diopters),
            _ => None,
        })
        .unwrap_or_default();

    let measure_type = extract_xmp_attr(xmp, "GDepth:MeasureType")
        .or_else(|| extract_xmp_attr(xmp, "DepthMap:MeasureType"))
        .and_then(|v| match v.as_str() {
            "OpticalAxis" => Some(GDepthMeasureType::OpticalAxis),
            "OpticRay" => Some(GDepthMeasureType::OpticRay),
            _ => None,
        })
        .unwrap_or_default();

    // Find the primary JPEG's end (EOI marker)
    let primary_end = find_primary_eoi(file_data)?;

    // Walk the container directory, calculating offsets for appended items.
    // The first item is always the primary JPEG (already decoded).
    // Subsequent items are appended sequentially after the primary's EOI.
    let mut offset = primary_end;
    let mut depth_data: Option<Vec<u8>> = None;
    let mut depth_mime = String::from("image/jpeg");
    let mut confidence_data: Option<Vec<u8>> = None;
    let mut confidence_mime: Option<String> = None;

    for (i, item) in items.iter().enumerate() {
        if i == 0 {
            // First item is the primary JPEG, skip it
            continue;
        }

        let length = item.length.unwrap_or(0);
        let end = offset.saturating_add(length);
        if end > file_data.len() {
            break;
        }

        let is_depth = matches!(item.semantic, ItemSemantic::DepthMap);
        let is_confidence = matches!(item.semantic, ItemSemantic::ConfidenceMap);

        if is_depth && depth_data.is_none() {
            depth_data = Some(file_data[offset..end].to_vec());
            depth_mime = item.mime.clone();
        } else if is_confidence && confidence_data.is_none() {
            confidence_data = Some(file_data[offset..end].to_vec());
            confidence_mime = Some(item.mime.clone());
        }

        offset = end;
    }

    let data = depth_data?;

    Some(DepthMapData {
        source: DepthSource::DynamicDepth,
        data,
        mime: depth_mime,
        metadata: Some(GDepthMetadata {
            format,
            near,
            far,
            units,
            measure_type,
        }),
        confidence: confidence_data,
        confidence_mime,
    })
}

// ============================================================================
// XMP attribute extraction helpers
// ============================================================================

/// Extract an XMP attribute value: `name="value"`.
fn extract_xmp_attr(xmp: &str, name: &str) -> Option<String> {
    extract_xml_attr(xmp, name)
}

/// Extract XML attribute value: `name="value"`.
fn extract_xml_attr(xml: &str, name: &str) -> Option<String> {
    // Pattern: name="value"
    let pattern = alloc::format!("{}=\"", name);
    if let Some(start) = xml.find(&pattern) {
        let value_start = start + pattern.len();
        let remaining = &xml[value_start..];
        if let Some(end) = remaining.find('"') {
            return Some(remaining[..end].to_string());
        }
    }

    // Also try single quotes: name='value'
    let pattern_sq = alloc::format!("{}='", name);
    if let Some(start) = xml.find(&pattern_sq) {
        let value_start = start + pattern_sq.len();
        let remaining = &xml[value_start..];
        if let Some(end) = remaining.find('\'') {
            return Some(remaining[..end].to_string());
        }
    }

    None
}

/// Extract XMP element content: `<name>content</name>`.
fn extract_xmp_element(xmp: &str, name: &str) -> Option<String> {
    let open_tag = alloc::format!("<{}>", name);
    let close_tag = alloc::format!("</{}>", name);

    let start = xmp.find(&open_tag)?;
    let content_start = start + open_tag.len();
    let remaining = &xmp[content_start..];
    let end = remaining.find(&close_tag)?;
    Some(remaining[..end].to_string())
}

// ============================================================================
// Base64 decoder (minimal, no external dependency)
// ============================================================================

/// Decode a base64 string, ignoring whitespace and newlines.
///
/// Supports both standard and URL-safe base64 alphabets, with
/// optional padding. Returns `None` on invalid input.
pub(crate) fn base64_decode(input: &str) -> Option<Vec<u8>> {
    // Strip whitespace
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();

    if clean.is_empty() {
        return None;
    }

    let mut output = Vec::with_capacity(clean.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &byte in &clean {
        let val = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62, // + (standard) or - (URL-safe)
            b'/' | b'_' => 63, // / (standard) or _ (URL-safe)
            b'=' => continue,  // Padding
            _ => return None,  // Invalid character
        };

        buf = (buf << 6) | val as u32;
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Some(output)
}

/// Encode bytes as standard base64.
#[cfg(test)]
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut output = String::with_capacity((input.len() + 2) / 3 * 4);

    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        output.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        output.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            output.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
        if chunk.len() > 2 {
            output.push(ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
    }

    output
}

// ============================================================================
// JPEG boundary scanner (shared with gainmap)
// ============================================================================

/// Find the end position of the primary JPEG (position after EOI marker).
///
/// Properly skips entropy-coded segments by parsing marker structure.
fn find_primary_eoi(data: &[u8]) -> Option<usize> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
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
                            pos += 2;
                        } else if next == 0xFF {
                            pos += 1;
                        } else if (0xD0..=0xD7).contains(&next) {
                            pos += 2;
                        } else {
                            break;
                        }
                    } else {
                        pos += 1;
                    }
                }
            }
            m if (0xD0..=0xD7).contains(&m) => {
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

// ============================================================================
// Integration with DecodedExtras
// ============================================================================

impl DecodedExtras {
    /// Extract a depth map from any available source.
    ///
    /// Tries sources in priority order:
    /// 1. GDepth XMP (most metadata-rich)
    /// 2. Dynamic Depth Format (DDF)
    /// 3. MPF Disparity secondary image (least metadata)
    ///
    /// For MPF extraction, `file_data` is not needed (data is already preserved).
    /// For GDepth/DDF, `file_data` is needed for DDF (appended images by offset).
    ///
    /// Pass the original JPEG bytes as `file_data` for DDF support.
    /// Pass `None` if you only need MPF and GDepth (inline base64) sources.
    #[must_use]
    pub fn extract_depth_map(&self, file_data: Option<&[u8]>) -> Option<DepthMapData> {
        // 1. Try GDepth XMP (base64-encoded depth in XMP, most metadata)
        if let Some(xmp) = self.xmp() {
            if let Some(depth) = parse_gdepth_xmp(xmp) {
                return Some(depth);
            }

            // 2. Try Dynamic Depth Format (DDF)
            if let Some(data) = file_data
                && let Some(depth) = parse_ddf(xmp, data)
            {
                return Some(depth);
            }
        }

        // 3. Fall back to MPF Disparity secondary image
        self.secondary_images
            .iter()
            .find(|img| img.image_type.is_depth())
            .map(|img| DepthMapData {
                source: DepthSource::MpfDisparity,
                data: img.data.clone(),
                mime: String::from("image/jpeg"),
                metadata: None,
                confidence: None,
                confidence_mime: None,
            })
    }

    /// Check if any depth map data is available without extracting it.
    ///
    /// This is cheaper than `extract_depth_map()` since it skips base64 decoding.
    #[must_use]
    pub fn has_depth_map(&self) -> bool {
        // Check MPF secondary images
        if self
            .secondary_images
            .iter()
            .any(|img| img.image_type.is_depth())
        {
            return true;
        }

        // Check XMP for GDepth namespace
        if let Some(xmp) = self.xmp() {
            if xmp.contains("GDepth:") || xmp.contains("gdepth:") {
                return true;
            }
            if xmp.contains("Container:Directory") && xmp.contains("Depth") {
                return true;
            }
        }

        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === Base64 roundtrip ===

    #[test]
    fn base64_roundtrip_empty() {
        assert_eq!(base64_decode(""), None);
    }

    #[test]
    fn base64_roundtrip_simple() {
        let original = b"Hello, World!";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64_roundtrip_binary() {
        let original: Vec<u8> = (0..=255).collect();
        let encoded = base64_encode(&original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64_with_whitespace() {
        let encoded = "SGVs\nbG8s\nIFdv\ncmxk\nIQ==";
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(decoded, b"Hello, World!");
    }

    #[test]
    fn base64_url_safe() {
        // URL-safe uses - instead of + and _ instead of /
        let standard = base64_encode(&[0xFB, 0xEF, 0xBE]);
        let url_safe = standard.replace('+', "-").replace('/', "_");
        let decoded = base64_decode(&url_safe).unwrap();
        assert_eq!(decoded, &[0xFB, 0xEF, 0xBE]);
    }

    #[test]
    fn base64_invalid_char() {
        assert!(base64_decode("SGVsbG8#").is_none());
    }

    #[test]
    fn base64_no_padding() {
        // Some encoders omit padding
        let original = b"Hi";
        let encoded = base64_encode(original);
        let no_pad = encoded.trim_end_matches('=');
        let decoded = base64_decode(no_pad).unwrap();
        assert_eq!(decoded, original);
    }

    // === XMP attribute extraction ===

    #[test]
    fn extract_attr_double_quotes() {
        let xmp = r#"<rdf:Description GDepth:Format="RangeLinear"/>"#;
        assert_eq!(
            extract_xmp_attr(xmp, "GDepth:Format"),
            Some("RangeLinear".to_string())
        );
    }

    #[test]
    fn extract_attr_single_quotes() {
        let xmp = "<rdf:Description GDepth:Near='0.5'/>";
        assert_eq!(
            extract_xmp_attr(xmp, "GDepth:Near"),
            Some("0.5".to_string())
        );
    }

    #[test]
    fn extract_attr_missing() {
        let xmp = r#"<rdf:Description GDepth:Format="RangeLinear"/>"#;
        assert_eq!(extract_xmp_attr(xmp, "GDepth:Near"), None);
    }

    #[test]
    fn extract_element_content() {
        let xmp = "<GDepth:Data>SGVsbG8=</GDepth:Data>";
        assert_eq!(
            extract_xmp_element(xmp, "GDepth:Data"),
            Some("SGVsbG8=".to_string())
        );
    }

    // === GDepth XMP parsing ===

    #[test]
    fn parse_gdepth_full_metadata() {
        let depth_jpeg = vec![0xFF, 0xD8, 0xFF, 0xD9]; // Minimal JPEG
        let b64_data = base64_encode(&depth_jpeg);
        let xmp = alloc::format!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF>
    <rdf:Description
      GDepth:Format="RangeInverse"
      GDepth:Near="0.15"
      GDepth:Far="100.0"
      GDepth:Mime="image/jpeg"
      GDepth:Units="Meters"
      GDepth:MeasureType="OpticRay"
      GDepth:Data="{b64_data}"/>
  </rdf:RDF>
</x:xmpmeta>"#
        );

        let result = parse_gdepth_xmp(&xmp).expect("should parse GDepth");
        assert_eq!(result.source, DepthSource::GDepthXmp);
        assert_eq!(result.mime, "image/jpeg");
        assert_eq!(result.data, depth_jpeg);

        let meta = result.metadata.unwrap();
        assert_eq!(meta.format, GDepthFormat::RangeInverse);
        assert!((meta.near - 0.15).abs() < 0.001);
        assert!((meta.far - 100.0).abs() < 0.001);
        assert_eq!(meta.units, GDepthUnits::Meters);
        assert_eq!(meta.measure_type, GDepthMeasureType::OpticRay);
    }

    #[test]
    fn parse_gdepth_minimal() {
        let depth_png = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic
        let b64_data = base64_encode(&depth_png);
        let xmp = alloc::format!(
            r#"<rdf:Description
      GDepth:Format="RangeLinear"
      GDepth:Mime="image/png"
      GDepth:Data="{b64_data}"/>"#
        );

        let result = parse_gdepth_xmp(&xmp).expect("should parse GDepth");
        assert_eq!(result.data, depth_png);
        assert_eq!(result.mime, "image/png");

        let meta = result.metadata.unwrap();
        assert_eq!(meta.format, GDepthFormat::RangeLinear);
        // Defaults when not specified
        assert_eq!(meta.units, GDepthUnits::Meters);
        assert_eq!(meta.measure_type, GDepthMeasureType::OpticalAxis);
    }

    #[test]
    fn parse_gdepth_lowercase_prefix() {
        let data = vec![1, 2, 3, 4];
        let b64 = base64_encode(&data);
        let xmp =
            alloc::format!(r#"<rdf:Description gdepth:Format="RangeLinear" gdepth:Data="{b64}"/>"#);

        let result = parse_gdepth_xmp(&xmp).expect("should parse lowercase gdepth");
        assert_eq!(result.data, data);
    }

    #[test]
    fn parse_gdepth_element_data() {
        let data = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let b64 = base64_encode(&data);
        let xmp = alloc::format!(
            r#"<rdf:Description GDepth:Format="RangeLinear">
  <GDepth:Data>{b64}</GDepth:Data>
</rdf:Description>"#
        );

        let result = parse_gdepth_xmp(&xmp).expect("should parse element data");
        assert_eq!(result.data, data);
    }

    #[test]
    fn parse_gdepth_with_confidence() {
        let depth_data = vec![1, 2, 3];
        let conf_data = vec![4, 5, 6];
        let depth_b64 = base64_encode(&depth_data);
        let conf_b64 = base64_encode(&conf_data);
        let xmp = alloc::format!(
            r#"<rdf:Description
      GDepth:Format="RangeLinear"
      GDepth:Data="{depth_b64}"
      GDepth:Confidence="{conf_b64}"
      GDepth:ConfidenceMime="image/png"/>"#
        );

        let result = parse_gdepth_xmp(&xmp).expect("should parse with confidence");
        assert_eq!(result.data, depth_data);
        assert_eq!(result.confidence.unwrap(), conf_data);
        assert_eq!(result.confidence_mime.unwrap(), "image/png");
    }

    #[test]
    fn parse_gdepth_no_namespace() {
        let xmp = r#"<rdf:Description hdrgm:Version="1.0"/>"#;
        assert!(parse_gdepth_xmp(xmp).is_none());
    }

    #[test]
    fn parse_gdepth_missing_data() {
        let xmp = r#"<rdf:Description GDepth:Format="RangeLinear" GDepth:Near="0.5"/>"#;
        // No GDepth:Data attribute
        assert!(parse_gdepth_xmp(xmp).is_none());
    }

    #[test]
    fn parse_gdepth_invalid_base64() {
        let xmp =
            r#"<rdf:Description GDepth:Format="RangeLinear" GDepth:Data="not{valid}base64!"/>"#;
        assert!(parse_gdepth_xmp(xmp).is_none());
    }

    // === DDF parsing ===

    #[test]
    fn parse_ddf_container_directory() {
        use ultrahdr_core::metadata::container::{ItemSemantic, parse_container_items};

        let xmp = r#"<x:xmpmeta>
  <rdf:RDF>
    <rdf:Description
      xmlns:Container="http://ns.google.com/photos/dd/1.0/container/"
      xmlns:Item="http://ns.google.com/photos/dd/1.0/item/">
      <Container:Directory>
        <rdf:Seq>
          <rdf:li>
            <Container:Item Item:Mime="image/jpeg" Item:Semantic="Primary" Item:Length="0"/>
          </rdf:li>
          <rdf:li>
            <Container:Item Item:Mime="image/jpeg" Item:Semantic="DepthMap" Item:Length="5000"/>
          </rdf:li>
          <rdf:li>
            <Container:Item Item:Mime="image/png" Item:Semantic="ConfidenceMap" Item:Length="3000"/>
          </rdf:li>
        </rdf:Seq>
      </Container:Directory>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#;

        let items = parse_container_items(xmp);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].semantic, ItemSemantic::Primary);
        assert_eq!(items[0].mime, "image/jpeg");
        assert_eq!(items[1].semantic, ItemSemantic::DepthMap);
        assert_eq!(items[1].length, Some(5000));
        assert_eq!(items[2].semantic, ItemSemantic::ConfidenceMap);
        assert_eq!(items[2].length, Some(3000));
        assert_eq!(items[2].mime, "image/png");
    }

    #[test]
    fn parse_ddf_extracts_depth() {
        // Build a fake DDF file: primary JPEG + depth JPEG + confidence PNG
        let primary_jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9]; // 8 bytes
        let depth_jpeg = vec![0xFF, 0xD8, 0x01, 0x02, 0x03, 0xFF, 0xD9]; // 7 bytes
        let conf_png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D]; // 5 bytes

        let mut file_data = primary_jpeg.clone();
        file_data.extend_from_slice(&depth_jpeg);
        file_data.extend_from_slice(&conf_png);

        let xmp = alloc::format!(
            r#"<x:xmpmeta>
  <rdf:RDF>
    <rdf:Description
      xmlns:Container="http://ns.google.com/photos/dd/1.0/container/"
      xmlns:Item="http://ns.google.com/photos/dd/1.0/item/"
      GDepth:Format="RangeInverse"
      GDepth:Near="0.2"
      GDepth:Far="50.0"
      Container:Directory="true">
      <Container:Directory>
        <rdf:Seq>
          <rdf:li>
            <Container:Item Item:Mime="image/jpeg" Item:Semantic="Primary" Item:Length="0"/>
          </rdf:li>
          <rdf:li>
            <Container:Item Item:Mime="image/jpeg" Item:Semantic="DepthMap" Item:Length="{depth_len}"/>
          </rdf:li>
          <rdf:li>
            <Container:Item Item:Mime="image/png" Item:Semantic="ConfidenceMap" Item:Length="{conf_len}"/>
          </rdf:li>
        </rdf:Seq>
      </Container:Directory>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#,
            depth_len = depth_jpeg.len(),
            conf_len = conf_png.len()
        );

        let result = parse_ddf(&xmp, &file_data).expect("should parse DDF");
        assert_eq!(result.source, DepthSource::DynamicDepth);
        assert_eq!(result.data, depth_jpeg);
        assert_eq!(result.mime, "image/jpeg");

        let meta = result.metadata.unwrap();
        assert_eq!(meta.format, GDepthFormat::RangeInverse);
        assert!((meta.near - 0.2).abs() < 0.001);
        assert!((meta.far - 50.0).abs() < 0.001);

        // Confidence map
        assert_eq!(result.confidence.unwrap(), conf_png);
        assert_eq!(result.confidence_mime.unwrap(), "image/png");
    }

    #[test]
    fn parse_ddf_no_directory() {
        let xmp = r#"<rdf:Description hdrgm:Version="1.0"/>"#;
        assert!(parse_ddf(xmp, &[0xFF, 0xD8, 0xFF, 0xD9]).is_none());
    }

    #[test]
    fn parse_ddf_truncated_file() {
        let primary_jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9];
        // File has no appended data, but directory claims 5000 bytes
        let xmp = r#"<x:xmpmeta>
  <rdf:RDF>
    <rdf:Description
      xmlns:Container="http://ns.google.com/photos/dd/1.0/container/"
      xmlns:Item="http://ns.google.com/photos/dd/1.0/item/"
      Container:Directory="true">
      <Container:Directory>
        <rdf:Seq>
          <rdf:li>
            <Container:Item Item:Mime="image/jpeg" Item:Semantic="Primary" Item:Length="0"/>
          </rdf:li>
          <rdf:li>
            <Container:Item Item:Mime="image/jpeg" Item:Semantic="DepthMap" Item:Length="5000"/>
          </rdf:li>
        </rdf:Seq>
      </Container:Directory>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#;

        // Should return None since depth data is beyond file boundary
        assert!(parse_ddf(xmp, &primary_jpeg).is_none());
    }

    // === MPF type code parsing ===

    #[test]
    fn mpf_type_disparity() {
        use crate::encode::extras::{MpfImageType, MpfImageTypeExt};
        let typ = MpfImageType::from_type_code(0x020002);
        assert_eq!(typ, MpfImageType::Disparity);
        assert!(typ.is_depth());
    }

    #[test]
    fn mpf_type_roundtrip() {
        use crate::encode::extras::{MpfImageType, MpfImageTypeExt};
        let types = [
            MpfImageType::Undefined,
            MpfImageType::LargeThumbnailVga,
            MpfImageType::LargeThumbnailFullHd,
            MpfImageType::Panorama,
            MpfImageType::Disparity,
            MpfImageType::MultiAngle,
            MpfImageType::BaselinePrimary,
        ];
        for typ in types {
            let code = typ.to_type_code();
            let back = MpfImageType::from_type_code(code);
            assert_eq!(
                back, typ,
                "roundtrip failed for {typ:?} (code=0x{code:06X})"
            );
        }
    }

    #[test]
    fn mpf_attr_masking() {
        // The attribute u32 has bits 31-24 for flags, bits 23-0 for type code.
        // The parser uses `attr & 0x00FFFFFF` to extract the type code.
        use crate::encode::extras::MpfImageType;

        // Simulate an MP entry attribute with flags set:
        // Dependent=1 (bit 31), data format=JPEG (bits 26-24 = 0)
        let attr: u32 = 0x80020002; // Dependent + Disparity type
        let type_code = attr & 0x00FFFFFF;
        assert_eq!(
            MpfImageType::from_type_code(type_code),
            MpfImageType::Disparity
        );
    }

    // === DecodedExtras integration ===

    #[test]
    fn extras_has_depth_map_false_when_empty() {
        let extras = DecodedExtras::new();
        assert!(!extras.has_depth_map());
    }

    #[test]
    fn extras_extract_depth_map_mpf() {
        use super::super::extras::PreservedMpfImage;
        use crate::encode::extras::MpfImageType;

        let mut extras = DecodedExtras::new();
        let depth_jpeg = vec![0xFF, 0xD8, 0x42, 0xFF, 0xD9];
        extras.secondary_images.push(PreservedMpfImage {
            mpf_index: 1,
            image_type: MpfImageType::Disparity,
            data: depth_jpeg.clone(),
        });

        assert!(extras.has_depth_map());

        let depth = extras.extract_depth_map(None).unwrap();
        assert_eq!(depth.source, DepthSource::MpfDisparity);
        assert_eq!(depth.data, depth_jpeg);
        assert_eq!(depth.mime, "image/jpeg");
        assert!(depth.metadata.is_none()); // MPF has no GDepth metadata
    }

    #[test]
    fn extras_extract_depth_map_gdepth_priority() {
        // When both GDepth XMP and MPF disparity are present,
        // GDepth should take priority (more metadata).
        use super::super::extras::{PreservedMpfImage, PreservedSegment, SegmentType};
        use crate::encode::extras::MpfImageType;

        let mut extras = DecodedExtras::new();

        // Add MPF depth image
        extras.secondary_images.push(PreservedMpfImage {
            mpf_index: 1,
            image_type: MpfImageType::Disparity,
            data: vec![0xFF, 0xD8, 0x01, 0xFF, 0xD9],
        });

        // Add XMP with GDepth
        let depth_data = vec![0xFF, 0xD8, 0x02, 0xFF, 0xD9];
        let b64 = base64_encode(&depth_data);
        // The XMP segment stored by the parser includes the namespace prefix
        let xmp_ns = b"http://ns.adobe.com/xap/1.0/\0";
        let mut xmp_data = xmp_ns.to_vec();
        let xmp_content = alloc::format!(
            r#"<rdf:Description GDepth:Format="RangeLinear" GDepth:Near="0.5" GDepth:Far="10.0" GDepth:Data="{b64}"/>"#
        );
        xmp_data.extend_from_slice(xmp_content.as_bytes());

        extras.segments.push(PreservedSegment {
            marker: 0xE1,
            data: xmp_data,
            segment_type: SegmentType::Xmp,
        });

        let depth = extras.extract_depth_map(None).unwrap();
        assert_eq!(depth.source, DepthSource::GDepthXmp);
        assert_eq!(depth.data, depth_data);
        assert!(depth.metadata.is_some());
    }

    // === find_primary_eoi ===

    #[test]
    fn find_eoi_simple() {
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9];
        assert_eq!(find_primary_eoi(&jpeg), Some(8));
    }

    #[test]
    fn find_eoi_with_entropy() {
        // JPEG with SOS marker followed by entropy data then EOI
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xDA, 0x00, 0x03, 0x00, // SOS (length=3, 1 byte data spec)
            0xAB, 0xCD, 0xEF, // Entropy data
            0xFF, 0xD9, // EOI
        ];
        assert_eq!(find_primary_eoi(&jpeg), Some(12));
    }

    #[test]
    fn find_eoi_not_jpeg() {
        let data = vec![0x89, 0x50, 0x4E, 0x47]; // PNG
        assert_eq!(find_primary_eoi(&data), None);
    }

    // === GDepth metadata field parsing ===

    #[test]
    fn gdepth_format_range_linear() {
        let data = vec![1, 2, 3];
        let b64 = base64_encode(&data);
        let xmp =
            alloc::format!(r#"<rdf:Description GDepth:Format="RangeLinear" GDepth:Data="{b64}"/>"#);
        let result = parse_gdepth_xmp(&xmp).unwrap();
        assert_eq!(result.metadata.unwrap().format, GDepthFormat::RangeLinear);
    }

    #[test]
    fn gdepth_format_range_inverse() {
        let data = vec![1, 2, 3];
        let b64 = base64_encode(&data);
        let xmp = alloc::format!(
            r#"<rdf:Description GDepth:Format="RangeInverse" GDepth:Data="{b64}"/>"#
        );
        let result = parse_gdepth_xmp(&xmp).unwrap();
        assert_eq!(result.metadata.unwrap().format, GDepthFormat::RangeInverse);
    }

    #[test]
    fn gdepth_units_diopters() {
        let data = vec![1, 2, 3];
        let b64 = base64_encode(&data);
        let xmp = alloc::format!(
            r#"<rdf:Description GDepth:Format="RangeLinear" GDepth:Units="Diopters" GDepth:Data="{b64}"/>"#
        );
        let result = parse_gdepth_xmp(&xmp).unwrap();
        assert_eq!(result.metadata.unwrap().units, GDepthUnits::Diopters);
    }

    #[test]
    fn gdepth_measure_optic_ray() {
        let data = vec![1, 2, 3];
        let b64 = base64_encode(&data);
        let xmp = alloc::format!(
            r#"<rdf:Description GDepth:Format="RangeLinear" GDepth:MeasureType="OpticRay" GDepth:Data="{b64}"/>"#
        );
        let result = parse_gdepth_xmp(&xmp).unwrap();
        assert_eq!(
            result.metadata.unwrap().measure_type,
            GDepthMeasureType::OpticRay
        );
    }

    // === Base64 decode with large data (simulating depth image) ===

    #[test]
    fn base64_roundtrip_large() {
        // Simulate a depth image (64KB+, the size that would need Extended XMP)
        let original: Vec<u8> = (0..70_000).map(|i| (i % 256) as u8).collect();
        let encoded = base64_encode(&original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded.len(), original.len());
        assert_eq!(decoded, original);
    }

    // === Conformance: Google GDepth spec example ===

    #[test]
    fn gdepth_google_spec_example() {
        // Based on Google's GDepth specification example
        let depth_data = vec![0xFF, 0xD8, 0x00, 0xFF, 0xD9]; // Fake JPEG
        let b64 = base64_encode(&depth_data);

        let xmp = alloc::format!(
            r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:GDepth="http://ns.google.com/photos/1.0/depthmap/"
      GDepth:Format="RangeInverse"
      GDepth:Near="0.6117252707481384"
      GDepth:Far="14.117647171020508"
      GDepth:Mime="image/jpeg"
      GDepth:Units="Meters"
      GDepth:MeasureType="OpticalAxis"
      GDepth:Data="{b64}"/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#
        );

        let result = parse_gdepth_xmp(&xmp).unwrap();
        assert_eq!(result.source, DepthSource::GDepthXmp);
        assert_eq!(result.data, depth_data);
        assert_eq!(result.mime, "image/jpeg");

        let meta = result.metadata.unwrap();
        assert_eq!(meta.format, GDepthFormat::RangeInverse);
        assert!((meta.near - 0.6117).abs() < 0.001);
        assert!((meta.far - 14.117).abs() < 0.01);
        assert_eq!(meta.units, GDepthUnits::Meters);
        assert_eq!(meta.measure_type, GDepthMeasureType::OpticalAxis);
    }
}
