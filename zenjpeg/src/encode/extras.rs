//! Encoder extras for segment injection and MPF assembly.
//!
//! This module provides types for injecting metadata segments (EXIF, XMP, ICC, etc.)
//! and assembling MPF (Multi-Picture Format) secondary images during encode.
//!
//! # Segment Ordering
//!
//! JPEG segments are written in this order for maximum compatibility:
//! 1. SOI (Start of Image)
//! 2. APP0 (JFIF) - if present
//! 3. APP1 (EXIF) - orientation needed early for display
//! 4. APP1 (XMP) - may include extended XMP chunks
//! 5. APP2 (ICC) - may be chunked if > 64KB
//! 6. APP2 (MPF) - references secondary images after EOI
//! 7. APP13 (IPTC)
//! 8. APP14 (Adobe)
//! 9. COM (Comments)
//! 10. DQT, SOF, DHT, SOS... (standard JPEG structure)
//! 11. Image data
//! 12. EOI (End of Image)
//! 13. Secondary images (complete JPEGs for MPF)
//!
//! # Usage
//!
//! ## Simple round-trip (preserve all metadata)
//!
//! ```rust,ignore
//! use zenjpeg::decoder::Decoder;
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
//!
//! // Decode with preservation
//! let decoded = Decoder::new().decode(&original)?;
//! let extras = decoded.extras().unwrap();
//!
//! // Re-encode with same metadata
//! let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
//!     .with_segments(extras.to_encoder_segments());
//! ```
//!
//! ## Build segments manually
//!
//! ```rust,ignore
//! use zenjpeg::encoder::{EncoderSegments, MpfImageType};
//!
//! let segments = EncoderSegments::new()
//!     .set_exif(exif_bytes)
//!     .set_xmp(&xmp_string)
//!     .set_icc(icc_profile)
//!     .add_gainmap(gainmap_jpeg);
//! ```

use alloc::string::String;
use alloc::vec::Vec;

/// Type of JPEG APP segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SegmentType {
    /// APP0 JFIF
    Jfif,
    /// APP1 EXIF
    Exif,
    /// APP1 XMP (standard)
    Xmp,
    /// APP1 XMP Extended
    XmpExtended,
    /// APP2 ICC Profile
    Icc,
    /// APP2 MPF (Multi-Picture Format)
    Mpf,
    /// APP13 IPTC/IIM
    Iptc,
    /// APP14 Adobe
    Adobe,
    /// COM (Comment)
    Comment,
    /// Unknown/unrecognized APP marker
    Unknown,
}

/// MPF image type codes (CIPA DC-007).
///
/// Re-exported from `ultrahdr_core` for cross-crate compatibility.
/// These correspond to the MPF Individual Image Attribute type codes.
pub use ultrahdr_core::MpImageType as MpfImageType;

/// Extension methods for [`MpfImageType`] used in zenjpeg.
///
/// Provides `to_type_code()` (compat alias for `type_code()`) and
/// category helpers (`is_gainmap`, `is_thumbnail`, etc.).
pub trait MpfImageTypeExt {
    /// Convert to MPF type code (compatibility alias for `type_code()`).
    fn to_type_code(self) -> u32;

    /// Check if this is a gain map type (Undefined).
    fn is_gainmap(&self) -> bool;

    /// Check if this is a thumbnail type.
    fn is_thumbnail(&self) -> bool;

    /// Check if this is a depth/disparity type.
    fn is_depth(&self) -> bool;

    /// Check if this is a multi-frame type (panorama, multi-angle).
    fn is_multiframe(&self) -> bool;
}

impl MpfImageTypeExt for MpfImageType {
    fn to_type_code(self) -> u32 {
        self.type_code()
    }

    fn is_gainmap(&self) -> bool {
        matches!(self, Self::Undefined)
    }

    fn is_thumbnail(&self) -> bool {
        matches!(self, Self::LargeThumbnailVga | Self::LargeThumbnailFullHd)
    }

    fn is_depth(&self) -> bool {
        matches!(self, Self::Disparity)
    }

    fn is_multiframe(&self) -> bool {
        matches!(self, Self::Panorama | Self::MultiAngle)
    }
}

/// JFIF density units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DensityUnits {
    /// No units - aspect ratio only
    #[default]
    None = 0,
    /// Pixels per inch
    PixelsPerInch = 1,
    /// Pixels per centimeter
    PixelsPerCm = 2,
}

/// JFIF segment info.
#[derive(Clone, Debug, Default)]
pub struct JfifInfo {
    /// Major version (usually 1)
    pub version_major: u8,
    /// Minor version (usually 1 or 2)
    pub version_minor: u8,
    /// Density units
    pub density_units: DensityUnits,
    /// Horizontal density/DPI
    pub x_density: u16,
    /// Vertical density/DPI
    pub y_density: u16,
}

/// Adobe segment color transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum AdobeColorTransform {
    /// Unknown or RGB/CMYK (no transform)
    #[default]
    Unknown = 0,
    /// YCbCr
    YCbCr = 1,
    /// YCCK
    Ycck = 2,
}

/// Adobe APP14 segment info.
#[derive(Clone, Debug, Default)]
pub struct AdobeInfo {
    /// Version (usually 100)
    pub version: u16,
    /// Color transform flag
    pub color_transform: AdobeColorTransform,
}

/// Maximum bytes for standard XMP (before extended XMP is needed).
/// 65535 - 2 (length) - 29 (namespace) - 2 (padding) = 65502
const MAX_STANDARD_XMP_BYTES: usize = 65502;

/// Maximum bytes per ICC chunk.
/// 65535 - 2 (length) - 12 (signature) - 2 (chunk info) = 65519
const MAX_ICC_CHUNK_BYTES: usize = 65519;

/// XMP namespace signature (with null terminator).
const XMP_NAMESPACE: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

/// Extended XMP namespace signature.
const XMP_EXTENDED_NAMESPACE: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";

/// ICC profile signature.
const ICC_SIGNATURE: &[u8] = b"ICC_PROFILE\0";

/// EXIF signature.
const EXIF_SIGNATURE: &[u8] = b"Exif\0\0";

/// JFIF signature.
const JFIF_SIGNATURE: &[u8] = b"JFIF\0";

/// IPTC signature (Photoshop 3.0).
const IPTC_SIGNATURE: &[u8] = b"Photoshop 3.0\0";

/// Adobe signature.
const ADOBE_SIGNATURE: &[u8] = b"Adobe";

/// MPF signature.
const MPF_SIGNATURE: &[u8] = b"MPF\0";

/// A segment to be injected into the encoder output.
#[derive(Clone, Debug)]
pub struct EncoderSegment {
    /// APP marker byte (0xE0-0xEF for APP0-APP15, 0xFE for COM).
    pub marker: u8,
    /// Segment data (without marker or length field).
    pub data: Vec<u8>,
    /// Detected segment type for ordering.
    pub segment_type: SegmentType,
}

/// A secondary image to append after EOI (for MPF).
#[derive(Clone, Debug)]
pub struct MpfImage {
    /// Type of the secondary image.
    pub image_type: MpfImageType,
    /// Complete JPEG data (including SOI/EOI).
    pub data: Vec<u8>,
}

/// Prepared segments for encoder injection.
///
/// This is the bridge type between decoder and encoder for round-trip workflows.
/// Create from `DecodedExtras::to_encoder_segments()` or build manually.
///
/// # Segment Ordering
///
/// Segments are automatically ordered for maximum compatibility when written:
/// JFIF → EXIF → XMP → ICC → IPTC → Adobe → Comments
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::encoder::EncoderSegments;
///
/// let segments = EncoderSegments::new()
///     .set_exif(exif_bytes)
///     .set_xmp("<?xml version=\"1.0\"?>...")
///     .set_icc(srgb_profile);
/// ```
#[derive(Clone, Debug, Default)]
pub struct EncoderSegments {
    /// Ordered segments to inject.
    segments: Vec<EncoderSegment>,
    /// Secondary images to append after EOI.
    mpf_images: Vec<MpfImage>,
}

impl EncoderSegments {
    /// Create empty segments.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // === Segment access ===

    /// Get segment by type (first match).
    #[must_use]
    pub fn get(&self, typ: SegmentType) -> Option<&[u8]> {
        self.segments
            .iter()
            .find(|s| s.segment_type == typ)
            .map(|s| s.data.as_slice())
    }

    /// Get all segments of a type.
    #[must_use]
    pub fn get_all(&self, typ: SegmentType) -> Vec<&[u8]> {
        self.segments
            .iter()
            .filter(|s| s.segment_type == typ)
            .map(|s| s.data.as_slice())
            .collect()
    }

    /// Check if segment type is present.
    #[must_use]
    pub fn has(&self, typ: SegmentType) -> bool {
        self.segments.iter().any(|s| s.segment_type == typ)
    }

    /// Get all segments.
    #[must_use]
    pub fn segments(&self) -> &[EncoderSegment] {
        &self.segments
    }

    /// Get segments of a specific type.
    pub fn segments_of_type(&self, typ: SegmentType) -> impl Iterator<Item = &EncoderSegment> {
        self.segments.iter().filter(move |s| s.segment_type == typ)
    }

    /// Check if there are any segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty() && self.mpf_images.is_empty()
    }

    // === Segment modification ===

    /// Add a segment (appended to list, will be ordered when written).
    #[must_use]
    pub fn add(mut self, marker: u8, data: Vec<u8>, typ: SegmentType) -> Self {
        self.segments.push(EncoderSegment {
            marker,
            data,
            segment_type: typ,
        });
        self
    }

    /// Add a segment (mutable version).
    pub fn add_mut(&mut self, marker: u8, data: Vec<u8>, typ: SegmentType) -> &mut Self {
        self.segments.push(EncoderSegment {
            marker,
            data,
            segment_type: typ,
        });
        self
    }

    /// Add raw segment (type inferred from marker + data).
    #[must_use]
    pub fn add_raw(mut self, marker: u8, data: Vec<u8>) -> Self {
        let typ = detect_segment_type(marker, &data);
        self.segments.push(EncoderSegment {
            marker,
            data,
            segment_type: typ,
        });
        self
    }

    /// Add raw segment (mutable version).
    pub fn add_raw_mut(&mut self, marker: u8, data: Vec<u8>) -> &mut Self {
        let typ = detect_segment_type(marker, &data);
        self.segments.push(EncoderSegment {
            marker,
            data,
            segment_type: typ,
        });
        self
    }

    /// Remove all segments of a type.
    #[must_use]
    pub fn remove(mut self, typ: SegmentType) -> Self {
        self.segments.retain(|s| s.segment_type != typ);
        self
    }

    /// Remove all segments of a type (mutable version).
    pub fn remove_mut(&mut self, typ: SegmentType) -> &mut Self {
        self.segments.retain(|s| s.segment_type != typ);
        self
    }

    /// Remove segments matching predicate.
    #[must_use]
    pub fn remove_where<F: Fn(&EncoderSegment) -> bool>(mut self, f: F) -> Self {
        self.segments.retain(|s| !f(s));
        self
    }

    /// Replace segment of a type (removes existing, adds new).
    #[must_use]
    pub fn replace(self, marker: u8, data: Vec<u8>, typ: SegmentType) -> Self {
        self.remove(typ).add(marker, data, typ)
    }

    // === Typed segment helpers ===

    /// Set/replace EXIF data.
    ///
    /// The data should be raw TIFF bytes (without the "Exif\0\0" prefix,
    /// which is added automatically).
    #[must_use]
    pub fn set_exif(self, data: Vec<u8>) -> Self {
        // Build full EXIF APP1 data with signature
        let mut full_data = Vec::with_capacity(EXIF_SIGNATURE.len() + data.len());
        full_data.extend_from_slice(EXIF_SIGNATURE);
        full_data.extend_from_slice(&data);
        self.replace(0xE1, full_data, SegmentType::Exif)
    }

    /// Set/replace EXIF data (mutable version).
    pub fn set_exif_mut(&mut self, data: Vec<u8>) -> &mut Self {
        self.remove_mut(SegmentType::Exif);
        let mut full_data = Vec::with_capacity(EXIF_SIGNATURE.len() + data.len());
        full_data.extend_from_slice(EXIF_SIGNATURE);
        full_data.extend_from_slice(&data);
        self.add_mut(0xE1, full_data, SegmentType::Exif)
    }

    /// Set/replace XMP string.
    ///
    /// Automatically handles extended XMP if the string is > 65502 bytes.
    #[must_use]
    pub fn set_xmp(mut self, xmp: &str) -> Self {
        // Remove existing XMP segments
        self.segments.retain(|s| {
            s.segment_type != SegmentType::Xmp && s.segment_type != SegmentType::XmpExtended
        });

        let xmp_bytes = xmp.as_bytes();

        if xmp_bytes.len() <= MAX_STANDARD_XMP_BYTES {
            // Standard XMP - fits in one segment
            let mut data = Vec::with_capacity(XMP_NAMESPACE.len() + xmp_bytes.len());
            data.extend_from_slice(XMP_NAMESPACE);
            data.extend_from_slice(xmp_bytes);
            self.segments.push(EncoderSegment {
                marker: 0xE1,
                data,
                segment_type: SegmentType::Xmp,
            });
        } else {
            // Extended XMP - split across segments
            // TODO: Implement extended XMP splitting
            // For now, truncate to standard XMP size
            let truncated = &xmp_bytes[..MAX_STANDARD_XMP_BYTES];
            let mut data = Vec::with_capacity(XMP_NAMESPACE.len() + truncated.len());
            data.extend_from_slice(XMP_NAMESPACE);
            data.extend_from_slice(truncated);
            self.segments.push(EncoderSegment {
                marker: 0xE1,
                data,
                segment_type: SegmentType::Xmp,
            });
        }

        self
    }

    /// Modify XMP in place (no-op if no XMP present).
    #[must_use]
    pub fn modify_xmp<F: FnOnce(&str) -> String>(mut self, f: F) -> Self {
        // Find existing XMP
        if let Some(idx) = self
            .segments
            .iter()
            .position(|s| s.segment_type == SegmentType::Xmp)
        {
            let seg = &self.segments[idx];
            // Extract XMP string (skip namespace prefix)
            if seg.data.len() > XMP_NAMESPACE.len() {
                let xmp_start = XMP_NAMESPACE.len();
                if let Ok(xmp_str) = core::str::from_utf8(&seg.data[xmp_start..]) {
                    let new_xmp = f(xmp_str);
                    // Remove old XMP and set new
                    self.segments.remove(idx);
                    return self.set_xmp(&new_xmp);
                }
            }
        }
        self
    }

    /// Set/replace ICC profile (auto-chunks if > 64KB).
    #[must_use]
    pub fn set_icc(mut self, profile: Vec<u8>) -> Self {
        // Remove existing ICC segments
        self.segments.retain(|s| s.segment_type != SegmentType::Icc);

        if profile.is_empty() {
            return self;
        }

        // Chunk the profile
        let num_chunks = (profile.len() + MAX_ICC_CHUNK_BYTES - 1) / MAX_ICC_CHUNK_BYTES;
        let mut offset = 0;

        for chunk_num in 0..num_chunks {
            let chunk_size = (profile.len() - offset).min(MAX_ICC_CHUNK_BYTES);

            // Build chunk data: signature + chunk_no + num_chunks + data
            let mut data = Vec::with_capacity(ICC_SIGNATURE.len() + 2 + chunk_size);
            data.extend_from_slice(ICC_SIGNATURE);
            data.push((chunk_num + 1) as u8);
            data.push(num_chunks as u8);
            data.extend_from_slice(&profile[offset..offset + chunk_size]);

            self.segments.push(EncoderSegment {
                marker: 0xE2,
                data,
                segment_type: SegmentType::Icc,
            });

            offset += chunk_size;
        }

        self
    }

    /// Remove ICC profile.
    #[must_use]
    pub fn remove_icc(self) -> Self {
        self.remove(SegmentType::Icc)
    }

    /// Set/replace IPTC data.
    #[must_use]
    pub fn set_iptc(self, data: Vec<u8>) -> Self {
        // Build full IPTC APP13 data with signature
        let mut full_data = Vec::with_capacity(IPTC_SIGNATURE.len() + data.len());
        full_data.extend_from_slice(IPTC_SIGNATURE);
        full_data.extend_from_slice(&data);
        self.replace(0xED, full_data, SegmentType::Iptc)
    }

    /// Set JFIF info (density/DPI).
    #[must_use]
    pub fn set_jfif(self, info: JfifInfo) -> Self {
        let mut data = Vec::with_capacity(16);
        data.extend_from_slice(JFIF_SIGNATURE);
        data.push(info.version_major);
        data.push(info.version_minor);
        data.push(info.density_units as u8);
        data.push((info.x_density >> 8) as u8);
        data.push(info.x_density as u8);
        data.push((info.y_density >> 8) as u8);
        data.push(info.y_density as u8);
        data.push(0); // Thumbnail width
        data.push(0); // Thumbnail height
        self.replace(0xE0, data, SegmentType::Jfif)
    }

    /// Set DPI for print workflows.
    ///
    /// Creates a JFIF APP0 segment with the specified DPI. Common values:
    /// - 72: Screen/web (though browsers ignore this)
    /// - 150: Draft print
    /// - 300: Standard print
    /// - 600: High-quality print
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let segments = EncoderSegments::new()
    ///     .set_printer_dpi(300);
    /// ```
    #[must_use]
    pub fn set_printer_dpi(self, dpi: u16) -> Self {
        self.set_jfif(JfifInfo {
            version_major: 1,
            version_minor: 2,
            density_units: DensityUnits::PixelsPerInch,
            x_density: dpi,
            y_density: dpi,
        })
    }

    /// Add comment.
    #[must_use]
    pub fn add_comment(mut self, comment: &str) -> Self {
        self.segments.push(EncoderSegment {
            marker: 0xFE,
            data: comment.as_bytes().to_vec(),
            segment_type: SegmentType::Comment,
        });
        self
    }

    // === MPF secondary images ===

    /// Add secondary image (will be appended after EOI).
    #[must_use]
    pub fn add_mpf_image(mut self, data: Vec<u8>, typ: MpfImageType) -> Self {
        self.mpf_images.push(MpfImage {
            image_type: typ,
            data,
        });
        self
    }

    /// Add secondary image (mutable version).
    pub fn add_mpf_image_mut(&mut self, data: Vec<u8>, typ: MpfImageType) -> &mut Self {
        self.mpf_images.push(MpfImage {
            image_type: typ,
            data,
        });
        self
    }

    /// Add gain map (convenience for MpfImageType::Undefined).
    #[must_use]
    pub fn add_gainmap(self, jpeg_data: Vec<u8>) -> Self {
        self.add_mpf_image(jpeg_data, MpfImageType::Undefined)
    }

    /// Add depth map.
    #[must_use]
    pub fn add_depth_map(self, jpeg_data: Vec<u8>) -> Self {
        self.add_mpf_image(jpeg_data, MpfImageType::Disparity)
    }

    /// Get MPF images.
    #[must_use]
    pub fn mpf_images(&self) -> &[MpfImage] {
        &self.mpf_images
    }

    /// Check if there are MPF images.
    #[must_use]
    pub fn has_mpf_images(&self) -> bool {
        !self.mpf_images.is_empty()
    }

    /// Remove all MPF images.
    #[must_use]
    pub fn clear_mpf_images(mut self) -> Self {
        self.mpf_images.clear();
        self
    }

    /// Remove MPF images by type.
    #[must_use]
    pub fn remove_mpf_images(mut self, typ: MpfImageType) -> Self {
        self.mpf_images.retain(|img| img.image_type != typ);
        self
    }

    // === Bulk operations ===

    /// Merge segments from another EncoderSegments.
    /// Existing segments of same type are kept (use replace for override).
    #[must_use]
    pub fn merge(mut self, other: &EncoderSegments) -> Self {
        for seg in &other.segments {
            self.segments.push(seg.clone());
        }
        for img in &other.mpf_images {
            self.mpf_images.push(img.clone());
        }
        self
    }

    /// Clear all segments (keeps MPF images).
    #[must_use]
    pub fn clear_segments(mut self) -> Self {
        self.segments.clear();
        self
    }

    /// Clear everything.
    #[must_use]
    pub fn clear(mut self) -> Self {
        self.segments.clear();
        self.mpf_images.clear();
        self
    }

    /// Keep only specified segment types.
    #[must_use]
    pub fn retain(mut self, types: &[SegmentType]) -> Self {
        self.segments.retain(|s| types.contains(&s.segment_type));
        self
    }
}

/// Detect segment type from marker and data prefix.
fn detect_segment_type(marker: u8, data: &[u8]) -> SegmentType {
    match marker {
        0xE0 if data.starts_with(JFIF_SIGNATURE) => SegmentType::Jfif,
        0xE1 if data.starts_with(EXIF_SIGNATURE) => SegmentType::Exif,
        0xE1 if data.starts_with(XMP_NAMESPACE) => SegmentType::Xmp,
        0xE1 if data.starts_with(XMP_EXTENDED_NAMESPACE) => SegmentType::XmpExtended,
        0xE2 if data.starts_with(ICC_SIGNATURE) => SegmentType::Icc,
        0xE2 if data.starts_with(MPF_SIGNATURE) => SegmentType::Mpf,
        0xED if data.starts_with(IPTC_SIGNATURE) => SegmentType::Iptc,
        0xEE if data.starts_with(ADOBE_SIGNATURE) => SegmentType::Adobe,
        0xFE => SegmentType::Comment,
        _ => SegmentType::Unknown,
    }
}

/// Write a single segment to output buffer.
pub(crate) fn write_segment(output: &mut Vec<u8>, marker: u8, data: &[u8]) {
    output.push(0xFF);
    output.push(marker);
    let length = (data.len() + 2) as u16;
    output.push((length >> 8) as u8);
    output.push(length as u8);
    output.extend_from_slice(data);
}

/// Write all encoder segments to output buffer (in correct order).
pub(crate) fn write_encoder_segments(output: &mut Vec<u8>, segments: &EncoderSegments) {
    // Write in order: JFIF → EXIF → XMP → ICC → IPTC → Adobe → Comments
    // Note: MPF directory is generated separately when MPF images exist

    for seg in segments.segments_of_type(SegmentType::Jfif) {
        write_segment(output, seg.marker, &seg.data);
    }

    for seg in segments.segments_of_type(SegmentType::Exif) {
        write_segment(output, seg.marker, &seg.data);
    }

    for seg in segments.segments_of_type(SegmentType::Xmp) {
        write_segment(output, seg.marker, &seg.data);
    }

    for seg in segments.segments_of_type(SegmentType::XmpExtended) {
        write_segment(output, seg.marker, &seg.data);
    }

    for seg in segments.segments_of_type(SegmentType::Icc) {
        write_segment(output, seg.marker, &seg.data);
    }

    // MPF directory is NOT written here - it's generated separately
    // when finishing the encode (after we know the primary image size)

    for seg in segments.segments_of_type(SegmentType::Iptc) {
        write_segment(output, seg.marker, &seg.data);
    }

    for seg in segments.segments_of_type(SegmentType::Adobe) {
        write_segment(output, seg.marker, &seg.data);
    }

    for seg in segments.segments_of_type(SegmentType::Comment) {
        write_segment(output, seg.marker, &seg.data);
    }

    // Unknown segments last
    for seg in segments.segments_of_type(SegmentType::Unknown) {
        write_segment(output, seg.marker, &seg.data);
    }
}

/// Generate MPF directory segment for secondary images.
///
/// Returns the MPF APP2 segment data (without marker/length).
///
/// # Arguments
/// * `num_images` - Number of secondary images
/// * `primary_size` - Total size of the primary JPEG in bytes
/// * `image_sizes` - Size and type of each secondary image
/// * `mpf_insert_offset` - File offset where the MPF segment will be inserted.
///   The TIFF header will be at `mpf_insert_offset + 8` (after marker, length, and "MPF\0").
///   Per CIPA DC-007, secondary image offsets must be relative to this TIFF header position.
pub(crate) fn generate_mpf_directory(
    num_images: usize,
    primary_size: u32,
    image_sizes: &[(u32, MpfImageType)],
    mpf_insert_offset: usize,
) -> Vec<u8> {
    // MPF structure:
    // - "MPF\0" signature (4 bytes)
    // - TIFF-like structure with:
    //   - Byte order (2 bytes): "II" for little-endian or "MM" for big-endian
    //   - Fixed value 0x002A (2 bytes)
    //   - Offset to first IFD (4 bytes)
    //   - IFD with MP Entry and related tags

    let total_images = 1 + num_images; // Primary + secondaries

    // We'll use little-endian (Intel) byte order
    let mut data = Vec::with_capacity(128);

    // MPF signature
    data.extend_from_slice(MPF_SIGNATURE);

    // TIFF header: byte order + magic + IFD offset
    data.extend_from_slice(b"II"); // Little-endian
    data.extend_from_slice(&0x002Au16.to_le_bytes()); // TIFF magic
    data.extend_from_slice(&0x00000008u32.to_le_bytes()); // Offset to IFD (immediately after header)

    // IFD starts here (offset 8 from start of TIFF header, 12 from start of data)
    // Number of entries
    let num_entries: u16 = 3; // MPFVersion, NumberOfImages, MPEntry
    data.extend_from_slice(&num_entries.to_le_bytes());

    // Calculate offsets for MP Entry array
    let ifd_size = 2 + 12 * num_entries as usize + 4; // count + entries + next IFD
    let mp_entry_offset = 8 + ifd_size; // offset from TIFF header start

    // Entry 1: MPFVersion (tag 0xB000)
    write_mpf_ifd_entry(&mut data, 0xB000, 7, 4, 0x30303130); // "0100" as u32

    // Entry 2: NumberOfImages (tag 0xB001)
    write_mpf_ifd_entry(&mut data, 0xB001, 4, 1, total_images as u32);

    // Entry 3: MPEntry (tag 0xB002)
    // Each entry is 16 bytes, value is offset to array
    let mp_entry_size = total_images * 16;
    write_mpf_ifd_entry(
        &mut data,
        0xB002,
        7,
        mp_entry_size as u32,
        mp_entry_offset as u32,
    );

    // Next IFD offset (0 = no more)
    data.extend_from_slice(&0u32.to_le_bytes());

    // MP Entry array
    // Primary image entry (offset 0, size = primary_size)
    write_mp_entry(&mut data, 0x030000, primary_size, 0, 0, 0);

    // Secondary image entries
    // Per CIPA DC-007, offsets are relative to the TIFF header position
    // TIFF header is at mpf_insert_offset + 8 (marker + length + "MPF\0")
    let tiff_header_pos = mpf_insert_offset + 8;
    // Use saturating_sub for size estimation calls where primary_size may be 0
    let mut current_offset = (primary_size as usize).saturating_sub(tiff_header_pos) as u32;
    for (size, typ) in image_sizes {
        let type_code = typ.to_type_code();
        write_mp_entry(&mut data, type_code, *size, current_offset, 0, 0);
        current_offset += size;
    }

    data
}

/// Write a single MPF IFD entry (12 bytes).
fn write_mpf_ifd_entry(buf: &mut Vec<u8>, tag: u16, type_: u16, count: u32, value: u32) {
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(&type_.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Write a single MP Entry (16 bytes).
fn write_mp_entry(
    buf: &mut Vec<u8>,
    type_code: u32,
    size: u32,
    offset: u32,
    dep_image1: u16,
    dep_image2: u16,
) {
    // Individual Image Attribute (4 bytes)
    buf.extend_from_slice(&type_code.to_le_bytes());
    // Individual Image Size (4 bytes)
    buf.extend_from_slice(&size.to_le_bytes());
    // Individual Image Data Offset (4 bytes)
    buf.extend_from_slice(&offset.to_le_bytes());
    // Dependent Image 1 Entry Number (2 bytes)
    buf.extend_from_slice(&dep_image1.to_le_bytes());
    // Dependent Image 2 Entry Number (2 bytes)
    buf.extend_from_slice(&dep_image2.to_le_bytes());
}

/// Inject encoder segments into a completed JPEG.
///
/// Inserts segments after SOI in correct order, generates MPF directory if needed,
/// and appends secondary images after EOI.
pub(crate) fn inject_encoder_segments(jpeg: Vec<u8>, segments: &EncoderSegments) -> Vec<u8> {
    if segments.is_empty() {
        return jpeg;
    }

    // Build the segment bytes to insert
    let mut segment_bytes = Vec::new();
    write_encoder_segments(&mut segment_bytes, segments);

    // Handle MPF images if present
    let has_mpf = segments.has_mpf_images();
    let mpf_images = segments.mpf_images();

    if has_mpf {
        // Calculate sizes for MPF directory
        // Primary size = original JPEG + inserted segments + MPF directory
        // We need to know MPF directory size first...

        // Get image sizes for MPF entries
        let image_sizes: Vec<(u32, MpfImageType)> = mpf_images
            .iter()
            .map(|img| (img.data.len() as u32, img.image_type))
            .collect();

        // MPF segment will be inserted after SOI (2 bytes) and segment_bytes
        let mpf_insert_offset = 2 + segment_bytes.len();

        // Generate MPF directory (without knowing exact primary size yet)
        // We'll generate it, then calculate actual size
        let mpf_data_temp =
            generate_mpf_directory(mpf_images.len(), 0, &image_sizes, mpf_insert_offset);
        let mpf_segment_size = 2 + 2 + mpf_data_temp.len(); // marker + length + data

        // Now calculate actual primary size
        let primary_size = jpeg.len() + segment_bytes.len() + mpf_segment_size;

        // Regenerate MPF directory with correct primary size
        let mpf_data = generate_mpf_directory(
            mpf_images.len(),
            primary_size as u32,
            &image_sizes,
            mpf_insert_offset,
        );

        // Build final output
        let total_size = primary_size + mpf_images.iter().map(|i| i.data.len()).sum::<usize>();
        let mut result = Vec::with_capacity(total_size);

        // SOI
        result.extend_from_slice(&jpeg[..2]);
        // Injected segments
        result.extend_from_slice(&segment_bytes);
        // MPF directory segment
        write_segment(&mut result, 0xE2, &mpf_data);
        // Rest of original JPEG (after SOI)
        result.extend_from_slice(&jpeg[2..]);
        // Secondary images after EOI
        for img in mpf_images {
            result.extend_from_slice(&img.data);
        }

        result
    } else {
        // No MPF - simple segment injection
        let mut result = Vec::with_capacity(jpeg.len() + segment_bytes.len());

        // SOI
        result.extend_from_slice(&jpeg[..2]);
        // Injected segments
        result.extend_from_slice(&segment_bytes);
        // Rest of original JPEG (after SOI)
        result.extend_from_slice(&jpeg[2..]);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_segments_new() {
        let segments = EncoderSegments::new();
        assert!(segments.is_empty());
        assert!(segments.segments().is_empty());
        assert!(segments.mpf_images().is_empty());
    }

    #[test]
    fn test_encoder_segments_add() {
        let segments = EncoderSegments::new().add(0xE1, b"test".to_vec(), SegmentType::Unknown);

        assert!(!segments.is_empty());
        assert_eq!(segments.segments().len(), 1);
        assert!(segments.has(SegmentType::Unknown));
    }

    #[test]
    fn test_set_exif() {
        let segments = EncoderSegments::new().set_exif(b"TIFF data".to_vec());

        assert!(segments.has(SegmentType::Exif));
        let exif_data = segments.get(SegmentType::Exif).unwrap();
        assert!(exif_data.starts_with(EXIF_SIGNATURE));
    }

    #[test]
    fn test_set_xmp() {
        let xmp = "<?xml version=\"1.0\"?><x:xmpmeta>test</x:xmpmeta>";
        let segments = EncoderSegments::new().set_xmp(xmp);

        assert!(segments.has(SegmentType::Xmp));
        let xmp_data = segments.get(SegmentType::Xmp).unwrap();
        assert!(xmp_data.starts_with(XMP_NAMESPACE));
    }

    #[test]
    fn test_set_icc_small() {
        let profile = vec![0u8; 1000];
        let segments = EncoderSegments::new().set_icc(profile);

        assert!(segments.has(SegmentType::Icc));
        // Should be a single chunk
        assert_eq!(segments.get_all(SegmentType::Icc).len(), 1);
    }

    #[test]
    fn test_set_icc_large() {
        // Large profile that needs chunking
        let profile = vec![0u8; 100_000];
        let segments = EncoderSegments::new().set_icc(profile);

        assert!(segments.has(SegmentType::Icc));
        // Should be multiple chunks
        let chunks = segments.get_all(SegmentType::Icc);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_add_gainmap() {
        let gainmap = b"fake jpeg data".to_vec();
        let segments = EncoderSegments::new().add_gainmap(gainmap);

        assert!(segments.has_mpf_images());
        assert_eq!(segments.mpf_images().len(), 1);
        assert_eq!(segments.mpf_images()[0].image_type, MpfImageType::Undefined);
    }

    #[test]
    fn test_remove() {
        let segments = EncoderSegments::new()
            .set_exif(b"exif".to_vec())
            .set_xmp("xmp")
            .remove(SegmentType::Exif);

        assert!(!segments.has(SegmentType::Exif));
        assert!(segments.has(SegmentType::Xmp));
    }

    #[test]
    fn test_replace() {
        let segments = EncoderSegments::new()
            .set_exif(b"old exif".to_vec())
            .set_exif(b"new exif".to_vec());

        // Should only have one EXIF segment
        assert_eq!(segments.get_all(SegmentType::Exif).len(), 1);
    }

    #[test]
    fn test_add_comment() {
        let segments = EncoderSegments::new()
            .add_comment("Comment 1")
            .add_comment("Comment 2");

        let comments = segments.get_all(SegmentType::Comment);
        assert_eq!(comments.len(), 2);
    }

    #[test]
    fn test_mpf_directory_generation() {
        let image_sizes = vec![(5000, MpfImageType::Undefined)];
        // MPF insert offset of 100 (simulating after SOI + some segments)
        let mpf_data = generate_mpf_directory(1, 10000, &image_sizes, 100);

        // Should start with MPF signature
        assert!(mpf_data.starts_with(MPF_SIGNATURE));
        // Should contain TIFF header
        assert_eq!(&mpf_data[4..6], b"II"); // Little-endian
    }

    #[test]
    fn test_mpf_secondary_offset_is_relative_to_tiff_header() {
        // Per CIPA DC-007, secondary image offsets must be relative to the TIFF header,
        // not absolute file positions. This was a bug that was fixed.
        //
        // Setup: MPF inserted at position 200, primary image total size 5000
        let mpf_insert_offset = 200;
        let primary_size = 5000u32;
        let secondary_size = 1000u32;
        let image_sizes = vec![(secondary_size, MpfImageType::Undefined)];

        let mpf_data = generate_mpf_directory(1, primary_size, &image_sizes, mpf_insert_offset);

        // Parse the generated MPF to extract the secondary image offset
        // Structure: MPF\0 (4) + II (2) + 0x2A00 (2) + IFD offset (4) = 12 bytes
        // Then IFD: count (2) + 3 entries (36) + next IFD (4) = 42 bytes
        // Then MP entries: primary (16) + secondary (16) = 32 bytes
        // Secondary offset is at bytes 8-11 of the secondary MP entry

        // Find MP entries - they start at offset 54 (12 + 42)
        // Primary entry: bytes 54-69
        // Secondary entry: bytes 70-85
        // Secondary offset is at bytes 78-81 (secondary entry bytes 8-11)
        let secondary_offset_bytes = &mpf_data[78..82];
        let secondary_offset = u32::from_le_bytes(secondary_offset_bytes.try_into().unwrap());

        // TIFF header is at mpf_insert_offset + 8 (marker + length + "MPF\0")
        // The TIFF header position in the file: 200 + 8 = 208
        // Secondary image starts at file position: 5000
        // Expected offset relative to TIFF header: 5000 - 208 = 4792
        let tiff_header_pos = mpf_insert_offset + 8;
        let expected_offset = primary_size as usize - tiff_header_pos;

        assert_eq!(
            secondary_offset, expected_offset as u32,
            "Secondary image offset should be {} (relative to TIFF header at {}), but got {}",
            expected_offset, tiff_header_pos, secondary_offset
        );

        // Also verify it's NOT the absolute position (the old bug)
        assert_ne!(
            secondary_offset, primary_size,
            "Secondary offset should NOT be the absolute file position ({})",
            primary_size
        );
    }

    #[test]
    fn test_segment_type_detection() {
        assert_eq!(detect_segment_type(0xE0, JFIF_SIGNATURE), SegmentType::Jfif);
        assert_eq!(detect_segment_type(0xE1, EXIF_SIGNATURE), SegmentType::Exif);
        assert_eq!(detect_segment_type(0xE1, XMP_NAMESPACE), SegmentType::Xmp);
        assert_eq!(detect_segment_type(0xE2, ICC_SIGNATURE), SegmentType::Icc);
        assert_eq!(detect_segment_type(0xFE, b"comment"), SegmentType::Comment);
        assert_eq!(detect_segment_type(0xE5, b"unknown"), SegmentType::Unknown);
    }

    #[test]
    fn test_set_printer_dpi() {
        let segments = EncoderSegments::new().set_printer_dpi(300);

        assert!(segments.has(SegmentType::Jfif));
        let jfif_data = segments.get(SegmentType::Jfif).unwrap();

        // Verify JFIF structure: "JFIF\0" + version + units + density
        assert!(jfif_data.starts_with(JFIF_SIGNATURE));
        assert_eq!(jfif_data[5], 1); // version major
        assert_eq!(jfif_data[6], 2); // version minor
        assert_eq!(jfif_data[7], 1); // units = pixels per inch
        // x_density = 300 = 0x012C (big-endian)
        assert_eq!(jfif_data[8], 0x01);
        assert_eq!(jfif_data[9], 0x2C);
        // y_density = 300
        assert_eq!(jfif_data[10], 0x01);
        assert_eq!(jfif_data[11], 0x2C);
    }
}
