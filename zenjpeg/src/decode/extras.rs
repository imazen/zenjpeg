//! Preserved metadata and secondary images from JPEG decode.
//!
//! This module provides types for preserving JPEG metadata segments (EXIF, XMP, ICC, etc.)
//! and MPF secondary images (gain maps, depth maps, thumbnails) during decode.
//!
//! # Example
//!
//! ```ignore
//! use zenjpeg::decode::{Decoder, PreserveConfig};
//!
//! // Default config preserves most metadata and gain maps
//! let result = Decoder::new().decode(&jpeg_data)?;
//! let extras = result.extras().expect("extras preserved by default");
//!
//! // Get XMP metadata
//! if let Some(xmp) = extras.xmp() {
//!     println!("XMP: {}", xmp);
//! }
//!
//! // Get gain map for UltraHDR
//! if let Some(gainmap) = extras.gainmap() {
//!     let gm_decoded = Decoder::new().decode(gainmap)?;
//! }
//! ```

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::OnceLock;

// Re-export shared types from encoder extras
pub use crate::encode::extras::{
    AdobeColorTransform, AdobeInfo, DensityUnits, JfifInfo, MpfImageType, MpfImageTypeExt,
    SegmentType,
};

// Re-export detect types for quality estimation through extras
pub use crate::detect::{
    Confidence, DqtTable, EncoderFamily, JpegProbe, QualityEstimate, QualityScale,
};

/// Configuration for what to preserve during decode.
#[derive(Clone)]
pub struct PreserveConfig {
    // === Metadata segments ===
    /// APP0 JFIF - DPI/density for print
    pub jfif: bool,

    /// APP1 EXIF - orientation, camera, GPS, copyright
    pub exif: bool,

    /// APP1 XMP - edit history, copyright, gainmap metadata
    /// Note: Extended XMP (chunked across multiple APP1) is reassembled
    pub xmp: bool,

    /// APP2 ICC - color profile preservation mode
    pub icc: IccPreserve,

    /// APP13 IPTC/IIM - copyright, creator, caption, keywords
    pub iptc: bool,

    /// APP14 Adobe - color transform flag
    pub adobe: bool,

    /// COM - comment markers (sometimes contain copyright)
    pub com: bool,

    /// Unknown APP markers (APP3-12, APP15 excluding known types)
    pub app_unknown: bool,

    // === MPF secondary images ===
    /// Undefined type - used for gain maps (UltraHDR)
    pub mpf_gainmaps: bool,

    /// Large thumbnails (VGA, Full HD)
    pub mpf_thumbnails: bool,

    /// Multi-frame images (panorama, multi-angle)
    pub mpf_multiframe: bool,

    /// Disparity/depth maps
    pub mpf_depth: bool,

    /// Custom filter for MPF images (overrides above if set)
    /// Called with (index, image_type, size_bytes) -> should_keep
    pub mpf_filter: Option<Arc<dyn Fn(usize, MpfImageType, u32) -> bool + Send + Sync>>,
}

impl core::fmt::Debug for PreserveConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreserveConfig")
            .field("jfif", &self.jfif)
            .field("exif", &self.exif)
            .field("xmp", &self.xmp)
            .field("icc", &self.icc)
            .field("iptc", &self.iptc)
            .field("adobe", &self.adobe)
            .field("com", &self.com)
            .field("app_unknown", &self.app_unknown)
            .field("mpf_gainmaps", &self.mpf_gainmaps)
            .field("mpf_thumbnails", &self.mpf_thumbnails)
            .field("mpf_multiframe", &self.mpf_multiframe)
            .field("mpf_depth", &self.mpf_depth)
            .field("mpf_filter", &self.mpf_filter.is_some())
            .finish()
    }
}

/// ICC profile preservation mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IccPreserve {
    /// Keep all ICC profiles
    #[default]
    All,
    /// Drop well-known standard profiles (sRGB IEC61966-2.1, Display P3)
    /// Saves space when profile is implicit
    DropStandard,
    /// Keep no ICC profiles
    None,
}

impl Default for PreserveConfig {
    fn default() -> Self {
        Self {
            // Metadata - keep by default (copyright, rendering)
            jfif: true,
            exif: true,
            xmp: true,
            icc: IccPreserve::All,
            iptc: true,
            adobe: true,
            com: true,
            app_unknown: false,

            // MPF - keep gain maps, drop redundant previews
            mpf_gainmaps: true,
            mpf_thumbnails: false,
            mpf_multiframe: false,
            mpf_depth: false,
            mpf_filter: None,
        }
    }
}

impl PreserveConfig {
    /// Preserve nothing (minimal memory).
    #[must_use]
    pub fn none() -> Self {
        Self {
            jfif: false,
            exif: false,
            xmp: false,
            icc: IccPreserve::None,
            iptc: false,
            adobe: false,
            com: false,
            app_unknown: false,
            mpf_gainmaps: false,
            mpf_thumbnails: false,
            mpf_multiframe: false,
            mpf_depth: false,
            mpf_filter: None,
        }
    }

    /// Preserve everything.
    #[must_use]
    pub fn all() -> Self {
        Self {
            jfif: true,
            exif: true,
            xmp: true,
            icc: IccPreserve::All,
            iptc: true,
            adobe: true,
            com: true,
            app_unknown: true,
            mpf_gainmaps: true,
            mpf_thumbnails: true,
            mpf_multiframe: true,
            mpf_depth: true,
            mpf_filter: None,
        }
    }

    /// Set JFIF preservation.
    #[must_use]
    pub fn jfif(mut self, keep: bool) -> Self {
        self.jfif = keep;
        self
    }

    /// Set EXIF preservation.
    #[must_use]
    pub fn exif(mut self, keep: bool) -> Self {
        self.exif = keep;
        self
    }

    /// Set XMP preservation.
    #[must_use]
    pub fn xmp(mut self, keep: bool) -> Self {
        self.xmp = keep;
        self
    }

    /// Set ICC profile preservation mode.
    #[must_use]
    pub fn icc(mut self, mode: IccPreserve) -> Self {
        self.icc = mode;
        self
    }

    /// Set IPTC preservation.
    #[must_use]
    pub fn iptc(mut self, keep: bool) -> Self {
        self.iptc = keep;
        self
    }

    /// Set Adobe segment preservation.
    #[must_use]
    pub fn adobe(mut self, keep: bool) -> Self {
        self.adobe = keep;
        self
    }

    /// Set comment preservation.
    #[must_use]
    pub fn com(mut self, keep: bool) -> Self {
        self.com = keep;
        self
    }

    /// Set unknown APP marker preservation.
    #[must_use]
    pub fn app_unknown(mut self, keep: bool) -> Self {
        self.app_unknown = keep;
        self
    }

    /// Set gain map preservation (UltraHDR).
    #[must_use]
    pub fn mpf_gainmaps(mut self, keep: bool) -> Self {
        self.mpf_gainmaps = keep;
        self
    }

    /// Set thumbnail preservation.
    #[must_use]
    pub fn mpf_thumbnails(mut self, keep: bool) -> Self {
        self.mpf_thumbnails = keep;
        self
    }

    /// Set multi-frame image preservation.
    #[must_use]
    pub fn mpf_multiframe(mut self, keep: bool) -> Self {
        self.mpf_multiframe = keep;
        self
    }

    /// Set depth map preservation.
    #[must_use]
    pub fn mpf_depth(mut self, keep: bool) -> Self {
        self.mpf_depth = keep;
        self
    }

    /// Custom MPF filter (called for each secondary image).
    #[must_use]
    pub fn mpf_filter<F>(mut self, f: F) -> Self
    where
        F: Fn(usize, MpfImageType, u32) -> bool + Send + Sync + 'static,
    {
        self.mpf_filter = Some(Arc::new(f));
        self
    }

    /// Check if any metadata preservation is enabled.
    pub(crate) fn preserves_any_metadata(&self) -> bool {
        self.jfif
            || self.exif
            || self.xmp
            || self.icc != IccPreserve::None
            || self.iptc
            || self.adobe
            || self.com
            || self.app_unknown
    }

    /// Check if any MPF preservation is enabled.
    pub(crate) fn preserves_any_mpf(&self) -> bool {
        self.mpf_gainmaps
            || self.mpf_thumbnails
            || self.mpf_multiframe
            || self.mpf_depth
            || self.mpf_filter.is_some()
    }
}

/// A preserved segment from the JPEG.
#[derive(Clone, Debug)]
pub struct PreservedSegment {
    /// The marker byte (e.g., 0xE0 for APP0)
    pub marker: u8,
    /// Raw segment data (excluding length bytes)
    pub data: Vec<u8>,
    /// Detected segment type
    pub segment_type: SegmentType,
}

/// A preserved MPF secondary image.
#[derive(Clone, Debug)]
pub struct PreservedMpfImage {
    /// Index in the MPF directory (0 = primary, 1+ = secondary)
    pub mpf_index: usize,
    /// Image type from MPF entry
    pub image_type: MpfImageType,
    /// Complete JPEG data (SOI to EOI)
    pub data: Vec<u8>,
}

/// MPF directory parsed from APP2.
#[derive(Clone, Debug)]
pub struct MpfDirectory {
    /// MPF version bytes
    pub version: [u8; 4],
    /// Image entries
    pub images: Vec<MpfEntry>,
}

/// Entry in the MPF directory.
#[derive(Clone, Debug)]
pub struct MpfEntry {
    /// Image type
    pub image_type: MpfImageType,
    /// Offset from start of file
    pub offset: u32,
    /// Image size in bytes
    pub size: u32,
    /// Dependent image 1 index (if any)
    pub dependent_image1: Option<u16>,
    /// Dependent image 2 index (if any)
    pub dependent_image2: Option<u16>,
}

/// Well-known standard ICC profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandardProfile {
    /// sRGB IEC61966-2.1
    SrgbIec61966,
    /// Display P3
    DisplayP3,
}

/// Preserved metadata and secondary images from decode.
///
/// Raw bytes are buffered during decode. Parsing is lazy (on first access).
pub struct DecodedExtras {
    /// Raw buffered segments
    pub(crate) segments: Vec<PreservedSegment>,
    /// Secondary images extracted from MPF
    pub(crate) secondary_images: Vec<PreservedMpfImage>,
    /// Probe result from header analysis (encoder ID, quality estimate, DQT tables).
    pub(crate) probe: Option<JpegProbe>,

    // Lazy parse cache
    xmp_cache: OnceLock<Option<String>>,
    icc_cache: OnceLock<Option<Vec<u8>>>,
    mpf_cache: OnceLock<Option<MpfDirectory>>,
}

impl core::fmt::Debug for DecodedExtras {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodedExtras")
            .field("segments", &self.segments.len())
            .field("secondary_images", &self.secondary_images.len())
            .field("has_probe", &self.probe.is_some())
            .finish()
    }
}

impl Clone for DecodedExtras {
    fn clone(&self) -> Self {
        Self {
            segments: self.segments.clone(),
            secondary_images: self.secondary_images.clone(),
            probe: self.probe.clone(),
            xmp_cache: OnceLock::new(),
            icc_cache: OnceLock::new(),
            mpf_cache: OnceLock::new(),
        }
    }
}

impl DecodedExtras {
    /// Create empty extras.
    pub(crate) fn new() -> Self {
        Self {
            segments: Vec::new(),
            secondary_images: Vec::new(),
            probe: None,
            xmp_cache: OnceLock::new(),
            icc_cache: OnceLock::new(),
            mpf_cache: OnceLock::new(),
        }
    }

    /// Check if there are any preserved segments or images.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty() && self.secondary_images.is_empty()
    }

    // === Raw segment access ===

    /// All preserved segments.
    #[must_use]
    pub fn segments(&self) -> &[PreservedSegment] {
        &self.segments
    }

    /// Segments by type.
    pub fn segments_by_type(&self, typ: SegmentType) -> impl Iterator<Item = &PreservedSegment> {
        self.segments.iter().filter(move |s| s.segment_type == typ)
    }

    /// Remove all segments of a given type.
    pub(crate) fn remove_segments_by_type(&mut self, typ: SegmentType) {
        self.segments.retain(|s| s.segment_type != typ);
    }

    /// Segments by marker.
    pub fn segments_by_marker(&self, marker: u8) -> impl Iterator<Item = &PreservedSegment> {
        self.segments.iter().filter(move |s| s.marker == marker)
    }

    // === Lazy-parsed metadata access ===

    /// JFIF info (density, aspect ratio).
    #[must_use]
    pub fn jfif(&self) -> Option<JfifInfo> {
        self.segments_by_type(SegmentType::Jfif)
            .next()
            .and_then(|seg| parse_jfif(&seg.data))
    }

    /// EXIF data (returns raw bytes for external parsing).
    #[must_use]
    pub fn exif(&self) -> Option<&[u8]> {
        self.segments_by_type(SegmentType::Exif)
            .next()
            .map(|seg| seg.data.as_slice())
    }

    /// XMP string (reassembled from extended XMP if needed).
    #[must_use]
    pub fn xmp(&self) -> Option<&str> {
        self.xmp_cache
            .get_or_init(|| self.reassemble_xmp())
            .as_deref()
    }

    /// ICC profile (reassembled from chunks).
    #[must_use]
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_cache
            .get_or_init(|| self.reassemble_icc())
            .as_deref()
    }

    /// Check if ICC is a standard profile (sRGB, Display P3).
    #[must_use]
    pub fn icc_is_standard(&self) -> Option<StandardProfile> {
        let icc = self.icc_profile()?;
        detect_standard_profile(icc)
    }

    /// IPTC data (returns raw bytes for external parsing).
    #[must_use]
    pub fn iptc(&self) -> Option<&[u8]> {
        self.segments_by_type(SegmentType::Iptc)
            .next()
            .map(|seg| seg.data.as_slice())
    }

    /// Adobe segment info.
    #[must_use]
    pub fn adobe(&self) -> Option<AdobeInfo> {
        self.segments_by_type(SegmentType::Adobe)
            .next()
            .and_then(|seg| parse_adobe(&seg.data))
    }

    /// Comment strings.
    pub fn comments(&self) -> impl Iterator<Item = &str> {
        self.segments_by_type(SegmentType::Comment)
            .filter_map(|seg| core::str::from_utf8(&seg.data).ok())
    }

    // === Quality estimation and encoder identification ===

    /// Full probe result from header analysis.
    ///
    /// Contains encoder identification, quality estimate, dimensions,
    /// subsampling, mode, and raw quantization tables. Available when
    /// metadata preservation is enabled (the default).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let extras = decoded.extras().unwrap();
    /// if let Some(probe) = extras.probe() {
    ///     println!("Encoder: {:?}", probe.encoder);
    ///     println!("Quality: {:.0} ({:?})", probe.quality.value, probe.quality.scale);
    /// }
    /// ```
    #[must_use]
    pub fn probe(&self) -> Option<&JpegProbe> {
        self.probe.as_ref()
    }

    /// Estimate the encoding quality from quantization tables.
    ///
    /// Returns the quality estimate with a value on the appropriate scale
    /// for the detected encoder (IJG 1-100, mozjpeg 1-100, or butteraugli distance).
    #[must_use]
    pub fn quality_estimate(&self) -> Option<&QualityEstimate> {
        self.probe.as_ref().map(|p| &p.quality)
    }

    /// Detected encoder family (libjpeg-turbo, mozjpeg, jpegli, Photoshop, etc.).
    #[must_use]
    pub fn encoder(&self) -> Option<EncoderFamily> {
        self.probe.as_ref().map(|p| p.encoder)
    }

    /// Raw luminance quantization table (64 values in natural/row-major order).
    ///
    /// This is DQT table 0, which is typically the luminance (Y) table.
    /// Values are in the range 1-255 for baseline JPEG, 1-65535 for extended.
    #[must_use]
    pub fn luminance_qt(&self) -> Option<&[u16; 64]> {
        self.probe
            .as_ref()?
            .dqt_tables
            .iter()
            .find(|t| t.index == 0)
            .map(|t| &t.values)
    }

    /// Raw chrominance quantization table (64 values in natural/row-major order).
    ///
    /// This is DQT table 1, which is typically the chrominance (Cb/Cr) table.
    /// Returns `None` for grayscale images (only one DQT table).
    #[must_use]
    pub fn chrominance_qt(&self) -> Option<&[u16; 64]> {
        self.probe
            .as_ref()?
            .dqt_tables
            .iter()
            .find(|t| t.index == 1)
            .map(|t| &t.values)
    }

    /// All quantization tables extracted from the JPEG.
    ///
    /// Typically 1 table for grayscale, 2 for standard color, 3 for jpegli.
    /// Tables are in natural (row-major) order with their original table indices.
    #[must_use]
    pub fn dqt_tables(&self) -> Option<&[DqtTable]> {
        self.probe.as_ref().map(|p| p.dqt_tables.as_slice())
    }

    /// MPF directory (parsed from APP2).
    #[must_use]
    pub fn mpf(&self) -> Option<&MpfDirectory> {
        self.mpf_cache.get_or_init(|| self.parse_mpf()).as_ref()
    }

    // === MPF secondary images ===

    /// All preserved secondary images.
    #[must_use]
    pub fn secondary_images(&self) -> &[PreservedMpfImage] {
        &self.secondary_images
    }

    /// Get secondary image by MPF index.
    #[must_use]
    pub fn secondary_image(&self, mpf_index: usize) -> Option<&[u8]> {
        self.secondary_images
            .iter()
            .find(|img| img.mpf_index == mpf_index)
            .map(|img| img.data.as_slice())
    }

    /// Get first gain map (first Undefined-type secondary image).
    #[must_use]
    pub fn gainmap(&self) -> Option<&[u8]> {
        self.secondary_images
            .iter()
            .find(|img| img.image_type.is_gainmap())
            .map(|img| img.data.as_slice())
    }

    /// Get MPF depth/disparity map JPEG bytes if present.
    ///
    /// Returns raw JPEG bytes of the first MPF secondary image with
    /// [`MpfImageType::Disparity`] type. For a richer API that also
    /// checks GDepth XMP and Dynamic Depth Format, use
    /// [`extract_depth_map()`](Self::extract_depth_map).
    #[must_use]
    pub fn depth_map(&self) -> Option<&[u8]> {
        self.secondary_images
            .iter()
            .find(|img| img.image_type.is_depth())
            .map(|img| img.data.as_slice())
    }

    // === For encoder round-trip ===

    /// Convert to encoder segments for round-trip encoding.
    ///
    /// Includes: JFIF, EXIF, XMP, ICC, IPTC, Adobe, Comments
    /// Excludes: MPF directory (encoder regenerates it), unknown segments
    ///
    /// Secondary images (gain maps, depth maps) are included.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decoder::Decoder;
    /// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
    ///
    /// let decoded = Decoder::new().decode(&original)?;
    /// let extras = decoded.extras().unwrap();
    ///
    /// let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    ///     .with_segments(extras.to_encoder_segments())
    ///     .encode_oneshot(&pixels, w, h, layout)?;
    /// ```
    #[must_use]
    pub fn to_encoder_segments(&self) -> crate::encode::extras::EncoderSegments {
        use crate::encode::extras::EncoderSegments;

        let mut segments = EncoderSegments::new();

        // Copy metadata segments (exclude MPF directory and unknown)
        for seg in &self.segments {
            match seg.segment_type {
                SegmentType::Jfif
                | SegmentType::Exif
                | SegmentType::Xmp
                | SegmentType::XmpExtended
                | SegmentType::Icc
                | SegmentType::Iptc
                | SegmentType::Adobe
                | SegmentType::Comment => {
                    segments.add_mut(seg.marker, seg.data.clone(), seg.segment_type);
                }
                SegmentType::Mpf => {
                    // Skip - encoder regenerates MPF directory
                }
                SegmentType::Unknown => {
                    // Skip unknown by default
                }
            }
        }

        // Copy secondary images
        for img in &self.secondary_images {
            segments.add_mpf_image_mut(img.data.clone(), img.image_type);
        }

        segments
    }

    /// Convert to encoder segments with custom filter.
    ///
    /// The filter function receives each preserved segment and returns true
    /// to include it in the output.
    #[must_use]
    pub fn to_encoder_segments_filtered<F>(
        &self,
        filter: F,
    ) -> crate::encode::extras::EncoderSegments
    where
        F: Fn(&PreservedSegment) -> bool,
    {
        use crate::encode::extras::EncoderSegments;

        let mut segments = EncoderSegments::new();

        for seg in &self.segments {
            if seg.segment_type != SegmentType::Mpf && filter(seg) {
                segments.add_mut(seg.marker, seg.data.clone(), seg.segment_type);
            }
        }

        // Copy secondary images
        for img in &self.secondary_images {
            segments.add_mpf_image_mut(img.data.clone(), img.image_type);
        }

        segments
    }

    /// Convert to raw segment tuples (legacy API).
    /// Maintains original order, excludes MPF (encoder regenerates it).
    #[must_use]
    pub fn to_raw_segments(&self) -> Vec<(u8, Vec<u8>)> {
        self.segments
            .iter()
            .filter(|seg| seg.segment_type != SegmentType::Mpf)
            .map(|seg| (seg.marker, seg.data.clone()))
            .collect()
    }

    // === Internal helpers ===

    /// Reassemble XMP from primary and extended chunks.
    fn reassemble_xmp(&self) -> Option<String> {
        // Get primary XMP
        let primary = self
            .segments_by_type(SegmentType::Xmp)
            .next()
            .map(|seg| &seg.data)?;

        // Check for XMP namespace prefix and strip it
        const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
        let primary_xmp = if primary.starts_with(XMP_NS) {
            &primary[XMP_NS.len()..]
        } else {
            primary
        };

        let primary_str = core::str::from_utf8(primary_xmp).ok()?;

        // Check for extended XMP
        let extended: Vec<_> = self.segments_by_type(SegmentType::XmpExtended).collect();
        if extended.is_empty() {
            return Some(primary_str.to_string());
        }

        // Parse and reassemble extended XMP chunks
        // Format: namespace + GUID (32 bytes) + total_length (4 bytes) + offset (4 bytes) + data
        const EXT_NS: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";
        let mut chunks: Vec<(u32, &[u8])> = Vec::new();

        for seg in extended {
            if seg.data.len() < EXT_NS.len() + 40 {
                continue;
            }
            if !seg.data.starts_with(EXT_NS) {
                continue;
            }
            let offset_start = EXT_NS.len() + 32 + 4; // After namespace + GUID + total_length
            if seg.data.len() < offset_start + 4 {
                continue;
            }
            let offset = u32::from_be_bytes([
                seg.data[offset_start],
                seg.data[offset_start + 1],
                seg.data[offset_start + 2],
                seg.data[offset_start + 3],
            ]);
            let data = &seg.data[offset_start + 4..];
            chunks.push((offset, data));
        }

        if chunks.is_empty() {
            return Some(primary_str.to_string());
        }

        // Sort by offset and concatenate
        chunks.sort_by_key(|(off, _)| *off);
        let extended_data: Vec<u8> = chunks
            .into_iter()
            .flat_map(|(_, data)| data)
            .copied()
            .collect();
        let extended_str = core::str::from_utf8(&extended_data).ok()?;

        Some(format!("{}{}", primary_str, extended_str))
    }

    /// Reassemble ICC profile from chunks.
    fn reassemble_icc(&self) -> Option<Vec<u8>> {
        let icc_segments: Vec<_> = self.segments_by_type(SegmentType::Icc).collect();
        if icc_segments.is_empty() {
            return None;
        }

        // Parse chunk numbers and sort
        // Format: "ICC_PROFILE\0" + chunk_num (1 byte) + total_chunks (1 byte) + data
        const ICC_SIG: &[u8] = b"ICC_PROFILE\0";
        let mut chunks: Vec<(u8, &[u8])> = Vec::new();

        for seg in icc_segments {
            if seg.data.len() < ICC_SIG.len() + 2 {
                continue;
            }
            if !seg.data.starts_with(ICC_SIG) {
                continue;
            }
            let chunk_num = seg.data[ICC_SIG.len()];
            let data = &seg.data[ICC_SIG.len() + 2..];
            chunks.push((chunk_num, data));
        }

        if chunks.is_empty() {
            return None;
        }

        // Sort by chunk number and concatenate
        chunks.sort_by_key(|(num, _)| *num);
        let profile: Vec<u8> = chunks
            .into_iter()
            .flat_map(|(_, data)| data)
            .copied()
            .collect();

        Some(profile)
    }

    /// Parse MPF directory from APP2 segment.
    fn parse_mpf(&self) -> Option<MpfDirectory> {
        let mpf_seg = self.segments_by_type(SegmentType::Mpf).next()?;
        parse_mpf_directory(&mpf_seg.data)
    }

    /// Add a segment (internal use during parsing).
    pub(crate) fn add_segment(&mut self, marker: u8, data: Vec<u8>, segment_type: SegmentType) {
        self.segments.push(PreservedSegment {
            marker,
            data,
            segment_type,
        });
    }

    /// Add a secondary image (internal use during parsing).
    pub(crate) fn add_secondary_image(
        &mut self,
        mpf_index: usize,
        image_type: MpfImageType,
        data: Vec<u8>,
    ) {
        self.secondary_images.push(PreservedMpfImage {
            mpf_index,
            image_type,
            data,
        });
    }
}

// === Segment parsing helpers ===

/// Parse JFIF APP0 segment.
fn parse_jfif(data: &[u8]) -> Option<JfifInfo> {
    // Format: "JFIF\0" + version_major + version_minor + units + x_density + y_density + ...
    const JFIF_SIG: &[u8] = b"JFIF\0";
    if data.len() < JFIF_SIG.len() + 7 {
        return None;
    }
    if !data.starts_with(JFIF_SIG) {
        return None;
    }

    let offset = JFIF_SIG.len();
    let version_major = data[offset];
    let version_minor = data[offset + 1];
    let units_byte = data[offset + 2];
    let x_density = u16::from_be_bytes([data[offset + 3], data[offset + 4]]);
    let y_density = u16::from_be_bytes([data[offset + 5], data[offset + 6]]);

    let density_units = match units_byte {
        0 => DensityUnits::None,
        1 => DensityUnits::PixelsPerInch,
        2 => DensityUnits::PixelsPerCm,
        _ => DensityUnits::None,
    };

    Some(JfifInfo {
        version_major,
        version_minor,
        density_units,
        x_density,
        y_density,
    })
}

/// Parse Adobe APP14 segment.
fn parse_adobe(data: &[u8]) -> Option<AdobeInfo> {
    // Format: "Adobe\0" + version (2 bytes) + flags0 (2) + flags1 (2) + color_transform (1)
    const ADOBE_SIG: &[u8] = b"Adobe";
    if data.len() < ADOBE_SIG.len() + 7 {
        return None;
    }
    if !data.starts_with(ADOBE_SIG) {
        return None;
    }

    let offset = ADOBE_SIG.len();
    let version = u16::from_be_bytes([data[offset], data[offset + 1]]);
    let color_transform_byte = data[offset + 6];

    let color_transform = match color_transform_byte {
        0 => AdobeColorTransform::Unknown,
        1 => AdobeColorTransform::YCbCr,
        2 => AdobeColorTransform::Ycck,
        _ => AdobeColorTransform::Unknown,
    };

    Some(AdobeInfo {
        version,
        color_transform,
    })
}

/// Parse MPF directory from APP2 data.
pub(crate) fn parse_mpf_directory(data: &[u8]) -> Option<MpfDirectory> {
    // Format: "MPF\0" + endianness marker + ...
    const MPF_SIG: &[u8] = b"MPF\0";
    if data.len() < MPF_SIG.len() + 8 {
        return None;
    }
    if !data.starts_with(MPF_SIG) {
        return None;
    }

    let offset = MPF_SIG.len();

    // Check endianness (II = little, MM = big)
    let is_little_endian = &data[offset..offset + 2] == b"II";

    // Read helper
    let read_u16 = |pos: usize| -> u16 {
        if is_little_endian {
            u16::from_le_bytes([data[pos], data[pos + 1]])
        } else {
            u16::from_be_bytes([data[pos], data[pos + 1]])
        }
    };

    let read_u32 = |pos: usize| -> u32 {
        if is_little_endian {
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
        } else {
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
        }
    };

    // Skip to IFD offset (at offset + 4)
    let ifd_offset = read_u32(offset + 4) as usize;
    let ifd_pos = offset + ifd_offset;

    if data.len() < ifd_pos + 2 {
        return None;
    }

    // Read version (typically from tag 0xB000)
    let version = [0u8; 4]; // Simplified - would need full TIFF parsing

    // Read number of directory entries
    let num_entries = read_u16(ifd_pos) as usize;

    // Find MP Entry tag (0xB002)
    // Some MPF generators produce non-standard structures with extra bytes between entries,
    // so we scan for the tag pattern instead of assuming fixed 12-byte entry spacing.
    let mut mp_entry_offset = None;
    let mut mp_entry_count = 0u32;

    // First try standard 12-byte entry spacing
    for i in 0..num_entries {
        let entry_pos = ifd_pos + 2 + i * 12;
        if data.len() < entry_pos + 12 {
            break;
        }

        let tag = read_u16(entry_pos);
        if tag == 0xB002 {
            // MP Entry - type should be UNDEFINED (7), count is total bytes
            mp_entry_count = read_u32(entry_pos + 4);
            mp_entry_offset = Some(read_u32(entry_pos + 8) as usize + offset);
            break;
        }
    }

    // If not found with standard spacing, scan for the 0xB002 tag pattern
    // This handles malformed MPF structures (e.g., from some Android camera apps)
    if mp_entry_offset.is_none() {
        let tag_bytes = if is_little_endian {
            [0x02, 0xB0] // 0xB002 in little-endian
        } else {
            [0xB0, 0x02] // 0xB002 in big-endian
        };

        // Scan from after the entry count through the expected IFD area
        // Limit search to a reasonable range (256 bytes covers most MPF structures)
        let scan_start = ifd_pos + 2;
        let scan_end = (scan_start + 256).min(data.len().saturating_sub(12));

        for pos in scan_start..scan_end {
            if data.len() >= pos + 12 && data[pos..pos + 2] == tag_bytes {
                // Found the tag, verify it looks like a valid entry
                let type_val = read_u16(pos + 2);
                if type_val == 7 {
                    // Type 7 = UNDEFINED, expected for MP Entry
                    mp_entry_count = read_u32(pos + 4);
                    mp_entry_offset = Some(read_u32(pos + 8) as usize + offset);
                    break;
                }
            }
        }
    }

    let mp_entry_offset = mp_entry_offset?;

    // Each MP entry is 16 bytes
    let num_images = (mp_entry_count / 16) as usize;
    let mut images = Vec::with_capacity(num_images);

    for i in 0..num_images {
        let entry_pos = mp_entry_offset + i * 16;
        if data.len() < entry_pos + 16 {
            break;
        }

        let attr = read_u32(entry_pos);
        let image_type = MpfImageType::from_type_code(attr & 0x00FFFFFF);
        let size = read_u32(entry_pos + 4);
        let entry_offset = read_u32(entry_pos + 8);
        let dep1 = read_u16(entry_pos + 12);
        let dep2 = read_u16(entry_pos + 14);

        images.push(MpfEntry {
            image_type,
            offset: entry_offset,
            size,
            dependent_image1: if dep1 != 0 { Some(dep1) } else { None },
            dependent_image2: if dep2 != 0 { Some(dep2) } else { None },
        });
    }

    Some(MpfDirectory { version, images })
}

/// Detect if ICC profile is a well-known standard profile.
fn detect_standard_profile(icc: &[u8]) -> Option<StandardProfile> {
    // Check profile description tag for known profiles
    // sRGB profiles typically have "sRGB" in description
    // Display P3 profiles have "Display P3" in description

    // Simple heuristic: check for signature strings
    if icc.windows(4).any(|w| w == b"sRGB") {
        return Some(StandardProfile::SrgbIec61966);
    }
    if icc.windows(10).any(|w| w == b"Display P3") {
        return Some(StandardProfile::DisplayP3);
    }

    None
}

/// Detect segment type from marker and data.
pub(crate) fn detect_segment_type(marker: u8, data: &[u8]) -> SegmentType {
    match marker {
        0xE0 => {
            // APP0 - JFIF?
            if data.starts_with(b"JFIF\0") {
                SegmentType::Jfif
            } else {
                SegmentType::Unknown
            }
        }
        0xE1 => {
            // APP1 - EXIF or XMP?
            if data.starts_with(b"Exif\0\0") {
                SegmentType::Exif
            } else if data.starts_with(b"http://ns.adobe.com/xap/1.0/\0") {
                SegmentType::Xmp
            } else if data.starts_with(b"http://ns.adobe.com/xmp/extension/\0") {
                SegmentType::XmpExtended
            } else {
                SegmentType::Unknown
            }
        }
        0xE2 => {
            // APP2 - ICC or MPF?
            if data.starts_with(b"ICC_PROFILE\0") {
                SegmentType::Icc
            } else if data.starts_with(b"MPF\0") {
                SegmentType::Mpf
            } else {
                SegmentType::Unknown
            }
        }
        0xED => {
            // APP13 - IPTC?
            if data.starts_with(b"Photoshop 3.0\0") {
                SegmentType::Iptc
            } else {
                SegmentType::Unknown
            }
        }
        0xEE => {
            // APP14 - Adobe?
            if data.starts_with(b"Adobe") {
                SegmentType::Adobe
            } else {
                SegmentType::Unknown
            }
        }
        0xFE => SegmentType::Comment,
        _ => SegmentType::Unknown,
    }
}

/// Check if a segment should be preserved based on config.
pub(crate) fn should_preserve_segment(config: &PreserveConfig, segment_type: SegmentType) -> bool {
    match segment_type {
        SegmentType::Jfif => config.jfif,
        SegmentType::Exif => config.exif,
        SegmentType::Xmp | SegmentType::XmpExtended => config.xmp,
        SegmentType::Icc => config.icc != IccPreserve::None,
        SegmentType::Mpf => config.preserves_any_mpf(), // Keep MPF if we want any secondary images
        SegmentType::Iptc => config.iptc,
        SegmentType::Adobe => config.adobe,
        SegmentType::Comment => config.com,
        SegmentType::Unknown => config.app_unknown,
    }
}

/// Check if an MPF image should be preserved based on config.
pub(crate) fn should_preserve_mpf_image(
    config: &PreserveConfig,
    index: usize,
    image_type: MpfImageType,
    size: u32,
) -> bool {
    // Custom filter takes precedence
    if let Some(ref filter) = config.mpf_filter {
        return filter(index, image_type, size);
    }

    match image_type {
        MpfImageType::Undefined => config.mpf_gainmaps,
        MpfImageType::LargeThumbnailVga | MpfImageType::LargeThumbnailFullHd => {
            config.mpf_thumbnails
        }
        MpfImageType::Panorama | MpfImageType::MultiAngle => config.mpf_multiframe,
        MpfImageType::Disparity => config.mpf_depth,
        MpfImageType::BaselinePrimary => false, // Primary is the main decode result
        MpfImageType::Other(_) | _ => config.app_unknown,
    }
}
