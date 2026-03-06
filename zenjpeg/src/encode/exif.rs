//! Minimal EXIF builder for common metadata fields.
//!
//! Provides a type-safe API for embedding EXIF metadata without requiring
//! users to construct raw TIFF/EXIF bytes.
//!
//! # Usage
//!
//! ```ignore
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, Exif, Orientation};
//!
//! // Build from fields (compile-time safe - can't mix with raw)
//! let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
//!     .exif(Exif::build()
//!         .orientation(Orientation::Rotate90)
//!         .copyright("© 2024 Example Corp"));
//!
//! // Or use raw EXIF bytes
//! let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
//!     .exif(Exif::raw(my_exif_bytes));
//! ```

/// EXIF orientation values (rotation/flip).
///
/// These correspond to the EXIF Orientation tag (0x0112) values 1-8.
/// Most image viewers and browsers respect this tag for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u16)]
pub enum Orientation {
    /// Normal (no transformation needed)
    #[default]
    Normal = 1,
    /// Flip horizontally
    FlipHorizontal = 2,
    /// Rotate 180°
    Rotate180 = 3,
    /// Flip vertically
    FlipVertical = 4,
    /// Transpose (flip + rotate 90° CW)
    Transpose = 5,
    /// Rotate 90° clockwise
    Rotate90 = 6,
    /// Transverse (flip + rotate 90° CCW)
    Transverse = 7,
    /// Rotate 90° counter-clockwise (270° CW)
    Rotate270 = 8,
}

/// EXIF metadata - either raw bytes or built from common fields.
///
/// Use [`Exif::raw`] for user-provided EXIF TIFF bytes, or [`Exif::build`]
/// to construct EXIF from common fields like orientation and copyright.
///
/// The two modes are mutually exclusive at compile time - you cannot
/// accidentally mix raw bytes with field-based building.
#[derive(Debug, Clone)]
pub enum Exif {
    /// Raw EXIF TIFF bytes (without the `Exif\0\0` APP1 prefix).
    Raw(Vec<u8>),
    /// Built from common fields.
    Fields(ExifFields),
}

impl Exif {
    /// Create EXIF from raw TIFF bytes.
    ///
    /// The bytes should be raw TIFF data without the `Exif\0\0` APP1 prefix
    /// (the encoder adds that automatically).
    #[must_use]
    pub fn raw(bytes: impl Into<Vec<u8>>) -> Self {
        Exif::Raw(bytes.into())
    }

    /// Start building EXIF from common fields.
    ///
    /// Returns an [`ExifFields`] builder that can be chained with
    /// `.orientation()` and `.copyright()` methods.
    #[must_use]
    pub fn build() -> ExifFields {
        ExifFields::default()
    }

    /// Convert to raw TIFF bytes for embedding.
    ///
    /// Returns `None` if no fields are set (for the `Fields` variant).
    #[must_use]
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        match self {
            Exif::Raw(bytes) => Some(bytes.clone()),
            Exif::Fields(fields) => fields.to_bytes(),
        }
    }
}

impl From<ExifFields> for Exif {
    fn from(fields: ExifFields) -> Self {
        Exif::Fields(fields)
    }
}

/// Common EXIF fields for building metadata.
///
/// Created via [`Exif::build()`], this struct provides a type-safe builder
/// for common EXIF tags. Chain methods to set fields, then pass to
/// [`EncodeRequest::exif()`](super::request::EncodeRequest::exif).
#[derive(Debug, Clone, Default)]
pub struct ExifFields {
    orientation: Option<Orientation>,
    copyright: Option<String>,
}

impl ExifFields {
    /// Set the EXIF orientation tag.
    ///
    /// This controls how image viewers should rotate/flip the image for display.
    #[must_use]
    pub fn orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = Some(orientation);
        self
    }

    /// Set the EXIF copyright tag.
    ///
    /// Standard format is "Copyright, Owner Name, Year" but any string works.
    #[must_use]
    pub fn copyright(mut self, copyright: impl Into<String>) -> Self {
        self.copyright = Some(copyright.into());
        self
    }

    /// Convert to raw TIFF bytes.
    ///
    /// Returns `None` if no fields are set.
    #[must_use]
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        if self.orientation.is_none() && self.copyright.is_none() {
            return None;
        }
        Some(build_exif_tiff(self.orientation, self.copyright.as_deref()))
    }
}

/// Build minimal EXIF TIFF data with orientation and optional copyright.
///
/// Returns raw TIFF data (without the `Exif\0\0` APP1 prefix - that's added
/// by the encoder automatically).
fn build_exif_tiff(orientation: Option<Orientation>, copyright: Option<&str>) -> Vec<u8> {
    // Count how many IFD entries we need
    let mut entry_count: u16 = 0;
    if orientation.is_some() {
        entry_count += 1;
    }
    if copyright.is_some() {
        entry_count += 1;
    }

    if entry_count == 0 {
        return Vec::new();
    }

    // Calculate sizes
    // TIFF header: 8 bytes
    // IFD: 2 (count) + 12*entries + 4 (next IFD offset)
    let ifd_size = 2 + 12 * entry_count as usize + 4;
    let header_and_ifd = 8 + ifd_size;

    // Copyright string goes after IFD if it doesn't fit inline (>4 bytes)
    let copyright_bytes = copyright.map(|s| {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0); // Null terminator
        bytes
    });
    let copyright_len = copyright_bytes.as_ref().map(|b| b.len()).unwrap_or(0);
    let copyright_inline = copyright_len <= 4;

    let total_size = if copyright_inline {
        header_and_ifd
    } else {
        header_and_ifd + copyright_len
    };

    let mut exif = Vec::with_capacity(total_size);

    // === TIFF Header (8 bytes) ===
    // Byte order: little-endian (Intel)
    exif.extend_from_slice(b"II");
    // TIFF magic number (42)
    exif.extend_from_slice(&42u16.to_le_bytes());
    // Offset to first IFD (immediately after header)
    exif.extend_from_slice(&8u32.to_le_bytes());

    // === IFD0 ===
    // Number of entries
    exif.extend_from_slice(&entry_count.to_le_bytes());

    // Track offset for non-inline values (after IFD)
    let value_offset = header_and_ifd as u32;

    // Entry 1: Orientation (tag 0x0112)
    if let Some(orient) = orientation {
        write_ifd_entry(
            &mut exif,
            0x0112,        // Tag: Orientation
            3,             // Type: SHORT
            1,             // Count: 1
            orient as u32, // Value (inline for SHORT)
        );
    }

    // Entry 2: Copyright (tag 0x8298)
    if let Some(ref bytes) = copyright_bytes {
        let count = bytes.len() as u32;
        let value_or_offset = if copyright_inline {
            // Inline: pad to 4 bytes
            let mut val = [0u8; 4];
            val[..bytes.len()].copy_from_slice(bytes);
            u32::from_le_bytes(val)
        } else {
            // Offset to value
            value_offset
        };

        write_ifd_entry(
            &mut exif,
            0x8298, // Tag: Copyright
            2,      // Type: ASCII
            count,
            value_or_offset,
        );
    }

    // Next IFD offset (0 = no more IFDs)
    exif.extend_from_slice(&0u32.to_le_bytes());

    // === Values that didn't fit inline ===
    if !copyright_inline && let Some(bytes) = copyright_bytes {
        exif.extend_from_slice(&bytes);
    }

    exif
}

/// Write a single IFD entry (12 bytes).
fn write_ifd_entry(buf: &mut Vec<u8>, tag: u16, type_: u16, count: u32, value: u32) {
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(&type_.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orientation_only() {
        let exif = Exif::build().orientation(Orientation::Rotate90);
        let bytes = exif.to_bytes().expect("should produce bytes");

        // Should have TIFF header + 1 IFD entry
        assert!(bytes.len() >= 8 + 2 + 12 + 4); // header + count + entry + next

        // Check TIFF header
        assert_eq!(&bytes[0..2], b"II"); // Little-endian
        assert_eq!(&bytes[2..4], &42u16.to_le_bytes()); // Magic

        // Check entry count
        assert_eq!(&bytes[8..10], &1u16.to_le_bytes());

        // Check orientation tag
        assert_eq!(&bytes[10..12], &0x0112u16.to_le_bytes()); // Tag
        assert_eq!(&bytes[12..14], &3u16.to_le_bytes()); // Type: SHORT
        assert_eq!(&bytes[14..18], &1u32.to_le_bytes()); // Count: 1
        assert_eq!(&bytes[18..20], &6u16.to_le_bytes()); // Value: Rotate90 = 6
    }

    #[test]
    fn test_copyright_short() {
        let exif = Exif::build().copyright("AB");
        let bytes = exif.to_bytes().expect("should produce bytes");

        // Short copyright fits inline
        assert_eq!(bytes.len(), 8 + 2 + 12 + 4); // No extra data

        // Check copyright tag
        assert_eq!(&bytes[10..12], &0x8298u16.to_le_bytes()); // Tag
        assert_eq!(&bytes[12..14], &2u16.to_le_bytes()); // Type: ASCII
        assert_eq!(&bytes[14..18], &3u32.to_le_bytes()); // Count: 3 (AB + null)
    }

    #[test]
    fn test_copyright_long() {
        let long_copyright = "Copyright 2024 Example Corp";
        let exif = Exif::build().copyright(long_copyright);
        let bytes = exif.to_bytes().expect("should produce bytes");

        // Long copyright stored after IFD
        let expected_len = 8 + 2 + 12 + 4 + long_copyright.len() + 1;
        assert_eq!(bytes.len(), expected_len);

        // Copyright string should be at the end
        let string_start = 8 + 2 + 12 + 4;
        assert_eq!(
            &bytes[string_start..string_start + long_copyright.len()],
            long_copyright.as_bytes()
        );
    }

    #[test]
    fn test_both_fields() {
        let exif = Exif::build()
            .orientation(Orientation::Rotate180)
            .copyright("Test");
        let bytes = exif.to_bytes().expect("should produce bytes");

        // 2 entries
        assert_eq!(&bytes[8..10], &2u16.to_le_bytes());

        // Both tags should be present (in order by tag number)
        // Orientation: 0x0112, Copyright: 0x8298
        assert_eq!(&bytes[10..12], &0x0112u16.to_le_bytes());
        assert_eq!(&bytes[22..24], &0x8298u16.to_le_bytes());
    }

    #[test]
    fn test_empty_fields() {
        let exif = Exif::build();
        assert!(exif.to_bytes().is_none(), "empty fields should return None");
    }

    #[test]
    fn test_raw_bytes() {
        let raw = vec![1u8, 2, 3, 4, 5];
        let exif = Exif::raw(raw.clone());
        let bytes = exif.to_bytes().expect("should produce bytes");
        assert_eq!(bytes, raw);
    }

    #[test]
    fn test_chaining_preserves_both() {
        // This is the key test - verify chaining works correctly
        let exif = Exif::build()
            .orientation(Orientation::Rotate90)
            .copyright("Test");

        let bytes = exif.to_bytes().expect("should produce bytes");

        // Should have 2 entries
        assert_eq!(&bytes[8..10], &2u16.to_le_bytes());
    }
}
