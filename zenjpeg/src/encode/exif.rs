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
    ///
    /// Serialization is delegated to [`zencodec::exif::Exif`] — the
    /// canonical authoring path (little-endian, type-2 ASCII strings,
    /// NUL-terminated count-inclusive, inline-if-≤4-bytes, out-of-line
    /// values padded to even offsets per TIFF 6.0 word alignment).
    #[must_use]
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        if self.orientation.is_none() && self.copyright.is_none() {
            return None;
        }
        let mut exif = zencodec::exif::Exif::new(zencodec::exif::TextEncoding::Ascii);
        if let Some(orient) = self.orientation {
            // zenjpeg's repr-u16 enum uses the EXIF values 1-8 directly, so
            // the mapping is total; Identity is an unreachable fallback.
            exif.set_orientation(
                zencodec::Orientation::from_exif(orient as u8)
                    .unwrap_or(zencodec::Orientation::Identity),
            );
        }
        if let Some(ref copyright) = self.copyright {
            exif.set_copyright(copyright);
        }
        Some(exif.to_bytes())
    }
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

    /// The delegated serializer's output must parse back (zencodec parser)
    /// with both fields intact — guards the zenjpeg↔zencodec orientation
    /// value mapping and string encoding across the delegation seam.
    #[test]
    fn test_zencodec_parse_roundtrip() {
        let bytes = Exif::build()
            .orientation(Orientation::Transverse) // EXIF value 7
            .copyright("© 2026 Example") // odd-length UTF-8, exercises padding
            .to_bytes()
            .expect("should produce bytes");
        let parsed = zencodec::exif::Exif::parse(&bytes).expect("zencodec must parse our output");
        assert_eq!(parsed.orientation().map(|o| o.to_exif()), Some(7));
        assert_eq!(parsed.copyright().as_deref(), Some("© 2026 Example"));
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
