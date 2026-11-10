//! Gain map preservation for UltraHDR JPEGs.
//!
//! Detects, extracts, and reattaches gain map secondary images through
//! layout transforms (lossless or lossy). The gain map maintains spatial
//! lock with the primary image by applying proportional transforms.

use alloc::vec::Vec;

use crate::encode::extras::{EncoderSegments, MpfImageType, inject_encoder_segments};

/// Assemble a primary JPEG + gain map JPEG into an UltraHDR JPEG with MPF.
///
/// Carries the source primary's ISO 21496-1 version record (the re-encoded
/// primary does not have it), injects an MPF APP2 directory into the primary
/// JPEG and appends the gain map JPEG after the primary's EOI.
pub(crate) fn assemble_ultrahdr(source: &[u8], primary: Vec<u8>, gain_map: Vec<u8>) -> Vec<u8> {
    let mut segments = EncoderSegments::new();
    for span in crate::container::iter(source) {
        if matches!(span.kind, crate::container::MarkerKind::Sos) {
            break;
        }
        if matches!(span.kind, crate::container::MarkerKind::App(2))
            && span.payload.starts_with(zencodec::ISO_21496_1_URN)
        {
            segments = segments.add_raw(0xE2, span.payload.to_vec());
        }
    }
    let segments = segments.add_mpf_image(gain_map, MpfImageType::Undefined);
    inject_encoder_segments(primary, &segments)
}

/// `xmp` with its GContainer directory's gain-map `Item:Length` set to
/// `gain_map_len` (zenmeta's directory editor, as the lossless pipeline uses
/// it). A packet without a directory comes back unchanged; one whose
/// directory does not match a single gain map is an error rather than a file
/// whose directory points at bytes it does not have.
pub(crate) fn with_gain_map_length(
    xmp: Option<&str>,
    gain_map_len: usize,
) -> crate::error::Result<Option<alloc::string::String>> {
    use zenmeta::xmp::container::Directory;
    let Some(xmp) = xmp else {
        return Ok(None);
    };
    Directory::parse(xmp)
        .and_then(|d| d.rewrite_item_lengths(&[("GainMap", gain_map_len)]))
        .map(Some)
        .map_err(|e| {
            crate::error::Error(
                crate::error::Error::unsupported_feature(
                    "the XMP GContainer directory cannot be pointed at the transformed gain map",
                )
                .0
                .at_string(|| alloc::format!("{e}")),
            )
        })
}
