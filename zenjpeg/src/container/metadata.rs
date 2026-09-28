//! Privacy filtering of a single JPEG component before gain-map assembly.
//!
//! Retains encoded scans byte-for-byte. Drops source XMP, comments, unknown APP
//! carriers, thumbnails and trailing auxiliary images. The assembler must write
//! new XMP/ISO/MPF from extracted typed parameters and actual output sizes.
//! ICC is an opaque rendering dependency, not certified free of identifying text.
use super::marker::{self, MarkerKind};
use alloc::vec::Vec;
use zencodec::{IccRetention, Metadata, MetadataPolicy};

/// Failure leaves the input untouched; no partial output is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    MalformedJpeg,
    /// Source XMP preservation conflicts with this privacy operation.
    KeepSourceXmp,
    /// Removing an existing ICC profile requires a separate color conversion.
    DropColorProfile,
    Allocation,
    /// Source EXIF could not be interpreted without guessing orientation.
    InvalidExif,
    /// Removing orientation would require rotating both image components.
    DropOrientation,
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "JPEG metadata filtering: {self:?}")
    }
}
impl core::error::Error for Error {}

/// Filter one JPEG base or gain-map component before reassembling the container.
///
/// Use `Web` for retained attribution or `ColorAndRotation` without attribution.
/// Policies retaining source XMP refuse: the complete packet may carry private
/// fields or stale image offsets. Required gain-map XMP is generated separately.
/// Existing ICC bytes and Adobe color-transform markers survive. Generic private
/// APP segments and anything after this image's EOI are deliberately omitted.
/// This is an encoded-byte copy/marker walk, with no pixel decode or re-encode.
pub fn filter_for_gain_map(jpeg: &[u8], policy: &MetadataPolicy) -> Result<Vec<u8>, Error> {
    if policy.fields().xmp.keeps() {
        return Err(Error::KeepSourceXmp);
    }
    if !jpeg.starts_with(&[0xff, 0xd8]) {
        return Err(Error::MalformedJpeg);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(jpeg.len())
        .map_err(|_| Error::Allocation)?;
    let mut expected = 0;
    let mut saw_scan = false;
    let mut icc_chunks = [false; 256];
    let mut icc_count = None;
    for span in marker::iter(jpeg) {
        // The iterator tolerates FF fill. Accept only fill, not unparsed gaps.
        if span.offset < expected || jpeg[expected..span.offset].iter().any(|b| *b != 0xff) {
            return Err(Error::MalformedJpeg);
        }
        expected = span.offset + span.length;
        let keep = match span.kind {
            MarkerKind::Soi => {
                if span.offset != 0 {
                    return Err(Error::MalformedJpeg);
                }
                true
            }
            MarkerKind::Sos => {
                saw_scan = true;
                true
            }
            MarkerKind::Eoi => {
                if icc_count.is_some_and(|count| !(1..=count).all(|i| icc_chunks[i as usize])) {
                    return Err(Error::MalformedJpeg);
                }
                if !saw_scan {
                    return Err(Error::MalformedJpeg);
                }
                out.extend_from_slice(&jpeg[span.offset..expected]);
                return Ok(out);
            }
            MarkerKind::App(1) if span.payload.starts_with(b"Exif\0\0") => {
                let orientation = zencodec::exif::Exif::parse(span.payload)
                    .ok_or(Error::InvalidExif)?
                    .orientation()
                    .unwrap_or_default();
                let source = Metadata::none()
                    .with_exif(span.payload.to_vec())
                    .with_orientation(orientation);
                let filtered = source.filtered(policy);
                if filtered.orientation != source.orientation {
                    return Err(Error::DropOrientation);
                }
                if let Some(exif) = filtered.exif {
                    let length = exif
                        .len()
                        .checked_add(2)
                        .and_then(|v| u16::try_from(v).ok())
                        .ok_or(Error::MalformedJpeg)?;
                    out.extend_from_slice(&[0xff, 0xe1]);
                    out.extend_from_slice(&length.to_be_bytes());
                    out.extend_from_slice(&exif);
                }
                false
            }
            MarkerKind::App(2) if span.payload.starts_with(b"ICC_PROFILE\0") => {
                let seq = *span.payload.get(12).ok_or(Error::MalformedJpeg)?;
                let count = *span.payload.get(13).ok_or(Error::MalformedJpeg)?;
                if seq == 0
                    || count == 0
                    || seq > count
                    || icc_chunks[seq as usize]
                    || icc_count.is_some_and(|c| c != count)
                {
                    return Err(Error::MalformedJpeg);
                }
                icc_count = Some(count);
                icc_chunks[seq as usize] = true;
                if matches!(policy.fields().icc, IccRetention::Drop) {
                    return Err(Error::DropColorProfile);
                }
                true
            }
            MarkerKind::App(14)
                if span.payload.starts_with(b"Adobe") && span.payload.len() == 12 =>
            {
                true
            }
            MarkerKind::App(0)
                if span.payload.starts_with(b"JFIF\0") && span.payload.len() >= 14 =>
            {
                // Keep density/version, remove any embedded RGB thumbnail.
                out.extend_from_slice(&[0xff, 0xe0, 0, 16]);
                out.extend_from_slice(&span.payload[..12]);
                out.extend_from_slice(&[0, 0]);
                false
            }
            MarkerKind::App(_) | MarkerKind::Com => false,
            MarkerKind::Other(_) | MarkerKind::Restart(_) => return Err(Error::MalformedJpeg),
            _ => true,
        };
        if keep {
            out.extend_from_slice(&jpeg[span.offset..expected]);
        }
    }
    Err(Error::MalformedJpeg)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = alloc::vec![0xff, marker];
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(payload);
        out
    }
    fn jpeg(app: &[u8]) -> Vec<u8> {
        let mut out = alloc::vec![0xff, 0xd8];
        out.extend_from_slice(app);
        out.extend_from_slice(&[0xff, 0xda, 0, 2, 17, 0xff, 0, 23, 0xff, 0xd9]);
        out
    }
    #[test]
    fn orientation_survives_exif_pruning_and_discard_refuses() {
        let mut exif = zencodec::exif::Exif::new(zencodec::exif::TextEncoding::Ascii);
        exif.set_orientation(zencodec::Orientation::Rotate90);
        exif.set_artist("PRIVATE");
        let bytes = exif.to_bytes();
        let payload = if bytes.starts_with(b"Exif\0\0") {
            bytes
        } else {
            [b"Exif\0\0".as_slice(), &bytes].concat()
        };
        let input = jpeg(&segment(0xe1, &payload));
        let output = filter_for_gain_map(&input, &MetadataPolicy::ColorAndRotation).unwrap();
        let exif = marker::iter(&output)
            .find(|s| s.kind == MarkerKind::App(1))
            .unwrap();
        assert_eq!(
            zencodec::exif::Exif::parse(exif.payload)
                .unwrap()
                .orientation(),
            Some(zencodec::Orientation::Rotate90)
        );
        assert!(!output.windows(7).any(|w| w == b"PRIVATE"));
        let policy = MetadataPolicy::Custom(zencodec::MetadataFields::DISCARD_ALL);
        assert_eq!(
            filter_for_gain_map(&input, &policy),
            Err(Error::DropOrientation)
        );
    }
    #[test]
    fn filters_between_scans_and_trailers_without_altering_scan_bytes() {
        let first = [0xff, 0xda, 0, 2, 17, 0xff, 0, 23];
        let second = [0xff, 0xda, 0, 2, 19, 0xff, 0xd0, 25];
        let mut input = alloc::vec![0xff, 0xd8];
        input.extend_from_slice(&first);
        input.extend_from_slice(&segment(
            0xe1,
            b"http://ns.adobe.com/xmp/extension/\0PRIVATE",
        ));
        input.extend_from_slice(&second);
        input.extend_from_slice(&[0xff, 0xd9]);
        input.extend_from_slice(b"PRIVATE");
        let expected = [&[0xff, 0xd8][..], &first, &second, &[0xff, 0xd9]].concat();
        assert_eq!(
            filter_for_gain_map(&input, &MetadataPolicy::Web).unwrap(),
            expected
        );
    }
    #[test]
    fn rejects_missing_duplicate_or_truncated_icc_chunks() {
        for payload in [b"ICC_PROFILE\0".as_slice(), b"ICC_PROFILE\0\x01\x02data"] {
            assert_eq!(
                filter_for_gain_map(&jpeg(&segment(0xe2, payload)), &MetadataPolicy::Web),
                Err(Error::MalformedJpeg)
            );
        }
        let one = segment(0xe2, b"ICC_PROFILE\0\x01\x01data");
        assert!(filter_for_gain_map(&jpeg(&one), &MetadataPolicy::Web).is_ok());
        assert_eq!(
            filter_for_gain_map(
                &jpeg(&[one.as_slice(), &one].concat()),
                &MetadataPolicy::Web
            ),
            Err(Error::MalformedJpeg)
        );
    }
}
