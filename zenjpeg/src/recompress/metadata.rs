//! Admission checks for strategies that can emit only one JPEG image.
use super::error::Error;
use crate::container::marker::{MarkerKind, iter};

const MESSAGE: &str = "lossy recompression cannot preserve auxiliary images; use lossless::restructure or explicitly extract the base image";

pub(super) fn check_single_image(jpeg: &[u8]) -> Result<(), Error> {
    if crate::container::find_jpeg_boundaries(jpeg).len() > 1 {
        return Err(Error::Unsupported(MESSAGE));
    }
    for span in iter(jpeg) {
        let marker = match span.kind {
            MarkerKind::App(n) => 0xe0 + n,
            _ => continue,
        };
        check_segment(marker, span.payload)?;
    }
    Ok(())
}

pub(super) fn check_segment(marker: u8, payload: &[u8]) -> Result<(), Error> {
    if marker == 0xe2
        && (payload.starts_with(b"MPF\0") || payload.starts_with(zencodec::ISO_21496_1_URN))
    {
        return Err(Error::Unsupported(MESSAGE));
    }
    if marker == 0xe1
        && let Some(xml) = payload.strip_prefix(crate::foundation::consts::XMP_NAMESPACE)
        && let Ok(xml) = core::str::from_utf8(xml)
    {
        let directory =
            zenmeta::xmp::container::Directory::parse(xml).is_ok_and(|d| d.items().len() > 1);
        let gainmap = zenmeta::xmp::Packet::parse(xml).is_ok_and(|p| {
            !matches!(
                p.property(crate::container::HDRGM_NAMESPACE, "Version"),
                Ok(None)
            )
        });
        if directory || gainmap {
            return Err(Error::Unsupported(MESSAGE));
        }
    }
    Ok(())
}
