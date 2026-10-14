//! JPEG framing for ISO gain-map parameters, independent of HDR rendering.
use crate::error::{Error, Result};

pub(crate) fn iso21496_record(gainmap_jpeg: &[u8]) -> Option<&[u8]> {
    let probe = crate::container::probe(gainmap_jpeg, crate::container::Wants::ISO_GAINMAP);
    let payload = probe.iso_gainmap()?;
    let start = (payload.start as usize).checked_sub(zencodec::ISO_21496_1_URN.len())?;
    gainmap_jpeg.get(start..payload.end as usize)
}

/// Gain-map metadata from a gain-map JPEG's ISO 21496-1 APP2 record (where
/// ISO 21496-1 and the Ultra HDR v1.1 format put the full record; the
/// primary image carries only a version record), or `None` when it has no
/// such record.
pub(crate) fn iso21496_from_gainmap_jpeg(
    gainmap_jpeg: &[u8],
) -> Option<Result<zencodec::GainMapParams>> {
    let body = iso21496_record(gainmap_jpeg)?;
    Some(
        zencodec::gainmap::parse_iso21496_fmt(
            body,
            zencodec::gainmap::Iso21496Format::JpegApp2BodyWithUrn,
        )
        .map_err(|e| Error::decode_error(alloc::format!("ISO 21496-1 gain-map metadata: {e}"))),
    )
}
