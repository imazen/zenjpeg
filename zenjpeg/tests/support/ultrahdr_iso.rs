//! Independently author ISO APP2 records and use DC-only JPEG rasters.

#[path = "jpeg_writer.rs"]
#[allow(dead_code)]
mod jpeg_writer;

use jpeg_writer::{Comp, Segment, Spec, app14, dc_block, write};

pub fn app2(id: &[u8], body: &[u8]) -> Vec<u8> {
    let len = u16::try_from(2 + id.len() + body.len()).unwrap();
    [&[0xff, 0xe2], len.to_be_bytes().as_slice(), id, body].concat()
}

/// One gray gain component with three wire parameter records. Numerators use
/// denominator 8; each record holds min, max, gamma, base and alternate offsets.
pub fn fixture(
    patches: &[[u8; 3]],
    base_icc: Option<&[u8]>,
    gain: u8,
    alternate_headroom: u32,
    channels: [[u32; 5]; 3],
) -> Vec<u8> {
    let mut base_segments = vec![Segment::PreFrame(app14(0))];
    if let Some(icc) = base_icc {
        let payload = [&[1, 1], icc].concat();
        base_segments.push(Segment::PreFrame(app2(b"ICC_PROFILE\0", &payload)));
    }
    let urn = b"urn:iso:std:iso:ts:21496:-1\0";
    base_segments.push(Segment::PreFrame(app2(urn, &[0, 0])));
    let base = write(&Spec {
        width: patches.len() as u32 * 8,
        height: 8,
        comps: b"RGB".iter().map(|&id| Comp { id, h: 1, v: 1 }).collect(),
        block: &|component, x, _| dc_block(patches[x][component]),
        restart_interval: 0,
        quant: 8,
        segments: base_segments,
    });
    // Min/writer versions and flags: three records, use base color space,
    // common denominator. Headroom and channel records follow in wire order.
    let mut iso = vec![0, 0, 0, 0, 0xc8];
    for n in [8, 0, alternate_headroom * 8]
        .into_iter()
        .chain(channels.into_iter().flatten())
    {
        iso.extend_from_slice(&n.to_be_bytes());
    }
    let secondary = write(&Spec {
        width: 8,
        height: 8,
        comps: vec![Comp { id: 1, h: 1, v: 1 }],
        block: &|_, _, _| dc_block(gain),
        restart_interval: 0,
        quant: 8,
        segments: vec![Segment::PreFrame(app2(urn, &iso))],
    });
    let marker_len = zenjpeg::container::create_mpf_header(0, secondary.len(), Some(2)).len();
    let mpf =
        zenjpeg::container::create_mpf_header(base.len() + marker_len, secondary.len(), Some(2));
    [&base[..2], &mpf, &base[2..], &secondary].concat()
}
