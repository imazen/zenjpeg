//! Structural inventory of a JPEG file for
//! [`DecodeJob::inventory`](zencodec::decode::DecodeJob::inventory).
//!
//! The walker follows the marker loop that the zencodec decode path runs —
//! `JpegParser::read_header` (decode/parser/markers.rs) up to the first frame
//! header, then `JpegParser::decode` (decode/parser/mod.rs) up to EOI — rather
//! than [`crate::container::marker`]. The two disagree on exactly the bytes an
//! audit cares about: the container iterator stops at the first stray byte and
//! hides fill bytes, while the decoder skips both and keeps reading.
//!
//! Dispositions describe two consumers:
//!
//! - `Decode::decode`, which parses the whole stream with
//!   `PreserveConfig::all()` and reports ICC, EXIF, XMP and JFIF density
//!   through `ImageInfo` (codec/info.rs `populate_info_from_jpeg_extras`);
//! - `DecodeJob::probe`, which stops at the first frame header
//!   (`DecodeConfig::read_info`) and reports the metadata it saw before it.
//!
//! A metadata segment either one reports is `Metadata`. Data that reaches the
//! caller only through the native `DecodedExtras` attached to `DecodeOutput`
//! (comments, Photoshop resources, MPF images other than a decoded gain map)
//! is `Skipped`, with a detail naming the native accessor.
//!
//! Failures the walker can see without entropy decoding (no SOI, an
//! unsupported or invalid frame header, a malformed segment length, 12-bit
//! precision, DNL mode) make the decode fail; parts the failing decode would
//! have consumed are then `Skipped`, except those `probe` still reads. Entropy
//! errors inside scan data are invisible here: scan data is `ImageData`
//! whenever the container is sound.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use zencodec::ImageFormat;
use zencodec::inventory::{
    Disposition, Inventory, InventoryError, MetadataKind, Part, PartId, PartKind, PartTag,
};

use crate::decode::{
    DecodedExtras, MpfImageType, SegmentType, detect_segment_type, parse_mpf_directory,
};

/// What the decode job is configured to do, where it changes a disposition.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    /// The job applies EXIF orientation to the pixels (`OrientationHint`
    /// `Correct*`), so `find_exif_orientation` reads an EXIF segment.
    pub(crate) auto_orient: bool,
    /// The job decodes the Ultra HDR gain-map image
    /// (`GainMapRender::Components` or `ReconstructHdr`, `ultrahdr` feature).
    pub(crate) gain_map_decoded: bool,
    /// The decoder's pixel cap (`0` = unlimited), checked against the frame
    /// header the way `parse_frame_header` does.
    pub(crate) max_pixels: u64,
}

/// Map `data` as the zencodec decode path reads it.
pub(crate) fn inventory(data: &[u8], opts: Options) -> Result<Inventory, InventoryError> {
    let mut w = Walk {
        data,
        inv: Inventory::new(ImageFormat::Jpeg, data.len() as u64),
        ids: Vec::new(),
        opts,
        repeats: [(None, 0); REPEAT_KINDS],
    };
    let len = data.len();
    let soi = if data.starts_with(&[0xFF, MARKER_SOI]) {
        Some(0)
    } else {
        find_soi(data)
    };
    match soi {
        None => {
            if len > 0 {
                w.add(
                    None,
                    gap(0..len, Disposition::Malformed)
                        .with_detail("no SOI marker: the decoder rejects the input"),
                )?;
            }
        }
        Some(start) => {
            if start > 0 {
                // `JpegParser::with_strictness`: input must begin with SOI.
                w.add(
                    None,
                    gap(0..start, Disposition::Malformed).with_detail(
                        "bytes before SOI: the decoder rejects input that does not start with SOI",
                    ),
                )?;
            }
            let mut st = w.walk_stream(None, start, len)?;
            if start > 0 {
                st.probe_fatal = true;
                st.decode_fatal = true;
            }
            w.resolve(
                &st,
                View {
                    probe: true,
                    embedded: false,
                    auto_orient: opts.auto_orient && start == 0,
                },
            )?;
            if let Some(eoi_end) = st.eoi_end {
                w.after_eoi(&st, eoi_end)?;
            }
        }
    }
    w.finish()
}

const MARKER_SOI: u8 = 0xD8;
const MARKER_EOI: u8 = 0xD9;
const MARKER_SOS: u8 = 0xDA;
const MARKER_DQT: u8 = 0xDB;
const MARKER_DNL: u8 = 0xDC;
const MARKER_DRI: u8 = 0xDD;
const MARKER_DHT: u8 = 0xC4;
const MARKER_DAC: u8 = 0xCC;
const MARKER_APP14: u8 = 0xEE;
const MARKER_COM: u8 = 0xFE;
const MARKER_TEM: u8 = 0x01;

/// `MAX_SCANS` in foundation/alloc.rs: the decoder fails on the 256th scan.
const MAX_SCANS: u32 = 256;
/// `JPEG_MAX_DIMENSION` in foundation/consts.rs, enforced by `validate_dimensions`.
const MAX_DIMENSION: u32 = 65500;
/// Longest label copied from a segment.
const MAX_LABEL: usize = 64;

const XMP_NS_LEN: usize = b"http://ns.adobe.com/xap/1.0/\0".len();
/// Most `rdf:li` occurrences a GContainer directory is parsed with.
const MAX_CONTAINER_LI: usize = 1024;
const XMP_EXT_NS_LEN: usize = b"http://ns.adobe.com/xmp/extension/\0".len();
const ICC_SIG_LEN: usize = b"ICC_PROFILE\0".len();
const ISO_21496_1: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";

/// Parts go straight into the inventory; walker code refers to them by
/// push index, and `ids` maps an index to its `PartId`.
struct Walk<'a> {
    data: &'a [u8],
    inv: Inventory,
    ids: Vec<PartId>,
    opts: Options,
    /// Per [`Repeat`] kind: the first part of that kind (the one carrying
    /// the detail) and how many more followed.
    repeats: [(Option<usize>, u32); REPEAT_KINDS],
}

/// Gaps and standalone markers that a crafted file can repeat once per
/// byte or two. Only the first of each kind carries a detail, so a flood of
/// them costs no string per part.
#[derive(Clone, Copy)]
enum Repeat {
    Stray,
    Fill,
    FfZero,
    HeaderMarker,
    Restart,
}
const REPEAT_KINDS: usize = 5;

/// Which consumers resolve a stream's metadata.
#[derive(Clone, Copy)]
struct View {
    /// `probe()` reads this stream's header (the primary image only).
    probe: bool,
    /// An MPF or GContainer image inside the file, not the primary.
    embedded: bool,
    /// `find_exif_orientation` runs on this stream.
    auto_orient: bool,
}

/// How an embedded image is used.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Decoded as the Ultra HDR gain map.
    GainMap,
    /// Not decoded by the zencodec path.
    Unread,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Before the first supported frame header (`read_header`).
    Header,
    /// After it, up to EOI (`JpegParser::decode`).
    Body,
    Done,
}

/// One APPn or COM segment, kept for metadata resolution.
struct App {
    node: usize,
    marker: u8,
    payload: Range<usize>,
    ty: SegmentType,
    before_sof: bool,
    before_sos: bool,
}

/// The frame header the decoder accepted.
#[derive(Clone, Copy)]
struct Frame {
    components: u8,
    ids: [u8; 4],
    /// Quantisation table index per component.
    qidx: [u8; 4],
    /// The SOF marker: 0xC0/C1 sequential and 0xC2 progressive Huffman,
    /// 0xC9/CA arithmetic.
    mode: u8,
}

/// What a [`Def`] defines: a table slot, a DAC conditioning entry or the
/// restart interval.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DefKind {
    Quant(u8),
    Dc(u8),
    Ac(u8),
    Restart,
    /// DAC DC conditioning (L, U) for an arithmetic DC table.
    DacDc(u8),
    /// DAC AC conditioning (Kx) for an arithmetic AC table.
    DacAc(u8),
}

impl core::fmt::Display for DefKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Quant(i) => write!(f, "quantization table {i}"),
            Self::Dc(i) => write!(f, "DC Huffman table {i}"),
            Self::Ac(i) => write!(f, "AC Huffman table {i}"),
            Self::Restart => f.write_str("restart interval"),
            Self::DacDc(i) => write!(f, "DC conditioning {i}"),
            Self::DacAc(i) => write!(f, "AC conditioning {i}"),
        }
    }
}

/// A definition in effect, and whether a scan (or the final
/// dequantisation) has used it.
#[derive(Clone)]
struct Def {
    node: usize,
    range: Range<usize>,
    what: DefKind,
    used: bool,
}

/// A segment that defines something: its definitions tile `area`.
struct DefSeg {
    node: usize,
    area: Range<usize>,
    count: u32,
    first: DefKind,
}

/// The definitions in effect, for marking units that are overwritten or
/// never used (`Dropped`). Memory is bounded by the number of segments and
/// scans, not by the number of tables a segment packs: a replaced
/// definition that was never used leaves nothing behind but its bytes.
#[derive(Default)]
struct Defs {
    q: [Option<Def>; 4],
    dc: [Option<Def>; 4],
    ac: [Option<Def>; 4],
    dri: Option<Def>,
    dac_dc: [Option<Def>; 4],
    dac_ac: [Option<Def>; 4],
    /// Every definition a scan used, recorded once.
    used: Vec<Def>,
    segs: Vec<DefSeg>,
}

impl Defs {
    fn slot(&mut self, k: DefKind) -> Option<&mut Option<Def>> {
        match k {
            DefKind::Quant(i) => self.q.get_mut(i as usize),
            DefKind::Dc(i) => self.dc.get_mut(i as usize),
            DefKind::Ac(i) => self.ac.get_mut(i as usize),
            DefKind::Restart => Some(&mut self.dri),
            DefKind::DacDc(i) => self.dac_dc.get_mut(i as usize),
            DefKind::DacAc(i) => self.dac_ac.get_mut(i as usize),
        }
    }

    /// Record a definition in segment `node`; it replaces the one in its slot.
    fn define(&mut self, node: usize, range: Range<usize>, what: DefKind) {
        match self.segs.last_mut() {
            Some(s) if s.node == node => {
                s.area.end = range.end;
                s.count += 1;
            }
            _ => self.segs.push(DefSeg {
                node,
                area: range.clone(),
                count: 1,
                first: what,
            }),
        }
        if let Some(slot) = self.slot(what) {
            *slot = Some(Def {
                node,
                range,
                what,
                used: false,
            });
        }
    }

    fn mark(&mut self, k: DefKind) {
        let fresh = match self.slot(k) {
            Some(Some(d)) if !d.used => {
                d.used = true;
                Some(d.clone())
            }
            _ => None,
        };
        if let Some(d) = fresh {
            self.used.push(d);
        }
    }

    /// The quantisation tables every frame component refers to.
    fn mark_quant(&mut self, f: &Frame) {
        for c in 0..f.components as usize {
            self.mark(DefKind::Quant(f.qidx[c]));
        }
    }
}

/// What walking one SOI..EOI stream found.
struct Stream {
    /// Node indices created for this stream (its own parts, not later ones).
    nodes: Range<usize>,
    eoi_end: Option<usize>,
    /// End of the first frame header; `probe()` reads everything before it.
    sof_end: Option<usize>,
    frame: Option<Frame>,
    apps: Vec<App>,
    /// `probe()` fails (the header cannot be parsed up to a supported frame).
    probe_fatal: bool,
    /// `decode()` fails for a container-level reason.
    decode_fatal: bool,
}

/// A length-bearing segment starting at `pos`.
enum SegEnd {
    /// The segment ends here.
    Ends(usize),
    /// Declared length below 2.
    TooShort(usize),
    /// The length field or the declared body runs past the data.
    Truncated,
}

fn part(kind: PartKind, tag: PartTag, r: Range<usize>, d: Disposition) -> Part {
    Part::new(kind, tag, r.start as u64..r.end as u64, d)
}

fn seg(marker: u8, r: Range<usize>, d: Disposition) -> Part {
    part(PartKind::Segment, PartTag::Marker(marker), r, d)
}

fn gap(r: Range<usize>, d: Disposition) -> Part {
    part(PartKind::Gap, PartTag::None, r, d)
}

fn find_soi(data: &[u8]) -> Option<usize> {
    let mut pos = 0;
    while let Some(rel) = data.get(pos..).and_then(|d| memchr::memchr(0xFF, d)) {
        let ff = pos + rel;
        if data.get(ff + 1) == Some(&MARKER_SOI) {
            return Some(ff);
        }
        pos = ff + 1;
    }
    None
}

/// A label copied from the start of a payload: bytes up to the first NUL,
/// with non-printable bytes escaped, at most [`MAX_LABEL`] characters.
fn label_of(payload: &[u8]) -> Option<String> {
    let end = payload
        .iter()
        .take(MAX_LABEL)
        .position(|&b| b == 0)
        .unwrap_or(payload.len().min(MAX_LABEL));
    if end == 0 {
        return None;
    }
    let mut s = String::with_capacity(end.min(MAX_LABEL));
    for &b in &payload[..end] {
        let printable = (0x20..0x7F).contains(&b) && b != b'\\';
        if s.len() + if printable { 1 } else { 4 } > MAX_LABEL {
            break;
        }
        if printable {
            s.push(b as char);
        } else {
            let _ = core::fmt::Write::write_fmt(&mut s, format_args!("\\x{b:02x}"));
        }
    }
    Some(s)
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*data.get(at)?, *data.get(at + 1)?]))
}

fn le32(data: &[u8], at: usize) -> Option<u32> {
    let b = data.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// End of the entropy-coded data starting at `start`: the next `0xFF` that
/// is not stuffing (`FF 00`) or a restart marker (`FF D0..D7`). Same rule as
/// the container iterator's `skip_entropy_scan`.
fn scan_end(data: &[u8], start: usize, limit: usize) -> usize {
    let mut pos = start;
    while pos < limit {
        let Some(rel) = memchr::memchr(0xFF, &data[pos..limit]) else {
            return limit;
        };
        let ff = pos + rel;
        match data.get(ff + 1) {
            Some(&next) if ff + 1 < limit => {
                if next == 0x00 || (0xD0..=0xD7).contains(&next) {
                    pos = ff + 2;
                } else {
                    return ff;
                }
            }
            _ => return limit,
        }
    }
    limit
}

/// The decoder's verdict on a DQT body (`JpegParser::parse_quant_table`).
fn check_dqt(body: &[u8]) -> Result<(), &'static str> {
    let mut rest = body;
    while let Some((&info, tail)) = rest.split_first() {
        if info >> 4 > 1 {
            return Err("invalid quantization table precision");
        }
        if info & 0x0F >= 4 {
            return Err("quantization table index out of range");
        }
        let n = if info >> 4 == 0 { 64 } else { 128 };
        if tail.len() < n {
            return Err("DQT length mismatch");
        }
        rest = &tail[n..];
    }
    Ok(())
}

/// The decoder's verdict on a DHT body (`JpegParser::parse_huffman_table`),
/// including the code-length check `HuffmanDecodeTable` makes.
fn check_dht(body: &[u8]) -> Result<(), &'static str> {
    use crate::huffman::HuffmanDecodeTable;
    let mut rest = body;
    while let Some((&info, tail)) = rest.split_first() {
        if info >> 4 > 1 {
            return Err("invalid Huffman table class");
        }
        if info & 0x0F >= 4 {
            return Err("Huffman table index out of range");
        }
        let Some(bits) = tail.get(..16) else {
            return Err("DHT length mismatch");
        };
        let mut counts = [0u8; 16];
        counts.copy_from_slice(bits);
        let n: usize = counts.iter().map(|&b| b as usize).sum();
        if n > 256 {
            return Err("Huffman symbol count exceeds 256");
        }
        let Some(values) = tail.get(16..16 + n) else {
            return Err("DHT length mismatch");
        };
        let built = if info >> 4 == 0 {
            HuffmanDecodeTable::from_bits_values(&counts, values).map(|_| ())
        } else {
            HuffmanDecodeTable::from_bits_values_ac(&counts, values).map(|_| ())
        };
        if built.is_err() {
            return Err("invalid Huffman code lengths");
        }
        rest = &tail[16 + n..];
    }
    Ok(())
}

/// The byte ranges of an MPF APP2 body that `parse_mpf_directory`
/// (decode/extras.rs) reads, in body offsets: `MPF\0` and the byte-order
/// mark, the IFD offset, the IFD entry count, the tag of every IFD entry it
/// steps over and the count and offset of the MP Entry tag (B002) — or the
/// window its non-standard-spacing fallback scans — and the 16-byte MP
/// entries. `None` when it finds no MP Entry, so nothing is extracted.
///
/// Arithmetic is checked, so a crafted offset that would overflow a 32-bit
/// `usize` inside `parse_mpf_directory` yields `None` here first; the walker
/// calls `parse_mpf_directory` only when this returns `Some`.
fn mpf_read_ranges(d: &[u8]) -> Option<Vec<Range<usize>>> {
    if d.len() < 12 || !d.starts_with(b"MPF\0") {
        return None;
    }
    let le = &d[4..6] == b"II";
    let r16 = |p: usize| {
        let b = d.get(p..p.checked_add(2)?)?;
        Some(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let r32 = |p: usize| {
        let b = d.get(p..p.checked_add(4)?)?;
        let b = [b[0], b[1], b[2], b[3]];
        Some(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };
    let mut read = vec![0..6, 8..12];
    let ifd = 4usize.checked_add(r32(8)? as usize)?;
    let n = r16(ifd)? as usize;
    read.push(ifd..ifd + 2);
    let mut mp = None;
    for i in 0..n {
        let e = ifd.checked_add(2)?.checked_add(i.checked_mul(12)?)?;
        if d.len() < e.checked_add(12)? {
            break;
        }
        read.push(e..e + 2);
        if r16(e)? == 0xB002 {
            read.push(e + 4..e + 12);
            mp = Some(((r32(e + 8)? as usize).checked_add(4)?, r32(e + 4)? as usize));
            break;
        }
    }
    if mp.is_none() {
        // The fallback reads a two-byte tag candidate at every position of
        // a window of up to 256 bytes, and the type of each candidate.
        let tag: [u8; 2] = if le { [0x02, 0xB0] } else { [0xB0, 0x02] };
        let start = ifd + 2;
        let scan_end = start.saturating_add(256).min(d.len().saturating_sub(12));
        let mut window_end = scan_end + 1;
        for p in start..scan_end {
            if d[p..p + 2] == tag {
                read.push(p + 2..p + 4);
                if r16(p + 2) == Some(7) {
                    read.push(p + 4..p + 12);
                    mp = Some(((r32(p + 8)? as usize).checked_add(4)?, r32(p + 4)? as usize));
                    window_end = p + 2;
                    break;
                }
            }
        }
        if start < scan_end {
            read.push(start..window_end);
        }
    }
    let (off, count) = mp?;
    let images = (count / 16).min(d.len().saturating_sub(off) / 16).min(256);
    if images > 0 {
        read.push(off..off + images * 16);
    }
    Some(read)
}

/// The IFD0 Orientation entry (12 bytes) in an APP1 `Exif\0\0` body.
fn exif_orientation_entry(p: &[u8]) -> Option<Range<usize>> {
    let t = p.get(6..)?;
    let be = match t.get(0..2)? {
        b"MM" => true,
        b"II" => false,
        _ => return None,
    };
    let r16 = |o: usize| {
        let b = t.get(o..o.checked_add(2)?)?;
        Some(if be {
            u16::from_be_bytes([b[0], b[1]])
        } else {
            u16::from_le_bytes([b[0], b[1]])
        })
    };
    let b = t.get(4..8)?;
    let b = [b[0], b[1], b[2], b[3]];
    let ifd = (if be {
        u32::from_be_bytes(b)
    } else {
        u32::from_le_bytes(b)
    }) as usize;
    let n = r16(ifd)? as usize;
    for i in 0..n.min(4096) {
        let e = ifd.checked_add(2 + 12 * i)?;
        t.get(e..e + 12)?;
        if r16(e)? == 0x0112 {
            return Some(6 + e..6 + e + 12);
        }
    }
    None
}

/// ICC bytes past the profile's declared size (its first four bytes):
/// zenjpeg forwards them in `ImageInfo::icc_profile` unchanged, so they stay
/// `Metadata(Icc)`, but as their own child parts so an audit sees the slack.
/// Chunks join in sequence-number order, as `reassemble_icc` joins them.
fn icc_past_declared_size(
    data: &[u8],
    apps: &[App],
    decided: &mut [Decided],
    extra: &mut Vec<(usize, Part)>,
) {
    let mut chunks: Vec<(u8, usize)> = (0..apps.len())
        .filter(|&i| {
            apps[i].ty == SegmentType::Icc
                && matches!(
                    decided[i],
                    Some((Disposition::Metadata(MetadataKind::Icc), _))
                )
        })
        .map(|i| (data[apps[i].payload.start + ICC_SIG_LEN], i))
        .collect();
    chunks.sort_by_key(|c| c.0);
    let profile = |i: usize| apps[i].payload.start + ICC_SIG_LEN + 2..apps[i].payload.end;
    let total: usize = chunks.iter().map(|&(_, i)| profile(i).len()).sum();
    let head: Vec<u8> = chunks
        .iter()
        .flat_map(|&(_, i)| data[profile(i)].iter().copied())
        .take(4)
        .collect();
    let Ok(head) = <[u8; 4]>::try_from(head) else {
        return;
    };
    let declared = u32::from_be_bytes(head) as usize;
    if declared > total {
        if let Some(&(_, i)) = chunks.first()
            && let Some((_, detail)) = decided[i].as_mut()
        {
            *detail = Some(format!(
                "the reassembled profile is {total} bytes; its header declares {declared}"
            ));
        }
        return;
    }
    let mut at = 0usize;
    for &(_, i) in &chunks {
        let r = profile(i);
        let (lo, hi) = (declared.max(at), at + r.len());
        if lo < hi {
            extra.push((
                apps[i].node,
                gap(
                    r.start + (lo - at)..r.start + (hi - at),
                    Disposition::Metadata(MetadataKind::Icc),
                )
                .with_detail(format!(
                    "past the ICC profile's declared size ({declared} bytes); forwarded \
                         unchanged in ImageInfo::icc_profile"
                )),
            ));
        }
        at = hi;
    }
}

/// `(selector byte, byte range)` of each table in a DQT or DHT body the
/// decoder accepted; `base` is the body's file offset.
fn table_ranges(marker: u8, body: &[u8], base: usize) -> Vec<(u8, Range<usize>)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(&info) = body.get(at) {
        let len = if marker == MARKER_DQT {
            1 + if info >> 4 == 0 { 64 } else { 128 }
        } else {
            let Some(bits) = body.get(at + 1..at + 17) else {
                break;
            };
            17 + bits.iter().map(|&b| b as usize).sum::<usize>()
        };
        if at + len > body.len() {
            break;
        }
        out.push((info, base + at..base + at + len));
        at += len;
    }
    out
}

/// Mark what a scan uses: every frame component's quantisation table (the
/// sequential path dequantises during the scan), the restart interval, and
/// the entropy-coding tables or conditioning entries its coding reads. `spec`
/// is the SOS body after the component count: the component selectors, then
/// Ss, Se, Ah/Al.
fn mark_scan_uses(defs: &mut Defs, f: &Frame, spec: &[u8]) {
    defs.mark_quant(f);
    defs.mark(DefKind::Restart);
    let (comps, params) = spec.split_at(spec.len().saturating_sub(3));
    let (Some(&ss), Some(&ahal)) = (params.first(), params.get(2)) else {
        return;
    };
    let first = ahal >> 4 == 0;
    let arithmetic = matches!(f.mode, 0xC9 | 0xCA);
    let (need_dc, need_ac) = match f.mode {
        // Sequential: every block decodes DC then AC.
        0xC0 | 0xC1 | 0xC9 => (true, true),
        // Progressive arithmetic (entropy/arithmetic.rs): DC first scans read
        // the DC conditioning, AC first scans the AC conditioning (Kx);
        // refinement scans use fixed or per-scan statistics only.
        0xCA => (ss == 0 && first, ss > 0 && first),
        // Progressive Huffman: DC first scans use the DC table, DC refinement
        // uses none, AC scans (first and refinement) use the AC table.
        _ => (ss == 0 && first, ss > 0),
    };
    for &[_, tables] in comps.as_chunks::<2>().0 {
        let (td, ta) = (tables >> 4, tables & 0x0F);
        let (dc, ac) = if arithmetic {
            (DefKind::DacDc(td), DefKind::DacAc(ta))
        } else {
            (DefKind::Dc(td), DefKind::Ac(ta))
        };
        if need_dc {
            defs.mark(dc);
        }
        if need_ac {
            defs.mark(ac);
        }
    }
}

/// Offset of the segment `find_exif_orientation` (decode/mod.rs) takes the
/// pixel orientation from. A byte-for-byte copy of its loop, including the
/// way a single fill byte desynchronises it; the `orientation_segment_agrees`
/// test pins the two together.
fn exif_orientation_segment(data: &[u8]) -> Option<usize> {
    const EXIF_PREFIX: &[u8] = b"Exif\0\0";
    if data.len() < 4 {
        return None;
    }
    let mut pos = 2;
    while pos + 4 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];
        if marker == MARKER_SOS || marker == MARKER_EOI {
            break;
        }
        if marker == 0xFF || marker == 0x00 || (0xD0..=0xD7).contains(&marker) {
            pos += 2;
            continue;
        }
        let length = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
        if length < 2 {
            break;
        }
        let seg_start = pos + 4;
        let seg_end = pos + 2 + length;
        if seg_end > data.len() {
            break;
        }
        if marker == 0xE1 && seg_end - seg_start >= EXIF_PREFIX.len() {
            let seg_data = &data[seg_start..seg_end];
            if seg_data.starts_with(EXIF_PREFIX)
                && crate::lossless::parse_exif_orientation(seg_data).is_some()
            {
                return Some(pos);
            }
        }
        pos = seg_end;
    }
    None
}

/// `(sof_name, supported)` for a SOFn marker, as `read_header` treats it.
fn sof_kind(m: u8) -> Option<(&'static str, bool)> {
    Some(match m {
        0xC0 => ("SOF0", true),
        0xC1 => ("SOF1", true),
        0xC2 => ("SOF2", true),
        0xC9 => ("SOF9", true),
        0xCA => ("SOF10", true),
        0xC3 => ("SOF3", false),
        0xC7 => ("SOF7", false),
        0xCB => ("SOF11", false),
        _ => return None,
    })
}

/// A disposition with an optional detail, as resolved for one segment.
type Decided = Option<(Disposition, Option<String>)>;

fn decide(slot: &mut Decided, d: Disposition, detail: Option<String>) {
    let rank = |d: &Disposition| match d {
        Disposition::Metadata(_) => 3,
        Disposition::Structure => 2,
        Disposition::Dropped => 1,
        _ => 0,
    };
    if slot.as_ref().is_none_or(|(old, _)| rank(&d) > rank(old)) {
        *slot = Some((d, detail));
    }
}

impl Walk<'_> {
    fn add(&mut self, parent: Option<usize>, part: Part) -> Result<usize, InventoryError> {
        let parent = parent.map(|p| self.ids[p]);
        let id = self.inv.push(parent, part)?;
        self.ids.push(id);
        Ok(self.ids.len() - 1)
    }

    /// Add a part of a [`Repeat`] kind; only the first one gets `detail`.
    fn add_repeated(
        &mut self,
        parent: Option<usize>,
        part: Part,
        kind: Repeat,
        detail: impl FnOnce() -> String,
    ) -> Result<usize, InventoryError> {
        let first = self.repeats[kind as usize].0.is_none();
        let part = if first {
            part.with_detail(detail())
        } else {
            part
        };
        let node = self.add(parent, part)?;
        let slot = &mut self.repeats[kind as usize];
        if first {
            slot.0 = Some(node);
        } else {
            slot.1 += 1;
        }
        Ok(node)
    }

    fn node(&self, node: usize) -> &Part {
        &self.inv.parts()[node]
    }

    fn set(&mut self, node: usize, d: Disposition, detail: Option<String>) {
        let id = self.ids[node];
        self.inv.set_disposition(id, d);
        if let Some(detail) = detail {
            self.inv.set_detail(id, detail);
        }
    }

    fn append_detail(&mut self, node: usize, more: &str) {
        let detail = match &self.node(node).detail {
            Some(d) => format!("{d}; {more}"),
            None => String::from(more),
        };
        self.inv.set_detail(self.ids[node], detail);
    }

    fn seg_end(&self, pos: usize, limit: usize) -> SegEnd {
        let Some(n) = be16(&self.data[..limit], pos + 2) else {
            return SegEnd::Truncated;
        };
        let n = n as usize;
        if n < 2 {
            return SegEnd::TooShort(pos + 4);
        }
        let end = pos + 2 + n;
        if end > limit {
            SegEnd::Truncated
        } else {
            SegEnd::Ends(end)
        }
    }

    /// Walk one SOI..EOI stream from `start` (which holds `FF D8`) to at most
    /// `limit`, recording every byte up to EOI or `limit`.
    fn walk_stream(
        &mut self,
        parent: Option<usize>,
        start: usize,
        limit: usize,
    ) -> Result<Stream, InventoryError> {
        let data = self.data;
        let first = self.ids.len();
        let mut st = Stream {
            nodes: first..first,
            eoi_end: None,
            sof_end: None,
            frame: None,
            apps: Vec::new(),
            probe_fatal: false,
            decode_fatal: false,
        };
        self.add(
            parent,
            seg(MARKER_SOI, start..start + 2, Disposition::Structure),
        )?;
        let mut pos = start + 2;
        let mut phase = Phase::Header;
        let mut scans = 0u32;
        let mut seen_sos = false;
        let mut height_known = false;
        let mut defs = Defs::default();

        // A container-level failure: the decoder errors at this part. Before
        // the first scan that is certain. After it, the decoder resumes
        // wherever its entropy decoder stopped in the previous scan, which
        // the walker cannot know, so the rest of the stream keeps its
        // dispositions and only this part says it would fail.
        let fatal =
            |w: &mut Self, st: &mut Stream, node: usize, scans: u32, phase: Phase, why: &str| {
                if phase == Phase::Header {
                    st.probe_fatal = true;
                }
                if phase == Phase::Header || scans == 0 {
                    if !st.decode_fatal {
                        w.append_detail(node, &format!("decode fails here: {why}"));
                    }
                    st.decode_fatal = true;
                } else {
                    w.append_detail(
                        node,
                        &format!("the decode fails here if the decoder reaches it: {why}"),
                    );
                }
            };

        while pos < limit && phase != Phase::Done {
            // `JpegParser::read_marker`: skip to the next 0xFF,
            // warning about the bytes in between, then skip fill bytes.
            let next_ff = memchr::memchr(0xFF, &data[pos..limit]).map_or(limit, |r| pos + r);
            if next_ff > pos {
                self.add_repeated(
                    parent,
                    gap(pos..next_ff, Disposition::Malformed),
                    Repeat::Stray,
                    || {
                        String::from(
                            "stray bytes outside any segment; the decoder skips them \
                             (ExtraneousBytesSkipped)",
                        )
                    },
                )?;
                pos = next_ff;
                if pos >= limit {
                    break;
                }
            }
            let fill_start = pos;
            while pos + 1 < limit && data[pos + 1] == 0xFF {
                pos += 1;
            }
            if pos > fill_start {
                self.add_repeated(
                    parent,
                    gap(fill_start..pos, Disposition::Padding),
                    Repeat::Fill,
                    || String::from("fill bytes"),
                )?;
            }
            if pos + 1 >= limit {
                let node = self.add(
                    parent,
                    gap(pos..limit, Disposition::Malformed)
                        .with_detail("truncated: marker prefix at the end of the data"),
                )?;
                if phase == Phase::Header || scans == 0 {
                    fatal(
                        self,
                        &mut st,
                        node,
                        scans,
                        phase,
                        "truncated before any scan",
                    );
                }
                break;
            }
            let m = data[pos + 1];
            if m == 0x00 {
                // `read_marker`: FF 00 outside a scan is skipped like stray bytes.
                self.add_repeated(
                    parent,
                    gap(pos..pos + 2, Disposition::Malformed),
                    Repeat::FfZero,
                    || String::from("FF 00 outside entropy-coded data; the decoder skips it"),
                )?;
                pos += 2;
                continue;
            }

            // Standalone markers.
            if m == MARKER_EOI {
                let node = self.add(parent, seg(m, pos..pos + 2, Disposition::Structure))?;
                if phase == Phase::Header {
                    // `read_header`: EOI before SOF is an error.
                    fatal(
                        self,
                        &mut st,
                        node,
                        scans,
                        phase,
                        "EOI before any frame header",
                    );
                } else if !height_known {
                    // `JpegParser::decode`: EOI with height 0 is an error.
                    fatal(
                        self,
                        &mut st,
                        node,
                        scans,
                        phase,
                        "image height is 0 and no DNL set it",
                    );
                }
                if let Some(f) = st.frame {
                    defs.mark_quant(&f);
                }
                pos += 2;
                st.eoi_end = Some(pos);
                phase = Phase::Done;
                continue;
            }
            if (0xD0..=0xD7).contains(&m) || (m == MARKER_TEM && phase == Phase::Header) {
                // `read_header` ignores RSTn/TEM; `decode` ignores RSTn between scans.
                let (kind, why) = if phase == Phase::Header {
                    (
                        Repeat::HeaderMarker,
                        "standalone marker before the frame header; the decoder ignores it",
                    )
                } else {
                    (
                        Repeat::Restart,
                        "restart marker between scans; the decoder ignores it",
                    )
                };
                self.add_repeated(
                    parent,
                    seg(m, pos..pos + 2, Disposition::Skipped),
                    kind,
                    || String::from(why),
                )?;
                pos += 2;
                continue;
            }
            // SOS: the decoder reads the header by its component count, not
            // by its declared length (`parse_scan`), then the entropy-coded data.
            if m == MARKER_SOS && phase == Phase::Body {
                let hdr_end = data
                    .get(pos + 4)
                    .filter(|_| pos + 4 < limit)
                    .map(|&ns| pos + 2 + 2 + 1 + 2 * ns as usize + 3);
                let Some(hdr_end) = hdr_end.filter(|&e| e <= limit) else {
                    // A cut inside the scan header is recovered as a
                    // truncated scan (`JpegParser::decode`, SOS arm).
                    self.add(
                        parent,
                        seg(m, pos..limit, Disposition::Structure)
                            .with_detail("truncated scan header; the decoder stops here"),
                    )?;
                    break;
                };
                let ns = data[pos + 4];
                let declared = be16(data, pos + 2).unwrap_or(0);
                let mut detail = None;
                if declared as usize != 6 + 2 * ns as usize {
                    detail = Some(format!(
                        "declared length {declared} ignored; the decoder reads {} bytes",
                        6 + 2 * ns as usize
                    ));
                }
                let node = self.add(parent, seg(m, pos..hdr_end, Disposition::Structure))?;
                if let Some(d) = detail {
                    self.append_detail(node, &d);
                }
                if let Some(why) = check_sos(data, pos, ns, st.frame) {
                    fatal(self, &mut st, node, scans, phase, why);
                }
                if let Some(f) = st.frame {
                    mark_scan_uses(&mut defs, &f, &data[pos + 5..hdr_end]);
                }
                let end = scan_end(data, hdr_end, limit);
                if end > hdr_end {
                    let mut scan = part(
                        PartKind::ScanData,
                        PartTag::None,
                        hdr_end..end,
                        Disposition::ImageData,
                    );
                    scan = scan.with_detail(if end == limit {
                        "no marker after the entropy-coded data (truncated; the decoder pads); \
                         bytes after the last MCU are not distinguished"
                    } else {
                        "bytes after the last MCU are not distinguished (that needs a Huffman \
                         decode)"
                    });
                    self.add(parent, scan)?;
                }
                scans += 1;
                seen_sos = true;
                if scans >= MAX_SCANS {
                    fatal(self, &mut st, node, scans, phase, "too many scans");
                }
                pos = end;
                continue;
            }

            // Every other marker carries a length (TEM included after the
            // frame header: `JpegParser::decode` sends it to `skip_segment`).
            let seg_end = match m {
                MARKER_DRI => match be16(&data[..limit], pos + 2) {
                    // `parse_restart_interval`: length + interval are read, then any
                    // excess the length declares is skipped.
                    Some(n) => {
                        let end = pos + 2 + (n as usize).max(4);
                        if end <= limit {
                            SegEnd::Ends(end)
                        } else {
                            SegEnd::Truncated
                        }
                    }
                    None => SegEnd::Truncated,
                },
                MARKER_DQT | MARKER_DHT | MARKER_DAC => match be16(&data[..limit], pos + 2) {
                    // These parse their tables by content: a length below 2
                    // reads no table and resumes right after the length word.
                    Some(n) if n < 2 => SegEnd::Ends(pos + 4),
                    Some(n) if m == MARKER_DAC => {
                        // `parse_dac`: two bytes per table while >= 2 remain.
                        let end = pos + 4 + (n as usize - 2) / 2 * 2;
                        if end <= limit {
                            SegEnd::Ends(end)
                        } else {
                            SegEnd::Truncated
                        }
                    }
                    _ => self.seg_end(pos, limit),
                },
                _ => self.seg_end(pos, limit),
            };
            let end = match seg_end {
                SegEnd::Ends(end) => end,
                SegEnd::TooShort(end) => {
                    let node = self.add(
                        parent,
                        seg(m, pos..end, Disposition::Malformed)
                            .with_detail("segment length below 2"),
                    )?;
                    fatal(
                        self,
                        &mut st,
                        node,
                        scans,
                        phase,
                        "segment length too short",
                    );
                    pos = end;
                    if phase == Phase::Header {
                        phase = Phase::Body;
                    }
                    continue;
                }
                SegEnd::Truncated => {
                    let node = self.add(
                        parent,
                        seg(m, pos..limit, Disposition::Malformed)
                            .with_detail("truncated: the declared segment runs past the data"),
                    )?;
                    // Between scans a cut is recovered (`JpegParser::decode`).
                    if phase == Phase::Header || scans == 0 {
                        fatal(
                            self,
                            &mut st,
                            node,
                            scans,
                            phase,
                            "truncated before any scan",
                        );
                    }
                    break;
                }
            };
            let payload = (pos + 4).min(end)..end;
            let body = &data[payload.clone()];

            match m {
                0xE0..=0xEF | MARKER_COM => {
                    let ty = detect_segment_type(m, body);
                    let mut p = seg(m, pos..end, Disposition::Unknown);
                    if let Some(l) = label_of(body) {
                        p = p.with_label(l);
                    }
                    let node = self.add(parent, p)?;
                    if ty == SegmentType::Jfif {
                        self.jfif_children(node, &payload)?;
                    }
                    st.apps.push(App {
                        node,
                        marker: m,
                        payload,
                        ty,
                        before_sof: phase == Phase::Header,
                        before_sos: !seen_sos,
                    });
                }
                MARKER_DQT | MARKER_DHT => {
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    let verdict = if m == MARKER_DQT {
                        check_dqt(body)
                    } else {
                        check_dht(body)
                    };
                    if let Err(why) = verdict {
                        self.set(node, Disposition::Malformed, None);
                        fatal(self, &mut st, node, scans, phase, why);
                    } else {
                        // Each table replaces the one in the same slot.
                        for (info, r) in table_ranges(m, body, payload.start) {
                            let i = info & 0x0F;
                            let what = if m == MARKER_DQT {
                                DefKind::Quant(i)
                            } else if info >> 4 == 0 {
                                DefKind::Dc(i)
                            } else {
                                DefKind::Ac(i)
                            };
                            defs.define(node, r, what);
                        }
                    }
                }
                MARKER_DAC => {
                    // `parse_dac`: two bytes per entry; class 0 sets a DC
                    // table's (L, U), any other class an AC table's Kx. Each
                    // entry replaces the one for the same table.
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    for (k, &[info, cs]) in body.as_chunks::<2>().0.iter().enumerate() {
                        let idx = info & 0x0F;
                        if idx >= 4 || (info >> 4 == 0 && cs & 0x0F > cs >> 4) {
                            self.set(node, Disposition::Malformed, None);
                            fatal(
                                self,
                                &mut st,
                                node,
                                scans,
                                phase,
                                "invalid DAC conditioning table",
                            );
                            break;
                        }
                        let what = if info >> 4 == 0 {
                            DefKind::DacDc(idx)
                        } else {
                            DefKind::DacAc(idx)
                        };
                        let at = payload.start + 2 * k;
                        defs.define(node, at..at + 2, what);
                    }
                }
                MARKER_DRI => {
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    defs.define(node, pos + 4..pos + 6, DefKind::Restart);
                    if end > pos + 6 {
                        // `parse_restart_interval` skips what a length above 4 declares.
                        self.add(
                            Some(node),
                            gap(pos + 6..end, Disposition::Dropped).with_detail(
                                "bytes after the restart interval; the decoder skips them",
                            ),
                        )?;
                    }
                }
                MARKER_DNL if phase == Phase::Body => {
                    // `parse_dnl`: the length must be 4.
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    if end - pos != 6 {
                        self.set(node, Disposition::Malformed, None);
                        fatal(self, &mut st, node, scans, phase, "DNL length is not 4");
                    } else if height_known {
                        // `parse_dnl` only sets the height when SOF left it 0.
                        self.set(
                            node,
                            Disposition::Dropped,
                            Some(
                                "the frame header already set the height; the decoder ignores it"
                                    .into(),
                            ),
                        );
                    } else if be16(data, pos + 4).is_some_and(|h| h > 0) {
                        height_known = true;
                    }
                }
                _ if phase == Phase::Header && sof_kind(m).is_some() => {
                    let (name, supported) = sof_kind(m).unwrap_or(("SOF", false));
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    st.sof_end = Some(end);
                    phase = Phase::Body;
                    if !supported {
                        // `read_header` rejects SOF3/SOF7/SOF11.
                        self.set(
                            node,
                            Disposition::Skipped,
                            Some(format!("lossless JPEG ({name}) is not supported")),
                        );
                        fatal(
                            self,
                            &mut st,
                            node,
                            scans,
                            Phase::Header,
                            "unsupported frame type",
                        );
                    } else {
                        match check_sof(body, self.opts.max_pixels) {
                            Err(why) => {
                                self.set(node, Disposition::Malformed, None);
                                fatal(self, &mut st, node, scans, Phase::Header, why);
                            }
                            Ok(f) => {
                                st.frame = Some(Frame { mode: m, ..f.frame });
                                height_known = f.height > 0;
                                if f.height == 0 {
                                    // `read_info` rejects DNL mode
                                    // and every scan decoder does too.
                                    fatal(
                                        self,
                                        &mut st,
                                        node,
                                        scans,
                                        Phase::Header,
                                        "DNL mode (height 0) is not supported",
                                    );
                                } else if f.precision != 8 {
                                    // `JpegParser::decode`: probe() reports the
                                    // header, decode() refuses the precision.
                                    fatal(
                                        self,
                                        &mut st,
                                        node,
                                        scans,
                                        Phase::Body,
                                        "12-bit precision is not supported",
                                    );
                                }
                            }
                        }
                    }
                }
                _ => {
                    // `skip_segment`: the decoder reads the
                    // length and skips the body.
                    let why = if m == MARKER_TEM {
                        "TEM is a standalone marker, but after the frame header the decoder \
                         reads the next two bytes as a length and skips that many bytes"
                    } else if m == MARKER_SOI {
                        "SOI inside the stream; the decoder reads a length and skips it"
                    } else if m == MARKER_SOS {
                        "scan header before any frame header; the decoder skips the header, \
                         and its entropy-coded data reads as stray bytes"
                    } else if phase == Phase::Header && matches!(m, 0xC5 | 0xC6 | 0xCD..=0xCF) {
                        "differential/hierarchical frame header; the decoder skips it"
                    } else if (phase == Phase::Body && sof_kind(m).is_some())
                        || matches!(m, 0xC5 | 0xC6 | 0xCD..=0xCF)
                    {
                        "frame header after the first; the decoder skips it"
                    } else if m == MARKER_DNL {
                        "DNL before any frame header; the decoder skips it"
                    } else {
                        "marker the decoder skips by its length"
                    };
                    self.add(
                        parent,
                        seg(m, pos..end, Disposition::Skipped).with_detail(why),
                    )?;
                }
            }
            pos = end;
        }
        if st.eoi_end.is_none() && phase == Phase::Header && !st.probe_fatal {
            // The data ended before a frame header.
            st.probe_fatal = true;
            st.decode_fatal = true;
        }
        if st.eoi_end.is_none() && phase == Phase::Body && scans == 0 && !st.decode_fatal {
            st.decode_fatal = true;
        }
        self.settle_defs(&mut defs)?;
        st.nodes = first..self.ids.len();
        Ok(st)
    }

    /// Definitions no scan used: overwritten before use, never referenced,
    /// or of a kind the frame's coding does not use. A segment whose every
    /// definition is unused is `Dropped`; one that mixes both gets a child
    /// per used definition and one per run of unused ones.
    fn settle_defs(&mut self, defs: &mut Defs) -> Result<(), InventoryError> {
        const WHY: &str = "no scan uses it before it is redefined, or nothing refers to it";
        defs.used.sort_by_key(|d| (d.node, d.range.start));
        let mut u = 0;
        for s in &defs.segs {
            while u < defs.used.len() && defs.used[u].node < s.node {
                u += 1;
            }
            let from = u;
            while u < defs.used.len() && defs.used[u].node == s.node {
                u += 1;
            }
            let used = &defs.used[from..u];
            if self.node(s.node).disposition != Disposition::Structure
                || used.len() == s.count as usize
            {
                continue;
            }
            if used.is_empty() {
                let what = if s.count == 1 {
                    format!("{}", s.first)
                } else {
                    format!("{} definitions, the first {}", s.count, s.first)
                };
                self.set(s.node, Disposition::Dropped, Some(format!("{what}: {WHY}")));
                continue;
            }
            let mut at = s.area.start;
            for d in used {
                if d.range.start > at {
                    self.add(
                        Some(s.node),
                        part(
                            PartKind::Attribute,
                            PartTag::None,
                            at..d.range.start,
                            Disposition::Dropped,
                        )
                        .with_detail(format!("unused definitions: {WHY}")),
                    )?;
                }
                let tag = PartTag::Code(u32::from(self.data[d.range.start]));
                self.add(
                    Some(s.node),
                    part(
                        PartKind::Attribute,
                        tag,
                        d.range.clone(),
                        Disposition::Structure,
                    )
                    .with_detail(format!("{}", d.what)),
                )?;
                at = d.range.end;
            }
            if at < s.area.end {
                self.add(
                    Some(s.node),
                    part(
                        PartKind::Attribute,
                        PartTag::None,
                        at..s.area.end,
                        Disposition::Dropped,
                    )
                    .with_detail(format!("unused definitions: {WHY}")),
                )?;
            }
        }
        Ok(())
    }

    /// The parts of a JFIF APP0 the decoder never reads: it reads the
    /// signature, version, units and densities (12 bytes, `parse_jfif`), not
    /// the thumbnail size, the uncompressed RGB thumbnail or anything after.
    fn jfif_children(&mut self, node: usize, payload: &Range<usize>) -> Result<(), InventoryError> {
        let body = &self.data[payload.clone()];
        if body.len() <= 12 {
            return Ok(());
        }
        let base = payload.start;
        let mut tail = 12;
        if let (Some(&w), Some(&h)) = (body.get(12), body.get(13)) {
            self.add(
                Some(node),
                part(
                    PartKind::Attribute,
                    PartTag::None,
                    base + 12..base + 14,
                    Disposition::Skipped,
                )
                .with_detail("JFIF thumbnail size; the decoder does not read it"),
            )?;
            tail = 14;
            let n = 3 * usize::from(w) * usize::from(h);
            if n > 0 && 14 + n <= body.len() {
                self.add(
                    Some(node),
                    part(
                        PartKind::EmbeddedImage,
                        PartTag::None,
                        base + 14..base + 14 + n,
                        Disposition::Skipped,
                    )
                    .with_detail(format!(
                        "{w}x{h} RGB JFIF thumbnail; the decoder does not read it"
                    )),
                )?;
                tail = 14 + n;
            }
        }
        if tail < body.len() {
            self.add(
                Some(node),
                gap(base + tail..payload.end, Disposition::Unreferenced)
                    .with_detail("bytes after the JFIF fields and thumbnail"),
            )?;
        }
        Ok(())
    }

    /// Settle every APPn/COM disposition of `st`, and demote the stream's
    /// structure and image data when the decode fails.
    fn resolve(&mut self, st: &Stream, view: View) -> Result<(), InventoryError> {
        let probe_ok = view.probe && !st.probe_fatal;
        let decode_ok = !st.decode_fatal;
        let mut decided: Vec<Decided> = vec![None; st.apps.len()];
        // Child parts for bytes inside consumed segments that nothing reads.
        let mut extra: Vec<(usize, Part)> = Vec::new();

        // The metadata both `probe()` and `decode()` report, each over the
        // segments it reads: `probe()` the ones before the first frame header.
        let views = [(probe_ok, true), (decode_ok, false)];
        for (active, header_only) in views {
            if !active {
                continue;
            }
            let sel: Vec<usize> = (0..st.apps.len())
                .filter(|&i| !header_only || st.apps[i].before_sof)
                .collect();
            self.resolve_view(st, &sel, &mut decided);
        }

        let data = self.data;
        // `probe()` falls back to `extract_icc_profile` (color/icc.rs), which
        // reads chunks up to the first SOS, when no chunk precedes the frame
        // header (decode/parser/mod.rs `info`).
        if probe_ok
            && !st
                .apps
                .iter()
                .any(|a| a.ty == SegmentType::Icc && a.before_sof)
            && crate::color::icc::extract_icc_profile(data).is_some()
        {
            for (i, a) in st.apps.iter().enumerate() {
                if a.ty == SegmentType::Icc && a.before_sos && a.payload.len() >= ICC_SIG_LEN + 2 {
                    decide(
                        &mut decided[i],
                        Disposition::Metadata(MetadataKind::Icc),
                        None,
                    );
                }
            }
        }
        if decode_ok {
            // APP14 Adobe: every one is parsed, the last one parsed sets the
            // colour transform (`process_app_or_com`).
            let adobe: Vec<usize> = (0..st.apps.len())
                .filter(|&i| {
                    st.apps[i].marker == MARKER_APP14 && st.apps[i].ty == SegmentType::Adobe
                })
                .collect();
            let valid: Vec<usize> = adobe
                .iter()
                .copied()
                .filter(|&i| st.apps[i].payload.len() >= 12)
                .collect();
            let components = st.frame.map_or(0, |f| f.components);
            for &i in &adobe {
                let a = &st.apps[i];
                if a.payload.len() < 12 {
                    decide(
                        &mut decided[i],
                        Disposition::Dropped,
                        Some("too short to carry a colour transform".into()),
                    );
                } else if Some(&i) != valid.last() {
                    decide(
                        &mut decided[i],
                        Disposition::Dropped,
                        Some("superseded by a later APP14 Adobe segment".into()),
                    );
                } else if (3..=4).contains(&components) {
                    let transform = data[a.payload.start + 11];
                    if a.payload.len() > 12 {
                        extra.push((
                            a.node,
                            gap(a.payload.start + 12..a.payload.end, Disposition::Unreferenced)
                                .with_detail("bytes after the APP14 fields; the decoder reads only the transform"),
                        ));
                    }
                    decide(
                        &mut decided[i],
                        Disposition::Metadata(MetadataKind::Colour),
                        Some(format!(
                            "colour transform {transform} applied to the pixels"
                        )),
                    );
                } else {
                    decide(
                        &mut decided[i],
                        Disposition::Dropped,
                        Some(format!(
                            "colour transform unused for a {components}-component frame"
                        )),
                    );
                }
            }
            // MPF: the first segment is the index the decoder follows.
            if let Some(i) = (0..st.apps.len()).find(|&i| st.apps[i].ty == SegmentType::Mpf) {
                if view.embedded {
                    decide(
                        &mut decided[i],
                        Disposition::Skipped,
                        Some("MPF index inside an embedded image is not followed".into()),
                    );
                } else {
                    let a = &st.apps[i];
                    match mpf_read_ranges(&data[a.payload.clone()]) {
                        None => decide(
                            &mut decided[i],
                            Disposition::Dropped,
                            Some("the MPF index does not parse; no image is extracted".into()),
                        ),
                        Some(mut read) => {
                            // Every hole between what the parser reads.
                            read.sort_by_key(|r| r.start);
                            let mut at = 0;
                            let len = a.payload.len();
                            for r in read.iter().chain(core::iter::once(&(len..len))) {
                                if r.start > at {
                                    extra.push((
                                        a.node,
                                        gap(
                                            a.payload.start + at..a.payload.start + r.start,
                                            Disposition::Unreferenced,
                                        )
                                        .with_detail("bytes the MPF index parser does not read"),
                                    ));
                                }
                                at = at.max(r.end);
                            }
                            decide(&mut decided[i], Disposition::Structure, None);
                        }
                    }
                }
            }
        }

        // Pixel orientation (OrientationHint::Correct*): find_exif_orientation
        // runs its own pre-SOS walk and takes the first EXIF segment that
        // carries an orientation tag.
        let mut orientation_from_inside = None;
        if decode_ok
            && view.auto_orient
            && let Some(at) = exif_orientation_segment(data)
        {
            match st.apps.iter().position(|a| a.payload.start == at + 4) {
                Some(i) => {
                    if !matches!(
                        decided[i],
                        Some((Disposition::Metadata(MetadataKind::Exif), _))
                    ) {
                        let a = &st.apps[i];
                        match exif_orientation_entry(&data[a.payload.clone()]) {
                            Some(r) => {
                                // Only the orientation entry is consumed.
                                decided[i] = Some((
                                    Disposition::Skipped,
                                    Some(
                                        "only its orientation entry is read (auto-orient); \
                                         ImageInfo::exif carries the first EXIF segment"
                                            .into(),
                                    ),
                                ));
                                extra.push((
                                    a.node,
                                    part(
                                        PartKind::Field,
                                        PartTag::Code(0x0112),
                                        a.payload.start + r.start..a.payload.start + r.end,
                                        Disposition::Metadata(MetadataKind::Orientation),
                                    )
                                    .with_detail("EXIF orientation entry, applied to the pixels"),
                                ));
                            }
                            None => {
                                decided[i] = Some((
                                    Disposition::Metadata(MetadataKind::Orientation),
                                    Some(
                                        "its orientation is applied to the pixels; \
                                         ImageInfo::exif carries the first EXIF segment"
                                            .into(),
                                    ),
                                ));
                            }
                        }
                    }
                }
                None => orientation_from_inside = Some(at),
            }
        }

        // The reverse case: the reported EXIF carries an orientation that the
        // pixel-orientation walk never reaches (it desynchronises on a single
        // fill byte, and stops at the first SOS).
        if decode_ok && view.auto_orient {
            let found = exif_orientation_segment(data);
            for (i, a) in st.apps.iter().enumerate() {
                if matches!(
                    decided[i],
                    Some((Disposition::Metadata(MetadataKind::Exif), _))
                ) && crate::lossless::parse_exif_orientation(&data[a.payload.clone()]).is_some()
                    && found != Some(a.payload.start - 4)
                    && let Some((_, detail)) = decided[i].as_mut()
                {
                    *detail = Some(String::from(
                        "its orientation is neither applied to the pixels nor reported: \
                         find_exif_orientation does not reach this segment",
                    ));
                }
            }
        }

        icc_past_declared_size(data, &st.apps, &mut decided, &mut extra);

        // Extended XMP: `reassemble_xmp` reads each chunk's offset (its sort
        // key) and data, never the GUID or the full length before them.
        for (i, a) in st.apps.iter().enumerate() {
            if a.ty == SegmentType::XmpExtended
                && matches!(
                    decided[i],
                    Some((Disposition::Metadata(MetadataKind::Xmp), _))
                )
            {
                let at = a.payload.start + XMP_EXT_NS_LEN;
                extra.push((
                    a.node,
                    part(
                        PartKind::Attribute,
                        PartTag::None,
                        at..at + 36,
                        Disposition::Dropped,
                    )
                    .with_detail(
                        "extended-XMP GUID and full length; reassemble_xmp reads only each \
                         chunk's offset and data",
                    ),
                ));
            }
        }

        for (i, a) in st.apps.iter().enumerate() {
            let (d, detail) = match decided[i].take() {
                Some(x) => x,
                None => self.default_disposition(a, decode_ok),
            };
            self.set(a.node, d, detail);
        }

        if let Some(at) = orientation_from_inside {
            // The orientation walk desynchronised (a single fill byte) and
            // found an APP1 EXIF header inside another part.
            if let Some(node) = (st.nodes.clone()).rev().find(|&n| {
                self.node(n).range.start <= at as u64 && (at as u64) < self.node(n).range.end
            }) {
                self.set(node, Disposition::Metadata(MetadataKind::Orientation), None);
                self.append_detail(
                    node,
                    &format!(
                        "find_exif_orientation reads an EXIF orientation from bytes at offset {at} \
                         inside this part"
                    ),
                );
            }
        }

        if !decode_ok {
            // Nothing reaches the caller from a failed decode, except what
            // `probe()` reads before the frame header.
            let keep_until = if probe_ok { st.sof_end.unwrap_or(0) } else { 0 };
            for n in st.nodes.clone() {
                let p = self.node(n);
                if matches!(
                    p.disposition,
                    Disposition::Structure | Disposition::ImageData
                ) && p.range.end > keep_until as u64
                {
                    self.inv.set_disposition(self.ids[n], Disposition::Skipped);
                }
            }
        }
        for (node, p) in extra {
            self.add(Some(node), p)?;
        }
        Ok(())
    }

    /// EXIF, XMP, ICC and JFIF as one consumer (`probe()` or `decode()`)
    /// reports them from the segments in `sel`.
    fn resolve_view(&self, st: &Stream, sel: &[usize], decided: &mut [Decided]) {
        let data = self.data;
        let apps = &st.apps;
        let of_type = |t: SegmentType| sel.iter().copied().filter(move |&i| apps[i].ty == t);
        let extras_of = |types: &[SegmentType]| {
            let mut ex = DecodedExtras::new();
            for &i in sel {
                let a = &apps[i];
                if types.contains(&a.ty) {
                    ex.add_segment(a.marker, data[a.payload.clone()].to_vec(), a.ty);
                }
            }
            ex
        };

        // EXIF: the first segment (extras.rs `exif`).
        if let Some(i) = of_type(SegmentType::Exif).next() {
            decide(
                &mut decided[i],
                Disposition::Metadata(MetadataKind::Exif),
                None,
            );
        }

        // XMP: the first packet plus every extended chunk (extras.rs
        // `reassemble_xmp`), all or nothing on UTF-8 validity.
        let primary = of_type(SegmentType::Xmp).next();
        let xmp_ok = primary.is_some()
            && extras_of(&[SegmentType::Xmp, SegmentType::XmpExtended])
                .xmp()
                .is_some();
        if let Some(i) = primary {
            if xmp_ok {
                decide(
                    &mut decided[i],
                    Disposition::Metadata(MetadataKind::Xmp),
                    None,
                );
            } else {
                decide(
                    &mut decided[i],
                    Disposition::Dropped,
                    Some("XMP is not valid UTF-8; nothing is reported".into()),
                );
            }
        }
        for i in of_type(SegmentType::XmpExtended) {
            let (d, why) = if primary.is_none() {
                (
                    Disposition::Dropped,
                    Some("extended XMP without a standard XMP packet is ignored"),
                )
            } else if apps[i].payload.len() < XMP_EXT_NS_LEN + 40 {
                (
                    Disposition::Dropped,
                    Some("too short for the extended-XMP header"),
                )
            } else if xmp_ok {
                (Disposition::Metadata(MetadataKind::Xmp), None)
            } else {
                (
                    Disposition::Dropped,
                    Some("XMP is not valid UTF-8; nothing is reported"),
                )
            };
            decide(&mut decided[i], d, why.map(String::from));
        }

        // ICC: every chunk, reassembled (extras.rs `reassemble_icc`).
        let icc_ok = extras_of(&[SegmentType::Icc]).icc_profile().is_some();
        for i in of_type(SegmentType::Icc) {
            if apps[i].payload.len() < ICC_SIG_LEN + 2 {
                decide(
                    &mut decided[i],
                    Disposition::Dropped,
                    Some("too short for the ICC chunk header".into()),
                );
            } else if icc_ok {
                decide(
                    &mut decided[i],
                    Disposition::Metadata(MetadataKind::Icc),
                    None,
                );
            } else {
                decide(
                    &mut decided[i],
                    Disposition::Dropped,
                    Some("reassembled profile exceeds MAX_ICC_PROFILE_SIZE; not reported".into()),
                );
            }
        }

        // JFIF: the first segment's density (codec/info.rs `jfif_to_resolution`).
        if let Some(i) = of_type(SegmentType::Jfif).next() {
            let jfif = extras_of(&[SegmentType::Jfif]).jfif();
            match jfif.as_ref().and_then(super::info::jfif_to_resolution) {
                Some(_) => decide(
                    &mut decided[i],
                    Disposition::Metadata(MetadataKind::Resolution),
                    None,
                ),
                None => decide(
                    &mut decided[i],
                    Disposition::Dropped,
                    Some(
                        if jfif.is_some() {
                            "density is aspect-ratio only or zero; no resolution is reported"
                        } else {
                            "too short to parse"
                        }
                        .into(),
                    ),
                ),
            }
        }
    }

    /// The disposition of a segment no consumer reports.
    fn default_disposition(&self, a: &App, decode_ok: bool) -> (Disposition, Option<String>) {
        let body = &self.data[a.payload.clone()];
        let not_reached = || String::from("not reported: the decode fails");
        let (d, why): (Disposition, String) = match a.ty {
            SegmentType::Unknown => {
                return if a.marker == 0xE2 && body.starts_with(ISO_21496_1) {
                    (
                        Disposition::Unknown,
                        Some("ISO 21496-1 gain-map metadata; the decoder does not parse it".into()),
                    )
                } else {
                    (Disposition::Unknown, None)
                };
            }
            SegmentType::Comment => (
                Disposition::Skipped,
                "native DecodedExtras::comments only".into(),
            ),
            SegmentType::Iptc => (
                Disposition::Skipped,
                "native DecodedExtras::iptc only".into(),
            ),
            _ if !decode_ok => (Disposition::Skipped, not_reached()),
            SegmentType::Exif => (
                Disposition::Skipped,
                "additional EXIF segment; only the first is reported".into(),
            ),
            SegmentType::Xmp => (
                Disposition::Skipped,
                "additional XMP packet; only the first is reported".into(),
            ),
            SegmentType::Jfif => (
                Disposition::Skipped,
                "additional JFIF segment; only the first is read".into(),
            ),
            SegmentType::Mpf => (
                Disposition::Skipped,
                "additional MPF segment; only the first is read".into(),
            ),
            _ => (Disposition::Skipped, not_reached()),
        };
        (d, Some(why))
    }

    /// Map an embedded image's resolved parts to what its role makes of them.
    fn apply_role(&mut self, nodes: Range<usize>, role: Role) {
        for n in nodes {
            let d = self.node(n).disposition;
            let (new, detail) = match (role, d) {
                (Role::GainMap, Disposition::Structure | Disposition::ImageData) => {
                    (Disposition::Metadata(MetadataKind::GainMap), None)
                }
                (Role::GainMap, Disposition::Metadata(MetadataKind::Colour)) => {
                    (Disposition::Metadata(MetadataKind::GainMap), None)
                }
                (Role::GainMap, Disposition::Metadata(MetadataKind::Xmp)) => (
                    Disposition::Metadata(MetadataKind::GainMap),
                    Some("read for hdrgm parameters when the primary XMP carries none"),
                ),
                (Role::GainMap, Disposition::Metadata(_)) => (
                    Disposition::Dropped,
                    Some("the gain-map decode keeps only pixels"),
                ),
                (Role::Unread, d) if d.is_consumed() => (Disposition::Skipped, None),
                _ => continue,
            };
            self.set(n, new, detail.map(String::from));
        }
    }

    /// Add an embedded image at `r` and walk it when it starts with SOI.
    fn embedded(
        &mut self,
        mut p: Part,
        r: Range<usize>,
        role: Role,
    ) -> Result<usize, InventoryError> {
        let is_jpeg = self.data.get(r.start..r.start + 2) == Some(&[0xFF, MARKER_SOI][..]);
        if is_jpeg {
            p = p.with_body(r.start as u64..r.end as u64);
        }
        let node = self.add(None, p)?;
        if is_jpeg {
            let st = self.walk_stream(Some(node), r.start, r.end)?;
            self.resolve(
                &st,
                View {
                    probe: false,
                    embedded: true,
                    auto_orient: false,
                },
            )?;
            // Every part of this stream, including children `resolve` added.
            let all = st.nodes.start..self.ids.len();
            self.apply_role(all, role);
        }
        Ok(node)
    }

    /// Everything after the primary EOI: the Samsung SEF trailer, MPF images
    /// and GContainer items, in that order of precedence.
    fn after_eoi(&mut self, st: &Stream, eoi_end: usize) -> Result<(), InventoryError> {
        let data = self.data;
        let len = data.len();
        let mut placed: Vec<Range<usize>> = Vec::new();
        let free = |placed: &[Range<usize>], r: &Range<usize>| {
            r.start >= eoi_end
                && r.end <= len
                && r.start < r.end
                && placed.iter().all(|p| r.end <= p.start || r.start >= p.end)
        };
        let decode_ok = !st.decode_fatal;

        // Samsung SEF trailer (the layout ExifTool's Samsung module reads).
        let mut seft_node = None;
        if let Some(t) = samsung_trailer(data, eoi_end) {
            let node = self.add(
                None,
                part(
                    PartKind::Trailer,
                    PartTag::None,
                    t.range.clone(),
                    Disposition::Skipped,
                )
                .with_label("SEFT")
                .with_detail(format!(
                    "Samsung SEF trailer, {} entries; the decoder does not read it",
                    t.count
                )),
            )?;
            for (r, ty, name) in t.blocks {
                let mut p = part(
                    PartKind::Chunk,
                    PartTag::Code(u32::from(ty)),
                    r,
                    Disposition::Skipped,
                );
                if let Some(l) = name {
                    p = p.with_label(l);
                }
                self.add(Some(node), p)?;
            }
            self.add(
                Some(node),
                part(
                    PartKind::Chunk,
                    PartTag::FourCc(*b"SEFH"),
                    t.directory,
                    Disposition::Skipped,
                )
                .with_label("SEFH"),
            )?;
            placed.push(t.range);
            seft_node = Some(node);
        }

        // MPF secondary images (`extract_mpf_secondary_images`): offsets are
        // relative to the TIFF header after the first MPF segment's `MPF\0`;
        // offset 0 means "right after the primary EOI".
        let mpf = st.apps.iter().find(|a| a.ty == SegmentType::Mpf);
        let mut mpf_images: Vec<(Range<usize>, usize)> = Vec::new();
        let mut gain_map_taken = false;
        if let Some(mpf) = mpf
            && mpf_read_ranges(&data[mpf.payload.clone()]).is_some()
            && let Some(dir) = parse_mpf_directory(&data[mpf.payload.clone()])
        {
            let base = mpf.payload.start + 4;
            let mut notes: Vec<String> = Vec::new();
            for (idx, entry) in dir.images.iter().enumerate() {
                if idx == 0 || matches!(entry.image_type, MpfImageType::BaselinePrimary) {
                    continue;
                }
                let start = if entry.offset == 0 {
                    Some(eoi_end)
                } else {
                    base.checked_add(entry.offset as usize)
                };
                let r = start.and_then(|s| Some(s..s.checked_add(entry.size as usize)?));
                let Some(r) = r.filter(|r| free(&placed, r)) else {
                    notes.push(format!(
                        "entry {idx} ({} bytes at relative offset {}) lies outside the data after \
                         EOI or overlaps another part",
                        entry.size, entry.offset
                    ));
                    continue;
                };
                let is_jpeg = data.get(r.start..r.start + 2) == Some(&[0xFF, MARKER_SOI][..]);
                let extracted = decode_ok && is_jpeg;
                let type_code = entry.image_type.type_code();
                let is_gain_map = extracted && entry.image_type == MpfImageType::Undefined;
                let role = if is_gain_map && !gain_map_taken && self.opts.gain_map_decoded {
                    Role::GainMap
                } else {
                    Role::Unread
                };
                gain_map_taken |= is_gain_map;
                let (d, why) = match (role, extracted) {
                    (Role::GainMap, _) => (
                        Disposition::Metadata(MetadataKind::GainMap),
                        String::from("decoded as the Ultra HDR gain map"),
                    ),
                    (_, true) => (
                        Disposition::Skipped,
                        String::from("copied to native DecodedExtras::secondary_images only"),
                    ),
                    (_, false) if !is_jpeg => (
                        Disposition::Skipped,
                        String::from("does not start with SOI; the decoder ignores it"),
                    ),
                    _ => (
                        Disposition::Skipped,
                        String::from("not read: the decode fails"),
                    ),
                };
                let p = part(
                    PartKind::EmbeddedImage,
                    PartTag::Code(idx as u32),
                    r.clone(),
                    d,
                )
                .with_label("MPF")
                .with_detail(format!("MPF image {idx}, type {type_code:#08x}; {why}"));
                let node = self.embedded(p, r.clone(), role)?;
                mpf_images.push((r.clone(), node));
                placed.push(r);
            }
            if !notes.is_empty() {
                self.append_detail(mpf.node, &notes.join("; "));
            }
        }

        // GContainer items named by the primary XMP packet
        // (`Container:Directory`): packed in order after the primary image
        // plus its `Item:Padding`. Only the standard packet is read, and only
        // when it names a bounded number of list items:
        // `parse_container_items` rescans the text per item.
        let xmp = st
            .apps
            .iter()
            .find(|a| a.ty == SegmentType::Xmp)
            .and_then(|a| {
                let body = &data[a.payload.clone()];
                core::str::from_utf8(body.get(XMP_NS_LEN..)?).ok()
            })
            .filter(|x| x.contains("Container:Directory"))
            .filter(|x| x.matches("rdf:li").take(MAX_CONTAINER_LI + 1).count() <= MAX_CONTAINER_LI);
        let items = xmp
            .map(crate::container::types::parse_container_items)
            .unwrap_or_default();
        if items.len() >= 2 {
            // Items are packed, so they cannot overlap one another; check
            // them against the trailer and MPF images only.
            let placed = placed.clone();
            let mut cursor = eoi_end;
            if let Some(pad) = items[0].padding.filter(|&p| p > 0) {
                let r = cursor..cursor.saturating_add(pad);
                if free(&placed, &r) {
                    self.add(
                        None,
                        gap(r.clone(), Disposition::Padding)
                            .with_label("Item:Padding")
                            .with_detail("GContainer padding after the primary image"),
                    )?;
                }
                cursor = cursor.saturating_add(pad);
            }
            for item in &items[1..] {
                let Some(length) = item.length.filter(|&l| l > 0) else {
                    // Length 0: the item shares the previous item's bytes.
                    continue;
                };
                let r = cursor..cursor.saturating_add(length);
                cursor = r.end;
                let semantic = String::from(item.semantic.as_xmp_str());
                if let Some(&(_, n)) = mpf_images.iter().find(|(m, _)| *m == r) {
                    self.append_detail(
                        n,
                        &format!("also GContainer item {semantic} ({})", item.mime),
                    );
                    continue;
                }
                if !free(&placed, &r) {
                    if let Some(node) = seft_node.filter(|_| r.end <= len && r.start >= eoi_end) {
                        self.append_detail(
                            node,
                            &format!(
                                "GContainer item {semantic} ({}) at {}..{} overlaps the trailer",
                                item.mime, r.start, r.end
                            ),
                        );
                    }
                    break;
                }
                let detail = format!(
                    "GContainer {semantic} item, {}; not read by the zencodec path{}",
                    item.mime,
                    if matches!(semantic.as_str(), "DepthMap" | "ConfidenceMap" | "Depth") {
                        " (native DecodedExtras::extract_depth_map(Some(file)) reads it)"
                    } else {
                        ""
                    }
                );
                let p = part(
                    PartKind::EmbeddedImage,
                    PartTag::None,
                    r.clone(),
                    Disposition::Skipped,
                )
                .with_label(semantic)
                .with_detail(detail);
                if item.mime == "image/jpeg" {
                    self.embedded(p, r.clone(), Role::Unread)?;
                } else {
                    self.add(None, p)?;
                }
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Inventory, InventoryError> {
        for (first, more) in self.repeats {
            if let Some(first) = first
                && more > 0
            {
                self.append_detail(
                    first,
                    &format!("{more} more parts like this one carry no detail"),
                );
            }
        }
        let bodies: Vec<PartId> = self
            .ids
            .iter()
            .copied()
            .filter(|id| self.inv.parts()[id.index()].body.is_some())
            .collect();
        for id in bodies {
            self.inv.fill_gaps(Some(id), Disposition::Unreferenced)?;
        }
        self.inv.fill_gaps(None, Disposition::Trailing)?;
        Ok(self.inv)
    }
}

struct SofInfo {
    frame: Frame,
    height: u16,
    precision: u8,
}

/// The decoder's verdict on a supported frame header (`parse_frame_header`,
/// in `read_header`). `body` follows the length word.
fn check_sof(body: &[u8], max_pixels: u64) -> Result<SofInfo, &'static str> {
    if body.len() + 2 < 8 {
        return Err("frame header too short");
    }
    let precision = body[0];
    if precision != 8 && precision != 12 {
        return Err("invalid data precision");
    }
    let height = be16(body, 1).unwrap_or(0);
    let width = be16(body, 3).unwrap_or(0);
    if width == 0 {
        return Err("zero width");
    }
    if height > 0 {
        if u32::from(width) > MAX_DIMENSION || u32::from(height) > MAX_DIMENSION {
            return Err("dimension exceeds 65500");
        }
        let max = if max_pixels == 0 {
            u64::MAX
        } else {
            max_pixels
        };
        if u64::from(width) * u64::from(height) > max {
            return Err("image exceeds the decoder's max_pixels limit");
        }
    } else if u32::from(width) > MAX_DIMENSION {
        return Err("dimension exceeds 65500");
    }
    let nc = body[5];
    if nc == 0 {
        return Err("number of components is zero");
    }
    if nc > 4 {
        return Err("more than 4 components");
    }
    if body.len() + 2 != 8 + 3 * nc as usize {
        return Err("SOF marker length mismatch");
    }
    let mut ids = [0u8; 4];
    let mut qidx = [0u8; 4];
    for (c, id) in ids.iter_mut().enumerate().take(nc as usize) {
        let at = 6 + 3 * c;
        *id = body[at];
        let (h, v) = (body[at + 1] >> 4, body[at + 1] & 0x0F);
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err("invalid sampling factor");
        }
        if body[at + 2] >= 4 {
            return Err("quantization table index out of range");
        }
        qidx[c] = body[at + 2];
    }
    Ok(SofInfo {
        frame: Frame {
            components: nc,
            ids,
            qidx,
            mode: 0,
        },
        height,
        precision,
    })
}

/// The decoder's verdict on a scan header (`parse_scan`), at the
/// default strictness. `None` when it is accepted.
fn check_sos(data: &[u8], pos: usize, ns: u8, frame: Option<Frame>) -> Option<&'static str> {
    let Some(frame) = frame else {
        return Some("no frame header");
    };
    if ns == 0 {
        return Some("scan with zero components");
    }
    if ns > frame.components {
        return Some("scan has more components than the frame");
    }
    let mut seen = [false; 4];
    for c in 0..ns as usize {
        let at = pos + 5 + 2 * c;
        let (id, tables) = (data[at], data[at + 1]);
        if tables >> 4 >= 4 || tables & 0x0F >= 4 {
            return Some("Huffman table index out of range");
        }
        let Some(idx) = frame.ids[..frame.components as usize]
            .iter()
            .position(|&x| x == id)
        else {
            return Some("unknown component in scan");
        };
        if seen[idx] {
            return Some("duplicate component in scan");
        }
        seen[idx] = true;
    }
    let at = pos + 5 + 2 * ns as usize;
    let (ss, se, ahal) = (data[at], data[at + 1], data[at + 2]);
    if ss > 63 || se > 63 {
        return Some("spectral selection beyond 63");
    }
    if ss > se {
        return Some("spectral selection start exceeds end");
    }
    if ahal >> 4 > 13 || ahal & 0x0F > 13 {
        return Some("successive approximation out of range");
    }
    None
}

struct SamsungTrailer {
    range: Range<usize>,
    directory: Range<usize>,
    count: u32,
    /// `(range, type, name)` of each valid data block, non-overlapping.
    blocks: Vec<(Range<usize>, u16, Option<String>)>,
}

/// The Samsung SEF trailer at the end of the file, if one is there and lies
/// entirely after `min_start`. Layout: data blocks, an `SEFH` directory of
/// 12-byte entries (type, distance back from the directory, size), then the
/// directory length and `SEFT`.
fn samsung_trailer(data: &[u8], min_start: usize) -> Option<SamsungTrailer> {
    let len = data.len();
    let foot = len.checked_sub(8)?;
    if data.get(foot + 4..)? != b"SEFT" {
        return None;
    }
    let dir_len = le32(data, foot)? as usize;
    let dir_pos = foot.checked_sub(dir_len)?;
    if dir_pos < min_start || dir_len < 12 || data.get(dir_pos..dir_pos + 4)? != b"SEFH" {
        return None;
    }
    let count = le32(data, dir_pos + 8)?;
    if 12usize.checked_add((count as usize).checked_mul(12)?)? > dir_len {
        return None;
    }
    let mut first_block = 0usize;
    let mut blocks: Vec<(Range<usize>, u16, Option<String>)> = Vec::new();
    for i in 0..count as usize {
        let e = dir_pos + 12 + 12 * i;
        let ty = u16::from_le_bytes([*data.get(e + 2)?, *data.get(e + 3)?]);
        let noff = le32(data, e + 4)? as usize;
        let size = le32(data, e + 8)? as usize;
        if noff > dir_pos - min_start || size > noff || size < 8 {
            continue;
        }
        let start = dir_pos - noff;
        let r = start..start + size;
        // Block header: u16 0, u16 type, u32 name length, name, data.
        let name = le32(data, start + 4)
            .map(|n| n as usize)
            .filter(|&n| n.checked_add(8).is_some_and(|e| e <= size))
            .and_then(|n| data.get(start + 8..start + 8 + n))
            .and_then(label_of);
        first_block = first_block.max(noff);
        blocks.push((r, ty, name));
    }
    // Keep the blocks that do not overlap an earlier-starting one.
    blocks.sort_by_key(|(r, _, _)| (r.start, r.end));
    let mut end = 0;
    blocks.retain(|(r, _, _)| {
        let keep = r.start >= end;
        if keep {
            end = r.end;
        }
        keep
    });
    Some(SamsungTrailer {
        range: dir_pos - first_block..len,
        directory: dir_pos..len,
        count,
        blocks,
    })
}

#[cfg(test)]
mod tests;
