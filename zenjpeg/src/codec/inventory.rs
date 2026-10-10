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
//! unsupported or invalid frame header, a malformed segment, 12-bit
//! precision, DNL mode, a 2-component frame) make the decode fail; parts
//! the failing decode would have consumed are then `Skipped`, except those
//! `probe` still reads. Entropy errors inside scan data are invisible here:
//! scan data is `ImageData` whenever the container is sound.
//!
//! # Job settings
//!
//! The walker follows what the job sets that changes the outcome:
//!
//! - strictness: `probe()` parses the header with the inner config's
//!   strictness; `decode()` with the same, raised to `Strict` by a
//!   `DecodePolicy` with `strict` or `allow_truncated: false`. Strict turns
//!   every decoder warning the walker can see into a failure (stray bytes,
//!   truncation, a scan-header or DRI length mismatch, a zero quantization
//!   value, a DNL height conflict, a missing Huffman table); Permissive
//!   skips header segments the parser rejects, and segments shorter than
//!   their length word, by their length;
//! - `DecodePolicy::allow_progressive`, `ResourceLimits::max_pixels`,
//!   `max_width` and `max_height`;
//! - the `OrientationHint` (which EXIF orientation reaches the pixels) and
//!   the `GainMapRender` (whether the gain map is decoded).
//!
//! It assumes `ResourceLimits::max_memory_bytes` is large enough: whether a
//! decode fits depends on the decode path's allocations, which the walker
//! does not model.
//!
//! # A failure after the first scan
//!
//! After a scan coded with a restart interval, a non-Strict decoder that
//! misses an RSTn scans up to 4096 bytes forward for the next one, over
//! other markers (`resync_to_restart`, foundation/bitstream.rs), so it may
//! never reach a later part. A container-level failure there is reported
//! on the part only ("the decode fails here if the decoder reaches it")
//! and demotes nothing. Everywhere else the decoder reaches the next
//! marker after a scan, so the failure is certain.
//!
//! # Output paths
//!
//! The parts a decode uses can depend on the output the caller asks for,
//! which `inventory()` does not see. A baseline frame that
//! `can_use_streaming` accepts (1 or 3 components, standard sampling, not
//! XYB), decoded to u8 RGB-family output, dequantises and converts colour
//! during its first all-component scan, with the quantisation tables and
//! APP14 transform in effect then, and ignores its other scans; every other
//! output (f32, Gray, CMYK, other formats, dequant bias, Knusperli
//! deblocking) and every other frame stores coefficients and converts at
//! the end, with what is in effect then. Dispositions follow the default
//! decode (no preferred descriptors: RGB8 for 3 components, GRAY8 for 1),
//! and a part the other path treats differently says so in its detail.

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
    DecodedExtras, MpfImageType, SegmentType, Strictness, detect_segment_type, parse_mpf_directory,
};

/// What the decode job is configured to do, where it changes a disposition.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    /// The job applies EXIF orientation to the pixels (`OrientationHint`
    /// `Correct*`), so `find_exif_orientation` reads an EXIF segment.
    pub(crate) auto_orient: bool,
    /// How the job renders an Ultra HDR gain map.
    pub(crate) render: Render,
    /// The decoder's pixel cap (`0` = unlimited), checked against the frame
    /// header the way `parse_frame_header` does.
    pub(crate) max_pixels: u64,
    /// The strictness `probe()` parses the header with: the inner config's.
    pub(crate) probe_strictness: Strictness,
    /// The strictness `decode()` runs at: the inner config's, raised to
    /// `Strict` by a `DecodePolicy` with `strict` or `allow_truncated: false`.
    pub(crate) decode_strictness: Strictness,
    /// `DecodePolicy::allow_progressive` (default true); `decode()` refuses
    /// SOF2/SOF10 frames when false.
    pub(crate) allow_progressive: bool,
    /// `ResourceLimits::max_width` / `max_height`, checked by `decode()`
    /// against the frame header.
    pub(crate) max_width: Option<u32>,
    pub(crate) max_height: Option<u32>,
}

/// The job's `GainMapRender`, as far as it changes what the decode reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Render {
    /// `BaseOnly`: the gain-map image is never decoded.
    Base,
    /// `Components`: `decode_gain_map_components` (codec/decode.rs) runs
    /// after the base decode.
    Components,
    /// `ReconstructHdr`. `hdrgm`: the XMP `read_info` returns contains
    /// `hdrgm:`, which is what sends the decode down
    /// `decode_reconstruct_hdr`; otherwise it decodes the base image only.
    Reconstruct { hdrgm: bool },
    /// The decode refuses the mode: `Components` or `ReconstructHdr`
    /// without the `ultrahdr` feature, or a mode it does not recognize.
    Refused,
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
            let mut st = w.walk_stream(None, start, len, Rules::primary(&opts))?;
            if start > 0 {
                st.probe_fatal = true;
                st.decode_fatal = true;
            }
            if opts.render == Render::Refused && !st.decode_fatal {
                w.append_detail(
                    st.nodes.start,
                    "decode fails: the job's GainMapRender needs the ultrahdr feature, or is \
                     not recognized",
                );
                st.decode_fatal = true;
            }
            let (mpf_node, images) = match st.eoi_end {
                Some(eoi_end) => mpf_images(data, &st, eoi_end),
                None => (None, Vec::new()),
            };
            let gain_map = if st.decode_fatal {
                GainMap::None
            } else {
                w.gain_map_use(&st, &images)
            };
            if let GainMap::Fails(why) = gain_map {
                w.append_detail(st.nodes.start, &format!("decode fails: {why}"));
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
                w.after_eoi(&st, eoi_end, mpf_node, &images, gain_map)?;
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

/// An MPF image entry `extract_mpf_secondary_images` considers.
struct MpfImage {
    idx: usize,
    ty: MpfImageType,
    size: u32,
    offset: u32,
    /// Absolute range (`start..start` when the offset overflows).
    range: Range<usize>,
    /// In the data and starting with SOI: copied to `secondary_images`.
    extracted: bool,
}

/// What the job's decode does with the Ultra HDR gain map.
#[derive(Clone, Copy)]
enum GainMap {
    /// Not decoded.
    None,
    /// MPF image `idx` is decoded; `xmp_at`: the start of the APP1 inside
    /// it that supplies the parameters, when that is where they come from.
    Decoded { idx: usize, xmp_at: Option<usize> },
    /// The decode fails.
    Fails(&'static str),
}

/// The MPF images the decode extracts at the primary EOI
/// (`extract_mpf_secondary_images`, decode/parser/mod.rs) from the first
/// MPF segment: every entry but the first and `BaselinePrimary` ones,
/// offset 0 meaning "right after EOI" and others relative to the TIFF
/// header after `MPF\0`, extracted when the range lies in the data and
/// starts with SOI, wherever that is (inside the primary too).
fn mpf_images(data: &[u8], st: &Stream, eoi_end: usize) -> (Option<usize>, Vec<MpfImage>) {
    let Some(mpf) = st.apps.iter().find(|a| a.ty == SegmentType::Mpf) else {
        return (None, Vec::new());
    };
    let body = &data[mpf.payload.clone()];
    let dir = mpf_read_ranges(body).and_then(|_| parse_mpf_directory(body));
    let Some(dir) = dir else {
        return (Some(mpf.node), Vec::new());
    };
    let base = mpf.payload.start + 4;
    let mut out = Vec::new();
    for (idx, entry) in dir.images.iter().enumerate() {
        if idx == 0 || matches!(entry.image_type, MpfImageType::BaselinePrimary) {
            continue;
        }
        let start = if entry.offset == 0 {
            Some(eoi_end)
        } else {
            base.checked_add(entry.offset as usize)
        };
        let range = start.map_or(0..0, |s| s..s.saturating_add(entry.size as usize));
        let extracted = data
            .get(range.clone())
            .is_some_and(|b| b.starts_with(&[0xFF, MARKER_SOI]));
        out.push(MpfImage {
            idx,
            ty: entry.image_type,
            size: entry.size,
            offset: entry.offset,
            range,
            extracted,
        });
    }
    (Some(mpf.node), out)
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
    /// Sampling factors per component, `h << 4 | v`.
    samp: [u8; 4],
    width: u16,
    height: u16,
    /// The SOF marker: 0xC0/C1 sequential and 0xC2 progressive Huffman,
    /// 0xC9/CA arithmetic.
    mode: u8,
}

/// The decode paths that use a definition or a part, as bits.
///
/// `STREAM`: `can_use_streaming` (decode/parser/scan.rs) frames decoded to
/// u8 RGB-family output dequantise and colour-convert during their first
/// scan that holds every component (baseline_streaming.rs), with the tables
/// and APP14 transform in effect then. `COEFF`: every other output (f32,
/// Gray, CMYK, other formats, dequant bias, Knusperli deblocking;
/// `DecodeConfig::decode`) and every other frame store coefficients and
/// convert at the end, with the tables and transform in effect then.
const STREAM: u8 = 1;
const COEFF: u8 = 2;
const BOTH: u8 = STREAM | COEFF;

impl Frame {
    /// `can_use_streaming` for this frame's all-component scan.
    fn streams(&self, xyb: bool) -> bool {
        let standard = match self.components {
            1 => self.samp[0] == 0x11,
            3 => {
                let [y, cb, cr, _] = self.samp;
                cb == cr && cb == 0x11 && matches!(y, 0x11 | 0x22 | 0x21)
            }
            _ => false,
        };
        matches!(self.mode, 0xC0 | 0xC1) && standard && !xyb
    }
}

/// Which path the default zencodec decode (no preferred descriptors) takes,
/// and the other one an explicit output format can select.
#[derive(Clone, Copy)]
struct Paths {
    default: u8,
    other: u8,
}

impl Paths {
    /// Default output: RGB8 for 3 components (streaming when eligible),
    /// GRAY8 for 1 (coefficients; RGB-family output streams instead).
    fn of(frame: Option<Frame>, all_component_scan: bool, xyb: bool) -> Self {
        match frame {
            Some(f) if all_component_scan && f.streams(xyb) && f.components == 3 => Self {
                default: STREAM,
                other: COEFF,
            },
            Some(f) if all_component_scan && f.streams(xyb) => Self {
                default: COEFF,
                other: STREAM,
            },
            _ => Self {
                default: COEFF,
                other: 0,
            },
        }
    }

    /// What the default and the other output paths do with a quantisation
    /// table (or APP14) used on `paths`, when they disagree.
    fn note(self, paths: u8) -> Option<&'static str> {
        let (d, o) = (paths & self.default != 0, paths & self.other != 0);
        if self.other == 0 || d == o {
            return None;
        }
        Some(match (self.default, d) {
            (STREAM, true) => {
                "used by the default u8 RGB output, converted during the scan; an output that \
                 needs coefficients (f32, Gray, CMYK, dequant bias) converts at the end and \
                 uses what is in effect then instead"
            }
            (STREAM, false) => {
                "unused by the default u8 RGB output, converted during the scan; used when the \
                 output needs coefficients (f32, Gray, CMYK, dequant bias), converted at the end"
            }
            (_, true) => {
                "used by the default Gray output, converted at the end; u8 RGB-family output \
                 converts during the scan and uses what is in effect then instead"
            }
            _ => {
                "unused by the default Gray output, converted at the end; used by u8 \
                 RGB-family output, converted during the scan"
            }
        })
    }
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

/// A definition in effect; `used` indexes its record in `Defs::used` once
/// a decode path uses it.
struct Def {
    node: usize,
    range: Range<usize>,
    what: DefKind,
    used: Option<usize>,
}

/// A definition some decode path uses, and which ([`STREAM`], [`COEFF`]).
struct Used {
    node: usize,
    range: Range<usize>,
    what: DefKind,
    paths: u8,
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
    /// Every definition a decode path used, recorded once.
    used: Vec<Used>,
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
                used: None,
            });
        }
    }

    /// The definition in effect for `k` is used on `paths`.
    fn mark(&mut self, k: DefKind, paths: u8) {
        let n = self.used.len();
        let (at, fresh) = match self.slot(k) {
            Some(Some(d)) => match d.used {
                Some(at) => (at, None),
                None => {
                    d.used = Some(n);
                    let u = Used {
                        node: d.node,
                        range: d.range.clone(),
                        what: d.what,
                        paths: 0,
                    };
                    (n, Some(u))
                }
            },
            _ => return,
        };
        if let Some(u) = fresh {
            self.used.push(u);
        }
        self.used[at].paths |= paths;
    }

    /// Whether a sequential Huffman scan selects a table no DHT defined
    /// (`parse_scan` warns `MissingHuffmanTables` and uses the K.3 tables).
    fn missing_huffman(&self, spec: &[u8], permissive: bool) -> bool {
        let comps = &spec[..spec.len().saturating_sub(3)];
        comps.as_chunks::<2>().0.iter().any(|&[_, tables]| {
            let (td, ta) = scan_tables(tables, permissive);
            let (td, ta) = (td.min(3) as usize, ta.min(3) as usize);
            self.dc[td].is_none() || self.ac[ta].is_none()
        })
    }

    /// The quantisation tables every frame component refers to.
    fn mark_quant(&mut self, f: &Frame, paths: u8) {
        for c in 0..f.components as usize {
            self.mark(DefKind::Quant(f.qidx[c]), paths);
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
    /// The decode paths this stream's frame can take.
    paths: Paths,
    /// Start of the first scan header that holds every frame component:
    /// the one the streaming path decodes.
    stream_scan: Option<usize>,
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

/// How the consumers read one stream, where it changes the outcome.
#[derive(Clone, Copy)]
struct Rules {
    /// `probe()`'s strictness; `None` for an embedded image nothing probes.
    probe: Option<Strictness>,
    decode: Strictness,
    max_pixels: u64,
    allow_progressive: bool,
    max_width: Option<u32>,
    max_height: Option<u32>,
}

impl Rules {
    fn primary(o: &Options) -> Self {
        Self {
            probe: Some(o.probe_strictness),
            decode: o.decode_strictness,
            max_pixels: o.max_pixels,
            allow_progressive: o.allow_progressive,
            max_width: o.max_width,
            max_height: o.max_height,
        }
    }

    /// An embedded image: the gain map is decoded by `decode_gainmap_jpeg`
    /// with a default `Decoder` (Balanced, default limits, no policy).
    fn embedded() -> Self {
        Self {
            probe: None,
            decode: Strictness::default(),
            max_pixels: crate::foundation::alloc::DEFAULT_MAX_PIXELS,
            allow_progressive: true,
            max_width: None,
            max_height: None,
        }
    }
}

/// The strictness levels at which a container-level problem stops the
/// decoder.
#[derive(Clone, Copy)]
enum Fails {
    Always,
    /// Only `Strict`: a warning (`JpegParser::warn`) there, an error here.
    Strict,
    /// Every level but `Permissive`, which skips the segment by its length.
    UnlessPermissive,
    /// `Strict` and `Balanced`; `Lenient` and `Permissive` recover.
    UnlessLenient,
}

impl Fails {
    fn at(self, s: Strictness) -> bool {
        match self {
            Self::Always => true,
            Self::Strict => s.is_strict(),
            Self::UnlessPermissive => !s.is_permissive(),
            Self::UnlessLenient => !s.lenient_entropy_recovery(),
        }
    }
}

/// Where a failing part sits.
#[derive(Clone, Copy)]
struct At {
    /// `probe()` reads it (before the first frame header).
    header: bool,
    /// The decoder certainly reaches it, given that it gets this far. Not
    /// so after a scan coded with a restart interval at a non-Strict level:
    /// when an interval ends without its RSTn, `resync_to_restart`
    /// (foundation/bitstream.rs) scans up to 4096 bytes forward for any
    /// RSTn, over other markers, and the decoder resumes there.
    certain: bool,
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

/// End of the entropy-coded data starting at `start`: the first `0xFF` of
/// a run that ends in neither stuffing (`00`) nor a restart marker
/// (`D0..D7`). A run of fill bytes before `00` is one stuffed data byte to
/// the decoder (`BitReader::read_byte_slow`).
fn scan_end(data: &[u8], start: usize, limit: usize) -> usize {
    let mut pos = start;
    while pos < limit {
        let Some(rel) = memchr::memchr(0xFF, &data[pos..limit]) else {
            return limit;
        };
        let ff = pos + rel;
        let mut q = ff + 1;
        while q < limit && data[q] == 0xFF {
            q += 1;
        }
        if q >= limit {
            return limit;
        }
        if data[q] == 0x00 || (0xD0..=0xD7).contains(&data[q]) {
            pos = q + 1;
        } else {
            return ff;
        }
    }
    limit
}

/// Why the decoder rejects a table segment: how many tables (or DAC
/// entries) it stored before the bad one, why, and whether it simply ran
/// out of bytes (a cut, in the bytes left of a truncated segment).
#[derive(Clone, Copy)]
struct Bad {
    stored: usize,
    why: &'static str,
    ran_out: bool,
}

fn bad(stored: usize, why: &'static str) -> Result<(), Bad> {
    Err(Bad {
        stored,
        why,
        ran_out: false,
    })
}

fn ran_out(stored: usize, why: &'static str) -> Result<(), Bad> {
    Err(Bad {
        stored,
        why,
        ran_out: true,
    })
}

/// The decoder's verdict on a DQT segment (`JpegParser::parse_quant_table`):
/// `declared` bytes of tables by the length field, read from `avail` (the
/// data from the start of the body on, which a cut can leave shorter).
fn check_dqt(avail: &[u8], declared: usize) -> Result<(), Bad> {
    let (mut at, mut k) = (0, 0);
    while at < declared {
        let Some(&info) = avail.get(at) else {
            return ran_out(k, "truncated");
        };
        if info >> 4 > 1 {
            return bad(k, "invalid quantization table precision");
        }
        if info & 0x0F >= 4 {
            return bad(k, "quantization table index out of range");
        }
        let n = if info >> 4 == 0 { 64 } else { 128 };
        if declared - at - 1 < n {
            return bad(k, "DQT length mismatch");
        }
        if avail.len() < at + 1 + n {
            return ran_out(k, "truncated");
        }
        at += 1 + n;
        k += 1;
    }
    Ok(())
}

/// Whether a valid DQT body holds a zero quantization value, which the
/// decoder clamps to 1 with a warning (an error when `Strict`).
fn has_zero_quant(body: &[u8]) -> bool {
    let mut rest = body;
    while let Some((&info, tail)) = rest.split_first() {
        let wide = info >> 4 == 1;
        let n = if wide { 128 } else { 64 };
        let Some(values) = tail.get(..n) else {
            return false;
        };
        let zero = if wide {
            values.as_chunks::<2>().0.iter().any(|v| *v == [0, 0])
        } else {
            values.contains(&0)
        };
        if zero {
            return true;
        }
        rest = &tail[n..];
    }
    false
}

/// The decoder's verdict on a DHT segment (`JpegParser::parse_huffman_table`),
/// including the code-length check `HuffmanDecodeTable` makes. Arguments
/// as for [`check_dqt`]: the parser reads a table's counts and symbols
/// before it compares them with the declared length, so a short length
/// field makes it read on into whatever follows.
fn check_dht(avail: &[u8], declared: usize) -> Result<(), Bad> {
    use crate::huffman::HuffmanDecodeTable;
    let (mut at, mut k) = (0, 0);
    while at < declared {
        let Some(&info) = avail.get(at) else {
            return ran_out(k, "truncated");
        };
        if info >> 4 > 1 {
            return bad(k, "invalid Huffman table class");
        }
        if info & 0x0F >= 4 {
            return bad(k, "Huffman table index out of range");
        }
        let Some(bits) = avail.get(at + 1..at + 17) else {
            return ran_out(k, "truncated");
        };
        let mut counts = [0u8; 16];
        counts.copy_from_slice(bits);
        let n: usize = counts.iter().map(|&b| b as usize).sum();
        if n > 256 {
            return bad(k, "Huffman symbol count exceeds 256");
        }
        let Some(values) = avail.get(at + 17..at + 17 + n) else {
            return ran_out(k, "truncated");
        };
        if at + 17 + n > declared {
            return bad(k, "DHT length mismatch");
        }
        let built = if info >> 4 == 0 {
            HuffmanDecodeTable::from_bits_values(&counts, values).map(|_| ())
        } else {
            HuffmanDecodeTable::from_bits_values_ac(&counts, values).map(|_| ())
        };
        if built.is_err() {
            return bad(k, "invalid Huffman code lengths");
        }
        at += 17 + n;
        k += 1;
    }
    Ok(())
}

/// The decoder's verdict on DAC entries (`JpegParser::parse_dac`): two
/// bytes per entry while two declared bytes remain.
fn check_dac(avail: &[u8], declared: usize) -> Result<(), Bad> {
    let (mut at, mut k) = (0, 0);
    while at + 2 <= declared {
        let Some(&info) = avail.get(at) else {
            return ran_out(k, "truncated");
        };
        if info & 0x0F >= 4 {
            return bad(k, "invalid DAC conditioning table");
        }
        let Some(&cs) = avail.get(at + 1) else {
            return ran_out(k, "truncated");
        };
        if info >> 4 == 0 && cs & 0x0F > cs >> 4 {
            return bad(k, "invalid DAC conditioning table");
        }
        at += 2;
        k += 1;
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

/// Mark what a scan uses on every path: the restart interval, and the
/// entropy-coding tables or conditioning entries its coding reads. `spec`
/// is the SOS body after the component count: the component selectors,
/// then Ss, Se, Ah/Al. Quantisation tables are marked by the caller: the
/// streaming path at its scan, the coefficient path at the end.
fn mark_scan_uses(defs: &mut Defs, f: &Frame, spec: &[u8], permissive: bool) {
    defs.mark(DefKind::Restart, BOTH);
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
        // Progressive Huffman (`decode_progressive_scan`: a DC scan has
        // Ss = Se = 0): DC first scans use the DC table, DC refinement uses
        // none, AC scans (first and refinement) use the AC table.
        _ => {
            let dc_scan = ss == 0 && params.get(1) == Some(&0);
            (dc_scan && first, !dc_scan)
        }
    };
    for &[_, tables] in comps.as_chunks::<2>().0 {
        let (td, ta) = scan_tables(tables, permissive);
        let (dc, ac) = if arithmetic {
            (DefKind::DacDc(td), DefKind::DacAc(ta))
        } else {
            (DefKind::Dc(td), DefKind::Ac(ta))
        };
        if need_dc {
            defs.mark(dc, BOTH);
        }
        if need_ac {
            defs.mark(ac, BOTH);
        }
    }
}

/// A Huffman table a scan reads.
enum Table {
    Owned(crate::huffman::HuffmanDecodeTable),
    Standard(&'static crate::huffman::HuffmanDecodeTable),
}

impl Table {
    fn get(&self) -> &crate::huffman::HuffmanDecodeTable {
        match self {
            Self::Owned(t) => t,
            Self::Standard(t) => t,
        }
    }

    /// The DHT definition in effect for the slot, or the K.3 table the
    /// decoder falls back to (`install_progressive_huffman_tables`).
    fn of(data: &[u8], def: Option<&Def>, ac: bool, idx: u8) -> Option<Self> {
        use crate::huffman::HuffmanDecodeTable as H;
        let Some(d) = def else {
            return Some(Self::Standard(match (ac, idx) {
                (false, 0) => H::std_dc_luminance(),
                (false, _) => H::std_dc_chrominance(),
                (true, 0) => H::std_ac_luminance(),
                (true, _) => H::std_ac_chrominance(),
            }));
        };
        let b = data.get(d.range.clone())?;
        let counts: [u8; 16] = b.get(1..17)?.try_into().ok()?;
        H::from_bits_values(&counts, b.get(17..)?)
            .ok()
            .map(Self::Owned)
    }
}

/// Run the count-only entropy pass over a Huffman scan (`spec`: the SOS
/// body after the component count).
#[allow(clippy::too_many_arguments)]
fn count_scan(
    data: &[u8],
    f: &Frame,
    defs: &Defs,
    spec: &[u8],
    start: usize,
    end: usize,
    restart: u16,
    permissive: bool,
) -> Option<entropy::Count> {
    let (comps, params) = spec.split_at(spec.len().checked_sub(3)?);
    let (&ss, &se, &ahal) = (params.first()?, params.get(1)?, params.get(2)?);
    let nc = f.components as usize;
    let mut tables: Vec<(Table, Table, u8, u8)> = Vec::with_capacity(4);
    for &[id, sel] in comps.as_chunks::<2>().0 {
        let ci = f.ids[..nc].iter().position(|&x| x == id)?;
        let (td, ta) = scan_tables(sel, permissive);
        let (td, ta) = (td.min(3), ta.min(3));
        let dc = Table::of(data, defs.dc[td as usize].as_ref(), false, td)?;
        let ac = Table::of(data, defs.ac[ta as usize].as_ref(), true, ta)?;
        tables.push((dc, ac, f.samp[ci] >> 4, f.samp[ci] & 0x0F));
    }
    let comps: Vec<entropy::ScanComp<'_>> = tables
        .iter()
        .map(|(dc, ac, h, v)| entropy::ScanComp {
            h: *h,
            v: *v,
            dc: dc.get(),
            ac: ac.get(),
        })
        .collect();
    let hmax = f.samp[..nc].iter().map(|s| s >> 4).max()?;
    let vmax = f.samp[..nc].iter().map(|s| s & 0x0F).max()?;
    let scan = entropy::Scan {
        comps: &comps,
        hmax,
        vmax,
        width: u32::from(f.width),
        height: u32::from(f.height),
        progressive: f.mode == 0xC2,
        ss,
        se,
        ah: ahal >> 4,
        restart,
    };
    Some(entropy::count(data, start, end, &scan))
}

/// Whether bytes the decoder skips between markers hold anything but
/// `FF` runs and stuffed `FF 00` pairs (which `read_marker` skips without
/// a warning).
fn data_has_stray(b: &[u8]) -> bool {
    let mut i = 0;
    while i < b.len() {
        if b[i] != 0xFF {
            return true;
        }
        while i < b.len() && b[i] == 0xFF {
            i += 1;
        }
        // The byte after a run: 00 (skipped) or a marker code.
        i += 1;
    }
    false
}

/// A scan component's (DC, AC) table selectors; Permissive clamps an
/// out-of-range one to 0 (`parse_scan`).
fn scan_tables(tables: u8, permissive: bool) -> (u8, u8) {
    let clamp = |t: u8| if permissive && t >= 4 { 0 } else { t };
    (clamp(tables >> 4), clamp(tables & 0x0F))
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

    /// A container-level problem at `node` that stops the decoder at the
    /// strictness levels `f` names. When it is certain the decoder reaches
    /// it, the stream's decode (and, in the header, `probe()`) fails;
    /// otherwise only the part says so.
    fn fail(&mut self, st: &mut Stream, rules: Rules, node: usize, at: At, f: Fails, why: &str) {
        if at.header && rules.probe.is_some_and(|s| f.at(s)) {
            st.probe_fatal = true;
        }
        if !f.at(rules.decode) {
            return;
        }
        if at.certain {
            if !st.decode_fatal {
                self.append_detail(node, &format!("decode fails here: {why}"));
            }
            st.decode_fatal = true;
        } else {
            self.append_detail(
                node,
                &format!(
                    "the decode fails here if the decoder reaches it ({why}); an earlier scan \
                     coded with a restart interval may resync past this part"
                ),
            );
        }
    }

    /// The details, children and failures the count-only entropy pass
    /// gives a scan-data part `hdr_end..end` (`limit`: where the stream's
    /// data ends).
    #[allow(clippy::too_many_arguments)]
    fn settle_scan(
        &mut self,
        st: &mut Stream,
        rules: Rules,
        node: usize,
        here: At,
        end: usize,
        limit: usize,
        counted: Option<&entropy::Count>,
    ) -> Result<(), InventoryError> {
        use entropy::End;
        let truncated = end == limit;
        let Some(c) = counted else {
            self.append_detail(
                node,
                if truncated {
                    "no marker after the entropy-coded data (truncated; the decoder pads); bytes \
                     after the last MCU are not distinguished"
                } else {
                    "bytes after the last MCU are not distinguished: AC refinement and \
                     arithmetic-coded scans are not counted"
                },
            );
            if truncated {
                self.fail(st, rules, node, here, Fails::Strict, "scan data truncated");
            }
            return Ok(());
        };
        match c.end {
            End::Complete => self.append_detail(
                node,
                "entropy-coded data counted to the last MCU (count-only Huffman pass)",
            ),
            End::Exhausted => {
                self.append_detail(
                    node,
                    "the entropy-coded data runs out before the last MCU; the decoder pads the \
                     rest with zero bits",
                );
                if truncated {
                    self.fail(st, rules, node, here, Fails::Strict, "scan data truncated");
                }
            }
            End::Resync { at } => {
                self.append_detail(
                    node,
                    &format!(
                        "a restart interval ends where its RSTn should be, but the marker at {at} \
                         is not it: a non-Strict decoder scans forward for any RSTn"
                    ),
                );
                self.fail(
                    st,
                    rules,
                    node,
                    here,
                    Fails::Strict,
                    "expected restart marker not found",
                );
            }
            End::InvalidCode { at } => {
                let why = format!("invalid Huffman code near offset {at}");
                self.append_detail(node, &why);
                self.fail(
                    st,
                    rules,
                    node,
                    here,
                    Fails::UnlessLenient,
                    "invalid Huffman code",
                );
            }
            End::BadDcCategory { at } => {
                self.append_detail(node, &format!("DC category above 16 near offset {at}"));
                self.fail(st, rules, node, here, Fails::Always, "DC category above 16");
            }
            End::Unsupported => self.append_detail(
                node,
                "bytes after the last MCU are not distinguished: this scan is not counted",
            ),
        }
        if c.ac_overflow {
            self.fail(
                st,
                rules,
                node,
                here,
                Fails::Strict,
                "AC coefficient run past the block",
            );
        }
        if c.rst_mismatch {
            self.fail(
                st,
                rules,
                node,
                here,
                Fails::Strict,
                "restart marker sequence mismatch",
            );
        }
        // Tails only where the count is clean: an irregular scan (AC runs
        // past the block, out-of-sequence RSTn, a resync, an invalid code)
        // may be read differently by the decoder's recovery paths.
        let clean =
            matches!(c.end, End::Complete | End::Exhausted) && !c.ac_overflow && !c.rst_mismatch;
        if !clean {
            if !c.tails.is_empty() || c.stop_at.is_some() {
                self.append_detail(
                    node,
                    "bytes after the last MCU are not distinguished: the count is irregular",
                );
            }
            return Ok(());
        }
        for (r, after_last) in &c.tails {
            let detail = if *after_last {
                "after the last MCU's entropy-coded data: never decoded; the decoder skips it as \
                 stray bytes before the next marker (Strict may reject them, depending on how \
                 far its bit reader read ahead)"
            } else {
                "after a restart interval's entropy-coded data, before its RSTn: never decoded; \
                 the decoder skips it"
            };
            self.add(
                Some(node),
                gap(r.clone(), Disposition::Unreferenced).with_detail(detail),
            )?;
        }
        if let Some(at) = c.stop_at {
            let child = self.add(
                Some(node),
                gap(at..end, Disposition::Malformed).with_detail(
                    "the decoder ends the scan at this restart marker (no interval expects it), \
                     ignores the marker and skips what follows as stray bytes",
                ),
            )?;
            if data_has_stray(&self.data[at..end]) {
                self.fail(
                    st,
                    rules,
                    child,
                    here,
                    Fails::Strict,
                    "extraneous bytes between markers",
                );
            }
        }
        Ok(())
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
        rules: Rules,
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
            paths: Paths {
                default: COEFF,
                other: 0,
            },
            stream_scan: None,
        };
        self.add(
            parent,
            seg(MARKER_SOI, start..start + 2, Disposition::Structure),
        )?;
        // Scan-data parts: (node, scan header offset).
        let mut scan_parts: Vec<(usize, usize)> = Vec::new();
        let mut pos = start + 2;
        let mut phase = Phase::Header;
        let mut scans = 0u32;
        let mut seen_sos = false;
        // The image height in effect (SOF, or a DNL when SOF left it 0).
        let mut height = 0u16;
        let mut defs = Defs::default();
        // The restart interval in effect, and whether a scan was coded with
        // a nonzero one (see `At::certain`).
        let mut restart = 0u16;
        let mut resync_risk = false;
        let at = |phase: Phase, scans: u32, resync_risk: bool| At {
            header: phase == Phase::Header,
            certain: phase == Phase::Header
                || scans == 0
                || !resync_risk
                || rules.decode.is_strict(),
        };
        // Failures `probe()` never meets: it stops at the frame header.
        let decode_only = |phase: Phase, scans: u32, resync_risk: bool| At {
            header: false,
            ..at(phase, scans, resync_risk)
        };

        while pos < limit && phase != Phase::Done {
            // `JpegParser::read_marker`: skip to the next 0xFF,
            // warning about the bytes in between, then skip fill bytes.
            let next_ff = memchr::memchr(0xFF, &data[pos..limit]).map_or(limit, |r| pos + r);
            if next_ff > pos {
                let node = self.add_repeated(
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
                // `read_marker` warns once it reaches the next 0xFF.
                self.fail(
                    &mut st,
                    rules,
                    node,
                    at(phase, scans, resync_risk),
                    Fails::Strict,
                    "extraneous bytes between markers",
                );
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
                // After a scan, a cut is recovered as the end of the image.
                let f = if phase == Phase::Header || scans == 0 {
                    Fails::Always
                } else {
                    Fails::Strict
                };
                self.fail(
                    &mut st,
                    rules,
                    node,
                    at(phase, scans, resync_risk),
                    f,
                    "truncated",
                );
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
                    self.fail(
                        &mut st,
                        rules,
                        node,
                        at(phase, scans, resync_risk),
                        Fails::Always,
                        "EOI before any frame header",
                    );
                } else if height == 0 {
                    // `JpegParser::decode`: EOI with height 0 is an error.
                    self.fail(
                        &mut st,
                        rules,
                        node,
                        at(phase, scans, resync_risk),
                        Fails::Always,
                        "image height is 0 and no DNL set it",
                    );
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
                    let node = self.add(
                        parent,
                        seg(m, pos..limit, Disposition::Structure)
                            .with_detail("truncated scan header; the decoder stops here"),
                    )?;
                    self.fail(
                        &mut st,
                        rules,
                        node,
                        at(phase, scans, resync_risk),
                        Fails::Strict,
                        "truncated scan header",
                    );
                    break;
                };
                let ns = data[pos + 4];
                let declared = be16(data, pos + 2).unwrap_or(0);
                let node = self.add(parent, seg(m, pos..hdr_end, Disposition::Structure))?;
                let here = at(phase, scans, resync_risk);
                if declared as usize != 6 + 2 * ns as usize {
                    self.append_detail(
                        node,
                        &format!(
                            "declared length {declared} ignored; the decoder reads {} bytes",
                            6 + 2 * ns as usize
                        ),
                    );
                    self.fail(
                        &mut st,
                        rules,
                        node,
                        here,
                        Fails::Strict,
                        "scan header length mismatch",
                    );
                }
                let permissive = rules.decode.is_permissive();
                let mut sos_ok = true;
                if let Some((f, why)) = check_sos(data, pos, ns, st.frame) {
                    self.fail(&mut st, rules, node, here, f, why);
                    sos_ok = !f.at(rules.decode);
                }
                if let Some(f) = st.frame {
                    let spec = &data[pos + 5..hdr_end];
                    if matches!(f.mode, 0xC0 | 0xC1) && defs.missing_huffman(spec, permissive) {
                        // `parse_scan`: K.3 tables stand in, with a warning.
                        self.fail(
                            &mut st,
                            rules,
                            node,
                            here,
                            Fails::Strict,
                            "missing Huffman table",
                        );
                    }
                    mark_scan_uses(&mut defs, &f, spec, permissive);
                    if ns == f.components && st.stream_scan.is_none() {
                        // The streaming path's scan: it dequantises here.
                        st.stream_scan = Some(pos);
                        defs.mark_quant(&f, STREAM);
                    }
                }
                let end = scan_end(data, hdr_end, limit);
                let counted = match st.frame {
                    Some(f) if sos_ok && matches!(f.mode, 0xC0 | 0xC1 | 0xC2) => {
                        let spec = &data[pos + 5..hdr_end];
                        count_scan(data, &f, &defs, spec, hdr_end, end, restart, permissive)
                    }
                    _ => None,
                };
                // Without a count, or when an interval missed its RSTn,
                // the decoder may resync past later parts.
                resync_risk |= match counted.as_ref().map(|c| c.end) {
                    Some(entropy::End::Complete | entropy::End::Exhausted) => false,
                    Some(entropy::End::Resync { .. }) => true,
                    _ => restart > 0,
                };
                if end > hdr_end {
                    let scan = part(
                        PartKind::ScanData,
                        PartTag::None,
                        hdr_end..end,
                        Disposition::ImageData,
                    );
                    let scan_node = self.add(parent, scan)?;
                    scan_parts.push((scan_node, pos));
                    self.settle_scan(
                        &mut st,
                        rules,
                        scan_node,
                        here,
                        end,
                        limit,
                        counted.as_ref(),
                    )?;
                }
                scans += 1;
                seen_sos = true;
                if scans >= MAX_SCANS {
                    self.fail(&mut st, rules, node, here, Fails::Always, "too many scans");
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
                    // `parse_frame_header` always fails; `skip_segment`,
                    // `process_app_or_com` and `parse_dnl` skip it when
                    // Permissive, resuming after the length word.
                    let sof = phase == Phase::Header && sof_kind(m).is_some();
                    let f = if sof {
                        Fails::Always
                    } else {
                        Fails::UnlessPermissive
                    };
                    self.fail(
                        &mut st,
                        rules,
                        node,
                        at(phase, scans, resync_risk),
                        f,
                        "segment length too short",
                    );
                    pos = end;
                    if sof {
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
                    // The table parsers read entry by entry, so a bad entry in
                    // the bytes that are there fails before the cut does.
                    let avail = data.get(pos + 4..limit).unwrap_or(&[]);
                    let declared =
                        be16(&data[..limit], pos + 2).map_or(0, |n| (n as usize).saturating_sub(2));
                    let content = match m {
                        MARKER_DQT => check_dqt(avail, declared),
                        MARKER_DHT => check_dht(avail, declared),
                        MARKER_DAC => check_dac(avail, declared),
                        _ => Ok(()),
                    };
                    // Between scans a cut is recovered (`JpegParser::decode`).
                    let (f, why) = match content {
                        Err(b) if !b.ran_out => (Fails::Always, b.why),
                        _ if phase == Phase::Header || scans == 0 => (Fails::Always, "truncated"),
                        _ => (Fails::Strict, "truncated"),
                    };
                    self.fail(&mut st, rules, node, at(phase, scans, resync_risk), f, why);
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
                    let here = at(phase, scans, resync_risk);
                    let avail = &data[payload.start..limit];
                    let verdict = if m == MARKER_DQT {
                        check_dqt(avail, body.len())
                    } else {
                        check_dht(avail, body.len())
                    };
                    let ranges = table_ranges(m, body, payload.start);
                    // The tables before a bad one are stored before the
                    // parser fails; Permissive then skips the rest of the
                    // segment by its length, but only in the header.
                    let valid = match verdict {
                        Ok(()) => ranges.len(),
                        Err(b) => {
                            self.set(node, Disposition::Malformed, None);
                            let f = if phase == Phase::Header {
                                Fails::UnlessPermissive
                            } else {
                                Fails::Always
                            };
                            self.fail(&mut st, rules, node, here, f, b.why);
                            b.stored
                        }
                    };
                    if m == MARKER_DQT && verdict.is_ok() && has_zero_quant(body) {
                        // Clamped to 1 with a warning; `Strict` fails.
                        self.fail(
                            &mut st,
                            rules,
                            node,
                            here,
                            Fails::Strict,
                            "zero quantization value",
                        );
                    }
                    // Each table replaces the one in the same slot.
                    for (info, r) in ranges.into_iter().take(valid) {
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
                MARKER_DAC => {
                    // `parse_dac`: two bytes per entry; class 0 sets a DC
                    // table's (L, U), any other class an AC table's Kx. Each
                    // entry replaces the one for the same table.
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    let mut entries = body.as_chunks::<2>().0;
                    if let Err(b) = check_dac(body, body.len()) {
                        self.set(node, Disposition::Malformed, None);
                        let f = if phase == Phase::Header {
                            Fails::UnlessPermissive
                        } else {
                            Fails::Always
                        };
                        self.fail(
                            &mut st,
                            rules,
                            node,
                            at(phase, scans, resync_risk),
                            f,
                            b.why,
                        );
                        entries = &entries[..b.stored];
                    }
                    for (k, &[info, _]) in entries.iter().enumerate() {
                        let idx = info & 0x0F;
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
                    restart = be16(data, pos + 4).unwrap_or(0);
                    if be16(data, pos + 2) != Some(4) {
                        self.fail(
                            &mut st,
                            rules,
                            node,
                            at(phase, scans, resync_risk),
                            Fails::Strict,
                            "DRI length is not 4",
                        );
                    }
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
                    // `parse_dnl`: the length must be 4 (Permissive skips it).
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    let here = at(phase, scans, resync_risk);
                    let lines = be16(data, pos + 4).unwrap_or(0);
                    if end - pos != 6 {
                        self.set(node, Disposition::Malformed, None);
                        self.fail(
                            &mut st,
                            rules,
                            node,
                            here,
                            Fails::UnlessPermissive,
                            "DNL length is not 4",
                        );
                    } else if height > 0 {
                        // `parse_dnl` only sets the height when it is 0.
                        self.set(
                            node,
                            Disposition::Dropped,
                            Some(
                                "the frame header already set the height; the decoder ignores it"
                                    .into(),
                            ),
                        );
                        if lines != height {
                            self.fail(
                                &mut st,
                                rules,
                                node,
                                here,
                                Fails::Strict,
                                "DNL height conflicts with the frame height",
                            );
                        }
                    } else {
                        height = lines;
                    }
                }
                _ if phase == Phase::Header && sof_kind(m).is_some() => {
                    let (name, supported) = sof_kind(m).unwrap_or(("SOF", false));
                    let node = self.add(parent, seg(m, pos..end, Disposition::Structure))?;
                    st.sof_end = Some(end);
                    phase = Phase::Body;
                    let header = at(Phase::Header, 0, false);
                    let decode = decode_only(Phase::Header, 0, false);
                    if !supported {
                        // `read_header` rejects SOF3/SOF7/SOF11.
                        self.set(
                            node,
                            Disposition::Skipped,
                            Some(format!("lossless JPEG ({name}) is not supported")),
                        );
                        self.fail(
                            &mut st,
                            rules,
                            node,
                            header,
                            Fails::Always,
                            "unsupported frame type",
                        );
                    } else {
                        match check_sof(body, rules.max_pixels) {
                            Err(why) => {
                                self.set(node, Disposition::Malformed, None);
                                self.fail(&mut st, rules, node, header, Fails::Always, why);
                            }
                            Ok(f) => {
                                st.frame = Some(Frame { mode: m, ..f.frame });
                                height = f.height;
                                let too_big = rules
                                    .max_width
                                    .is_some_and(|w| u32::from(f.width) > w)
                                    || rules.max_height.is_some_and(|h| u32::from(f.height) > h);
                                let why = if f.height == 0 {
                                    // `read_info` rejects DNL mode
                                    // and every scan decoder does too.
                                    Some((header, "DNL mode (height 0) is not supported"))
                                } else if too_big {
                                    // `decode()` checks the frame against
                                    // `ResourceLimits::max_width`/`max_height`.
                                    Some((
                                        decode,
                                        "the frame exceeds the job's max_width/max_height",
                                    ))
                                } else if !rules.allow_progressive && matches!(m, 0xC2 | 0xCA) {
                                    Some((decode, "progressive JPEG rejected by the decode policy"))
                                } else if f.precision != 8 {
                                    // `JpegParser::decode`: probe() reports the
                                    // header, decode() refuses the precision.
                                    Some((decode, "12-bit precision is not supported"))
                                } else if f.frame.components == 2 {
                                    // The output stage (decode/parser/output.rs)
                                    // converts 1, 3 and 4 components only.
                                    Some((decode, "no colour conversion for 2 components"))
                                } else {
                                    None
                                };
                                if let Some((here, why)) = why {
                                    self.fail(&mut st, rules, node, here, Fails::Always, why);
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
        if st.eoi_end.is_none() && phase == Phase::Body && scans > 0 && rules.decode.is_strict() {
            // `TruncatedBetweenScans` / `TruncatedScan` are errors when Strict.
            st.decode_fatal = true;
        }
        // The coefficient path dequantises at the end of the decode.
        if let Some(f) = st.frame {
            defs.mark_quant(&f, COEFF);
        }
        let xyb = crate::color::icc::extract_icc_profile(&data[start..limit])
            .is_some_and(|p| crate::color::icc::is_xyb_profile(&p));
        st.paths = Paths::of(st.frame, st.stream_scan.is_some(), xyb);
        if st.paths.default == STREAM {
            // `to_pixels` returns the streaming result: every other scan is
            // decoded into coefficients that the default output never uses.
            for &(node, at) in &scan_parts {
                if Some(at) != st.stream_scan {
                    self.set(
                        node,
                        Disposition::Dropped,
                        Some(String::from(
                            "decoded, but unused by the default u8 RGB output, which comes \
                             from the frame's first all-component scan; used when the output \
                             needs coefficients (f32, Gray, CMYK, dequant bias)",
                        )),
                    );
                }
            }
        }
        self.settle_defs(&mut defs, st.paths)?;
        st.nodes = first..self.ids.len();
        Ok(st)
    }

    /// Definitions the default decode path never used: overwritten before
    /// use, never referenced, or of a kind the frame's coding does not use.
    /// A segment whose every definition is unused is `Dropped`; one that
    /// mixes both gets a child per used definition and one per run of
    /// unused ones. Where the other output path disagrees (quantisation
    /// tables only), the part says so.
    fn settle_defs(&mut self, defs: &mut Defs, paths: Paths) -> Result<(), InventoryError> {
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
            let recorded = &defs.used[from..u];
            let on_default = |d: &Used| d.paths & paths.default != 0;
            let used = recorded.iter().filter(|d| on_default(d)).count();
            // A Malformed segment the Permissive header parser skipped
            // after storing its first tables: those that a scan uses get
            // children; the segment stays Malformed.
            let malformed = match self.node(s.node).disposition {
                Disposition::Structure => false,
                Disposition::Malformed => true,
                _ => continue,
            };
            let notes: Vec<&str> = recorded
                .iter()
                .filter_map(|d| paths.note(d.paths))
                .collect();
            if used == s.count as usize && !malformed {
                if let Some(note) = notes.first() {
                    self.append_detail(s.node, note);
                }
                continue;
            }
            if recorded.is_empty() {
                if !malformed {
                    let what = if s.count == 1 {
                        format!("{}", s.first)
                    } else {
                        format!("{} definitions, the first {}", s.count, s.first)
                    };
                    self.set(s.node, Disposition::Dropped, Some(format!("{what}: {WHY}")));
                }
                continue;
            }
            if used == 0 && s.count == 1 && !malformed {
                // One definition, used only by the other output path.
                let note = notes.first().copied().unwrap_or(WHY);
                self.set(
                    s.node,
                    Disposition::Dropped,
                    Some(format!("{}: {note}", s.first)),
                );
                continue;
            }
            if used == 0 && !malformed {
                self.set(
                    s.node,
                    Disposition::Dropped,
                    Some(String::from(
                        "the default output path uses none of its definitions",
                    )),
                );
            }
            let mut at = s.area.start;
            for d in recorded {
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
                let disposition = if on_default(d) {
                    Disposition::Structure
                } else {
                    Disposition::Dropped
                };
                let detail = match paths.note(d.paths) {
                    Some(note) => format!("{}: {note}", d.what),
                    None => format!("{}", d.what),
                };
                self.add(
                    Some(s.node),
                    part(PartKind::Attribute, tag, d.range.clone(), disposition)
                        .with_detail(detail),
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
            // APP14 Adobe: every valid one parsed sets the colour transform
            // (`process_app_or_com`). The streaming path converts colour
            // during its scan, with the transform in effect then; the
            // coefficient path at the end, with the last one parsed.
            let valid: Vec<usize> = (0..st.apps.len())
                .filter(|&i| {
                    let a = &st.apps[i];
                    a.marker == MARKER_APP14 && a.ty == SegmentType::Adobe && a.payload.len() >= 12
                })
                .collect();
            let pick = |paths: u8| match paths {
                STREAM => valid.iter().copied().rfind(|&i| {
                    st.stream_scan
                        .is_some_and(|at| st.apps[i].payload.start < at)
                }),
                COEFF => valid.last().copied(),
                _ => None,
            };
            let (default_pick, other_pick) = (pick(st.paths.default), pick(st.paths.other));
            let components = st.frame.map_or(0, |f| f.components);
            for (i, a) in st.apps.iter().enumerate() {
                if a.marker != MARKER_APP14 || a.ty != SegmentType::Adobe {
                    continue;
                }
                let (d, why) = if a.payload.len() < 12 {
                    (
                        Disposition::Dropped,
                        String::from("too short to carry a colour transform"),
                    )
                } else if !(3..=4).contains(&components) {
                    (
                        Disposition::Dropped,
                        format!("colour transform unused for a {components}-component frame"),
                    )
                } else {
                    let paths = if Some(i) == default_pick {
                        st.paths.default
                    } else {
                        0
                    } | if Some(i) == other_pick {
                        st.paths.other
                    } else {
                        0
                    };
                    let note = st.paths.note(paths);
                    if Some(i) == default_pick {
                        if a.payload.len() > 12 {
                            extra.push((
                                a.node,
                                gap(
                                    a.payload.start + 12..a.payload.end,
                                    Disposition::Unreferenced,
                                )
                                .with_detail(
                                    "bytes after the APP14 fields; the decoder reads only the \
                                         transform",
                                ),
                            ));
                        }
                        let transform = data[a.payload.start + 11];
                        let mut why = format!("colour transform {transform} applied to the pixels");
                        if let Some(note) = note {
                            why = format!("{why}; {note}");
                        }
                        (Disposition::Metadata(MetadataKind::Colour), why)
                    } else if let Some(note) = note {
                        (Disposition::Dropped, String::from(note))
                    } else if st.paths.default == STREAM
                        && st.stream_scan.is_some_and(|at| a.payload.start > at)
                    {
                        (
                            Disposition::Dropped,
                            String::from(
                                "after the scan the default u8 RGB output converts colour in; an \
                                 output that needs coefficients uses a later APP14 Adobe segment",
                            ),
                        )
                    } else {
                        (
                            Disposition::Dropped,
                            String::from(
                                "superseded: a later APP14 Adobe segment sets the transform \
                                 before the colour conversion",
                            ),
                        )
                    }
                };
                decide(&mut decided[i], d, Some(why));
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
                                // parse_exif_orientation found a tag the
                                // entry search does not locate to the byte.
                                decided[i] = Some((
                                    Disposition::Skipped,
                                    Some(
                                        "find_exif_orientation applies an orientation from this \
                                         segment, at an entry the inventory does not locate; \
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
        if let Some(at) = orientation_from_inside {
            self.orientation_inside(st, at)?;
        }
        Ok(())
    }

    /// The orientation walk desynchronised (a single fill byte is enough)
    /// and read an APP1 EXIF header at `at`, inside another part. The part
    /// keeps its disposition; the 12-byte orientation entry the decoder
    /// applies becomes a `Field` child of the innermost part around it.
    fn orientation_inside(&mut self, st: &Stream, at: usize) -> Result<(), InventoryError> {
        let data = self.data;
        let entry = be16(data, at + 2).and_then(|n| {
            let payload = data.get(at + 4..at + 2 + n as usize)?;
            let r = exif_orientation_entry(payload)?;
            Some(at + 4 + r.start..at + 4 + r.end)
        });
        let nodes = st.nodes.start..self.ids.len();
        let around = |w: &Self, r: &Range<usize>| {
            nodes.clone().rev().find(|&n| {
                let p = &w.node(n).range;
                p.start <= r.start as u64 && r.end as u64 <= p.end
            })
        };
        if let Some(r) = entry
            && let Some(node) = around(self, &r)
            && self.inv.children(Some(self.ids[node])).iter().all(|&c| {
                let c = &self.inv.parts()[c.index()].range;
                c.end <= r.start as u64 || c.start >= r.end as u64
            })
        {
            self.add(
                Some(node),
                part(
                    PartKind::Field,
                    PartTag::Code(0x0112),
                    r,
                    Disposition::Metadata(MetadataKind::Orientation),
                )
                .with_detail(format!(
                    "EXIF orientation entry of an APP1 lookalike at offset {at}: \
                     find_exif_orientation applies it to the pixels"
                )),
            )?;
        } else if let Some(node) = around(self, &(at..at + 1)) {
            self.append_detail(
                node,
                &format!(
                    "find_exif_orientation applies an EXIF orientation read from an APP1 \
                     lookalike at offset {at} inside this part"
                ),
            );
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
                        Some(
                            "ISO 21496-1 gain-map metadata; no zencodec decode or probe path \
                             reads it: gain-map parameters come from hdrgm XMP \
                             (UltraHdrExtras::ultrahdr_metadata)"
                                .into(),
                        ),
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

    /// Map an embedded image's resolved parts to what its role makes of
    /// them. `xmp_at`: where the APP1 that `extract_xmp_from_jpeg` reads for
    /// the gain-map parameters starts, when the decode takes them from it.
    fn apply_role(&mut self, nodes: Range<usize>, role: Role, xmp_at: Option<usize>) {
        for n in nodes {
            let p = self.node(n);
            let d = p.disposition;
            let is_xmp_segment =
                p.kind == PartKind::Segment && Some(p.range.start) == xmp_at.map(|a| a as u64);
            let (new, detail) = match (role, d) {
                (Role::GainMap, Disposition::Structure | Disposition::ImageData) => {
                    (Disposition::Metadata(MetadataKind::GainMap), None)
                }
                (Role::GainMap, Disposition::Metadata(MetadataKind::Colour)) => {
                    (Disposition::Metadata(MetadataKind::GainMap), None)
                }
                (Role::GainMap, Disposition::Metadata(MetadataKind::Xmp)) if is_xmp_segment => (
                    Disposition::Metadata(MetadataKind::GainMap),
                    Some(
                        "the gain-map parameters: ultrahdr_metadata() reads this packet \
                         (extract_xmp_from_jpeg) because the primary XMP carries none",
                    ),
                ),
                (Role::GainMap, Disposition::Metadata(MetadataKind::Xmp)) => (
                    Disposition::Dropped,
                    Some(
                        "the gain-map decode keeps only pixels; ultrahdr_metadata() reads the \
                         primary XMP, or the first standard XMP packet of this image only",
                    ),
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

    /// Add an embedded image at `r` under `parent` and walk it when it
    /// starts with SOI.
    fn embedded(
        &mut self,
        parent: Option<usize>,
        mut p: Part,
        r: Range<usize>,
        role: Role,
        xmp_at: Option<usize>,
    ) -> Result<usize, InventoryError> {
        let is_jpeg = self.data.get(r.start..r.start + 2) == Some(&[0xFF, MARKER_SOI][..]);
        if is_jpeg {
            p = p.with_body(r.start as u64..r.end as u64);
        }
        let node = self.add(parent, p)?;
        if is_jpeg {
            let st = self.walk_stream(Some(node), r.start, r.end, Rules::embedded())?;
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
            self.apply_role(all, role, xmp_at);
        }
        Ok(node)
    }

    /// Whether the container-level walk of the image at `r` finds the
    /// failures that would make `decode_gainmap_jpeg` fail (a default
    /// `Decoder`). Entropy-level failures stay invisible, as for the
    /// primary image. Walks a scratch inventory that is then dropped.
    fn stream_decodes(&self, r: Range<usize>) -> bool {
        let mut w = Walk {
            data: self.data,
            inv: Inventory::new(ImageFormat::Jpeg, self.data.len() as u64),
            ids: Vec::new(),
            opts: self.opts,
            repeats: [(None, 0); REPEAT_KINDS],
        };
        w.walk_stream(None, r.start, r.end, Rules::embedded())
            .is_ok_and(|st| !st.decode_fatal)
    }

    /// What the job's decode does with the Ultra HDR gain map, decided by
    /// the calls the decode itself makes: `ultrahdr_metadata()` and
    /// `gainmap()` on extras built the way the decoder builds them (every
    /// XMP segment, and the first extracted `Undefined` MPF image), and the
    /// `decode_gainmap()` outcome approximated by [`Self::stream_decodes`].
    #[cfg(feature = "ultrahdr")]
    fn gain_map_use(&self, st: &Stream, images: &[MpfImage]) -> GainMap {
        use crate::ultrahdr::UltraHdrExtras;
        let required = match self.opts.render {
            Render::Components => false,
            Render::Reconstruct { hdrgm: true } => true,
            _ => return GainMap::None,
        };
        let data = self.data;
        let mut ex = DecodedExtras::new();
        for a in &st.apps {
            if matches!(a.ty, SegmentType::Xmp | SegmentType::XmpExtended) {
                ex.add_segment(a.marker, data[a.payload.clone()].to_vec(), a.ty);
            }
        }
        let candidate = images
            .iter()
            .find(|m| m.extracted && m.ty == MpfImageType::Undefined);
        if let Some(m) = candidate {
            ex.add_secondary_image(m.idx, m.ty, data[m.range.clone()].to_vec());
        }
        match ex.ultrahdr_metadata() {
            None if required => {
                GainMap::Fails("ReconstructHdr: ultrahdr_metadata() finds no gain-map parameters")
            }
            None => GainMap::None,
            Some(Err(_)) => GainMap::Fails(
                "ultrahdr_metadata() returns an error (no hdrgm parameters in the XMP it reads)",
            ),
            Some(Ok(_)) => match candidate {
                None if required => GainMap::Fails(
                    "ReconstructHdr: the MPF index names no extracted gain-map image",
                ),
                None => GainMap::None,
                Some(m) if !self.stream_decodes(m.range.clone()) => {
                    GainMap::Fails("the gain-map image fails to decode")
                }
                Some(m) => {
                    let gm = &data[m.range.clone()];
                    let from_gain_map = crate::ultrahdr::primary_xmp_gain_map(&ex).is_none()
                        && crate::ultrahdr::extract_xmp_from_jpeg(gm).is_some();
                    GainMap::Decoded {
                        idx: m.idx,
                        xmp_at: from_gain_map
                            .then(|| crate::ultrahdr::xmp_segment_range(gm))
                            .flatten()
                            .map(|r| m.range.start + r.start),
                    }
                }
            },
        }
    }

    #[cfg(not(feature = "ultrahdr"))]
    fn gain_map_use(&self, _st: &Stream, _images: &[MpfImage]) -> GainMap {
        GainMap::None
    }

    /// The innermost part of the primary stream that holds all of `r` and
    /// has no child overlapping it: where an MPF image inside the primary
    /// is nested.
    fn host_for(&self, st: &Stream, r: &Range<usize>) -> Option<usize> {
        let (a, b) = (r.start as u64, r.end as u64);
        let host = (st.nodes.start..self.ids.len()).rev().find(|&n| {
            let p = &self.node(n).range;
            p.start <= a && b <= p.end
        })?;
        let clear = self.inv.children(Some(self.ids[host])).iter().all(|&c| {
            let c = &self.inv.parts()[c.index()].range;
            c.end <= a || c.start >= b
        });
        clear.then_some(host)
    }

    /// Everything after the primary EOI, in the order the decoder's
    /// precedence gives it: MPF images (which the decode extracts wherever
    /// they lie, inside the primary too), then the Samsung SEF trailer and
    /// GContainer items in the space still free.
    fn after_eoi(
        &mut self,
        st: &Stream,
        eoi_end: usize,
        mpf_node: Option<usize>,
        images: &[MpfImage],
        gain_map: GainMap,
    ) -> Result<(), InventoryError> {
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
        let (gain_map_idx, xmp_at) = match gain_map {
            GainMap::Decoded { idx, xmp_at } => (Some(idx), xmp_at),
            _ => (None, None),
        };

        // MPF images (`extract_mpf_secondary_images`).
        let mut notes: Vec<String> = Vec::new();
        let mut mpf_parts: Vec<(Range<usize>, usize)> = Vec::new();
        for m in images {
            let idx = m.idx;
            let type_code = m.ty.type_code();
            let r = m.range.clone();
            let role = if decode_ok && gain_map_idx == Some(idx) {
                Role::GainMap
            } else {
                Role::Unread
            };
            let in_data = r.end <= len && r.start < r.end;
            let (d, why) = if role == Role::GainMap {
                (
                    Disposition::Metadata(MetadataKind::GainMap),
                    "decoded as the Ultra HDR gain map",
                )
            } else if !in_data {
                (Disposition::Skipped, "lies outside the data; not extracted")
            } else if !m.extracted {
                (
                    Disposition::Skipped,
                    "does not start with SOI; the decoder ignores it",
                )
            } else if !decode_ok {
                (Disposition::Skipped, "not read: the decode fails")
            } else {
                (
                    Disposition::Skipped,
                    "copied to native DecodedExtras::secondary_images only",
                )
            };
            if !in_data {
                notes.push(format!(
                    "entry {idx} ({} bytes at relative offset {}) {why}",
                    m.size, m.offset
                ));
                continue;
            }
            let p = part(
                PartKind::EmbeddedImage,
                PartTag::Code(idx as u32),
                r.clone(),
                d,
            )
            .with_label("MPF")
            .with_detail(format!("MPF image {idx}, type {type_code:#08x}; {why}"));
            let at = if role == Role::GainMap { xmp_at } else { None };
            if free(&placed, &r) {
                let node = self.embedded(None, p, r.clone(), role, at)?;
                mpf_parts.push((r.clone(), node));
                placed.push(r);
            } else if r.end <= eoi_end
                && let Some(host) = self.host_for(st, &r)
            {
                // Inside the primary: the bytes are both the part around
                // them and this image.
                let node = self.embedded(Some(host), p, r.clone(), role, at)?;
                mpf_parts.push((r.clone(), node));
            } else {
                notes.push(format!(
                    "entry {idx} at {}..{} {why}; it overlaps other parts, so it has no part \
                     of its own",
                    r.start, r.end
                ));
            }
        }
        if !notes.is_empty()
            && let Some(node) = mpf_node
        {
            self.append_detail(node, &notes.join("; "));
        }

        // Samsung SEF trailer (the layout ExifTool's Samsung module reads),
        // in the space MPF images leave free.
        let mut seft_node = None;
        if let Some(t) = samsung_trailer(data, eoi_end)
            && free(&placed, &t.directory)
        {
            let floor = placed
                .iter()
                .map(|p| p.end)
                .filter(|&e| e <= t.directory.start)
                .max()
                .unwrap_or(eoi_end);
            let blocks: Vec<_> = t
                .blocks
                .into_iter()
                .filter(|(r, _, _)| r.start >= floor && free(&placed, r))
                .collect();
            let start = blocks
                .iter()
                .map(|(r, _, _)| r.start)
                .min()
                .unwrap_or(t.directory.start);
            let node = self.add(
                None,
                part(
                    PartKind::Trailer,
                    PartTag::None,
                    start..len,
                    Disposition::Skipped,
                )
                .with_label("SEFT")
                .with_detail(format!(
                    "Samsung SEF trailer, {} entries; the decoder does not read it",
                    t.count
                )),
            )?;
            for (r, ty, name) in blocks {
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
            placed.push(start..len);
            seft_node = Some(node);
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
                if let Some(&(_, n)) = mpf_parts.iter().find(|(m, _)| *m == r) {
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
                    self.embedded(None, p, r.clone(), Role::Unread, None)?;
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
    width: u16,
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
    let mut samp = [0u8; 4];
    for (c, id) in ids.iter_mut().enumerate().take(nc as usize) {
        let at = 6 + 3 * c;
        *id = body[at];
        samp[c] = body[at + 1];
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
            samp,
            width,
            height,
            mode: 0,
        },
        width,
        height,
        precision,
    })
}

/// The decoder's verdict on a scan header (`parse_scan`): the first
/// problem, and the strictness levels it fails at.
fn check_sos(
    data: &[u8],
    pos: usize,
    ns: u8,
    frame: Option<Frame>,
) -> Option<(Fails, &'static str)> {
    let fail = |why| Some((Fails::Always, why));
    let Some(frame) = frame else {
        return fail("no frame header");
    };
    if ns == 0 {
        return fail("scan with zero components");
    }
    if ns > frame.components {
        return fail("scan has more components than the frame");
    }
    let mut seen = [false; 4];
    let mut bad_table = false;
    for c in 0..ns as usize {
        let at = pos + 5 + 2 * c;
        let (id, tables) = (data[at], data[at + 1]);
        // Permissive clamps an out-of-range selector to 0 and goes on.
        bad_table |= tables >> 4 >= 4 || tables & 0x0F >= 4;
        if bad_table {
            return Some((Fails::UnlessPermissive, "Huffman table index out of range"));
        }
        let Some(idx) = frame.ids[..frame.components as usize]
            .iter()
            .position(|&x| x == id)
        else {
            return fail("unknown component in scan");
        };
        if seen[idx] {
            return fail("duplicate component in scan");
        }
        seen[idx] = true;
    }
    let at = pos + 5 + 2 * ns as usize;
    let (ss, se, ahal) = (data[at], data[at + 1], data[at + 2]);
    if ss > 63 || se > 63 {
        return fail("spectral selection beyond 63");
    }
    if ss > se {
        return fail("spectral selection start exceeds end");
    }
    if ahal >> 4 > 13 || ahal & 0x0F > 13 {
        return fail("successive approximation out of range");
    }
    // `decode_progressive_scan`: only DC scans (Ss = Se = 0) may interleave.
    if frame.mode == 0xC2 && !(ss == 0 && se == 0) && ns != 1 {
        return fail("progressive AC scan must have a single component");
    }
    None
}

struct SamsungTrailer {
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
        directory: dir_pos..len,
        count,
        blocks,
    })
}

mod entropy;

#[cfg(test)]
mod tests;
