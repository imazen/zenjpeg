//! Encoder parameter record (APP15 `ZENJ`): authoritative encode history
//! instead of fingerprint estimation (zenjpeg#189).
//!
//! A JPEG carries no record of how it was encoded beyond its tables, so
//! [`probe`](super::probe) *estimates* encoder family and quality. For our own
//! output the estimate is measurably lossy — Robidoux-table zenjpeg output
//! probes as `Unknown` with the quality 10-15 points low, and every invisible
//! knob (adaptive quantization, trellis, pre-blur) is gone. This record carries
//! the truth, and a per-generation hash of the quantization tables lets a
//! reader tell whether the record still describes the tables in front of it.
//!
//! # What goes in, and why
//!
//! Fields were chosen by measurement, not by listing the encoder's knobs
//! (zensr `benchmarks/zenjpeg_knobs_2026-10-30.md`: restoration gain per knob
//! at an equal quality dial, paired per file on clean references):
//!
//! - **quantization-table family** dominates (−25 SSIM2 of input fidelity
//!   between jpegli and Robidoux tables at q15) and is where the fingerprinter
//!   fails on our own output → `table_source` + the dial on its native scale;
//! - **adaptive quantization** is the only header-invisible knob above the
//!   metric floor (+0.3..0.6 SSIM2 restoration gain without it at q15-35,
//!   80-92 % of files) → `AQ` flag;
//! - **pre-blur** shifts the input by −3 SSIM2 and the gain by −0.2 → `pre_blur`;
//! - **trellis** moves things by ~0.1-0.15 (at the floor) → a flag, useful to
//!   `recompress`, not for model routing;
//! - deringing, scan script and Huffman strategy have no measured restoration
//!   effect; they ride along as flags because recompression wants them.
//!
//! # History, not a snapshot
//!
//! Generation loss is where header estimation fails hardest: a q60 file
//! re-encoded at q90 carries q90 tables and q60 damage (zensr `gen_eval`,
//! chain `g2-upq`). A record that described only the last encode would lie
//! in exactly the same way, so the record is an **append-only chain**: each
//! encoder appends one [`EncodeEntry`] and keeps the ones it was handed
//! (newest last, oldest dropped past [`MAX_ENTRIES`]). Severity for gating is
//! the worst entry, not the last.
//!
//! The newest entry's `dqt_crc32` must match the file's tables; otherwise the
//! tables were rewritten after the record (a transcode that copied markers)
//! and the whole chain is *prior context* only, never a description of this
//! file. [`EncodeRecord::verified`] carries that verdict.
//!
//! # Wire format (version 1)
//!
//! APP15 (`0xFFEF`) segment. The `ZENJ\0` signature distinguishes this
//! record from other applications using APP15. Payload:
//!
//! ```text
//! "ZENJ\0"            5  signature
//! u8 version          1  = 1
//! u8 n_entries        1  1..=MAX_ENTRIES
//! entries             20 bytes each, oldest first:
//!   u8   encoder          EncoderId
//!   u8 u8 u8  version     semver major, minor, patch of that encoder
//!   u8   quality_scale    DialScale
//!   f32  quality_value    little-endian, on that scale
//!   u8   table_source     TableSource
//!   u8   color_front      ColorFront
//!   u8   subsampling      SubsamplingCode
//!   u8   flags            EncodeFlags bits
//!   u8   pre_blur_x16     Gaussian sigma * 16, saturating (0 = none)
//!   u8   reserved         0
//!   u8   reserved         0
//!   u32  dqt_crc32        little-endian, see [`dqt_crc32`]
//! ```
//!
//! Everything after a known entry count is ignored by this reader, so a
//! future version can append fields per entry only by bumping `version`;
//! unknown versions are skipped (the record is an optimization, probing
//! falls back to estimation).
//!
//! Metadata strippers drop APP15 like any other APPn. The record is never
//! load-bearing: a file without one behaves exactly as before.

use alloc::vec::Vec;

use crate::foundation::consts::{MARKER_DQT, MARKER_SOS};

/// APP15 marker byte used for the record.
pub const MARKER_APP15: u8 = 0xEF;
/// Payload signature.
pub const SIGNATURE: &[u8; 5] = b"ZENJ\0";
/// Current wire-format version.
pub const VERSION: u8 = 1;
/// Oldest entries are dropped past this many.
pub const MAX_ENTRIES: usize = 8;
const ENTRY_LEN: usize = 20;

/// Which encoder wrote an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncoderId {
    /// zenjpeg.
    Zenjpeg,
    /// A value this reader does not know.
    Unknown(u8),
}

/// Scale of the quality dial as the caller set it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DialScale {
    /// jpegli-style 0-100 (zenjpeg `Quality::ApproxJpegli`).
    Jpegli,
    /// mozjpeg/libjpeg 0-100 (`Quality::ApproxMozjpeg`).
    Mozjpeg,
    /// SSIMULACRA2 target (`Quality::ApproxSsim2`).
    Ssim2,
    /// Butteraugli distance (`Quality::ApproxButteraugli`).
    Butteraugli,
    /// zensim Profile A target, as `recompress` takes it.
    ZensimA,
    /// A value this reader does not know.
    Unknown(u8),
}

/// Quantization-table family the encoder used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TableSource {
    /// jpegli perceptual tables, three (Y, Cb, Cr).
    Jpegli,
    /// jpegli tables, shared chroma (two tables).
    JpegliSharedChroma,
    /// mozjpeg Robidoux tables.
    MozjpegRobidoux,
    /// Caller-supplied tables.
    Custom,
    /// zenjpeg piecewise v4 tables.
    PiecewiseV4,
    /// zenjpeg Glassa low-bpp tables.
    GlassaLowBpp,
    /// A value this reader does not know.
    Unknown(u8),
}

/// Colour front-end of the encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorFront {
    /// Standard YCbCr.
    YCbCr,
    /// XYB (jpegli-style, ICC-tagged).
    Xyb,
    /// Single-component grayscale.
    Grayscale,
    /// RGB without colour transform.
    Rgb,
    /// A value this reader does not know.
    Unknown(u8),
}

/// Chroma layout as the caller set it (cross-check against SOF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SubsamplingCode {
    /// 4:4:4 (or XYB full).
    Full,
    /// 4:2:0 (or XYB B-quarter).
    Quarter,
    /// 4:2:2.
    HalfHorizontal,
    /// 4:4:0.
    HalfVertical,
    /// Not applicable (grayscale, RGB).
    NotApplicable,
    /// A value this reader does not know.
    Unknown(u8),
}

/// Boolean knobs of one encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EncodeFlags(pub u8);

impl EncodeFlags {
    /// Adaptive quantization was on.
    pub const AQ: u8 = 1 << 0;
    /// AC trellis quantization was on.
    pub const TRELLIS_AC: u8 = 1 << 1;
    /// DC trellis was on.
    pub const TRELLIS_DC: u8 = 1 << 2;
    /// Overshoot deringing was on.
    pub const DERINGING: u8 = 1 << 3;
    /// Progressive scan script.
    pub const PROGRESSIVE: u8 = 1 << 4;
    /// Scan-script search ran.
    pub const SCAN_SEARCH: u8 = 1 << 5;
    /// Gamma-aware (sharp) chroma downsampling.
    pub const GAMMA_AWARE_CHROMA: u8 = 1 << 6;

    /// Whether `bit` is set.
    #[must_use]
    pub fn has(self, bit: u8) -> bool {
        self.0 & bit != 0
    }
}

/// One generation of the encode history.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodeEntry {
    /// Which encoder.
    pub encoder: EncoderId,
    /// That encoder's semver (major, minor, patch).
    pub version: (u8, u8, u8),
    /// Scale of `quality`.
    pub quality_scale: DialScale,
    /// Quality dial as set, on `quality_scale`.
    pub quality: f32,
    /// Quantization-table family.
    pub table_source: TableSource,
    /// Colour front-end.
    pub color_front: ColorFront,
    /// Chroma layout as set.
    pub subsampling: SubsamplingCode,
    /// Boolean knobs.
    pub flags: EncodeFlags,
    /// Pre-encode Gaussian blur sigma (0.0 = none), 1/16 resolution.
    pub pre_blur: f32,
    /// CRC-32 of the DQT tables this generation wrote ([`dqt_crc32`]).
    pub dqt_crc32: u32,
}

/// The parsed record: the chain plus whether its newest entry matches the
/// file's tables.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodeRecord {
    /// Oldest first; the last entry is the encode that wrote these bytes
    /// (when `verified`).
    pub entries: Vec<EncodeEntry>,
    /// The newest entry's `dqt_crc32` equals the CRC of the DQT tables in
    /// this file. When false, the tables were rewritten after the record and
    /// `entries` are prior generations only.
    pub verified: bool,
}

impl EncodeRecord {
    /// The entry describing this file's coefficients, if the record is
    /// verified.
    #[must_use]
    pub fn current(&self) -> Option<&EncodeEntry> {
        if self.verified {
            self.entries.last()
        } else {
            None
        }
    }

    /// Number of recorded generations.
    #[must_use]
    pub fn generations(&self) -> usize {
        self.entries.len()
    }

    /// Serialize as an APP15 payload (signature included, marker and length
    /// excluded), keeping the newest [`MAX_ENTRIES`].
    #[must_use]
    pub fn to_payload(&self) -> Vec<u8> {
        let skip = self.entries.len().saturating_sub(MAX_ENTRIES);
        let entries = &self.entries[skip..];
        let mut out = Vec::with_capacity(7 + ENTRY_LEN * entries.len());
        out.extend_from_slice(SIGNATURE);
        out.push(VERSION);
        out.push(entries.len() as u8);
        for e in entries {
            out.push(e.encoder.code());
            out.extend_from_slice(&[e.version.0, e.version.1, e.version.2]);
            out.push(e.quality_scale.code());
            out.extend_from_slice(&e.quality.to_le_bytes());
            out.push(e.table_source.code());
            out.push(e.color_front.code());
            out.push(e.subsampling.code());
            out.push(e.flags.0);
            out.push((e.pre_blur * 16.0).round().clamp(0.0, 255.0) as u8);
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&e.dqt_crc32.to_le_bytes());
        }
        out
    }

    /// Parse an APP15 payload. `None` for anything that is not a record this
    /// reader understands. `verified` is left `false`; see
    /// [`EncodeRecord::verify_against`].
    #[must_use]
    pub fn from_payload(p: &[u8]) -> Option<Self> {
        if p.len() < 7 || &p[..5] != SIGNATURE || p[5] != VERSION {
            return None;
        }
        let n = p[6] as usize;
        if n == 0 || n > MAX_ENTRIES || p.len() < 7 + n * ENTRY_LEN {
            return None;
        }
        let entries = p[7..7 + n * ENTRY_LEN]
            .as_chunks::<ENTRY_LEN>()
            .0
            .iter()
            .map(|c| EncodeEntry {
                encoder: EncoderId::from_code(c[0]),
                version: (c[1], c[2], c[3]),
                quality_scale: DialScale::from_code(c[4]),
                quality: f32::from_le_bytes([c[5], c[6], c[7], c[8]]),
                table_source: TableSource::from_code(c[9]),
                color_front: ColorFront::from_code(c[10]),
                subsampling: SubsamplingCode::from_code(c[11]),
                flags: EncodeFlags(c[12]),
                pre_blur: c[13] as f32 / 16.0,
                dqt_crc32: u32::from_le_bytes([c[16], c[17], c[18], c[19]]),
            })
            .collect();
        Some(Self {
            entries,
            verified: false,
        })
    }

    /// Set `verified` from the CRC of the tables actually in `jpeg`.
    pub fn verify_against(&mut self, jpeg: &[u8]) {
        self.verified =
            dqt_crc32(jpeg).is_some_and(|c| self.entries.last().is_some_and(|e| e.dqt_crc32 == c));
    }
}

/// The entries a re-encode of `jpeg` should carry forward: the source's
/// chain when its record still describes the source's tables, nothing
/// otherwise (an unverified chain describes some earlier file, and
/// claiming continuity through an unknown generation would be a lie).
#[must_use]
pub fn history_of(jpeg: &[u8]) -> Vec<EncodeEntry> {
    match read_record(jpeg) {
        Some(r) if r.verified => r.entries,
        _ => Vec::new(),
    }
}

/// This crate's version as a record triple.
pub(crate) fn crate_version() -> (u8, u8, u8) {
    let mut parts = env!("CARGO_PKG_VERSION")
        .split('.')
        .map(|x| x.parse::<u8>().unwrap_or(255));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

/// Splice a record for `entries` into `jpeg` (before the first DQT), with
/// the newest entry's `dqt_crc32` set from the tables in `jpeg`. Used by
/// coefficient-domain rewrites that do not go through the encoder. No-op
/// on an empty chain.
#[cfg(any(test, feature = "recompress"))]
pub(crate) fn insert_record(jpeg: &mut Vec<u8>, mut entries: Vec<EncodeEntry>) {
    let (Some(crc), Some(last)) = (dqt_crc32(jpeg), entries.last_mut()) else {
        return;
    };
    last.dqt_crc32 = crc;
    let payload = EncodeRecord {
        entries,
        verified: true,
    }
    .to_payload();
    let at = record_insert_position(jpeg);
    let len = payload.len() + 2;
    let mut seg = Vec::with_capacity(len + 2);
    seg.extend_from_slice(&[0xFF, MARKER_APP15, (len >> 8) as u8, len as u8]);
    seg.extend_from_slice(&payload);
    jpeg.splice(at..at, seg);
}

/// Extract the record from a JPEG's headers (first APP15 `ZENJ` segment
/// before SOS), verified against the file's DQT tables. `None` when absent
/// or unreadable.
#[must_use]
pub fn read_record(jpeg: &[u8]) -> Option<EncodeRecord> {
    let mut rec = None;
    for_each_segment(jpeg, |marker, payload| {
        // The visitor stops as soon as a record is found.
        if marker == MARKER_APP15 {
            rec = EncodeRecord::from_payload(payload);
        }
        rec.is_none()
    });
    let mut rec = rec?;
    rec.verify_against(jpeg);
    Some(rec)
}

/// CRC-32 (IEEE, as in PNG/zlib) over the file's quantization tables, in
/// the order they appear: for each table `Pq | Tq` byte followed by its 64
/// values as big-endian u16 (zigzag order as stored, so the hash is of the
/// segment contents, including its 8/16-bit storage precision). `None` if there
/// is no DQT before SOS.
#[must_use]
pub fn dqt_crc32(jpeg: &[u8]) -> Option<u32> {
    let mut crc = Crc32::new();
    let mut seen = false;
    for_each_segment(jpeg, |marker, payload| {
        if marker == MARKER_DQT {
            let mut p = payload;
            while let Some(&pqtq) = p.first() {
                let wide = pqtq >> 4 != 0;
                let n = if wide { 129 } else { 65 };
                if p.len() < n {
                    break;
                }
                seen = true;
                crc.update(&[pqtq]);
                if wide {
                    crc.update(&p[1..129]);
                } else {
                    for &v in &p[1..65] {
                        crc.update(&[0, v]);
                    }
                }
                p = &p[n..];
            }
        }
        true
    });
    seen.then(|| crc.finish())
}

/// Walk marker segments from SOI to SOS (exclusive), calling `f(marker,
/// payload)`; stop early when `f` returns false.
fn for_each_segment(jpeg: &[u8], mut f: impl FnMut(u8, &[u8]) -> bool) {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return;
    }
    let mut pos = 2;
    while pos + 4 <= jpeg.len() {
        if jpeg[pos] != 0xFF {
            return;
        }
        let marker = jpeg[pos + 1];
        if marker == 0xFF {
            pos += 1;
            continue;
        }
        if marker == MARKER_SOS || marker == 0xD9 {
            return;
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            pos += 2;
            continue;
        }
        let len = ((jpeg[pos + 2] as usize) << 8) | jpeg[pos + 3] as usize;
        if len < 2 || pos + 2 + len > jpeg.len() {
            return;
        }
        if !f(marker, &jpeg[pos + 4..pos + 2 + len]) {
            return;
        }
        pos += 2 + len;
    }
}

/// Position just before the first DQT (or SOS) marker: where the record is
/// inserted so it follows every other APPn the encoder wrote.
pub(crate) fn record_insert_position(jpeg: &[u8]) -> usize {
    let mut at = jpeg.len();
    let mut pos = 2;
    while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF {
        let marker = jpeg[pos + 1];
        if marker == MARKER_DQT || marker == MARKER_SOS || (0xC0..=0xCF).contains(&marker) {
            at = pos;
            break;
        }
        let len = ((jpeg[pos + 2] as usize) << 8) | jpeg[pos + 3] as usize;
        if len < 2 {
            break;
        }
        pos += 2 + len;
    }
    at
}

/// Bitwise CRC-32 (IEEE 802.3, reflected, init/xorout 0xFFFFFFFF). Input is
/// at most a few hundred bytes, so no table is needed.
struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Self(0xFFFF_FFFF)
    }
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u32;
            for _ in 0..8 {
                let mask = (self.0 & 1).wrapping_neg();
                self.0 = (self.0 >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
    }
    fn finish(self) -> u32 {
        !self.0
    }
}

macro_rules! codes {
    ($t:ident { $($v:ident = $c:literal),* $(,)? }) => {
        impl $t {
            fn code(self) -> u8 {
                match self { $($t::$v => $c,)* $t::Unknown(c) => c }
            }
            fn from_code(c: u8) -> Self {
                match c { $($c => $t::$v,)* c => $t::Unknown(c) }
            }
        }
    };
}
codes!(EncoderId { Zenjpeg = 1 });
codes!(DialScale { Jpegli = 0, Mozjpeg = 1, Ssim2 = 2, Butteraugli = 3, ZensimA = 4 });
codes!(TableSource {
    Jpegli = 0,
    JpegliSharedChroma = 1,
    MozjpegRobidoux = 2,
    Custom = 3,
    PiecewiseV4 = 4,
    GlassaLowBpp = 5,
});
codes!(ColorFront { YCbCr = 0, Xyb = 1, Grayscale = 2, Rgb = 3 });
codes!(SubsamplingCode {
    Full = 0,
    Quarter = 1,
    HalfHorizontal = 2,
    HalfVertical = 3,
    NotApplicable = 4,
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_reference_vector() {
        let mut c = Crc32::new();
        c.update(b"123456789");
        assert_eq!(c.finish(), 0xCBF4_3926);
    }

    fn entry(q: f32, crc: u32) -> EncodeEntry {
        EncodeEntry {
            encoder: EncoderId::Zenjpeg,
            version: (0, 9, 0),
            quality_scale: DialScale::Jpegli,
            quality: q,
            table_source: TableSource::MozjpegRobidoux,
            color_front: ColorFront::YCbCr,
            subsampling: SubsamplingCode::Quarter,
            flags: EncodeFlags(EncodeFlags::AQ | EncodeFlags::TRELLIS_AC),
            pre_blur: 0.4375,
            dqt_crc32: crc,
        }
    }

    #[test]
    fn coefficient_rewrite_inserts_app15_with_verified_tables() {
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xDB, 0, 67, 0];
        jpeg.extend_from_slice(&[1; 64]);
        jpeg.extend_from_slice(&[0xFF, 0xDA, 0, 2]);
        insert_record(&mut jpeg, vec![entry(70.0, 0)]);
        assert_eq!(&jpeg[2..4], &[0xFF, 0xEF]);
        let record = read_record(&jpeg).unwrap();
        assert!(record.verified);
        assert_eq!(record.current().unwrap().quality, 70.0);
    }

    #[test]
    fn unknown_wire_codes_roundtrip_without_aliasing_known_values() {
        for code in 0..=255 {
            assert_eq!(EncoderId::from_code(code).code(), code);
            assert_eq!(DialScale::from_code(code).code(), code);
            assert_eq!(TableSource::from_code(code).code(), code);
            assert_eq!(ColorFront::from_code(code).code(), code);
            assert_eq!(SubsamplingCode::from_code(code).code(), code);
        }
    }

    #[test]
    fn truncated_payloads_and_invalid_counts_are_rejected() {
        let record = EncodeRecord {
            entries: vec![entry(70.0, 0)],
            verified: false,
        };
        assert!(record.current().is_none());
        let payload = record.to_payload();
        for end in 0..payload.len() {
            assert!(EncodeRecord::from_payload(&payload[..end]).is_none());
        }
        for (offset, value) in [(0, 0), (5, 99), (6, 0), (6, 9)] {
            let mut bad = payload.clone();
            bad[offset] = value;
            assert!(EncodeRecord::from_payload(&bad).is_none());
        }
    }

    #[test]
    fn malformed_headers_do_not_yield_records_or_table_checksums() {
        for bytes in [
            vec![],
            vec![0; 4],
            vec![255, 0, 0, 0],
            vec![255, 216, 0, 0, 0, 0],
            vec![255, 216, 255, 217, 0, 0],
            vec![255, 216, 255, 218, 0, 0],
            vec![255, 216, 255, 224, 0, 1],
            vec![255, 216, 255, 224, 0, 8],
            vec![255, 216, 255, 255, 224, 0, 2],
            vec![255, 216, 255, 208, 255, 1, 255, 224, 0, 2],
            vec![255, 216, 255, 219, 0, 3, 0],
            vec![255, 216, 255, 219, 0, 3, 16],
        ] {
            assert!(read_record(&bytes).is_none());
            assert!(dqt_crc32(&bytes).is_none());
            assert!(history_of(&bytes).is_empty());
        }
    }

    #[test]
    fn wide_tables_hash_their_full_precision_and_history_requires_a_match() {
        let mut jpeg = vec![255, 216, 255, 219, 0, 131, 16];
        for _ in 0..64 {
            jpeg.extend_from_slice(&[1, 2]);
        }
        jpeg.extend_from_slice(&[255, 218, 0, 2]);
        let crc = dqt_crc32(&jpeg).unwrap();
        let mut reference = Crc32::new();
        reference.update(&[16]);
        for _ in 0..64 {
            reference.update(&[1, 2]);
        }
        assert_eq!(crc, reference.finish());
        insert_record(&mut jpeg, vec![entry(70.0, 0)]);
        assert_eq!(history_of(&jpeg).len(), 1);
        let dqt = jpeg.windows(2).position(|b| b == [255, 219]).unwrap();
        jpeg[dqt + 5] ^= 1;
        assert!(history_of(&jpeg).is_empty());
        let mut empty = vec![255, 216, 255, 218, 0, 2];
        let original = empty.clone();
        insert_record(&mut empty, vec![]);
        assert_eq!(empty, original);
        insert_record(&mut jpeg, vec![]);
    }

    #[test]
    fn record_position_follows_metadata_and_precedes_image_markers() {
        assert_eq!(record_insert_position(&[255, 216]), 2);
        assert_eq!(record_insert_position(&[255, 216, 0, 224, 0, 2]), 6);
        assert_eq!(record_insert_position(&[255, 216, 255, 224, 0, 1]), 6);
        assert_eq!(record_insert_position(&[255, 216, 255, 224, 0, 2]), 6);
        for marker in [MARKER_DQT, MARKER_SOS, 0xC0] {
            assert_eq!(
                record_insert_position(&[255, 216, 255, 224, 0, 2, 255, marker, 0, 2]),
                6
            );
        }
    }

    #[test]
    fn payload_roundtrip_and_cap() {
        let rec = EncodeRecord {
            entries: (0..10).map(|i| entry(i as f32, i)).collect(),
            verified: true,
        };
        let p = rec.to_payload();
        assert_eq!(p.len(), 7 + ENTRY_LEN * MAX_ENTRIES);
        let back = EncodeRecord::from_payload(&p).unwrap();
        assert_eq!(back.entries, rec.entries[2..].to_vec());
        assert!(!back.verified);
        assert!(EncodeRecord::from_payload(&p[..20]).is_none());
        let mut bad = p.clone();
        bad[5] = 99;
        assert!(EncodeRecord::from_payload(&bad).is_none());
    }
}
