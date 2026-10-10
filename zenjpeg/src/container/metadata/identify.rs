//! Signature identification of marker segments and their fixed-layout fields.
//!
//! One function, [`identify`], maps `(marker, payload)` to an
//! [`ItemKind`]. Signatures and layouts follow ExifTool's `ProcessJPEG` and
//! per-format readers. Where 6.99 and the 12.40 oracle differ, the 12.40
//! rule is used and noted (JFXX extension codes 0x11/0x13, the 14-byte ICC
//! minimum); 12.40 leniencies that 6.99 lacks (case-insensitive `Exif`,
//! stray bytes before it, signature-less XMP) are not followed.

use super::ItemKind;

const XMP_NAMESPACE: &[u8] = crate::foundation::consts::XMP_NAMESPACE;
const EXTENDED_XMP_NAMESPACE: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";
const ICC_SIGNATURE: &[u8] = crate::color::icc::ICC_PROFILE_SIGNATURE;
pub(super) const PHOTOSHOP_SIGNATURE: &[u8] = b"Photoshop 3.0\0";
/// The pre-3.0 Photoshop APP13 identifier; ExifTool skips a 27-byte header
/// after it (`$psAPP13old`, `$hdrLen = $isOld ? 27 : 14`).
const PHOTOSHOP_OLD_SIGNATURE: &[u8] = b"Adobe_Photoshop2.5:";
const PHOTOSHOP_OLD_HEADER_LEN: usize = 27;

/// Identify a length-bearing marker segment and parse its fixed fields.
///
/// `marker` is the byte after `0xFF`; `payload` is everything after the
/// length word (for SOS, the scan header) and `entropy` the entropy-coded
/// bytes after an SOS header (empty for every other marker).
pub(super) fn identify<'a>(marker: u8, payload: &'a [u8], entropy: &'a [u8]) -> ItemKind<'a> {
    let kind = match marker {
        0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
            Sof::parse(marker, payload).map(ItemKind::Sof)
        }
        0xC4 => Some(ItemKind::Dht),
        0xCC => Some(ItemKind::Dac),
        0xDB => Some(ItemKind::Dqt),
        0xDA => Sos::parse(payload, entropy).map(ItemKind::Sos),
        0xDC => be16(payload, 0).map(ItemKind::Dnl),
        0xDD => be16(payload, 0).map(ItemKind::Dri),
        0xE0 => identify_app0(payload),
        0xE1 => identify_app1(payload),
        0xE2 => identify_app2(payload),
        0xE3 => strip(payload, &[b"Meta\0\0", b"META\0\0", b"Exif\0\0"]).map(ItemKind::KodakMeta),
        0xE5 => strip(payload, &[b"RMETA\0"]).map(ItemKind::RicohRmeta),
        0xE6 => strip(payload, &[b"EPPIM\0"]).map(|tiff| ItemKind::Eppim(Eppim::parse(tiff))),
        0xE8 => Spiff::parse(payload).map(ItemKind::Spiff),
        0xEA => {
            strip(payload, &[b"UNICODE\0"]).map(|text| ItemKind::PhotoStudioComment(Utf16Be(text)))
        }
        0xEB => Jumbf::parse(payload).map(ItemKind::Jumbf),
        0xEC => identify_app12(payload),
        0xED => identify_app13(payload),
        0xEE => Adobe::parse(payload).map(ItemKind::Adobe),
        0xEF => graphic_converter_quality(payload).map(ItemKind::GraphicConverter),
        0xFE => Some(ItemKind::Comment),
        _ => None,
    };
    kind.unwrap_or(ItemKind::Unknown)
}

fn identify_app0(payload: &[u8]) -> Option<ItemKind<'_>> {
    if payload.starts_with(b"JFIF\0") {
        return Jfif::parse(payload).map(ItemKind::Jfif);
    }
    // ExifTool accepts only the three extension codes JFIF 1.02 defines.
    if let Some(rest) = payload.strip_prefix(b"JFXX\0")
        && let Some((&code, data)) = rest.split_first()
        && matches!(code, 0x10 | 0x11 | 0x13)
    {
        return Some(ItemKind::Jfxx(Jfxx {
            extension_code: code,
            data,
        }));
    }
    if payload.len() >= 14
        && (payload.starts_with(b"II") || payload.starts_with(b"MM"))
        && &payload[6..14] == b"HEAPJPGM"
    {
        return Some(ItemKind::Ciff);
    }
    if let Some(rest) = payload.strip_prefix(b"AVI1") {
        return Some(ItemKind::Avi1(Avi1 {
            interleaved_field: rest.first().copied(),
        }));
    }
    None
}

fn identify_app1(payload: &[u8]) -> Option<ItemKind<'_>> {
    if payload.starts_with(b"Exif\0") {
        // The TIFF starts after the 6-byte `Exif\0\0` header whatever byte 5
        // holds (ExifTool: `DirStart => length("Exif\0\0")`; some Kodak
        // cameras write a non-NUL there). Writers that drop byte 5 entirely
        // put the TIFF at 5; take that when 5 begins a TIFF header (then 6
        // cannot: `I*` and `M\0` are no byte order). ExifTool reports that
        // layout as a malformed EXIF segment; accepting it is a leniency.
        let tiff = match payload.get(5..) {
            Some(at5) if is_tiff_header(at5) => at5,
            _ => payload.get(6..).unwrap_or(&[]),
        };
        return Some(ItemKind::Exif(tiff));
    }
    if let Some(packet) = payload.strip_prefix(XMP_NAMESPACE) {
        return Some(ItemKind::Xmp(packet));
    }
    if let Some(rest) = payload.strip_prefix(EXTENDED_XMP_NAMESPACE) {
        return ExtendedXmpChunk::parse(rest).map(ItemKind::ExtendedXmp);
    }
    None
}

fn is_tiff_header(b: &[u8]) -> bool {
    b.starts_with(b"II*\0") || b.starts_with(b"MM\0*")
}

fn identify_app2(payload: &[u8]) -> Option<ItemKind<'_>> {
    if let Some(rest) = payload.strip_prefix(ICC_SIGNATURE) {
        // ExifTool requires the two chunk bytes (segment length >= 14).
        let (&sequence, rest) = rest.split_first()?;
        let (&count, data) = rest.split_first()?;
        return Some(ItemKind::IccProfile(IccChunk {
            sequence,
            count,
            data,
        }));
    }
    if payload.starts_with(b"FPXR\0") {
        return Some(ItemKind::Fpxr);
    }
    if let Some(tiff) = payload.strip_prefix(super::super::mpf::MPF_IDENTIFIER) {
        return Some(ItemKind::Mpf(tiff));
    }
    if let Some(data) = payload.strip_prefix(zencodec::ISO_21496_1_URN) {
        return Some(ItemKind::GainMapIso21496(GainMapIso21496 { data }));
    }
    None
}

fn identify_app12(payload: &[u8]) -> Option<ItemKind<'_>> {
    if let Some(records) = payload.strip_prefix(b"Ducky") {
        return Some(ItemKind::Ducky(Ducky { records }));
    }
    is_picture_info(payload).then_some(ItemKind::PictureInfo)
}

fn identify_app13(payload: &[u8]) -> Option<ItemKind<'_>> {
    if let Some(resources) = payload.strip_prefix(PHOTOSHOP_SIGNATURE) {
        return Some(ItemKind::Photoshop(resources));
    }
    if payload.starts_with(PHOTOSHOP_OLD_SIGNATURE) {
        return payload
            .get(PHOTOSHOP_OLD_HEADER_LEN..)
            .map(ItemKind::Photoshop);
    }
    if let Some(rest) = payload.strip_prefix(b"Adobe_CM") {
        return be16(rest, 0).map(ItemKind::AdobeCm);
    }
    None
}

/// ExifTool's APP12 `ProcessAPP12` accepts the segment when its token regex
/// matches anywhere:
///
/// `(\[.*?\]|[\w#-]+=[\x20-\x7e]+?(?=\s*([\n\r\0]|[\w#-]+=|\[|$)))`
///
/// This is that test in linear time. A section is `[` then `]` with no
/// newline between. A pair is a tag character, `=`, one or more printable
/// characters, then (after optional whitespace) a line end, NUL, `[`, the
/// end of the data, or another `tag=`. Inside the printable run after `=`, a
/// `[` or `tag=` after the first value byte satisfies the lookahead;
/// otherwise the run's end decides it. A later `=` inside the same run has
/// the same run end and a subset of these chances, so the scan resumes at
/// the run's end, and every byte is read a bounded number of times. The
/// oracle test (`extended_metadata_exiftool`) checks this against the regex
/// run by Perl.
fn is_picture_info(p: &[u8]) -> bool {
    let tag_char = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'#' | b'-');
    let printable = |b: u8| (0x20..=0x7E).contains(&b);
    // Perl's `\s`: space, \t, \n, \x0b, \x0c, \r.
    let space = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r');
    // An unclosed `[` since the last newline.
    let mut section_open = false;
    let mut i = 0;
    while i < p.len() {
        match p[i] {
            b'[' => section_open = true,
            b']' if section_open => return true,
            b'\n' => section_open = false,
            b'=' if i > 0 && tag_char(p[i - 1]) && p.get(i + 1).is_some_and(|&v| printable(v)) => {
                let mut end = i + 1;
                while end < p.len() && printable(p[end]) {
                    end += 1;
                }
                for q in i + 1..end {
                    match p[q] {
                        b'[' if q >= i + 2 => return true,
                        b'[' => section_open = true,
                        b']' if section_open => return true,
                        b'=' if q >= i + 3 && tag_char(p[q - 1]) => return true,
                        _ => {}
                    }
                }
                // At the run's end: whitespace (a line end in it qualifies),
                // then the end of the data, NUL, `[` or `tag=`.
                let mut r = end;
                while r < p.len() && space(p[r]) {
                    if matches!(p[r], b'\n' | b'\r') {
                        return true;
                    }
                    r += 1;
                }
                if r == p.len() || matches!(p[r], 0 | b'[') {
                    return true;
                }
                let tag_end = (r..p.len()).find(|&t| !tag_char(p[t])).unwrap_or(p.len());
                if tag_end > r && p.get(tag_end) == Some(&b'=') {
                    return true;
                }
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

/// APP15 GraphicConverter: ExifTool requires a 4-byte payload matching
/// `^Q\s*(\d+)` and reports the digits as the quality.
fn graphic_converter_quality(payload: &[u8]) -> Option<u16> {
    if payload.len() != 4 || payload[0] != b'Q' {
        return None;
    }
    let rest = &payload[1..];
    // Perl's `\s` also matches the vertical tab.
    let start = rest
        .iter()
        .position(|&b| !(b.is_ascii_whitespace() || b == 0x0B))?;
    let digits = &rest[start..];
    let end = digits
        .iter()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(digits.len());
    if end == 0 {
        return None;
    }
    // At most three digits fit in a 4-byte payload, so this cannot overflow.
    Some(
        digits[..end]
            .iter()
            .fold(0u16, |acc, &d| acc * 10 + u16::from(d - b'0')),
    )
}

/// The bytes after the first of `signatures` that `payload` starts with.
fn strip<'a>(payload: &'a [u8], signatures: &[&[u8]]) -> Option<&'a [u8]> {
    signatures.iter().find_map(|sig| payload.strip_prefix(*sig))
}

#[inline]
fn be16(data: &[u8], at: usize) -> Option<u16> {
    match data.get(at..at.checked_add(2)?)? {
        &[a, b] => Some(u16::from_be_bytes([a, b])),
        _ => None,
    }
}

#[inline]
fn be32(data: &[u8], at: usize) -> Option<u32> {
    match data.get(at..at.checked_add(4)?)? {
        &[a, b, c, d] => Some(u32::from_be_bytes([a, b, c, d])),
        _ => None,
    }
}

// ── Frame structure ───────────────────────────────────────────────────────

/// Start-of-frame fields (T.81 B.2.2), the ExifTool "JPEG SOF" tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Sof<'a> {
    /// Encoding process: the marker's low nibble (ExifTool `EncodingProcess`:
    /// 0 baseline, 1 extended sequential, 2 progressive, 3 lossless, 5..=7
    /// differential, 9..=11 and 13..=15 arithmetic).
    pub process: u8,
    /// Sample precision in bits (`BitsPerSample`).
    pub precision: u8,
    /// Number of lines (`ImageHeight`); 0 means a DNL segment defines it.
    pub height: u16,
    /// Samples per line (`ImageWidth`).
    pub width: u16,
    /// Declared component count (`ColorComponents`).
    pub num_components: u8,
    components: &'a [u8],
}

impl<'a> Sof<'a> {
    fn parse(marker: u8, payload: &'a [u8]) -> Option<Self> {
        let &[precision, h0, h1, w0, w1, num_components, ref table @ ..] = payload else {
            return None;
        };
        // Keep only whole component records that are present.
        let present = (table.len() / 3).min(usize::from(num_components));
        Some(Self {
            process: marker & 0x0F,
            precision,
            height: u16::from_be_bytes([h0, h1]),
            width: u16::from_be_bytes([w0, w1]),
            num_components,
            components: &table[..present * 3],
        })
    }

    /// The component records present in the segment (at most
    /// [`num_components`](Self::num_components)).
    #[must_use]
    pub fn components(&self) -> FrameComponents<'a> {
        FrameComponents(self.components.as_chunks::<3>().0.iter())
    }

    /// ExifTool's `YCbCrSubSampling` for a three-component frame: the ratio
    /// of the largest to the smallest sampling factor on each axis, as
    /// `(horizontal, vertical)`. `None` for other component counts, a zero
    /// factor, or a ratio that is not a whole number.
    #[must_use]
    pub fn ycbcr_subsampling(&self) -> Option<(u8, u8)> {
        if self.num_components != 3 || self.components.len() < 9 {
            return None;
        }
        let (mut h_min, mut h_max, mut v_min, mut v_max) = (u8::MAX, 0, u8::MAX, 0);
        for c in self.components() {
            h_min = h_min.min(c.h);
            h_max = h_max.max(c.h);
            v_min = v_min.min(c.v);
            v_max = v_max.max(c.v);
        }
        if h_min == 0 || v_min == 0 || h_max % h_min != 0 || v_max % v_min != 0 {
            return None;
        }
        Some((h_max / h_min, v_max / v_min))
    }
}

/// One frame component record of a [`crate::container::metadata::Sof`] segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FrameComponent {
    /// Component identifier (`C`).
    pub id: u8,
    /// Horizontal sampling factor (`H`).
    pub h: u8,
    /// Vertical sampling factor (`V`).
    pub v: u8,
    /// Quantization table selector (`Tq`).
    pub quant_table: u8,
}

/// Iterator over the [`crate::container::metadata::FrameComponent`]s of a [`crate::container::metadata::Sof`].
#[derive(Debug, Clone)]
pub struct FrameComponents<'a>(core::slice::Iter<'a, [u8; 3]>);

impl Iterator for FrameComponents<'_> {
    type Item = FrameComponent;

    fn next(&mut self) -> Option<FrameComponent> {
        let &[id, hv, quant_table] = self.0.next()?;
        Some(FrameComponent {
            id,
            h: hv >> 4,
            v: hv & 0x0F,
            quant_table,
        })
    }
}

/// Start-of-scan header fields (T.81 B.2.3) and the scan's entropy-coded data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Sos<'a> {
    /// Start of spectral selection (`Ss`).
    pub spectral_start: u8,
    /// End of spectral selection (`Se`).
    pub spectral_end: u8,
    /// Successive approximation high bit position (`Ah`).
    pub approx_high: u8,
    /// Successive approximation low bit position (`Al`).
    pub approx_low: u8,
    /// The entropy-coded data that follows the header, up to the next marker
    /// that is not a restart marker (restart markers and stuffed bytes
    /// included). A restart marker preceded by `0xFF` fill bytes currently
    /// ends it (see [`crate::container::metadata::ItemKind::Restart`]).
    /// Where the data ends at fill bytes, or at a restart marker the decoder
    /// does not expect, may change (see the
    /// [module docs](crate::container::metadata)).
    pub entropy: &'a [u8],
    components: &'a [u8],
}

impl<'a> Sos<'a> {
    fn parse(payload: &'a [u8], entropy: &'a [u8]) -> Option<Self> {
        let (&ns, rest) = payload.split_first()?;
        let table_len = usize::from(ns) * 2;
        let (components, tail) = rest.split_at_checked(table_len)?;
        let &[spectral_start, spectral_end, approx, ..] = tail else {
            return None;
        };
        Some(Self {
            spectral_start,
            spectral_end,
            approx_high: approx >> 4,
            approx_low: approx & 0x0F,
            entropy,
            components,
        })
    }

    /// The scan's component selectors.
    #[must_use]
    pub fn components(&self) -> ScanComponents<'a> {
        ScanComponents(self.components.as_chunks::<2>().0.iter())
    }
}

/// One component selector of a [`crate::container::metadata::Sos`] header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScanComponent {
    /// Component selector (`Cs`), matching a [`crate::container::metadata::FrameComponent::id`].
    pub selector: u8,
    /// DC entropy coding table selector (`Td`).
    pub dc_table: u8,
    /// AC entropy coding table selector (`Ta`).
    pub ac_table: u8,
}

/// Iterator over the [`crate::container::metadata::ScanComponent`]s of a [`crate::container::metadata::Sos`].
#[derive(Debug, Clone)]
pub struct ScanComponents<'a>(core::slice::Iter<'a, [u8; 2]>);

impl Iterator for ScanComponents<'_> {
    type Item = ScanComponent;

    fn next(&mut self) -> Option<ScanComponent> {
        let &[selector, tables] = self.0.next()?;
        Some(ScanComponent {
            selector,
            dc_table: tables >> 4,
            ac_table: tables & 0x0F,
        })
    }
}

// ── APP0 ──────────────────────────────────────────────────────────────────

/// JFIF APP0 fields (JFIF 1.02; ExifTool JFIF tags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Jfif<'a> {
    /// Major version (`JFIFVersion`, first byte).
    pub version_major: u8,
    /// Minor version (`JFIFVersion`, second byte).
    pub version_minor: u8,
    /// Density units (`ResolutionUnit`): 0 aspect ratio only, 1 dots per
    /// inch, 2 dots per cm; other values are kept as written.
    pub units: u8,
    /// Horizontal density (`XResolution`).
    pub x_density: u16,
    /// Vertical density (`YResolution`).
    pub y_density: u16,
    /// Thumbnail width (`ThumbnailWidth`).
    pub thumbnail_width: u8,
    /// Thumbnail height (`ThumbnailHeight`).
    pub thumbnail_height: u8,
    /// The uncompressed RGB thumbnail: the `3 * width * height` bytes after
    /// the fixed fields, or as many of them as the segment holds (ExifTool
    /// `ThumbnailTIFF`). Bytes past the declared size are not part of it.
    pub thumbnail: &'a [u8],
}

impl<'a> Jfif<'a> {
    fn parse(payload: &'a [u8]) -> Option<Self> {
        let &[
            _,
            _,
            _,
            _,
            _,
            version_major,
            version_minor,
            units,
            x0,
            x1,
            y0,
            y1,
            tw,
            th,
            ref rest @ ..,
        ] = payload
        else {
            return None;
        };
        let declared = 3 * usize::from(tw) * usize::from(th);
        let thumbnail = &rest[..declared.min(rest.len())];
        Some(Self {
            version_major,
            version_minor,
            units,
            x_density: u16::from_be_bytes([x0, x1]),
            y_density: u16::from_be_bytes([y0, y1]),
            thumbnail_width: tw,
            thumbnail_height: th,
            thumbnail,
        })
    }
}

/// JFXX APP0 extension (JFIF 1.02 section "JFIF extension APP0").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Jfxx<'a> {
    /// Extension code: `0x10` JPEG thumbnail, `0x11` 1 byte/pixel palette
    /// thumbnail, `0x13` 3 bytes/pixel RGB thumbnail.
    pub extension_code: u8,
    /// The extension data after the code (for `0x10`, a JPEG stream).
    pub data: &'a [u8],
}

/// AVI1 APP0 fields (ExifTool `JPEG::AVI1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Avi1 {
    /// `InterleavedField`: 0 not interleaved, 1 odd, 2 even. `None` when the
    /// segment ends after the signature.
    pub interleaved_field: Option<u8>,
}

// ── APP1 / APP2 chunks ────────────────────────────────────────────────────

/// One extended XMP chunk (XMP Specification Part 3, 1.1.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExtendedXmpChunk<'a> {
    /// The 32-byte GUID (uppercase hex MD5 of the full extended packet)
    /// that names the packet this chunk belongs to.
    pub guid: &'a [u8],
    /// Full length of the extended packet.
    pub full_length: u32,
    /// Offset of this chunk's data within the extended packet.
    pub offset: u32,
    /// This chunk's portion of the packet.
    pub data: &'a [u8],
}

impl<'a> ExtendedXmpChunk<'a> {
    fn parse(rest: &'a [u8]) -> Option<Self> {
        let (guid, rest) = rest.split_at_checked(32)?;
        Some(Self {
            guid,
            full_length: be32(rest, 0)?,
            offset: be32(rest, 4)?,
            data: rest.get(8..)?,
        })
    }
}

/// One `ICC_PROFILE` chunk (ICC.1:2010 Annex B.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct IccChunk<'a> {
    /// 1-based sequence number of this chunk.
    pub sequence: u8,
    /// Total number of chunks.
    pub count: u8,
    /// This chunk's portion of the profile.
    pub data: &'a [u8],
}

/// APP2 ISO 21496-1 gain map metadata ([`zencodec::ISO_21496_1_URN`]).
///
/// Neither `Copy` nor `Eq`, so parsed fields (floating-point values among
/// them) can be added later.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GainMapIso21496<'a> {
    /// The bytes after the URN and its NUL: the ISO 21496-1 metadata.
    pub data: &'a [u8],
}

// ── APP6 EPPIM ────────────────────────────────────────────────────────────

/// APP6 `EPPIM` segment: a TIFF structure whose IFD0 carries the PrintIM
/// block in tag `0xC4A5` (ExifTool `JPEG::EPPIM`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Eppim<'a> {
    /// The TIFF structure after the `EPPIM\0` identifier.
    pub tiff: &'a [u8],
    /// The value of tag `0xC4A5` (PrintIM: `PrintIM\0`, a 4-character
    /// version, then the PrintIM records), when IFD0 holds it in bounds.
    /// ExifTool's TIFF reader rules apply: `II` or `MM` byte order with any
    /// magic number, and an IFD0 offset of at least 8.
    pub print_im: Option<&'a [u8]>,
}

impl<'a> Eppim<'a> {
    const TAG_PRINT_IM: u16 = 0xC4A5;

    fn parse(tiff: &'a [u8]) -> Self {
        Self {
            tiff,
            print_im: tiff_ifd0_value(tiff, Self::TAG_PRINT_IM),
        }
    }

    /// The 4-character PrintIM version (e.g. `b"0300"`, bytes 8..12), when
    /// the block passes ExifTool's `ProcessPrintIM` checks: more than 15
    /// bytes, starting with `PrintIM`, and room for the record count at
    /// byte 14 (read in either byte order) of 6-byte records after byte 16.
    #[must_use]
    pub fn print_im_version(&self) -> Option<&'a [u8]> {
        let block = self.print_im?;
        if block.len() <= 15 || !block.starts_with(b"PrintIM") {
            return None;
        }
        let count = [block[14], block[15]];
        let fits = |n: u16| block.len() >= 16 + usize::from(n) * 6;
        if !fits(u16::from_be_bytes(count)) && !fits(u16::from_le_bytes(count)) {
            return None;
        }
        block.get(8..12)
    }
}

/// The value bytes of `tag` in IFD0 of a TIFF structure, when the entry and
/// its value lie within `tiff`. Handles both byte orders and inline values.
/// Header rules follow ExifTool's `DoProcessTIFF`: the byte order must be
/// `II` or `MM`, the magic number is not checked, and IFD0 must start at
/// offset 8 or later.
fn tiff_ifd0_value(tiff: &[u8], tag: u16) -> Option<&[u8]> {
    let le = match tiff.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |at: usize| -> Option<u16> {
        let b = tiff.get(at..at.checked_add(2)?)?;
        Some(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let b = tiff.get(at..at.checked_add(4)?)?;
        let b = [b[0], b[1], b[2], b[3]];
        Some(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };
    let ifd = usize::try_from(u32_at(4)?).ok()?;
    if ifd < 8 {
        return None;
    }
    let count = usize::from(u16_at(ifd)?);
    for i in 0..count {
        let entry = ifd.checked_add(2)?.checked_add(i.checked_mul(12)?)?;
        if u16_at(entry)? != tag {
            continue;
        }
        let size = tiff_type_size(u16_at(entry + 2)?)?;
        let len = usize::try_from(u32_at(entry + 4)?)
            .ok()?
            .checked_mul(size)?;
        let start = if len <= 4 {
            entry + 8
        } else {
            usize::try_from(u32_at(entry + 8)?).ok()?
        };
        return tiff.get(start..start.checked_add(len)?);
    }
    None
}

/// Byte size of one value of TIFF field type `t` (TIFF 6.0 section 2).
fn tiff_type_size(t: u16) -> Option<usize> {
    match t {
        1 | 2 | 6 | 7 => Some(1),
        3 | 8 => Some(2),
        4 | 9 | 11 | 13 => Some(4),
        5 | 10 | 12 => Some(8),
        _ => None,
    }
}

// ── APP8 SPIFF ────────────────────────────────────────────────────────────

/// APP8 SPIFF header fields (ITU-T T.84 Annex F.2.1).
///
/// The standard header after `SPIFF\0` is 24 bytes. ExifTool identifies
/// SPIFF only for a 26-byte body and reads the fields after the component
/// count two bytes later, citing its one (Kodak) sample; both layouts are
/// accepted, each read at its own offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Spiff {
    /// Major version (`SPIFFVersion`, first byte).
    pub version_major: u8,
    /// Minor version (`SPIFFVersion`, second byte).
    pub version_minor: u8,
    /// Application profile (`ProfileID`): 0 none, 1 continuous-tone base,
    /// 2 continuous-tone progressive, 3 bi-level facsimile, 4
    /// continuous-tone facsimile.
    pub profile_id: u8,
    /// Number of components (`ColorComponents`).
    pub num_components: u8,
    /// Image height (`ImageHeight`).
    pub height: u32,
    /// Image width (`ImageWidth`).
    pub width: u32,
    /// Colour space code (`ColorSpace`; e.g. 3 YCbCr BT.601 RGB, 8
    /// grayscale, 10 RGB, 12 CMYK).
    pub color_space: u8,
    /// Bits per sample (`BitsPerSample`).
    pub bits_per_sample: u8,
    /// Compression type (`Compression`; 5 JPEG).
    pub compression: u8,
    /// Resolution units (`ResolutionUnit`): 0 aspect ratio, 1 inch, 2 cm.
    pub resolution_units: u8,
    /// Vertical resolution (`YResolution`).
    pub vertical_resolution: u32,
    /// Horizontal resolution (`XResolution`).
    pub horizontal_resolution: u32,
}

impl Spiff {
    fn parse(payload: &[u8]) -> Option<Self> {
        let body = payload.strip_prefix(b"SPIFF\0")?;
        // T.84: 24-byte body. ExifTool's sample: 26 bytes, 2 extra after NC.
        let skip = match body.len() {
            24 => 0,
            26 => 2,
            _ => return None,
        };
        let at = |i: usize| if i >= 4 { i + skip } else { i };
        Some(Self {
            version_major: body[0],
            version_minor: body[1],
            profile_id: body[2],
            num_components: body[3],
            height: be32(body, at(4))?,
            width: be32(body, at(8))?,
            color_space: body[at(12)],
            bits_per_sample: body[at(13)],
            compression: body[at(14)],
            resolution_units: body[at(15)],
            vertical_resolution: be32(body, at(16))?,
            horizontal_resolution: be32(body, at(20))?,
        })
    }
}

// ── APP10 / APP12 text ────────────────────────────────────────────────────

/// Big-endian UTF-16 (UCS-2) text, as in the APP10 PhotoStudio comment and
/// the Ducky comment and copyright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Utf16Be<'a>(&'a [u8]);

impl<'a> Utf16Be<'a> {
    /// The raw bytes (an odd trailing byte is ignored by [`chars`](Self::chars)).
    #[must_use]
    pub fn bytes(&self) -> &'a [u8] {
        self.0
    }

    /// The decoded characters; unpaired surrogates become U+FFFD.
    #[must_use]
    pub fn chars(&self) -> Utf16BeChars<'a> {
        Utf16BeChars(char::decode_utf16(Utf16BeUnits(
            self.0.as_chunks::<2>().0.iter(),
        )))
    }
}

/// Iterator over the characters of a [`crate::container::metadata::Utf16Be`] text.
#[derive(Debug, Clone)]
pub struct Utf16BeChars<'a>(core::char::DecodeUtf16<Utf16BeUnits<'a>>);

impl Iterator for Utf16BeChars<'_> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        self.0
            .next()
            .map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER))
    }
}

/// The big-endian code units of a [`crate::container::metadata::Utf16Be`] text.
#[derive(Debug, Clone)]
struct Utf16BeUnits<'a>(core::slice::Iter<'a, [u8; 2]>);

impl Iterator for Utf16BeUnits<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        self.0.next().map(|&u| u16::from_be_bytes(u))
    }
}

/// APP12 `Ducky` segment (ExifTool `APP12::Ducky`): records of
/// `tag u16, length u16, data`, big-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ducky<'a> {
    records: &'a [u8],
}

impl<'a> Ducky<'a> {
    /// The records in order, read as ExifTool's `ProcessDucky` does:
    /// iteration ends when fewer than 4 bytes remain or at a record whose
    /// length runs past the segment. A zero tag (Photoshop's terminator) is
    /// returned like any other record.
    #[must_use]
    pub fn records(&self) -> DuckyRecords<'a> {
        DuckyRecords(self.records)
    }

    /// Tag 1, `Quality`: the 4-byte big-endian "Save for Web" quality, from
    /// the first such record. ExifTool gives all three Ducky tags priority 0,
    /// so it keeps the first value of each.
    #[must_use]
    pub fn quality(&self) -> Option<u32> {
        let data = self.record(1)?;
        be32(data, 0)
    }

    /// Tag 2, `Comment`, after its 4-byte character count, from the first
    /// such record (ExifTool priority 0).
    #[must_use]
    pub fn comment(&self) -> Option<Utf16Be<'a>> {
        self.record(2)?.get(4..).map(Utf16Be)
    }

    /// Tag 3, `Copyright`, after its 4-byte character count, from the first
    /// such record (ExifTool priority 0).
    #[must_use]
    pub fn copyright(&self) -> Option<Utf16Be<'a>> {
        self.record(3)?.get(4..).map(Utf16Be)
    }

    fn record(&self, tag: u16) -> Option<&'a [u8]> {
        self.records().find(|r| r.tag == tag).map(|r| r.data)
    }
}

/// One record of a [`crate::container::metadata::Ducky`] segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DuckyRecord<'a> {
    /// Record tag (1 quality, 2 comment, 3 copyright; 0 terminator).
    pub tag: u16,
    /// Record data.
    pub data: &'a [u8],
}

/// Iterator over the [`crate::container::metadata::DuckyRecord`]s of a [`crate::container::metadata::Ducky`] segment.
#[derive(Debug, Clone)]
pub struct DuckyRecords<'a>(&'a [u8]);

impl<'a> Iterator for DuckyRecords<'a> {
    type Item = DuckyRecord<'a>;

    fn next(&mut self) -> Option<DuckyRecord<'a>> {
        let tag = be16(self.0, 0)?;
        let len = usize::from(be16(self.0, 2)?);
        let Some(data) = self.0.get(4..4 + len) else {
            self.0 = &[];
            return None;
        };
        self.0 = &self.0[4 + len..];
        Some(DuckyRecord { tag, data })
    }
}

// ── APP11 JUMBF ───────────────────────────────────────────────────────────

/// APP11 JUMBF segment header (ISO/IEC 19566-5 JPEG-1 embedding; ExifTool
/// identifies `^JP..` with at least 16 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Jumbf<'a> {
    /// Box instance number (`En`): segments with the same instance and box
    /// type belong to one box.
    pub instance: u16,
    /// Packet sequence number (`Z`), starting at 1.
    pub sequence: u32,
    /// The superbox type (`TBox`), normally `jumb`.
    pub box_type: [u8; 4],
    /// The box bytes after the 8-byte segment header: `LBox`, `TBox`
    /// (and `XLBox`) followed by this segment's portion of the content.
    pub data: &'a [u8],
}

impl<'a> Jumbf<'a> {
    fn parse(payload: &'a [u8]) -> Option<Self> {
        if payload.len() < 16 || !payload.starts_with(b"JP") {
            return None;
        }
        Some(Self {
            instance: be16(payload, 2)?,
            sequence: be32(payload, 4)?,
            box_type: [payload[12], payload[13], payload[14], payload[15]],
            data: &payload[8..],
        })
    }
}

// ── APP14 Adobe ───────────────────────────────────────────────────────────

/// APP14 `Adobe` fields (Adobe Technical Note 5116; ExifTool `JPEG::Adobe`).
///
/// Metadata only: how the decoder interprets the colour transform is decided
/// elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Adobe {
    /// `DCTEncodeVersion` (100 or 101 in practice).
    pub dct_encode_version: u16,
    /// `APP14Flags0` (bit 15: encoded with Blend=1 downsampling).
    pub flags0: u16,
    /// `APP14Flags1`.
    pub flags1: u16,
    /// `ColorTransform` as written: 0 unknown (RGB or CMYK), 1 YCbCr, 2 YCCK.
    pub color_transform: u8,
}

impl Adobe {
    fn parse(payload: &[u8]) -> Option<Self> {
        let body = payload.strip_prefix(b"Adobe")?;
        Some(Self {
            dct_encode_version: be16(body, 0)?,
            flags0: be16(body, 2)?,
            flags1: be16(body, 4)?,
            color_transform: *body.get(6)?,
        })
    }
}
