//! Every segment and trailer of a JPEG file, in file order, parsed or opaque.
//!
//! [`crate::container::metadata::JpegMetadata::parse`] walks a JPEG byte buffer
//! once and returns one [`crate::container::metadata::Item`] per structural unit: each marker
//! segment (SOI..EOI, every APPn,
//! COM, DQT/DHT/SOF/SOS/DRI/DNL, unknown markers), every chunk of data
//! appended after the primary image's EOI (MPF images, vendor trailers), and
//! every run of bytes the walker could not structure. Nothing is dropped:
//! the items tile the input exactly, so `items[i].offset + items[i].length
//! == items[i + 1].offset` and the lengths sum to the input length.
//!
//! The walk does not decode pixels, and items borrow from the input: no
//! payload is copied. The current walker skips entropy-coded scans with a
//! byte search, and parsing allocates, fallibly, only the item list
//! (reserved once for up to 32 items, then grown by doubling; it never holds
//! more than [`crate::container::metadata::MetadataLimits::max_items`]
//! items) and, when the file ends in vendor trailers, a list of those
//! trailers. Those two are properties of this walker, not promises: a
//! walker that follows the decoder's marker loop may read the entropy-coded
//! data (see "What is stable" below).
//!
//! The accessors ([`crate::container::metadata::JpegMetadata::exif`],
//! [`crate::container::metadata::JpegMetadata::xmp`],
//! [`crate::container::metadata::JpegMetadata::extended_xmp`],
//! [`crate::container::metadata::JpegMetadata::icc_profile`],
//! [`crate::container::metadata::JpegMetadata::photoshop`],
//! [`crate::container::metadata::JpegMetadata::mpf`]) read the header. The
//! header is the primary image's items before its first SOS; when it has no
//! SOS, the items before its EOI; when it has neither, every item. A second
//! SOI does not start a new header (ExifTool reads on; libjpeg-turbo stops
//! with an error), and streams appended after the primary image's EOI are
//! never part of it. This is where libjpeg-turbo reads metadata, and where
//! ExifTool does by default (with `-ee` or `-validate` it also reads segments
//! between scans). Segments outside the header stay visible as items.
//! Payloads that span several segments (ICC profile, extended XMP, Photoshop
//! resources) are reassembled with validation; each borrows when one segment
//! holds it and otherwise allocates the returned buffer, fallibly, only after
//! its checks pass. Either way the result is at most
//! [`crate::container::metadata::MetadataLimits::max_reassembled_bytes`] long.
//!
//! # What is stable
//!
//! The item list may later be built from a walker that follows the
//! decoder's marker loop, so that metadata reading and decoding see one
//! structure of the file. These hold now and will hold then:
//!
//! - the items tile the input in file order;
//! - each item's [`kind`](crate::container::metadata::Item::kind)
//!   describes the bytes it covers;
//! - the accessors' results on well-formed files, that is, files whose
//!   primary image has one SOI, one frame, no reserved markers (TEM),
//!   length words that match their content and no stray bytes.
//!
//! Not yet stable: item boundaries on malformed or unusual layouts, the
//! extents of [`crate::container::metadata::ItemKind::Unparsed`] items,
//! [`crate::container::metadata::JpegMetadata::primary_image_end`] on such
//! files, and the item list after the primary image's EOI. The differences
//! known today are listed, as information rather than a contract, in the
//! "Follow-up" section of
//! [`docs/METADATA.md`](https://github.com/imazen/zenjpeg/blob/main/zenjpeg/docs/METADATA.md).
//! Accessor changes that are not walker changes (another MPF directory
//! layout, a namespace-aware XMP reader) are announced in the CHANGELOG.
//!
//! # Coverage
//!
//! Identification follows the
//! [ExifTool 6.99 JPEG tag table](https://web.mit.edu/graphics/src/Image-ExifTool-6.99/html/TagNames/JPEG.html)
//! plus MPF and ISO 21496-1 gain map metadata (which zenjpeg reads and
//! writes), JUMBF, and the Samsung trailer that current ExifTool versions
//! recognize. Formats the table defines inline are parsed into
//! fields: JFIF/JFXX/AVI1 (APP0), SPIFF (APP8), Adobe (APP14), Adobe_CM and
//! Photoshop resources (APP13), Ducky (APP12), the PrintIM block in EPPIM
//! (APP6), the PhotoStudio comment (APP10), GraphicConverter quality
//! (APP15), and the SOF/SOS/DRI/DNL frame fields. Formats the table only
//! links to (EXIF, XMP, ICC, FlashPix, Kodak Meta, Ricoh RMETA, CIFF, MPF,
//! trailers) are identified and exposed as borrowed bytes for a dedicated
//! parser. Everything else is an [`crate::container::metadata::ItemKind::Unknown`]
//! segment or an [`crate::container::metadata::ItemKind::UnknownTrailer`],
//! with its bytes. Where this reading differs from ExifTool on purpose (a
//! dropped EXIF byte, signatures matched literally, a Photoshop run that
//! starts with the old header, extended XMP edge cases), `docs/METADATA.md`
//! lists it.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "extended-metadata")] {
//! use zenjpeg::container::metadata::{ItemKind, JpegMetadata};
//!
//! # let jpeg: &[u8] = &[
//! #     0xFF, 0xD8,
//! #     0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0, 1, 2, 1, 0, 72, 0, 72, 0, 0,
//! #     0xFF, 0xFE, 0x00, 0x07, b'h', b'e', b'l', b'l', b'o',
//! #     0xFF, 0xD9,
//! # ];
//! let meta = JpegMetadata::parse(jpeg).expect("allocation");
//! for item in meta.items() {
//!     println!("{:>8} {:>6}  {}", item.offset, item.length, item.kind.name());
//!     if let ItemKind::Jfif(jfif) = item.kind {
//!         println!("  density {}x{} (units {})", jfif.x_density, jfif.y_density, jfif.units);
//!     }
//! }
//! assert_eq!(meta.items().len(), 4); // SOI, JFIF, COM, EOI
//! # }
//! ```

#[path = "identify.rs"]
mod identify;
#[path = "reassemble.rs"]
mod reassemble;
#[path = "trailer.rs"]
mod trailer;
#[path = "walk.rs"]
mod walk;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

use alloc::borrow::Cow;
use alloc::vec::Vec;

pub use identify::{
    Adobe, Avi1, Ducky, DuckyRecord, DuckyRecords, Eppim, ExtendedXmpChunk, FrameComponent,
    FrameComponents, GainMapIso21496, IccChunk, Jfif, Jfxx, Jumbf, ScanComponent, ScanComponents,
    Sof, Sos, Spiff, Utf16Be, Utf16BeChars,
};
pub use reassemble::{PhotoshopIrb, PhotoshopResource, PhotoshopResources};

use crate::container::mpf::{MpfError, parse_mpf_segment};
use crate::container::types::MpfEntry;

/// One structural unit of a JPEG file.
///
/// Offsets and lengths are in bytes, relative to the start of the buffer
/// passed to [`crate::container::metadata::JpegMetadata::parse`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Item<'a> {
    /// Offset of the item's first byte (the `0xFF` of a marker segment).
    pub offset: usize,
    /// Total size: marker, length word and payload for a marker segment
    /// (plus the entropy-coded data that follows an SOS header); the whole
    /// run of bytes for trailers, padding and unparsed bytes.
    pub length: usize,
    /// Marker identifier (the byte after `0xFF`) for marker segments,
    /// including a truncated or bad-length one; `None` for trailers,
    /// padding and stray bytes.
    pub marker: Option<u8>,
    /// When [`marker`](Self::marker) is set: the bytes after the length
    /// word (for SOS, the scan header as its length word sizes it; for a
    /// truncated segment, what is present; for a bad-length one, nothing).
    /// Otherwise every byte of the trailer, padding or unparsed run. The
    /// module docs say which of these extents are stable.
    pub payload: &'a [u8],
    /// What the item is, with its parsed fields.
    pub kind: ItemKind<'a>,
}

/// Identification of an [`crate::container::metadata::Item`], with the fields parsed from its payload.
///
/// Signatures and field layouts follow ExifTool's JPEG reader; each variant
/// documents its source. A segment whose signature matches but whose payload
/// is too short for the format's fixed layout is reported as
/// [`crate::container::metadata::ItemKind::Unknown`] rather than half-parsed.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ItemKind<'a> {
    // ── Frame structure (ITU-T T.81 Table B.1) ────────────────────────────
    /// Start of image (`FFD8`).
    Soi,
    /// End of image (`FFD9`). Items after the primary image's EOI are
    /// trailers.
    Eoi,
    /// A restart marker (`FFD0`..`FFD7`) that is not part of an
    /// [`crate::container::metadata::ItemKind::Sos`] item's entropy-coded
    /// data. Restart markers inside a scan belong to its SOS item, except
    /// that one preceded by `0xFF` fill bytes (`FF FF D0`) currently ends
    /// the scan's data: the fill becomes a
    /// [`crate::container::metadata::ItemKind::Padding`] item, the marker
    /// this item, and the rest of the scan usually
    /// [`crate::container::metadata::Unparsed::ExtraneousBytes`] items,
    /// although a decoder reads it as part of the scan. Where a scan ends at
    /// fill bytes or at a restart marker the decoder does not expect may
    /// change (see the module docs).
    Restart,
    /// Start of frame (`SOF0`..`SOF15`, excluding DHT, JPG and DAC).
    Sof(Sof<'a>),
    /// Define Huffman tables (`FFC4`).
    Dht,
    /// Define quantization tables (`FFDB`).
    Dqt,
    /// Define arithmetic coding conditioning (`FFCC`).
    Dac,
    /// Start of scan (`FFDA`): the scan header plus its entropy-coded data.
    Sos(Sos<'a>),
    /// Define restart interval (`FFDD`), in MCUs.
    Dri(u16),
    /// Define number of lines (`FFDC`).
    Dnl(u16),

    // ── APP0 ──────────────────────────────────────────────────────────────
    /// `JFIF\0` (JFIF 1.02).
    Jfif(Jfif<'a>),
    /// `JFXX\0` JFIF extension (thumbnail).
    Jfxx(Jfxx<'a>),
    /// Canon CIFF (`II`/`MM` + 4 bytes + `HEAPJPGM`); the payload is the CIFF heap.
    Ciff,
    /// `AVI1` (Motion JPEG frames from AVI files).
    Avi1(Avi1),

    // ── APP1 ──────────────────────────────────────────────────────────────
    /// `Exif\0` and one more byte of any value (ExifTool reads the TIFF
    /// structure from byte 6; some Kodak cameras write a non-NUL byte 5).
    /// Holds the TIFF structure: from byte 6, or from byte 5 when byte 5
    /// begins a TIFF header (a writer that dropped the second NUL; ExifTool
    /// reports that layout as malformed). Empty when the segment ends within
    /// the prefix.
    Exif(&'a [u8]),
    /// `http://ns.adobe.com/xap/1.0/\0` standard XMP. Holds the packet.
    Xmp(&'a [u8]),
    /// `http://ns.adobe.com/xmp/extension/\0` extended XMP chunk (XMP
    /// Specification Part 3, 1.1.3.1).
    ExtendedXmp(ExtendedXmpChunk<'a>),

    // ── APP2 ──────────────────────────────────────────────────────────────
    /// `ICC_PROFILE\0` chunk (ICC.1 Annex B.4).
    IccProfile(IccChunk<'a>),
    /// `FPXR\0` FlashPix ready data; the payload is opaque.
    Fpxr,
    /// `MPF\0` Multi-Picture Format directory (CIPA DC-007). Holds the TIFF
    /// structure after the identifier; [`crate::container::metadata::JpegMetadata::mpf`] parses it.
    Mpf(&'a [u8]),
    /// ISO 21496-1 gain map metadata ([`zencodec::ISO_21496_1_URN`]), with
    /// the bytes after the URN. A struct, so parsed fields can be added.
    GainMapIso21496(GainMapIso21496<'a>),

    // ── APP3..APP15 ───────────────────────────────────────────────────────
    /// APP3 Kodak `Meta\0\0`, `META\0\0` or `Exif\0\0`. Holds the TIFF
    /// structure after the 6-byte identifier.
    KodakMeta(&'a [u8]),
    /// APP5 Ricoh `RMETA\0`. Holds the bytes after the identifier.
    RicohRmeta(&'a [u8]),
    /// APP6 Toshiba `EPPIM\0`: a TIFF structure carrying PrintIM.
    Eppim(Eppim<'a>),
    /// APP8 `SPIFF\0` (ITU-T T.84 Annex F).
    Spiff(Spiff),
    /// APP10 PhotoStudio `UNICODE\0` comment.
    PhotoStudioComment(Utf16Be<'a>),
    /// APP11 JUMBF box segment (ISO/IEC 19566-5): `JP` + instance +
    /// sequence + box.
    Jumbf(Jumbf<'a>),
    /// APP12 `Ducky` (Photoshop "Save for Web").
    Ducky(Ducky<'a>),
    /// APP12 Picture Info: ASCII `[section]` and `key=value` text, identified
    /// with ExifTool's token test. The text is not parsed into fields; the
    /// payload holds it.
    PictureInfo,
    /// APP13 `Photoshop 3.0\0` image resource blocks, or the older
    /// `Adobe_Photoshop2.5:` form with its 27-byte header (ExifTool
    /// `$psAPP13old`). Holds the resource data after the header;
    /// [`crate::container::metadata::JpegMetadata::photoshop`]
    /// joins consecutive segments and walks the resources.
    Photoshop(&'a [u8]),
    /// APP13 `Adobe_CM`: the AdobeCMType value.
    AdobeCm(u16),
    /// APP14 `Adobe` (Adobe Technical Note 5116).
    Adobe(Adobe),
    /// APP15 GraphicConverter `Q` quality.
    GraphicConverter(u16),
    /// `COM` comment; the payload is the comment bytes (often, but not
    /// always, text).
    Comment,
    /// A marker segment that is not identified: an APPn with an unknown
    /// signature, an unclassified marker (`JPGn`, `DHP`, `EXP`, `TEM`,
    /// reserved), or a known signature too short for its layout. The
    /// payload holds its bytes. An SOS whose scan header does not parse (the
    /// component count overruns it) is `Unknown` too; like an
    /// [`crate::container::metadata::ItemKind::Sos`] item it covers the
    /// entropy-coded data after the header, and its payload is the header.
    Unknown,

    // ── After the primary image's EOI ─────────────────────────────────────
    /// A JPEG stream appended after the primary image: an MPF image (gain
    /// map, depth map, large thumbnail), a preview image, or a concatenated
    /// image. The current walker reports a stream from SOI to EOI that it
    /// structures without unparsed bytes; how appended streams are found and
    /// sized may change (see the module docs). The payload is the whole
    /// stream; pass it to
    /// [`crate::container::metadata::JpegMetadata::parse`] to walk it.
    EmbeddedJpeg,
    /// AXS File Concatenation Protocol trailer (`AXS!`/`AXS*`).
    Afcp,
    /// Canon Digital Photo Professional recipe data (`CANON OPTIONAL DATA\0`).
    CanonVrd,
    /// One FotoWare FotoStation trailer record; holds its record tag
    /// (1 IPTC, 2 SoftEdit, 3 thumbnail, 4 preview).
    FotoStation(u16),
    /// Photo Mechanic trailer (`cbipcbbl`).
    PhotoMechanic,
    /// MIE trailer (`zmie` signature).
    Mie,
    /// Samsung trailer (`SEFH`/`SEFT` directory, optionally `QDIO`).
    Samsung,
    /// Bytes that are all `0x00` or all `0xFF`: fill bytes before a marker
    /// (T.81 B.1.1.2) or padding between appended images.
    Padding,
    /// Data after the primary image that is not identified. This includes
    /// the Insta360 and NikonApp trailers ExifTool 12.40 names but this
    /// module does not size, and any vendor trailer before them.
    UnknownTrailer,

    // ── Not structurable ──────────────────────────────────────────────────
    /// Bytes the walker could not structure, and why.
    Unparsed(Unparsed),
}

/// Why an [`crate::container::metadata::ItemKind::Unparsed`] run of bytes could not be structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Unparsed {
    /// The input does not begin with SOI (`FFD8`). The current walker then
    /// makes the whole input this one item.
    NotJpeg,
    /// Bytes between segments that are not a segment: a run that does not
    /// begin with a marker, or a stuffed `FF 00` where a marker belongs. The
    /// current walker extends the run to the next marker other than `RSTn`,
    /// the way libjpeg's `next_marker` and `read_markers` skip it, so
    /// entropy-coded data that lost its scan header (restart markers
    /// included) is one item. Where such runs start and end may change (see
    /// the module docs).
    ExtraneousBytes,
    /// A marker whose length word is below 2. The current walker makes the
    /// marker and the length word this item and resumes right after it; how
    /// those bytes and the ones after them are split may change (see the
    /// module docs).
    BadLength,
    /// A segment cut off by the end of the input (a cut marker, length word
    /// or payload). [`crate::container::metadata::Item::marker`] holds the marker when it is present.
    TruncatedSegment,
    /// [`crate::container::metadata::MetadataLimits::max_items`] was reached; the item covers the rest
    /// of the input.
    LimitExceeded,
}

impl ItemKind<'_> {
    /// Short display name. ExifTool's segment names where one exists
    /// (`JFIF`, `ICC_Profile`, `Photoshop`, `Adobe_CM`, ...).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Soi => "SOI",
            Self::Eoi => "EOI",
            Self::Restart => "RST",
            Self::Sof(_) => "SOF",
            Self::Dht => "DHT",
            Self::Dqt => "DQT",
            Self::Dac => "DAC",
            Self::Sos(_) => "SOS",
            Self::Dri(_) => "DRI",
            Self::Dnl(_) => "DNL",
            Self::Jfif(_) => "JFIF",
            Self::Jfxx(_) => "JFXX",
            Self::Ciff => "CIFF",
            Self::Avi1(_) => "AVI1",
            Self::Exif(_) => "EXIF",
            Self::Xmp(_) => "XMP",
            Self::ExtendedXmp(_) => "ExtendedXMP",
            Self::IccProfile(_) => "ICC_Profile",
            Self::Fpxr => "FPXR",
            Self::Mpf(_) => "MPF",
            Self::GainMapIso21496(_) => "ISO21496-1",
            Self::KodakMeta(_) => "Meta",
            Self::RicohRmeta(_) => "RMETA",
            Self::Eppim(_) => "EPPIM",
            Self::Spiff(_) => "SPIFF",
            Self::PhotoStudioComment(_) => "PhotoStudio",
            Self::Jumbf(_) => "JUMBF",
            Self::Ducky(_) => "Ducky",
            Self::PictureInfo => "PictureInfo",
            Self::Photoshop(_) => "Photoshop",
            Self::AdobeCm(_) => "Adobe_CM",
            Self::Adobe(_) => "Adobe",
            Self::GraphicConverter(_) => "GraphicConverter",
            Self::Comment => "Comment",
            Self::Unknown => "Unknown",
            Self::EmbeddedJpeg => "EmbeddedJPEG",
            Self::Afcp => "AFCP",
            Self::CanonVrd => "CanonVRD",
            Self::FotoStation(_) => "FotoStation",
            Self::PhotoMechanic => "PhotoMechanic",
            Self::Mie => "MIE",
            Self::Samsung => "Samsung",
            Self::Padding => "Padding",
            Self::UnknownTrailer => "UnknownTrailer",
            Self::Unparsed(_) => "Unparsed",
        }
    }
}

/// Resource limits for [`crate::container::metadata::JpegMetadata::parse_with_limits`] and the
/// reassembly accessors.
///
/// ```
/// # #[cfg(feature = "extended-metadata")] {
/// use zenjpeg::container::metadata::MetadataLimits;
///
/// let limits = MetadataLimits::default()
///     .with_max_items(256)
///     .with_max_reassembled_bytes(1 << 20);
/// assert_eq!((limits.max_items(), limits.max_reassembled_bytes()), (256, 1 << 20));
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataLimits {
    max_items: usize,
    max_reassembled_bytes: usize,
}

impl Default for MetadataLimits {
    fn default() -> Self {
        Self {
            max_items: 16_384,
            max_reassembled_bytes: 64 * 1024 * 1024,
        }
    }
}

impl MetadataLimits {
    /// Set the maximum number of items collected (0 is treated as 1). When
    /// the walk would produce more, the last item is
    /// [`crate::container::metadata::Unparsed::LimitExceeded`] covering the rest of the input. Default
    /// 16384; real files carry tens.
    #[must_use]
    pub fn with_max_items(mut self, max_items: usize) -> Self {
        self.max_items = max_items;
        self
    }

    /// Set the maximum size of one reassembled payload (ICC profile,
    /// extended XMP, joined Photoshop resources). Default 64 MiB. A larger
    /// declared or summed size is [`crate::container::metadata::MetadataError::TooLarge`], checked before
    /// allocating. The bound also applies when one segment holds the whole
    /// payload and the result borrows it.
    #[must_use]
    pub fn with_max_reassembled_bytes(mut self, bytes: usize) -> Self {
        self.max_reassembled_bytes = bytes;
        self
    }

    /// The maximum number of items, as set (see
    /// [`with_max_items`](Self::with_max_items)).
    #[must_use]
    pub fn max_items(&self) -> usize {
        self.max_items
    }

    /// The maximum size of one reassembled payload in bytes, borrowed or
    /// allocated (see
    /// [`with_max_reassembled_bytes`](Self::with_max_reassembled_bytes)).
    #[must_use]
    pub fn max_reassembled_bytes(&self) -> usize {
        self.max_reassembled_bytes
    }
}

/// Errors from [`crate::container::metadata::JpegMetadata`] parsing and reassembly.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MetadataError {
    /// An allocation failed.
    #[error("out of memory")]
    OutOfMemory,
    /// A reassembled payload would exceed [`crate::container::metadata::MetadataLimits::max_reassembled_bytes`].
    #[non_exhaustive]
    #[error("reassembled payload of {size} bytes exceeds the {limit}-byte limit")]
    TooLarge {
        /// Bytes the payload needs.
        size: usize,
        /// The configured limit.
        limit: usize,
    },
    /// An ICC chunk declares a total chunk count of 0.
    #[error("ICC_PROFILE chunk count is zero")]
    IccZeroChunkCount,
    /// ICC chunks disagree on the total chunk count.
    #[non_exhaustive]
    #[error("ICC_PROFILE chunk declares {found} chunks, the first declared {expected}")]
    IccChunkCountMismatch {
        /// Count declared by the first chunk.
        expected: u8,
        /// Count declared by a later chunk.
        found: u8,
    },
    /// An ICC chunk's sequence number is 0 or above the declared count.
    #[non_exhaustive]
    #[error("ICC_PROFILE chunk sequence number {sequence} outside 1..={count}")]
    IccSequenceOutOfRange {
        /// The chunk's sequence number.
        sequence: u8,
        /// The declared chunk count.
        count: u8,
    },
    /// Two ICC chunks carry the same sequence number.
    #[non_exhaustive]
    #[error("duplicate ICC_PROFILE chunk {sequence}")]
    IccDuplicateChunk {
        /// The repeated sequence number.
        sequence: u8,
    },
    /// The ICC chunks are complete but hold no profile data (libjpeg-turbo's
    /// `jpeg_read_icc_profile` rejects a total length of 0).
    #[error("ICC_PROFILE chunks hold no profile data")]
    IccEmptyProfile,
    /// An ICC chunk in `1..=count` is absent.
    #[non_exhaustive]
    #[error("missing ICC_PROFILE chunk {sequence}")]
    IccMissingChunk {
        /// The first absent sequence number.
        sequence: u8,
    },
    /// The standard XMP declares `xmpNote:HasExtendedXMP` but no extended
    /// XMP chunk carries that GUID.
    #[error("extended XMP declared but no chunk carries its GUID")]
    ExtendedXmpMissing,
    /// Extended XMP chunks with the declared GUID disagree on the full length.
    #[non_exhaustive]
    #[error("extended XMP chunks disagree on the full length ({expected} vs {found})")]
    ExtendedXmpLengthMismatch {
        /// Full length declared by the first chunk.
        expected: u32,
        /// Full length declared by a later chunk.
        found: u32,
    },
    /// Extended XMP chunks overlap or leave a gap.
    #[non_exhaustive]
    #[error("extended XMP chunk at offset {offset} does not continue the data at {expected}")]
    ExtendedXmpDiscontinuous {
        /// Offset the next chunk should start at.
        expected: u32,
        /// Offset the chunk declares.
        offset: u32,
    },
    /// An extended XMP chunk runs past the declared full length.
    #[non_exhaustive]
    #[error(
        "extended XMP chunk at offset {offset} with {length} bytes runs past the full length {full_length}"
    )]
    ExtendedXmpChunkPastEnd {
        /// Offset the chunk declares.
        offset: u32,
        /// Bytes the chunk carries.
        length: usize,
        /// Declared full length.
        full_length: u32,
    },
    /// The reassembled extended XMP is shorter than its declared full length.
    #[non_exhaustive]
    #[error("extended XMP ends at {got} of {full_length} bytes")]
    ExtendedXmpIncomplete {
        /// Bytes covered by the chunks.
        got: u32,
        /// Declared full length.
        full_length: u32,
    },
    /// The MD5 digest of the reassembled extended XMP does not match the
    /// GUID that names it (XMP Part 3 1.1.3.1).
    #[error("extended XMP digest does not match its GUID")]
    ExtendedXmpDigestMismatch,
    /// [`crate::container::metadata::MetadataLimits::max_items`] cut the walk before the header (or a
    /// Photoshop run) ended, so the requested payload may lie in segments
    /// that were not collected.
    #[error("the item limit cut the walk before the payload's segments ended")]
    ItemLimitReached,
    /// The MPF directory does not parse. (An allocation failure while
    /// parsing it is [`crate::container::metadata::MetadataError::OutOfMemory`].)
    #[error("MPF directory: {0}")]
    Mpf(#[source] MpfError),
}

impl From<MpfError> for MetadataError {
    /// [`crate::container::mpf::MpfError::OutOfMemory`] becomes
    /// [`crate::container::metadata::MetadataError::OutOfMemory`], so callers
    /// check one out-of-memory variant; every other error is
    /// [`crate::container::metadata::MetadataError::Mpf`].
    fn from(error: MpfError) -> Self {
        match error {
            MpfError::OutOfMemory => Self::OutOfMemory,
            error => Self::Mpf(error),
        }
    }
}

/// Every [`crate::container::metadata::Item`] of a JPEG buffer, plus validated reassembly of the
/// payloads that span several segments. See the [metadata documentation](crate::container::metadata).
#[derive(Debug, Clone)]
pub struct JpegMetadata<'a> {
    items: Vec<Item<'a>>,
    limits: MetadataLimits,
    primary_end: Option<usize>,
}

impl<'a> JpegMetadata<'a> {
    /// Walk `data` with [`crate::container::metadata::MetadataLimits::default`].
    ///
    /// Never panics on malformed input: unstructurable bytes become
    /// [`crate::container::metadata::ItemKind::Unparsed`] items. Fails only when an allocation fails
    /// (the item list, or the list of vendor trailers).
    pub fn parse(data: &'a [u8]) -> Result<Self, MetadataError> {
        Self::parse_with_limits(data, MetadataLimits::default())
    }

    /// Walk `data` with explicit limits.
    pub fn parse_with_limits(
        data: &'a [u8],
        limits: MetadataLimits,
    ) -> Result<Self, MetadataError> {
        let mut items = Vec::new();
        let primary_end = walk::collect(data, limits.max_items.max(1), &mut items)?;
        Ok(Self {
            items,
            limits,
            primary_end,
        })
    }

    /// The items in file order. They tile the input: each item starts where
    /// the previous one ends.
    #[must_use]
    pub fn items(&self) -> &[Item<'a>] {
        &self.items
    }

    /// Offset just past the primary image's EOI, where trailers begin.
    /// `None` when the walk did not reach an EOI: the input is truncated or
    /// not a JPEG, or [`crate::container::metadata::MetadataLimits::max_items`] was reached first (the
    /// [`crate::container::metadata::Unparsed::LimitExceeded`] item then covers the rest of the input,
    /// any EOI included).
    #[must_use]
    pub fn primary_image_end(&self) -> Option<usize> {
        self.primary_end
    }

    /// The header: the items before the primary image's first SOS, or
    /// before its EOI when there is no SOS, or all of them when there is
    /// neither; and whether the item limit cut it. A limit item that begins
    /// (after `0xFF` fill) with the SOS or EOI marker replaced the item that
    /// ends the header, so that header is complete.
    fn header(&self) -> (&[Item<'a>], bool) {
        let end = self
            .items
            .iter()
            .position(|item| item.marker == Some(0xDA) || matches!(item.kind, ItemKind::Eoi))
            .unwrap_or(self.items.len());
        let header = &self.items[..end];
        match header.split_last() {
            Some((last, rest))
                if matches!(last.kind, ItemKind::Unparsed(Unparsed::LimitExceeded)) =>
            {
                let after_fill = last.payload.iter().position(|&b| b != 0xFF);
                let ends_header =
                    after_fill.is_some_and(|i| i > 0 && matches!(last.payload[i], 0xDA | 0xD9));
                if ends_header {
                    (rest, false)
                } else {
                    (header, true)
                }
            }
            _ => (header, false),
        }
    }

    /// The first header item `f` accepts. A miss in a header that the item
    /// limit cut is [`crate::container::metadata::MetadataError::ItemLimitReached`]: the segment may lie
    /// past the cut.
    fn first_in_header<T>(
        &self,
        f: impl FnMut(&Item<'a>) -> Option<T>,
    ) -> Result<Option<T>, MetadataError> {
        let (header, cut) = self.header();
        match header.iter().find_map(f) {
            Some(found) => Ok(Some(found)),
            None if cut => Err(MetadataError::ItemLimitReached),
            None => Ok(None),
        }
    }

    /// The TIFF structure of the first EXIF APP1 segment in the header, as
    /// [`crate::container::metadata::ItemKind::Exif`] holds
    /// it, ready for a TIFF/EXIF parser.
    pub fn exif(&self) -> Result<Option<&'a [u8]>, MetadataError> {
        self.first_in_header(|item| match item.kind {
            ItemKind::Exif(tiff) => Some(tiff),
            _ => None,
        })
    }

    /// The first standard XMP packet in the header. Extended XMP is a
    /// separate packet (XMP Part 3): see
    /// [`extended_xmp`](crate::container::metadata::JpegMetadata::extended_xmp),
    /// which follows the last `HasExtendedXMP` declaration and so may be
    /// named by a later standard packet than this one.
    pub fn xmp(&self) -> Result<Option<&'a [u8]>, MetadataError> {
        self.first_in_header(|item| match item.kind {
            ItemKind::Xmp(packet) => Some(packet),
            _ => None,
        })
    }

    /// The extended XMP packet named by the header's `xmpNote:HasExtendedXMP`
    /// GUID, reassembled per XMP Specification Part 3, 1.1.3.1.
    ///
    /// The GUID is the last `HasExtendedXMP` value in the header's standard
    /// XMP packets, as in ExifTool (later values win). Chunks are selected by
    /// GUID (ignoring hex digit case), ordered by offset, and must agree on
    /// the full length and cover it exactly; the MD5 of the result must equal
    /// the GUID. The digest is computed over the chunks in place. Borrowed
    /// when one chunk holds the whole packet; otherwise the packet is
    /// allocated once, after the digest matches. `Ok(None)` when the header
    /// declares no extended XMP. Chunks with other GUIDs, and chunks after
    /// the first SOS, are not used and remain visible as items.
    ///
    /// The declaring packet need not be the one
    /// [`xmp`](crate::container::metadata::JpegMetadata::xmp) returns: with
    /// several standard packets, `xmp()` returns the first and this method
    /// follows the last declaration, which may sit in a later one. Where this
    /// differs from ExifTool (a GUID padded with whitespace, a repeated
    /// chunk, chunks that disagree on the full length, the name inside a
    /// comment) is listed in `docs/METADATA.md`.
    pub fn extended_xmp(&self) -> Result<Option<Cow<'a, [u8]>>, MetadataError> {
        let (header, cut) = self.header();
        if cut {
            return Err(MetadataError::ItemLimitReached);
        }
        reassemble::extended_xmp(header, &self.limits)
    }

    /// The ICC profile, reassembled from the header's `ICC_PROFILE` chunks
    /// per ICC.1:2010 Annex B.4.
    ///
    /// Borrowed when the profile is a single chunk; otherwise the chunks are
    /// joined in sequence order into one allocation. Chunks may appear in
    /// any order, but must agree on the chunk count, cover `1..=count`
    /// exactly once and hold at least one byte (the checks libjpeg-turbo's
    /// `jpeg_read_icc_profile` applies to the markers it reads before the
    /// first scan). Chunks after the first SOS are not used and remain
    /// visible as items. `Ok(None)` when the header holds no chunk.
    pub fn icc_profile(&self) -> Result<Option<Cow<'a, [u8]>>, MetadataError> {
        let (header, cut) = self.header();
        if cut {
            return Err(MetadataError::ItemLimitReached);
        }
        reassemble::icc_profile(header, &self.limits)
    }

    /// The Photoshop image resources of the header's first run of APP13
    /// Photoshop segments, each segment's header stripped. Borrowed for a
    /// single segment. The run rules follow ExifTool's `ProcessJPEG`, except
    /// for a run that starts with an `Adobe_Photoshop2.5:` segment (see
    /// `docs/METADATA.md`). `Ok(None)` when the header holds no Photoshop
    /// segment.
    pub fn photoshop(&self) -> Result<Option<PhotoshopIrb<'a>>, MetadataError> {
        let (header, _) = self.header();
        reassemble::photoshop(header, &self.limits)
    }

    /// The MPF directory of the header's first `MPF\0` APP2 segment, parsed
    /// by [`crate::container::mpf::parse_mpf_segment`] with absolute image
    /// offsets. `Ok(None)` when the header holds no MPF segment.
    pub fn mpf(&self) -> Result<Option<Vec<MpfEntry>>, MetadataError> {
        let segment = self.first_in_header(|item| match item.kind {
            // FF E2 + length word + "MPF\0" precede the TIFF header.
            ItemKind::Mpf(tiff) => Some((tiff, item.offset + 8)),
            _ => None,
        })?;
        match segment {
            Some((tiff, offset)) => Ok(Some(parse_mpf_segment(tiff, offset)?)),
            None => Ok(None),
        }
    }
}
