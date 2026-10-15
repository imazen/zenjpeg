//! End-to-end lossless JPEG transform pipeline.
//!
//! Takes JPEG bytes → Huffman-decodes to coefficients → transforms → re-encodes → JPEG bytes.
//! No IDCT or forward DCT is performed. Zero generation loss.

use alloc::vec::Vec;

use crate::decode::{DecodeConfig, DecodedExtras, PreserveConfig, PreservedSegment, SegmentType};
use crate::encode::extras::{MpfImage, XMP_NAMESPACE, generate_mpf_directory};
use crate::entropy::encoder::EntropyEncoder;
use crate::error::{Error, Result};
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER, MARKER_APP2, MARKER_DHT, MARKER_DQT, MARKER_DRI,
    MARKER_EOI, MARKER_SOF0, MARKER_SOI, MARKER_SOS,
};
use crate::huffman::encode::{HuffmanEncodeTable, build_code_lengths, lengths_to_bits_values};
use enough::Stop;
use zenmeta::xmp::container::{Directory, ItemSemantic};

/// Build a [`HuffmanEncodeTable`] from a 256-entry frequency array.
///
/// Appends the pseudo-symbol 256 (with frequency 1) before calling
/// `build_code_lengths`, ensuring the resulting Kraft sum is strictly less
/// than 2^16. Without this, tables built from exactly-fitting symbol sets
/// produce Kraft sum == 2^16, which is rejected as a "Bad Huffman Table" by
/// many decoders (e.g. zune-jpeg).
fn build_huffman_table(freq: &[u64; 256]) -> Result<HuffmanEncodeTable> {
    let mut freqs = alloc::vec::Vec::with_capacity(257);
    freqs.extend_from_slice(freq);
    freqs.push(1); // pseudo-symbol 256 ensures Kraft sum < 2^16
    let depths = build_code_lengths(&freqs, 16);
    let (bits, vals) = lengths_to_bits_values(&depths[..256]);
    HuffmanEncodeTable::from_bits_values(&bits, &vals)
}

use super::coeff_transform::{
    EdgeHandling, LosslessTransform, TransformConfig, TransformedCoefficients,
    transform_coefficients,
};
use super::exif::{parse_exif_orientation, set_exif_orientation};
use super::geometry::{McuGeom, ScanEvent, for_each_interleaved_event};

/// Perform a lossless JPEG transform.
///
/// Takes JPEG bytes, applies the specified transform to the DCT coefficients
/// (without decoding to pixels), and returns new JPEG bytes.
///
/// # Performance
///
/// Typically 3-5x faster than decode + pixel transform + encode, because
/// it skips IDCT, forward DCT, quantization, and color space conversion.
///
/// # Metadata
///
/// All metadata (EXIF, ICC, XMP, IPTC, comments) is preserved from the source.
/// EXIF orientation is NOT automatically updated — the caller should handle that.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::lossless::{transform, LosslessTransform, TransformConfig, EdgeHandling};
///
/// let rotated = transform(&jpeg_data, &TransformConfig {
///     transform: LosslessTransform::Rotate90,
///     edge_handling: EdgeHandling::RejectPartialBlocks,
/// }, enough::Unstoppable)?;
/// ```
pub fn transform(jpeg_data: &[u8], config: &TransformConfig, stop: impl Stop) -> Result<Vec<u8>> {
    transform_with(jpeg_data, config, false, Level::Primary, &stop)
}

/// Where an image sits in its file: the primary, or a secondary of it.
/// Secondaries are carried one level deep: a secondary's own MPF images are
/// never extracted, and a secondary that has secondaries of its own is
/// refused ([`refuse_nested`]), so a crafted file cannot multiply the work
/// by nesting containers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Primary,
    Secondary,
}

/// Shared body of [`transform`] and the path for MPF secondary images. With
/// `reset_exif_orientation`, any EXIF orientation tag in the output is set to
/// 1 (Normal) — the [`apply_exif_orientation`] contract, applied to
/// secondaries that carry their own tag.
fn transform_with(
    jpeg_data: &[u8],
    config: &TransformConfig,
    reset_exif_orientation: bool,
    level: Level,
    stop: &impl Stop,
) -> Result<Vec<u8>> {
    stop.check()?;

    // Step 1: Decode to coefficients + extract metadata in a single pass
    let preserve = match level {
        Level::Primary => PreserveConfig::all(),
        Level::Secondary => PreserveConfig::all()
            .mpf_gainmaps(false)
            .mpf_thumbnails(false)
            .mpf_multiframe(false)
            .mpf_depth(false),
    };
    let decoder = DecodeConfig::new().preserve(preserve);
    let (decoded_coeffs, extras) = decoder.decode_coefficients_with_extras(jpeg_data, stop)?;

    stop.check()?;

    // Step 2: Transform coefficients
    let transformed = transform_coefficients(&decoded_coeffs, config)
        .map_err(|e| Error::invalid_config(alloc::format!("{e}")))?;

    stop.check()?;

    // Step 3: The secondary images get the same transform.
    let carry = match level {
        Level::Primary => transform_secondary_images(
            jpeg_data,
            extras.as_ref(),
            config,
            reset_exif_orientation,
            (decoded_coeffs.width, decoded_coeffs.height),
            (transformed.width, transformed.height),
            stop,
        )?,
        Level::Secondary => {
            refuse_nested(jpeg_data, extras.as_ref())?;
            // Bytes after the secondary's EOI stay with it.
            Carry {
                tail: crate::container::marker::primary_bounds(jpeg_data)
                    .map_or(&[][..], |b| &jpeg_data[b.end..]),
                ..Carry::none()
            }
        }
    };

    // Step 4: Re-encode as JPEG
    let mut segments: Vec<PreservedSegment> =
        extras.map(|e| e.segments().to_vec()).unwrap_or_default();
    if reset_exif_orientation {
        for seg in &mut segments {
            if seg.segment_type == SegmentType::Exif {
                set_exif_orientation(&mut seg.data, 1);
            }
        }
    }
    encode_from_coefficients(&transformed, Some(&segments), &carry, 0, stop)
}

/// Each carried secondary is decoded and re-encoded at its declared size, so
/// the carry is bounded against the primary: one secondary may hold at most
/// the primary's pixels (gain maps, depth and confidence maps and MPF
/// thumbnails are subsampled or the same size; the second frame of an MPF
/// stereo pair is the same size), and all of them together at most this many
/// times the primary's pixels. A crafted file declaring many large
/// secondaries then costs at most this many extra passes of the primary's
/// size, however many it declares.
const SECONDARY_PIXEL_BUDGET: u64 = 4;

/// What the lossless writers place after the primary image's EOI.
pub(crate) struct Carry<'a> {
    /// The JPEG secondaries, in file order, after the primary's transform:
    /// listed in the rebuilt MPF index and placed directly after the primary.
    pub(super) images: Vec<MpfImage>,
    /// The bytes after the last JPEG secondary (or after the primary): a
    /// Motion Photo or `GCamera:MicroVideo` video, a vendor trailer. Copied
    /// verbatim after the images, so it keeps its distance from the end of
    /// the file, which is what those formats' offsets count.
    pub(super) tail: &'a [u8],
    /// `(Item:Semantic, Item:Length)` of each GContainer directory item that
    /// no carried image fills (a Motion Photo video in `tail`, or an item
    /// whose bytes are not in the file): its length is written back as it is.
    pub(super) verbatim_items: Vec<(String, usize)>,
}

impl Carry<'_> {
    /// Nothing follows the primary.
    pub(crate) fn none() -> Self {
        Self {
            images: Vec::new(),
            tail: &[],
            verbatim_items: Vec::new(),
        }
    }
}

/// Carry what follows the primary (Ultra HDR gain maps, depth maps, MPF
/// thumbnails and frames, then any other trailing bytes) through a
/// transform. Each JPEG secondary is geometrically bound to the primary, so
/// it receives the **same** transform through the same pipeline, one level
/// deep. Explicit trimming is accepted only when every image retains the same
/// proportional source rectangle. With [`LosslessTransform::None`] and no
/// EXIF fix-up the secondaries pass through verbatim.
///
/// Nothing is dropped silently: [`plan_carry`] says what is carried, and a
/// secondary that cannot be carried or transformed is an error, never
/// dropped or carried untransformed (either would leave the file describing
/// a gain map or thumbnail that no longer matches the primary).
pub(super) fn transform_secondary_images<'a>(
    jpeg_data: &'a [u8],
    extras: Option<&DecodedExtras>,
    config: &TransformConfig,
    reset_exif_orientation: bool,
    primary_source: (u32, u32),
    primary_output: (u32, u32),
    stop: &impl Stop,
) -> Result<Carry<'a>> {
    let verbatim = config.transform == LosslessTransform::None && !reset_exif_orientation;
    let plan = plan_carry(jpeg_data, extras, primary_source, !verbatim)?;
    let primary_px = u64::from(primary_source.0) * u64::from(primary_source.1);
    let budget = primary_px.saturating_mul(SECONDARY_PIXEL_BUDGET);
    let mut spent = 0u64;
    let mut images = Vec::new();
    images
        .try_reserve_exact(plan.sources.len())
        .map_err(|_| Error::allocation_failed(plan.sources.len(), "secondary image list"))?;
    for img in &plan.sources {
        stop.check()?;
        let data = if verbatim {
            let mut copy = Vec::new();
            copy.try_reserve_exact(img.data.len())
                .map_err(|_| Error::allocation_failed(img.data.len(), "secondary image"))?;
            copy.extend_from_slice(img.data);
            copy
        } else {
            let source = DecodeConfig::new()
                .read_info(img.data)
                .map_err(|e| in_secondary(e, img))?
                .dimensions;
            let px = u64::from(source.width) * u64::from(source.height);
            spent = spent.saturating_add(px);
            if px > primary_px {
                return Err(in_secondary(Error::image_too_large(px, primary_px), img));
            }
            if spent > budget {
                return Err(in_secondary(Error::image_too_large(spent, budget), img));
            }
            transform_with(
                img.data,
                config,
                reset_exif_orientation,
                Level::Secondary,
                stop,
            )
            .map_err(|e| in_secondary(e, img))?
        };
        if !verbatim && config.edge_handling == EdgeHandling::TrimPartialBlocks {
            // Trimming removes right/bottom source edges before the same D4
            // transform. Equal retained fractions on both axes therefore mean
            // equal normalized crop rectangles. Compare integer products to
            // avoid rounding away a primary/gain-map registration mismatch.
            let decoder = DecodeConfig::new();
            let source = decoder
                .read_info(img.data)
                .map_err(|e| in_secondary(e, img))?
                .dimensions;
            let output = decoder
                .read_info(&data)
                .map_err(|e| in_secondary(e, img))?
                .dimensions;
            let (pw, ph, sw, sh) = if config.transform.swaps_dimensions() {
                (
                    primary_source.1,
                    primary_source.0,
                    source.height,
                    source.width,
                )
            } else {
                (
                    primary_source.0,
                    primary_source.1,
                    source.width,
                    source.height,
                )
            };
            if u64::from(primary_output.0) * u64::from(sw)
                != u64::from(output.width) * u64::from(pw)
                || u64::from(primary_output.1) * u64::from(sh)
                    != u64::from(output.height) * u64::from(ph)
            {
                return Err(Error::invalid_config(alloc::format!(
                    "MPF secondary image #{} would retain a different region than the primary after trimming; use a coordinated crop or RejectPartialBlocks",
                    img.index,
                )));
            }
        }
        images.push(MpfImage {
            image_type: img.image_type,
            data,
        });
    }
    Ok(Carry {
        images,
        tail: plan.tail,
        verbatim_items: plan.verbatim_items,
    })
}

/// One JPEG secondary to carry: its MPF index (one past the last MPF entry
/// for a gain map found without an index entry), its MPF type, and its bytes
/// in the source, which start at `start`.
pub(super) struct SourceSecondary<'a> {
    index: usize,
    image_type: crate::encode::extras::MpfImageType,
    data: &'a [u8],
    start: usize,
}

/// What [`transform_secondary_images`] carries.
struct CarryPlan<'a> {
    sources: Vec<SourceSecondary<'a>>,
    tail: &'a [u8],
    verbatim_items: Vec<(String, usize)>,
}

/// Plan the carry of everything after the primary's EOI. Every byte there
/// is carried, or the plan is an error:
/// - Each MPF entry after the primary that the decoder extracted is a JPEG
///   secondary, transformed with the primary. An entry whose bytes lie past
///   the end of the file (the trailing images were stripped, as
///   `jpegtran -copy all` does) describes nothing that can be carried and is
///   left out of the rebuilt index; the XMP keeps its signalling as it was.
///   Any other entry that was not extracted (inside the primary, overlapping
///   or repeating another, a second primary, bytes that are not a JPEG) is
///   refused.
/// - When the XMP signals a gain map (`hdrgm` properties or a GContainer
///   `GainMap` item) and no MPF entry is one, a JPEG that starts right after
///   the last secondary is it, provided it parses and is no larger than the
///   primary; its length comes from the directory when it gives one, else
///   from its EOI. It is carried like an MPF secondary and gains an index
///   entry. When `transforming`, such a JPEG that cannot be read as a gain
///   map is refused rather than carried untransformed (without a transform
///   it stays in the tail). Bytes that do not start a JPEG are not taken as
///   one (a Motion Photo's MP4 may hold JPEGs inside).
/// - Bytes between the secondaries must be fill (`00` or `FF`, as GContainer
///   padding is written); they are dropped and the directory's padding set
///   to zero. Anything else there is refused.
/// - Whatever follows the last secondary is the verbatim [`Carry::tail`].
/// - When `transforming`, a GContainer directory image item (`image/*`) that
///   no carried JPEG fills is refused while bytes follow the secondaries:
///   carried verbatim it would no longer match the primary.
fn plan_carry<'a>(
    jpeg_data: &'a [u8],
    extras: Option<&DecodedExtras>,
    primary_dims: (u32, u32),
    transforming: bool,
) -> Result<CarryPlan<'a>> {
    use crate::encode::extras::MpfImageType;
    let Some(primary) = crate::container::marker::primary_bounds(jpeg_data) else {
        // No EOI: the primary runs to the end of the input.
        return Ok(CarryPlan {
            sources: Vec::new(),
            tail: &[],
            verbatim_items: Vec::new(),
        });
    };
    let xmp = extras.and_then(standard_xmp);
    // A directory that does not parse cannot be rewritten; that is refused
    // where a rewrite is needed (`write_preserved_segments`). When
    // transforming, it is also refused while bytes follow the primary: the
    // directory could describe an image there that must not stay untransformed.
    let parsed = xmp.map(Directory::parse);
    if transforming
        && primary.end < jpeg_data.len()
        && let Some(Err(e)) = &parsed
    {
        return Err(Error(
            Error::invalid_jpeg_data(
                "invalid XMP directory; lossless transforms cannot tell what the bytes after \
                 the primary are",
            )
            .0
            .at_string(|| alloc::format!("{e}")),
        ));
    }
    let directory = parsed.and_then(|d| d.ok());
    let items: Vec<&zenmeta::xmp::container::ContainerItem> = directory
        .as_ref()
        .map(|d| {
            d.items()
                .filter(|i| i.semantic != ItemSemantic::Primary)
                .collect()
        })
        .unwrap_or_default();

    let mut sources: Vec<SourceSecondary<'a>> = Vec::new();
    let mut mpf_entries = 0;
    if let Some((extras, dir)) = extras.and_then(|e| e.mpf().map(|d| (e, d))) {
        mpf_entries = dir.images.len();
        let mpf = crate::container::marker::iter(&jpeg_data[..primary.end])
            .find(|s| {
                s.kind == crate::container::MarkerKind::App(2) && s.payload.starts_with(b"MPF\0")
            })
            .ok_or_else(|| Error::invalid_jpeg_data("missing MPF segment"))?;
        // MPF offsets count from the index's TIFF header; 0 means "right
        // after the primary" (as the decoder reads them).
        let base = mpf.offset + 8;
        for (index, entry) in dir.images.iter().enumerate().skip(1) {
            let start = if entry.offset == 0 {
                primary.end
            } else {
                base.saturating_add(entry.offset as usize)
            };
            let range = start..start.saturating_add(entry.size as usize);
            if start < primary.end {
                // Decoding permits JPEGs embedded in APP payloads. This
                // writer emits independent trailing images and cannot
                // relocate those references without also rewriting the
                // containing payload.
                return Err(Error::invalid_jpeg_data(
                    "lossless rewriting cannot relocate an MPF image embedded in the primary JPEG",
                ));
            }
            match extras
                .secondary_images()
                .iter()
                .find(|s| s.mpf_index == index)
            {
                Some(s) => {
                    let data = jpeg_data
                        .get(range)
                        .filter(|d| *d == s.data.as_slice())
                        .ok_or_else(|| {
                            Error::internal("an extracted MPF image does not match its entry")
                        })?;
                    sources.push(SourceSecondary {
                        index,
                        image_type: s.image_type,
                        data,
                        start,
                    });
                }
                // Stripped: nothing of it is left to carry.
                None if range.end > jpeg_data.len() => {}
                None => {
                    return Err(Error::invalid_jpeg_data(
                        "the MPF index names an image that is not a separate JPEG after the \
                         primary (a repeated or overlapping entry, a second primary, or bytes \
                         that are not a JPEG); lossless transforms do not drop secondary images",
                    ));
                }
            }
        }
    }
    sources.sort_by_key(|s| s.start);
    let mut cursor = primary.end;
    for s in &sources {
        let gap = jpeg_data
            .get(cursor..s.start)
            .ok_or_else(|| Error::invalid_jpeg_data("overlapping MPF images"))?;
        if !gap.iter().all(|&b| b == 0x00 || b == 0xFF) {
            return Err(Error::invalid_jpeg_data(
                "bytes between the primary and an MPF image, or between two MPF images, that \
                 no index describes; lossless transforms do not drop them",
            ));
        }
        cursor = s.start + s.data.len();
    }

    let directory_gain_map = items.iter().find(|i| i.semantic == ItemSemantic::GainMap);
    let signals_gain_map = directory_gain_map.is_some()
        || xmp.is_some_and(|x| crate::container::xmp::parse_xmp(x).is_ok());
    if signals_gain_map
        && !sources
            .iter()
            .any(|s| s.image_type == MpfImageType::Undefined)
    {
        let rest = &jpeg_data[cursor..];
        if rest.starts_with(&[0xFF, 0xD8, 0xFF]) {
            let length = directory_gain_map.and_then(|i| i.length).or_else(|| {
                crate::container::marker::find_jpeg_boundaries(rest)
                    .first()
                    .filter(|r| r.start == 0)
                    .map(|r| r.end)
            });
            match length
                .and_then(|n| rest.get(..n))
                .filter(|g| fits_in_primary(g, primary_dims))
            {
                Some(gain_map) => {
                    sources.push(SourceSecondary {
                        index: mpf_entries.max(1),
                        image_type: MpfImageType::Undefined,
                        data: gain_map,
                        start: cursor,
                    });
                    cursor += gain_map.len();
                }
                // Untransformed, it stays where it is, in the tail.
                None if !transforming => {}
                None => {
                    return Err(Error::invalid_jpeg_data(
                        "the XMP describes a gain map after the primary that cannot be read as \
                         one (no EOI, or not a JPEG no larger than the primary); transforming \
                         the primary would leave it misaligned",
                    ));
                }
            }
        }
    }

    let carried: Vec<&str> = sources
        .iter()
        .map(|s| item_semantic(s.image_type))
        .collect();
    let mut verbatim_items = Vec::new();
    for item in &items {
        let semantic = item.semantic.as_xmp_str();
        if carried.contains(&semantic) {
            continue;
        }
        if transforming && cursor < jpeg_data.len() && item.mime.starts_with("image/") {
            return Err(Error::unsupported_feature(
                "the XMP GContainer directory lists an image that is not an MPF secondary of \
                 this file; carried untransformed it would no longer match the primary",
            ));
        }
        verbatim_items.push((String::from(semantic), item.length.unwrap_or(0)));
    }
    Ok(CarryPlan {
        sources,
        tail: &jpeg_data[cursor..],
        verbatim_items,
    })
}

/// `jpeg` parses as a JPEG no larger than `primary` on either axis.
fn fits_in_primary(jpeg: &[u8], primary: (u32, u32)) -> bool {
    DecodeConfig::new()
        .read_info(jpeg)
        .is_ok_and(|i| i.dimensions.width <= primary.0 && i.dimensions.height <= primary.1)
}

/// The standard XMP packet (without the extended-XMP chunks, which are
/// copied as they are): the GContainer directory and the `hdrgm` signalling
/// live there, and it is the packet the directory rewrite edits.
fn standard_xmp(extras: &DecodedExtras) -> Option<&str> {
    extras
        .segments()
        .iter()
        .find(|s| s.segment_type == SegmentType::Xmp)
        .and_then(|s| s.data.strip_prefix(XMP_NAMESPACE))
        .and_then(|x| core::str::from_utf8(x).ok())
}

/// `e`, raised while carrying `img`, with the secondary named. The kind is
/// kept, so a cancellation stays `Cancelled` and an edge rejection stays
/// `InvalidConfig`; kinds that carry a message get the secondary's index and
/// type in front of it, the others get it as trace context.
fn in_secondary(e: Error, img: &SourceSecondary<'_>) -> Error {
    let label = || alloc::format!("MPF secondary image #{} ({:?})", img.index, img.image_type);
    match e.kind() {
        crate::error::ErrorKind::InvalidConfig(m) => {
            Error::invalid_config(alloc::format!("{}: {m}", label()))
        }
        crate::error::ErrorKind::DecodeError(m) => {
            Error::decode_error(alloc::format!("{}: {m}", label()))
        }
        _ => Error(e.0.at_string(label)),
    }
}

/// A secondary image is carried as one image: it may not hold secondaries of
/// its own (an MPF index naming further images, a GContainer directory with
/// secondary items, or a second JPEG in its bytes). Those would be dropped,
/// and following them would let nesting multiply the work.
fn refuse_nested(jpeg_data: &[u8], extras: Option<&DecodedExtras>) -> Result<()> {
    let mpf = extras
        .and_then(|e| e.mpf())
        .is_some_and(|d| d.images.len() > 1);
    let directory = extras
        .and_then(standard_xmp)
        .map(Directory::parse)
        .transpose()
        .map_err(|e| {
            Error(
                Error::invalid_jpeg_data("invalid XMP directory")
                    .0
                    .at_string(|| alloc::format!("{e}")),
            )
        })?
        .is_some_and(|d| d.items().any(|i| i.semantic != ItemSemantic::Primary));
    let trailing = crate::container::marker::find_jpeg_boundaries(jpeg_data).len() > 1;
    if mpf || directory || trailing {
        return Err(Error::unsupported_feature(
            "an MPF secondary image holds secondary images of its own; lossless transforms \
             carry secondaries one level deep",
        ));
    }
    Ok(())
}

/// Write the preserved APPn/COM segments right after SOI.
///
/// The source's MPF index (APP2 `MPF\0`) is never copied: its offsets and
/// sizes describe the source layout. When `carry` holds images, a
/// placeholder index of the final size is written in its place (or after the
/// other segments if the source had none) and the GContainer XMP
/// `Item:Length` values are rewritten to the new image lengths (items carried
/// in the tail keep theirs). Without images the XMP is copied as it is, and
/// so is its signalling: an index whose images were stripped from the file
/// describes nothing the output holds, and is not written.
/// Returns the placeholder's offset for [`finish_container`], which patches
/// in the primary length once it is known.
pub(super) fn write_preserved_segments(
    output: &mut Vec<u8>,
    preserved: Option<&[PreservedSegment]>,
    carry: &Carry<'_>,
) -> Result<Option<usize>> {
    let secondaries = carry.images.as_slice();
    let mut mpf_at = None;
    let mut place_mpf = |output: &mut Vec<u8>| {
        if !secondaries.is_empty() && mpf_at.is_none() {
            let off = output.len();
            let data = generate_mpf_directory(secondaries.len(), 0, &mpf_sizes(secondaries), off);
            write_marker_segment(output, MARKER_APP2, &data);
            mpf_at = Some(off);
        }
    };
    if let Some(segments) = preserved {
        for seg in segments {
            match seg.segment_type {
                SegmentType::Mpf => place_mpf(output),
                SegmentType::Xmp if !secondaries.is_empty() => {
                    let carried: Vec<(&str, usize)> = secondaries
                        .iter()
                        .map(|s| (item_semantic(s.image_type), s.data.len()))
                        .chain(carry.verbatim_items.iter().map(|(s, n)| (s.as_str(), *n)))
                        .collect();
                    let rewritten = seg
                        .data
                        .strip_prefix(XMP_NAMESPACE)
                        .and_then(|x| core::str::from_utf8(x).ok())
                        .and_then(|x| {
                            Directory::parse(x)
                                .and_then(|d| d.rewrite_item_lengths(&carried))
                                .ok()
                        })
                        // Must still fit one APP1 segment (the digits may grow).
                        .filter(|x| XMP_NAMESPACE.len() + x.len() + 2 <= usize::from(u16::MAX));
                    match rewritten {
                        Some(x) => {
                            let mut data = Vec::with_capacity(XMP_NAMESPACE.len() + x.len());
                            data.extend_from_slice(XMP_NAMESPACE);
                            data.extend_from_slice(x.as_bytes());
                            write_marker_segment(output, seg.marker, &data);
                        }
                        None => {
                            return Err(Error::unsupported_feature(
                                "the XMP GContainer directory describes items the carried \
                                 secondary images do not match, or the rewritten packet no \
                                 longer fits one APP1 segment",
                            ));
                        }
                    }
                }
                _ => write_marker_segment(output, seg.marker, &seg.data),
            }
        }
    }
    place_mpf(output);
    Ok(mpf_at)
}

/// Complete a container assembled with [`write_preserved_segments`]: after the
/// primary's EOI, patch the MPF index with the primary's byte length, append
/// the secondary images, then the verbatim tail.
pub(super) fn finish_container(
    output: &mut Vec<u8>,
    carry: &Carry<'_>,
    mpf_at: Option<usize>,
) -> Result<()> {
    let secondaries = carry.images.as_slice();
    let Some(off) = mpf_at else {
        return append_tail(output, carry.tail);
    };
    let primary_len = u32::try_from(output.len())
        .map_err(|_| Error::unsupported_feature("MPF container with a primary image over 4 GiB"))?;
    let data = generate_mpf_directory(secondaries.len(), primary_len, &mpf_sizes(secondaries), off);
    // Same entry count as the placeholder → same length; only the primary size
    // and the secondary offsets (relative to the index) change.
    let start = off + 4;
    output[start..start + data.len()].copy_from_slice(&data);
    let total = secondaries
        .iter()
        .map(|s| s.data.len())
        .fold(0usize, usize::saturating_add);
    output
        .try_reserve(total)
        .map_err(|_| Error::allocation_failed(total, "MPF container"))?;
    for s in secondaries {
        output.extend_from_slice(&s.data);
    }
    append_tail(output, carry.tail)
}

fn append_tail(output: &mut Vec<u8>, tail: &[u8]) -> Result<()> {
    output
        .try_reserve(tail.len())
        .map_err(|_| Error::allocation_failed(tail.len(), "trailing data"))?;
    output.extend_from_slice(tail);
    Ok(())
}

/// The GContainer `Item:Semantic` an MPF secondary of this type fills: Ultra
/// HDR stores its gain map as MPF type Undefined, Dynamic Depth its depth map
/// as Disparity. Other MPF images (thumbnails, frames) are not directory items.
fn item_semantic(t: crate::encode::extras::MpfImageType) -> &'static str {
    use crate::encode::extras::MpfImageType;
    match t {
        MpfImageType::Undefined => "GainMap",
        MpfImageType::Disparity => "DepthMap",
        _ => "",
    }
}

fn mpf_sizes(secondaries: &[MpfImage]) -> Vec<(u32, crate::encode::extras::MpfImageType)> {
    secondaries
        .iter()
        // A secondary over 4 GiB cannot be indexed by MPF; clamp rather than
        // wrap so a reader sees a bounded, obviously-too-large entry instead
        // of a small wrong one.
        .map(|s| {
            (
                u32::try_from(s.data.len()).unwrap_or(u32::MAX),
                s.image_type,
            )
        })
        .collect()
}

/// Encode transformed coefficients back to JPEG bytes.
///
/// Writes a baseline sequential JPEG with:
/// - The coefficients' quantization tables
/// - Optimized Huffman tables (built from coefficient frequencies)
/// - Preserved metadata segments (if provided), with the MPF index rebuilt
/// - The MPF secondary images appended after EOI
/// - Optional restart markers at specified MCU intervals
pub(super) fn encode_from_coefficients(
    coeffs: &TransformedCoefficients,
    preserved_segments: Option<&[PreservedSegment]>,
    carry: &Carry<'_>,
    restart_interval: u16,
    stop: &impl Stop,
) -> Result<Vec<u8>> {
    let num_components = coeffs.components.len();
    // The emitter writes luma tables for component 0 and shared chroma tables
    // for components 1..3. Anything else (e.g. 4-component Adobe CMYK) would
    // previously have been silently dropped from the scan while still being
    // declared in the SOF — refuse loudly instead of emitting a corrupt file.
    if num_components != 1 && num_components != 3 {
        return Err(Error::unsupported_feature(
            "lossless re-encode of JPEGs with other than 1 or 3 components",
        ));
    }
    let is_color = num_components == 3;

    // Validated grid geometry — the single source of truth for the scan
    // traversal. Fails loudly if any component grid is inconsistent with the
    // declared dimensions (instead of silently emitting a scrambled stream).
    let geom = McuGeom::from_components(coeffs.width, coeffs.height, &coeffs.components)?;

    // Convert coefficients to block arrays
    let blocks: Vec<Vec<[i16; DCT_BLOCK_SIZE]>> =
        coeffs.components.iter().map(component_to_blocks).collect();

    stop.check()?;

    // ---- Pass 1: count symbol frequencies in EXACT encode order ----
    //
    // Counting and encoding share `for_each_interleaved_event`, so the
    // optimized tables cover precisely the symbols the encoder will emit.
    // (A frequency count taken in any other order can miss a DC category,
    // and a symbol without a code encodes as zero bits — a silently corrupt
    // stream. That was issue #194.)
    let total_mcus = geom.total_mcus();
    let mut dc_freq = [[0u64; 256]; 2];
    let mut ac_freq = [[0u64; 256]; 2];
    {
        let mut prev_dc = [0i16; 3];
        let mut restart_counter = restart_interval;
        let mut uncodable = false;
        for_each_interleaved_event(&geom, |ev| match ev {
            ScanEvent::Block { comp, idx } => {
                let t = usize::from(comp != 0);
                let block = &blocks[comp][idx];
                let dc_diff = block[0].wrapping_sub(prev_dc[comp]);
                uncodable |= dc_diff == i16::MIN || block[1..].contains(&i16::MIN);
                prev_dc[comp] = block[0];
                dc_freq[t][category(dc_diff) as usize] += 1;
                count_block_ac(block, &mut ac_freq[t]);
            }
            ScanEvent::McuEnd { mcu_idx } => {
                // Match the encoder: no restart after the final MCU.
                if restart_interval > 0 && mcu_idx + 1 < total_mcus {
                    restart_counter -= 1;
                    if restart_counter == 0 {
                        prev_dc = [0i16; 3];
                        restart_counter = restart_interval;
                    }
                }
            }
        });
        if uncodable {
            return Err(coefficient_out_of_range());
        }
    }

    let dc_luma_table = build_huffman_table(&dc_freq[0])?;
    let ac_luma_table = build_huffman_table(&ac_freq[0])?;
    let (dc_chroma_table, ac_chroma_table) = if is_color {
        (
            build_huffman_table(&dc_freq[1])?,
            build_huffman_table(&ac_freq[1])?,
        )
    } else {
        (
            HuffmanEncodeTable::std_dc_chrominance().clone(),
            HuffmanEncodeTable::std_ac_chrominance().clone(),
        )
    };

    stop.check()?;

    // ---- Pass 2: entropy-encode, identical traversal ----
    let total_blocks: usize = blocks.iter().map(|b| b.len()).sum();
    let mut encoder = EntropyEncoder::with_capacity(total_blocks * 3);
    encoder.set_dc_table(0, &dc_luma_table);
    encoder.set_ac_table(0, &ac_luma_table);
    if is_color {
        encoder.set_dc_table(1, &dc_chroma_table);
        encoder.set_ac_table(1, &ac_chroma_table);
    }
    if restart_interval > 0 {
        encoder.set_restart_interval(restart_interval);
    }
    for_each_interleaved_event(&geom, |ev| match ev {
        ScanEvent::Block { comp, idx } => {
            let t = usize::from(comp != 0);
            encoder.encode_block(&blocks[comp][idx], comp, t, t);
        }
        ScanEvent::McuEnd { mcu_idx } => {
            if mcu_idx + 1 < total_mcus {
                encoder.check_restart();
            }
        }
    });
    let scan_data = encoder.finish();

    stop.check()?;

    // Assemble the JPEG container
    let mut output = Vec::with_capacity(scan_data.len() + 1024);

    // SOI
    output.push(0xFF);
    output.push(MARKER_SOI);

    // Preserved metadata segments (EXIF, ICC, XMP, ...) + MPF index placeholder
    let mpf_at = write_preserved_segments(&mut output, preserved_segments, carry)?;

    // DQT - Write quantization tables
    write_quant_tables(&mut output, &coeffs.quant_tables, num_components);

    // SOF0 - Start of Frame (baseline)
    write_sof(&mut output, coeffs.width, coeffs.height, &coeffs.components);

    // DHT - Huffman tables
    write_huffman_table(&mut output, 0x00, &dc_luma_table); // DC luma, table 0
    write_huffman_table(&mut output, 0x10, &ac_luma_table); // AC luma, table 0
    if is_color {
        write_huffman_table(&mut output, 0x01, &dc_chroma_table); // DC chroma, table 1
        write_huffman_table(&mut output, 0x11, &ac_chroma_table); // AC chroma, table 1
    }

    // DRI - Restart interval (if enabled)
    if restart_interval > 0 {
        write_dri(&mut output, restart_interval);
    }

    // SOS - Start of Scan
    write_sos(&mut output, &coeffs.components);

    // Scan data
    output.extend_from_slice(&scan_data);

    // EOI
    output.push(0xFF);
    output.push(MARKER_EOI);

    // MPF index + secondary images
    finish_container(&mut output, carry, mpf_at)?;

    Ok(output)
}

/// Count AC symbol frequencies for one block (run-length/category symbols).
fn count_block_ac(block: &[i16; DCT_BLOCK_SIZE], ac_freq: &mut [u64; 256]) {
    let mut run = 0u8;
    for &ac in &block[1..] {
        if ac == 0 {
            run += 1;
        } else {
            while run >= 16 {
                ac_freq[0xF0] += 1; // ZRL
                run -= 16;
            }
            let ac_cat = category(ac);
            ac_freq[((run << 4) | ac_cat) as usize] += 1;
            run = 0;
        }
    }
    if run > 0 {
        ac_freq[0x00] += 1; // EOB
    }
}

/// Convert a `ComponentCoefficients` to a Vec of `[i16; 64]` blocks.
pub(super) fn component_to_blocks(
    comp: &crate::decode::ComponentCoefficients,
) -> Vec<[i16; DCT_BLOCK_SIZE]> {
    let num_blocks = comp.num_blocks();
    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let mut block = [0i16; DCT_BLOCK_SIZE];
        block.copy_from_slice(comp.block(i));
        blocks.push(block);
    }
    blocks
}

/// The error for a coefficient the re-encode cannot code.
///
/// Decoders keep DCT coefficients modulo 2^16 (the DC prediction wraps, as
/// libjpeg's `(JCOEF)` store does), so the re-encode takes DC differences
/// modulo 2^16 as well and reproduces every stored coefficient. The one value
/// no Huffman category holds is -32768 (category 16), as an AC value or as a
/// DC difference. No conforming 8-bit stream produces it; a crafted one can,
/// and libjpeg's encoder refuses it too (`JERR_BAD_DCT_COEF`).
pub(super) fn coefficient_out_of_range() -> Error {
    Error::invalid_jpeg_data(
        "DCT coefficient out of range for re-encoding: -32768 has no Huffman category",
    )
}

/// Return the Huffman category for a coefficient value.
///
/// Delegates to `entropy::category()` which uses a lookup table for the
/// common range and a scalar fallback for out-of-range values.
#[inline]
fn category(val: i16) -> u8 {
    crate::entropy::category(val)
}

// ===== JPEG container writing =====

pub(super) fn write_marker_segment(output: &mut Vec<u8>, marker: u8, data: &[u8]) {
    output.push(0xFF);
    output.push(marker);
    let len = (data.len() + 2) as u16;
    output.push((len >> 8) as u8);
    output.push((len & 0xFF) as u8);
    output.extend_from_slice(data);
}

pub(super) fn write_quant_tables(
    output: &mut Vec<u8>,
    quant_tables: &[Option<[u16; 64]>],
    _num_components: usize,
) {
    // Write ALL present quant tables (not just 2)
    for (idx, table) in quant_tables.iter().enumerate() {
        if let Some(qt) = table {
            let needs_16bit = qt.iter().any(|&v| v > 255);

            output.push(0xFF);
            output.push(MARKER_DQT);

            if needs_16bit {
                let len: u16 = 2 + 1 + 128;
                output.push((len >> 8) as u8);
                output.push((len & 0xFF) as u8);
                output.push(0x10 | idx as u8); // Pq=1 (16-bit), Tq=idx
                // Write in JPEG zigzag order (quant_tables are stored in natural order)
                for z in 0..64 {
                    let v = qt[JPEG_NATURAL_ORDER[z] as usize];
                    output.push((v >> 8) as u8);
                    output.push((v & 0xFF) as u8);
                }
            } else {
                let len: u16 = 2 + 1 + 64;
                output.push((len >> 8) as u8);
                output.push((len & 0xFF) as u8);
                output.push(idx as u8); // Pq=0 (8-bit), Tq=idx
                // Write in JPEG zigzag order (quant_tables are stored in natural order)
                for z in 0..64 {
                    let v = qt[JPEG_NATURAL_ORDER[z] as usize];
                    output.push(v as u8);
                }
            }
        }
    }
}

pub(super) fn write_sof(
    output: &mut Vec<u8>,
    width: u32,
    height: u32,
    components: &[crate::decode::ComponentCoefficients],
) {
    let num_components = components.len();
    let len = 2 + 1 + 2 + 2 + 1 + num_components * 3;

    output.push(0xFF);
    output.push(MARKER_SOF0);
    output.push((len >> 8) as u8);
    output.push((len & 0xFF) as u8);
    output.push(8); // Sample precision (8-bit)
    output.push((height >> 8) as u8);
    output.push((height & 0xFF) as u8);
    output.push((width >> 8) as u8);
    output.push((width & 0xFF) as u8);
    output.push(num_components as u8);

    for comp in components {
        output.push(comp.id);
        output.push((comp.h_samp << 4) | comp.v_samp);
        output.push(comp.quant_table_idx);
    }
}

pub(super) fn write_huffman_table(
    output: &mut Vec<u8>,
    table_class_and_id: u8,
    table: &HuffmanEncodeTable,
) {
    let (bits, values) = crate::huffman::encode::lengths_to_bits_values(&table.lengths);

    let len = 2 + 1 + 16 + values.len();
    output.push(0xFF);
    output.push(MARKER_DHT);
    output.push((len >> 8) as u8);
    output.push((len & 0xFF) as u8);
    output.push(table_class_and_id);
    output.extend_from_slice(&bits);
    output.extend_from_slice(&values);
}

fn write_sos(output: &mut Vec<u8>, components: &[crate::decode::ComponentCoefficients]) {
    let num_components = components.len();
    let len = 2 + 1 + num_components * 2 + 3;

    output.push(0xFF);
    output.push(MARKER_SOS);
    output.push((len >> 8) as u8);
    output.push((len & 0xFF) as u8);
    output.push(num_components as u8);

    for (i, comp) in components.iter().enumerate() {
        let table_sel = if i == 0 { 0x00 } else { 0x11 };
        output.push(comp.id);
        output.push(table_sel);
    }

    output.push(0x00); // Ss
    output.push(0x3F); // Se (63)
    output.push(0x00); // Ah/Al
}

/// Write a DRI (Define Restart Interval) marker.
pub(super) fn write_dri(output: &mut Vec<u8>, restart_interval: u16) {
    output.push(0xFF);
    output.push(MARKER_DRI);
    output.push(0x00);
    output.push(0x04); // Length = 4
    output.push((restart_interval >> 8) as u8);
    output.push((restart_interval & 0xFF) as u8);
}

/// Apply the EXIF orientation tag as a lossless DCT-domain transform.
///
/// Reads the EXIF orientation from the JPEG's metadata, applies the corresponding
/// lossless transform, and resets the orientation tag to 1 (Normal) in the output.
///
/// Returns an error if the primary or a retained MPF secondary would lose pixels.
/// Use [`crate::lossless::apply_exif_orientation_with_edge_handling`] to explicitly permit trimming.
/// This is a behavior change from the previous trimming default (#204).
///
/// If the orientation is already 1 (Normal), absent, or unrecognized, the input
/// is returned unchanged (fast path — no decode/re-encode).
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::lossless::apply_exif_orientation;
///
/// // Rotated camera photo → pixel-correct orientation, zero generation loss
/// let corrected = apply_exif_orientation(&jpeg_data, enough::Unstoppable)?;
/// ```
pub fn apply_exif_orientation(jpeg_data: &[u8], stop: impl Stop) -> Result<Vec<u8>> {
    apply_exif_orientation_with_edge_handling(jpeg_data, EdgeHandling::RejectPartialBlocks, stop)
}

/// Apply EXIF orientation with an explicit partial-MCU policy.
///
/// Like [`crate::lossless::apply_exif_orientation`], this resets EXIF orientation after a successful
/// transform. [`crate::lossless::EdgeHandling::TrimPartialBlocks`] explicitly permits pixel removal.
/// With MPF images, all images must retain the same proportional source region;
/// incompatible grids return an error instead of misaligning a gain map.
/// No output is returned if any required image cannot satisfy the policy.
pub fn apply_exif_orientation_with_edge_handling(
    jpeg_data: &[u8],
    edge_handling: EdgeHandling,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    // Step 1: Decode to coefficients + metadata in one pass and read the
    // EXIF orientation (a tag-less or upright file returns unchanged).
    let decoder = DecodeConfig::new().preserve(PreserveConfig::all());
    let (coeffs, extras) = decoder.decode_coefficients_with_extras(jpeg_data, &stop)?;

    let orientation = extras
        .as_ref()
        .and_then(|e| e.exif())
        .and_then(parse_exif_orientation);
    let orientation = match orientation {
        Some(o) if o != 1 => o,
        _ => return Ok(jpeg_data.to_vec()),
    };
    let lossless_transform = match LosslessTransform::from_exif_orientation(orientation) {
        Some(t) => t,
        None => return Ok(jpeg_data.to_vec()),
    };

    // Step 2: Transform coefficients
    let config = TransformConfig {
        transform: lossless_transform,
        edge_handling,
    };
    let transformed = transform_coefficients(&coeffs, &config)
        .map_err(|e| Error::invalid_config(alloc::format!("{e}")))?;

    stop.check()?;

    // Step 3: The MPF secondaries get the same rotation (and the same EXIF
    // orientation reset, should one carry the tag).
    let carry = transform_secondary_images(
        jpeg_data,
        extras.as_ref(),
        &config,
        true,
        (coeffs.width, coeffs.height),
        (transformed.width, transformed.height),
        &stop,
    )?;

    // Step 4: Re-encode, rewriting EXIF orientation to 1 (Normal)
    let mut segments: Vec<PreservedSegment> =
        extras.map(|e| e.segments().to_vec()).unwrap_or_default();
    for seg in &mut segments {
        if seg.segment_type == SegmentType::Exif {
            set_exif_orientation(&mut seg.data, 1);
        }
    }

    encode_from_coefficients(&transformed, Some(&segments), &carry, 0, &stop)
}

#[cfg(test)]
mod secondary_error_tests {
    use super::*;
    use crate::error::ErrorKind;

    fn img() -> SourceSecondary<'static> {
        SourceSecondary {
            index: 2,
            image_type: crate::encode::extras::MpfImageType::Undefined,
            data: &[],
            start: 0,
        }
    }

    /// A secondary's error keeps its kind; message-carrying kinds name the
    /// secondary.
    #[test]
    fn in_secondary_keeps_the_kind() {
        let e = in_secondary(Error::cancelled(), &img());
        assert!(matches!(e.kind(), ErrorKind::Cancelled(_)), "{e:?}");
        let e = in_secondary(Error::invalid_config("partial blocks".into()), &img());
        assert!(
            matches!(e.kind(), ErrorKind::InvalidConfig(m) if m.contains("#2") && m.contains("partial blocks")),
            "{e:?}"
        );
        let e = in_secondary(Error::allocation_failed(7, "x"), &img());
        assert!(
            matches!(e.kind(), ErrorKind::AllocationFailed { .. }),
            "{e:?}"
        );
    }
}
