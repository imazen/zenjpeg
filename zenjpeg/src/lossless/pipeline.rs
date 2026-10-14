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
    let secondaries = match level {
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
            Vec::new()
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
    encode_from_coefficients(&transformed, Some(&segments), &secondaries, 0, stop)
}

/// Carry the secondary images (Ultra HDR gain maps, depth maps, MPF
/// thumbnails and frames) through a transform. Each secondary is
/// geometrically bound to the primary, so it receives the **same** transform
/// through the same pipeline, one level deep. Explicit trimming is accepted
/// only when every image retains the same proportional source rectangle.
/// With [`LosslessTransform::None`] and no EXIF fix-up the bytes pass through
/// verbatim.
///
/// A secondary that cannot be carried or transformed is an error, never
/// silently dropped or carried untransformed: either would leave the file
/// describing a gain map (or thumbnail) that no longer matches the primary
/// ([`secondary_sources`] lists what must be carried).
pub(super) fn transform_secondary_images(
    jpeg_data: &[u8],
    extras: Option<&DecodedExtras>,
    config: &TransformConfig,
    reset_exif_orientation: bool,
    primary_source: (u32, u32),
    primary_output: (u32, u32),
    stop: &impl Stop,
) -> Result<Vec<MpfImage>> {
    let sources = secondary_sources(jpeg_data, extras)?;
    let verbatim = config.transform == LosslessTransform::None && !reset_exif_orientation;
    let mut out = Vec::new();
    out.try_reserve_exact(sources.len())
        .map_err(|_| Error::allocation_failed(sources.len(), "secondary image list"))?;
    for img in &sources {
        stop.check()?;
        let data = if verbatim {
            let mut copy = Vec::new();
            copy.try_reserve_exact(img.data.len())
                .map_err(|_| Error::allocation_failed(img.data.len(), "secondary image"))?;
            copy.extend_from_slice(img.data);
            copy
        } else {
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
        out.push(MpfImage {
            image_type: img.image_type,
            data,
        });
    }
    Ok(out)
}

/// One secondary image to carry: its MPF index (1 for a gain map found
/// without an MPF index), its MPF type, and its bytes in the source.
pub(super) struct SourceSecondary<'a> {
    index: usize,
    image_type: crate::encode::extras::MpfImageType,
    data: &'a [u8],
}

/// The secondary images of a file that the lossless pipeline must carry, or
/// an error when the file describes one it cannot carry. Nothing is dropped
/// silently:
/// - Every MPF entry after the primary must have been extracted. The decoder
///   takes readable, disjoint ranges inside the input, including embedded
///   JPEGs. This writer additionally requires placement after the primary EOI;
///   an index naming anything
///   else (a repeated or overlapping entry, a second primary, bytes that are
///   not a JPEG) is refused here instead of being dropped.
/// - A gain map signalled only by XMP (`hdrgm` properties or a GContainer
///   `GainMap` item, no MPF index, as some Android files are written) is the
///   JPEG that follows the primary; it is carried, and the output gains an
///   MPF index for it.
/// - The XMP GContainer directory may not list more secondary items than are
///   carried: an item that is not a carried JPEG (e.g. a Motion Photo video)
///   would be dropped while the directory still described it.
///
/// The carried ranges are disjoint parts of the input, so their total never
/// exceeds its length (checked again here).
fn secondary_sources<'a>(
    jpeg_data: &'a [u8],
    extras: Option<&'a DecodedExtras>,
) -> Result<Vec<SourceSecondary<'a>>> {
    let Some(extras) = extras else {
        return Ok(Vec::new());
    };
    let mut out: Vec<SourceSecondary<'a>> = extras
        .secondary_images()
        .iter()
        .map(|s| SourceSecondary {
            index: s.mpf_index,
            image_type: s.image_type,
            data: &s.data,
        })
        .collect();
    let xmp = extras.xmp();
    let directory = xmp.map(Directory::parse).transpose().map_err(|e| {
        Error(
            Error::invalid_jpeg_data("invalid XMP directory")
                .0
                .at_string(|| alloc::format!("{e}")),
        )
    })?;
    match extras.mpf() {
        Some(dir) => {
            // Decoding permits JPEGs embedded in APP payloads. This writer
            // emits independent trailing images and cannot relocate those
            // references without also rewriting the containing payload.
            let primary = crate::container::marker::primary_bounds(jpeg_data)
                .ok_or_else(|| Error::invalid_jpeg_data("missing primary JPEG bounds"))?;
            let mpf = crate::container::marker::iter(&jpeg_data[..primary.end])
                .find(|s| {
                    s.kind == crate::container::MarkerKind::App(2)
                        && s.payload.starts_with(b"MPF\0")
                })
                .ok_or_else(|| Error::invalid_jpeg_data("missing MPF segment"))?;
            let base = mpf.offset + 8;
            if dir.images.iter().skip(1).any(|entry| {
                entry.offset != 0 && base.saturating_add(entry.offset as usize) < primary.end
            }) {
                return Err(Error::invalid_jpeg_data(
                    "lossless rewriting cannot relocate an MPF image embedded in the primary JPEG",
                ));
            }

            if (1..dir.images.len()).any(|i| !out.iter().any(|s| s.index == i)) {
                return Err(Error::invalid_jpeg_data(
                    "the MPF index names an image that is not a separate JPEG after the primary \
                     (a repeated or overlapping entry, a second primary, or bytes outside the \
                     file); lossless transforms do not drop secondary images",
                ));
            }
        }
        None => {
            let signals_gain_map = xmp.is_some_and(|x| {
                crate::container::xmp::parse_xmp(x).is_ok()
                    || directory
                        .as_ref()
                        .is_some_and(|d| d.items().any(|i| i.semantic == ItemSemantic::GainMap))
            });
            if signals_gain_map {
                let ranges = crate::container::marker::find_jpeg_boundaries(jpeg_data);
                if let [primary, gain_map, ..] = ranges.as_slice()
                    && primary.start == 0
                    && gain_map.start >= primary.end
                {
                    out.push(SourceSecondary {
                        index: 1,
                        image_type: crate::encode::extras::MpfImageType::Undefined,
                        data: &jpeg_data[gain_map.clone()],
                    });
                }
            }
        }
    }
    let listed = directory.as_ref().map_or(0, |d| {
        d.items()
            .filter(|i| i.semantic != ItemSemantic::Primary)
            .count()
    });
    if listed > out.len() {
        return Err(Error::unsupported_feature(
            "the XMP GContainer directory lists items that are not secondary JPEGs of this \
             file (e.g. a Motion Photo video); lossless transforms do not drop them",
        ));
    }
    let total = out
        .iter()
        .map(|s| s.data.len())
        .fold(0usize, usize::saturating_add);
    if total > jpeg_data.len() {
        return Err(Error::invalid_jpeg_data(
            "secondary images larger than the file that holds them",
        ));
    }
    Ok(out)
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
        .and_then(|e| e.xmp())
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
/// sizes describe the source layout. When `secondaries` will follow the
/// primary, a placeholder index of the final size is written in its place
/// (or after the other segments if the source had none) and the GContainer
/// XMP `Item:Length` values are rewritten to the new secondary lengths.
/// Returns the placeholder's offset for [`finish_container`], which patches
/// in the primary length once it is known.
pub(super) fn write_preserved_segments(
    output: &mut Vec<u8>,
    preserved: Option<&[PreservedSegment]>,
    secondaries: &[MpfImage],
) -> Result<Option<usize>> {
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
/// primary's EOI, patch the MPF index with the primary's byte length and
/// append the secondary images.
pub(super) fn finish_container(
    output: &mut Vec<u8>,
    secondaries: &[MpfImage],
    mpf_at: Option<usize>,
) -> Result<()> {
    let Some(off) = mpf_at else {
        return Ok(());
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
    secondaries: &[MpfImage],
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
    let mpf_at = write_preserved_segments(&mut output, preserved_segments, secondaries)?;

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
    finish_container(&mut output, secondaries, mpf_at)?;

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
    let secondaries = transform_secondary_images(
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

    encode_from_coefficients(&transformed, Some(&segments), &secondaries, 0, &stop)
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
