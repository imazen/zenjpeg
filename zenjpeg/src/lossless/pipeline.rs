//! End-to-end lossless JPEG transform pipeline.
//!
//! Takes JPEG bytes → Huffman-decodes to coefficients → transforms → re-encodes → JPEG bytes.
//! No IDCT or forward DCT is performed. Zero generation loss.

use alloc::vec::Vec;

use crate::decode::{DecodeConfig, PreserveConfig};
use crate::entropy::encoder::EntropyEncoder;
use crate::error::{Error, Result};
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER, MARKER_DHT, MARKER_DQT, MARKER_DRI, MARKER_EOI,
    MARKER_SOF0, MARKER_SOI, MARKER_SOS,
};
use crate::huffman::encode::{HuffmanEncodeTable, build_code_lengths, lengths_to_bits_values};
use enough::Stop;

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
    LosslessTransform, TransformConfig, TransformedCoefficients, transform_coefficients,
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
///     edge_handling: EdgeHandling::TrimPartialBlocks,
/// }, enough::Unstoppable)?;
/// ```
pub fn transform(jpeg_data: &[u8], config: &TransformConfig, stop: impl Stop) -> Result<Vec<u8>> {
    stop.check()?;

    // Step 1: Decode to coefficients + extract metadata in a single pass
    let decoder = DecodeConfig::new().preserve(PreserveConfig::all());
    let (decoded_coeffs, extras) = decoder.decode_coefficients_with_extras(jpeg_data, &stop)?;

    stop.check()?;

    // Step 2: Transform coefficients
    let transformed = transform_coefficients(&decoded_coeffs, config)
        .map_err(|e| Error::io_error(alloc::format!("{e}")))?;

    stop.check()?;

    // Step 3: Re-encode as JPEG
    let preserved = extras.as_ref().map(|e| e.segments());
    let output = encode_from_coefficients(&transformed, preserved, 0, &stop)?;

    Ok(output)
}

/// Encode transformed coefficients back to JPEG bytes.
///
/// Writes a baseline sequential JPEG with:
/// - Same quantization tables as the source
/// - Optimized Huffman tables (built from coefficient frequencies)
/// - Preserved metadata segments (if provided)
/// - Optional restart markers at specified MCU intervals
pub(super) fn encode_from_coefficients(
    coeffs: &TransformedCoefficients,
    preserved_segments: Option<&[crate::decode::PreservedSegment]>,
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
        for_each_interleaved_event(&geom, |ev| match ev {
            ScanEvent::Block { comp, idx } => {
                let t = usize::from(comp != 0);
                let block = &blocks[comp][idx];
                let dc_diff = block[0] - prev_dc[comp];
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

    // Write preserved metadata segments (EXIF, ICC, XMP, etc.)
    if let Some(segments) = preserved_segments {
        for seg in segments {
            write_marker_segment(&mut output, seg.marker, &seg.data);
        }
    }

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
    // Step 1: Quick scan for EXIF orientation without full decode
    let decoder = DecodeConfig::new().preserve(PreserveConfig::all());
    let (coeffs, extras) = decoder.decode_coefficients_with_extras(jpeg_data, &stop)?;

    // Find EXIF segment and parse orientation
    let orientation = extras
        .as_ref()
        .and_then(|e| e.exif())
        .and_then(parse_exif_orientation);

    // Fast path: no rotation needed
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
        ..Default::default()
    };
    let transformed = transform_coefficients(&coeffs, &config)
        .map_err(|e| Error::io_error(alloc::format!("{e}")))?;

    stop.check()?;

    // Step 3: Re-encode, rewriting EXIF orientation to 1
    // Clone the preserved segments so we can modify the EXIF orientation
    let mut segments: Vec<crate::decode::PreservedSegment> =
        extras.map(|e| e.segments().to_vec()).unwrap_or_default();

    // Find and rewrite EXIF orientation to 1 (Normal)
    for seg in &mut segments {
        if seg.segment_type == crate::decode::SegmentType::Exif {
            set_exif_orientation(&mut seg.data, 1);
        }
    }

    let output = encode_from_coefficients(&transformed, Some(&segments), 0, &stop)?;
    Ok(output)
}
