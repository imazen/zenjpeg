//! End-to-end lossless JPEG transform pipeline.
//!
//! Takes JPEG bytes → Huffman-decodes to coefficients → transforms → re-encodes → JPEG bytes.
//! No IDCT or forward DCT is performed. Zero generation loss.

use alloc::vec;
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

use super::coeff_transform::{
    LosslessTransform, TransformConfig, TransformedCoefficients, transform_coefficients,
};
use super::exif::{parse_exif_orientation, set_exif_orientation};

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
    let is_color = num_components >= 3;

    // Determine subsampling from component sampling factors
    let (h_samp, v_samp) = if is_color {
        (
            coeffs.components[0].h_samp as usize,
            coeffs.components[0].v_samp as usize,
        )
    } else {
        (1, 1)
    };

    // Convert coefficients to block arrays
    let y_blocks = component_to_blocks(&coeffs.components[0]);
    let (cb_blocks, cr_blocks) = if is_color {
        (
            component_to_blocks(&coeffs.components[1]),
            component_to_blocks(&coeffs.components[2]),
        )
    } else {
        (vec![], vec![])
    };

    stop.check()?;

    // Build optimized Huffman tables from coefficient frequencies.
    // When restart markers are enabled, DC prediction resets at interval
    // boundaries, so frequency counting must match the encoder's MCU order.
    let (dc_luma_table, ac_luma_table, dc_chroma_table, ac_chroma_table) = if restart_interval > 0 {
        build_tables_with_restart(
            &y_blocks,
            &cb_blocks,
            &cr_blocks,
            is_color,
            coeffs.width as usize,
            coeffs.height as usize,
            h_samp,
            v_samp,
            restart_interval,
        )?
    } else {
        let (dc_luma, ac_luma) = build_tables_from_blocks(&y_blocks)?;
        let (dc_chroma, ac_chroma) = if is_color {
            let mut dc_freq = [0u64; 256];
            let mut ac_freq = [0u64; 256];
            count_frequencies(&cb_blocks, &mut dc_freq, &mut ac_freq);
            count_frequencies(&cr_blocks, &mut dc_freq, &mut ac_freq);
            let dc_lengths = build_code_lengths(&dc_freq, 16);
            let ac_lengths = build_code_lengths(&ac_freq, 16);
            let (dc_bits, dc_vals) = lengths_to_bits_values(&dc_lengths);
            let (ac_bits, ac_vals) = lengths_to_bits_values(&ac_lengths);
            (
                HuffmanEncodeTable::from_bits_values(&dc_bits, &dc_vals)?,
                HuffmanEncodeTable::from_bits_values(&ac_bits, &ac_vals)?,
            )
        } else {
            (
                HuffmanEncodeTable::std_dc_chrominance().clone(),
                HuffmanEncodeTable::std_ac_chrominance().clone(),
            )
        };
        (dc_luma, ac_luma, dc_chroma, ac_chroma)
    };

    stop.check()?;

    // Entropy-encode all blocks
    let scan_data = encode_scan_data(
        &y_blocks,
        &cb_blocks,
        &cr_blocks,
        is_color,
        coeffs.width as usize,
        coeffs.height as usize,
        h_samp,
        v_samp,
        &dc_luma_table,
        &ac_luma_table,
        &dc_chroma_table,
        &ac_chroma_table,
        restart_interval,
    );

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

/// Build optimized Huffman tables with restart-aware DC prediction.
///
/// When restart markers are enabled, DC prediction resets to 0 at each
/// restart boundary. Frequency counting must match the encoder's MCU
/// iteration order to produce correct Huffman tables.
#[allow(clippy::too_many_arguments)]
fn build_tables_with_restart(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    is_color: bool,
    width: usize,
    height: usize,
    h_samp: usize,
    v_samp: usize,
    restart_interval: u16,
) -> Result<(
    HuffmanEncodeTable,
    HuffmanEncodeTable,
    HuffmanEncodeTable,
    HuffmanEncodeTable,
)> {
    let mut dc_luma_freq = [0u64; 256];
    let mut ac_luma_freq = [0u64; 256];
    let mut dc_chroma_freq = [0u64; 256];
    let mut ac_chroma_freq = [0u64; 256];

    // DC predictors per component
    let mut prev_dc = [0i16; 4]; // Y=0, Cb=1, Cr=2
    let mut restart_counter = restart_interval;

    let count_block_ac = |block: &[i16; DCT_BLOCK_SIZE], ac_freq: &mut [u64; 256]| {
        let mut run = 0u8;
        for &ac in &block[1..] {
            if ac == 0 {
                run += 1;
            } else {
                while run >= 16 {
                    ac_freq[0xF0] += 1;
                    run -= 16;
                }
                let ac_cat = category(ac);
                ac_freq[((run << 4) | ac_cat) as usize] += 1;
                run = 0;
            }
        }
        if run > 0 {
            ac_freq[0x00] += 1;
        }
    };

    if h_samp == 1 && v_samp == 1 {
        let total_mcus = y_blocks.len();
        for (i, y_block) in y_blocks.iter().enumerate() {
            // Y DC
            let dc_diff = y_block[0] - prev_dc[0];
            prev_dc[0] = y_block[0];
            dc_luma_freq[category(dc_diff) as usize] += 1;
            count_block_ac(y_block, &mut ac_luma_freq);

            if is_color {
                // Cb DC
                let dc_diff = cb_blocks[i][0] - prev_dc[1];
                prev_dc[1] = cb_blocks[i][0];
                dc_chroma_freq[category(dc_diff) as usize] += 1;
                count_block_ac(&cb_blocks[i], &mut ac_chroma_freq);

                // Cr DC
                let dc_diff = cr_blocks[i][0] - prev_dc[2];
                prev_dc[2] = cr_blocks[i][0];
                dc_chroma_freq[category(dc_diff) as usize] += 1;
                count_block_ac(&cr_blocks[i], &mut ac_chroma_freq);
            }

            // Check restart (match encoder: skip on last MCU)
            if i + 1 < total_mcus {
                restart_counter -= 1;
                if restart_counter == 0 {
                    prev_dc = [0; 4];
                    restart_counter = restart_interval;
                }
            }
        }
    } else {
        let y_blocks_w = (width + 7) / 8;
        let y_blocks_h = (height + 7) / 8;
        let c_blocks_w = (y_blocks_w + h_samp - 1) / h_samp;
        let c_blocks_h = (y_blocks_h + v_samp - 1) / v_samp;
        let total_mcus = c_blocks_w * c_blocks_h;

        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        let mut mcu_idx = 0;
        for mcu_y in 0..c_blocks_h {
            for mcu_x in 0..c_blocks_w {
                for dy in 0..v_samp {
                    for dx in 0..h_samp {
                        let y_bx = mcu_x * h_samp + dx;
                        let y_by = mcu_y * v_samp + dy;
                        let block = if y_bx < y_blocks_w && y_by < y_blocks_h {
                            &y_blocks[y_by * y_blocks_w + y_bx]
                        } else {
                            &ZERO_BLOCK
                        };
                        let dc_diff = block[0] - prev_dc[0];
                        prev_dc[0] = block[0];
                        dc_luma_freq[category(dc_diff) as usize] += 1;
                        count_block_ac(block, &mut ac_luma_freq);
                    }
                }

                if is_color {
                    let c_idx = mcu_y * c_blocks_w + mcu_x;
                    let cb = if c_idx < cb_blocks.len() {
                        &cb_blocks[c_idx]
                    } else {
                        &ZERO_BLOCK
                    };
                    let cr = if c_idx < cr_blocks.len() {
                        &cr_blocks[c_idx]
                    } else {
                        &ZERO_BLOCK
                    };

                    let dc_diff = cb[0] - prev_dc[1];
                    prev_dc[1] = cb[0];
                    dc_chroma_freq[category(dc_diff) as usize] += 1;
                    count_block_ac(cb, &mut ac_chroma_freq);

                    let dc_diff = cr[0] - prev_dc[2];
                    prev_dc[2] = cr[0];
                    dc_chroma_freq[category(dc_diff) as usize] += 1;
                    count_block_ac(cr, &mut ac_chroma_freq);
                }

                mcu_idx += 1;
                if mcu_idx < total_mcus {
                    restart_counter -= 1;
                    if restart_counter == 0 {
                        prev_dc = [0; 4];
                        restart_counter = restart_interval;
                    }
                }
            }
        }
    }

    // Build tables from frequencies
    let dc_luma_lengths = build_code_lengths(&dc_luma_freq, 16);
    let ac_luma_lengths = build_code_lengths(&ac_luma_freq, 16);
    let (dc_luma_bits, dc_luma_vals) = lengths_to_bits_values(&dc_luma_lengths);
    let (ac_luma_bits, ac_luma_vals) = lengths_to_bits_values(&ac_luma_lengths);

    let dc_luma_table = HuffmanEncodeTable::from_bits_values(&dc_luma_bits, &dc_luma_vals)?;
    let ac_luma_table = HuffmanEncodeTable::from_bits_values(&ac_luma_bits, &ac_luma_vals)?;

    let (dc_chroma_table, ac_chroma_table) = if is_color {
        let dc_lengths = build_code_lengths(&dc_chroma_freq, 16);
        let ac_lengths = build_code_lengths(&ac_chroma_freq, 16);
        let (dc_bits, dc_vals) = lengths_to_bits_values(&dc_lengths);
        let (ac_bits, ac_vals) = lengths_to_bits_values(&ac_lengths);
        (
            HuffmanEncodeTable::from_bits_values(&dc_bits, &dc_vals)?,
            HuffmanEncodeTable::from_bits_values(&ac_bits, &ac_vals)?,
        )
    } else {
        (
            HuffmanEncodeTable::std_dc_chrominance().clone(),
            HuffmanEncodeTable::std_ac_chrominance().clone(),
        )
    };

    Ok((
        dc_luma_table,
        ac_luma_table,
        dc_chroma_table,
        ac_chroma_table,
    ))
}

/// Build optimized Huffman tables from a set of coefficient blocks.
pub(super) fn build_tables_from_blocks(
    blocks: &[[i16; DCT_BLOCK_SIZE]],
) -> Result<(HuffmanEncodeTable, HuffmanEncodeTable)> {
    let mut dc_freq = [0u64; 256];
    let mut ac_freq = [0u64; 256];
    count_frequencies(blocks, &mut dc_freq, &mut ac_freq);

    let dc_lengths = build_code_lengths(&dc_freq, 16);
    let ac_lengths = build_code_lengths(&ac_freq, 16);
    let (dc_bits, dc_vals) = lengths_to_bits_values(&dc_lengths);
    let (ac_bits, ac_vals) = lengths_to_bits_values(&ac_lengths);

    Ok((
        HuffmanEncodeTable::from_bits_values(&dc_bits, &dc_vals)?,
        HuffmanEncodeTable::from_bits_values(&ac_bits, &ac_vals)?,
    ))
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

/// Count Huffman symbol frequencies for DC and AC coefficients.
pub(super) fn count_frequencies(
    blocks: &[[i16; DCT_BLOCK_SIZE]],
    dc_freq: &mut [u64; 256],
    ac_freq: &mut [u64; 256],
) {
    let mut prev_dc: i16 = 0;
    for block in blocks {
        // DC
        let dc_diff = block[0] - prev_dc;
        prev_dc = block[0];
        let dc_cat = category(dc_diff);
        dc_freq[dc_cat as usize] += 1;

        // AC
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
                let symbol = (run << 4) | ac_cat;
                ac_freq[symbol as usize] += 1;
                run = 0;
            }
        }
        if run > 0 {
            ac_freq[0x00] += 1; // EOB
        }
    }
}

/// Encode scan data using the entropy encoder.
#[allow(clippy::too_many_arguments)]
fn encode_scan_data(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    is_color: bool,
    width: usize,
    height: usize,
    h_samp: usize,
    v_samp: usize,
    dc_luma: &HuffmanEncodeTable,
    ac_luma: &HuffmanEncodeTable,
    dc_chroma: &HuffmanEncodeTable,
    ac_chroma: &HuffmanEncodeTable,
    restart_interval: u16,
) -> Vec<u8> {
    let total_blocks = y_blocks.len() + cb_blocks.len() + cr_blocks.len();
    let mut encoder = EntropyEncoder::with_capacity(total_blocks * 3);

    encoder.set_dc_table(0, dc_luma);
    encoder.set_ac_table(0, ac_luma);
    if is_color {
        encoder.set_dc_table(1, dc_chroma);
        encoder.set_ac_table(1, ac_chroma);
    }

    if restart_interval > 0 {
        encoder.set_restart_interval(restart_interval);
    }

    if h_samp == 1 && v_samp == 1 {
        // 4:4:4 — simple interleaving
        let total_mcus = y_blocks.len();
        for (i, y_block) in y_blocks.iter().enumerate() {
            encoder.encode_block(y_block, 0, 0, 0);
            if is_color {
                encoder.encode_block(&cb_blocks[i], 1, 1, 1);
                encoder.encode_block(&cr_blocks[i], 2, 1, 1);
            }
            // Only check restart if not the last MCU
            if i + 1 < total_mcus {
                encoder.check_restart();
            }
        }
    } else {
        // Subsampled — MCU interleaving
        let y_blocks_w = (width + 7) / 8;
        let y_blocks_h = (height + 7) / 8;
        let c_blocks_w = (y_blocks_w + h_samp - 1) / h_samp;
        let c_blocks_h = (y_blocks_h + v_samp - 1) / v_samp;
        let total_mcus = c_blocks_w * c_blocks_h;

        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        let mut mcu_idx = 0;
        for mcu_y in 0..c_blocks_h {
            for mcu_x in 0..c_blocks_w {
                // Y blocks in this MCU
                for dy in 0..v_samp {
                    for dx in 0..h_samp {
                        let y_bx = mcu_x * h_samp + dx;
                        let y_by = mcu_y * v_samp + dy;
                        if y_bx < y_blocks_w && y_by < y_blocks_h {
                            let y_idx = y_by * y_blocks_w + y_bx;
                            encoder.encode_block(&y_blocks[y_idx], 0, 0, 0);
                        } else {
                            encoder.encode_block(&ZERO_BLOCK, 0, 0, 0);
                        }
                    }
                }
                // Cb, Cr blocks
                if is_color {
                    let c_idx = mcu_y * c_blocks_w + mcu_x;
                    if c_idx < cb_blocks.len() {
                        encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1);
                        encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1);
                    } else {
                        encoder.encode_block(&ZERO_BLOCK, 1, 1, 1);
                        encoder.encode_block(&ZERO_BLOCK, 2, 1, 1);
                    }
                }
                // Only check restart if not the last MCU
                mcu_idx += 1;
                if mcu_idx < total_mcus {
                    encoder.check_restart();
                }
            }
        }
    }

    encoder.finish()
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
