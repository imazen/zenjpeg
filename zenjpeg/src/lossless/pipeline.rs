//! End-to-end lossless JPEG transform pipeline.
//!
//! Takes JPEG bytes → Huffman-decodes to coefficients → transforms → re-encodes → JPEG bytes.
//! No IDCT or forward DCT is performed. Zero generation loss.

use alloc::vec;
use alloc::vec::Vec;

use crate::decode::{DecodeConfig, DecodedCoefficients, PreserveConfig};
use crate::entropy::encoder::EntropyEncoder;
use crate::error::{Error, Result};
use crate::foundation::consts::{
    JPEG_NATURAL_ORDER,
    DCT_BLOCK_SIZE, MARKER_DHT, MARKER_DQT, MARKER_EOI, MARKER_SOF0, MARKER_SOI, MARKER_SOS,
};
use crate::huffman::encode::{build_code_lengths, lengths_to_bits_values, HuffmanEncodeTable};
use crate::types::Subsampling;
use enough::Stop;

use super::coeff_transform::{
    transform_coefficients, EdgeHandling, LosslessTransform, TransformConfig,
    TransformedCoefficients,
};

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
///     edge_handling: EdgeHandling::Trim,
/// }, enough::Unstoppable)?;
/// ```
pub fn transform(jpeg_data: &[u8], config: &TransformConfig, stop: impl Stop) -> Result<Vec<u8>> {
    stop.check()?;

    // Step 1: Decode to coefficients + extract metadata for round-tripping
    let decoder = DecodeConfig::new().preserve(PreserveConfig::all());
    let mut decode_result = decoder.decode(jpeg_data, &stop)?;
    let extras = decode_result.take_extras();

    // Also get the coefficients
    let decoded_coeffs = decoder.decode_coefficients(jpeg_data, &stop)?;

    stop.check()?;

    // Step 2: Transform coefficients
    let transformed = transform_coefficients(&decoded_coeffs, config)
        .map_err(|e| Error::io_error(alloc::format!("{e}")))?;

    stop.check()?;

    // Step 3: Re-encode as JPEG
    let preserved = extras.as_ref().map(|e| e.segments());
    let output = encode_from_coefficients(&transformed, preserved, &stop)?;

    Ok(output)
}

/// Encode transformed coefficients back to JPEG bytes.
///
/// Writes a baseline sequential JPEG with:
/// - Same quantization tables as the source
/// - Optimized Huffman tables (built from coefficient frequencies)
/// - Preserved metadata segments (if provided)
fn encode_from_coefficients(
    coeffs: &TransformedCoefficients,
    preserved_segments: Option<&[crate::decode::PreservedSegment]>,
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

    // Build optimized Huffman tables from coefficient frequencies
    let (dc_luma_table, ac_luma_table) = build_tables_from_blocks(&y_blocks)?;
    let (dc_chroma_table, ac_chroma_table) = if is_color {
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
        // Unused but needed for type consistency
        (
            HuffmanEncodeTable::std_dc_chrominance().clone(),
            HuffmanEncodeTable::std_ac_chrominance().clone(),
        )
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
    write_sof(
        &mut output,
        coeffs.width,
        coeffs.height,
        &coeffs.components,
    );

    // DHT - Huffman tables
    write_huffman_table(&mut output, 0x00, &dc_luma_table); // DC luma, table 0
    write_huffman_table(&mut output, 0x10, &ac_luma_table); // AC luma, table 0
    if is_color {
        write_huffman_table(&mut output, 0x01, &dc_chroma_table); // DC chroma, table 1
        write_huffman_table(&mut output, 0x11, &ac_chroma_table); // AC chroma, table 1
    }

    // SOS - Start of Scan
    write_sos(&mut output, num_components);

    // Scan data
    output.extend_from_slice(&scan_data);

    // EOI
    output.push(0xFF);
    output.push(MARKER_EOI);

    Ok(output)
}

/// Build optimized Huffman tables from a set of coefficient blocks.
fn build_tables_from_blocks(blocks: &[[i16; DCT_BLOCK_SIZE]]) -> Result<(HuffmanEncodeTable, HuffmanEncodeTable)> {
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
fn component_to_blocks(
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
fn count_frequencies(
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
) -> Vec<u8> {
    let total_blocks = y_blocks.len() + cb_blocks.len() + cr_blocks.len();
    let mut encoder = EntropyEncoder::with_capacity(total_blocks * 3);

    encoder.set_dc_table(0, dc_luma);
    encoder.set_ac_table(0, ac_luma);
    if is_color {
        encoder.set_dc_table(1, dc_chroma);
        encoder.set_ac_table(1, ac_chroma);
    }

    if h_samp == 1 && v_samp == 1 {
        // 4:4:4 — simple interleaving
        for (i, y_block) in y_blocks.iter().enumerate() {
            encoder.encode_block(y_block, 0, 0, 0);
            if is_color {
                encoder.encode_block(&cb_blocks[i], 1, 1, 1);
                encoder.encode_block(&cr_blocks[i], 2, 1, 1);
            }
        }
    } else {
        // Subsampled — MCU interleaving
        let y_blocks_w = (width + 7) / 8;
        let y_blocks_h = (height + 7) / 8;
        let c_blocks_w = (y_blocks_w + h_samp - 1) / h_samp;
        let c_blocks_h = (y_blocks_h + v_samp - 1) / v_samp;

        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

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
            }
        }
    }

    encoder.finish()
}

/// Return the Huffman category for a coefficient value.
fn category(val: i16) -> u8 {
    let abs = val.unsigned_abs();
    if abs == 0 {
        0
    } else {
        16 - abs.leading_zeros() as u8
    }
}

// ===== JPEG container writing =====

fn write_marker_segment(output: &mut Vec<u8>, marker: u8, data: &[u8]) {
    output.push(0xFF);
    output.push(marker);
    let len = (data.len() + 2) as u16;
    output.push((len >> 8) as u8);
    output.push((len & 0xFF) as u8);
    output.extend_from_slice(data);
}

fn write_quant_tables(
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

fn write_sof(
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

    for (i, comp) in components.iter().enumerate() {
        output.push(comp.id);
        output.push((comp.h_samp << 4) | comp.v_samp);
        output.push(comp.quant_table_idx);
    }
}

fn write_huffman_table(
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

fn write_sos(output: &mut Vec<u8>, num_components: usize) {
    let len = 2 + 1 + num_components * 2 + 3;

    output.push(0xFF);
    output.push(MARKER_SOS);
    output.push((len >> 8) as u8);
    output.push((len & 0xFF) as u8);
    output.push(num_components as u8);

    for i in 0..num_components {
        let component_id = (i + 1) as u8;
        let table_sel = if i == 0 { 0x00 } else { 0x11 };
        output.push(component_id);
        output.push(table_sel);
    }

    output.push(0x00); // Ss
    output.push(0x3F); // Se (63)
    output.push(0x00); // Ah/Al
}
