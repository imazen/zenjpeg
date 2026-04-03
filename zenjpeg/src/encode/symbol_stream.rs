//! Table-independent JPEG entropy symbol stream.
//!
//! Captures quantization decisions (symbols + extra bits) without committing
//! to a Huffman table. The stream can then be:
//! - Scanned for symbol frequencies (for optimal table construction)
//! - Encoded with any Huffman table set (cheap linear scan)
//!
//! Used by [`super::fused_parallel_encode`] to avoid recomputing DCT/quantize
//! in the second pass of Huffman-optimized encoding.
//!
//! Memory: ~4 bytes per symbol, ~5 symbols per block.
//! A 4K 4:2:0 image ≈ 130K blocks × 5 symbols × 4 bytes ≈ 2.6MB.

use crate::foundation::consts::DCT_BLOCK_SIZE;

/// A single entropy symbol (Huffman-table-independent).
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Symbol {
    /// Huffman symbol to look up: DC category (0-15) or AC (run<<4|category).
    pub symbol: u8,
    /// Number of extra magnitude bits (0-15).
    pub extra_len: u8,
    /// Extra magnitude bits (right-aligned).
    pub extra_bits: u16,
}

/// Table class for each symbol.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum TableClass {
    DcLuma = 0,
    DcChroma = 1,
    AcLuma = 2,
    AcChroma = 3,
    /// Restart marker (symbol field = marker number 0-7).
    Restart = 0x80,
}

/// Stream of JPEG symbols for one restart segment.
pub(crate) struct SymbolStream {
    pub symbols: Vec<Symbol>,
    pub classes: Vec<TableClass>,
}

impl SymbolStream {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(cap),
            classes: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.symbols.clear();
        self.classes.clear();
    }

    #[inline(always)]
    fn push(&mut self, class: TableClass, symbol: u8, extra_bits: u16, extra_len: u8) {
        self.symbols.push(Symbol {
            symbol,
            extra_len,
            extra_bits,
        });
        self.classes.push(class);
    }

    /// Collect symbol frequencies into the provided counters.
    pub fn collect_frequencies(
        &self,
        dc_luma: &mut crate::huffman::optimize::FrequencyCounter,
        ac_luma: &mut crate::huffman::optimize::FrequencyCounter,
        dc_chroma: &mut crate::huffman::optimize::FrequencyCounter,
        ac_chroma: &mut crate::huffman::optimize::FrequencyCounter,
    ) {
        for (sym, class) in self.symbols.iter().zip(self.classes.iter()) {
            match class {
                TableClass::DcLuma => dc_luma.count(sym.symbol),
                TableClass::DcChroma => dc_chroma.count(sym.symbol),
                TableClass::AcLuma => ac_luma.count(sym.symbol),
                TableClass::AcChroma => ac_chroma.count(sym.symbol),
                TableClass::Restart => {}
            }
        }
    }

    /// Encode this stream into a JPEG bitstream using the given Huffman tables.
    pub fn encode_to_bytes(
        &self,
        dc_luma: &crate::huffman::HuffmanEncodeTable,
        ac_luma: &crate::huffman::HuffmanEncodeTable,
        dc_chroma: &crate::huffman::HuffmanEncodeTable,
        ac_chroma: &crate::huffman::HuffmanEncodeTable,
    ) -> Vec<u8> {
        let mut writer =
            crate::foundation::bitstream::BitWriter::with_capacity(self.symbols.len() * 2);

        for (sym, class) in self.symbols.iter().zip(self.classes.iter()) {
            if *class == TableClass::Restart {
                let _ = writer.flush_restart_marker(sym.symbol);
                continue;
            }

            let table = match class {
                TableClass::DcLuma => dc_luma,
                TableClass::DcChroma => dc_chroma,
                TableClass::AcLuma => ac_luma,
                TableClass::AcChroma => ac_chroma,
                TableClass::Restart => unreachable!(),
            };

            let (code, len) = table.encode(sym.symbol);
            writer.write_bits(code, len);

            if sym.extra_len > 0 {
                writer.write_bits(sym.extra_bits as u32, sym.extra_len);
            }
        }

        writer.into_bytes()
    }
}

/// Convert a quantized block (zigzag order) into symbols.
///
/// Updates `prev_dc` with this block's DC value for differential encoding.
#[inline]
pub(crate) fn block_to_symbols(
    stream: &mut SymbolStream,
    coeffs: &[i16; DCT_BLOCK_SIZE],
    prev_dc: &mut i16,
    is_chroma: bool,
) {
    let dc_class = if is_chroma {
        TableClass::DcChroma
    } else {
        TableClass::DcLuma
    };
    let ac_class = if is_chroma {
        TableClass::AcChroma
    } else {
        TableClass::AcLuma
    };

    // DC: differential encoding
    let dc_diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let dc_cat = crate::entropy::category(dc_diff);
    if dc_cat > 0 {
        let extra = crate::entropy::additional_bits_with_cat(dc_diff, dc_cat);
        stream.push(dc_class, dc_cat, extra as u16, dc_cat);
    } else {
        stream.push(dc_class, 0, 0, 0);
    }

    // AC: run-length encoding
    let mut run = 0u8;
    for &ac in &coeffs[1..] {
        if ac == 0 {
            run += 1;
        } else {
            while run >= 16 {
                stream.push(ac_class, 0xF0, 0, 0); // ZRL
                run -= 16;
            }
            let ac_cat = crate::entropy::category(ac);
            let extra = crate::entropy::additional_bits_with_cat(ac, ac_cat);
            stream.push(ac_class, (run << 4) | ac_cat, extra as u16, ac_cat);
            run = 0;
        }
    }
    if run > 0 {
        stream.push(ac_class, 0x00, 0, 0); // EOB
    }
}
