//! Symbol stream: captures JPEG entropy symbols for deferred Huffman encoding.
//!
//! Instead of encoding directly to bits (which requires Huffman tables upfront),
//! this captures the table-independent symbol decisions. The symbol stream can
//! then be encoded with any Huffman table set — enabling single-pass quantization
//! with deferred optimal table construction.
//!
//! Each symbol is 4 bytes: compact enough to fit the entire 4K image (~3MB) in L3 cache.

use crate::foundation::consts::DCT_BLOCK_SIZE;

/// A single entropy coding symbol (table-independent).
///
/// Layout: `[symbol_byte, extra_bits_hi, extra_bits_lo, extra_len | flags]`
/// - `symbol_byte`: the Huffman symbol (DC category or AC run/size)
/// - `extra_bits`: the additional magnitude bits (up to 15 bits)
/// - `extra_len`: number of extra bits (0-15), with flags in upper bits
#[derive(Clone, Copy)]
#[repr(C)]
pub struct JpegSymbol {
    /// The Huffman symbol to look up in the table.
    /// DC: category (0-15). AC: (run << 4) | category, or 0x00 (EOB), or 0xF0 (ZRL).
    pub symbol: u8,
    /// Number of extra bits (magnitude bits after the Huffman code). 0-15.
    pub extra_len: u8,
    /// The extra bits value (right-aligned).
    pub extra_bits: u16,
}

/// Flags for symbol classification (packed into a separate array for cache efficiency).
pub const FLAG_DC_LUMA: u8 = 0;
pub const FLAG_DC_CHROMA: u8 = 1;
pub const FLAG_AC_LUMA: u8 = 2;
pub const FLAG_AC_CHROMA: u8 = 3;
pub const FLAG_RESTART: u8 = 0x80;

/// A stream of JPEG symbols for one segment.
pub struct SymbolStream {
    /// The symbols in encoding order.
    pub symbols: Vec<JpegSymbol>,
    /// Table class for each symbol (DC_LUMA, AC_LUMA, DC_CHROMA, AC_CHROMA, RESTART).
    pub flags: Vec<u8>,
}

impl SymbolStream {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(cap),
            flags: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn push_dc(&mut self, symbol: u8, extra_bits: u16, extra_len: u8, is_chroma: bool) {
        self.symbols.push(JpegSymbol { symbol, extra_len, extra_bits });
        self.flags.push(if is_chroma { FLAG_DC_CHROMA } else { FLAG_DC_LUMA });
    }

    #[inline]
    pub fn push_ac(&mut self, symbol: u8, extra_bits: u16, extra_len: u8, is_chroma: bool) {
        self.symbols.push(JpegSymbol { symbol, extra_len, extra_bits });
        self.flags.push(if is_chroma { FLAG_AC_CHROMA } else { FLAG_AC_LUMA });
    }

    #[inline]
    pub fn push_restart(&mut self, restart_num: u8) {
        self.symbols.push(JpegSymbol { symbol: restart_num, extra_len: 0, extra_bits: 0 });
        self.flags.push(FLAG_RESTART);
    }

    /// Encode this symbol stream into a bitstream using the given Huffman tables.
    pub fn encode_with_tables(
        &self,
        dc_luma: &crate::huffman::HuffmanEncodeTable,
        ac_luma: &crate::huffman::HuffmanEncodeTable,
        dc_chroma: &crate::huffman::HuffmanEncodeTable,
        ac_chroma: &crate::huffman::HuffmanEncodeTable,
    ) -> Vec<u8> {
        let mut writer = crate::foundation::bitstream::BitWriter::with_capacity(self.symbols.len() * 2);

        for (sym, &flag) in self.symbols.iter().zip(self.flags.iter()) {
            if flag == FLAG_RESTART {
                let _ = writer.flush_restart_marker(sym.symbol);
                continue;
            }

            let table = match flag {
                FLAG_DC_LUMA => dc_luma,
                FLAG_DC_CHROMA => dc_chroma,
                FLAG_AC_LUMA => ac_luma,
                FLAG_AC_CHROMA => ac_chroma,
                _ => unreachable!(),
            };

            let (code, len) = table.encode(sym.symbol);
            writer.write_bits(code, len);

            if sym.extra_len > 0 {
                writer.write_bits(sym.extra_bits as u32, sym.extra_len);
            }
        }

        writer.into_bytes()
    }

    /// Collect Huffman symbol frequencies from this stream.
    pub fn collect_frequencies(
        &self,
        dc_luma_freq: &mut crate::huffman::optimize::FrequencyCounter,
        ac_luma_freq: &mut crate::huffman::optimize::FrequencyCounter,
        dc_chroma_freq: &mut crate::huffman::optimize::FrequencyCounter,
        ac_chroma_freq: &mut crate::huffman::optimize::FrequencyCounter,
    ) {
        for (sym, &flag) in self.symbols.iter().zip(self.flags.iter()) {
            if flag == FLAG_RESTART { continue; }
            match flag {
                FLAG_DC_LUMA => dc_luma_freq.count(sym.symbol),
                FLAG_DC_CHROMA => dc_chroma_freq.count(sym.symbol),
                FLAG_AC_LUMA => ac_luma_freq.count(sym.symbol),
                FLAG_AC_CHROMA => ac_chroma_freq.count(sym.symbol),
                _ => {}
            }
        }
    }
}

/// Convert a quantized block to symbols (table-independent).
///
/// This does the same work as `EntropyEncoder::encode_block_scalar` but outputs
/// symbols instead of bits. The symbols can later be encoded with any Huffman table.
#[inline]
pub fn block_to_symbols(
    stream: &mut SymbolStream,
    coeffs: &[i16; DCT_BLOCK_SIZE],
    prev_dc: &mut i16,
    is_chroma: bool,
) {
    let dc = coeffs[0];
    let dc_diff = dc - *prev_dc;
    *prev_dc = dc;

    let dc_cat = crate::entropy::category(dc_diff);
    if dc_cat > 0 {
        let extra = crate::entropy::additional_bits_with_cat(dc_diff, dc_cat);
        stream.push_dc(dc_cat, extra as u16, dc_cat, is_chroma);
    } else {
        stream.push_dc(0, 0, 0, is_chroma);
    }

    // AC coefficients
    let mut run = 0u8;
    for i in 1..DCT_BLOCK_SIZE {
        let ac = coeffs[i];
        if ac == 0 {
            run += 1;
        } else {
            while run >= 16 {
                stream.push_ac(0xF0, 0, 0, is_chroma); // ZRL
                run -= 16;
            }
            let ac_cat = crate::entropy::category(ac);
            let symbol = (run << 4) | ac_cat;
            let extra = crate::entropy::additional_bits_with_cat(ac, ac_cat);
            stream.push_ac(symbol, extra as u16, ac_cat, is_chroma);
            run = 0;
        }
    }

    if run > 0 {
        stream.push_ac(0x00, 0, 0, is_chroma); // EOB
    }
}
