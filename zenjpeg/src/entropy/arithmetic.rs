//! Arithmetic entropy decoder for JPEG.
//!
//! Implements the QM-coder from ITU-T T.81 Appendix D for decoding
//! arithmetic-coded JPEGs (SOF9/SOF10).
//!
//! This is a pure Rust port of libjpeg-turbo's jdarith.c implementation.

#![allow(dead_code)]

use crate::error::{Error, Result, ScanRead, ScanResult};

/// Number of DC statistics bins per table.
pub const DC_STAT_BINS: usize = 64;

/// Number of AC statistics bins per table.
pub const AC_STAT_BINS: usize = 256;

/// Maximum number of arithmetic coding tables.
pub const NUM_ARITH_TBLS: usize = 4;

/// QE probability table from ITU-T T.81 Table D.2.
///
/// Each entry encodes:
/// - Bits 0-15: Qe value (probability estimate)
/// - Bits 16-22: Next_Index_LPS (next state after LPS)
/// - Bit 23: Switch_MPS (whether to switch MPS after LPS)
/// - Bits 24-30: Next_Index_MPS (next state after MPS)
const ARITAB: [u32; 114] = [
    0x0181_5a1d,
    0x020e_2586,
    0x0310_1114,
    0x0412_080b,
    0x0514_03d8,
    0x0617_01da,
    0x0719_00e5,
    0x081c_006f,
    0x091e_0036,
    0x0a21_001a,
    0x0b23_000d,
    0x0c09_0006,
    0x0d0a_0003,
    0x0d0c_0001,
    0x0f8f_5a7f,
    0x1024_3f25,
    0x1126_2cf2,
    0x1227_207c,
    0x1328_17b9,
    0x142a_1182,
    0x152b_0cef,
    0x162d_09a1,
    0x172e_072f,
    0x1830_055c,
    0x1931_0406,
    0x1a33_0303,
    0x1b34_0240,
    0x1c36_01b1,
    0x1d38_0144,
    0x1e39_00f5,
    0x1f3b_00b7,
    0x203c_008a,
    0x213e_0068,
    0x223f_004e,
    0x2320_003b,
    0x0921_002c,
    0x25a5_5ae1,
    0x2640_484c,
    0x2741_3a0d,
    0x2843_2ef1,
    0x2944_261f,
    0x2a45_1f33,
    0x2b46_19a8,
    0x2c48_1518,
    0x2d49_1177,
    0x2e4a_0e74,
    0x2f4b_0bfb,
    0x304d_09f8,
    0x314e_0861,
    0x324f_0706,
    0x3330_05cd,
    0x3432_04de,
    0x3532_040f,
    0x3633_0363,
    0x3734_02d4,
    0x3835_025c,
    0x3936_01f8,
    0x3a37_01a4,
    0x3b38_0160,
    0x3c39_0125,
    0x3d3a_00f6,
    0x3e3b_00cb,
    0x3f3d_00ab,
    0x203d_008f,
    0x41c1_5b12,
    0x4250_4d04,
    0x4351_412c,
    0x4452_37d8,
    0x4553_2fe8,
    0x4654_293c,
    0x4756_2379,
    0x4857_1edf,
    0x4957_1aa9,
    0x4a48_174e,
    0x4b48_1424,
    0x4c4a_119c,
    0x4d4a_0f6b,
    0x4e4b_0d51,
    0x4f4d_0bb6,
    0x304d_0a40,
    0x51d0_5832,
    0x5258_4d1c,
    0x5359_438e,
    0x545a_3bdd,
    0x555b_34ee,
    0x565c_2eae,
    0x575d_299a,
    0x4756_2516,
    0x59d8_5570,
    0x5a5f_4ca9,
    0x5b60_44d9,
    0x5c61_3e22,
    0x5d63_3824,
    0x5e63_32b4,
    0x565d_2e17,
    0x60df_56a8,
    0x6165_4f46,
    0x6266_47e5,
    0x6367_41cf,
    0x6468_3c3d,
    0x5d63_375e,
    0x6669_5231,
    0x676a_4c0f,
    0x686b_4639,
    0x6367_415e,
    0x6ae9_5627,
    0x6b6c_50e7,
    0x676d_4b85,
    0x6d6e_5597,
    0x6b6f_504f,
    0x6fee_5a10,
    0x6d70_5522,
    0x6ff0_59eb,
    0x7171_5a1d,
];

/// Core arithmetic decoding state (separated from stats tables for borrow checker).
struct ArithState<'data> {
    data: &'data [u8],
    position: usize,
    c: i32,
    a: i32,
    ct: i32,
    unread_marker: Option<u8>,
}

impl<'data> ArithState<'data> {
    fn new(data: &'data [u8]) -> Self {
        Self {
            data,
            position: 0,
            c: 0,
            a: 0,
            ct: -16,
            unread_marker: None,
        }
    }

    fn reset(&mut self) {
        self.c = 0;
        self.a = 0;
        self.ct = -16;
        // Note: unread_marker is NOT cleared - it persists across resets
    }

    #[inline]
    fn get_byte(&mut self) -> i32 {
        if self.position >= self.data.len() {
            return 0;
        }

        let byte = self.data[self.position];
        self.position += 1;

        if byte != 0xFF {
            return byte as i32;
        }

        // Handle 0xFF - could be stuffed byte or marker
        loop {
            if self.position >= self.data.len() {
                return 0xFF;
            }
            let next = self.data[self.position];
            self.position += 1;

            if next == 0xFF {
                // Skip padding 0xFF bytes
                continue;
            }
            if next == 0x00 {
                // Stuffed zero byte - return the 0xFF
                return 0xFF;
            }
            // Found a marker - store it and return 0
            self.unread_marker = Some(next);
            return 0;
        }
    }

    /// Core arithmetic decode - returns 0 or 1 based on context state.
    /// This is a direct port of libjpeg-turbo's arith_decode().
    #[inline]
    fn decode(&mut self, st: &mut u8) -> u8 {
        // Renormalization & data input per section D.2.6
        while self.a < 0x8000 {
            // Decrement ct FIRST, then check if need more data
            self.ct -= 1;
            if self.ct < 0 {
                // Need to fetch next data byte
                let data = if self.unread_marker.is_some() {
                    0 // Stuff zero data after marker
                } else {
                    self.get_byte()
                };
                self.c = (self.c << 8) | data;
                self.ct += 8;
                if self.ct < 0 {
                    // Need more initial bytes
                    self.ct += 1;
                    if self.ct == 0 {
                        // Got 2 initial bytes -> re-init A and exit loop
                        self.a = 0x8000; // => a = 0x10000 after loop exit due to a <<= 1
                    }
                }
            }
            self.a <<= 1;
        }

        // Fetch values from ARITAB
        let sv = *st;
        let entry = ARITAB[(sv & 0x7F) as usize];
        let qe = (entry & 0xFFFF) as i32;
        let nl = ((entry >> 16) & 0x7F) as u8;
        let nm = ((entry >> 24) & 0x7F) as u8;
        let switch_mps = (entry >> 23) & 1 != 0;

        // Decode & estimation procedures per sections D.2.4 & D.2.5
        let temp = self.a - qe;
        self.a = temp;
        let temp_shifted = temp << self.ct;

        if self.c >= temp_shifted {
            self.c -= temp_shifted;
            // Conditional LPS (less probable symbol) exchange
            if self.a < qe {
                self.a = qe;
                *st = (sv & 0x80) ^ nm; // Estimate_after_MPS
            } else {
                self.a = qe;
                *st = (sv & 0x80) ^ nl; // Estimate_after_LPS
                if switch_mps {
                    *st ^= 0x80;
                }
                return (sv ^ 0x80) >> 7; // Exchange LPS/MPS
            }
        } else if self.a < 0x8000 {
            // Conditional MPS (more probable symbol) exchange
            if self.a < qe {
                *st = (sv & 0x80) ^ nl; // Estimate_after_LPS
                if switch_mps {
                    *st ^= 0x80;
                }
                return (sv ^ 0x80) >> 7; // Exchange LPS/MPS
            } else {
                *st = (sv & 0x80) ^ nm; // Estimate_after_MPS
            }
        }

        sv >> 7
    }
}

/// Arithmetic decoder for JPEG.
pub struct ArithmeticDecoder<'data> {
    state: ArithState<'data>,
    dc_stats: [[u8; DC_STAT_BINS]; NUM_ARITH_TBLS],
    ac_stats: [[u8; AC_STAT_BINS]; NUM_ARITH_TBLS],
    fixed_bin: [u8; 4],
    last_dc_val: [i16; 4],
    dc_context: [u8; 4],
    dc_cond: [(u8, u8); NUM_ARITH_TBLS],
    ac_kx: [u8; NUM_ARITH_TBLS],
    restart_interval: u16,
    restarts_to_go: u32,
}

impl<'data> ArithmeticDecoder<'data> {
    /// Creates a new arithmetic decoder.
    pub fn new(data: &'data [u8]) -> Self {
        Self {
            state: ArithState::new(data),
            dc_stats: [[0; DC_STAT_BINS]; NUM_ARITH_TBLS],
            ac_stats: [[0; AC_STAT_BINS]; NUM_ARITH_TBLS],
            fixed_bin: [113, 0, 0, 0],
            last_dc_val: [0; 4],
            dc_context: [0; 4],
            dc_cond: [(0, 1); NUM_ARITH_TBLS],
            ac_kx: [5; NUM_ARITH_TBLS],
            restart_interval: 0,
            restarts_to_go: 0,
        }
    }

    pub fn set_restart_interval(&mut self, interval: u16) {
        self.restart_interval = interval;
        self.restarts_to_go = interval as u32;
    }

    pub fn set_dc_conditioning(&mut self, tbl: usize, l: u8, u: u8) {
        if tbl < NUM_ARITH_TBLS {
            self.dc_cond[tbl] = (l, u);
        }
    }

    pub fn set_ac_conditioning(&mut self, tbl: usize, kx: u8) {
        if tbl < NUM_ARITH_TBLS {
            self.ac_kx[tbl] = kx;
        }
    }

    pub fn position(&self) -> usize {
        // If we found a marker in get_byte(), we advanced position past both
        // the 0xFF and the marker byte, but those bytes belong to the parser.
        // Subtract 2 to point back to the 0xFF.
        if self.state.unread_marker.is_some() {
            self.state.position.saturating_sub(2)
        } else {
            self.state.position
        }
    }

    pub fn reset_for_scan(&mut self) {
        for tbl in &mut self.dc_stats {
            tbl.fill(0);
        }
        for tbl in &mut self.ac_stats {
            tbl.fill(0);
        }
        self.last_dc_val = [0; 4];
        self.dc_context = [0; 4];
        self.state.reset();
        self.restarts_to_go = self.restart_interval as u32;
    }

    pub fn process_restart(&mut self) -> Result<()> {
        if let Some(marker) = self.state.unread_marker.take()
            && !(0xD0..=0xD7).contains(&marker)
        {
            return Err(Error::invalid_jpeg_data("expected restart marker"));
        }

        for tbl in &mut self.dc_stats {
            tbl.fill(0);
        }
        for tbl in &mut self.ac_stats {
            tbl.fill(0);
        }
        self.last_dc_val = [0; 4];
        self.dc_context = [0; 4];
        self.state.reset();
        self.restarts_to_go = self.restart_interval as u32;

        Ok(())
    }

    /// Decodes a DC coefficient for a component.
    /// Direct port of libjpeg-turbo's decode_mcu_DC_first.
    pub fn decode_dc(&mut self, ci: usize, tbl: usize) -> ScanResult<i16> {
        if self.state.ct == -1 {
            return Ok(ScanRead::EndOfScan);
        }

        let context = self.dc_context[ci] as usize;

        // Figure F.19: Decode_DC_DIFF
        if self.state.decode(&mut self.dc_stats[tbl][context]) == 0 {
            self.dc_context[ci] = 0;
            return Ok(ScanRead::Value(self.last_dc_val[ci]));
        }

        // Figure F.21: Decoding nonzero value v
        // Figure F.22: Decoding the sign of v
        let sign = self.state.decode(&mut self.dc_stats[tbl][context + 1]);
        let mut st = context + 2 + sign as usize;

        // Figure F.23: Decoding the magnitude category of v
        let mut m: i32 = self.state.decode(&mut self.dc_stats[tbl][st]) as i32;
        if m != 0 {
            // If m != 0, st gets reassigned to 20 for extended magnitude decoding
            st = 20; // Table F.4: X1 = 20
            while self.state.decode(&mut self.dc_stats[tbl][st]) != 0 {
                m <<= 1;
                if m == 0x8000 {
                    self.state.ct = -1;
                    return Err(Error::invalid_jpeg_data("arithmetic DC magnitude overflow"));
                }
                st += 1;
            }
        }

        // Section F.1.4.4.1.2: Establish dc_context conditioning category
        let (l, u) = self.dc_cond[tbl];
        let half_l = (1i32 << l) >> 1;
        let half_u = (1i32 << u) >> 1;

        self.dc_context[ci] = if m < half_l {
            0 // zero diff category
        } else if m > half_u {
            12 + sign * 4 // large diff category
        } else {
            4 + sign * 4 // small diff category
        };

        // Figure F.24: Decoding the magnitude bit pattern of v
        let mut v = m;
        st += 14; // Bit pattern stats are 14 positions after magnitude stats
        let mut m2 = m;
        while m2 > 1 {
            m2 >>= 1;
            if self.state.decode(&mut self.dc_stats[tbl][st]) != 0 {
                v |= m2;
            }
        }

        v += 1;
        if sign != 0 {
            v = -v;
        }

        let prev = self.last_dc_val[ci];
        let new_dc = ((prev as i32 + v) & 0xFFFF) as i16;
        self.last_dc_val[ci] = new_dc;

        Ok(ScanRead::Value(new_dc))
    }

    /// Decodes AC coefficients for a block.
    /// Direct port of libjpeg-turbo's decode_mcu_AC_first for sequential mode.
    pub fn decode_ac(&mut self, block: &mut [i16; 64], tbl: usize, se: u8) -> ScanResult<()> {
        if self.state.ct == -1 {
            return Ok(ScanRead::EndOfScan);
        }

        let kx = self.ac_kx[tbl];
        let mut k: usize = 1;

        // Figure F.20: Decode_AC_coefficients
        while k <= se as usize {
            // st tracks current statistics index, starts at 3*(k-1)
            let mut st = 3 * (k - 1);

            // EOB flag
            if self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                break;
            }

            // Run of zeros - st and k advance together
            while self.state.decode(&mut self.ac_stats[tbl][st + 1]) == 0 {
                st += 3;
                k += 1;
                if k > se as usize {
                    self.state.ct = -1;
                    return Err(Error::invalid_jpeg_data("arithmetic AC spectral overflow"));
                }
            }

            // Figure F.21: Decoding nonzero value v
            // Figure F.22: Decoding the sign of v
            let sign = self.state.decode(&mut self.fixed_bin[0]);

            // Figure F.23: Decoding the magnitude category of v
            st += 2;
            let mut m: i32 = self.state.decode(&mut self.ac_stats[tbl][st]) as i32;
            if m != 0 && self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                m <<= 1;
                st = if (k as u8) <= kx { 189 } else { 217 };
                while self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                    m <<= 1;
                    if m == 0x8000 {
                        self.state.ct = -1;
                        return Err(Error::invalid_jpeg_data("arithmetic AC magnitude overflow"));
                    }
                    st += 1;
                }
            }

            // Figure F.24: Decoding the magnitude bit pattern of v
            let mut v = m;
            st += 14;
            let mut m2 = m;
            while m2 > 1 {
                m2 >>= 1;
                if self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                    v |= m2;
                }
            }

            v += 1;
            if sign != 0 {
                v = -v;
            }

            // Store in zigzag order (position k), not natural order
            // The output path will do the zigzag-to-natural conversion
            block[k] = v as i16;
            k += 1;
        }

        Ok(ScanRead::Value(()))
    }

    /// Decodes a full block (DC + AC) for sequential mode.
    pub fn decode_block(
        &mut self,
        block: &mut [i16; 64],
        ci: usize,
        dc_tbl: usize,
        ac_tbl: usize,
    ) -> ScanResult<()> {
        if self.restart_interval > 0 && self.restarts_to_go == 0 {
            self.process_restart()?;
        }

        let dc = match self.decode_dc(ci, dc_tbl)? {
            ScanRead::Value(v) => v,
            other => return Ok(other.map(|_| ())),
        };
        block[0] = dc;

        self.decode_ac(block, ac_tbl, 63)?;

        if self.restart_interval > 0 {
            self.restarts_to_go -= 1;
        }

        Ok(ScanRead::Value(()))
    }

    /// Decodes DC coefficient for progressive first scan.
    pub fn decode_dc_first(&mut self, ci: usize, tbl: usize, al: u8) -> ScanResult<i16> {
        let dc = match self.decode_dc(ci, tbl)? {
            ScanRead::Value(v) => v,
            other => return Ok(other),
        };
        Ok(ScanRead::Value(dc << al))
    }

    /// Decodes DC coefficient for progressive refinement scan.
    pub fn decode_dc_refine(&mut self, block: &mut [i16; 64], al: u8) -> ScanResult<()> {
        if self.state.ct == -1 {
            return Ok(ScanRead::EndOfScan);
        }

        let p1 = 1i16 << al;
        if self.state.decode(&mut self.fixed_bin[0]) != 0 {
            block[0] |= p1;
        }

        Ok(ScanRead::Value(()))
    }

    /// Decodes AC coefficients for progressive first scan.
    pub fn decode_ac_first(
        &mut self,
        block: &mut [i16; 64],
        bitmap: &mut u64,
        tbl: usize,
        ss: u8,
        se: u8,
        al: u8,
    ) -> ScanResult<()> {
        if self.state.ct == -1 {
            return Ok(ScanRead::EndOfScan);
        }

        let kx = self.ac_kx[tbl];
        let mut k = ss as usize;

        while k <= se as usize {
            let mut st = 3 * (k - 1);

            // EOB flag
            if self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                break;
            }

            // Run of zeros - st and k advance together
            while self.state.decode(&mut self.ac_stats[tbl][st + 1]) == 0 {
                st += 3;
                k += 1;
                if k > se as usize {
                    self.state.ct = -1;
                    return Err(Error::invalid_jpeg_data("arithmetic AC spectral overflow"));
                }
            }

            // Decode sign
            let sign = self.state.decode(&mut self.fixed_bin[0]);

            // Decode magnitude category
            st += 2;
            let mut m: i32 = self.state.decode(&mut self.ac_stats[tbl][st]) as i32;
            if m != 0 && self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                m <<= 1;
                st = if (k as u8) <= kx { 189 } else { 217 };
                while self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                    m <<= 1;
                    if m == 0x8000 {
                        self.state.ct = -1;
                        return Err(Error::invalid_jpeg_data("arithmetic AC magnitude overflow"));
                    }
                    st += 1;
                }
            }

            // Decode magnitude bit pattern
            let mut v = m;
            st += 14;
            let mut m2 = m;
            while m2 > 1 {
                m2 >>= 1;
                if self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                    v |= m2;
                }
            }

            v += 1;
            if sign != 0 {
                v = -v;
            }

            // Store in zigzag order (position k), not natural order
            block[k] = (v << al) as i16;
            *bitmap |= 1u64 << (k & 63);
            k += 1;
        }

        Ok(ScanRead::Value(()))
    }

    /// Decodes AC coefficients for progressive refinement scan.
    /// Direct port of libjpeg-turbo's decode_mcu_AC_refine.
    pub fn decode_ac_refine(
        &mut self,
        block: &mut [i16; 64],
        bitmap: &mut u64,
        tbl: usize,
        ss: u8,
        se: u8,
        al: u8,
    ) -> ScanResult<()> {
        if self.state.ct == -1 {
            return Ok(ScanRead::EndOfScan);
        }

        let p1 = 1i16 << al;
        let m1 = (-1i16) << al;

        // Establish EOBx (previous stage end-of-block) index
        // Coefficients are stored in zigzag order (position k), not natural order
        let mut kex = se as usize;
        while kex > 0 {
            if block[kex] != 0 {
                break;
            }
            kex -= 1;
        }

        let mut k = ss as usize;
        while k <= se as usize {
            // st is reset at start of each outer loop iteration
            let mut st = 3 * (k - 1);

            // EOB flag - only check if past previous EOB position
            if k > kex && self.state.decode(&mut self.ac_stats[tbl][st]) != 0 {
                break;
            }

            loop {
                // Use zigzag order (position k)
                let coef = block[k];

                if coef != 0 {
                    // Previously nonzero coefficient - potentially refine it
                    if self.state.decode(&mut self.ac_stats[tbl][st + 2]) != 0 {
                        if coef < 0 {
                            block[k] = coef.wrapping_add(m1);
                        } else {
                            block[k] = coef.wrapping_add(p1);
                        }
                    }
                    break;
                }

                // Zero coefficient - check if it becomes nonzero
                if self.state.decode(&mut self.ac_stats[tbl][st + 1]) != 0 {
                    // Newly nonzero coefficient
                    if self.state.decode(&mut self.fixed_bin[0]) != 0 {
                        block[k] = m1;
                    } else {
                        block[k] = p1;
                    }
                    *bitmap |= 1u64 << (k & 63);
                    break;
                }

                // Continue scanning - st and k advance together
                st += 3;
                k += 1;
                if k > se as usize {
                    self.state.ct = -1;
                    return Err(Error::invalid_jpeg_data("arithmetic AC spectral overflow"));
                }
            }

            k += 1;
        }

        Ok(ScanRead::Value(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let data = [0u8; 100];
        let decoder = ArithmeticDecoder::new(&data);
        assert_eq!(decoder.position(), 0);
    }
}
