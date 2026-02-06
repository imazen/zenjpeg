//! Arithmetic scan decoding for JPEG.
//!
//! This module handles decoding of arithmetic-coded JPEG scans (SOF9/SOF10).

use crate::entropy::ArithmeticDecoder;
use crate::error::{Error, Result, ScanRead};
use crate::foundation::alloc::{checked_size_2d, try_alloc_dct_blocks, try_alloc_filled};
use enough::Stop;

use super::JpegParser;

/// Arithmetic scan decoding methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Decode an arithmetic-coded sequential scan.
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn decode_arithmetic_scan(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        stop: &impl Stop,
    ) -> Result<()> {
        // DNL mode not supported
        if self.height == 0 {
            return Err(Error::unsupported_feature(
                "DNL mode (height=0 in SOF) not yet supported for arithmetic scan decoding",
            ));
        }

        // Calculate MCU structure
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Initialize coefficient storage
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                let comp_blocks_h = checked_size_2d(mcu_cols, h_samp)?;
                let comp_blocks_v = checked_size_2d(mcu_rows, v_samp)?;
                let num_blocks = checked_size_2d(comp_blocks_h, comp_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
                self.coeff_counts.push(try_alloc_filled(
                    num_blocks,
                    64u8,
                    "allocating coefficient counts",
                )?);
                self.nonzero_bitmaps.push(try_alloc_filled(
                    num_blocks,
                    0u64,
                    "allocating nonzero bitmaps",
                )?);
            }
        }

        // Set up arithmetic decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = ArithmeticDecoder::new(scan_data);

        // Configure decoder with conditioning parameters and restart interval
        decoder.set_restart_interval(self.restart_interval);
        for tbl in 0..4 {
            let (l, u) = self.arith_dc_cond[tbl];
            decoder.set_dc_conditioning(tbl, l, u);
            decoder.set_ac_conditioning(tbl, self.arith_ac_kx[tbl]);
        }

        // Reset decoder for this scan
        decoder.reset_for_scan();

        // Decode MCUs
        for mcu_y in 0..mcu_rows {
            // Check for cancellation at each MCU row
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            for mcu_x in 0..mcu_cols {
                // For each component in the scan
                for (comp_idx, dc_table, ac_table) in scan_components {
                    let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                    let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                    let comp_blocks_h = mcu_cols * h_samp;

                    // Decode all blocks for this component in this MCU
                    for v in 0..v_samp {
                        for h in 0..h_samp {
                            let block_x = mcu_x * h_samp + h;
                            let block_y = mcu_y * v_samp + v;
                            let block_idx = block_y * comp_blocks_h + block_x;

                            // Decode block
                            match decoder.decode_block(
                                &mut self.coeffs[*comp_idx][block_idx],
                                *comp_idx,
                                *dc_table as usize,
                                *ac_table as usize,
                            )? {
                                ScanRead::Value(()) => {
                                    // Count non-zero coefficients (approximation for tiered IDCT)
                                    let block = &self.coeffs[*comp_idx][block_idx];
                                    let mut count = 0u8;
                                    for (i, &coef) in block.iter().enumerate() {
                                        if coef != 0 {
                                            count = (i + 1) as u8;
                                        }
                                    }
                                    self.coeff_counts[*comp_idx][block_idx] = count.max(1);
                                }
                                ScanRead::EndOfScan | ScanRead::Truncated => {
                                    // Handle truncation - zero remaining blocks
                                    self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                    self.coeff_counts[*comp_idx][block_idx] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.position += decoder.position();
        Ok(())
    }

    /// Decode an arithmetic-coded progressive scan.
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn decode_arithmetic_progressive_scan(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        stop: &impl Stop,
    ) -> Result<()> {
        // DNL mode not supported
        if self.height == 0 {
            return Err(Error::unsupported_feature(
                "DNL mode (height=0 in SOF) not supported for progressive scan",
            ));
        }

        // Calculate MCU structure
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Initialize coefficient storage if needed
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                let comp_blocks_h = checked_size_2d(mcu_cols, h_samp)?;
                let comp_blocks_v = checked_size_2d(mcu_rows, v_samp)?;
                let num_blocks = checked_size_2d(comp_blocks_h, comp_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
                self.coeff_counts.push(try_alloc_filled(
                    num_blocks,
                    64u8,
                    "allocating coefficient counts",
                )?);
                self.nonzero_bitmaps.push(try_alloc_filled(
                    num_blocks,
                    0u64,
                    "allocating nonzero bitmaps",
                )?);
            }
        }

        // Set up arithmetic decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = ArithmeticDecoder::new(scan_data);

        // Configure decoder
        decoder.set_restart_interval(self.restart_interval);
        for tbl in 0..4 {
            let (l, u) = self.arith_dc_cond[tbl];
            decoder.set_dc_conditioning(tbl, l, u);
            decoder.set_ac_conditioning(tbl, self.arith_ac_kx[tbl]);
        }

        // Reset for scan (clears stats appropriate for this scan type)
        decoder.reset_for_scan();

        let is_dc_scan = ss == 0;
        let is_first_scan = ah == 0;

        if is_dc_scan {
            // DC scan
            if is_first_scan {
                self.decode_arith_dc_first(
                    &mut decoder,
                    scan_components,
                    mcu_cols,
                    mcu_rows,
                    al,
                    stop,
                )?;
            } else {
                self.decode_arith_dc_refine(
                    &mut decoder,
                    scan_components,
                    mcu_cols,
                    mcu_rows,
                    al,
                    stop,
                )?;
            }
        } else {
            // AC scan (single component only)
            if scan_components.len() != 1 {
                return Err(Error::invalid_jpeg_data(
                    "progressive AC scan must have exactly one component",
                ));
            }
            let (comp_idx, _dc_tbl, ac_tbl) = scan_components[0];

            if is_first_scan {
                self.decode_arith_ac_first(
                    &mut decoder,
                    comp_idx,
                    ac_tbl as usize,
                    mcu_cols,
                    mcu_rows,
                    ss,
                    se,
                    al,
                    stop,
                )?;
            } else {
                self.decode_arith_ac_refine(
                    &mut decoder,
                    comp_idx,
                    ac_tbl as usize,
                    mcu_cols,
                    mcu_rows,
                    ss,
                    se,
                    al,
                    stop,
                )?;
            }
        }

        self.position += decoder.position();
        Ok(())
    }

    /// Decode DC first scan (progressive arithmetic).
    fn decode_arith_dc_first(
        &mut self,
        decoder: &mut ArithmeticDecoder,
        scan_components: &[(usize, u8, u8)],
        mcu_cols: usize,
        mcu_rows: usize,
        al: u8,
        stop: &impl Stop,
    ) -> Result<()> {
        for mcu_y in 0..mcu_rows {
            // Check for cancellation at each MCU row
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            for mcu_x in 0..mcu_cols {
                for (comp_idx, dc_tbl, _ac_tbl) in scan_components {
                    let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                    let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                    let comp_blocks_h = mcu_cols * h_samp;

                    for v in 0..v_samp {
                        for h in 0..h_samp {
                            let block_x = mcu_x * h_samp + h;
                            let block_y = mcu_y * v_samp + v;
                            let block_idx = block_y * comp_blocks_h + block_x;

                            match decoder.decode_dc_first(*comp_idx, *dc_tbl as usize, al)? {
                                ScanRead::Value(dc) => {
                                    self.coeffs[*comp_idx][block_idx][0] = dc;
                                }
                                ScanRead::EndOfScan | ScanRead::Truncated => {
                                    // Truncated - leave as zero
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Decode DC refinement scan (progressive arithmetic).
    fn decode_arith_dc_refine(
        &mut self,
        decoder: &mut ArithmeticDecoder,
        scan_components: &[(usize, u8, u8)],
        mcu_cols: usize,
        mcu_rows: usize,
        al: u8,
        stop: &impl Stop,
    ) -> Result<()> {
        for mcu_y in 0..mcu_rows {
            // Check for cancellation at each MCU row
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            for mcu_x in 0..mcu_cols {
                for (comp_idx, _dc_tbl, _ac_tbl) in scan_components {
                    let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                    let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                    let comp_blocks_h = mcu_cols * h_samp;

                    for v in 0..v_samp {
                        for h in 0..h_samp {
                            let block_x = mcu_x * h_samp + h;
                            let block_y = mcu_y * v_samp + v;
                            let block_idx = block_y * comp_blocks_h + block_x;

                            decoder.decode_dc_refine(&mut self.coeffs[*comp_idx][block_idx], al)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Decode AC first scan (progressive arithmetic).
    #[allow(clippy::too_many_arguments)]
    fn decode_arith_ac_first(
        &mut self,
        decoder: &mut ArithmeticDecoder,
        comp_idx: usize,
        ac_tbl: usize,
        mcu_cols: usize,
        mcu_rows: usize,
        ss: u8,
        se: u8,
        al: u8,
        stop: &impl Stop,
    ) -> Result<()> {
        let h_samp = self.components[comp_idx].h_samp_factor as usize;
        let v_samp = self.components[comp_idx].v_samp_factor as usize;
        let comp_blocks_h = mcu_cols * h_samp;
        let comp_blocks_v = mcu_rows * v_samp;

        // AC progressive scans process blocks in raster order (not MCU order)
        for block_y in 0..comp_blocks_v {
            // Check for cancellation at each block row
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            for block_x in 0..comp_blocks_h {
                let block_idx = block_y * comp_blocks_h + block_x;
                decoder.decode_ac_first(
                    &mut self.coeffs[comp_idx][block_idx],
                    &mut self.nonzero_bitmaps[comp_idx][block_idx],
                    ac_tbl,
                    ss,
                    se,
                    al,
                )?;
            }
        }
        Ok(())
    }

    /// Decode AC refinement scan (progressive arithmetic).
    #[allow(clippy::too_many_arguments)]
    fn decode_arith_ac_refine(
        &mut self,
        decoder: &mut ArithmeticDecoder,
        comp_idx: usize,
        ac_tbl: usize,
        mcu_cols: usize,
        mcu_rows: usize,
        ss: u8,
        se: u8,
        al: u8,
        stop: &impl Stop,
    ) -> Result<()> {
        let h_samp = self.components[comp_idx].h_samp_factor as usize;
        let v_samp = self.components[comp_idx].v_samp_factor as usize;
        let comp_blocks_h = mcu_cols * h_samp;
        let comp_blocks_v = mcu_rows * v_samp;

        for block_y in 0..comp_blocks_v {
            // Check for cancellation at each block row
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            for block_x in 0..comp_blocks_h {
                let block_idx = block_y * comp_blocks_h + block_x;
                decoder.decode_ac_refine(
                    &mut self.coeffs[comp_idx][block_idx],
                    &mut self.nonzero_bitmaps[comp_idx][block_idx],
                    ac_tbl,
                    ss,
                    se,
                    al,
                )?;
            }
        }
        Ok(())
    }
}
