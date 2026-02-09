//! DCT-coefficient transform for decode-time orientation correction.
//!
//! Mutates the parser's internal coefficient storage to apply a lossless
//! transform before IDCT, enabling zero-quality-loss orientation correction
//! during decode.

use crate::lossless::{remap_block, BlockTransform, LosslessTransform};

use super::JpegParser;

impl<'a> JpegParser<'a> {
    /// Apply a lossless transform to the parser's decoded coefficients in-place.
    ///
    /// This rearranges blocks on the grid and transforms coefficients within
    /// each block, then updates the parser's width/height and sampling factors
    /// to reflect the transformed image.
    ///
    /// Must be called after `decode()` and before `to_pixels()`.
    pub(in crate::decode) fn apply_dct_transform(&mut self, transform: LosslessTransform) {
        if transform == LosslessTransform::None {
            return;
        }

        let block_transform = BlockTransform::for_transform(transform);
        let swaps = transform.swaps_dimensions();

        let num_comps = self.num_components as usize;

        // Calculate MCU dimensions for block grid sizes
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..num_comps {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }
        let mcu_width = max_h_samp as usize * 8;
        let mcu_height = max_v_samp as usize * 8;
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Transform each component's coefficient data
        for comp_idx in 0..num_comps {
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let src_bw = mcu_cols * h_samp;
            let src_bh = mcu_rows * v_samp;

            let (dst_bw, dst_bh) = if swaps {
                (src_bh, src_bw)
            } else {
                (src_bw, src_bh)
            };

            let total_dst_blocks = dst_bw * dst_bh;

            // Transform coefficients into new buffer
            let mut dst_coeffs = vec![[0i16; 64]; total_dst_blocks];
            let mut dst_counts = vec![0u8; total_dst_blocks];
            let has_bitmaps = !self.nonzero_bitmaps.is_empty()
                && comp_idx < self.nonzero_bitmaps.len()
                && !self.nonzero_bitmaps[comp_idx].is_empty();
            let mut dst_bitmaps = if has_bitmaps {
                vec![0u64; total_dst_blocks]
            } else {
                Vec::new()
            };

            for src_by in 0..src_bh {
                for src_bx in 0..src_bw {
                    let src_idx = src_by * src_bw + src_bx;
                    let (dst_bx, dst_by) = remap_block(src_bx, src_by, src_bw, src_bh, transform);
                    let dst_idx = dst_by * dst_bw + dst_bx;

                    // Transform coefficients within the block
                    dst_coeffs[dst_idx] = block_transform.apply(&self.coeffs[comp_idx][src_idx]);

                    // Copy coefficient count (block complexity doesn't change)
                    dst_counts[dst_idx] = self.coeff_counts[comp_idx][src_idx];

                    // Remap nonzero bitmap if present
                    if has_bitmaps {
                        // The bitmap tracks which zigzag positions are nonzero.
                        // After block transform, positions move, so we need to remap.
                        let src_bitmap = self.nonzero_bitmaps[comp_idx][src_idx];
                        let mut new_bitmap = 0u64;
                        let mut remaining = src_bitmap;
                        while remaining != 0 {
                            let src_z = remaining.trailing_zeros() as usize;
                            remaining &= remaining - 1; // clear lowest set bit
                            let (dst_z, _negate) = block_transform.entries[src_z];
                            new_bitmap |= 1u64 << dst_z;
                        }
                        dst_bitmaps[dst_idx] = new_bitmap;
                    }
                }
            }

            // Replace the component's data
            self.coeffs[comp_idx] = dst_coeffs;
            self.coeff_counts[comp_idx] = dst_counts;
            if has_bitmaps {
                self.nonzero_bitmaps[comp_idx] = dst_bitmaps;
            }
        }

        // Update dimensions
        if swaps {
            core::mem::swap(&mut self.width, &mut self.height);
            // Swap sampling factors for each component
            for i in 0..num_comps {
                let h = self.components[i].h_samp_factor;
                let v = self.components[i].v_samp_factor;
                self.components[i].h_samp_factor = v;
                self.components[i].v_samp_factor = h;
            }
        }
    }
}
