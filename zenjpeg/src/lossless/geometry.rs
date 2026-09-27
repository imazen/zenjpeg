//! Shared MCU/block-grid geometry for the lossless transform pipeline.
//!
//! Every stage of the lossless pipeline (coefficient transform, sequential
//! emitter, progressive emitter) MUST derive block-grid geometry from this one
//! module. The #194/#195 corruption class came from each stage recomputing
//! geometry independently: the emitters rederived grid dimensions from
//! `width`/`height` while the decoder produced MCU-padded grids, and frequency
//! counting walked blocks in a different order than entropy encoding. This
//! module makes the geometry a validated, shared fact instead of a per-call
//! recomputation.
//!
//! Conventions (matching `decode_coefficients` output):
//! - Each component's stored grid is padded to whole MCUs:
//!   `padded_bw = mcus_wide * h_samp`, `padded_bh = mcus_high * v_samp`.
//! - The "true" grid is what a non-interleaved scan emits per T.81 A.2.2:
//!   `true_bw = ceil(ceil(width * h_samp / max_h) / 8)` (same for height).
//! - An interleaved scan emits the full padded grid in MCU order.

use alloc::vec::Vec;

use crate::decode::ComponentCoefficients;
use crate::error::{Error, Result};

/// Validated per-component grid geometry.
#[derive(Debug, Clone, Copy)]
pub(super) struct ComponentGeom {
    /// Horizontal sampling factor.
    pub h_samp: usize,
    /// Vertical sampling factor.
    pub v_samp: usize,
    /// Stored grid width in blocks (MCU-padded).
    pub padded_bw: usize,
    /// Stored grid height in blocks (MCU-padded).
    pub padded_bh: usize,
    /// Data units per row in a non-interleaved scan (T.81 A.2.2).
    pub true_bw: usize,
    /// Data-unit rows in a non-interleaved scan.
    pub true_bh: usize,
}

/// Validated whole-image MCU geometry for a component set.
#[derive(Debug, Clone)]
pub(super) struct McuGeom {
    /// MCU columns.
    pub mcus_wide: usize,
    /// MCU rows.
    pub mcus_high: usize,
    /// Per-component geometry, same order as the component slice.
    pub comps: Vec<ComponentGeom>,
}

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

impl McuGeom {
    /// Compute and validate geometry for `components` at `width`×`height`.
    ///
    /// Returns an error if any component's stored grid does not match the
    /// MCU-padded dimensions implied by the image size and sampling factors.
    /// A mismatch means upstream produced a grid the emitters cannot encode
    /// faithfully — failing loudly here prevents silently corrupt output.
    pub fn from_components(
        width: u32,
        height: u32,
        components: &[ComponentCoefficients],
    ) -> Result<Self> {
        if components.is_empty() {
            return Err(Error::internal("lossless geometry: no components"));
        }
        if width == 0 || height == 0 {
            return Err(Error::internal("lossless geometry: zero dimension"));
        }
        let max_h = components.iter().map(|c| c.h_samp as usize).max().unwrap();
        let max_v = components.iter().map(|c| c.v_samp as usize).max().unwrap();
        if max_h == 0 || max_v == 0 {
            return Err(Error::internal("lossless geometry: zero sampling factor"));
        }

        let mcus_wide = div_ceil(width as usize, 8 * max_h);
        let mcus_high = div_ceil(height as usize, 8 * max_v);

        let mut comps = Vec::with_capacity(components.len());
        for c in components {
            let h_samp = c.h_samp as usize;
            let v_samp = c.v_samp as usize;
            if h_samp == 0 || v_samp == 0 || h_samp > max_h || v_samp > max_v {
                return Err(Error::internal("lossless geometry: bad sampling factor"));
            }
            let padded_bw = mcus_wide * h_samp;
            let padded_bh = mcus_high * v_samp;
            // T.81 A.1.1: component sample dimensions round up.
            let comp_w = div_ceil(width as usize * h_samp, max_h);
            let comp_h = div_ceil(height as usize * v_samp, max_v);
            let true_bw = div_ceil(comp_w, 8);
            let true_bh = div_ceil(comp_h, 8);

            if c.blocks_wide != padded_bw || c.blocks_high != padded_bh {
                return Err(Error::decode_error(alloc::format!(
                    "lossless geometry: component id {} grid {}x{} blocks does not match \
                     MCU-padded {}x{} for {}x{} px (samp {}x{}, MCU grid {}x{})",
                    c.id,
                    c.blocks_wide,
                    c.blocks_high,
                    padded_bw,
                    padded_bh,
                    width,
                    height,
                    h_samp,
                    v_samp,
                    mcus_wide,
                    mcus_high,
                )));
            }
            if c.coeffs.len() != padded_bw * padded_bh * 64 {
                return Err(Error::internal(
                    "lossless geometry: coefficient buffer length mismatch",
                ));
            }
            comps.push(ComponentGeom {
                h_samp,
                v_samp,
                padded_bw,
                padded_bh,
                true_bw,
                true_bh,
            });
        }

        Ok(Self {
            mcus_wide,
            mcus_high,
            comps,
        })
    }

    /// Total number of MCUs.
    pub fn total_mcus(&self) -> usize {
        self.mcus_wide * self.mcus_high
    }
}

/// One step of an interleaved-scan traversal.
#[derive(Debug, Clone, Copy)]
pub(super) enum ScanEvent {
    /// A data unit: `idx = by * padded_bw + bx` in `comp`'s stored grid.
    Block { comp: usize, idx: usize },
    /// An MCU just finished (`mcu_idx` counts from 0).
    McuEnd { mcu_idx: usize },
}

/// Walk every data unit of an interleaved scan in T.81 MCU order.
///
/// This is the ONE definition of the sequential scan order. Both frequency
/// counting and entropy encoding must go through it so the optimized Huffman
/// tables always cover exactly the symbols the encoder emits — a divergence
/// produces zero-length codes and a silently corrupt stream (issue #194).
#[inline]
pub(super) fn for_each_interleaved_event(geom: &McuGeom, mut f: impl FnMut(ScanEvent)) {
    let mut mcu_idx = 0usize;
    for mcu_y in 0..geom.mcus_high {
        for mcu_x in 0..geom.mcus_wide {
            for (comp_idx, cg) in geom.comps.iter().enumerate() {
                for dy in 0..cg.v_samp {
                    for dx in 0..cg.h_samp {
                        let bx = mcu_x * cg.h_samp + dx;
                        let by = mcu_y * cg.v_samp + dy;
                        debug_assert!(bx < cg.padded_bw && by < cg.padded_bh);
                        f(ScanEvent::Block {
                            comp: comp_idx,
                            idx: by * cg.padded_bw + bx,
                        });
                    }
                }
            }
            f(ScanEvent::McuEnd { mcu_idx });
            mcu_idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn comp(id: u8, bw: usize, bh: usize, h: u8, v: u8) -> ComponentCoefficients {
        ComponentCoefficients {
            id,
            coeffs: vec![0i16; bw * bh * 64],
            blocks_wide: bw,
            blocks_high: bh,
            h_samp: h,
            v_samp: v,
            quant_table_idx: 0,
        }
    }

    #[test]
    fn geometry_420_nonaligned() {
        // 2000x1333 4:2:0 — luma padded to 250x168, chroma 125x84.
        let comps = vec![
            comp(1, 250, 168, 2, 2),
            comp(2, 125, 84, 1, 1),
            comp(3, 125, 84, 1, 1),
        ];
        let g = McuGeom::from_components(2000, 1333, &comps).unwrap();
        assert_eq!((g.mcus_wide, g.mcus_high), (125, 84));
        assert_eq!((g.comps[0].true_bw, g.comps[0].true_bh), (250, 167));
        assert_eq!((g.comps[1].true_bw, g.comps[1].true_bh), (125, 84));
    }

    #[test]
    fn geometry_rejects_mismatched_grid() {
        // Luma grid claims the unpadded height (167) — must be rejected.
        let comps = vec![
            comp(1, 250, 167, 2, 2),
            comp(2, 125, 84, 1, 1),
            comp(3, 125, 84, 1, 1),
        ];
        assert!(McuGeom::from_components(2000, 1333, &comps).is_err());
    }

    #[test]
    fn traversal_order_and_count() {
        // 66x50 4:2:0: luma 10x8 padded, chroma 5x4; 20 MCUs of 6 blocks.
        let comps = vec![
            comp(1, 10, 8, 2, 2),
            comp(2, 5, 4, 1, 1),
            comp(3, 5, 4, 1, 1),
        ];
        let g = McuGeom::from_components(66, 50, &comps).unwrap();
        let mut count = 0usize;
        let mut mcus = 0usize;
        let mut first_mcu: Vec<(usize, usize)> = Vec::new();
        for_each_interleaved_event(&g, |ev| match ev {
            ScanEvent::Block { comp, idx } => {
                if mcus == 0 {
                    first_mcu.push((comp, idx));
                }
                count += 1;
            }
            ScanEvent::McuEnd { .. } => mcus += 1,
        });
        assert_eq!(mcus, g.total_mcus());
        assert_eq!(count, 20 * 6);
        // First MCU: luma (0,0),(1,0),(0,1),(1,1) then Cb(0,0), Cr(0,0).
        assert_eq!(
            first_mcu,
            vec![(0, 0), (0, 1), (0, 10), (0, 11), (1, 0), (2, 0)]
        );
    }
}
