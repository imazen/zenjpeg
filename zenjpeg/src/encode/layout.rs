//! Immutable image layout parameters computed once from (subsampling, use_xyb, width, height).
//!
//! Single source of truth for all geometry — never recomputed, never diverges.
//! This eliminates the class of bugs where `v_samp`, `padded_width`, or block
//! dimensions are computed independently in multiple locations and get out of sync.

use crate::types::Subsampling;

/// Immutable image layout computed once from (subsampling, use_xyb, width, height).
/// Passed by reference everywhere — never recomputed, never diverges.
#[derive(Debug, Clone)]
pub(crate) struct LayoutParams {
    // === Inputs (stored for reference) ===
    pub width: usize,
    pub height: usize,
    pub subsampling: Subsampling,
    pub use_xyb: bool,

    // === Luma geometry ===
    pub mcu_size: usize,
    /// MCU-aligned width for block extraction
    pub padded_width: usize,
    /// Strip height in pixels (8 or 16)
    pub strip_height: usize,
    /// max_v_samp_factor: 2 for XYB, 4:2:0, or 4:4:0; 1 otherwise
    pub v_samp: usize,

    // === Luma blocks ===
    /// (width + 7) / 8
    pub blocks_w: usize,
    /// (height + 7) / 8
    pub blocks_h: usize,
    /// padded_width / 8
    pub padded_blocks_w: usize,

    // === Chroma geometry ===
    pub c_width: usize,
    pub c_strip_height: usize,
    pub padded_c_width: usize,

    // === Block counts (luma) ===
    pub y_blocks_h: usize,
    pub y_blocks_v: usize,
    pub total_y_blocks: usize,

    // === Block counts (chroma) ===
    pub c_blocks_h: usize,
    pub c_blocks_v: usize,
    pub total_c_blocks: usize,
    pub padded_c_blocks_w: usize,

    // === XYB B-channel geometry ===
    pub b_width: usize,
    pub padded_b_width: usize,
    pub b_blocks_h: usize,
    pub b_blocks_v: usize,

    // === Pending buffer capacities ===
    pub pending_y_capacity: usize,
    pub pending_c_capacity: usize,

    // === iMCU dimensions (for AQ) ===
    /// padded_width + 1 for edge replication during HF modulation
    pub y_buffer_stride: usize,
}

impl LayoutParams {
    /// Creates layout parameters from image dimensions and encoding options.
    ///
    /// This consolidates ALL geometry that was previously computed independently
    /// in `StripProcessor::with_xyb()`, `StreamingAQ::new()`, `init_aq()`,
    /// and `estimate_memory_*` functions.
    pub fn new(width: usize, height: usize, subsampling: Subsampling, use_xyb: bool) -> Self {
        // Strip height is 16 for:
        // - 4:2:0 and 4:4:0 (2 MCU rows of chroma)
        // - XYB mode (B component is always 2x2 downsampled)
        let strip_height = match (subsampling, use_xyb) {
            (Subsampling::S420 | Subsampling::S440, _) => 16,
            (_, true) => 16, // XYB always needs 16-row strips for B component
            _ => 8,
        };

        // MCU size for padding calculation
        let mcu_size = subsampling.mcu_size();

        // Calculate padded width (MCU-aligned)
        let padded_width = (width + mcu_size - 1) / mcu_size * mcu_size;

        // Chroma dimensions
        let (c_width, c_strip_height) = match subsampling {
            Subsampling::S420 => ((width + 1) / 2, strip_height / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, strip_height / 2),
            Subsampling::S444 => (width, strip_height),
        };

        // Chroma planes are padded to multiples of 8 (block size)
        let padded_c_width = (c_width + 7) / 8 * 8;

        // B channel width for XYB mode (always 2x2 downsampled)
        let b_width = (width + 1) / 2;
        let padded_b_width = if use_xyb {
            (b_width + 7) / 8 * 8
        } else {
            padded_c_width // Not XYB, use same as chroma
        };

        // Block counts (luma)
        let blocks_w = (width + 7) / 8;
        let blocks_h = (height + 7) / 8;
        let padded_blocks_w = padded_width / 8;
        let y_blocks_h = blocks_w;
        let y_blocks_v = blocks_h;
        let total_y_blocks = y_blocks_h * y_blocks_v;

        // Block counts (chroma)
        let (c_blocks_h, c_blocks_v, total_c_blocks) = match subsampling {
            Subsampling::S420 => {
                let h = (width + 15) / 16;
                let v = (height + 15) / 16;
                (h, v, h * v)
            }
            Subsampling::S422 => {
                let h = (width + 15) / 16;
                (h, y_blocks_v, h * y_blocks_v)
            }
            Subsampling::S440 => {
                let v = (height + 15) / 16;
                (y_blocks_h, v, y_blocks_h * v)
            }
            Subsampling::S444 => (y_blocks_h, y_blocks_v, total_y_blocks),
        };

        let padded_c_blocks_w = padded_c_width / 8;

        // B channel block dimensions for XYB mode (always 2x2 downsampled)
        let (b_blocks_h, b_blocks_v) = if use_xyb {
            ((width + 15) / 16, (height + 15) / 16)
        } else {
            (c_blocks_h, c_blocks_v)
        };

        // In XYB mode, JPEG header uses R:2×2, G:2×2, B:1×1, so max_v_samp_factor=2
        // regardless of the chroma subsampling enum.
        let v_samp = if use_xyb {
            2
        } else {
            match subsampling {
                Subsampling::S420 | Subsampling::S440 => 2,
                _ => 1,
            }
        };

        // Pending buffer capacities: one iMCU row of blocks
        let pending_y_capacity = padded_blocks_w * v_samp;
        let pending_c_capacity = padded_c_blocks_w;

        // Y buffer stride for AQ (padded_width + 1 for edge replication)
        let y_buffer_stride = padded_width + 1;

        Self {
            width,
            height,
            subsampling,
            use_xyb,
            mcu_size,
            padded_width,
            strip_height,
            v_samp,
            blocks_w,
            blocks_h,
            padded_blocks_w,
            c_width,
            c_strip_height,
            padded_c_width,
            y_blocks_h,
            y_blocks_v,
            total_y_blocks,
            c_blocks_h,
            c_blocks_v,
            total_c_blocks,
            padded_c_blocks_w,
            b_width,
            padded_b_width,
            b_blocks_h,
            b_blocks_v,
            pending_y_capacity,
            pending_c_capacity,
            y_buffer_stride,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s444_no_xyb() {
        let lp = LayoutParams::new(1920, 1080, Subsampling::S444, false);
        assert_eq!(lp.strip_height, 8);
        assert_eq!(lp.mcu_size, 8);
        assert_eq!(lp.padded_width, 1920); // 1920 is already 8-aligned
        assert_eq!(lp.v_samp, 1);
        assert_eq!(lp.blocks_w, 240);
        assert_eq!(lp.blocks_h, 135);
        assert_eq!(lp.c_width, 1920);
        assert_eq!(lp.c_strip_height, 8);
        assert_eq!(lp.padded_c_width, 1920);
        // 4:4:4 chroma = same as luma
        assert_eq!(lp.c_blocks_h, 240);
        assert_eq!(lp.c_blocks_v, 135);
        assert_eq!(lp.total_c_blocks, 240 * 135);
        assert_eq!(lp.pending_y_capacity, 240); // padded_blocks_w * v_samp(1)
        assert_eq!(lp.pending_c_capacity, 240);
        assert_eq!(lp.y_buffer_stride, 1921);
    }

    #[test]
    fn test_s420_no_xyb() {
        let lp = LayoutParams::new(1920, 1080, Subsampling::S420, false);
        assert_eq!(lp.strip_height, 16);
        assert_eq!(lp.mcu_size, 16);
        assert_eq!(lp.padded_width, 1920); // 1920 is 16-aligned
        assert_eq!(lp.v_samp, 2);
        assert_eq!(lp.blocks_w, 240);
        assert_eq!(lp.blocks_h, 135);
        assert_eq!(lp.c_width, 960);
        assert_eq!(lp.c_strip_height, 8);
        assert_eq!(lp.padded_c_width, 960);
        assert_eq!(lp.c_blocks_h, 120); // (1920 + 15) / 16
        assert_eq!(lp.c_blocks_v, 68); // (1080 + 15) / 16
        assert_eq!(lp.pending_y_capacity, 240 * 2); // padded_blocks_w * v_samp
        assert_eq!(lp.pending_c_capacity, 120);
    }

    #[test]
    fn test_s422_no_xyb() {
        let lp = LayoutParams::new(1920, 1080, Subsampling::S422, false);
        assert_eq!(lp.strip_height, 8);
        assert_eq!(lp.mcu_size, 16);
        assert_eq!(lp.padded_width, 1920);
        assert_eq!(lp.v_samp, 1);
        assert_eq!(lp.c_width, 960);
        assert_eq!(lp.c_strip_height, 8);
        assert_eq!(lp.c_blocks_h, 120);
        assert_eq!(lp.c_blocks_v, 135); // same as y_blocks_v for 422
    }

    #[test]
    fn test_s440_no_xyb() {
        let lp = LayoutParams::new(1920, 1080, Subsampling::S440, false);
        assert_eq!(lp.strip_height, 16);
        assert_eq!(lp.mcu_size, 16);
        assert_eq!(lp.padded_width, 1920);
        assert_eq!(lp.v_samp, 2);
        assert_eq!(lp.c_width, 1920);
        assert_eq!(lp.c_strip_height, 8);
        assert_eq!(lp.c_blocks_h, 240); // same as y_blocks_h for 440
        assert_eq!(lp.c_blocks_v, 68); // (1080 + 15) / 16
    }

    #[test]
    fn test_s444_xyb() {
        let lp = LayoutParams::new(1920, 1080, Subsampling::S444, true);
        // XYB always uses 16-row strips
        assert_eq!(lp.strip_height, 16);
        // XYB always has v_samp=2 (R:2×2, G:2×2, B:1×1)
        assert_eq!(lp.v_samp, 2);
        // B channel is always 2x2 downsampled in XYB
        assert_eq!(lp.b_width, 960);
        assert_eq!(lp.b_blocks_h, 120);
        assert_eq!(lp.b_blocks_v, 68);
        // Pending Y capacity reflects v_samp=2
        assert_eq!(lp.pending_y_capacity, 240 * 2);
    }

    #[test]
    fn test_s420_xyb() {
        let lp = LayoutParams::new(1920, 1080, Subsampling::S420, true);
        assert_eq!(lp.strip_height, 16);
        assert_eq!(lp.v_samp, 2);
        // B channel dimensions are independent of subsampling in XYB mode
        assert_eq!(lp.b_width, 960);
        assert_eq!(lp.b_blocks_h, 120);
        assert_eq!(lp.b_blocks_v, 68);
    }

    #[test]
    fn test_non_aligned_dimensions() {
        // 1118x1105 - not aligned to 8 or 16
        let lp = LayoutParams::new(1118, 1105, Subsampling::S420, false);
        assert_eq!(lp.mcu_size, 16);
        assert_eq!(lp.padded_width, 1120); // (1118 + 15) / 16 * 16
        assert_eq!(lp.blocks_w, 140); // (1118 + 7) / 8
        assert_eq!(lp.blocks_h, 139); // (1105 + 7) / 8
        assert_eq!(lp.c_width, 559); // (1118 + 1) / 2
        assert_eq!(lp.padded_c_width, 560); // (559 + 7) / 8 * 8
        assert_eq!(lp.c_blocks_h, 70); // (1118 + 15) / 16
        assert_eq!(lp.c_blocks_v, 70); // (1105 + 15) / 16
        assert_eq!(lp.padded_blocks_w, 140); // 1120 / 8
    }

    #[test]
    fn test_small_image() {
        let lp = LayoutParams::new(1, 1, Subsampling::S420, false);
        assert_eq!(lp.padded_width, 16); // MCU-aligned
        assert_eq!(lp.blocks_w, 1);
        assert_eq!(lp.blocks_h, 1);
        assert_eq!(lp.c_blocks_h, 1);
        assert_eq!(lp.c_blocks_v, 1);
    }

    #[test]
    fn test_padded_b_width_xyb_vs_no_xyb() {
        // Without XYB, padded_b_width matches padded_c_width
        let lp_no = LayoutParams::new(1920, 1080, Subsampling::S444, false);
        assert_eq!(lp_no.padded_b_width, lp_no.padded_c_width);

        // With XYB, padded_b_width is computed from b_width (always 2x2 downsampled)
        let lp_xyb = LayoutParams::new(1920, 1080, Subsampling::S444, true);
        assert_eq!(lp_xyb.b_width, 960);
        assert_eq!(lp_xyb.padded_b_width, 960); // 960 is already 8-aligned
    }
}
