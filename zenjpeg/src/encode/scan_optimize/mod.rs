//! Progressive scan optimization (mozjpeg-style `optimize_scans`).
//!
//! Tries 64 candidate progressive scan configurations and picks the smallest.
//! This is a **lossless** optimization: decoded pixels are identical, but the
//! progressive scan structure is chosen to minimize file size.
//!
//! Expected savings: 1-3% smaller progressive JPEGs.

mod config;
mod estimate;
mod generate;
mod select;

pub(crate) use config::ScanSearchConfig;

use super::config::ProgressiveScan;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;

/// Optimize the progressive scan script for minimum file size.
///
/// Generates 64 candidate scan configurations, estimates the encoded size of
/// each using Huffman frequency analysis, and selects the optimal combination
/// of Al levels, frequency splits, and DC interleaving.
///
/// # Arguments
/// * `y_blocks` - Y channel quantized DCT blocks (zigzag order)
/// * `cb_blocks` - Cb channel quantized DCT blocks
/// * `cr_blocks` - Cr channel quantized DCT blocks
/// * `num_components` - Number of components (1 or 3)
///
/// # Returns
/// Optimized `Vec<ProgressiveScan>` ready for the progressive encoder.
pub(crate) fn optimize_scan_script(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    num_components: u8,
) -> Result<Vec<ProgressiveScan>> {
    let config = ScanSearchConfig::default();

    // Generate candidate scans
    let trial_scans = generate::generate_search_scans(num_components, &config);

    // Estimate sizes for all candidates
    let scan_sizes =
        estimate::estimate_all_scan_sizes(&trial_scans, y_blocks, cb_blocks, cr_blocks);

    // Select best parameters
    let selector = select::ScanSelector::new(num_components, config.clone());
    let result = selector.select_best(&scan_sizes);

    // Build final scan script
    Ok(result.build_final_scans(num_components, &config))
}
