//! Core coefficient-level transform operations.
//!
//! Implements the seven standard lossless JPEG transforms by manipulating
//! DCT coefficients in zigzag order.

use crate::decode::{ComponentCoefficients, DecodedCoefficients};
use crate::foundation::consts::JPEG_NATURAL_ORDER;

/// Lossless JPEG transform operations.
///
/// These correspond to the transforms supported by jpegtran and libjpeg-turbo.
/// Each is a combination of block rearrangement on the image grid and
/// coefficient manipulation within each 8×8 block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LosslessTransform {
    /// No transform (can still re-optimize Huffman tables).
    None,
    /// Horizontal flip (mirror left/right).
    FlipHorizontal,
    /// Vertical flip (mirror top/bottom).
    FlipVertical,
    /// Transpose (swap rows and columns, reflect across main diagonal).
    Transpose,
    /// Rotate 90° clockwise = transpose + horizontal flip.
    Rotate90,
    /// Rotate 180° = horizontal flip + vertical flip.
    Rotate180,
    /// Rotate 270° clockwise = transpose + vertical flip.
    Rotate270,
    /// Transverse = transpose + rotate 180° (reflect across anti-diagonal).
    Transverse,
}

impl LosslessTransform {
    /// Whether this transform swaps image width and height.
    #[must_use]
    pub fn swaps_dimensions(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90 | Self::Rotate270 | Self::Transverse
        )
    }

    /// Returns the output dimensions after applying this transform.
    ///
    /// Equivalent to manually checking [`swaps_dimensions()`](Self::swaps_dimensions)
    /// and swapping width/height, but less error-prone.
    #[must_use]
    pub fn output_dimensions(self, width: u32, height: u32) -> (u32, u32) {
        if self.swaps_dimensions() {
            (height, width)
        } else {
            (width, height)
        }
    }

    /// All 8 elements of the D4 dihedral group, in enum order.
    pub const ALL: [Self; 8] = [
        Self::None,
        Self::FlipHorizontal,
        Self::FlipVertical,
        Self::Transpose,
        Self::Rotate90,
        Self::Rotate180,
        Self::Rotate270,
        Self::Transverse,
    ];

    /// Map each variant to a dense index for table lookup.
    #[inline]
    const fn to_index(self) -> usize {
        match self {
            Self::None => 0,
            Self::FlipHorizontal => 1,
            Self::FlipVertical => 2,
            Self::Transpose => 3,
            Self::Rotate90 => 4,
            Self::Rotate180 => 5,
            Self::Rotate270 => 6,
            Self::Transverse => 7,
        }
    }

    /// Compose two transforms: apply `self` first, then `other`.
    ///
    /// This follows D4 group multiplication. The result is the single transform
    /// equivalent to applying both in sequence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zenjpeg::lossless::LosslessTransform;
    ///
    /// let combined = LosslessTransform::Rotate90.then(LosslessTransform::FlipHorizontal);
    /// // Rotate90 then FlipH = Transpose
    /// ```
    #[must_use]
    pub fn then(self, other: Self) -> Self {
        // D4 Cayley table: CAYLEY[a][b] = a.then(b)
        // Derived from geometric composition of transforms on (x, y) coordinates.
        //
        // Each transform maps a point (x, y) in [0, W) × [0, H) as follows:
        //   None:           (x, y)
        //   FlipHorizontal: (W-1-x, y)
        //   FlipVertical:   (x, H-1-y)
        //   Transpose:      (y, x)           [swaps dimensions]
        //   Rotate90:       (H-1-y, x)       [swaps dimensions]
        //   Rotate180:      (W-1-x, H-1-y)
        //   Rotate270:      (y, W-1-x)       [swaps dimensions]
        //   Transverse:     (H-1-y, W-1-x)   [swaps dimensions]
        //
        // For "a then b", we compute b(a(x, y)) accounting for dimension swaps.
        Self::ALL[Self::CAYLEY[self.to_index()][other.to_index()]]
    }

    /// Returns the inverse of this transform.
    ///
    /// `t.then(t.inverse()) == LosslessTransform::None` for all `t`.
    #[must_use]
    pub fn inverse(self) -> Self {
        // All reflections (FlipH, FlipV, Transpose, Transverse) are self-inverse.
        // Rotations pair up: 90 ↔ 270, 180 ↔ 180.
        match self {
            Self::None => Self::None,
            Self::FlipHorizontal => Self::FlipHorizontal,
            Self::FlipVertical => Self::FlipVertical,
            Self::Transpose => Self::Transpose,
            Self::Rotate90 => Self::Rotate270,
            Self::Rotate180 => Self::Rotate180,
            Self::Rotate270 => Self::Rotate90,
            Self::Transverse => Self::Transverse,
        }
    }

    /// D4 Cayley table. `CAYLEY[a][b]` gives the index of `a.then(b)`.
    ///
    /// Row = first transform applied, Column = second transform applied.
    /// Indices map to `ALL` array: 0=None, 1=FlipH, 2=FlipV, 3=Transpose,
    /// 4=Rot90, 5=Rot180, 6=Rot270, 7=Transverse.
    ///
    /// Derived by composing the (x, y) coordinate mappings of each pair.
    #[rustfmt::skip]
    const CAYLEY: [[usize; 8]; 8] = [
        //              None  FlipH FlipV Trans Rot90 R180  R270  Trnvs
        /* None      */ [0,    1,    2,    3,    4,    5,    6,    7],
        /* FlipH     */ [1,    0,    5,    6,    7,    2,    3,    4],
        /* FlipV     */ [2,    5,    0,    4,    3,    1,    7,    6],
        /* Transpose */ [3,    4,    6,    0,    1,    7,    2,    5],
        /* Rotate90  */ [4,    3,    7,    2,    5,    6,    0,    1],
        /* Rotate180 */ [5,    2,    1,    7,    6,    0,    4,    3],
        /* Rotate270 */ [6,    7,    3,    1,    0,    4,    5,    2],
        /* Transvrse */ [7,    6,    4,    5,    2,    3,    1,    0],
    ];
}

/// How to handle images with non-MCU-aligned dimensions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EdgeHandling {
    /// Trim partial MCU blocks (output may be slightly smaller).
    #[default]
    TrimPartialBlocks,
    /// Error if dimensions aren't MCU-aligned.
    RejectPartialBlocks,
}

/// Configuration for lossless JPEG transforms.
#[derive(Clone, Debug)]
pub struct TransformConfig {
    /// The transform to apply.
    pub transform: LosslessTransform,
    /// How to handle non-MCU-aligned dimensions.
    pub edge_handling: EdgeHandling,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            transform: LosslessTransform::None,
            edge_handling: EdgeHandling::default(),
        }
    }
}

/// Precomputed per-coefficient transform for a single 8×8 block.
///
/// For each zigzag position `z` in [0, 64), `entries[z]` gives the destination
/// zigzag position and whether to negate the coefficient.
pub struct BlockTransform {
    /// `entries[src_zigzag] = (dst_zigzag, negate)`
    pub entries: [(u8, bool); 64],
}

impl BlockTransform {
    /// Build the block-level coefficient transform for the given operation.
    ///
    /// The transform maps `(row, col)` in the source 8×8 block to
    /// `(new_row, new_col)` in the destination block, with optional negation.
    #[must_use]
    pub fn for_transform(transform: LosslessTransform) -> Self {
        let mut entries = [(0u8, false); 64];

        for src_z in 0..64usize {
            // Convert zigzag index to 8×8 (row, col)
            let linear = JPEG_NATURAL_ORDER[src_z] as usize;
            let src_row = linear / 8;
            let src_col = linear % 8;

            // Apply transform to get destination (row, col) and sign
            let (dst_row, dst_col, negate) = match transform {
                LosslessTransform::None => (src_row, src_col, false),

                LosslessTransform::FlipHorizontal => {
                    // Negate odd columns (antisymmetric horizontal basis functions)
                    (src_row, src_col, src_col % 2 == 1)
                }

                LosslessTransform::FlipVertical => {
                    // Negate odd rows (antisymmetric vertical basis functions)
                    (src_row, src_col, src_row % 2 == 1)
                }

                LosslessTransform::Transpose => {
                    // Swap row and col indices, no sign change
                    (src_col, src_row, false)
                }

                LosslessTransform::Rotate90 => {
                    // Transpose + horizontal flip
                    // Transpose: (row, col) → (col, row)
                    // Then H-flip negates odd columns of the transposed block
                    // But odd cols after transpose = odd rows before transpose
                    (src_col, src_row, src_row % 2 == 1)
                }

                LosslessTransform::Rotate180 => {
                    // H-flip + V-flip: negate when (row + col) is odd
                    (src_row, src_col, (src_row + src_col) % 2 == 1)
                }

                LosslessTransform::Rotate270 => {
                    // Transpose + vertical flip
                    // Transpose: (row, col) → (col, row)
                    // Then V-flip negates odd rows of the transposed block
                    // Odd rows after transpose = odd cols before transpose
                    (src_col, src_row, src_col % 2 == 1)
                }

                LosslessTransform::Transverse => {
                    // Transpose + rotate 180°
                    // Transpose: (row, col) → (col, row)
                    // Then 180°: negate when (row + col) is odd
                    // After transpose, row=col, col=row, so (col + row) parity is same
                    (src_col, src_row, (src_row + src_col) % 2 == 1)
                }
            };

            // Convert destination (row, col) back to zigzag index
            let dst_linear = dst_row * 8 + dst_col;
            // Use the inverse of JPEG_NATURAL_ORDER to find zigzag index
            let dst_z = linear_to_zigzag(dst_linear);

            entries[src_z] = (dst_z, negate);
        }

        Self { entries }
    }

    /// Apply this block transform to a coefficient block in-place.
    ///
    /// `src` contains 64 i16 coefficients in zigzag order.
    /// Returns the transformed block.
    #[must_use]
    pub fn apply(&self, src: &[i16; 64]) -> [i16; 64] {
        let mut dst = [0i16; 64];
        for (src_z, &(dst_z, negate)) in self.entries.iter().enumerate() {
            let val = src[src_z];
            dst[dst_z as usize] = if negate { -val } else { val };
        }
        dst
    }
}

/// Convert a linear 8×8 index (row*8+col) to zigzag scan index.
fn linear_to_zigzag(linear: usize) -> u8 {
    debug_assert!(linear < 64);
    // Build the inverse mapping from JPEG_NATURAL_ORDER
    // JPEG_NATURAL_ORDER[zigzag] = linear, so we want the reverse
    ZIGZAG_FROM_LINEAR[linear]
}

/// Inverse of `JPEG_NATURAL_ORDER`: maps linear index → zigzag index.
///
/// `ZIGZAG_FROM_LINEAR[row*8+col] = zigzag_position`
pub(crate) const ZIGZAG_FROM_LINEAR: [u8; 64] = build_zigzag_from_linear();

const fn build_zigzag_from_linear() -> [u8; 64] {
    let mut table = [0u8; 64];
    let mut z = 0;
    while z < 64 {
        table[JPEG_NATURAL_ORDER[z] as usize] = z as u8;
        z += 1;
    }
    table
}

/// Result of transforming decoded coefficients.
pub struct TransformedCoefficients {
    /// New image width in pixels.
    pub width: u32,
    /// New image height in pixels.
    pub height: u32,
    /// Per-component transformed coefficient data.
    pub components: Vec<ComponentCoefficients>,
    /// Quantization tables (unchanged from source).
    pub quant_tables: Vec<Option<[u16; 64]>>,
}

/// Transform decoded DCT coefficients losslessly.
///
/// This rearranges blocks on the image grid and manipulates coefficients within
/// each block according to the specified transform.
///
/// # Arguments
/// * `coeffs` - Decoded coefficients from `Decoder::decode_coefficients()`
/// * `config` - Transform configuration
///
/// # Returns
/// Transformed coefficients ready for Huffman encoding, or an error if the
/// image dimensions aren't MCU-aligned and `EdgeHandling::RejectPartialBlocks` was requested.
pub fn transform_coefficients(
    coeffs: &DecodedCoefficients,
    config: &TransformConfig,
) -> Result<TransformedCoefficients, TransformError> {
    if config.transform == LosslessTransform::None {
        return Ok(TransformedCoefficients {
            width: coeffs.width,
            height: coeffs.height,
            components: coeffs.components.clone(),
            quant_tables: coeffs.quant_tables.clone(),
        });
    }

    let block_transform = BlockTransform::for_transform(config.transform);
    let swaps = config.transform.swaps_dimensions();

    // Calculate MCU dimensions
    let max_h_samp = coeffs
        .components
        .iter()
        .map(|c| c.h_samp)
        .max()
        .unwrap_or(1);
    let max_v_samp = coeffs
        .components
        .iter()
        .map(|c| c.v_samp)
        .max()
        .unwrap_or(1);
    let mcu_width = max_h_samp as u32 * 8;
    let mcu_height = max_v_samp as u32 * 8;

    // Check MCU alignment for transforms that need it
    let needs_h_trim = coeffs.width % mcu_width != 0;
    let needs_v_trim = coeffs.height % mcu_height != 0;

    let needs_trim = match config.transform {
        LosslessTransform::FlipHorizontal => needs_h_trim,
        LosslessTransform::FlipVertical => needs_v_trim,
        LosslessTransform::Rotate90 => needs_v_trim,
        LosslessTransform::Rotate180 => needs_h_trim || needs_v_trim,
        LosslessTransform::Rotate270 => needs_h_trim,
        LosslessTransform::Transpose => false, // never has edge issues
        LosslessTransform::Transverse => needs_h_trim || needs_v_trim,
        LosslessTransform::None => false,
    };

    if needs_trim && config.edge_handling == EdgeHandling::RejectPartialBlocks {
        return Err(TransformError::NotMcuAligned {
            width: coeffs.width,
            height: coeffs.height,
            mcu_width,
            mcu_height,
        });
    }

    // For trim mode, we may reduce the block grid for affected components.
    // For simplicity in this initial implementation, we transform the full
    // block grid (including padding blocks) and adjust the output dimensions.

    let (new_width, new_height) = if swaps {
        // For trim: trim the source dimension that will become problematic after swap
        let trimmed_w = if needs_trim {
            (coeffs.width / mcu_width) * mcu_width
        } else {
            coeffs.width
        };
        let trimmed_h = if needs_trim {
            (coeffs.height / mcu_height) * mcu_height
        } else {
            coeffs.height
        };
        (trimmed_h, trimmed_w)
    } else {
        let trimmed_w = if needs_h_trim && needs_trim {
            (coeffs.width / mcu_width) * mcu_width
        } else {
            coeffs.width
        };
        let trimmed_h = if needs_v_trim && needs_trim {
            (coeffs.height / mcu_height) * mcu_height
        } else {
            coeffs.height
        };
        (trimmed_w, trimmed_h)
    };

    let mut transformed_components = Vec::with_capacity(coeffs.components.len());

    for comp in &coeffs.components {
        let src_bw = comp.blocks_wide;
        let src_bh = comp.blocks_high;

        // For transforms that swap dimensions, swap block grid too
        let (dst_bw, dst_bh) = if swaps {
            (src_bh, src_bw)
        } else {
            (src_bw, src_bh)
        };

        let total_blocks = dst_bw * dst_bh;
        let mut dst_coeffs = vec![0i16; total_blocks * 64];

        for src_by in 0..src_bh {
            for src_bx in 0..src_bw {
                // Calculate destination block position
                let (dst_bx, dst_by) =
                    remap_block(src_bx, src_by, src_bw, src_bh, config.transform);

                // Get source block
                let src_idx = src_by * src_bw + src_bx;
                let src_block = comp.block(src_idx);
                let mut src_arr = [0i16; 64];
                src_arr.copy_from_slice(src_block);

                // Transform coefficients within the block
                let dst_block = block_transform.apply(&src_arr);

                // Write to destination
                let dst_idx = dst_by * dst_bw + dst_bx;
                let dst_start = dst_idx * 64;
                dst_coeffs[dst_start..dst_start + 64].copy_from_slice(&dst_block);
            }
        }

        // For dimension-swapping transforms, also swap sampling factors
        let (dst_h_samp, dst_v_samp) = if swaps {
            (comp.v_samp, comp.h_samp)
        } else {
            (comp.h_samp, comp.v_samp)
        };

        transformed_components.push(ComponentCoefficients {
            id: comp.id,
            coeffs: dst_coeffs,
            blocks_wide: dst_bw,
            blocks_high: dst_bh,
            h_samp: dst_h_samp,
            v_samp: dst_v_samp,
            quant_table_idx: comp.quant_table_idx,
        });
    }

    Ok(TransformedCoefficients {
        width: new_width,
        height: new_height,
        components: transformed_components,
        quant_tables: coeffs.quant_tables.clone(),
    })
}

/// Remap a block position for the given transform.
///
/// Returns the destination (bx, by) for a source block at (src_bx, src_by)
/// in a grid of (blocks_wide, blocks_high).
pub(crate) fn remap_block(
    src_bx: usize,
    src_by: usize,
    blocks_wide: usize,
    blocks_high: usize,
    transform: LosslessTransform,
) -> (usize, usize) {
    match transform {
        LosslessTransform::None => (src_bx, src_by),

        LosslessTransform::FlipHorizontal => (blocks_wide - 1 - src_bx, src_by),

        LosslessTransform::FlipVertical => (src_bx, blocks_high - 1 - src_by),

        LosslessTransform::Transpose => {
            // (bx, by) → (by, bx); grid becomes (blocks_high, blocks_wide)
            (src_by, src_bx)
        }

        LosslessTransform::Rotate90 => {
            // Transpose + H-flip
            // Transpose: (bx, by) → (by, bx) in grid (bh, bw)
            // H-flip: (bx, by) → (bh-1-bx, by) in transposed grid
            // Combined: (bx, by) → (blocks_high - 1 - src_by, src_bx)
            (blocks_height_minus_1(blocks_high) - src_by, src_bx)
        }

        LosslessTransform::Rotate180 => (blocks_wide - 1 - src_bx, blocks_high - 1 - src_by),

        LosslessTransform::Rotate270 => {
            // Transpose + V-flip
            // Transpose: (bx, by) → (by, bx) in grid (bh, bw)
            // V-flip: (bx, by) → (bx, bw-1-by) in transposed grid
            // Combined: (bx, by) → (src_by, blocks_wide - 1 - src_bx)
            (src_by, blocks_wide - 1 - src_bx)
        }

        LosslessTransform::Transverse => {
            // Transpose + 180°
            // Transpose: (bx, by) → (by, bx)
            // Then 180°: (bx, by) → (bh-1-bx, bw-1-by) in transposed grid (bh, bw)
            // Combined: (bx, by) → (blocks_height-1-src_by, blocks_wide-1-src_bx)
            // Wait, after transpose grid is (blocks_high, blocks_wide) → but then
            // within that grid, 180° mirrors both axes.
            // Grid after transpose = (src_bh × src_bw) but we label dst_bw=src_bh, dst_bh=src_bw
            // So: transpose gives (by, bx) in grid (bh, bw)
            // 180° in that grid: (bh-1-by, bw-1-bx) → but bh here = src_bh, bw = src_bw
            // Wait no. After transpose, dst_bw = src_bh, dst_bh = src_bw.
            // So 180° in the transposed grid: (dst_bw - 1 - x, dst_bh - 1 - y)
            // where (x, y) = (src_by, src_bx)
            // = (src_bh - 1 - src_by, src_bw - 1 - src_bx)
            // Hmm, but blocks_high here refers to the SOURCE grid. Let me just think
            // of it as blocks_high - 1 - src_by and blocks_wide - 1 - src_bx.
            // After transpose+180, dst_bw = blocks_high, dst_bh = blocks_wide:
            // dst_bx = blocks_high - 1 - src_by, dst_by = blocks_wide - 1 - src_bx
            (blocks_high - 1 - src_by, blocks_wide - 1 - src_bx)
        }
    }
}

fn blocks_height_minus_1(blocks_high: usize) -> usize {
    debug_assert!(blocks_high > 0);
    blocks_high - 1
}

/// Errors from lossless transform operations.
#[derive(Clone, Debug)]
pub enum TransformError {
    /// Image dimensions are not MCU-aligned and `EdgeHandling::RejectPartialBlocks` was requested.
    NotMcuAligned {
        width: u32,
        height: u32,
        mcu_width: u32,
        mcu_height: u32,
    },
}

impl core::fmt::Display for TransformError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotMcuAligned {
                width,
                height,
                mcu_width,
                mcu_height,
            } => {
                write!(
                    f,
                    "image dimensions {}×{} are not aligned to MCU size {}×{} \
                     (use EdgeHandling::TrimPartialBlocks or resize to {}×{})",
                    width,
                    height,
                    mcu_width,
                    mcu_height,
                    (width / mcu_width) * mcu_width,
                    (height / mcu_height) * mcu_height,
                )
            }
        }
    }
}
