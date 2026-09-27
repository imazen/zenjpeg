//! Lossless JPEG transforms.
//!
//! Performs rotation, flip, and transpose operations directly on DCT coefficients
//! without decoding to pixels. This is mathematically lossless — zero generation loss.
//!
//! # How It Works
//!
//! JPEG stores image data as 8×8 blocks of DCT coefficients. The DCT basis functions
//! have symmetry properties that allow spatial transforms (flip, rotate, transpose) to
//! be performed by rearranging blocks on the image grid and selectively negating
//! coefficients within each block. Dimension-swapping transforms transpose the
//! quantization tables along with the blocks.
//!
//! # Metadata and secondary images
//!
//! Every APPn/COM segment of the source is carried through. A Multi-Picture
//! (MPF) container — an Ultra HDR JPEG with its gain map, a depth map, MPF
//! thumbnails — is carried as a whole: each secondary image receives the same
//! transform, the MPF index is rebuilt for the new byte layout, and a GContainer
//! XMP directory's `Item:Length` values are updated. A secondary that cannot be
//! transformed is an error rather than a silently dropped or mismatched image.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenjpeg::lossless::{transform, LosslessTransform, TransformConfig, EdgeHandling};
//!
//! let rotated = transform(&jpeg_data, &TransformConfig {
//!     transform: LosslessTransform::Rotate90,
//!     ..Default::default()
//! }, enough::Unstoppable)?;
//! ```

mod coeff_transform;
// Crate-internal handle for the recompress preserve emitter's progressive
// smallest-trial (#143); NOT part of the public lossless API. `coeff_transform`
// is a private module, so this re-export IS the only path for
// recompress/strategies/preserve_emit.rs. It is unused without that feature
// (hence the cfg) — removing it "as unused" under a default clippy run broke
// the `recompress` build once (b4ec5574); keep the cfg, not a deletion.
#[cfg(feature = "recompress")]
pub(crate) use coeff_transform::TransformedCoefficients;
// Crate-internal handle for the recompress preserve emitter's progressive
// smallest-trial (#143); NOT part of the public lossless API.
mod exif;
mod geometry;
mod pipeline;
pub(crate) mod restructure;
#[cfg(test)]
mod tests;

pub use coeff_transform::{
    BlockTransform, EdgeHandling, LosslessTransform, TransformConfig, transform_coefficients,
};
pub use exif::{parse_exif_orientation, set_exif_orientation};
pub use pipeline::{apply_exif_orientation, transform};
pub use restructure::{OutputMode, RestartInterval, RestructureConfig, restructure};
