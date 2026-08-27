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
//! coefficients within each block.
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
