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
//! Every APPn/COM segment of the source is carried through, and so is every
//! byte after the primary image:
//!
//! - A Multi-Picture (MPF) container (an Ultra HDR JPEG with its gain map, a
//!   depth map, MPF thumbnails) is carried as a whole: each secondary JPEG
//!   receives the same transform, the MPF index is rebuilt for the new byte
//!   layout, and a GContainer XMP directory's `Item:Length` values are
//!   updated.
//! - A gain map that follows the primary without an MPF entry (signalled by
//!   `hdrgm` XMP or a directory `GainMap` item, as some Android files are
//!   written) is carried the same way and gains an index entry.
//! - Whatever follows the secondary JPEGs (a Motion Photo or
//!   `GCamera:MicroVideo` video, a vendor trailer) is copied verbatim at the
//!   end, where those formats' offsets, counted from the end of the file,
//!   still find it; a directory item there keeps its length.
//! - An MPF index whose images were stripped from the file (as
//!   `jpegtran -copy all` leaves it) describes nothing the output holds and
//!   is not written; the XMP is kept as it is.
//!
//! What cannot be carried is an error, never silently dropped or left
//! untransformed beside a transformed primary:
//! [`ErrorKind::UnsupportedFeature`](crate::decoder::ErrorKind::UnsupportedFeature)
//! when the current single-scan writer would have to drop or activate a JFIF, Adobe,
//! or ICC segment found after the source's first scan (use progressive
//! [`lossless::restructure`](fn@crate::lossless::restructure) to retain it),
//! for a secondary that must be transformed and holds secondary images of
//! its own, or a directory image item that is not a secondary JPEG of the
//! file;
//! [`ErrorKind::InvalidJpegData`](crate::decoder::ErrorKind::InvalidJpegData)
//! for an MPF entry that is not a separate JPEG after the primary (inside the
//! primary, overlapping or repeating another), bytes between secondary JPEGs
//! that no index describes, a signalled gain map that cannot be read (no EOI,
//! larger than the primary), or an XMP directory that does not parse while
//! bytes follow the primary; and
//! [`ErrorKind::ImageTooLarge`](crate::decoder::ErrorKind::ImageTooLarge)
//! when the secondaries to transform declare more pixels than the primary
//! (one) or four times the primary (all together), which bounds their decode
//! work. A [`lossless::restructure`](fn@crate::lossless::restructure)
//! without a transform copies the secondaries verbatim and so refuses less.
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
// Same handle for that emitter, which carries nothing after the primary.
#[cfg(feature = "recompress")]
pub(crate) use pipeline::Carry;
// Crate-internal handle for the recompress preserve emitter's progressive
// smallest-trial (#143); NOT part of the public lossless API.
mod exif;
mod geometry;
mod pipeline;
pub(crate) mod placement;
pub(crate) mod restructure;
#[cfg(test)]
mod tests;

pub use coeff_transform::{
    BlockTransform, EdgeHandling, LosslessTransform, TransformConfig, transform_coefficients,
};
pub use exif::{parse_exif_orientation, set_exif_orientation};
pub use pipeline::{apply_exif_orientation, apply_exif_orientation_with_edge_handling, transform};
pub use restructure::{OutputMode, RestartInterval, RestructureConfig, restructure};
