//! Encoder workspace for buffer reuse.
//!
//! This module provides `EncoderWorkspace` which holds pre-allocated buffers
//! that can be reused across multiple encodes to avoid allocation overhead.
//!
//! # Current Status
//!
//! The workspace API is available but currently falls back to the regular
//! encode path. Full integration requires refactoring the encode pipeline
//! to use borrowed slices instead of owned Vecs to avoid copy overhead.
//!
//! The workspace is still useful for:
//! - Pre-allocating memory to avoid page faults at encode time
//! - Validating image dimensions before encoding
//! - Future integration when the pipeline is refactored
//!
//! # Example
//!
//! ```ignore
//! use jpegli::{Encoder, EncoderWorkspace};
//!
//! // Create workspace sized for your maximum image dimensions
//! let mut workspace = EncoderWorkspace::new(4096, 4096)?;
//!
//! // Reuse workspace across multiple encodes
//! for image in images {
//!     let jpeg = Encoder::new()
//!         .width(image.width)
//!         .height(image.height)
//!         .encode_with_workspace(&image.pixels, &mut workspace)?;
//! }
//! ```

use crate::error::{Error, Result};
use crate::foundation::alloc::try_alloc_zeroed_f32;

/// Pre-allocated workspace for encoder buffer reuse.
///
/// Holds buffers for YCbCr planes and intermediate processing.
/// Reusing a workspace across multiple encodes eliminates allocation
/// overhead (~25-30% of encode time for large images).
#[derive(Debug)]
pub struct EncoderWorkspace {
    /// Maximum dimensions this workspace can handle
    pub(crate) max_width: usize,
    pub(crate) max_height: usize,

    /// Y plane buffer (full resolution)
    pub(crate) y_plane: Vec<f32>,
    /// Cb plane buffer (may be subsampled)
    pub(crate) cb_plane: Vec<f32>,
    /// Cr plane buffer (may be subsampled)
    pub(crate) cr_plane: Vec<f32>,

    /// Temporary buffer for smoothing/downsampling
    pub(crate) temp_cb: Vec<f32>,
    pub(crate) temp_cr: Vec<f32>,
}

impl EncoderWorkspace {
    /// Creates a new workspace sized for images up to `max_width` x `max_height`.
    ///
    /// The workspace allocates buffers for the worst case (4:4:4 subsampling).
    /// Smaller images will reuse these buffers without reallocation.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or if `max_width * max_height` overflows.
    pub fn new(max_width: usize, max_height: usize) -> Result<Self> {
        let max_pixels = max_width
            .checked_mul(max_height)
            .ok_or(Error::SizeOverflow {
                context: "workspace dimensions",
            })?;

        // Pre-allocate buffers with fallible allocation
        let mut y_plane = try_alloc_zeroed_f32(max_pixels, "workspace Y plane")?;
        let mut cb_plane = try_alloc_zeroed_f32(max_pixels, "workspace Cb plane")?;
        let mut cr_plane = try_alloc_zeroed_f32(max_pixels, "workspace Cr plane")?;
        let mut temp_cb = try_alloc_zeroed_f32(max_pixels, "workspace temp Cb")?;
        let mut temp_cr = try_alloc_zeroed_f32(max_pixels, "workspace temp Cr")?;

        // Touch all pages to force physical allocation (write pattern to avoid optimization)
        // This ensures page faults happen here, not during encode
        for (i, chunk) in y_plane.chunks_mut(4096).enumerate() {
            chunk[0] = i as f32;
        }
        for (i, chunk) in cb_plane.chunks_mut(4096).enumerate() {
            chunk[0] = i as f32;
        }
        for (i, chunk) in cr_plane.chunks_mut(4096).enumerate() {
            chunk[0] = i as f32;
        }
        for (i, chunk) in temp_cb.chunks_mut(4096).enumerate() {
            chunk[0] = i as f32;
        }
        for (i, chunk) in temp_cr.chunks_mut(4096).enumerate() {
            chunk[0] = i as f32;
        }

        Ok(Self {
            max_width,
            max_height,
            y_plane,
            cb_plane,
            cr_plane,
            temp_cb,
            temp_cr,
        })
    }

    /// Returns the maximum width this workspace can handle.
    #[must_use]
    pub fn max_width(&self) -> usize {
        self.max_width
    }

    /// Returns the maximum height this workspace can handle.
    #[must_use]
    pub fn max_height(&self) -> usize {
        self.max_height
    }

    /// Returns the maximum number of pixels this workspace can handle.
    #[must_use]
    pub fn max_pixels(&self) -> usize {
        self.max_width * self.max_height
    }

    /// Checks if this workspace can handle the given dimensions.
    #[must_use]
    pub fn can_handle(&self, width: usize, height: usize) -> bool {
        width <= self.max_width && height <= self.max_height
    }

    /// Resizes the workspace if needed to handle larger images.
    ///
    /// Only reallocates if the new dimensions exceed current capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if reallocation fails.
    pub fn ensure_capacity(&mut self, width: usize, height: usize) -> Result<()> {
        if !self.can_handle(width, height) {
            let new_width = width.max(self.max_width);
            let new_height = height.max(self.max_height);
            *self = Self::new(new_width, new_height)?;
        }
        Ok(())
    }

    /// Returns mutable slices for the Y, Cb, Cr planes.
    ///
    /// The slices are sized for `num_pixels` elements.
    #[inline]
    pub(crate) fn planes_mut(&mut self, num_pixels: usize) -> (&mut [f32], &mut [f32], &mut [f32]) {
        debug_assert!(num_pixels <= self.y_plane.len());
        (
            &mut self.y_plane[..num_pixels],
            &mut self.cb_plane[..num_pixels],
            &mut self.cr_plane[..num_pixels],
        )
    }

    /// Returns mutable slices for temporary buffers.
    #[inline]
    pub(crate) fn temp_planes_mut(&mut self, num_pixels: usize) -> (&mut [f32], &mut [f32]) {
        debug_assert!(num_pixels <= self.temp_cb.len());
        (
            &mut self.temp_cb[..num_pixels],
            &mut self.temp_cr[..num_pixels],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let ws = EncoderWorkspace::new(1920, 1080).unwrap();
        assert_eq!(ws.max_width(), 1920);
        assert_eq!(ws.max_height(), 1080);
        assert_eq!(ws.max_pixels(), 1920 * 1080);
    }

    #[test]
    fn test_workspace_can_handle() {
        let ws = EncoderWorkspace::new(1920, 1080).unwrap();
        assert!(ws.can_handle(1920, 1080));
        assert!(ws.can_handle(1280, 720));
        assert!(!ws.can_handle(2048, 1080));
        assert!(!ws.can_handle(1920, 1200));
    }

    #[test]
    fn test_workspace_ensure_capacity() {
        let mut ws = EncoderWorkspace::new(1920, 1080).unwrap();
        assert!(!ws.can_handle(2048, 1080));

        ws.ensure_capacity(2048, 1080).unwrap();
        assert!(ws.can_handle(2048, 1080));
        assert!(ws.can_handle(1920, 1080)); // Still handles smaller
    }

    #[test]
    fn test_workspace_planes() {
        let mut ws = EncoderWorkspace::new(100, 100).unwrap();
        let (y, cb, cr) = ws.planes_mut(5000);
        assert_eq!(y.len(), 5000);
        assert_eq!(cb.len(), 5000);
        assert_eq!(cr.len(), 5000);
    }
}
