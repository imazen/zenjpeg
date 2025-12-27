//! Image buffer types for butteraugli.
//!
//! These types provide efficient storage for floating-point image data
//! with row-stride support for cache-friendly access patterns.

use std::ops::{Index, IndexMut};

/// Single-channel floating point image.
///
/// This is the primary image type used throughout butteraugli for
/// intermediate computations. It stores pixel values as f32 with
/// optional row padding for alignment.
#[derive(Debug, Clone)]
pub struct ImageF {
    data: Vec<f32>,
    width: usize,
    height: usize,
    stride: usize, // pixels per row (may be > width for alignment)
}

impl ImageF {
    /// Creates a new image filled with zeros.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        // Align stride to 16 floats (64 bytes) for SIMD
        let stride = (width + 15) & !15;
        Self {
            data: vec![0.0; stride * height],
            width,
            height,
            stride,
        }
    }

    /// Creates an image from existing data.
    ///
    /// # Panics
    /// Panics if data length doesn't match width * height.
    #[must_use]
    pub fn from_vec(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        // For imported data, don't add padding
        Self {
            data,
            width,
            height,
            stride: width,
        }
    }

    /// Creates an image with padding for aligned rows.
    #[must_use]
    pub fn from_vec_padded(data: Vec<f32>, width: usize, height: usize, stride: usize) -> Self {
        assert!(stride >= width);
        assert_eq!(data.len(), stride * height);
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    /// Creates an image filled with a constant value.
    #[must_use]
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        let stride = (width + 15) & !15;
        Self {
            data: vec![value; stride * height],
            width,
            height,
            stride,
        }
    }

    /// Image width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of pixels per row (may include padding).
    #[inline]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns a reference to a row.
    #[inline]
    #[must_use]
    pub fn row(&self, y: usize) -> &[f32] {
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Returns a mutable reference to a row.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        let start = y * self.stride;
        &mut self.data[start..start + self.width]
    }

    /// Returns a reference to a full row including padding.
    #[inline]
    #[must_use]
    pub fn row_full(&self, y: usize) -> &[f32] {
        let start = y * self.stride;
        &self.data[start..start + self.stride]
    }

    /// Returns a mutable reference to a full row including padding.
    #[inline]
    pub fn row_full_mut(&mut self, y: usize) -> &mut [f32] {
        let start = y * self.stride;
        &mut self.data[start..start + self.stride]
    }

    /// Gets a pixel value.
    #[inline]
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.stride + x]
    }

    /// Sets a pixel value.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.data[y * self.stride + x] = value;
    }

    /// Returns a raw pointer to the data for SIMD operations.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Returns a mutable raw pointer to the data for SIMD operations.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    /// Returns the raw data as a slice.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns the raw data as a mutable slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Checks if two images have the same dimensions.
    #[must_use]
    pub fn same_size(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height
    }

    /// Copies data from another image.
    ///
    /// # Panics
    /// Panics if dimensions don't match.
    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.same_size(other));
        for y in 0..self.height {
            self.row_mut(y).copy_from_slice(other.row(y));
        }
    }

    /// Fills the image with a constant value.
    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }
}

impl Index<(usize, usize)> for ImageF {
    type Output = f32;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[y * self.stride + x]
    }
}

impl IndexMut<(usize, usize)> for ImageF {
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.data[y * self.stride + x]
    }
}

/// Three-channel floating point image (for XYB data).
#[derive(Debug, Clone)]
pub struct Image3F {
    planes: [ImageF; 3],
}

impl Image3F {
    /// Creates a new 3-channel image.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            planes: [
                ImageF::new(width, height),
                ImageF::new(width, height),
                ImageF::new(width, height),
            ],
        }
    }

    /// Creates from three separate planes.
    #[must_use]
    pub fn from_planes(plane0: ImageF, plane1: ImageF, plane2: ImageF) -> Self {
        assert!(plane0.same_size(&plane1));
        assert!(plane0.same_size(&plane2));
        Self {
            planes: [plane0, plane1, plane2],
        }
    }

    /// Image width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.planes[0].width()
    }

    /// Image height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.planes[0].height()
    }

    /// Returns a reference to a specific plane.
    #[inline]
    #[must_use]
    pub fn plane(&self, index: usize) -> &ImageF {
        &self.planes[index]
    }

    /// Returns a mutable reference to a specific plane.
    #[inline]
    pub fn plane_mut(&mut self, index: usize) -> &mut ImageF {
        &mut self.planes[index]
    }

    /// Returns a row from a specific plane.
    #[inline]
    #[must_use]
    pub fn plane_row(&self, plane: usize, y: usize) -> &[f32] {
        self.planes[plane].row(y)
    }

    /// Returns a mutable row from a specific plane.
    #[inline]
    pub fn plane_row_mut(&mut self, plane: usize, y: usize) -> &mut [f32] {
        self.planes[plane].row_mut(y)
    }
}

impl Index<usize> for Image3F {
    type Output = ImageF;

    fn index(&self, index: usize) -> &Self::Output {
        &self.planes[index]
    }
}

impl IndexMut<usize> for Image3F {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.planes[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation() {
        let img = ImageF::new(100, 50);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
        assert!(img.stride() >= 100);
        assert_eq!(img.stride() % 16, 0); // Aligned
    }

    #[test]
    fn test_pixel_access() {
        let mut img = ImageF::new(10, 10);
        img.set(5, 3, 42.0);
        assert!((img.get(5, 3) - 42.0).abs() < 0.001);
        assert!((img[(5, 3)] - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_row_access() {
        let mut img = ImageF::new(10, 10);
        img.row_mut(5)[3] = 99.0;
        assert!((img.row(5)[3] - 99.0).abs() < 0.001);
    }

    #[test]
    fn test_image3f() {
        let img = Image3F::new(100, 50);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
    }
}
