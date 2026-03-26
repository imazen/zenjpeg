//! Aligned, fallible allocation utilities for SIMD operations.
//!
//! All internal buffers should use these types to ensure:
//! - 32-byte alignment for AVX SIMD operations
//! - Fallible allocation (no panic on OOM)

#![allow(dead_code)]

use aligned_vec::{AVec, ConstAlign};

/// 32-byte alignment for AVX SIMD (f32x8 requires 32-byte alignment)
pub type Align32 = ConstAlign<32>;

/// Aligned vector type - guaranteed 32-byte aligned base pointer.
///
/// Use this for all internal buffers that will be accessed with SIMD.
pub type AlignedVec<T> = AVec<T, Align32>;

/// Allocation error type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    /// Failed to reserve memory (out of memory)
    OutOfMemory,
    /// Overflow computing allocation size
    Overflow,
}

impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::Overflow => write!(f, "allocation size overflow"),
        }
    }
}

impl std::error::Error for AllocError {}

/// Try to allocate an aligned vector with `count` elements, all zeroed.
#[inline]
pub fn try_alloc_zeroed(count: usize) -> Result<AlignedVec<f32>, AllocError> {
    let mut vec = AVec::new(0);
    vec.try_reserve_exact(count)
        .map_err(|_| AllocError::OutOfMemory)?;
    vec.resize(count, 0.0);
    Ok(vec)
}

/// Try to allocate an aligned vector with `count` elements of type T.
#[inline]
pub fn try_alloc<T: Copy + Default>(count: usize) -> Result<AlignedVec<T>, AllocError> {
    let mut vec = AVec::new(0);
    vec.try_reserve_exact(count)
        .map_err(|_| AllocError::OutOfMemory)?;
    vec.resize(count, T::default());
    Ok(vec)
}

/// Try to allocate an aligned image buffer with proper stride for SIMD.
///
/// Returns (buffer, stride) where stride is rounded up to multiple of 8
/// to ensure all rows remain 32-byte aligned.
///
/// Buffer contains `stride * height` elements, with actual image data
/// in the first `width` elements of each row.
#[inline]
pub fn try_alloc_image(
    width: usize,
    height: usize,
) -> Result<(AlignedVec<f32>, usize), AllocError> {
    let stride = (width + 7) & !7; // Round up to multiple of 8
    let count = stride.checked_mul(height).ok_or(AllocError::Overflow)?;
    let buffer = try_alloc_zeroed(count)?;
    Ok((buffer, stride))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::{AlignTo, f32x8};

    #[test]
    fn test_aligned_vec_is_32_byte_aligned() {
        let vec = try_alloc_zeroed(64).unwrap();
        let addr = vec.as_ptr() as usize;
        assert_eq!(addr % 32, 0, "AlignedVec should be 32-byte aligned");
    }

    #[test]
    fn test_simd_align_to_has_no_prefix() {
        let vec = try_alloc_zeroed(128).unwrap();
        let (prefix, _aligned, _suffix) = f32x8::simd_align_to(&vec);
        assert_eq!(prefix.len(), 0, "32-byte aligned vec should have no prefix");
    }

    #[test]
    fn test_image_buffer_rows_aligned() {
        let (buffer, stride) = try_alloc_image(100, 4).unwrap();
        assert_eq!(stride, 104); // 100 rounded up to multiple of 8

        // Check that the base is 32-byte aligned
        let base_addr = buffer.as_ptr() as usize;
        assert_eq!(base_addr % 32, 0, "Base buffer should be 32-byte aligned");

        // Since stride is a multiple of 8, and each f32 is 4 bytes,
        // stride * 4 bytes = stride * 4, which is multiple of 32 when stride % 8 == 0.
        // Thus all rows are 32-byte aligned if base is.
        assert_eq!(
            (stride * 4) % 32,
            0,
            "Row stride in bytes should be 32-byte aligned"
        );
    }

    #[test]
    fn test_image_buffer_stride_multiple_of_8() {
        for width in [1, 7, 8, 9, 15, 16, 17, 100, 1000, 1920] {
            let (_buffer, stride) = try_alloc_image(width, 1).unwrap();
            assert_eq!(
                stride % 8,
                0,
                "Stride for width {} should be multiple of 8",
                width
            );
            assert!(stride >= width, "Stride should be >= width");
        }
    }
}
