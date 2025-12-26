//! Safe allocation helpers for DoS protection.
//!
//! This module provides fallible allocation functions that return errors instead
//! of panicking on OOM. These are used throughout the codebase to prevent
//! memory exhaustion attacks.
//!
//! Based on patterns from libjpeg-turbo's memory management and Rust's
//! `try_reserve` API (stabilized in Rust 1.57).

use crate::jpegli::error::{Error, Result};

/// Maximum dimension for JPEG images (matches libjpeg-turbo's JPEG_MAX_DIMENSION).
/// Slightly under 64K to prevent overflow in 16-bit calculations.
pub const JPEG_MAX_DIMENSION: u32 = 65500;

/// Default maximum pixels (100 megapixels).
/// This is a reasonable limit for most applications.
pub const DEFAULT_MAX_PIXELS: u64 = 100_000_000;

/// Maximum number of progressive scans allowed.
pub const MAX_SCANS: usize = 256;

/// Maximum ICC profile size (16 MB).
pub const MAX_ICC_PROFILE_SIZE: usize = 16 * 1024 * 1024;

/// Default maximum memory for decode operations (512 MB).
/// This limits total allocations during a single decode operation.
pub const DEFAULT_MAX_MEMORY: usize = 512 * 1024 * 1024;

/// Tracks cumulative memory allocations during decode operations.
///
/// This prevents DoS attacks where many small allocations (each under limit)
/// combine to exhaust memory. Used to enforce a global memory budget.
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Total bytes allocated so far
    pub allocated: usize,
    /// Maximum bytes allowed
    pub limit: usize,
}

impl MemoryTracker {
    /// Creates a new tracker with the specified limit.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            allocated: 0,
            limit,
        }
    }

    /// Creates a new tracker with default limit (512 MB).
    #[must_use]
    pub fn with_default_limit() -> Self {
        Self::new(DEFAULT_MAX_MEMORY)
    }

    /// Creates an unlimited tracker (for testing or trusted inputs).
    #[must_use]
    pub fn unlimited() -> Self {
        Self::new(usize::MAX)
    }

    /// Attempts to allocate bytes, returning error if limit exceeded.
    pub fn try_alloc(&mut self, bytes: usize, context: &'static str) -> Result<()> {
        let new_total = self
            .allocated
            .checked_add(bytes)
            .ok_or(Error::SizeOverflow { context })?;

        if new_total > self.limit {
            return Err(Error::AllocationFailed { bytes, context });
        }

        self.allocated = new_total;
        Ok(())
    }

    /// Frees previously allocated bytes.
    pub fn free(&mut self, bytes: usize) {
        self.allocated = self.allocated.saturating_sub(bytes);
    }

    /// Returns remaining available bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.allocated)
    }

    /// Returns current allocation total.
    #[must_use]
    pub fn current(&self) -> usize {
        self.allocated
    }

    /// Resets the tracker for reuse.
    pub fn reset(&mut self) {
        self.allocated = 0;
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::with_default_limit()
    }
}

/// Calculate size with overflow checking.
///
/// Returns an error if the multiplication would overflow.
#[inline]
pub fn checked_size(width: usize, height: usize, bytes_per_pixel: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or(Error::SizeOverflow {
            context: "calculating buffer size",
        })
}

/// Calculate size for a 2D array with overflow checking.
#[inline]
pub fn checked_size_2d(dim1: usize, dim2: usize) -> Result<usize> {
    dim1.checked_mul(dim2).ok_or(Error::SizeOverflow {
        context: "calculating 2D size",
    })
}

/// Validate image dimensions against limits.
///
/// Checks:
/// - Neither dimension is zero
/// - Neither dimension exceeds JPEG_MAX_DIMENSION
/// - Total pixels don't exceed max_pixels
pub fn validate_dimensions(width: u32, height: u32, max_pixels: u64) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions {
            width,
            height,
            reason: "dimensions cannot be zero",
        });
    }

    if width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION {
        return Err(Error::InvalidDimensions {
            width,
            height,
            reason: "exceeds JPEG_MAX_DIMENSION (65500)",
        });
    }

    let total_pixels = (width as u64)
        .checked_mul(height as u64)
        .ok_or(Error::SizeOverflow {
            context: "calculating total pixels",
        })?;

    if total_pixels > max_pixels {
        return Err(Error::ImageTooLarge {
            pixels: total_pixels,
            limit: max_pixels,
        });
    }

    Ok(())
}

/// Allocate a Vec with fallible allocation.
///
/// Returns an error instead of panicking if allocation fails.
#[inline]
pub fn try_alloc_vec<T: Default + Clone>(count: usize, context: &'static str) -> Result<Vec<T>> {
    let byte_size = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::SizeOverflow { context })?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            bytes: byte_size,
            context,
        })?;
    v.resize(count, T::default());
    Ok(v)
}

/// Allocate a Vec of zeros with fallible allocation.
#[inline]
pub fn try_alloc_zeroed(count: usize, context: &'static str) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            bytes: count,
            context,
        })?;
    v.resize(count, 0u8);
    Ok(v)
}

/// Allocate a Vec of f32 zeros with fallible allocation.
#[inline]
pub fn try_alloc_zeroed_f32(count: usize, context: &'static str) -> Result<Vec<f32>> {
    let byte_size = count
        .checked_mul(4)
        .ok_or(Error::SizeOverflow { context })?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            bytes: byte_size,
            context,
        })?;
    v.resize(count, 0.0f32);
    Ok(v)
}

/// Allocate a Vec with specific capacity (no initialization).
#[inline]
pub fn try_with_capacity<T>(capacity: usize, context: &'static str) -> Result<Vec<T>> {
    let byte_size = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::SizeOverflow { context })?;

    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| Error::AllocationFailed {
            bytes: byte_size,
            context,
        })?;
    Ok(v)
}

/// Allocate a Vec of DCT blocks (64 i16 values each) with fallible allocation.
#[inline]
pub fn try_alloc_dct_blocks(count: usize, context: &'static str) -> Result<Vec<[i16; 64]>> {
    let byte_size = count
        .checked_mul(64 * 2) // 64 i16 = 128 bytes per block
        .ok_or(Error::SizeOverflow { context })?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            bytes: byte_size,
            context,
        })?;
    v.resize(count, [0i16; 64]);
    Ok(v)
}

/// Allocate a Vec filled with a specific value using fallible allocation.
#[inline]
pub fn try_alloc_filled<T: Clone>(count: usize, value: T, context: &'static str) -> Result<Vec<T>> {
    let byte_size = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::SizeOverflow { context })?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            bytes: byte_size,
            context,
        })?;
    v.resize(count, value);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_size() {
        assert!(checked_size(100, 100, 3).is_ok());
        assert_eq!(checked_size(100, 100, 3).unwrap(), 30000);

        // Overflow case
        assert!(checked_size(usize::MAX, 2, 1).is_err());
    }

    #[test]
    fn test_validate_dimensions() {
        // Valid case
        assert!(validate_dimensions(1920, 1080, DEFAULT_MAX_PIXELS).is_ok());

        // Zero dimension
        assert!(validate_dimensions(0, 100, DEFAULT_MAX_PIXELS).is_err());
        assert!(validate_dimensions(100, 0, DEFAULT_MAX_PIXELS).is_err());

        // Exceeds JPEG_MAX_DIMENSION
        assert!(validate_dimensions(70000, 100, DEFAULT_MAX_PIXELS).is_err());

        // Exceeds max_pixels
        assert!(validate_dimensions(20000, 20000, 100_000_000).is_err()); // 400M > 100M
    }

    #[test]
    fn test_try_alloc_vec() {
        let v: Vec<u8> = try_alloc_vec(1000, "test").unwrap();
        assert_eq!(v.len(), 1000);
        assert!(v.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_try_alloc_zeroed() {
        let v = try_alloc_zeroed(1000, "test").unwrap();
        assert_eq!(v.len(), 1000);
        assert!(v.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_try_with_capacity() {
        let v: Vec<u8> = try_with_capacity(1000, "test").unwrap();
        assert_eq!(v.capacity(), 1000);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_try_alloc_filled() {
        let v: Vec<u8> = try_alloc_filled(1000, 128u8, "test").unwrap();
        assert_eq!(v.len(), 1000);
        assert!(v.iter().all(|&x| x == 128));
    }

    #[test]
    fn test_memory_tracker_basic() {
        let mut tracker = MemoryTracker::new(1000);
        assert_eq!(tracker.remaining(), 1000);
        assert_eq!(tracker.current(), 0);

        // Allocate some bytes
        tracker.try_alloc(400, "test1").unwrap();
        assert_eq!(tracker.current(), 400);
        assert_eq!(tracker.remaining(), 600);

        // Allocate more
        tracker.try_alloc(300, "test2").unwrap();
        assert_eq!(tracker.current(), 700);
        assert_eq!(tracker.remaining(), 300);
    }

    #[test]
    fn test_memory_tracker_limit() {
        let mut tracker = MemoryTracker::new(1000);

        // Allocate up to limit
        tracker.try_alloc(500, "test1").unwrap();
        tracker.try_alloc(500, "test2").unwrap();
        assert_eq!(tracker.current(), 1000);
        assert_eq!(tracker.remaining(), 0);

        // Exceed limit
        let result = tracker.try_alloc(1, "test3");
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_tracker_free() {
        let mut tracker = MemoryTracker::new(1000);
        tracker.try_alloc(800, "test").unwrap();

        // Free some
        tracker.free(300);
        assert_eq!(tracker.current(), 500);
        assert_eq!(tracker.remaining(), 500);

        // Can allocate again
        tracker.try_alloc(400, "test2").unwrap();
        assert_eq!(tracker.current(), 900);
    }

    #[test]
    fn test_memory_tracker_reset() {
        let mut tracker = MemoryTracker::new(1000);
        tracker.try_alloc(800, "test").unwrap();

        tracker.reset();
        assert_eq!(tracker.current(), 0);
        assert_eq!(tracker.remaining(), 1000);
    }

    #[test]
    fn test_memory_tracker_overflow() {
        let mut tracker = MemoryTracker::new(usize::MAX);
        tracker.try_alloc(usize::MAX - 10, "test1").unwrap();

        // This would overflow
        let result = tracker.try_alloc(100, "test2");
        assert!(result.is_err());
    }
}
