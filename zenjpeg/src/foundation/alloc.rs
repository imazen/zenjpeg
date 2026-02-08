//! Safe allocation helpers for DoS protection.
//!
//! This module provides fallible allocation functions that return errors instead
//! of panicking on OOM. These are used throughout the codebase to prevent
//! memory exhaustion attacks.
//!
//! Based on patterns from libjpeg-turbo's memory management and Rust's
//! `try_reserve` API (stabilized in Rust 1.57).
//!
//! ## Allocation Profiling
//!
//! When the `alloc-instrument` feature is enabled, allocation functions return
//! `ProfiledVec<T>` which logs utilization stats on drop. This helps identify
//! over-allocated buffers.

#![allow(dead_code)] // Tracking utilities and optional alloc helpers

use crate::error::{Error, Result};

// Re-export for use by callers who want to profile specific allocations
#[allow(unused_imports)]
pub use crate::foundation::instrumented_vec::{
    InstrumentedVec, ProfiledVec, ProfiledVecExt, VecStats,
};

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
pub const DEFAULT_MAX_MEMORY: u64 = 512 * 1024 * 1024;

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
        Self::new(DEFAULT_MAX_MEMORY as usize)
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
            .ok_or_else(|| Error::size_overflow(context))?;

        if new_total > self.limit {
            return Err(Error::allocation_failed(bytes, context));
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

/// Detailed allocation info captured with `#[track_caller]`.
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Bytes allocated
    pub bytes: usize,
    /// Type name (if available)
    pub type_name: &'static str,
    /// Size of one element (for array allocations)
    pub element_size: usize,
    /// Number of elements (1 for single allocations)
    pub count: usize,
    /// Context string (e.g., "DCT blocks", "color plane")
    pub context: &'static str,
    /// Source location (file:line:column)
    pub location: &'static std::panic::Location<'static>,
}

impl AllocationInfo {
    /// Format as a single-line summary
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{:>10} | {:<30} | {:<40} | {}:{}",
            format_bytes(self.bytes),
            self.context,
            if self.count > 1 {
                format!("{}[{}] × {}", self.type_name, self.count, self.element_size)
            } else {
                format!("{} ({}B)", self.type_name, self.element_size)
            },
            self.location
                .file()
                .rsplit('/')
                .next()
                .unwrap_or(self.location.file()),
            self.location.line()
        )
    }
}

/// Tracks allocation statistics during encoding/decoding operations.
///
/// Unlike `MemoryTracker`, this doesn't enforce limits - it just records
/// what was allocated for analysis and prediction purposes.
///
/// When the `detailed` field is `true`, stores full `AllocationInfo` for each
/// allocation including source location via `#[track_caller]`.
#[derive(Debug, Clone, Default)]
pub struct EncodeStats {
    /// Number of allocations made
    pub count: usize,
    /// Total bytes allocated (sum of all allocations)
    pub total_bytes: usize,
    /// Peak bytes allocated at any point (requires manual tracking)
    pub peak_bytes: usize,
    /// Current bytes allocated (for tracking peak)
    current_bytes: usize,
    /// Per-allocation breakdown by context name
    pub by_context: Vec<(&'static str, usize)>,
    /// Detailed allocation info (when enabled)
    pub allocations: Vec<AllocationInfo>,
    /// Whether to capture detailed allocation info
    pub detailed: bool,
}

impl EncodeStats {
    /// Creates a new empty stats tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new stats tracker with detailed tracking enabled.
    #[must_use]
    pub fn with_detailed_tracking() -> Self {
        Self {
            detailed: true,
            ..Self::default()
        }
    }

    /// Records an allocation.
    #[inline]
    pub fn record_alloc(&mut self, bytes: usize) {
        self.count += 1;
        self.total_bytes += bytes;
        self.current_bytes += bytes;
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
    }

    /// Records an allocation with context name for detailed tracking.
    #[inline]
    pub fn record_alloc_named(&mut self, bytes: usize, context: &'static str) {
        self.record_alloc(bytes);
        self.by_context.push((context, bytes));
    }

    /// Records a typed allocation with full source location tracking.
    ///
    /// Uses `#[track_caller]` to capture the allocation site. When `detailed`
    /// is true, stores full `AllocationInfo` for later analysis.
    #[inline]
    #[track_caller]
    pub fn record_alloc_typed<T>(&mut self, count: usize, context: &'static str) {
        let element_size = core::mem::size_of::<T>();
        let bytes = count * element_size;

        self.record_alloc(bytes);
        self.by_context.push((context, bytes));

        if self.detailed {
            self.allocations.push(AllocationInfo {
                bytes,
                type_name: std::any::type_name::<T>(),
                element_size,
                count,
                context,
                location: std::panic::Location::caller(),
            });
        }
    }

    /// Records an allocation with explicit type info and source location.
    ///
    /// Use this when you have the type info but can't use generics.
    #[inline]
    #[track_caller]
    pub fn record_alloc_explicit(
        &mut self,
        bytes: usize,
        type_name: &'static str,
        element_size: usize,
        count: usize,
        context: &'static str,
    ) {
        self.record_alloc(bytes);
        self.by_context.push((context, bytes));

        if self.detailed {
            self.allocations.push(AllocationInfo {
                bytes,
                type_name,
                element_size,
                count,
                context,
                location: std::panic::Location::caller(),
            });
        }
    }

    /// Records a deallocation (for peak tracking).
    #[inline]
    pub fn record_dealloc(&mut self, bytes: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.total_bytes = 0;
        self.peak_bytes = 0;
        self.current_bytes = 0;
        self.by_context.clear();
        self.allocations.clear();
    }

    /// Merges another stats tracker into this one.
    pub fn merge(&mut self, other: &EncodeStats) {
        self.count += other.count;
        self.total_bytes += other.total_bytes;
        self.by_context.extend(other.by_context.iter().cloned());
        self.allocations.extend(other.allocations.iter().cloned());
        // Peak is the max of either tracker's peak
        if other.peak_bytes > self.peak_bytes {
            self.peak_bytes = other.peak_bytes;
        }
    }

    /// Returns a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} allocations, {} total, {} peak",
            self.count,
            format_bytes(self.total_bytes),
            format_bytes(self.peak_bytes)
        )
    }

    /// Returns a detailed breakdown of allocations by source location.
    ///
    /// Only available when `detailed` tracking is enabled.
    #[must_use]
    pub fn detailed_report(&self) -> String {
        if !self.detailed || self.allocations.is_empty() {
            return "Detailed tracking not enabled or no allocations recorded.".to_string();
        }

        let mut lines = Vec::with_capacity(self.allocations.len() + 4);
        lines.push(format!(
            "{:>10} | {:<30} | {:<40} | Location",
            "Size", "Context", "Type"
        ));
        lines.push("-".repeat(100));

        // Sort by size descending
        let mut sorted: Vec<_> = self.allocations.iter().collect();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.bytes));

        for info in sorted {
            lines.push(info.summary());
        }

        lines.push("-".repeat(100));
        lines.push(format!(
            "{:>10} | {} allocations, {} peak",
            format_bytes(self.total_bytes),
            self.count,
            format_bytes(self.peak_bytes)
        ));

        lines.join("\n")
    }

    /// Returns allocations grouped by context, sorted by total size.
    #[must_use]
    pub fn by_context_summary(&self) -> String {
        use std::collections::HashMap;

        let mut by_ctx: HashMap<&'static str, usize> = HashMap::new();
        for (ctx, bytes) in &self.by_context {
            *by_ctx.entry(*ctx).or_default() += bytes;
        }

        let mut sorted: Vec<_> = by_ctx.into_iter().collect();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.1));

        let mut lines = Vec::with_capacity(sorted.len() + 2);
        lines.push(format!("{:>12} | Context", "Total Size"));
        lines.push("-".repeat(50));

        for (ctx, bytes) in &sorted {
            lines.push(format!("{:>12} | {}", format_bytes(*bytes), ctx));
        }

        lines.join("\n")
    }
}

/// Formats bytes as human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

// ============================================================================
// Tracked allocation functions
// ============================================================================

/// Allocate a Vec with tracking.
///
/// When `stats.detailed` is true, captures the allocation site via `#[track_caller]`.
#[inline]
#[track_caller]
pub fn try_alloc_vec_tracked<T: Default + Clone>(
    count: usize,
    context: &'static str,
    stats: &mut EncodeStats,
) -> Result<Vec<T>> {
    let byte_size = count
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, T::default());

    stats.record_alloc_typed::<T>(count, context);
    Ok(v)
}

/// Allocate a Vec of f32 zeros with tracking.
///
/// When `stats.detailed` is true, captures the allocation site via `#[track_caller]`.
#[inline]
#[track_caller]
pub fn try_alloc_zeroed_f32_tracked(
    count: usize,
    context: &'static str,
    stats: &mut EncodeStats,
) -> Result<Vec<f32>> {
    let byte_size = count
        .checked_mul(4)
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, 0.0f32);

    stats.record_alloc_typed::<f32>(count, context);
    Ok(v)
}

/// Allocate a Vec with specific capacity with tracking.
///
/// When `stats.detailed` is true, captures the allocation site via `#[track_caller]`.
#[inline]
#[track_caller]
pub fn try_with_capacity_tracked<T>(
    capacity: usize,
    context: &'static str,
    stats: &mut EncodeStats,
) -> Result<Vec<T>> {
    let byte_size = capacity
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;

    stats.record_alloc_typed::<T>(capacity, context);
    Ok(v)
}

/// Allocate a Vec of DCT blocks with tracking.
///
/// When `stats.detailed` is true, captures the allocation site via `#[track_caller]`.
#[inline]
#[track_caller]
pub fn try_alloc_dct_blocks_tracked(
    count: usize,
    context: &'static str,
    stats: &mut EncodeStats,
) -> Result<Vec<[i16; 64]>> {
    let byte_size = count
        .checked_mul(64 * 2) // 64 i16 = 128 bytes per block
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, [0i16; 64]);

    stats.record_alloc_typed::<[i16; 64]>(count, context);
    Ok(v)
}

/// Calculate size with overflow checking.
///
/// Returns an error if the multiplication would overflow.
#[inline]
pub fn checked_size(width: usize, height: usize, bytes_per_pixel: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or_else(|| Error::size_overflow("calculating buffer size"))
}

/// Calculate size for a 2D array with overflow checking.
#[inline]
pub fn checked_size_2d(dim1: usize, dim2: usize) -> Result<usize> {
    dim1.checked_mul(dim2)
        .ok_or_else(|| Error::size_overflow("calculating 2D size"))
}

/// Validate image dimensions against limits.
///
/// Checks:
/// - Neither dimension is zero
/// - Neither dimension exceeds JPEG_MAX_DIMENSION
/// - Total pixels don't exceed max_pixels
pub fn validate_dimensions(width: u32, height: u32, max_pixels: u64) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::invalid_dimensions(
            width,
            height,
            "dimensions cannot be zero",
        ));
    }

    if width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION {
        return Err(Error::invalid_dimensions(
            width,
            height,
            "exceeds JPEG_MAX_DIMENSION (65500)",
        ));
    }

    let total_pixels = (width as u64)
        .checked_mul(height as u64)
        .ok_or_else(|| Error::size_overflow("calculating total pixels"))?;

    if total_pixels > max_pixels {
        return Err(Error::image_too_large(total_pixels, max_pixels));
    }

    Ok(())
}

/// Allocate a Vec with fallible allocation.
///
/// Returns an error instead of panicking if allocation fails.
#[inline]
pub fn try_alloc_vec<T: Default + Clone>(count: usize, context: &'static str) -> Result<Vec<T>> {
    let byte_size = count
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, T::default());
    Ok(v)
}

/// Allocate a Vec of zeros with fallible allocation.
#[inline]
pub fn try_alloc_zeroed(count: usize, context: &'static str) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(count, context))?;
    v.resize(count, 0u8);
    Ok(v)
}

/// Allocate a Vec of f32 zeros with fallible allocation.
#[inline]
pub fn try_alloc_zeroed_f32(count: usize, context: &'static str) -> Result<Vec<f32>> {
    let byte_size = count
        .checked_mul(4)
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, 0.0f32);
    Ok(v)
}

/// Allocate a Vec with specific capacity (no initialization).
#[inline]
pub fn try_with_capacity<T>(capacity: usize, context: &'static str) -> Result<Vec<T>> {
    let byte_size = capacity
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::with_capacity(capacity);
    v.try_reserve_exact(capacity)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    Ok(v)
}

/// Allocate a Vec with zeroed memory using fallible allocation.
///
/// This function allocates and zero-initializes memory. All callers are expected
/// to overwrite the contents before reading, so the zeroing is technically
/// unnecessary but ensures memory safety.
///
/// # Future optimization path
/// When performance is critical, this could be replaced with `MaybeUninit<T>`-based
/// allocation that skips zeroing, provided callers guarantee complete initialization
/// before any reads. Such optimization would require `unsafe` and careful auditing.
#[inline]
pub fn try_alloc_maybeuninit<T: Default + Clone>(
    count: usize,
    context: &'static str,
) -> Result<Vec<T>> {
    let byte_size = count
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, T::default());
    Ok(v)
}

/// Allocate a Vec of DCT blocks (64 i16 values each) with fallible allocation.
#[inline]
pub fn try_alloc_dct_blocks(count: usize, context: &'static str) -> Result<Vec<[i16; 64]>> {
    let byte_size = count
        .checked_mul(64 * 2) // 64 i16 = 128 bytes per block
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, [0i16; 64]);
    Ok(v)
}

/// Allocate a Vec filled with a specific value using fallible allocation.
#[inline]
pub fn try_alloc_filled<T: Clone>(count: usize, value: T, context: &'static str) -> Result<Vec<T>> {
    let byte_size = count
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.resize(count, value);
    Ok(v)
}

/// Clone a slice into a new Vec using fallible allocation.
#[inline]
pub fn try_clone_slice<T: Clone>(slice: &[T], context: &'static str) -> Result<Vec<T>> {
    let byte_size = slice
        .len()
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(slice.len())
        .map_err(|_| Error::allocation_failed(byte_size, context))?;
    v.extend_from_slice(slice);
    Ok(v)
}

// ============================================================================
// Pixel format conversion with fallible allocation
// ============================================================================

/// Convert grayscale to RGB with fallible allocation.
/// Each gray byte becomes [gray, gray, gray].
#[inline]
pub fn try_gray_to_rgb(data: &[u8], context: &'static str) -> Result<Vec<u8>> {
    let len = data
        .len()
        .checked_mul(3)
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::allocation_failed(len, context))?;

    for &byte in data {
        v.push(byte);
        v.push(byte);
        v.push(byte);
    }
    Ok(v)
}

/// Convert RGBA to RGB with fallible allocation.
/// Drops alpha channel: [R, G, B, A] -> [R, G, B]
#[inline]
pub fn try_rgba_to_rgb(data: &[u8], context: &'static str) -> Result<Vec<u8>> {
    let num_pixels = data.len() / 4;
    let len = num_pixels
        .checked_mul(3)
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::allocation_failed(len, context))?;

    for chunk in data.chunks_exact(4) {
        v.push(chunk[0]);
        v.push(chunk[1]);
        v.push(chunk[2]);
    }
    Ok(v)
}

/// Convert BGR to RGB with fallible allocation.
/// Swaps B and R: [B, G, R] -> [R, G, B]
#[inline]
pub fn try_bgr_to_rgb(data: &[u8], context: &'static str) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(data.len())
        .map_err(|_| Error::allocation_failed(data.len(), context))?;

    for chunk in data.chunks_exact(3) {
        v.push(chunk[2]);
        v.push(chunk[1]);
        v.push(chunk[0]);
    }
    Ok(v)
}

/// Convert BGRA to RGB with fallible allocation.
/// Swaps B and R, drops alpha: [B, G, R, A] -> [R, G, B]
#[inline]
pub fn try_bgra_to_rgb(data: &[u8], context: &'static str) -> Result<Vec<u8>> {
    let num_pixels = data.len() / 4;
    let len = num_pixels
        .checked_mul(3)
        .ok_or_else(|| Error::size_overflow(context))?;

    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::allocation_failed(len, context))?;

    for chunk in data.chunks_exact(4) {
        v.push(chunk[2]);
        v.push(chunk[1]);
        v.push(chunk[0]);
    }
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

    #[test]
    fn test_encode_stats_basic() {
        let mut stats = EncodeStats::new();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);

        stats.record_alloc(1000);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_bytes, 1000);
        assert_eq!(stats.peak_bytes, 1000);

        stats.record_alloc(500);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_bytes, 1500);
        assert_eq!(stats.peak_bytes, 1500);
    }

    #[test]
    fn test_encode_stats_peak_tracking() {
        let mut stats = EncodeStats::new();

        stats.record_alloc(1000);
        assert_eq!(stats.peak_bytes, 1000);

        stats.record_dealloc(400);
        assert_eq!(stats.peak_bytes, 1000); // Peak unchanged

        stats.record_alloc(200);
        assert_eq!(stats.peak_bytes, 1000); // Still at 800, peak unchanged

        stats.record_alloc(500); // Now at 1300 (600 + 200 + 500)
        assert_eq!(stats.peak_bytes, 1300); // New peak
    }

    #[test]
    fn test_encode_stats_tracked_alloc() {
        let mut stats = EncodeStats::new();

        let v: Vec<f32> = try_alloc_zeroed_f32_tracked(100, "test", &mut stats).unwrap();
        assert_eq!(v.len(), 100);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_bytes, 400); // 100 * 4 bytes

        let v2: Vec<[i16; 64]> = try_alloc_dct_blocks_tracked(10, "blocks", &mut stats).unwrap();
        assert_eq!(v2.len(), 10);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_bytes, 400 + 1280); // + 10 * 128 bytes
    }

    #[test]
    fn test_encode_stats_summary() {
        let mut stats = EncodeStats::new();
        stats.record_alloc(1024 * 1024); // 1 MB
        stats.record_alloc(512 * 1024); // 512 KB

        let summary = stats.summary();
        assert!(summary.contains("2 allocations"));
        assert!(summary.contains("MB")); // Total should be in MB
    }
}
