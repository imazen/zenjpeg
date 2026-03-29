#![allow(dead_code)] // Used when alloc-instrument feature is enabled
//! Instrumented Vec for allocation analysis.
//!
//! When `alloc-instrument` feature is enabled, allocations are tracked and
//! utilization stats are logged on drop. Otherwise, this is a zero-cost
//! wrapper around Vec.
//!
//! Usage:
//! ```ignore
//! use crate::foundation::instrumented_vec::ProfiledVec;
//!
//! // Works like Vec, logs stats on drop when feature enabled
//! let mut v: ProfiledVec<u8> = ProfiledVec::with_capacity(100, "my_buffer");
//! v.push(1);
//! ```

use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

// ============================================================================
// Type alias - switches between instrumented and plain Vec
// ============================================================================

/// A Vec that optionally tracks allocation stats.
///
/// When `alloc-instrument` feature is enabled, logs utilization on drop.
/// Otherwise, this is just `Vec<T>` with zero overhead.
#[cfg(feature = "__alloc-instrument")]
pub type ProfiledVec<T> = InstrumentedVec<T>;

#[cfg(not(feature = "__alloc-instrument"))]
pub type ProfiledVec<T> = Vec<T>;

// ============================================================================
// Helper trait for consistent creation API
// ============================================================================

/// Extension trait for creating profiled vecs with context.
pub trait ProfiledVecExt<T> {
    /// Create with capacity and context for profiling.
    fn with_capacity_profiled(capacity: usize, context: &'static str) -> Self;

    /// Create empty with context for profiling.
    fn new_profiled(context: &'static str) -> Self;
}

#[cfg(feature = "__alloc-instrument")]
impl<T> ProfiledVecExt<T> for ProfiledVec<T> {
    fn with_capacity_profiled(capacity: usize, context: &'static str) -> Self {
        InstrumentedVec::with_capacity(capacity, context)
    }

    fn new_profiled(context: &'static str) -> Self {
        InstrumentedVec::new(context)
    }
}

#[cfg(not(feature = "__alloc-instrument"))]
impl<T> ProfiledVecExt<T> for ProfiledVec<T> {
    #[inline]
    fn with_capacity_profiled(capacity: usize, _context: &'static str) -> Self {
        Vec::with_capacity(capacity)
    }

    #[inline]
    fn new_profiled(_context: &'static str) -> Self {
        Vec::new()
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics collected from a ProfiledVec on drop.
#[derive(Debug, Clone)]
pub struct VecStats {
    pub context: &'static str,
    pub final_len: usize,
    pub final_capacity: usize,
    pub initial_capacity: usize,
    pub realloc_count: u32,
    pub peak_capacity: usize,
    pub element_size: usize,
}

impl VecStats {
    /// Utilization as a percentage (0-100)
    pub fn utilization_pct(&self) -> f32 {
        if self.final_capacity == 0 {
            100.0
        } else {
            (self.final_len as f32 / self.final_capacity as f32) * 100.0
        }
    }

    /// Wasted bytes
    pub fn wasted_bytes(&self) -> usize {
        (self.final_capacity.saturating_sub(self.final_len)) * self.element_size
    }
}

// ============================================================================
// Global stats control
// ============================================================================

#[cfg(feature = "__alloc-instrument")]
static STATS_ENABLED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// Minimum wasted bytes to report (avoids noise from small allocations).
#[cfg(feature = "__alloc-instrument")]
static MIN_WASTE_REPORT: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(1024); // 1KB default

/// Enable or disable stats logging.
#[cfg(feature = "__alloc-instrument")]
pub fn set_stats_enabled(enabled: bool) {
    STATS_ENABLED.store(enabled, core::sync::atomic::Ordering::Relaxed);
}

/// Set minimum wasted bytes threshold for reporting.
#[cfg(feature = "__alloc-instrument")]
pub fn set_min_waste_report(bytes: usize) {
    MIN_WASTE_REPORT.store(bytes, core::sync::atomic::Ordering::Relaxed);
}

// ============================================================================
// InstrumentedVec implementation
// ============================================================================

/// A Vec wrapper that tracks allocation statistics.
#[derive(Debug)]
pub struct InstrumentedVec<T> {
    inner: Vec<T>,
    context: &'static str,
    initial_capacity: usize,
    realloc_count: u32,
    peak_capacity: usize,
    last_capacity: usize,
}

impl<T> InstrumentedVec<T> {
    /// Create empty.
    pub fn new(context: &'static str) -> Self {
        Self {
            inner: Vec::new(),
            context,
            initial_capacity: 0,
            realloc_count: 0,
            peak_capacity: 0,
            last_capacity: 0,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize, context: &'static str) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
            context,
            initial_capacity: capacity,
            realloc_count: 0,
            peak_capacity: capacity,
            last_capacity: capacity,
        }
    }

    /// Wrap an existing Vec.
    pub fn from_vec(vec: Vec<T>, context: &'static str) -> Self {
        let cap = vec.capacity();
        Self {
            inner: vec,
            context,
            initial_capacity: cap,
            realloc_count: 0,
            peak_capacity: cap,
            last_capacity: cap,
        }
    }

    /// Check for reallocation.
    #[inline]
    fn check_realloc(&mut self) {
        let current = self.inner.capacity();
        if current != self.last_capacity {
            if self.last_capacity > 0 {
                self.realloc_count += 1;
            }
            self.last_capacity = current;
            if current > self.peak_capacity {
                self.peak_capacity = current;
            }
        }
    }

    /// Push with realloc tracking.
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
        self.check_realloc();
    }

    /// Extend with realloc tracking.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter);
        self.check_realloc();
    }

    /// Reserve with realloc tracking.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
        self.check_realloc();
    }

    /// Resize with realloc tracking.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.inner.resize(new_len, value);
        self.check_realloc();
    }

    /// Get current statistics.
    pub fn stats(&self) -> VecStats {
        VecStats {
            context: self.context,
            final_len: self.inner.len(),
            final_capacity: self.inner.capacity(),
            initial_capacity: self.initial_capacity,
            realloc_count: self.realloc_count,
            peak_capacity: self.peak_capacity,
            element_size: core::mem::size_of::<T>(),
        }
    }

    /// Consume and return inner Vec (skips drop logging).
    pub fn into_inner(self) -> Vec<T> {
        // Use ManuallyDrop to prevent running our Drop impl while extracting inner
        let mut this = core::mem::ManuallyDrop::new(self);
        // Take ownership of inner. This is safe because:
        // 1. We wrapped self in ManuallyDrop so our Drop won't run
        // 2. We're taking the only reference to inner via mutable borrow
        // 3. The ManuallyDrop wrapper will be forgotten without any cleanup
        core::mem::take(&mut this.inner)
    }

    /// Get inner Vec reference.
    pub fn inner(&self) -> &Vec<T> {
        &self.inner
    }
}

impl<T> Deref for InstrumentedVec<T> {
    type Target = Vec<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for InstrumentedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T> From<InstrumentedVec<T>> for Vec<T> {
    fn from(v: InstrumentedVec<T>) -> Vec<T> {
        v.into_inner()
    }
}

#[cfg(feature = "__alloc-instrument")]
impl<T> Drop for InstrumentedVec<T> {
    fn drop(&mut self) {
        if !STATS_ENABLED.load(core::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let stats = self.stats();
        let wasted = stats.wasted_bytes();
        let min_report = MIN_WASTE_REPORT.load(core::sync::atomic::Ordering::Relaxed);

        // Only log if waste exceeds threshold or there were reallocations
        if wasted >= min_report || stats.realloc_count > 0 {
            let util = stats.utilization_pct();
            eprintln!(
                "[alloc] {}: len={} cap={} ({:.0}% util, {}B wasted) init={} reallocs={}",
                stats.context,
                stats.final_len,
                stats.final_capacity,
                util,
                wasted,
                stats.initial_capacity,
                stats.realloc_count,
            );
        }
    }
}

#[cfg(not(feature = "__alloc-instrument"))]
impl<T> Drop for InstrumentedVec<T> {
    fn drop(&mut self) {
        // No-op
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiled_vec_basic() {
        let mut v: ProfiledVec<u8> = ProfiledVec::with_capacity_profiled(100, "test");
        for i in 0..50u8 {
            v.push(i);
        }
        assert_eq!(v.len(), 50);
    }

    #[test]
    fn test_into_inner() {
        let mut v = InstrumentedVec::with_capacity(10, "test");
        v.push(1u8);
        v.push(2);
        let inner: Vec<u8> = v.into_inner();
        assert_eq!(inner, vec![1, 2]);
    }
}
