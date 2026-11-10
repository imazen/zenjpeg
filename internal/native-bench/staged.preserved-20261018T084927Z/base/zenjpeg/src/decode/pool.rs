//! Decode pool for server scenarios with concurrent decode requests.
//!
//! [`DecodePool`] tracks the number of active concurrent decodes and
//! automatically adapts threading strategy: parallel (rayon) when few
//! decodes are active, sequential when many are active. This prevents
//! the catastrophic tail latency that occurs when many parallel decodes
//! compete for the same rayon thread pool.
//!
//! # Example
//!
//! ```ignore
//! use zenjpeg::decode::{Decoder, DecodePool};
//!
//! // Built once at server startup
//! let decoder = Decoder::new()
//!     .output_format(PixelFormat::Rgb)
//!     .chroma_upsampling(ChromaUpsampling::NearestNeighbor);
//!
//! let pool = DecodePool::new();
//!
//! // Per-request (pool is &DecodePool, Send + Sync)
//! let result = decoder.request(&jpeg_data)
//!     .pool(&pool)
//!     .decode()?;
//! ```

use core::sync::atomic::{AtomicUsize, Ordering};

/// Shared concurrency tracker for decode operations.
///
/// Tracks the number of active concurrent decodes and selects between
/// parallel and sequential decoding based on a configurable threshold.
///
/// When `active_count <= parallel_threshold`, decodes use rayon for
/// lowest single-image latency. When above the threshold, decodes run
/// sequentially to maximize total system throughput and prevent tail
/// latency blowup.
///
/// Thread-safe: `Send + Sync`. Create once, share via `&DecodePool`.
pub struct DecodePool {
    active: AtomicUsize,
    parallel_threshold: usize,
}

impl DecodePool {
    /// Creates a new decode pool with the default parallel threshold (4).
    ///
    /// The default threshold of 4 was determined empirically: at 32 concurrent
    /// 2048x2048 decodes on a 16-core/32-thread system, adaptive-4 achieves
    /// the best combination of throughput (4689 MP/s) and tail latency
    /// (p95/p50 = 1.5x vs 8.6x for unbounded parallel).
    #[must_use]
    pub fn new() -> Self {
        Self {
            active: AtomicUsize::new(0),
            parallel_threshold: 4,
        }
    }

    /// Sets the parallel threshold.
    ///
    /// When the number of active decodes is less than or equal to this value,
    /// new decodes use parallel (rayon) for lowest latency. Above this value,
    /// new decodes run sequentially for predictable throughput.
    ///
    /// Guidelines:
    /// - `1-2`: Conservative, mostly sequential. Best tail latency.
    /// - `4` (default): Good balance of latency and throughput.
    /// - `physical_cores`: Aggressive parallel. Higher throughput, worse tail latency.
    #[must_use]
    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Returns the current number of active decodes.
    pub fn active_count(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Acquires a decode slot, returning a guard that releases on drop.
    ///
    /// The guard carries the recommended `num_threads` value based on
    /// the active count at acquisition time.
    pub(super) fn acquire(&self) -> PoolGuard<'_> {
        let prev = self.active.fetch_add(1, Ordering::Relaxed);
        let num_threads = if prev < self.parallel_threshold {
            0 // parallel (rayon)
        } else {
            1 // sequential
        };
        PoolGuard {
            pool: self,
            num_threads,
        }
    }
}

impl Default for DecodePool {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for DecodePool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodePool")
            .field("active", &self.active.load(Ordering::Relaxed))
            .field("parallel_threshold", &self.parallel_threshold)
            .finish()
    }
}

/// RAII guard that decrements the pool's active count on drop.
///
/// Carries the `num_threads` decision made at acquisition time.
pub(super) struct PoolGuard<'a> {
    pool: &'a DecodePool,
    /// `0` = parallel (rayon), `1` = sequential.
    pub(super) num_threads: usize,
}

impl Drop for PoolGuard<'_> {
    fn drop(&mut self) {
        self.pool.active.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_threshold() {
        let pool = DecodePool::new();
        assert_eq!(pool.parallel_threshold, 4);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn acquire_increments_and_drop_decrements() {
        let pool = DecodePool::new();
        assert_eq!(pool.active_count(), 0);

        let g1 = pool.acquire();
        assert_eq!(pool.active_count(), 1);
        assert_eq!(g1.num_threads, 0); // parallel (< threshold 4)

        let g2 = pool.acquire();
        assert_eq!(pool.active_count(), 2);
        assert_eq!(g2.num_threads, 0); // still parallel

        drop(g1);
        assert_eq!(pool.active_count(), 1);
        drop(g2);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn threshold_triggers_sequential() {
        let pool = DecodePool::new().parallel_threshold(2);

        let g1 = pool.acquire();
        assert_eq!(g1.num_threads, 0); // active was 0, < 2

        let g2 = pool.acquire();
        assert_eq!(g2.num_threads, 0); // active was 1, < 2

        let g3 = pool.acquire();
        assert_eq!(g3.num_threads, 1); // active was 2, >= 2 → sequential

        let g4 = pool.acquire();
        assert_eq!(g4.num_threads, 1); // active was 3, >= 2 → sequential

        drop(g3);
        drop(g4);
        drop(g1);

        // Back to 1 active, next should be parallel
        let g5 = pool.acquire();
        assert_eq!(g5.num_threads, 0); // active was 1, < 2

        drop(g2);
        drop(g5);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn custom_threshold() {
        let pool = DecodePool::new().parallel_threshold(1);

        let g1 = pool.acquire();
        assert_eq!(g1.num_threads, 0); // active was 0, < 1

        let g2 = pool.acquire();
        assert_eq!(g2.num_threads, 1); // active was 1, >= 1

        drop(g1);
        drop(g2);
    }
}
