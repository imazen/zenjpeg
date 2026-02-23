//! Per-job decode request binding data, stop token, and pool.
//!
//! [`DecodeRequest`] is the per-request type that binds JPEG data with
//! a cancellation token and optional [`DecodePool`] for adaptive threading.
//! Created via [`Decoder::request()`](super::DecodeConfig::request).

use super::DecodeConfig;
use super::config::DecodeResult;
use super::pool::DecodePool;
use super::scanline::ScanlineReader;
use crate::error::Result;
use enough::{Stop, Unstoppable};

/// A per-job decode request binding data with stop token and pool.
///
/// Created via [`Decoder::request()`](DecodeConfig::request). Holds references
/// to the shared config and data, plus optional pool for adaptive threading.
///
/// # Example
///
/// ```ignore
/// use zenjpeg::decode::{Decoder, DecodePool};
///
/// let decoder = Decoder::new();
/// let pool = DecodePool::new();
///
/// // Full-buffer decode with pool
/// let result = decoder.request(&jpeg_data)
///     .pool(&pool)
///     .stop(&cancel_token)
///     .decode()?;
///
/// // Streaming decode with pool
/// let reader = decoder.request(&jpeg_data)
///     .pool(&pool)
///     .scanline_reader()?;
/// ```
pub struct DecodeRequest<'a, S: Stop = Unstoppable> {
    pub(super) config: &'a DecodeConfig,
    pub(super) data: &'a [u8],
    pub(super) pool: Option<&'a DecodePool>,
    pub(super) stop: S,
}

impl<'a> DecodeRequest<'a, Unstoppable> {
    /// Creates a new decode request with default stop (Unstoppable).
    pub(super) fn new(config: &'a DecodeConfig, data: &'a [u8]) -> Self {
        Self {
            config,
            data,
            pool: None,
            stop: Unstoppable,
        }
    }
}

impl<'a, S: Stop> DecodeRequest<'a, S> {
    /// Attaches a [`DecodePool`] for adaptive threading under concurrent load.
    ///
    /// When a pool is attached, the decode automatically uses parallel mode
    /// when few decodes are active and sequential mode when many are active.
    /// This prevents tail latency blowup from rayon thread pool contention.
    ///
    /// Without a pool, uses the config's `num_threads` setting directly.
    #[must_use]
    pub fn pool(mut self, pool: &'a DecodePool) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Sets the cancellation token for this decode.
    ///
    /// The stop token is checked periodically during decoding. If it signals
    /// stop, the decode returns early with a cancellation error.
    #[must_use]
    pub fn stop<S2: Stop>(self, stop: S2) -> DecodeRequest<'a, S2> {
        DecodeRequest {
            config: self.config,
            data: self.data,
            pool: self.pool,
            stop,
        }
    }

    /// Decodes the JPEG into a full-buffer result.
    ///
    /// This is equivalent to [`Decoder::decode()`](DecodeConfig::decode) but
    /// with pool-aware adaptive threading.
    pub fn decode(self) -> Result<DecodeResult> {
        let guard = self.pool.map(DecodePool::acquire);
        let mut config = self.config.clone();
        if let Some(ref g) = guard {
            config.num_threads = g.num_threads;
        }
        config.decode(self.data, self.stop)
        // guard drops here, releasing pool slot
    }

    /// Creates a streaming scanline reader.
    ///
    /// This is equivalent to [`Decoder::scanline_reader()`](DecodeConfig::scanline_reader)
    /// but with pool-aware adaptive threading. The pool slot is held for the
    /// lifetime of the reader and released when the reader is dropped.
    pub fn scanline_reader(self) -> Result<ScanlineReader<'a>> {
        let guard = self.pool.map(DecodePool::acquire);
        let mut config = self.config.clone();
        if let Some(ref g) = guard {
            config.num_threads = g.num_threads;
        }
        let mut reader = config.scanline_reader(self.data)?;
        reader.pool_guard = guard;
        Ok(reader)
        // guard ownership moved to reader — released when reader drops
    }
}
