# JpegEncoder API Design

## Current API Surface (v0.4)

### Configuration (StreamingEncoderBuilder)

```rust
// Entry point - creates builder
JpegEncoder::new(width: u32, height: u32) -> StreamingEncoderBuilder

// Builder methods (all return Self for chaining)
.quality(impl Into<Quality>)     // 1-100 or Quality enum
.distance(f32)                   // Butteraugli distance (0.5-3.0)
.progressive(bool)               // Enable progressive JPEG
.subsampling(Subsampling)        // S444, S422, S420, S440
.pixel_format(PixelFormat)       // Rgb, Rgba, Gray, Rgb16, etc.
.mode(JpegMode)                  // Baseline, Progressive
.optimize_huffman(bool)          // Two-pass Huffman optimization
.chroma_downsampling(ChromaDownsampling)  // Box, GammaAwareIterative
.sharp_yuv(bool)                 // Alias for GammaAwareIterative
.restart_interval(u16)           // MCUs between restart markers
.custom_quant_matrices(CustomQuantMatrices)  // Custom quant tables
.use_xyb(bool)                   // XYB color space mode

// Feature-gated
.parallel(bool)                  // [parallel] Multi-threaded encoding
.hybrid_trellis(bool)            // [experimental-hybrid-trellis]
.hybrid_config(HybridConfig)     // [experimental-hybrid-trellis]
.aq_map(AQStrengthMap)           // [experimental-hybrid-trellis]

// Terminal methods
.start() -> Result<StreamingEncoder>           // Start streaming
.encode(data: &[u8]) -> Result<Vec<u8>>        // One-shot encode
.encode_all_with_stop(data, stop) -> Result<Vec<u8>>  // With cancellation
.estimate_memory_usage() -> usize              // Memory estimate
```

### Encoder State (StreamingEncoder)

```rust
// Inspection
.rows_pushed() -> usize          // Rows received so far
.bytes_per_row() -> usize        // Expected bytes per row
.height() -> usize               // Total image height
.strip_height() -> usize         // Internal strip size

// Push data
.push_row(&[u8]) -> Result<()>
.push_row_with_stop(&[u8], impl Stop) -> Result<()>
.push_rows(&[u8], num_rows) -> Result<()>
.push_rows_with_stop(&[u8], num_rows, impl Stop) -> Result<()>

// Direct YCbCr input (bypass RGB conversion)
.push_ycbcr_strip_f32(y, cb, cr, num_rows) -> Result<()>
.push_ycbcr_strip_f32_subsampled(y, cb, cr, num_rows) -> Result<()>

// Finalize
.finish() -> Result<Vec<u8>>
.finish_with_stop(impl Stop) -> Result<Vec<u8>>
```

### Convenience Functions

```rust
jpegli::encode_rgb(w, h, data, quality) -> Result<Vec<u8>>
jpegli::encode_rgba(w, h, data, quality) -> Result<Vec<u8>>
jpegli::encode_gray(w, h, data, quality) -> Result<Vec<u8>>
jpegli::decode(data) -> Result<DecodedImage>
jpegli::decode_f32(data) -> Result<DecodedImageF32>
jpegli::decode_to_format(data, format) -> Result<DecodedImage>
```

---

## Proposed API Improvements (v0.5)

### Goal: Separate Config from Encoder

Current problem: `JpegEncoder::new(w, h)` creates a builder that has dimensions baked in.
This prevents reusing configuration across different image sizes.

### New Design

```rust
/// Reusable encoding configuration (no dimensions)
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    quality: Quality,
    subsampling: Subsampling,
    pixel_format: PixelFormat,
    mode: JpegMode,
    optimize_huffman: bool,
    chroma_downsampling: ChromaDownsampling,
    restart_interval: u16,
    custom_quant_matrices: Option<CustomQuantMatrices>,
    use_xyb: bool,
    #[cfg(feature = "parallel")]
    parallel: bool,
}

impl EncoderConfig {
    /// Default config: Q85, 4:4:4, RGB, baseline, optimize on
    pub fn new() -> Self;

    // Builder methods (same as current, return &mut Self or Self)
    pub fn quality(&mut self, q: impl Into<Quality>) -> &mut Self;
    pub fn distance(&mut self, d: f32) -> &mut Self;
    pub fn progressive(&mut self, enable: bool) -> &mut Self;
    pub fn subsampling(&mut self, s: Subsampling) -> &mut Self;
    pub fn pixel_format(&mut self, f: PixelFormat) -> &mut Self;
    // ... etc

    /// Estimate memory for specific dimensions
    pub fn estimate_memory(&self, width: u32, height: u32) -> MemoryEstimate;

    /// Estimate output size range (min, typical, max)
    pub fn estimate_output_size(&self, width: u32, height: u32) -> OutputSizeEstimate;

    /// Create encoder for specific dimensions
    pub fn encoder(&self, width: u32, height: u32) -> Result<Encoder>;

    /// One-shot encode with this config
    pub fn encode(&self, width: u32, height: u32, data: &[u8]) -> Result<Vec<u8>>;

    /// One-shot encode with cancellation
    pub fn encode_with_stop(
        &self,
        width: u32,
        height: u32,
        data: &[u8],
        stop: impl Stop
    ) -> Result<Vec<u8>>;
}

/// Detailed memory breakdown
#[derive(Clone, Debug)]
pub struct MemoryEstimate {
    /// Peak memory during encoding
    pub peak_bytes: usize,
    /// Memory for input buffering (one strip)
    pub input_buffer: usize,
    /// Memory for DCT coefficient storage
    pub coefficient_storage: usize,
    /// Memory for internal working buffers
    pub working_buffers: usize,
    /// Estimated output buffer (before actual encoding)
    pub output_buffer_estimate: usize,
}

impl MemoryEstimate {
    /// Total including output buffer estimate
    pub fn total(&self) -> usize;

    /// Peak during encoding (excludes final output)
    pub fn encoding_peak(&self) -> usize;
}

/// Output size estimate
#[derive(Clone, Debug)]
pub struct OutputSizeEstimate {
    /// Minimum likely size (highly compressible content)
    pub min_bytes: usize,
    /// Typical size for photographic content
    pub typical_bytes: usize,
    /// Maximum likely size (incompressible content)
    pub max_bytes: usize,
    /// Bits per pixel estimate
    pub typical_bpp: f32,
}

/// Stateful encoder (created from config + dimensions)
pub struct Encoder {
    // ... internal state
}

impl Encoder {
    /// Create with default config
    pub fn new(width: u32, height: u32) -> Result<Self>;

    /// Create from config
    pub fn with_config(config: &EncoderConfig, width: u32, height: u32) -> Result<Self>;

    // Status
    pub fn rows_pushed(&self) -> usize;
    pub fn rows_remaining(&self) -> usize;
    pub fn bytes_per_row(&self) -> usize;
    pub fn progress(&self) -> f32;  // 0.0 - 1.0
    pub fn is_complete(&self) -> bool;

    // Memory tracking
    pub fn current_memory_usage(&self) -> usize;
    pub fn peak_memory_usage(&self) -> usize;

    // Push data
    pub fn push_row(&mut self, row: &[u8]) -> Result<()>;
    pub fn push_row_cancellable(&mut self, row: &[u8], stop: &impl Stop) -> Result<()>;
    pub fn push_rows(&mut self, data: &[u8], num_rows: usize) -> Result<()>;
    pub fn push_rows_cancellable(&mut self, data: &[u8], num_rows: usize, stop: &impl Stop) -> Result<()>;

    // Direct YCbCr (for transcoding pipelines)
    pub fn push_ycbcr_f32(&mut self, y: &[f32], cb: &[f32], cr: &[f32], rows: usize) -> Result<()>;

    // Finish
    pub fn finish(self) -> Result<Vec<u8>>;
    pub fn finish_cancellable(self, stop: impl Stop) -> Result<Vec<u8>>;

    // Write to existing buffer (zero-copy for proxy servers)
    pub fn finish_into(self, output: &mut Vec<u8>) -> Result<usize>;
    pub fn finish_into_cancellable(self, output: &mut Vec<u8>, stop: impl Stop) -> Result<usize>;
}
```

### Proxy Server Usage Pattern

```rust
use jpegli::{EncoderConfig, Subsampling};
use std::sync::Arc;

// Create shared config once at startup
let config = Arc::new(
    EncoderConfig::new()
        .quality(85)
        .subsampling(Subsampling::S420)
        .progressive(true)
        .optimize_huffman(true)
        .clone()
);

// Per-request handler
async fn handle_resize(
    config: Arc<EncoderConfig>,
    source: Image,
    target_width: u32,
    target_height: u32,
    cancel: CancellationToken,
) -> Result<Vec<u8>> {
    // Estimate memory before committing
    let estimate = config.estimate_memory(target_width, target_height);
    if estimate.peak_bytes > MAX_MEMORY_PER_REQUEST {
        return Err(Error::ImageTooLarge);
    }

    // Pre-allocate output buffer
    let output_estimate = config.estimate_output_size(target_width, target_height);
    let mut output = Vec::with_capacity(output_estimate.typical_bytes);

    // Create encoder
    let mut encoder = config.encoder(target_width, target_height)?;

    // Stream rows from resizer, checking cancellation
    let stop = cancel.as_stop();
    for row in source.resize_rows(target_width, target_height) {
        encoder.push_row_cancellable(&row, &stop)?;
    }

    // Finish into pre-allocated buffer
    let size = encoder.finish_into_cancellable(&mut output, stop)?;
    output.truncate(size);

    Ok(output)
}
```

### Memory Estimate Accuracy

Current `estimate_memory_usage()` is approximate. Proposed breakdown:

```rust
impl EncoderConfig {
    pub fn estimate_memory(&self, width: u32, height: u32) -> MemoryEstimate {
        let w = width as usize;
        let h = height as usize;
        let strip_h = self.subsampling.strip_height();
        let mcu = self.subsampling.mcu_size();

        // Pad to MCU boundaries
        let pw = (w + mcu - 1) / mcu * mcu;
        let ph = (h + mcu - 1) / mcu * mcu;

        // Block counts
        let y_blocks = (pw / 8) * (ph / 8);
        let c_blocks = self.subsampling.chroma_blocks(pw, ph);

        // Input buffer: one strip of RGB
        let input_buffer = w * strip_h * self.pixel_format.bytes_per_pixel();

        // Working buffers: f32 YCbCr planes for one strip
        let strip_f32 = pw * strip_h * 4 * 3;  // Y, Cb, Cr

        // Coefficient storage: all blocks as i16
        let coeff_storage = (y_blocks + 2 * c_blocks) * 64 * 2;  // i16 = 2 bytes

        // DCT working buffers (double-buffered f32)
        let dct_buffers = (pw / 8) * 2 * 256 * 2;  // 2 iMCU rows, f32

        // AQ map
        let aq_map = y_blocks * 4;  // f32 per block

        // Huffman optimization (if enabled)
        let huffman_buffers = if self.optimize_huffman {
            y_blocks * 2 + c_blocks * 4  // frequency counts
        } else {
            0
        };

        // Output estimate (quality-dependent)
        let bpp = self.quality.estimated_bpp();
        let output_estimate = (w * h) as f32 * bpp / 8.0;

        MemoryEstimate {
            peak_bytes: input_buffer + strip_f32 + coeff_storage + dct_buffers + aq_map,
            input_buffer,
            coefficient_storage: coeff_storage,
            working_buffers: strip_f32 + dct_buffers + aq_map + huffman_buffers,
            output_buffer_estimate: output_estimate as usize,
        }
    }
}
```

### Output Size Estimation

```rust
impl Quality {
    /// Estimated bits per pixel for this quality level
    pub fn estimated_bpp(&self) -> f32 {
        let d = self.to_distance();
        // Empirical formula from benchmarks
        match d {
            d if d < 0.5 => 4.5,   // Very high quality
            d if d < 1.0 => 3.0,   // High quality
            d if d < 1.5 => 2.0,   // Medium-high
            d if d < 2.0 => 1.5,   // Medium
            d if d < 3.0 => 1.0,   // Low
            _ => 0.7,             // Very low
        }
    }
}

impl EncoderConfig {
    pub fn estimate_output_size(&self, width: u32, height: u32) -> OutputSizeEstimate {
        let pixels = (width * height) as f32;
        let base_bpp = self.quality.estimated_bpp();

        // Adjust for subsampling
        let subsample_factor = match self.subsampling {
            Subsampling::S444 => 1.0,
            Subsampling::S422 => 0.85,
            Subsampling::S420 => 0.75,
            Subsampling::S440 => 0.85,
        };

        // Adjust for progressive (typically 3% smaller)
        let prog_factor = if self.mode == JpegMode::Progressive { 0.97 } else { 1.0 };

        let typical_bpp = base_bpp * subsample_factor * prog_factor;

        OutputSizeEstimate {
            min_bytes: (pixels * typical_bpp * 0.3 / 8.0) as usize,
            typical_bytes: (pixels * typical_bpp / 8.0) as usize,
            max_bytes: (pixels * typical_bpp * 2.0 / 8.0) as usize,
            typical_bpp,
        }
    }
}
```

---

## Migration Path

### v0.4 (Current)
```rust
let jpeg = JpegEncoder::new(800, 600)
    .quality(85)
    .progressive(true)
    .encode(&pixels)?;
```

### v0.5 (Proposed) - Same syntax still works
```rust
// Option 1: Existing syntax (backwards compatible)
let jpeg = JpegEncoder::new(800, 600)
    .quality(85)
    .progressive(true)
    .encode(&pixels)?;

// Option 2: Separate config (new)
let config = EncoderConfig::new()
    .quality(85)
    .progressive(true);

let jpeg = config.encode(800, 600, &pixels)?;

// Option 3: Reusable config across sizes
let small = config.encode(400, 300, &small_pixels)?;
let large = config.encode(1600, 1200, &large_pixels)?;
```

---

---

## Resource Estimation API (TODO)

### Requirements

Memory estimation depends on:
- **Input method**: streaming (row-by-row) vs one-shot vs YCbCr direct
- **Dimensions**: width × height
- **Config**: subsampling, optimize_huffman, parallel, etc.

### Proposed Structs

```rust
/// Pre-encode resource estimate
pub struct ResourceEstimate {
    /// Peak memory required (only public field)
    pub peak_bytes: usize,

    // Internal tracking (not exposed, used for validation)
    // - total_alloc_count: usize
    // - total_alloc_bytes: usize
    // - max_single_alloc: usize
}

impl ResourceEstimate {
    /// Estimated compute time in milliseconds for current architecture
    pub fn compute_cost_ms(&self) -> f32;
}

/// Input method affects memory profile
pub enum InputMethod {
    /// One-shot: entire image in memory
    OneShot,
    /// Streaming: row-by-row, lower peak memory
    Streaming,
    /// Direct YCbCr: pre-converted planes
    YCbCrDirect,
    /// Direct YCbCr with pre-subsampled chroma
    YCbCrSubsampled,
}

impl EncoderConfig {
    /// Estimate resources for specific dimensions and input method
    pub fn estimate_resources(
        &self,
        width: u32,
        height: u32,
        input_method: InputMethod,
    ) -> ResourceEstimate;
}
```

### Post-Encode Metrics

```rust
/// Actual resource usage after encoding completes
pub struct EncodeMetrics {
    /// Actual peak memory during encoding
    pub peak_bytes: usize,
    /// Total allocations made
    pub alloc_count: usize,
    /// Total bytes allocated (may exceed peak due to churn)
    pub total_alloc_bytes: usize,
    /// Wall-clock time spent encoding
    pub elapsed_ms: f32,
    /// Output size in bytes
    pub output_bytes: usize,
}

impl Encoder {
    /// Finish encoding and return metrics along with output
    pub fn finish_with_metrics(self) -> Result<(Vec<u8>, EncodeMetrics)>;

    /// Finish into existing buffer, return metrics
    pub fn finish_into_with_metrics(
        self,
        output: &mut Vec<u8>,
    ) -> Result<EncodeMetrics>;
}
```

### Compute Cost Estimation

```rust
impl ResourceEstimate {
    /// Estimated encode time based on:
    /// - Image dimensions (pixels)
    /// - Config (progressive 2x slower, parallel speedup)
    /// - Current CPU (detected at runtime)
    pub fn compute_cost_ms(&self) -> f32 {
        // Base: ~90 MP/s sequential, ~45 MP/s progressive
        // Adjusted for: parallel (1.4x for large), sharp_yuv (slower)
        // CPU detection: AVX2 vs SSE vs scalar fallback
    }
}
```

### Usage Pattern for Proxy Servers

```rust
async fn handle_resize(
    config: &EncoderConfig,
    width: u32,
    height: u32,
    pixels: &[u8],
    cancel: CancellationToken,
) -> Result<(Vec<u8>, EncodeMetrics)> {
    // Pre-flight check
    let estimate = config.estimate_resources(width, height, InputMethod::OneShot);

    if estimate.peak_bytes > MAX_MEMORY_PER_REQUEST {
        return Err(Error::ImageTooLarge);
    }

    if estimate.compute_cost_ms() > MAX_ENCODE_TIME_MS {
        return Err(Error::WouldTakeToLong);
    }

    // Encode
    let mut encoder = config.encoder(width, height)?;
    // ... push rows with cancellation ...

    // Get actual metrics for logging/billing
    let (jpeg, metrics) = encoder.finish_with_metrics()?;

    log::info!(
        "Encoded {}x{} in {:.1}ms, peak {}KB, output {}KB",
        width, height,
        metrics.elapsed_ms,
        metrics.peak_bytes / 1024,
        metrics.output_bytes / 1024,
    );

    Ok((jpeg, metrics))
}
```

---

## Implementation TODO

- [ ] Extract `EncoderConfig` as dimension-independent config
- [ ] Add `InputMethod` enum
- [ ] Implement `estimate_resources()` with accurate memory modeling
- [ ] Add `compute_cost_ms()` with CPU detection
- [ ] Add allocation tracking (behind feature flag for zero overhead)
- [ ] Add `EncodeMetrics` returned from `finish_with_metrics()`
- [ ] Add `finish_into()` for zero-copy output to existing buffer
- [ ] Benchmark to calibrate `compute_cost_ms()` estimates

---

## Open Questions

1. Should `EncoderConfig` be `Clone + Send + Sync` for easy sharing?
2. Should we add `encode_to_writer(impl Write)` for zero-copy streaming output?
3. Should memory/timing tracking be opt-in via feature flag (slight overhead)?
4. Should we expose strip-level progress callbacks for large images?
5. How to handle allocation tracking without global allocator hooks?
