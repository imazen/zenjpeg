# Streaming HDR Encoder/Decoder Design

## Overview

This document describes a redesigned streaming HDR architecture with:
1. **Pluggable tonemappers** with variable lag (0 rows, N rows, or full image)
2. **Proper color space handling** (primaries + transfer functions)
3. **Multiple output formats** (f32, i16, u8)
4. **BT.601 constraint awareness** (JPEG always uses fixed YCbCr matrix)

## API Requirements (See CLAUDE.md)

**All pixel APIs MUST follow these rules:**

- **Type-safe pixels**: Use `rgb::RGB<T>` or `rgb::RGBA<T>`, NEVER raw `&[u8]`
- **Stride ALWAYS required**: Via `imgref::ImgRef` or explicit `stride_pixels` parameter
- **Caller owns buffers**: Write into caller-provided `&mut [T]`, don't allocate
- **Fallible allocation**: Use `try_reserve()`, return `Result` on OOM
- **16-32 bit precision**: Internal processing MUST be f32 or i32, NEVER u8 arithmetic

## Color Pipeline Architecture

### The BT.601 Constraint

**CRITICAL**: JPEG encoding ALWAYS uses BT.601 RGB→YCbCr, regardless of input gamut.

```
BT.601 RGB→YCbCr matrix (always used):
Y  = 0.299R + 0.587G + 0.114B
Cb = -0.169R - 0.331G + 0.500B + 128
Cr = 0.500R - 0.419G - 0.081B + 128
```

The ICC profile embedded in the JPEG tells decoders what gamut the *decoded* RGB is in.
The YCbCr encoding math doesn't change - only the interpretation of the final RGB.

### Full Encoding Pipeline

```
HDR Input (any colorspace)
    │
    ▼ [1] apply_eotf(input.transfer)
Linear RGB in source primaries
    │
    ▼ [2] convert_primaries(src → working)  [if needed]
Linear RGB in working primaries
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  │
Tonemapper (with lag)            Buffer HDR
    │                              (for gain)
    │                                  │
    ▼                                  │
SDR Linear RGB                         │
    │                                  │
    ▼ [3] compute_gain(hdr/sdr) ◄──────┘
    │
    ├──────────────────────────┐
    ▼                          ▼
SDR path                   Gain Map path
    │                          │
    ▼ [4] convert_primaries    │
    │     (working → output)   │
    │                          │
    ▼ [5] apply_oetf           │
    │     (output.transfer)    │
    │                          │
    ▼ [6] BT.601 RGB→YCbCr     ▼
    │     (ALWAYS!)        Encode grayscale
    │                          │
    ▼                          │
JPEG encode                    │
    │                          │
    ▼ [7] Embed ICC profile    │
    │     (output primaries)   │
    │                          │
    └──────────┬───────────────┘
               ▼
        Assemble UltraHDR
        (SDR + XMP + MPF + Gain Map)
```

### Color Space Types

```rust
/// Color primaries (gamut definition)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ColorPrimaries {
    #[default]
    Srgb,       // sRGB / BT.709 (identical primaries)
    DisplayP3,  // DCI-P3 with D65 white point
    Rec2020,    // ITU-R BT.2020 (very wide for HDR)
    AdobeRgb,   // Adobe RGB 1998
}

/// Transfer function (EOTF for decode, OETF for encode)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TransferFunction {
    Linear,     // Gamma 1.0 (scene-referred linear light)
    #[default]
    Srgb,       // sRGB piecewise (~2.2 with linear toe)
    Pq,         // SMPTE ST 2084 (HDR10, up to 10000 nits)
    Hlg,        // ITU-R BT.2100 HLG (broadcast HDR)
    Gamma22,    // Pure 2.2 power function
    Gamma24,    // Pure 2.4 power function (BT.1886)
}

/// Full color space = primaries + transfer
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorSpace {
    pub primaries: ColorPrimaries,
    pub transfer: TransferFunction,
}

impl ColorSpace {
    pub const SRGB: Self = Self { primaries: ColorPrimaries::Srgb, transfer: TransferFunction::Srgb };
    pub const LINEAR_SRGB: Self = Self { primaries: ColorPrimaries::Srgb, transfer: TransferFunction::Linear };
    pub const DISPLAY_P3: Self = Self { primaries: ColorPrimaries::DisplayP3, transfer: TransferFunction::Srgb };
    pub const LINEAR_P3: Self = Self { primaries: ColorPrimaries::DisplayP3, transfer: TransferFunction::Linear };
    pub const REC2020_PQ: Self = Self { primaries: ColorPrimaries::Rec2020, transfer: TransferFunction::Pq };
    pub const REC2020_HLG: Self = Self { primaries: ColorPrimaries::Rec2020, transfer: TransferFunction::Hlg };
    pub const LINEAR_REC2020: Self = Self { primaries: ColorPrimaries::Rec2020, transfer: TransferFunction::Linear };
}
```

## Tonemapper Trait with Variable Lag

### Lag Specification

```rust
/// How much buffering a tonemapper needs before producing output
#[derive(Clone, Copy, Debug, Default)]
pub enum TonemapperLag {
    /// Zero lag - output available immediately per row
    /// Examples: Reinhard, ACES filmic, linear clamp
    #[default]
    Zero,

    /// Fixed row count lag (lookahead window)
    /// Examples: Local histogram-based, bilateral filter
    Rows(usize),

    /// Percentage of image height (0.0 - 1.0)
    /// Examples: Adaptive local contrast
    Percent(f32),

    /// Full image buffering required
    /// Examples: Global histogram, percentile-based exposure
    Full,
}

impl TonemapperLag {
    pub fn to_rows(&self, height: usize) -> usize {
        match *self {
            TonemapperLag::Zero => 0,
            TonemapperLag::Rows(n) => n,
            TonemapperLag::Percent(p) => (height as f32 * p.clamp(0.0, 1.0)) as usize,
            TonemapperLag::Full => height,
        }
    }
}
```

### Streaming Tonemapper Trait

```rust
/// A tonemapper that operates on streaming row data with configurable lag.
///
/// # Lifetime Contract
/// 1. `init()` called once at start with image dimensions
/// 2. `process_rows()` called repeatedly with input HDR rows
/// 3. `flush()` called after all input to drain buffered rows
///
/// # Color Space Contract
/// - Input: Linear RGB in `input_primaries()` gamut
/// - Output: Linear RGB in `output_primaries()` gamut
/// - The encoder handles EOTF/OETF and gamut conversions around the tonemapper
pub trait StreamingTonemapper: Send {
    /// Lag before first output row is available
    fn lag(&self) -> TonemapperLag;

    /// Required input color primaries (always linear transfer)
    fn input_primaries(&self) -> ColorPrimaries;

    /// Output color primaries (always linear transfer)
    fn output_primaries(&self) -> ColorPrimaries;

    /// Initialize for a specific image size
    fn init(&mut self, width: usize, height: usize);

    /// Process HDR rows and produce SDR rows.
    ///
    /// # Arguments
    /// - `hdr_rows`: Input linear RGB f32, with stride (use imgref or explicit stride)
    /// - `hdr_stride`: Stride in pixels (not bytes!) for input buffer
    /// - `sdr_out`: Output buffer, caller-provided, same layout as input
    /// - `sdr_stride`: Stride in pixels for output buffer
    /// - `width`: Actual pixel width (may be less than stride)
    /// - `row_index`: Index of first input row (for position-aware algorithms)
    /// - `count`: Number of input rows
    ///
    /// # Returns
    /// `(consumed, produced)` - rows consumed from input, rows written to output
    ///
    /// For zero-lag: consumed == produced == count
    /// For lagged: may buffer input, output may be delayed
    fn process_rows(
        &mut self,
        hdr_rows: &[rgb::RGB<f32>],
        hdr_stride: usize,
        sdr_out: &mut [rgb::RGB<f32>],
        sdr_stride: usize,
        width: usize,
        row_index: usize,
        count: usize,
    ) -> (usize, usize);

    /// Flush remaining buffered rows after all input is processed.
    ///
    /// Caller provides output buffer with stride. May need multiple calls until returns 0.
    fn flush(&mut self, sdr_out: &mut [rgb::RGB<f32>], stride: usize, width: usize) -> usize;

    /// Optional metadata learned during tonemapping (peak luminance, etc.)
    fn metadata(&self) -> Option<TonemapperMetadata> { None }
}

/// Metadata that a tonemapper can provide after processing
#[derive(Clone, Debug, Default)]
pub struct TonemapperMetadata {
    pub hdr_peak_luminance: Option<f32>,  // nits
    pub hdr_avg_luminance: Option<f32>,   // nits
    pub dynamic_range_stops: Option<f32>,
    pub custom: Option<Vec<u8>>,
}
```

### Simple Pixel Tonemapper Adapter

For zero-lag per-pixel tonemappers:

```rust
/// Per-pixel tonemapper (zero lag, no buffering)
pub trait PixelTonemapper: Send {
    /// Tonemap a single pixel. Input/output are linear RGB.
    fn tonemap(&self, hdr: [f32; 3]) -> [f32; 3];
}

/// Adapter to use PixelTonemapper as StreamingTonemapper
pub struct PixelTonemapperAdapter<T: PixelTonemapper> {
    inner: T,
    primaries: ColorPrimaries,
    width: usize,
}

impl<T: PixelTonemapper> StreamingTonemapper for PixelTonemapperAdapter<T> {
    fn lag(&self) -> TonemapperLag { TonemapperLag::Zero }
    fn input_primaries(&self) -> ColorPrimaries { self.primaries }
    fn output_primaries(&self) -> ColorPrimaries { self.primaries }

    fn init(&mut self, width: usize, _height: usize) {
        self.width = width;
    }

    fn process_rows(
        &mut self,
        hdr_rows: &[rgb::RGB<f32>],
        hdr_stride: usize,
        sdr_out: &mut [rgb::RGB<f32>],
        sdr_stride: usize,
        width: usize,
        _row_index: usize,
        count: usize,
    ) -> (usize, usize) {
        for row in 0..count {
            let hdr_row_start = row * hdr_stride;
            let sdr_row_start = row * sdr_stride;
            for x in 0..width {
                let hdr = hdr_rows[hdr_row_start + x];
                let sdr = self.inner.tonemap([hdr.r, hdr.g, hdr.b]);
                sdr_out[sdr_row_start + x] = rgb::RGB { r: sdr[0], g: sdr[1], b: sdr[2] };
            }
        }
        (count, count)
    }

    fn flush(&mut self, _sdr_out: &mut [rgb::RGB<f32>], _stride: usize, _width: usize) -> usize { 0 }
}
```

### Example Tonemappers

```rust
/// Simple Reinhard tonemapper (zero lag)
pub struct ReinhardTonemapper {
    white_point: f32,  // Luminance that maps to 1.0
}

impl PixelTonemapper for ReinhardTonemapper {
    fn tonemap(&self, hdr: [f32; 3]) -> [f32; 3] {
        let wp2 = self.white_point * self.white_point;
        hdr.map(|c| c * (1.0 + c / wp2) / (1.0 + c))
    }
}

/// Local histogram tonemapper (N-row lag for lookahead)
pub struct LocalHistogramTonemapper {
    lookahead_rows: usize,
    // ... histogram state, ring buffers
}

impl StreamingTonemapper for LocalHistogramTonemapper {
    fn lag(&self) -> TonemapperLag {
        TonemapperLag::Rows(self.lookahead_rows)
    }
    // ... implementation with row buffering
}

/// Global auto-exposure tonemapper (full-image lag)
pub struct GlobalAutoExposure {
    target_middle_gray: f32,
    // ... accumulated stats
}

impl StreamingTonemapper for GlobalAutoExposure {
    fn lag(&self) -> TonemapperLag { TonemapperLag::Full }

    fn process_rows(&mut self, hdr: &[f32], _sdr: &mut [f32], ...) -> (usize, usize) {
        // First pass: accumulate statistics, don't output yet
        self.accumulate_stats(hdr);
        (count, 0)  // Consume input, produce nothing
    }

    fn flush(&mut self, sdr_out: &mut [f32]) -> usize {
        // Now we know the exposure, emit all buffered rows
        // ...
    }
}
```

## Streaming HDR Encoder

### Configuration

```rust
/// Configuration for streaming HDR encoding
#[derive(Clone, Debug)]
pub struct StreamingHdrConfig {
    /// Input color space (how to interpret input data)
    pub input_colorspace: ColorSpace,

    /// Working primaries for tonemapping (typically Rec.2020 for wide gamut)
    pub working_primaries: ColorPrimaries,

    /// Output SDR color space (embedded as ICC profile)
    pub output_colorspace: ColorSpace,

    /// Gain map computation settings
    pub gainmap_config: GainMapConfig,

    /// JPEG quality for SDR base image
    pub sdr_quality: f32,

    /// JPEG quality for gain map
    pub gainmap_quality: f32,
}

impl Default for StreamingHdrConfig {
    fn default() -> Self {
        Self {
            input_colorspace: ColorSpace::LINEAR_REC2020,
            working_primaries: ColorPrimaries::Rec2020,
            output_colorspace: ColorSpace::SRGB,
            gainmap_config: GainMapConfig::default(),
            sdr_quality: 85.0,
            gainmap_quality: 75.0,
        }
    }
}

impl StreamingHdrConfig {
    /// P3 HDR input → sRGB SDR output
    pub fn p3_to_srgb() -> Self {
        Self {
            input_colorspace: ColorSpace::LINEAR_P3,
            working_primaries: ColorPrimaries::DisplayP3,
            output_colorspace: ColorSpace::SRGB,
            ..Default::default()
        }
    }

    /// P3 HDR input → P3 SDR output (wide gamut SDR)
    pub fn p3_to_p3() -> Self {
        Self {
            input_colorspace: ColorSpace::LINEAR_P3,
            working_primaries: ColorPrimaries::DisplayP3,
            output_colorspace: ColorSpace::DISPLAY_P3,
            ..Default::default()
        }
    }

    /// Rec.2020 PQ input → sRGB SDR output
    pub fn rec2020_pq_to_srgb() -> Self {
        Self {
            input_colorspace: ColorSpace::REC2020_PQ,
            working_primaries: ColorPrimaries::Rec2020,
            output_colorspace: ColorSpace::SRGB,
            ..Default::default()
        }
    }
}
```

### Encoder State Machine

```rust
/// Streaming UltraHDR encoder with pluggable tonemapper
pub struct StreamingHdrEncoder<T: StreamingTonemapper> {
    config: StreamingHdrConfig,
    width: usize,
    height: usize,

    // Tonemapper
    tonemapper: T,
    lag_rows: usize,

    // Ring buffers for lag management
    hdr_ring: RingBuffer<f32>,    // Buffer HDR until SDR ready

    // Position tracking
    hdr_rows_in: usize,
    sdr_rows_out: usize,
    rows_encoded: usize,

    // Color conversion matrices (precomputed)
    input_to_working: Option<Matrix3x3>,
    working_to_output: Option<Matrix3x3>,

    // JPEG encoders (created lazily)
    sdr_encoder: Option<StreamingJpegEncoder>,
    gainmap_encoder: Option<StreamingJpegEncoder>,

    // Gain map state
    gainmap_computer: Option<RowEncoder>,
}

impl<T: StreamingTonemapper> StreamingHdrEncoder<T> {
    pub fn new(
        width: usize,
        height: usize,
        config: StreamingHdrConfig,
        tonemapper: T,
    ) -> Result<Self> {
        let mut tm = tonemapper;
        tm.init(width, height);

        let lag_rows = tm.lag().to_rows(height);

        // Verify tonemapper color space compatibility
        if config.working_primaries != tm.input_primaries() {
            // Need to convert input → working → tonemapper input
            // (or require they match)
        }

        // Precompute color matrices
        let input_to_working = compute_gamut_matrix(
            config.input_colorspace.primaries,
            config.working_primaries,
        );
        let working_to_output = compute_gamut_matrix(
            tm.output_primaries(),
            config.output_colorspace.primaries,
        );

        Ok(Self {
            config,
            width,
            height,
            tonemapper: tm,
            lag_rows,
            hdr_ring: RingBuffer::new(width * 3, lag_rows + 16),
            hdr_rows_in: 0,
            sdr_rows_out: 0,
            rows_encoded: 0,
            input_to_working,
            working_to_output,
            sdr_encoder: None,
            gainmap_encoder: None,
            gainmap_computer: None,
        })
    }

    /// Push HDR rows into the encoder.
    ///
    /// # Arguments
    /// - `hdr_input`: HDR pixel data with stride, type encodes format
    /// - `stride`: Stride in PIXELS (not bytes!)
    /// - `width`: Actual pixel width (may be less than stride)
    /// - `count`: Number of rows
    ///
    /// Input transfer function depends on `config.input_colorspace.transfer`:
    /// - Linear: f32 linear light values
    /// - Srgb: f32 gamma-encoded [0,1]
    /// - Pq: f32 PQ-encoded [0,1]
    /// - Hlg: f32 HLG-encoded [0,1]
    ///
    /// Returns number of rows encoded to JPEG (may be delayed due to lag).
    pub fn push_rows(
        &mut self,
        hdr_input: &[rgb::RGB<f32>],
        stride: usize,
        width: usize,
        count: usize,
    ) -> Result<usize> {
        // [1] Linearize input if needed
        let linear = self.linearize_input(hdr_input);

        // [2] Convert to working primaries
        let working = self.to_working_primaries(&linear);

        // Store HDR in ring buffer (needed for gain map after SDR ready)
        self.hdr_ring.push(&working, count);
        self.hdr_rows_in += count;

        // [3] Process through tonemapper
        let mut sdr_linear = vec![0.0f32; working.len()];
        let (consumed, produced) = self.tonemapper.process_rows(
            &working,
            &mut sdr_linear,
            self.hdr_rows_in - count,
            count,
        );

        // Process available rows (where we have both HDR and SDR)
        let mut total_encoded = 0;
        if produced > 0 {
            total_encoded = self.encode_available_rows(&sdr_linear[..produced * self.width * 3], produced)?;
        }

        Ok(total_encoded)
    }

    fn encode_available_rows(&mut self, sdr_linear: &[f32], count: usize) -> Result<usize> {
        // Pop corresponding HDR rows from ring buffer
        let hdr_linear = self.hdr_ring.pop(count);

        // Compute gain map: gain = HDR / SDR (both in linear working primaries)
        let gain = self.compute_gain(&hdr_linear, sdr_linear, count);

        // [4] Convert SDR to output primaries
        let sdr_output = self.to_output_primaries(sdr_linear);

        // [5] Apply output OETF (e.g., sRGB gamma)
        let sdr_gamma = self.apply_oetf(&sdr_output);

        // [6] Encode to JPEG (BT.601 RGB→YCbCr applied internally)
        self.encode_sdr_rows(&sdr_gamma, count)?;
        self.encode_gain_rows(&gain, count)?;

        self.rows_encoded += count;
        Ok(count)
    }

    /// Finish encoding and return complete UltraHDR JPEG
    pub fn finish(mut self) -> Result<Vec<u8>> {
        // Flush tonemapper
        loop {
            let mut sdr_buf = vec![0.0f32; self.width * 3 * 64];
            let flushed = self.tonemapper.flush(&mut sdr_buf);
            if flushed == 0 { break; }
            self.encode_available_rows(&sdr_buf[..flushed * self.width * 3], flushed)?;
        }

        // Finish JPEG encoders
        let sdr_jpeg = self.sdr_encoder.take().unwrap().finish()?;
        let gain_jpeg = self.gainmap_encoder.take().unwrap().finish()?;

        // [7] Assemble with ICC profile for output primaries
        let icc = self.get_output_icc_profile();
        let metadata = self.build_gainmap_metadata();

        assemble_ultrahdr(sdr_jpeg, gain_jpeg, metadata, icc)
    }

    fn linearize_input(&self, data: &[f32]) -> Vec<f32> {
        match self.config.input_colorspace.transfer {
            TransferFunction::Linear => data.to_vec(),
            TransferFunction::Srgb => data.iter().map(|&v| srgb_eotf(v)).collect(),
            TransferFunction::Pq => data.iter().map(|&v| pq_eotf(v)).collect(),
            TransferFunction::Hlg => data.iter().map(|&v| hlg_eotf(v)).collect(),
            TransferFunction::Gamma22 => data.iter().map(|&v| v.powf(2.2)).collect(),
            TransferFunction::Gamma24 => data.iter().map(|&v| v.powf(2.4)).collect(),
        }
    }

    fn apply_oetf(&self, linear: &[f32]) -> Vec<f32> {
        match self.config.output_colorspace.transfer {
            TransferFunction::Linear => linear.to_vec(),
            TransferFunction::Srgb => linear.iter().map(|&v| srgb_oetf(v)).collect(),
            TransferFunction::Pq => linear.iter().map(|&v| pq_oetf(v)).collect(),
            TransferFunction::Hlg => linear.iter().map(|&v| hlg_oetf(v)).collect(),
            TransferFunction::Gamma22 => linear.iter().map(|&v| v.powf(1.0/2.2)).collect(),
            TransferFunction::Gamma24 => linear.iter().map(|&v| v.powf(1.0/2.4)).collect(),
        }
    }

    fn get_output_icc_profile(&self) -> Option<Vec<u8>> {
        match self.config.output_colorspace.primaries {
            ColorPrimaries::Srgb => None,  // sRGB is assumed, no profile needed
            ColorPrimaries::DisplayP3 => Some(display_p3_icc_profile()),
            ColorPrimaries::Rec2020 => Some(rec2020_icc_profile()),
            ColorPrimaries::AdobeRgb => Some(adobe_rgb_icc_profile()),
        }
    }
}
```

## Streaming HDR Decoder

### Configuration

```rust
/// Configuration for streaming HDR decode
#[derive(Clone, Debug)]
pub struct StreamingHdrDecoderConfig {
    /// Output color space for reconstructed HDR
    pub output_colorspace: ColorSpace,

    /// Output pixel format
    pub output_format: HdrPixelFormat,

    /// Display boost factor (1.0 = SDR, 4.0 = typical HDR, 8.0 = high-end)
    pub display_boost: f32,

    /// Also produce SDR output (for dual-output workflows)
    pub also_output_sdr: bool,

    /// Memory strategy for gain map
    pub gain_map_memory: GainMapMemory,
}

/// Output pixel format for HDR
#[derive(Clone, Copy, Debug, Default)]
pub enum HdrPixelFormat {
    /// Linear light f32 RGB/RGBA [0, peak_nits]
    #[default]
    LinearF32,
    LinearF32A,

    /// PQ-encoded f32 RGB [0, 1]
    PqF32,

    /// HLG-encoded f32 RGB [0, 1]
    HlgF32,

    /// Linear light i16 RGB (1.0 = 10000, for 0.0001 precision)
    LinearI16,

    /// Half-precision float
    LinearF16,
}
```

### Decoder

```rust
/// Streaming UltraHDR decoder
pub struct StreamingHdrDecoder<'a> {
    config: StreamingHdrDecoderConfig,

    // Base JPEG decoder
    sdr_reader: ScanlineReader<'a>,

    // Gain map state
    gain_map: Option<GainMapState>,
    metadata: Option<GainMapMetadata>,

    // Color conversion
    sdr_primaries: ColorPrimaries,  // From ICC profile or assume sRGB
    output_matrix: Option<Matrix3x3>,

    // Position
    width: usize,
    height: usize,
    current_row: usize,
}

impl<'a> StreamingHdrDecoder<'a> {
    /// Read rows into caller-provided output buffers.
    ///
    /// # Arguments
    /// - `hdr_out`: Caller's HDR buffer, format per `config.output_format`
    /// - `hdr_stride`: Stride in PIXELS for HDR output
    /// - `sdr_out`: Optional caller's SDR buffer (RGB8)
    /// - `sdr_stride`: Stride in PIXELS for SDR output
    /// - `width`: Actual pixel width (from decoder, may be less than stride)
    /// - `max_rows`: Maximum rows to read
    ///
    /// Returns actual rows read (may be less at end of image).
    pub fn read_rows(
        &mut self,
        hdr_out: &mut [rgb::RGB<f32>],
        hdr_stride: usize,
        sdr_out: Option<&mut [rgb::RGB<u8>]>,
        sdr_stride: usize,
        max_rows: usize,
    ) -> Result<usize> {
        let actual = max_rows.min(self.height - self.current_row);

        // Read SDR rows into caller's buffer or temp buffer
        // (implementation handles stride internally)

        // Reconstruct HDR if gain map available
        if let Some(ref gain) = self.gain_map {
            self.reconstruct_hdr(&sdr_buf, actual, hdr_out, gain)?;
        } else {
            // No gain map - convert SDR to HDR format
            self.sdr_to_hdr_fallback(&sdr_buf, actual, hdr_out);
        }

        self.current_row += actual;
        Ok(actual)
    }

    fn reconstruct_hdr(
        &self,
        sdr: &[u8],
        rows: usize,
        hdr_out: &mut [f32],
        gain: &GainMapState,
    ) -> Result<()> {
        let meta = self.metadata.as_ref().unwrap();

        for row in 0..rows {
            for x in 0..self.width {
                let sdr_idx = (row * self.width + x) * 3;

                // SDR → linear (apply sRGB EOTF)
                let sdr_r = srgb_eotf(sdr[sdr_idx] as f32 / 255.0);
                let sdr_g = srgb_eotf(sdr[sdr_idx + 1] as f32 / 255.0);
                let sdr_b = srgb_eotf(sdr[sdr_idx + 2] as f32 / 255.0);

                // Get gain value (interpolated if gain map is smaller)
                let gain_val = gain.sample(x, self.current_row + row, self.width, self.height);

                // Apply gain: HDR = SDR * 2^(gain * boost)
                let boost = self.config.display_boost;
                let multiplier = 2.0f32.powf(gain_val * boost);

                let hdr_r = sdr_r * multiplier;
                let hdr_g = sdr_g * multiplier;
                let hdr_b = sdr_b * multiplier;

                // Convert to output primaries if needed
                let (out_r, out_g, out_b) = if let Some(ref m) = self.output_matrix {
                    m.transform([hdr_r, hdr_g, hdr_b])
                } else {
                    (hdr_r, hdr_g, hdr_b)
                };

                // Apply output transfer function
                self.write_output_pixel(hdr_out, row, x, out_r, out_g, out_b);
            }
        }
        Ok(())
    }

    fn write_output_pixel(&self, out: &mut [f32], row: usize, x: usize, r: f32, g: f32, b: f32) {
        match self.config.output_format {
            HdrPixelFormat::LinearF32 => {
                let idx = (row * self.width + x) * 3;
                out[idx] = r;
                out[idx + 1] = g;
                out[idx + 2] = b;
            }
            HdrPixelFormat::LinearF32A => {
                let idx = (row * self.width + x) * 4;
                out[idx] = r;
                out[idx + 1] = g;
                out[idx + 2] = b;
                out[idx + 3] = 1.0;
            }
            HdrPixelFormat::PqF32 => {
                let idx = (row * self.width + x) * 3;
                out[idx] = pq_oetf(r);
                out[idx + 1] = pq_oetf(g);
                out[idx + 2] = pq_oetf(b);
            }
            // ... other formats
        }
    }
}
```

## Color Space Conversion Details

### Gamut Conversion Matrices

All conversions go through XYZ as intermediate:

```rust
// Primaries → XYZ matrices (D65 white point)
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

const P3_TO_XYZ: [[f32; 3]; 3] = [
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
];

const REC2020_TO_XYZ: [[f32; 3]; 3] = [
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
];

fn compute_gamut_matrix(from: ColorPrimaries, to: ColorPrimaries) -> Option<Matrix3x3> {
    if from == to { return None; }

    let from_to_xyz = match from {
        ColorPrimaries::Srgb => SRGB_TO_XYZ,
        ColorPrimaries::DisplayP3 => P3_TO_XYZ,
        ColorPrimaries::Rec2020 => REC2020_TO_XYZ,
        // ...
    };

    let xyz_to_to = match to {
        ColorPrimaries::Srgb => XYZ_TO_SRGB,
        ColorPrimaries::DisplayP3 => XYZ_TO_P3,
        ColorPrimaries::Rec2020 => XYZ_TO_REC2020,
        // ...
    };

    Some(matrix_multiply(xyz_to_to, from_to_xyz))
}
```

### Transfer Functions

```rust
// sRGB (IEC 61966-2-1)
fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 }
    else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn srgb_oetf(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 }
    else { 1.055 * v.powf(1.0/2.4) - 0.055 }
}

// PQ (SMPTE ST 2084) - assumes 10000 nit peak
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

fn pq_eotf(v: f32) -> f32 {
    let vp = v.max(0.0).powf(1.0 / PQ_M2);
    let num = (vp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * vp;
    10000.0 * (num / den).powf(1.0 / PQ_M1)
}

fn pq_oetf(nits: f32) -> f32 {
    let y = (nits / 10000.0).max(0.0);
    let yp = y.powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * yp) / (1.0 + PQ_C3 * yp)).powf(PQ_M2)
}

// HLG (ITU-R BT.2100)
fn hlg_eotf(v: f32) -> f32 {
    if v <= 0.5 { (v * v) / 3.0 }
    else { (((v - 0.55991073) / 0.17883277).exp() + 0.28466892) / 12.0 }
}

fn hlg_oetf(linear: f32) -> f32 {
    let l = linear.max(0.0);
    if l <= 1.0/12.0 { (3.0 * l).sqrt() }
    else { 0.17883277 * (12.0 * l - 0.28466892).ln() + 0.55991073 }
}
```

## i16 Output Format

For memory-constrained scenarios, support i16 output:

```rust
/// i16 linear format: value of 10000 = 1.0 linear light
/// Range: [-32768, 32767] → [-3.2768, 3.2767] linear
/// Precision: 0.0001 linear light units
const I16_SCALE: f32 = 10000.0;

fn f32_to_linear_i16(v: f32) -> i16 {
    (v * I16_SCALE).clamp(-32768.0, 32767.0) as i16
}

fn linear_i16_to_f32(v: i16) -> f32 {
    v as f32 / I16_SCALE
}
```

## Implementation Plan

1. **Phase 1: Color Space Infrastructure**
   - Add `ColorPrimaries`, `TransferFunction`, `ColorSpace` types
   - Implement gamut conversion matrices
   - Implement all EOTF/OETF functions

2. **Phase 2: Tonemapper Trait**
   - Define `StreamingTonemapper` trait
   - Implement `PixelTonemapperAdapter`
   - Port existing tonemapper to new trait

3. **Phase 3: Streaming Encoder**
   - Implement `StreamingHdrEncoder` with lag management
   - Add ring buffer for HDR row storage
   - Integrate gain map computation

4. **Phase 4: Streaming Decoder Enhancement**
   - Update `StreamingHdrDecoder` with color space awareness
   - Add i16 output format support
   - Add output colorspace conversion

5. **Phase 5: Testing**
   - Color space roundtrip tests
   - Tonemapper lag tests (0, N, Full)
   - Cross-decoder compatibility tests
