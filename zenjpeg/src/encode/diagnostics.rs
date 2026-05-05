//! Per-block encode diagnostics — UNSTABLE.
//!
//! Gated behind the `__diagnostics` cargo feature. Captures the per-block
//! state encoders consume (forward-DCT coefficients, post-quant levels,
//! AQ multiplier, entropy bits) plus image-level state (resolved config,
//! AQ field, scan script). Designed for visualizer / calibration tooling
//! that wants to drive interactive quant-table editing live in WASM.
//!
//! All public types in this module are `#[doc(hidden)]`. The shape of
//! the structs is intentionally evolvable — patch versions may add,
//! remove, or rename fields without bumping the major.
//!
//! # Coefficient ordering
//!
//! Two orderings appear in this module and they are NOT the same:
//!
//! - [`BlockDiagnostics::coef_pre_quant`] is in **natural row-major**
//!   order: index `i = row * 8 + col` with `row, col in [0, 7]`. This
//!   is the layout produced by the forward DCT before any zigzag
//!   reordering.
//! - [`BlockDiagnostics::coef_levels`] is in **JPEG zigzag** order:
//!   index `k` corresponds to natural position
//!   `JPEG_ZIGZAG_ORDER[k]`. This is what the Huffman entropy coder
//!   reads — DC at index 0, then a frequency-distance walk through the
//!   AC band.
//!
//! Visualizers that draw 8×8 heatmaps will usually want to plot in
//! natural order (so DC sits at top-left). Convert by indexing
//! `coef_levels[zigzag_to_natural[i]]` if you need natural-order levels.
//!
//! # Memory cost
//!
//! For an `H × W` image with three components and 4:4:4 sampling:
//!
//! ```text
//! per-block payload  = sizeof(coef_pre_quant) + sizeof(coef_levels)
//!                      + sizeof(misc)
//!                    ≈ 256 + 128 + 16 = 400 bytes
//! components         = 3
//! blocks per comp    = ceil(W / 8) × ceil(H / 8)
//! total payload      = 3 × ceil(W / 8) × ceil(H / 8) × 400 bytes
//! ```
//!
//! e.g. 512 × 512 ≈ 4.9 MB, 1024 × 1024 ≈ 19.7 MB. Subsampled chroma
//! halves (4:2:0) the chroma block count. This module is intentionally
//! not designed for production-size images — it exists to drive
//! tooling on small demo imagery.

#![allow(dead_code)]
#![cfg(feature = "__diagnostics")]

use alloc::vec::Vec;

/// Top-level diagnostic record for a single encode.
///
/// Populated by the encoder when running with `__diagnostics` enabled
/// AND a sink is requested. Field semantics:
///
/// - [`Self::components`] holds one entry per JPEG component, in
///   declaration order (Y, Cb, Cr for YCbCr; X, Y, B for XYB; one
///   entry for grayscale).
/// - [`Self::global`] carries image-level state that doesn't fit per
///   component (AQ field, scan script, heuristic decisions).
///
/// See module docs for ordering conventions.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct EncodeDiagnostics {
    /// Image dimensions and sampling layout.
    pub image: ImageInfo,

    /// Encoder configuration as actually resolved at encode time.
    pub config_snapshot: ConfigSnapshot,

    /// One entry per JPEG component (Y, Cb, Cr / X, Y, B / Y).
    pub components: Vec<ComponentDiagnostics>,

    /// Image-level state (AQ field, scan script, decisions).
    pub global: GlobalDiagnostics,
}

/// Image dimensions and sampling layout snapshot.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    /// Color path used by this encode (YCbCr / XYB / Grayscale).
    pub color_path: ColorPathTag,
    /// Per-component (h_samp, v_samp) sampling factors. Length matches
    /// `EncodeDiagnostics::components.len()`.
    pub sampling_factors: Vec<(u8, u8)>,
}

/// Resolved encoder configuration at encode time.
///
/// Snapshot of the knobs that affect quantization and bit allocation.
/// Mirrors [`InternalParams`](super::internal_params::InternalParams)
/// at a stable shape (visualizer doesn't need every internal detail —
/// just what matters for re-quant decisions).
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct ConfigSnapshot {
    /// Quality value at encode time (jpegli-internal scale).
    pub quality: f32,
    /// Whether AQ was enabled.
    pub aq_enabled: bool,
    /// Whether deringing preprocessing was enabled.
    pub deringing: bool,
    /// Quantization mode the encoder dispatched to for this encode.
    pub quant_mode: QuantModeTag,
    /// Source of the base quantization tables (default / user supplied
    /// / mozjpeg / etc.).
    pub quant_table_source: QuantTableSourceTag,
    /// Chroma distance scale, if applicable (XYB / YCbCr).
    pub chroma_distance_scale: Option<f32>,
    /// Tiny-file mode active.
    pub tiny_file_mode: Option<TinyFileModeTag>,
    /// Whether progressive scans were used.
    pub progressive: bool,
    /// Whether scan optimization (`optimize_scans`) ran.
    pub optimize_scans: bool,
    /// Whether Huffman tables were optimized vs taken from defaults.
    pub optimize_huffman: bool,
    /// Whether 16-bit DQT was permitted.
    pub allow_16bit_quant_tables: bool,
    /// `auto_optimize(true)` was set.
    pub auto_optimize: bool,
}

/// High-level color path classifier.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorPathTag {
    #[default]
    YCbCr,
    Xyb,
    Grayscale,
}

/// Which quantization branch the encoder dispatched to.
///
/// Three configurations are observable:
///
/// - [`Self::Plain`]: zero-bias quantize only (default fast path,
///   `aq_enabled` may still be true and modulate per-block).
/// - [`Self::Trellis`]: standalone AC trellis (`trellis` feature on,
///   no hybrid context).
/// - [`Self::Hybrid`]: hybrid AQ + trellis (`auto_optimize` /
///   `HybridConfig`).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QuantModeTag {
    #[default]
    Plain,
    Trellis,
    Hybrid,
}

/// Where the base quantization table came from.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QuantTableSourceTag {
    #[default]
    Default,
    UserSupplied,
    MozjpegPreset,
    LearnedXyb,
    AutoOptimized,
    Other,
}

/// Tiny-file mode classifier.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TinyFileModeTag {
    Off,
    Auto,
    Force,
}

/// Per-component diagnostic record.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ComponentDiagnostics {
    /// JPEG component identifier (Y=1, Cb=2, Cr=3 for YCbCr; X/Y/B
    /// for XYB; whatever the SOF declared).
    pub component_id: u8,

    /// Block grid dimensions (cols, rows) — number of 8×8 blocks
    /// horizontally and vertically for *this* component (post
    /// subsampling). For chroma in 4:2:0 these are half the luma
    /// counts.
    pub block_grid: (u32, u32),

    /// Base quantization table written to JFIF (DQT) for this
    /// component, in **natural row-major** order (index `i = row*8 +
    /// col`). Per-block AQ scaling is applied on top of this base at
    /// quantize time using `BlockDiagnostics::aq_multiplier`. Always
    /// 64 entries; values may exceed 255 when 16-bit DQT is permitted.
    pub quant_table_base: [u16; 64],

    /// Zero-bias table (jpegli adaptive dead-zone bias) for this
    /// component, in natural order. Values are coefficient-space
    /// thresholds: a coefficient with absolute value below
    /// `bias[uv] * quant[uv]` is snapped to zero.  When trellis is
    /// active for this component, the trellis lambda overrides this
    /// at the per-coefficient level. Length 64.
    pub zero_bias: [f32; 64],

    /// Per-block records, length `block_grid.0 * block_grid.1`,
    /// indexed by `row * block_grid.0 + col`.
    pub blocks: Vec<BlockDiagnostics>,
}

/// Per-block diagnostic record.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct BlockDiagnostics {
    /// Forward-DCT coefficients before quantization, in **natural
    /// row-major** order (index `i = row*8 + col`). DC sits at
    /// position 0; AC fills 1..=63. Captured directly from the
    /// encoder's `Block8x8f` after deringing+DCT, before any
    /// quantization.
    ///
    /// Memory: 256 bytes per block. This is the dominant cost of
    /// diagnostics output.
    pub coef_pre_quant: [f32; 64],

    /// Quantized coefficient levels in **JPEG zigzag** order — what
    /// the Huffman entropy coder consumed for this block.
    /// `coef_levels[0]` is the DC level; `coef_levels[1..=63]` is
    /// the AC band in zigzag traversal order.
    pub coef_levels: [i16; 64],

    /// AQ multiplier applied to the base quant table for this block.
    /// `1.0` means "use base table as-is"; `<1.0` means "spend more
    /// bits here" (finer quantization); `>1.0` means "save bits here"
    /// (coarser). Encoder sources this from `aq_strengths` after the
    /// optional [`AqController`](crate::encode::aq_controller::AqController)
    /// hook.
    pub aq_multiplier: f32,

    /// Entropy-coded byte cost contributed by this block, summed
    /// over Huffman code lengths + value bits. Excludes restart
    /// markers and DRI bookkeeping. `0` if entropy capture is
    /// disabled (e.g. progressive multi-scan layouts where one block
    /// contributes to several scans — see
    /// [`Self::progressive_scan_bits`]).
    pub entropy_bits: u32,

    /// Per-progressive-scan entropy bit attribution. For baseline
    /// (single-scan) encodes this stays empty and
    /// [`Self::entropy_bits`] is the source of truth. For
    /// progressive scan layouts, length matches the scan count and
    /// the sum equals [`Self::entropy_bits`].
    pub progressive_scan_bits: Vec<u32>,
}

impl Default for BlockDiagnostics {
    fn default() -> Self {
        Self {
            coef_pre_quant: [0.0; 64],
            coef_levels: [0; 64],
            aq_multiplier: 1.0,
            entropy_bits: 0,
            progressive_scan_bits: Vec::new(),
        }
    }
}

/// Image-level diagnostic state that doesn't fit per component.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct GlobalDiagnostics {
    /// AQ field after `StreamingAQ` finalized strengths (and after
    /// any [`AqController`](crate::encode::aq_controller::AqController)
    /// adjustments). One entry per luma 8×8 block in raster order;
    /// length matches `components[0].blocks.len()` for YCbCr/XYB.
    /// `None` when AQ was disabled.
    pub aq_field: Option<AqFieldDump>,

    /// Resolved progressive scan script, when progressive encoding
    /// was used. `None` for baseline.
    pub scan_script: Option<ScanScriptDump>,

    /// Heuristic / auto-optimize decisions logged during encode (why
    /// `auto_optimize` chose hybrid vs plain, why a quant table was
    /// up/down-shifted, etc.).
    pub heuristics: Vec<HeuristicEntry>,
}

/// AQ field dump — per-luma-block strengths in raster order.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct AqFieldDump {
    /// Cols, rows in luma-block units.
    pub block_grid: (u32, u32),
    /// Per-block strengths in raster order (row-major).
    pub strengths: Vec<f32>,
}

/// Resolved progressive scan script.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct ScanScriptDump {
    pub scans: Vec<ScanInfo>,
}

/// Single progressive scan descriptor.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ScanInfo {
    /// Component IDs in this scan (1-based JFIF identifiers).
    pub components: Vec<u8>,
    /// Spectral band start index (Ss).
    pub ss: u8,
    /// Spectral band end index (Se).
    pub se: u8,
    /// Successive approximation high bit (Ah).
    pub ah: u8,
    /// Successive approximation low bit (Al).
    pub al: u8,
}

/// One heuristic / decision log line.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HeuristicEntry {
    /// Stage that recorded this entry (`"auto_optimize"`,
    /// `"scan_script"`, `"quant_table_select"`, …).
    pub stage: &'static str,
    /// Free-form decision text.
    pub message: alloc::string::String,
}

/// Configuration toggle for what to capture.
///
/// All fields default to `true` — the most useful default is "capture
/// everything." Disable individual fields when chasing memory or
/// per-block CPU cost.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct DiagnosticsCapture {
    pub coef_pre_quant: bool,
    pub coef_levels: bool,
    pub entropy_bits: bool,
    pub aq_field: bool,
    pub scan_script: bool,
    pub heuristics: bool,
}

impl Default for DiagnosticsCapture {
    fn default() -> Self {
        Self {
            coef_pre_quant: true,
            coef_levels: true,
            entropy_bits: true,
            aq_field: true,
            scan_script: true,
            heuristics: true,
        }
    }
}

impl EncodeDiagnostics {
    /// Reset all per-encode state. Useful for reusing a diagnostics
    /// buffer across multiple encodes.
    pub fn clear(&mut self) {
        self.image = ImageInfo::default();
        self.config_snapshot = ConfigSnapshot::default();
        self.components.clear();
        self.global = GlobalDiagnostics::default();
    }

    /// Total per-block payload size in bytes. Useful for choosing a
    /// safe input image size given a memory budget.
    pub fn approx_bytes(&self) -> usize {
        let per_block = core::mem::size_of::<BlockDiagnostics>();
        self.components
            .iter()
            .map(|c| c.blocks.len() * per_block)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    #[test]
    fn default_construction_zero_blocks() {
        let d = EncodeDiagnostics::default();
        assert_eq!(d.image.width, 0);
        assert!(d.components.is_empty());
        assert!(d.global.aq_field.is_none());
        assert!(d.global.scan_script.is_none());
        assert_eq!(d.approx_bytes(), 0);
    }

    #[test]
    fn approx_bytes_scales_with_block_count() {
        let mut d = EncodeDiagnostics::default();
        d.components.push(ComponentDiagnostics {
            component_id: 1,
            block_grid: (2, 2),
            quant_table_base: [16; 64],
            zero_bias: [0.5; 64],
            blocks: vec![BlockDiagnostics::default(); 4],
        });
        let per_block = core::mem::size_of::<BlockDiagnostics>();
        assert_eq!(d.approx_bytes(), per_block * 4);
    }

    #[test]
    fn capture_default_is_everything_on() {
        let c = DiagnosticsCapture::default();
        assert!(c.coef_pre_quant);
        assert!(c.coef_levels);
        assert!(c.entropy_bits);
        assert!(c.aq_field);
        assert!(c.scan_script);
        assert!(c.heuristics);
    }

    #[test]
    fn clear_returns_to_default() {
        let mut d = EncodeDiagnostics::default();
        d.image.width = 100;
        d.components.push(ComponentDiagnostics {
            component_id: 1,
            block_grid: (1, 1),
            quant_table_base: [16; 64],
            zero_bias: [0.5; 64],
            blocks: vec![BlockDiagnostics::default()],
        });
        d.global.heuristics.push(HeuristicEntry {
            stage: "test",
            message: String::from("placeholder"),
        });
        d.clear();
        assert_eq!(d.image.width, 0);
        assert!(d.components.is_empty());
        assert!(d.global.heuristics.is_empty());
    }
}
