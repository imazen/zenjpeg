//! WASM bindings for zenjpeg's UNSTABLE `__diagnostics` capture.
//!
//! Exposes a single `encode_with_diagnostics(rgba_bytes, width, height,
//! options)` entry that runs the encoder and returns both the JPEG byte
//! payload and a serializable per-block diagnostics record. Used by the
//! demo viewer in `../web/`.
//!
//! ## Output shape (TypeScript-friendly)
//!
//! The returned object has:
//! - `bytes: Uint8Array` — the encoded JPEG bytes
//! - `diagnostics: Diagnostics` — JSON-friendly diagnostics record
//!
//! See `Diagnostics` below for the full shape; field names match the
//! Rust struct exactly so a TS consumer can re-derive types from the
//! JSON Schema if needed.

#![allow(clippy::missing_safety_doc)]

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use zenjpeg::encoder::diagnostics as diag_src;
use zenjpeg::encoder::{ChromaSubsampling, ColorMode, EncoderConfig, PixelLayout, XybSubsampling};

/// Per-block diagnostic record, JS-friendly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockDiagnostics {
    /// Forward-DCT coefficients in natural row-major order
    /// (index `i = row*8 + col`). Length 64.
    pub coef_pre_quant: Vec<f32>,
    /// Quantized levels in JPEG zigzag order. Length 64.
    pub coef_levels: Vec<i16>,
    /// AQ strength supplied to the block quantizer.
    pub aq_strength: f32,
}

/// Per-component diagnostic record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentDiagnostics {
    /// JFIF component identifier (1=Y/X, 2=Cb/Y-XYB, 3=Cr/B-XYB).
    pub component_id: u8,
    /// (cols, rows) in 8x8 blocks.
    pub block_grid: (u32, u32),
    /// Base quantization table (natural row-major). Length 64.
    pub quant_table_base: Vec<u16>,
    /// Zero-bias offset table (natural row-major). Length 64.
    pub zero_bias: Vec<f32>,
    pub zero_bias_mul: Vec<f32>,
    /// Per-block records, raster order
    /// (index `r * cols + c` for `(r, c)` in the block grid).
    pub blocks: Vec<BlockDiagnostics>,
}

/// Top-level diagnostic record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostics {
    pub width: u32,
    pub height: u32,
    pub color_path: String,
    pub sampling_factors: Vec<(u8, u8)>,
    pub restart_interval_mcus: u16,
    pub components: Vec<ComponentDiagnostics>,
}

fn to_js_diagnostics(src: diag_src::EncodeDiagnostics) -> Diagnostics {
    Diagnostics {
        width: src.width,
        height: src.height,
        color_path: src.color_path.into(),
        sampling_factors: src.sampling_factors,
        restart_interval_mcus: 0,
        components: src
            .components
            .into_iter()
            .map(|c| ComponentDiagnostics {
                component_id: c.component_id,
                block_grid: c.block_grid,
                quant_table_base: c.quant_table_base.to_vec(),
                zero_bias: c.zero_bias.to_vec(),
                zero_bias_mul: c.zero_bias_mul.to_vec(),
                blocks: c
                    .blocks
                    .into_iter()
                    .map(|b| BlockDiagnostics {
                        coef_pre_quant: b.coef_pre_quant.to_vec(),
                        coef_levels: b.coef_levels.to_vec(),
                        aq_strength: b.aq_strength,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Options controlling the encode. Mirrors essentially every dial on
/// `EncoderConfig` so the diagnostics viewer's UI can offer expert-mode
/// access. Field names use camelCase for TS-natural input. All fields
/// are optional with documented defaults.
///
/// ## Mode dispatch
///
/// `mode` selects which optimization to run:
/// - `"baseline"` (default): no trellis, no auto-optimize. Honors
///   `optimizeHuffman`, `progressive`, AQ, deringing, sharp_yuv,
///   pre_blur, chroma_distance_scale, and any custom-table override.
/// - `"trellis"`: standalone trellis with the explicit `trellis*`
///   sub-config. Plays nicely with AQ.
/// - `"hybrid"`: AQ-coupled trellis, reads the
///   `hybrid*` sub-config. Always overrides standalone trellis.
///
/// Knobs not relevant to the chosen mode do not affect the encode; the
/// viewer exposes only controls for the active mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncodeOptions {
    // ── Always-applied basics ───────────────────────────────────────
    /// jpegli quality (0-100). Default 85.
    #[serde(default = "default_quality")]
    pub quality: f32,
    /// "ycbcr" or "xyb". Default "ycbcr".
    #[serde(default = "default_color_path")]
    pub color_path: String,
    /// "none" (4:4:4), "halfHorizontal" (4:2:2), "quarter" (4:2:0),
    /// "halfVertical" (4:4:0). Used when colorPath = "ycbcr".
    #[serde(default = "default_subsampling")]
    pub subsampling: String,
    /// "full" or "bQuarter". Used when colorPath = "xyb".
    #[serde(default = "default_xyb_subsampling")]
    pub xyb_subsampling: String,
    /// Adaptive quantization on/off. Default true.
    #[serde(default = "default_true")]
    pub aq_enabled: bool,
    /// Deringing on/off. Default true.
    #[serde(default = "default_true")]
    pub deringing: bool,
    /// Optimize Huffman tables (second-pass pixel sweep). Default true.
    #[serde(default = "default_true")]
    pub optimize_huffman: bool,
    /// Progressive scan (multiple SOS markers, smaller for high-q
    /// images). Default false (baseline).
    #[serde(default)]
    pub progressive: bool,
    /// Sharp-YUV chroma downsampling (gamma-aware). Only meaningful
    /// for YCbCr 4:2:2/4:2:0/4:4:0; ignored for 4:4:4 and XYB.
    #[serde(default)]
    pub sharp_yuv: bool,
    /// Pre-encode Gaussian blur sigma (px). 0 disables.
    #[serde(default)]
    pub pre_blur: f32,
    /// Multiplicative scale on chroma's perceptual distance budget.
    /// 1.0 = default. Larger values increase chroma distance.
    #[serde(default = "default_one")]
    pub chroma_distance_scale: f32,
    /// Restart-marker cadence in MCU rows. None uses the codec default; 0 disables.
    #[serde(default)]
    pub restart_mcu_rows: Option<u16>,

    // ── Mode picker ─────────────────────────────────────────────────
    /// "baseline" | "trellis" | "hybrid".
    #[serde(default = "default_mode")]
    pub mode: String,

    // ── Standalone trellis sub-config (used when mode == "trellis") ─
    /// Enable DC-coefficient trellis. Default false.
    #[serde(default)]
    pub trellis_dc_enabled: bool,
    /// Lambda log-scale 1 (rate penalty). Default 14.75.
    #[serde(default = "default_trellis_lambda1")]
    pub trellis_lambda_log_scale1: f32,
    /// Lambda log-scale 2 (distortion sensitivity). Default 16.5.
    #[serde(default = "default_trellis_lambda2")]
    pub trellis_lambda_log_scale2: f32,
    /// "thorough" | "balanced" | "fast".
    #[serde(default = "default_trellis_speed")]
    pub trellis_speed_mode: String,
    /// Vertical-DC-gradient weight in DC trellis. 0.0 disables.
    #[serde(default)]
    pub trellis_delta_dc_weight: f32,

    // ── Hybrid sub-config (used when mode == "hybrid") ──────────────
    /// Lambda gain per unit AQ strength. Default 2.0.
    #[serde(default = "default_hybrid_aq_lambda_scale")]
    pub hybrid_aq_lambda_scale: f32,
    /// Base lambda log-scale 1. Default 14.75.
    #[serde(default = "default_trellis_lambda1")]
    pub hybrid_base_lambda_scale1: f32,
    /// Base lambda log-scale 2. Default 16.5.
    #[serde(default = "default_trellis_lambda2")]
    pub hybrid_base_lambda_scale2: f32,
    /// Enable DC-coefficient trellis in hybrid mode. Default false.
    #[serde(default)]
    pub hybrid_dc_enabled: bool,
    /// AQ-strength exponent (1.0=linear, 2.0=squared, 0.5=sqrt).
    #[serde(default = "default_one")]
    pub hybrid_aq_exponent: f32,
    /// Minimum AQ strength to start adjusting lambda.
    #[serde(default)]
    pub hybrid_aq_threshold: f32,
    /// Quality-adaptive lambda dampening. Default true.

    // ── Custom quantization tables (overrides everything) ───────────
    /// When set, replaces the source-derived quant tables with the
    /// caller's f32 values. Other table fields (zero_bias_*) fall
    /// back to jpegli defaults if not provided.
    #[serde(default)]
    pub custom_quant_tables: Option<CustomQuantTables>,
}

/// f32 quant tables + optional zero-bias overrides for a custom
/// `EncodingTables`. Each component table is 64 floats in natural
/// row-major order (DC at index 0, AC[7,7] at index 63).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomQuantTables {
    /// Y / X (luma) quant table. Length 64.
    pub y: Vec<f32>,
    /// Cb / Y-XYB quant table. Length 64.
    pub cb: Vec<f32>,
    /// Cr / B-XYB quant table. Length 64.
    pub cr: Vec<f32>,
    /// Optional zero-bias multipliers per component (each length 64).
    /// Higher values = more aggressive zeroing of small coefficients.
    #[serde(default)]
    pub y_zero_bias_mul: Option<Vec<f32>>,
    #[serde(default)]
    pub cb_zero_bias_mul: Option<Vec<f32>>,
    #[serde(default)]
    pub cr_zero_bias_mul: Option<Vec<f32>>,
    /// Optional zero-bias DC offset, per component [Y, Cb, Cr]. Length 3.
    #[serde(default)]
    pub zero_bias_offset_dc: Option<[f32; 3]>,
    /// Optional zero-bias AC offset, per component [Y, Cb, Cr]. Length 3.
    #[serde(default)]
    pub zero_bias_offset_ac: Option<[f32; 3]>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quality: default_quality(),
            color_path: default_color_path(),
            subsampling: default_subsampling(),
            xyb_subsampling: default_xyb_subsampling(),
            aq_enabled: true,
            deringing: true,
            optimize_huffman: true,
            progressive: false,
            sharp_yuv: false,
            pre_blur: 0.0,
            chroma_distance_scale: 1.0,
            restart_mcu_rows: None,
            mode: default_mode(),
            trellis_dc_enabled: false,
            trellis_lambda_log_scale1: default_trellis_lambda1(),
            trellis_lambda_log_scale2: default_trellis_lambda2(),
            trellis_speed_mode: default_trellis_speed(),
            trellis_delta_dc_weight: 0.0,
            hybrid_aq_lambda_scale: default_hybrid_aq_lambda_scale(),
            hybrid_base_lambda_scale1: default_trellis_lambda1(),
            hybrid_base_lambda_scale2: default_trellis_lambda2(),
            hybrid_dc_enabled: false,
            hybrid_aq_exponent: 1.0,
            hybrid_aq_threshold: 0.0,
            custom_quant_tables: None,
        }
    }
}

fn default_quality() -> f32 {
    85.0
}
fn default_color_path() -> String {
    "ycbcr".into()
}
fn default_subsampling() -> String {
    "none".into()
}
fn default_xyb_subsampling() -> String {
    "full".into()
}
fn default_true() -> bool {
    true
}
fn default_one() -> f32 {
    1.0
}
fn default_mode() -> String {
    "baseline".into()
}
fn default_trellis_lambda1() -> f32 {
    14.75
}
fn default_trellis_lambda2() -> f32 {
    16.5
}
fn default_trellis_speed() -> String {
    "balanced".into()
}
fn default_hybrid_aq_lambda_scale() -> f32 {
    2.0
}

// Note: `EncodeResult` is built up on the JS side as
// `{ bytes: Uint8Array, diagnostics: Diagnostics }`. We can't go through
// serde_wasm_bindgen for the whole struct because it serializes
// `Vec<u8>` as a plain JS Array<number>, which produces invalid bytes
// when fed to `new Blob([...])`. We hand-pack the result with
// `js_sys::Uint8Array` to keep the byte path zero-copy.

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Defaults resolved from the encoder's own types and constructor.
#[wasm_bindgen(js_name = codecDefaults)]
pub fn codec_defaults() -> Result<JsValue, JsValue> {
    let (color, sampling) = match ColorMode::default() {
        ColorMode::YCbCr { subsampling } => (
            "ycbcr",
            match subsampling {
                ChromaSubsampling::None => "none",
                ChromaSubsampling::HalfHorizontal => "halfHorizontal",
                ChromaSubsampling::Quarter => "quarter",
                ChromaSubsampling::HalfVertical => "halfVertical",
                _ => return Err(JsValue::from_str("unsupported default YCbCr sampling")),
            },
        ),
        _ => return Err(JsValue::from_str("unsupported default color mode")),
    };
    let xyb_sampling = match XybSubsampling::default() {
        XybSubsampling::Full => "full",
        XybSubsampling::BQuarter => "bQuarter",
        _ => return Err(JsValue::from_str("unsupported default XYB sampling")),
    };
    let config = EncoderConfig::ycbcr(
        85,
        match ColorMode::default() {
            ColorMode::YCbCr { subsampling } => subsampling,
            _ => return Err(JsValue::from_str("unsupported default color mode")),
        },
    );
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Defaults<'a> {
        color_path: &'a str,
        subsampling: &'a str,
        xyb_subsampling: &'a str,
        progressive: bool,
        optimize_huffman: bool,
        aq_enabled: bool,
        deringing: bool,
    }
    serde_wasm_bindgen::to_value(&Defaults {
        color_path: color,
        subsampling: sampling,
        xyb_subsampling: xyb_sampling,
        progressive: config.is_progressive(),
        optimize_huffman: config.is_optimize_huffman(),
        aq_enabled: config.is_aq_enabled(),
        deringing: config.resolve_plan(64, 64).deringing,
    })
    .map_err(|e| JsValue::from_str(&format!("serialize defaults: {e}")))
}

/// Encode an RGB or RGBA pixel buffer and return the resulting JPEG
/// bytes plus a per-block diagnostics record.
///
/// `pixels` must be packed `[r, g, b]` triples (length = `width *
/// height * 3`) — RGBA input should be flattened by the caller before
/// passing in.
///
/// `options` is a JS object matching `EncodeOptions` (all fields
/// optional; defaults documented inline). Pass `null` / `undefined`
/// for defaults.
#[wasm_bindgen(js_name = encodeWithDiagnostics)]
pub fn encode_with_diagnostics(
    pixels: &[u8],
    width: u32,
    height: u32,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let opts: EncodeOptions = if options.is_null() || options.is_undefined() {
        EncodeOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsValue::from_str(&format!("invalid options: {e}")))?
    };

    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| JsValue::from_str("image dimensions overflow"))?;
    if pixels.len() != expected {
        return Err(JsValue::from_str(&format!(
            "pixels length mismatch: expected {expected} ({width}*{height}*3), \
             got {}",
            pixels.len()
        )));
    }

    let default_color = ColorMode::default();
    let color_path = if opts.color_path == "auto" {
        match default_color {
            ColorMode::YCbCr { .. } => "ycbcr",
            ColorMode::Xyb { .. } => "xyb",
            _ => return Err(JsValue::from_str("unsupported codec default color mode")),
        }
    } else {
        opts.color_path.as_str()
    };
    let config = match color_path {
        "ycbcr" => {
            let sub = match opts.subsampling.as_str() {
                "auto" => match default_color {
                    ColorMode::YCbCr { subsampling } => subsampling,
                    _ => return Err(JsValue::from_str("codec has no default YCbCr subsampling")),
                },
                "none" | "s444" | "4:4:4" => ChromaSubsampling::None,
                "halfHorizontal" | "s422" | "4:2:2" => ChromaSubsampling::HalfHorizontal,
                "quarter" | "s420" | "4:2:0" => ChromaSubsampling::Quarter,
                "halfVertical" | "s440" | "4:4:0" => ChromaSubsampling::HalfVertical,
                other => {
                    return Err(JsValue::from_str(&format!("unknown subsampling: {other}")));
                }
            };
            EncoderConfig::ycbcr(opts.quality, sub)
        }
        "xyb" => {
            let xyb_sub = match opts.xyb_subsampling.as_str() {
                "auto" => XybSubsampling::default(),
                "full" => XybSubsampling::Full,
                "bQuarter" | "b_quarter" => XybSubsampling::BQuarter,
                other => {
                    return Err(JsValue::from_str(&format!(
                        "unknown xyb subsampling: {other}"
                    )));
                }
            };
            EncoderConfig::xyb(opts.quality, xyb_sub)
        }
        other => {
            return Err(JsValue::from_str(&format!("unknown color path: {other}")));
        }
    };

    use zenjpeg::encoder::ProgressiveScanMode;
    use zenjpeg::encoder::QuantTableConfig;
    use zenjpeg::encoder::diagnostics::{EncodingTables, ScalingParams};
    use zenjpeg::encoder::{AqCoupling, TrellisConfig, TrellisSpeedMode};

    // Always-applied dials. Knobs that don't apply to the chosen
    // (color path × subsampling × mode) combination get silently
    // ignored — the JS side flags them as "ignored: …" in the UI.
    let scan_mode = if opts.progressive {
        ProgressiveScanMode::Progressive
    } else {
        ProgressiveScanMode::Baseline
    };
    let mut config = config
        .aq_enabled(opts.aq_enabled)
        .deringing(opts.deringing)
        .optimize_huffman(opts.optimize_huffman)
        .scan_mode(scan_mode)
        .sharp_yuv(opts.sharp_yuv)
        .pre_blur(opts.pre_blur)
        .quant_table_config(QuantTableConfig::jpegli_chroma_scale(
            opts.chroma_distance_scale,
        ));
    if let Some(rows) = opts.restart_mcu_rows {
        config = config.restart_mcu_rows(rows);
    }

    // Custom quant tables override `quant_table_config` entirely.
    // PerComponent uses generic c0/c1/c2 fields (Y/Cb/Cr in YCbCr mode).
    //
    // CRITICAL: when the caller passes a custom table, we set
    // `scaling: ScalingParams::Exact` so the f32 values are rounded
    // to JPEG table entries without additional quality scaling. With the default
    // `ScalingParams::Scaled`, the table would be re-multiplied by
    // distance_to_scale × global_scale at encode time — a "Q-scale =
    // 0.5" edit on the JS side would compound, exploding to
    // unreasonable values.
    if let Some(tables) = opts.custom_quant_tables.as_ref() {
        let is_xyb = color_path == "xyb";
        let mut t = if is_xyb {
            EncodingTables::default_xyb()
        } else {
            EncodingTables::default_ycbcr()
        };
        if tables.y.len() == 64 {
            t.quant.c0.copy_from_slice(&tables.y);
        }
        if tables.cb.len() == 64 {
            t.quant.c1.copy_from_slice(&tables.cb);
        }
        if tables.cr.len() == 64 {
            t.quant.c2.copy_from_slice(&tables.cr);
        }
        if let Some(v) = tables.y_zero_bias_mul.as_ref()
            && v.len() == 64
        {
            t.zero_bias_mul.c0.copy_from_slice(v);
        }
        if let Some(v) = tables.cb_zero_bias_mul.as_ref()
            && v.len() == 64
        {
            t.zero_bias_mul.c1.copy_from_slice(v);
        }
        if let Some(v) = tables.cr_zero_bias_mul.as_ref()
            && v.len() == 64
        {
            t.zero_bias_mul.c2.copy_from_slice(v);
        }
        if let Some(dc) = tables.zero_bias_offset_dc {
            t.zero_bias_offset_dc = dc;
        }
        if let Some(ac) = tables.zero_bias_offset_ac {
            t.zero_bias_offset_ac = ac;
        }
        // The caller is editing final tables, not pre-scale base
        // values — switch to Exact so the values pass through.
        t.scaling = ScalingParams::exact();
        config = config.quant_table_config(QuantTableConfig::Custom(Box::new(t)));
    }

    // Mode picker: hybrid > trellis > baseline.
    match opts.mode.as_str() {
        "hybrid" => {
            let h = TrellisConfig {
                enabled: true,
                dc_enabled: opts.hybrid_dc_enabled,
                lambda_log_scale1: opts.hybrid_base_lambda_scale1,
                lambda_log_scale2: opts.hybrid_base_lambda_scale2,
                aq_coupling: AqCoupling {
                    scale: opts.hybrid_aq_lambda_scale,
                    exponent: opts.hybrid_aq_exponent,
                    threshold: opts.hybrid_aq_threshold,
                    ..AqCoupling::OFF
                },
                ..TrellisConfig::default()
            };
            config = config.trellis(h);
        }
        "trellis" => {
            let speed = match opts.trellis_speed_mode.as_str() {
                "thorough" => TrellisSpeedMode::Thorough,
                "fast" => TrellisSpeedMode::Level(8),
                _ => TrellisSpeedMode::Adaptive,
            };
            let mut t = TrellisConfig::new();
            t.dc_enabled = opts.trellis_dc_enabled;
            t.lambda_log_scale1 = opts.trellis_lambda_log_scale1;
            t.lambda_log_scale2 = opts.trellis_lambda_log_scale2;
            t.speed_mode = speed;
            t.delta_dc_weight = opts.trellis_delta_dc_weight;
            config = config.trellis(t);
        }
        // "baseline" or unknown — no trellis/hybrid wiring.
        _ => {}
    }

    let restart_interval = config.resolve_plan(width, height).restart_interval_mcus;
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| JsValue::from_str(&format!("encoder build: {e}")))?;
    let capture = encoder
        .capture_diagnostics()
        .map_err(|e| JsValue::from_str(&format!("diagnostics: {e}")))?;
    encoder
        .push_packed(pixels, enough::Unstoppable)
        .map_err(|e| JsValue::from_str(&format!("push pixels: {e}")))?;
    let bytes = encoder
        .finish()
        .map_err(|e| JsValue::from_str(&format!("finish: {e}")))?;
    let diag = capture
        .try_take()
        .map_err(|_| JsValue::from_str("encoder did not release diagnostics"))?;

    let js_bytes = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
    js_bytes.copy_from(&bytes);
    let mut diagnostic_result = to_js_diagnostics(diag);
    diagnostic_result.restart_interval_mcus = restart_interval;
    let js_diag = serde_wasm_bindgen::to_value(&diagnostic_result)
        .map_err(|e| JsValue::from_str(&format!("serialize diagnostics: {e}")))?;
    let out = js_sys::Object::new();
    js_sys::Reflect::set(&out, &JsValue::from_str("bytes"), &js_bytes)
        .map_err(|_| JsValue::from_str("failed to set bytes on result"))?;
    js_sys::Reflect::set(&out, &JsValue::from_str("diagnostics"), &js_diag)
        .map_err(|_| JsValue::from_str("failed to set diagnostics on result"))?;
    Ok(out.into())
}

/// Compute SSIMULACRA 2 between two sRGB-encoded RGB byte buffers.
///
/// Both `source` and `distorted` must be packed `[r, g, b]` triples
/// with the same dimensions (length = `width * height * 3`). Returns
/// the f64 SSIMULACRA 2 score (higher = more similar; ~80+ "good",
/// 100 = identical, can go negative for severe distortions).
///
/// On any input mismatch or computation failure, returns a JS error.
#[wasm_bindgen(js_name = computeSsimulacra2)]
pub fn compute_ssimulacra2_rgb(
    source: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, JsValue> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| JsValue::from_str("image dimensions overflow"))?;
    if source.len() != expected {
        return Err(JsValue::from_str(&format!(
            "source pixels length mismatch: expected {expected} ({width}*{height}*3), got {}",
            source.len()
        )));
    }
    if distorted.len() != expected {
        return Err(JsValue::from_str(&format!(
            "distorted pixels length mismatch: expected {expected} ({width}*{height}*3), got {}",
            distorted.len()
        )));
    }
    if width == 0 || height == 0 {
        return Err(JsValue::from_str("width and height must be > 0"));
    }
    let lin_src = rgb_bytes_to_linear(source, width as usize, height as usize);
    let lin_dst = rgb_bytes_to_linear(distorted, width as usize, height as usize);
    fast_ssim2::compute_ssimulacra2(lin_src, lin_dst)
        .map_err(|e| JsValue::from_str(&format!("ssimulacra2: {e}")))
}

/// Butteraugli max-norm distance for sRGB RGB8 buffers (lower is better).
#[wasm_bindgen(js_name = computeButteraugli)]
pub fn compute_butteraugli_rgb(
    source: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, JsValue> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| JsValue::from_str("image dimensions overflow"))?;
    if source.len() != expected || distorted.len() != expected {
        return Err(JsValue::from_str(
            "RGB buffer length does not match image dimensions",
        ));
    }
    if width < 8 || height < 8 {
        return Err(JsValue::from_str(
            "Butteraugli requires images of at least 8×8 pixels",
        ));
    }
    let pack = |bytes: &[u8]| -> Result<Vec<butteraugli::RGB8>, JsValue> {
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(bytes.len() / 3)
            .map_err(|_| JsValue::from_str("could not allocate Butteraugli pixels"))?;
        pixels.extend(
            bytes
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| butteraugli::RGB8::new(p[0], p[1], p[2])),
        );
        Ok(pixels)
    };
    let original = pack(source)?;
    let decoded = pack(distorted)?;
    let original = butteraugli::Img::new(original, width as usize, height as usize);
    let decoded = butteraugli::Img::new(decoded, width as usize, height as usize);
    butteraugli::butteraugli(
        original.as_ref(),
        decoded.as_ref(),
        &butteraugli::ButteraugliParams::default(),
    )
    .map(|result| result.score)
    .map_err(|e| JsValue::from_str(&format!("butteraugli: {e}")))
}

/// Compute zensim score between two sRGB-encoded RGB byte buffers.
///
/// Same input shape as [`compute_ssimulacra2_rgb`]. Returns the f64
/// zensim score (0-100; 100 = identical). Uses the latest profile.
#[wasm_bindgen(js_name = computeZensim)]
pub fn compute_zensim_rgb(
    source: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, JsValue> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| JsValue::from_str("image dimensions overflow"))?;
    if source.len() != expected || distorted.len() != expected {
        return Err(JsValue::from_str(&format!(
            "rgb length mismatch: expected {expected} (={width}*{height}*3), \
             got source={} distorted={}",
            source.len(),
            distorted.len()
        )));
    }
    if width == 0 || height == 0 {
        return Err(JsValue::from_str("width and height must be > 0"));
    }
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let src_rgb = pack_rgb_triples(source, n);
    let dst_rgb = pack_rgb_triples(distorted, n);
    let z = zensim::Zensim::new(zensim::ZensimProfile::latest());
    let src = zensim::RgbSlice::new(&src_rgb, w, h);
    let dst = zensim::RgbSlice::new(&dst_rgb, w, h);
    z.compute(&src, &dst)
        .map(|r| r.score())
        .map_err(|e| JsValue::from_str(&format!("zensim: {e}")))
}

fn pack_rgb_triples(rgb: &[u8], n: usize) -> Vec<[u8; 3]> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]]);
    }
    out
}

fn rgb_bytes_to_linear(rgb: &[u8], w: usize, h: usize) -> fast_ssim2::LinearRgbImage {
    let n = w * h;
    let mut data: Vec<[f32; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        data.push([
            fast_ssim2::srgb_u8_to_linear(r),
            fast_ssim2::srgb_u8_to_linear(g),
            fast_ssim2::srgb_u8_to_linear(b),
        ]);
    }
    fast_ssim2::LinearRgbImage::new(data, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butteraugli_identical_and_distorted_inputs() {
        let source = synthetic(32, 32);
        let identical = compute_butteraugli_rgb(&source, &source, 32, 32).unwrap();
        let mut distorted = source.clone();
        for pixel in distorted.chunks_exact_mut(3) {
            pixel[0] = pixel[0].saturating_add(30);
        }
        let changed = compute_butteraugli_rgb(&source, &distorted, 32, 32).unwrap();
        assert!(identical.abs() < 0.0001, "identical distance {identical}");
        assert!(changed > identical, "distorted distance {changed}");
    }

    fn synthetic(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32;
                let fy = y as f32;
                let r = (128.0 + 64.0 * (fx * 0.3).sin()).clamp(0.0, 255.0) as u8;
                let g = (128.0 + 64.0 * (fy * 0.25).cos()).clamp(0.0, 255.0) as u8;
                let b = (128.0 + 64.0 * ((fx + fy) * 0.15).sin()).clamp(0.0, 255.0) as u8;
                buf.extend_from_slice(&[r, g, b]);
            }
        }
        buf
    }

    /// Native side test: ensures the conversion path produces the same
    /// shape as the Rust diagnostics. Doesn't go through wasm-bindgen
    /// (that's covered by the web/Playwright suite).
    #[test]
    #[allow(unused_variables)]
    fn to_js_diagnostics_preserves_shape() {
        let w = 32u32;
        let h = 32u32;
        let pixels = synthetic(w, h);
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).aq_enabled(true);
        let mut enc = config
            .request()
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .expect("encoder");
        let capture = enc.capture_diagnostics().expect("capture");
        enc.push_packed(&pixels, enough::Unstoppable).expect("push");
        let _bytes = enc.finish().expect("finish");
        let diag = capture.try_take().expect("diag");
        let js = to_js_diagnostics(diag);
        assert_eq!(js.width, w);
        assert_eq!(js.height, h);
        assert_eq!(js.color_path, "YCbCr");
        assert_eq!(js.components.len(), 3);
        assert_eq!(js.components[0].block_grid, (4, 4));
        assert_eq!(js.components[1].block_grid, (2, 2));
        assert_eq!(js.components[2].block_grid, (2, 2));
        assert_eq!(js.components[0].quant_table_base.len(), 64);
        assert_eq!(js.components[0].zero_bias.len(), 64);
        for comp in &js.components {
            for block in &comp.blocks {
                assert_eq!(block.coef_pre_quant.len(), 64);
                assert_eq!(block.coef_levels.len(), 64);
            }
        }
    }
}
