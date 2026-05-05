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
use zenjpeg::encode::diagnostics as diag_src;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

/// Per-block diagnostic record, JS-friendly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockDiagnostics {
    /// Forward-DCT coefficients in natural row-major order
    /// (index `i = row*8 + col`). Length 64.
    pub coef_pre_quant: Vec<f32>,
    /// Quantized levels in JPEG zigzag order. Length 64.
    pub coef_levels: Vec<i16>,
    /// Per-block AQ multiplier (1.0 = neutral, <1.0 = finer, >1.0 = coarser).
    pub aq_multiplier: f32,
    /// Entropy bits attributed to this block (0 if not yet captured).
    pub entropy_bits: u32,
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
    pub components: Vec<ComponentDiagnostics>,
}

fn to_js_diagnostics(src: diag_src::EncodeDiagnostics) -> Diagnostics {
    Diagnostics {
        width: src.image.width,
        height: src.image.height,
        color_path: match src.image.color_path {
            diag_src::ColorPathTag::YCbCr => "YCbCr".into(),
            diag_src::ColorPathTag::Xyb => "XYB".into(),
            diag_src::ColorPathTag::Grayscale => "Grayscale".into(),
        },
        sampling_factors: src.image.sampling_factors,
        components: src
            .components
            .into_iter()
            .map(|c| ComponentDiagnostics {
                component_id: c.component_id,
                block_grid: c.block_grid,
                quant_table_base: c.quant_table_base.to_vec(),
                zero_bias: c.zero_bias.to_vec(),
                blocks: c
                    .blocks
                    .into_iter()
                    .map(|b| BlockDiagnostics {
                        coef_pre_quant: b.coef_pre_quant.to_vec(),
                        coef_levels: b.coef_levels.to_vec(),
                        aq_multiplier: b.aq_multiplier,
                        entropy_bits: b.entropy_bits,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Options controlling the encode. Mirrors the most useful knobs from
/// `EncoderConfig`. Field names use camelCase for TS-natural input.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EncodeOptions {
    /// jpegli quality (0-100). Default 85.
    #[serde(default = "default_quality")]
    pub quality: f32,
    /// "ycbcr" or "xyb". Default "ycbcr".
    #[serde(default = "default_color_path")]
    pub color_path: String,
    /// "none" (4:4:4), "halfHorizontal" (4:2:2), "quarter" (4:2:0),
    /// "halfVertical" (4:4:0). Used when colorPath = "ycbcr".
    /// Default "none".
    #[serde(default = "default_subsampling")]
    pub subsampling: String,
    /// "full" or "bQuarter". Used when colorPath = "xyb". Default "full".
    #[serde(default = "default_xyb_subsampling")]
    pub xyb_subsampling: String,
    /// Adaptive quantization on/off. Default true.
    #[serde(default = "default_true")]
    pub aq_enabled: bool,
    /// Standalone trellis quantization on/off. Default false.
    #[serde(default)]
    pub trellis: bool,
    /// Auto-optimize (hybrid AQ+trellis) on/off. Default false.
    #[serde(default)]
    pub auto_optimize: bool,
    /// Deringing on/off. Default true.
    #[serde(default = "default_true")]
    pub deringing: bool,
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

/// Output of `encodeWithDiagnostics`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncodeResult {
    /// JPEG bytes.
    pub bytes: Vec<u8>,
    /// Per-block diagnostics.
    pub diagnostics: Diagnostics,
}

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
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
        EncodeOptions {
            quality: default_quality(),
            color_path: default_color_path(),
            subsampling: default_subsampling(),
            xyb_subsampling: default_xyb_subsampling(),
            aq_enabled: true,
            trellis: false,
            auto_optimize: false,
            deringing: true,
        }
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsValue::from_str(&format!("invalid options: {e}")))?
    };

    let expected = (width as usize) * (height as usize) * 3;
    if pixels.len() != expected {
        return Err(JsValue::from_str(&format!(
            "pixels length mismatch: expected {expected} ({width}*{height}*3), \
             got {}",
            pixels.len()
        )));
    }

    let config = match opts.color_path.as_str() {
        "ycbcr" => {
            let sub = match opts.subsampling.as_str() {
                "none" | "s444" | "4:4:4" => ChromaSubsampling::None,
                "halfHorizontal" | "s422" | "4:2:2" => ChromaSubsampling::HalfHorizontal,
                "quarter" | "s420" | "4:2:0" => ChromaSubsampling::Quarter,
                "halfVertical" | "s440" | "4:4:0" => ChromaSubsampling::HalfVertical,
                other => {
                    return Err(JsValue::from_str(&format!(
                        "unknown subsampling: {other}"
                    )));
                }
            };
            EncoderConfig::ycbcr(opts.quality, sub)
        }
        "xyb" => {
            let xyb_sub = match opts.xyb_subsampling.as_str() {
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
            return Err(JsValue::from_str(&format!(
                "unknown color path: {other}"
            )));
        }
    };

    let mut config = config
        .aq_enabled(opts.aq_enabled)
        .deringing(opts.deringing)
        .with_diagnostics(true);

    #[cfg(feature = "trellis")]
    {
        if opts.auto_optimize {
            config = config.auto_optimize(true);
        } else if opts.trellis {
            use zenjpeg::encode::trellis::TrellisConfig;
            config = config.trellis(TrellisConfig::new());
        }
    }
    // Suppress unused-warning when trellis isn't enabled.
    let _ = (opts.trellis, opts.auto_optimize);

    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| JsValue::from_str(&format!("encoder build: {e}")))?;
    encoder
        .push_packed(pixels, enough::Unstoppable)
        .map_err(|e| JsValue::from_str(&format!("push pixels: {e}")))?;
    let (bytes, diag) = encoder
        .finish_with_diagnostics()
        .map_err(|e| JsValue::from_str(&format!("finish: {e}")))?;

    let diag = diag.ok_or_else(|| {
        JsValue::from_str("with_diagnostics(true) was set but encoder returned None")
    })?;

    let result = EncodeResult {
        bytes,
        diagnostics: to_js_diagnostics(diag),
    };
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("serialize result: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn to_js_diagnostics_preserves_shape() {
        let w = 32u32;
        let h = 32u32;
        let pixels = synthetic(w, h);
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .aq_enabled(true)
            .with_diagnostics(true);
        let mut enc = config
            .request()
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .expect("encoder");
        enc.push_packed(&pixels, enough::Unstoppable).expect("push");
        let (_bytes, diag) = enc.finish_with_diagnostics().expect("finish");
        let diag = diag.expect("diag");
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
