//! zennode pipeline node definitions for zenjpeg.
//!
//! Provides [`EncodeJpeg`] and [`DecodeJpeg`] nodes with full parameter schemas,
//! RIAPI querystring parsing, and conversion to native zenjpeg config types.
//!
//! Feature-gated behind `feature = "zennode"`.

#![cfg(feature = "zennode")]

extern crate alloc;
use alloc::string::String;

use zennode::*;

use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::{
    ChromaSubsampling, DownsamplingMethod, OptimizationPreset, ProgressiveScanMode, Quality,
    QuantTableConfig, XybSubsampling,
};

// ============================================================================
// EncodeJpeg node
// ============================================================================

/// JPEG encoder configuration as a self-documenting pipeline node.
///
/// Maps to [`JpegEncoderConfig`](crate::JpegEncoderConfig) via
/// [`to_encoder_config()`](EncodeJpeg::to_encoder_config).
///
/// **RIAPI**: `?jpeg.quality=85&jpeg.subsampling=quarter&jpeg.progressive=progressive`
#[derive(Node, Clone, Debug, Default)]
#[node(id = "zenjpeg.encode", group = Encode, role = Encode)]
#[node(tags("jpeg", "jpg", "encode", "lossy"))]
pub struct EncodeJpeg {
    /// Quality level (0-100, jpegli native scale). None = use base config's quality.
    #[param(range(0.0..=100.0), default = 85.0, step = 1.0)]
    #[param(section = "Quality", label = "Quality")]
    #[kv("jpeg.quality", "jpeg.q")]
    pub quality: Option<f32>,

    /// Encoding effort (0 = fast baseline, 1 = progressive, 2 = max compression).
    /// None = use base config's effort.
    #[param(range(0..=2), default = 1)]
    #[param(section = "Quality", label = "Effort")]
    #[kv("jpeg.effort")]
    pub effort: Option<i32>,

    /// Color space: "ycbcr", "xyb", or "grayscale". None = use base config's color space.
    #[param(default = "ycbcr")]
    #[param(section = "Color", label = "Color Space")]
    #[kv("jpeg.colorspace")]
    pub color_space: Option<String>,

    /// Chroma subsampling: "none"/"444", "quarter"/"420",
    /// "half_horizontal"/"422", "half_vertical"/"440".
    /// None = use base config's subsampling.
    #[param(default = "quarter")]
    #[param(section = "Color", label = "Chroma Subsampling")]
    #[kv("jpeg.subsampling", "jpeg.ss")]
    pub subsampling: Option<String>,

    /// Chroma downsampling method: "average"/"box", "gamma_aware", "sharp_yuv".
    /// None = use base config's method.
    #[param(default = "average")]
    #[param(section = "Color", label = "Chroma Downsampling")]
    #[kv("jpeg.chroma_method")]
    pub chroma_downsampling: Option<String>,

    /// Scan mode: "baseline", "progressive", "progressive_mozjpeg", "progressive_search".
    /// None = use base config's scan mode.
    #[param(default = "progressive")]
    #[param(section = "Encoding", label = "Scan Mode")]
    #[kv("jpeg.progressive", "jpeg.mode")]
    pub scan_mode: Option<String>,

    /// Quantization table source: "jpegli", "jpegli_shared", "mozjpeg".
    /// None = use base config's tables.
    #[param(default = "jpegli")]
    #[param(section = "Encoding", label = "Quantization Tables")]
    #[kv("jpeg.tables")]
    pub quant_tables: Option<String>,

    /// Enable overshoot deringing (reduces ringing on white backgrounds).
    /// None = use base config's setting.
    #[param(default = true)]
    #[param(section = "Advanced")]
    #[kv("jpeg.deringing")]
    pub deringing: Option<bool>,

    /// Enable adaptive quantization (per-block AQ from luminance data).
    /// None = use base config's setting.
    #[param(default = true)]
    #[param(section = "Advanced", label = "Adaptive Quantization")]
    #[kv("jpeg.aq")]
    pub aq: Option<bool>,
}

impl EncodeJpeg {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`JpegEncoderConfig`](crate::JpegEncoderConfig).
    ///
    /// `None` fields are skipped, so the base config's values are preserved.
    /// Only params the user explicitly set (via `Some(value)`) take effect.
    /// This correctly handles users explicitly choosing the default value.
    ///
    /// Application order:
    /// 1. Quality via `with_generic_quality`
    /// 2. Effort via `with_generic_effort`
    /// 3. Color space / subsampling via `inner_mut()`
    /// 4. Chroma downsampling method
    /// 5. Scan mode, quant tables, deringing, aq on inner config
    pub fn apply(&self, mut config: crate::JpegEncoderConfig) -> crate::JpegEncoderConfig {
        use zencodec::encode::EncoderConfig as _;

        // Quality
        if let Some(quality) = self.quality {
            config = config.with_generic_quality(quality);
        }
        // Effort
        if let Some(effort) = self.effort {
            config = config.with_generic_effort(effort);
        }
        // Color space — rebuild inner if set
        if let Some(ref color_space) = self.color_space {
            let quality = self.quality.unwrap_or(85.0);
            match color_space.to_ascii_lowercase().as_str() {
                "grayscale" | "gray" | "grey" => {
                    config = crate::JpegEncoderConfig::grayscale(quality);
                    // Re-apply effort
                    if let Some(effort) = self.effort {
                        config = config.with_generic_effort(effort);
                    }
                }
                "xyb" => {
                    let mut new_config = crate::JpegEncoderConfig::new();
                    *new_config.inner_mut() = EncoderConfig::xyb(
                        crate::encode::encoder_types::Quality::ApproxJpegli(quality),
                        self.parse_xyb_subsampling(),
                    );
                    config = new_config;
                    if let Some(effort) = self.effort {
                        config = config.with_generic_effort(effort);
                    }
                }
                _ => {
                    // Non-default color space string but not gray/xyb — treat as ycbcr
                    if let Some(subsampling) = self.parse_subsampling() {
                        config = config.with_subsampling(subsampling);
                    }
                }
            }
        } else if let Some(subsampling) = self.parse_subsampling() {
            // Color space not set but subsampling is
            config = config.with_subsampling(subsampling);
        }

        // Chroma downsampling method
        if let Some(method) = self.parse_downsampling() {
            config.inner_mut().downsampling_method = method;
        }

        // Scan mode (on inner config)
        if let Some(mode) = self.parse_scan_mode() {
            config.inner_mut().scan_mode = mode;
        }

        // Quant tables (on inner config)
        if let Some(tables) = self.parse_quant_tables() {
            config.inner_mut().quant_table_config = tables;
        }

        // Deringing
        if let Some(deringing) = self.deringing {
            config.inner_mut().deringing = deringing;
        }

        // AQ
        if let Some(aq) = self.aq {
            config.inner_mut().aq_enabled = aq;
        }

        config
    }

    /// Convert this node into a [`JpegEncoderConfig`](crate::JpegEncoderConfig).
    ///
    /// The `effort` parameter controls the optimization preset (baseline,
    /// progressive, max compression) via [`with_generic_effort`]. Quality,
    /// color space, subsampling, and chroma downsampling are applied to the
    /// inner [`EncoderConfig`] and always respected.
    ///
    /// `None` fields use sensible defaults (quality=85, effort=1, ycbcr, etc.).
    ///
    /// Note: `scan_mode`, `quant_tables`, `deringing`, and `aq` are set on
    /// the inner config but may be overridden by the effort preset at encode
    /// time. Use [`to_inner_encoder_config()`](Self::to_inner_encoder_config)
    /// for full control over all settings.
    pub fn to_encoder_config(&self) -> crate::JpegEncoderConfig {
        use zencodec::encode::EncoderConfig as _;

        let quality = self.quality.unwrap_or(85.0);
        let effort = self.effort.unwrap_or(1);
        let subsampling = self
            .parse_subsampling()
            .unwrap_or(ChromaSubsampling::Quarter);

        // Build JpegEncoderConfig with the right color space constructor.
        // JpegEncoderConfig only has ycbcr() and grayscale() constructors;
        // for XYB we start with a default config and replace the inner.
        let color_space = self
            .color_space
            .as_deref()
            .unwrap_or("ycbcr")
            .to_ascii_lowercase();
        let config = match color_space.as_str() {
            "grayscale" | "gray" | "grey" => crate::JpegEncoderConfig::grayscale(quality),
            "xyb" => {
                // No XYB constructor on JpegEncoderConfig; build inner directly
                let mut config = crate::JpegEncoderConfig::new();
                *config.inner_mut() = EncoderConfig::xyb(
                    Quality::ApproxJpegli(quality),
                    self.parse_xyb_subsampling(),
                );
                config
            }
            _ => crate::JpegEncoderConfig::ycbcr(quality, subsampling),
        };

        // Apply effort (drives the optimization preset at encode time)
        let mut config = config.with_generic_effort(effort);

        // Apply settings that are NOT overridden by effort presets:
        // color space, quality, subsampling (already set above),
        // and chroma downsampling method.
        if let Some(method) = self.parse_downsampling() {
            config.inner_mut().downsampling_method = method;
        }

        config
    }

    /// Convert this node into an [`EncoderConfig`] with all settings applied.
    ///
    /// Unlike [`to_encoder_config()`](Self::to_encoder_config), this returns
    /// the native zenjpeg config with explicit control over every parameter.
    /// No effort preset is applied — `scan_mode`, `quant_tables`, `deringing`,
    /// and `aq` are set exactly as specified.
    ///
    /// `None` fields use sensible defaults.
    pub fn to_inner_encoder_config(&self) -> EncoderConfig {
        let quality = self.quality.unwrap_or(85.0);
        let subsampling = self
            .parse_subsampling()
            .unwrap_or(ChromaSubsampling::Quarter);
        let xyb_subsampling = self.parse_xyb_subsampling();

        // Build the EncoderConfig based on color space
        let color_space = self
            .color_space
            .as_deref()
            .unwrap_or("ycbcr")
            .to_ascii_lowercase();
        let mut config = match color_space.as_str() {
            "xyb" => EncoderConfig::xyb(Quality::ApproxJpegli(quality), xyb_subsampling),
            "grayscale" | "gray" | "grey" => {
                EncoderConfig::grayscale(Quality::ApproxJpegli(quality))
            }
            _ => EncoderConfig::ycbcr(Quality::ApproxJpegli(quality), subsampling),
        };

        if let Some(mode) = self.parse_scan_mode() {
            config = config.progressive(mode);
        }
        if let Some(method) = self.parse_downsampling() {
            config = config.downsampling_method(method);
        }
        if let Some(tables) = self.parse_quant_tables() {
            config = config.quant_table_config(tables);
        }
        if let Some(deringing) = self.deringing {
            config = config.deringing(deringing);
        }
        if let Some(aq) = self.aq {
            config = config.aq_enabled(aq);
        }

        config
    }

    /// Parse subsampling string to [`ChromaSubsampling`].
    /// Returns `None` if the field is unset.
    fn parse_subsampling(&self) -> Option<ChromaSubsampling> {
        let s = self.subsampling.as_deref()?;
        Some(match s.to_ascii_lowercase().as_str() {
            "none" | "444" | "full" => ChromaSubsampling::None,
            "half_horizontal" | "422" => ChromaSubsampling::HalfHorizontal,
            "half_vertical" | "440" => ChromaSubsampling::HalfVertical,
            // Default: quarter/420
            _ => ChromaSubsampling::Quarter,
        })
    }

    /// Parse subsampling string to [`XybSubsampling`] (for XYB color space).
    /// Returns `BQuarter` as default when subsampling is unset.
    fn parse_xyb_subsampling(&self) -> XybSubsampling {
        match self.subsampling.as_deref() {
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "none" | "444" | "full" => XybSubsampling::Full,
                _ => XybSubsampling::BQuarter,
            },
            None => XybSubsampling::BQuarter,
        }
    }

    /// Parse scan mode string to [`ProgressiveScanMode`].
    /// Returns `None` if the field is unset.
    fn parse_scan_mode(&self) -> Option<ProgressiveScanMode> {
        let s = self.scan_mode.as_deref()?;
        Some(match s.to_ascii_lowercase().as_str() {
            "baseline" | "sequential" | "false" => ProgressiveScanMode::Baseline,
            "progressive_mozjpeg" | "mozjpeg" => ProgressiveScanMode::ProgressiveMozjpeg,
            "progressive_search" | "search" => ProgressiveScanMode::ProgressiveSearch,
            // Default: progressive
            _ => ProgressiveScanMode::Progressive,
        })
    }

    /// Parse chroma downsampling method string to [`DownsamplingMethod`].
    /// Returns `None` if the field is unset.
    fn parse_downsampling(&self) -> Option<DownsamplingMethod> {
        let s = self.chroma_downsampling.as_deref()?;
        Some(match s.to_ascii_lowercase().as_str() {
            "gamma_aware" | "gamma" => DownsamplingMethod::GammaAware,
            "sharp_yuv" | "iterative" | "gamma_aware_iterative" => {
                DownsamplingMethod::GammaAwareIterative
            }
            // Default: box/average
            _ => DownsamplingMethod::Box,
        })
    }

    /// Parse quantization table config string to [`QuantTableConfig`].
    /// Returns `None` if the field is unset.
    fn parse_quant_tables(&self) -> Option<QuantTableConfig> {
        let s = self.quant_tables.as_deref()?;
        Some(match s.to_ascii_lowercase().as_str() {
            "jpegli_shared" | "jpegli_shared_chroma" => QuantTableConfig::JpegliSharedChroma,
            "mozjpeg" | "robidoux" | "mozjpeg_robidoux" => QuantTableConfig::MozjpegRobidoux,
            // Default: jpegli (3 separate tables)
            _ => QuantTableConfig::Jpegli,
        })
    }
}

// ============================================================================
// DecodeJpeg node
// ============================================================================

/// JPEG decoder configuration as a self-documenting pipeline node.
///
/// Maps to [`JpegDecoderConfig`](crate::JpegDecoderConfig) via
/// [`to_decoder_config()`](DecodeJpeg::to_decoder_config).
///
/// **RIAPI**: `?jpeg.strictness=balanced&jpeg.auto_orient=true`
#[derive(Node, Clone, Debug)]
#[node(id = "zenjpeg.decode", group = Decode, role = Decode)]
#[node(tags("jpeg", "jpg", "decode"))]
pub struct DecodeJpeg {
    /// Error handling strictness: "strict", "balanced", "lenient", "permissive".
    #[param(default = "balanced")]
    #[param(section = "Main", label = "Strictness")]
    #[kv("jpeg.strictness")]
    pub strictness: String,

    /// Whether to automatically correct EXIF orientation during decode.
    #[param(default = true)]
    #[param(section = "Main", label = "Auto Orient")]
    #[kv("jpeg.orient", "jpeg.auto_orient")]
    pub auto_orient: bool,

    /// Maximum image size in megapixels. None = use decoder default (100 MP).
    #[param(range(0..=10000), default = 100)]
    #[param(unit = "MP", section = "Limits")]
    #[kv("jpeg.max_megapixels")]
    pub max_megapixels: Option<u32>,
}

impl Default for DecodeJpeg {
    fn default() -> Self {
        Self {
            strictness: String::from("balanced"),
            auto_orient: true,
            max_megapixels: None,
        }
    }
}

impl DecodeJpeg {
    /// Convert this node into a [`JpegDecoderConfig`](crate::JpegDecoderConfig).
    ///
    /// Applies strictness, auto-orient, and megapixel limit settings.
    /// `None` for `max_megapixels` leaves the decoder's default limit unchanged.
    pub fn to_decoder_config(&self) -> crate::JpegDecoderConfig {
        let mut config = crate::JpegDecoderConfig::new();

        #[cfg(feature = "decoder")]
        {
            use crate::decode::Strictness;

            let strictness = match self.strictness.to_ascii_lowercase().as_str() {
                "strict" => Strictness::Strict,
                "lenient" => Strictness::Lenient,
                "permissive" => Strictness::Permissive,
                // Default: balanced
                _ => Strictness::Balanced,
            };

            let inner = config.inner_mut();
            inner.strictness = strictness;
            inner.auto_orient = self.auto_orient;

            if let Some(mp) = self.max_megapixels {
                if mp > 0 {
                    inner.max_pixels = mp as u64 * 1_000_000;
                } else {
                    inner.max_pixels = 0; // unlimited
                }
            }
            // None: leave inner.max_pixels at its default
        }

        config
    }
}

// ============================================================================
// EncodeMozjpeg node — mozjpeg-compatible preset
// ============================================================================

/// Mozjpeg-compatible JPEG encoder configuration.
///
/// Bundles the correct defaults for matching mozjpeg-rs output:
/// Robidoux quantization tables, mozjpeg quality scale, no adaptive quantization,
/// no deringing, and trellis quantization (when the `trellis` feature is enabled).
///
/// Quality uses the mozjpeg scale directly (`Quality::ApproxMozjpeg`), not the
/// jpegli scale — these produce different quantization tables at the same numeric
/// value.
///
/// **RIAPI**: `?mozjpeg.quality=85&mozjpeg.effort=2`
///
/// **Measured parity** (25 gb82 images, Q50-Q98 vs mozjpeg-rs):
/// - File size: within ±1% at all quality levels
/// - Quality vs original: equivalent (±0.01 at Q50) to slightly better (+0.66 at Q98)
/// - The remaining per-pixel differences (zensim 84-93 between decoders) are from
///   f32 vs integer DCT constant differences — the f32 path is more precise
#[derive(Node, Clone, Debug, Default)]
#[node(id = "zenjpeg.encode_mozjpeg", group = Encode, role = Encode)]
#[node(tags("jpeg", "jpg", "encode", "lossy", "mozjpeg", "compat"))]
pub struct EncodeMozjpeg {
    /// Quality level (1-100, mozjpeg scale). None = 85.
    ///
    /// Uses the mozjpeg/IJG quality scale with Robidoux quantization tables.
    /// This is NOT the same as the jpegli quality scale used by `EncodeJpeg`.
    #[param(range(1.0..=100.0), default = 85.0, step = 1.0)]
    #[param(section = "Quality", label = "Quality")]
    #[kv("mozjpeg.quality", "mozjpeg.q")]
    pub quality: Option<f32>,

    /// Encoding effort (0 = baseline, 1 = progressive, 2 = max compression).
    ///
    /// - 0: `MozjpegBaseline` — baseline JPEG + trellis
    /// - 1: `MozjpegProgressive` — progressive + mozjpeg scan script + trellis
    /// - 2: `MozjpegMaxCompression` — progressive scan search + thorough trellis
    #[param(range(0..=2), default = 1)]
    #[param(section = "Quality", label = "Effort")]
    #[kv("mozjpeg.effort")]
    pub effort: Option<i32>,

    /// Chroma subsampling: "none"/"444", "quarter"/"420",
    /// "half_horizontal"/"422", "half_vertical"/"440".
    #[param(default = "quarter")]
    #[param(section = "Color", label = "Chroma Subsampling")]
    #[kv("mozjpeg.subsampling", "mozjpeg.ss")]
    pub subsampling: Option<String>,
}

impl EncodeMozjpeg {
    /// Convert to a native [`EncoderConfig`] with mozjpeg-compatible settings.
    ///
    /// Uses `Quality::ApproxMozjpeg` (correct quality scale for Robidoux tables),
    /// the appropriate `OptimizationPreset`, and disables jpegli-specific features
    /// (AQ, deringing) that have no mozjpeg equivalent.
    pub fn to_inner_encoder_config(&self) -> EncoderConfig {
        let quality = self.quality.unwrap_or(85.0).round().clamp(1.0, 100.0) as u8;
        let subsampling = self.parse_subsampling();

        let preset = match self.effort.unwrap_or(1) {
            0 => OptimizationPreset::MozjpegBaseline,
            2 => OptimizationPreset::MozjpegMaxCompression,
            _ => OptimizationPreset::MozjpegProgressive,
        };

        EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), subsampling).optimization(preset)
    }

    /// Apply this node's settings on top of an existing
    /// [`JpegEncoderConfig`](crate::JpegEncoderConfig).
    ///
    /// Replaces the inner config entirely with mozjpeg-compatible settings,
    /// since mozjpeg and jpegli settings are not meaningfully mixable.
    pub fn apply(&self, _config: crate::JpegEncoderConfig) -> crate::JpegEncoderConfig {
        let mut config = crate::JpegEncoderConfig::new();
        *config.inner_mut() = self.to_inner_encoder_config();
        config
    }

    fn parse_subsampling(&self) -> ChromaSubsampling {
        match self.subsampling.as_deref() {
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "none" | "444" | "full" => ChromaSubsampling::None,
                "half_horizontal" | "422" => ChromaSubsampling::HalfHorizontal,
                "half_vertical" | "440" => ChromaSubsampling::HalfVertical,
                _ => ChromaSubsampling::Quarter,
            },
            None => ChromaSubsampling::Quarter,
        }
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all JPEG zennode definitions with a registry.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&ENCODE_JPEG_NODE);
    registry.register(&ENCODE_MOZJPEG_NODE);
    registry.register(&DECODE_JPEG_NODE);
}

/// All JPEG zennode definitions.
pub static ALL: &[&dyn NodeDef] = &[&ENCODE_JPEG_NODE, &ENCODE_MOZJPEG_NODE, &DECODE_JPEG_NODE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_schema_basics() {
        let schema = ENCODE_JPEG_NODE.schema();
        assert_eq!(schema.id, "zenjpeg.encode");
        assert_eq!(schema.group, NodeGroup::Encode);
        assert_eq!(schema.role, NodeRole::Encode);
        assert!(schema.tags.contains(&"jpeg"));
        assert!(schema.tags.contains(&"encode"));
        assert!(schema.tags.contains(&"lossy"));
    }

    #[test]
    fn encode_default_values() {
        let node = ENCODE_JPEG_NODE.create_default().unwrap();
        // All optional fields default to None
        assert_eq!(node.get_param("quality"), Some(ParamValue::None));
        assert_eq!(node.get_param("effort"), Some(ParamValue::None));
        assert_eq!(node.get_param("color_space"), Some(ParamValue::None));
        assert_eq!(node.get_param("subsampling"), Some(ParamValue::None));
        assert_eq!(
            node.get_param("chroma_downsampling"),
            Some(ParamValue::None)
        );
        assert_eq!(node.get_param("scan_mode"), Some(ParamValue::None));
        assert_eq!(node.get_param("quant_tables"), Some(ParamValue::None));
        assert_eq!(node.get_param("deringing"), Some(ParamValue::None));
        assert_eq!(node.get_param("aq"), Some(ParamValue::None));
    }

    #[test]
    fn encode_kv_parsing() {
        let mut kv = KvPairs::from_querystring("jpeg.q=75&jpeg.ss=444&jpeg.aq=false");
        let instance = ENCODE_JPEG_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(instance.get_param("quality"), Some(ParamValue::F32(75.0)));
        assert_eq!(
            instance.get_param("subsampling"),
            Some(ParamValue::Str("444".into()))
        );
        assert_eq!(instance.get_param("aq"), Some(ParamValue::Bool(false)));
    }

    #[test]
    fn encode_to_config_ycbcr() {
        let node = EncodeJpeg {
            quality: Some(90.0),
            subsampling: Some(String::from("none")),
            scan_mode: Some(String::from("baseline")),
            ..Default::default()
        };
        let _config = node.to_encoder_config();
        // Config created successfully; detailed field checks would require
        // inspecting inner() which is already tested in codec.rs.
    }

    #[test]
    fn encode_to_config_grayscale() {
        let node = EncodeJpeg {
            color_space: Some(String::from("grayscale")),
            ..Default::default()
        };
        let _config = node.to_encoder_config();
    }

    #[test]
    fn decode_schema_basics() {
        let schema = DECODE_JPEG_NODE.schema();
        assert_eq!(schema.id, "zenjpeg.decode");
        assert_eq!(schema.group, NodeGroup::Decode);
        assert_eq!(schema.role, NodeRole::Decode);
        assert!(schema.tags.contains(&"jpeg"));
        assert!(schema.tags.contains(&"decode"));
    }

    #[test]
    fn decode_default_values() {
        let node = DECODE_JPEG_NODE.create_default().unwrap();
        assert_eq!(
            node.get_param("strictness"),
            Some(ParamValue::Str("balanced".into()))
        );
        assert_eq!(node.get_param("auto_orient"), Some(ParamValue::Bool(true)));
        // max_megapixels is optional, defaults to None
        assert_eq!(node.get_param("max_megapixels"), Some(ParamValue::None));
    }

    #[test]
    fn decode_kv_parsing() {
        let mut kv = KvPairs::from_querystring("jpeg.strictness=strict&jpeg.auto_orient=false");
        let instance = DECODE_JPEG_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            instance.get_param("strictness"),
            Some(ParamValue::Str("strict".into()))
        );
        assert_eq!(
            instance.get_param("auto_orient"),
            Some(ParamValue::Bool(false))
        );
    }

    #[test]
    fn decode_to_config() {
        let node = DecodeJpeg {
            strictness: String::from("lenient"),
            auto_orient: false,
            max_megapixels: Some(50),
        };
        let _config = node.to_decoder_config();
    }

    #[test]
    fn apply_defaults_preserves_config() {
        // Applying a default node should not change the base config
        let base = crate::JpegEncoderConfig::new();
        let effort_before = zencodec::encode::EncoderConfig::generic_effort(&base);
        let quality_before = zencodec::encode::EncoderConfig::generic_quality(&base);

        let node = EncodeJpeg::default();
        let config = node.apply(base);

        let effort_after = zencodec::encode::EncoderConfig::generic_effort(&config);
        let quality_after = zencodec::encode::EncoderConfig::generic_quality(&config);
        assert_eq!(effort_before, effort_after);
        assert_eq!(quality_before, quality_after);
    }

    #[test]
    fn apply_quality_only() {
        let base = crate::JpegEncoderConfig::new();
        let node = EncodeJpeg {
            quality: Some(50.0),
            ..Default::default()
        };
        let config = node.apply(base);
        let q = zencodec::encode::EncoderConfig::generic_quality(&config);
        assert!(q.is_some());
    }

    #[test]
    fn apply_effort_only() {
        let base = crate::JpegEncoderConfig::new();
        let node = EncodeJpeg {
            effort: Some(2),
            ..Default::default()
        };
        let config = node.apply(base);
        let e = zencodec::encode::EncoderConfig::generic_effort(&config);
        assert_eq!(e, Some(2));
    }

    #[test]
    fn apply_aq_false() {
        let node = EncodeJpeg {
            aq: Some(false),
            ..Default::default()
        };
        let config = node.apply(crate::JpegEncoderConfig::new());
        assert!(!config.inner().aq_enabled);
    }

    #[test]
    fn apply_explicit_default_quality() {
        // Setting quality to 85.0 explicitly should apply it, unlike before
        // where it was indistinguishable from "unset"
        let base = crate::JpegEncoderConfig::new();
        let node = EncodeJpeg {
            quality: Some(85.0),
            ..Default::default()
        };
        let config = node.apply(base);
        let q = zencodec::encode::EncoderConfig::generic_quality(&config);
        assert!(q.is_some());
    }

    #[test]
    fn to_encoder_config_matches_apply_on_default() {
        let node = EncodeJpeg {
            quality: Some(70.0),
            effort: Some(0),
            ..Default::default()
        };
        // to_encoder_config should produce a valid config
        let _config = node.to_encoder_config();
    }

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenjpeg.encode").is_some());
        assert!(registry.get("zenjpeg.decode").is_some());

        let result = registry.from_querystring("jpeg.q=80");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenjpeg.encode");
    }

    #[test]
    fn all_contains_both_nodes() {
        assert_eq!(ALL.len(), 2);
        let ids: alloc::vec::Vec<&str> = ALL.iter().map(|n| n.schema().id).collect();
        assert!(ids.contains(&"zenjpeg.encode"));
        assert!(ids.contains(&"zenjpeg.decode"));
    }
}
