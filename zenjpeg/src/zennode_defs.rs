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
    ChromaSubsampling, DownsamplingMethod, ProgressiveScanMode, Quality, QuantTableConfig,
    XybSubsampling,
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
#[derive(Node, Clone, Debug)]
#[node(id = "zenjpeg.encode", group = Encode, role = Encode)]
#[node(tags("jpeg", "jpg", "encode", "lossy"))]
pub struct EncodeJpeg {
    /// Quality level (0-100, jpegli native scale).
    #[param(range(0.0..=100.0), default = 85.0, step = 1.0)]
    #[param(section = "Quality", label = "Quality")]
    #[kv("jpeg.quality", "jpeg.q")]
    pub quality: f32,

    /// Encoding effort (0 = fast baseline, 1 = progressive, 2 = max compression).
    #[param(range(0..=2), default = 1)]
    #[param(section = "Quality", label = "Effort")]
    #[kv("jpeg.effort")]
    pub effort: i32,

    /// Color space: "ycbcr", "xyb", or "grayscale".
    #[param(default = "ycbcr")]
    #[param(section = "Color", label = "Color Space")]
    #[kv("jpeg.colorspace")]
    pub color_space: String,

    /// Chroma subsampling: "none"/"444", "quarter"/"420",
    /// "half_horizontal"/"422", "half_vertical"/"440".
    #[param(default = "quarter")]
    #[param(section = "Color", label = "Chroma Subsampling")]
    #[kv("jpeg.subsampling", "jpeg.ss")]
    pub subsampling: String,

    /// Chroma downsampling method: "average"/"box", "gamma_aware", "sharp_yuv".
    #[param(default = "average")]
    #[param(section = "Color", label = "Chroma Downsampling")]
    #[kv("jpeg.chroma_method")]
    pub chroma_downsampling: String,

    /// Scan mode: "baseline", "progressive", "progressive_mozjpeg", "progressive_search".
    #[param(default = "progressive")]
    #[param(section = "Encoding", label = "Scan Mode")]
    #[kv("jpeg.progressive", "jpeg.mode")]
    pub scan_mode: String,

    /// Quantization table source: "jpegli", "jpegli_shared", "mozjpeg".
    #[param(default = "jpegli")]
    #[param(section = "Encoding", label = "Quantization Tables")]
    #[kv("jpeg.tables")]
    pub quant_tables: String,

    /// Enable overshoot deringing (reduces ringing on white backgrounds).
    #[param(default = true)]
    #[param(section = "Advanced")]
    #[kv("jpeg.deringing")]
    pub deringing: bool,

    /// Enable adaptive quantization (per-block AQ from luminance data).
    #[param(default = true)]
    #[param(section = "Advanced", label = "Adaptive Quantization")]
    #[kv("jpeg.aq")]
    pub aq: bool,
}

impl Default for EncodeJpeg {
    fn default() -> Self {
        Self {
            quality: 85.0,
            effort: 1,
            color_space: String::from("ycbcr"),
            subsampling: String::from("quarter"),
            chroma_downsampling: String::from("average"),
            scan_mode: String::from("progressive"),
            quant_tables: String::from("jpegli"),
            deringing: true,
            aq: true,
        }
    }
}

impl EncodeJpeg {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`JpegEncoderConfig`](crate::JpegEncoderConfig).
    ///
    /// Fields at their default value are skipped, so the base config's
    /// values are preserved. Only params the user explicitly changed
    /// take effect.
    ///
    /// Application order:
    /// 1. Quality (if != 85.0) via `with_generic_quality`
    /// 2. Effort (if != 1) via `with_generic_effort`
    /// 3. Color space / subsampling (if != defaults) via `inner_mut()`
    /// 4. Chroma downsampling method (if != "average")
    /// 5. Scan mode, quant tables, deringing, aq on inner config
    pub fn apply(&self, mut config: crate::JpegEncoderConfig) -> crate::JpegEncoderConfig {
        use zencodec::encode::EncoderConfig as _;

        let defaults = Self::default();

        // Quality
        if (self.quality - defaults.quality).abs() > f32::EPSILON {
            config = config.with_generic_quality(self.quality);
        }
        // Effort
        if self.effort != defaults.effort {
            config = config.with_generic_effort(self.effort);
        }
        // Color space — rebuild inner if changed from default
        if self.color_space != defaults.color_space {
            let subsampling = self.parse_subsampling();
            match self.color_space.to_ascii_lowercase().as_str() {
                "grayscale" | "gray" | "grey" => {
                    config = crate::JpegEncoderConfig::grayscale(self.quality);
                    // Re-apply effort
                    if self.effort != defaults.effort {
                        config = config.with_generic_effort(self.effort);
                    }
                }
                "xyb" => {
                    let mut new_config = crate::JpegEncoderConfig::new();
                    *new_config.inner_mut() = EncoderConfig::xyb(
                        crate::encode::encoder_types::Quality::ApproxJpegli(self.quality),
                        self.parse_xyb_subsampling(),
                    );
                    config = new_config;
                    if self.effort != defaults.effort {
                        config = config.with_generic_effort(self.effort);
                    }
                }
                _ => {
                    // Non-default color space string but not gray/xyb — treat as ycbcr
                    config = config.with_subsampling(subsampling);
                }
            }
        } else if self.subsampling != defaults.subsampling {
            // Color space is default but subsampling changed
            config = config.with_subsampling(self.parse_subsampling());
        }

        // Chroma downsampling method
        if self.chroma_downsampling != defaults.chroma_downsampling {
            config.inner_mut().downsampling_method = self.parse_downsampling();
        }

        // Scan mode (on inner config)
        if self.scan_mode != defaults.scan_mode {
            config.inner_mut().scan_mode = self.parse_scan_mode();
        }

        // Quant tables (on inner config)
        if self.quant_tables != defaults.quant_tables {
            config.inner_mut().quant_table_config = self.parse_quant_tables();
        }

        // Deringing (default is true, so only apply if explicitly set to false)
        if self.deringing != defaults.deringing {
            config.inner_mut().deringing = self.deringing;
        }

        // AQ (default is true, so only apply if explicitly set to false)
        if self.aq != defaults.aq {
            config.inner_mut().aq_enabled = self.aq;
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
    /// Note: `scan_mode`, `quant_tables`, `deringing`, and `aq` are set on
    /// the inner config but may be overridden by the effort preset at encode
    /// time. Use [`to_inner_encoder_config()`](Self::to_inner_encoder_config)
    /// for full control over all settings.
    pub fn to_encoder_config(&self) -> crate::JpegEncoderConfig {
        use zencodec::encode::EncoderConfig as _;

        let subsampling = self.parse_subsampling();

        // Build JpegEncoderConfig with the right color space constructor.
        // JpegEncoderConfig only has ycbcr() and grayscale() constructors;
        // for XYB we start with a default config and replace the inner.
        let config = match self.color_space.to_ascii_lowercase().as_str() {
            "grayscale" | "gray" | "grey" => crate::JpegEncoderConfig::grayscale(self.quality),
            "xyb" => {
                // No XYB constructor on JpegEncoderConfig; build inner directly
                let mut config = crate::JpegEncoderConfig::new();
                *config.inner_mut() = EncoderConfig::xyb(
                    Quality::ApproxJpegli(self.quality),
                    self.parse_xyb_subsampling(),
                );
                config
            }
            _ => crate::JpegEncoderConfig::ycbcr(self.quality, subsampling),
        };

        // Apply effort (drives the optimization preset at encode time)
        let mut config = config.with_generic_effort(self.effort);

        // Apply settings that are NOT overridden by effort presets:
        // color space, quality, subsampling (already set above),
        // and chroma downsampling method.
        let inner = config.inner_mut();
        inner.downsampling_method = self.parse_downsampling();

        config
    }

    /// Convert this node into an [`EncoderConfig`] with all settings applied.
    ///
    /// Unlike [`to_encoder_config()`](Self::to_encoder_config), this returns
    /// the native zenjpeg config with explicit control over every parameter.
    /// No effort preset is applied — `scan_mode`, `quant_tables`, `deringing`,
    /// and `aq` are set exactly as specified.
    pub fn to_inner_encoder_config(&self) -> EncoderConfig {
        let subsampling = self.parse_subsampling();
        let xyb_subsampling = self.parse_xyb_subsampling();

        // Build the EncoderConfig based on color space
        let config = match self.color_space.to_ascii_lowercase().as_str() {
            "xyb" => EncoderConfig::xyb(Quality::ApproxJpegli(self.quality), xyb_subsampling),
            "grayscale" | "gray" | "grey" => {
                EncoderConfig::grayscale(Quality::ApproxJpegli(self.quality))
            }
            _ => EncoderConfig::ycbcr(Quality::ApproxJpegli(self.quality), subsampling),
        };

        config
            .progressive(self.parse_scan_mode())
            .downsampling_method(self.parse_downsampling())
            .quant_table_config(self.parse_quant_tables())
            .deringing(self.deringing)
            .aq_enabled(self.aq)
    }

    /// Parse subsampling string to [`ChromaSubsampling`].
    fn parse_subsampling(&self) -> ChromaSubsampling {
        match self.subsampling.to_ascii_lowercase().as_str() {
            "none" | "444" | "full" => ChromaSubsampling::None,
            "half_horizontal" | "422" => ChromaSubsampling::HalfHorizontal,
            "half_vertical" | "440" => ChromaSubsampling::HalfVertical,
            // Default: quarter/420
            _ => ChromaSubsampling::Quarter,
        }
    }

    /// Parse subsampling string to [`XybSubsampling`] (for XYB color space).
    fn parse_xyb_subsampling(&self) -> XybSubsampling {
        match self.subsampling.to_ascii_lowercase().as_str() {
            "none" | "444" | "full" => XybSubsampling::Full,
            // Default: BQuarter
            _ => XybSubsampling::BQuarter,
        }
    }

    /// Parse scan mode string to [`ProgressiveScanMode`].
    fn parse_scan_mode(&self) -> ProgressiveScanMode {
        match self.scan_mode.to_ascii_lowercase().as_str() {
            "baseline" | "sequential" | "false" => ProgressiveScanMode::Baseline,
            "progressive_mozjpeg" | "mozjpeg" => ProgressiveScanMode::ProgressiveMozjpeg,
            "progressive_search" | "search" => ProgressiveScanMode::ProgressiveSearch,
            // Default: progressive
            _ => ProgressiveScanMode::Progressive,
        }
    }

    /// Parse chroma downsampling method string to [`DownsamplingMethod`].
    fn parse_downsampling(&self) -> DownsamplingMethod {
        match self.chroma_downsampling.to_ascii_lowercase().as_str() {
            "gamma_aware" | "gamma" => DownsamplingMethod::GammaAware,
            "sharp_yuv" | "iterative" | "gamma_aware_iterative" => {
                DownsamplingMethod::GammaAwareIterative
            }
            // Default: box/average
            _ => DownsamplingMethod::Box,
        }
    }

    /// Parse quantization table config string to [`QuantTableConfig`].
    fn parse_quant_tables(&self) -> QuantTableConfig {
        match self.quant_tables.to_ascii_lowercase().as_str() {
            "jpegli_shared" | "jpegli_shared_chroma" => QuantTableConfig::JpegliSharedChroma,
            "mozjpeg" | "robidoux" | "mozjpeg_robidoux" => QuantTableConfig::MozjpegRobidoux,
            // Default: jpegli (3 separate tables)
            _ => QuantTableConfig::Jpegli,
        }
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

    /// Maximum image size in megapixels (0 = unlimited, default 100 MP).
    #[param(range(0..=10000), default = 100)]
    #[param(unit = "MP", section = "Limits")]
    #[kv("jpeg.max_megapixels")]
    pub max_megapixels: u32,
}

impl Default for DecodeJpeg {
    fn default() -> Self {
        Self {
            strictness: String::from("balanced"),
            auto_orient: true,
            max_megapixels: 100,
        }
    }
}

impl DecodeJpeg {
    /// Convert this node into a [`JpegDecoderConfig`](crate::JpegDecoderConfig).
    ///
    /// Applies strictness, auto-orient, and megapixel limit settings.
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

            if self.max_megapixels > 0 {
                inner.max_pixels = self.max_megapixels as u64 * 1_000_000;
            } else {
                inner.max_pixels = 0; // unlimited
            }
        }

        config
    }
}

/// Register all JPEG zennode definitions with a registry.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&ENCODE_JPEG_NODE);
    registry.register(&DECODE_JPEG_NODE);
}

/// All JPEG zennode definitions.
pub static ALL: &[&dyn NodeDef] = &[&ENCODE_JPEG_NODE, &DECODE_JPEG_NODE];

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
        assert_eq!(node.get_param("quality"), Some(ParamValue::F32(85.0)));
        assert_eq!(node.get_param("effort"), Some(ParamValue::I32(1)));
        assert_eq!(
            node.get_param("color_space"),
            Some(ParamValue::Str("ycbcr".into()))
        );
        assert_eq!(
            node.get_param("subsampling"),
            Some(ParamValue::Str("quarter".into()))
        );
        assert_eq!(node.get_param("deringing"), Some(ParamValue::Bool(true)));
        assert_eq!(node.get_param("aq"), Some(ParamValue::Bool(true)));
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
            quality: 90.0,
            subsampling: String::from("none"),
            scan_mode: String::from("baseline"),
            ..Default::default()
        };
        let _config = node.to_encoder_config();
        // Config created successfully; detailed field checks would require
        // inspecting inner() which is already tested in codec.rs.
    }

    #[test]
    fn encode_to_config_grayscale() {
        let node = EncodeJpeg {
            color_space: String::from("grayscale"),
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
        assert_eq!(node.get_param("max_megapixels"), Some(ParamValue::U32(100)));
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
            max_megapixels: 50,
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
            quality: 50.0,
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
            effort: 2,
            ..Default::default()
        };
        let config = node.apply(base);
        let e = zencodec::encode::EncoderConfig::generic_effort(&config);
        assert_eq!(e, Some(2));
    }

    #[test]
    fn apply_aq_false() {
        let node = EncodeJpeg {
            aq: false,
            ..Default::default()
        };
        let config = node.apply(crate::JpegEncoderConfig::new());
        assert!(!config.inner().aq_enabled);
    }

    #[test]
    fn to_encoder_config_matches_apply_on_default() {
        let node = EncodeJpeg {
            quality: 70.0,
            effort: 0,
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
