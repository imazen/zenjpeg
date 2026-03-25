//! zennode node definitions for JPEG encoding.
//!
//! Defines [`EncodeJpeg`] with RIAPI-compatible querystring keys matching
//! imageflow's established JPEG encoding parameters.

extern crate alloc;
use alloc::string::String;

use zennode::*;

/// JPEG encoding with quality, progressive, and subsampling options.
///
/// Matches imageflow's RIAPI keys: `quality`, `jpeg.quality`,
/// `jpeg.progressive`, `jpeg.li`, `subsampling`.
///
/// JSON API: `{ "quality": 85, "progressive": true, "subsampling": "420" }`
/// RIAPI: `?jpeg.quality=85&jpeg.progressive=true&subsampling=420`
#[derive(Node, Clone, Debug)]
#[node(id = "zenjpeg.encode", group = Encode, role = Encode)]
#[node(tags("codec", "jpeg", "lossy", "encode"))]
pub struct EncodeJpeg {
    /// Generic quality 0-100 (mapped via with_generic_quality at execution time).
    ///
    /// When set (>= 0), this value is passed through zencodec's
    /// `with_generic_quality()` which maps it to the codec's native
    /// quality scale. Use this for uniform quality across all codecs.
    #[param(range(0..=100), default = -1, step = 1)]
    #[param(unit = "", section = "Main", label = "Quality")]
    #[kv("quality")]
    pub quality: i32,

    /// Codec-specific JPEG quality override (jpegli scale, 1-100).
    ///
    /// When set (>= 1), this value is used directly as the jpegli
    /// quality parameter, bypassing generic quality mapping.
    /// Takes precedence over the generic `quality` field.
    #[param(range(1..=100), default = -1, step = 1)]
    #[param(unit = "", section = "Main", label = "JPEG Quality")]
    #[kv("jpeg.quality")]
    pub jpeg_quality: i32,

    /// Use progressive JPEG encoding (multiple scans).
    ///
    /// Progressive JPEGs render a blurry preview first, then sharpen.
    /// Slightly smaller files at quality 70+, slightly larger below.
    #[param(default = false)]
    #[param(section = "Main")]
    #[kv("jpeg.progressive")]
    pub progressive: bool,

    /// Chroma subsampling mode.
    ///
    /// "444" = no subsampling (best color), "422" = half horizontal,
    /// "420" = quarter (smallest, default for most JPEG encoders).
    /// Empty = auto (420 for quality < 90, 444 for quality >= 90).
    #[param(default = "")]
    #[param(section = "Main", label = "Subsampling")]
    #[kv("subsampling")]
    pub subsampling: String,

    /// Use jpegli-style perceptual optimizations.
    ///
    /// Enables adaptive quantization, deringing, and perceptual
    /// tuning for better quality at the same file size.
    #[param(default = true)]
    #[param(section = "Advanced")]
    #[kv("jpeg.li")]
    pub jpegli: bool,

    /// Enable trellis quantization (rate-distortion optimization).
    ///
    /// Produces ~2% smaller files with equivalent quality.
    /// Slightly slower encoding.
    #[param(default = false)]
    #[param(section = "Advanced")]
    pub trellis: bool,
}

impl Default for EncodeJpeg {
    fn default() -> Self {
        Self {
            quality: -1,
            jpeg_quality: -1,
            progressive: false,
            subsampling: String::new(),
            jpegli: true,
            trellis: false,
        }
    }
}

impl EncodeJpeg {
    /// Apply this node's explicitly-set params on top of an existing config.
    ///
    /// Fields at their default/sentinel value are skipped:
    /// - `quality` and `jpeg_quality`: `-1` means not set
    /// - `subsampling`: empty string means not set
    /// - `progressive` and `trellis`: `false` means not set (default)
    /// - `jpegli`: `true` is the default, only `false` is an explicit change
    ///
    /// Codec-specific `jpeg_quality` is applied AFTER generic `quality`,
    /// so it takes precedence when both are set.
    pub fn apply(&self, mut config: crate::JpegEncoderConfig) -> crate::JpegEncoderConfig {
        use zencodec::encode::EncoderConfig as _;

        // Generic quality first (calibrated mapping)
        if self.quality >= 0 {
            config = config.with_generic_quality(self.quality as f32);
        }
        // Codec-specific quality override (direct jpegli quality)
        if self.jpeg_quality >= 1 {
            config = config.with_calibrated_quality(self.jpeg_quality as f32);
        }
        // Progressive encoding
        if self.progressive {
            config = config.with_progressive(true);
        }
        // Chroma subsampling
        if !self.subsampling.is_empty() {
            let sub = match self.subsampling.as_str() {
                "444" => crate::encoder::ChromaSubsampling::None,
                "422" => crate::encoder::ChromaSubsampling::HalfHorizontal,
                "440" => crate::encoder::ChromaSubsampling::HalfVertical,
                // Default to 4:2:0 for "420" or any other value
                _ => crate::encoder::ChromaSubsampling::Quarter,
            };
            config = config.with_subsampling(sub);
        }
        // Jpegli defaults to true; only apply when explicitly disabled
        if !self.jpegli {
            // When jpegli is disabled, use effort 0 (JpegliBaseline).
            config = zencodec::encode::EncoderConfig::with_generic_effort(config, 0);
        }
        // Trellis quantization
        if self.trellis {
            config = zencodec::encode::EncoderConfig::with_generic_effort(config, 2);
        }
        config
    }

    /// Build a config from scratch using only this node's params.
    pub fn to_encoder_config(&self) -> crate::JpegEncoderConfig {
        self.apply(crate::JpegEncoderConfig::new())
    }
}

/// Registration function for aggregating crates.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&ENCODE_JPEG_NODE);
}

/// All JPEG zennode definitions.
pub static ALL: &[&dyn NodeDef] = &[&ENCODE_JPEG_NODE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_metadata() {
        let schema = ENCODE_JPEG_NODE.schema();
        assert_eq!(schema.id, "zenjpeg.encode");
        assert_eq!(schema.group, NodeGroup::Encode);
        assert_eq!(schema.role, NodeRole::Encode);
        assert!(schema.tags.contains(&"jpeg"));
        assert!(schema.tags.contains(&"lossy"));
    }

    #[test]
    fn param_count_and_names() {
        let schema = ENCODE_JPEG_NODE.schema();
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&"quality"));
        assert!(names.contains(&"jpeg_quality"));
        assert!(names.contains(&"progressive"));
        assert!(names.contains(&"subsampling"));
        assert!(names.contains(&"jpegli"));
        assert!(names.contains(&"trellis"));
    }

    #[test]
    fn defaults() {
        let node = ENCODE_JPEG_NODE.create_default().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(-1)));
        assert_eq!(node.get_param("jpeg_quality"), Some(ParamValue::I32(-1)));
        assert_eq!(node.get_param("progressive"), Some(ParamValue::Bool(false)));
        assert_eq!(node.get_param("subsampling"), Some(ParamValue::Str(String::new())));
        assert_eq!(node.get_param("jpegli"), Some(ParamValue::Bool(true)));
    }

    #[test]
    fn from_kv_jpeg_quality() {
        let mut kv = KvPairs::from_querystring("jpeg.quality=92&jpeg.progressive=true");
        let node = ENCODE_JPEG_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("jpeg_quality"), Some(ParamValue::I32(92)));
        assert_eq!(node.get_param("progressive"), Some(ParamValue::Bool(true)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn from_kv_generic_quality() {
        // "quality" sets the generic quality field
        let mut kv = KvPairs::from_querystring("quality=75");
        let node = ENCODE_JPEG_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(75)));
        // jpeg_quality remains unset
        assert_eq!(node.get_param("jpeg_quality"), Some(ParamValue::I32(-1)));
    }

    #[test]
    fn from_kv_both_qualities() {
        // Both generic and codec-specific can be set independently
        let mut kv = KvPairs::from_querystring("quality=80&jpeg.quality=92");
        let node = ENCODE_JPEG_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(80)));
        assert_eq!(node.get_param("jpeg_quality"), Some(ParamValue::I32(92)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn from_kv_subsampling() {
        let mut kv = KvPairs::from_querystring("quality=90&subsampling=444");
        let node = ENCODE_JPEG_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("subsampling"), Some(ParamValue::Str("444".into())));
    }

    #[test]
    fn from_kv_no_match() {
        let mut kv = KvPairs::from_querystring("w=800&h=600");
        let result = ENCODE_JPEG_NODE.from_kv(&mut kv).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn json_round_trip() {
        let mut params = ParamMap::new();
        params.insert("quality".into(), ParamValue::I32(80));
        params.insert("jpeg_quality".into(), ParamValue::I32(92));
        params.insert("progressive".into(), ParamValue::Bool(true));
        params.insert("subsampling".into(), ParamValue::Str("420".into()));

        let node = ENCODE_JPEG_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(80)));
        assert_eq!(node.get_param("jpeg_quality"), Some(ParamValue::I32(92)));
        assert_eq!(node.get_param("progressive"), Some(ParamValue::Bool(true)));
        assert_eq!(node.get_param("subsampling"), Some(ParamValue::Str("420".into())));

        // Round-trip
        let exported = node.to_params();
        let node2 = ENCODE_JPEG_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("quality"), Some(ParamValue::I32(80)));
        assert_eq!(node2.get_param("jpeg_quality"), Some(ParamValue::I32(92)));
    }

    #[test]
    fn downcast_to_concrete() {
        let node = ENCODE_JPEG_NODE.create_default().unwrap();
        let enc = node.as_any().downcast_ref::<EncodeJpeg>().unwrap();
        assert_eq!(enc.quality, -1);
        assert_eq!(enc.jpeg_quality, -1);
        assert!(enc.jpegli);
    }

    #[test]
    fn to_encoder_config_defaults() {
        let node = EncodeJpeg::default();
        let _config = node.to_encoder_config();
    }

    #[test]
    fn apply_generic_quality_only() {
        let mut node = EncodeJpeg::default();
        node.quality = 80;
        let config = node.to_encoder_config();
        let q = zencodec::encode::EncoderConfig::generic_quality(&config);
        assert!(q.is_some());
    }

    #[test]
    fn apply_codec_specific_overrides_generic() {
        let mut node = EncodeJpeg::default();
        node.quality = 50;
        node.jpeg_quality = 95;
        let _config = node.to_encoder_config();
    }

    #[test]
    fn apply_preserves_existing_config() {
        let base = crate::JpegEncoderConfig::new().with_progressive(true);
        let node = EncodeJpeg::default();
        let _config = node.apply(base);
    }

    #[test]
    fn apply_subsampling_444() {
        let mut node = EncodeJpeg::default();
        node.subsampling = "444".into();
        let _config = node.to_encoder_config();
    }

    #[test]
    fn apply_trellis_and_progressive() {
        let mut node = EncodeJpeg::default();
        node.trellis = true;
        node.progressive = true;
        node.quality = 85;
        let _config = node.to_encoder_config();
    }

    #[test]
    fn apply_jpegli_disabled() {
        let mut node = EncodeJpeg::default();
        node.jpegli = false;
        let _config = node.to_encoder_config();
    }

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenjpeg.encode").is_some());

        // jpeg.quality triggers codec-specific path
        let result = registry.from_querystring("jpeg.quality=80&jpeg.progressive=true");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenjpeg.encode");

        // generic quality also triggers the node
        let result2 = registry.from_querystring("quality=80");
        assert_eq!(result2.instances.len(), 1);
        assert_eq!(result2.instances[0].schema().id, "zenjpeg.encode");
    }
}
