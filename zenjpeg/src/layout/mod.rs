//! Layout pipeline: lossless transforms + lossy decode → resize → encode.
//!
//! Integrates [`zenlayout`] for geometry computation and [`zenresize`] for image
//! resampling. Automatically selects the lossless DCT-domain path when only
//! orientation changes are requested (zero generation loss, ~3-5x faster).
//!
//! # Example
//!
//! ```rust,ignore
//! use zenjpeg::layout::{LayoutConfig, LayoutResult};
//!
//! // Lossless auto-orient (uses DCT-domain transforms)
//! let result = LayoutConfig::new(85.0)
//!     .request(&jpeg_data)
//!     .auto_orient(6)
//!     .execute(&enough::Unstoppable)?;
//! assert!(result.lossless);
//!
//! // Lossy resize (decode → resize → encode)
//! let result = LayoutConfig::new(85.0)
//!     .request(&jpeg_data)
//!     .auto_orient(6)
//!     .fit(800, 600)
//!     .execute(&enough::Unstoppable)?;
//! assert!(!result.lossless);
//! ```

mod gainmap;
mod lossless;
mod lossy;

use alloc::vec::Vec;

use enough::Stop;
use zenlayout::{Command, Constraint, ConstraintMode, FlipAxis, Rotation, SourceCrop};

use crate::decode::DecodeConfig;
use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::ChromaSubsampling;
use crate::error::Result;
pub use crate::lossless::EdgeHandling;

/// Layout pipeline configuration. Reusable across operations.
///
/// Controls encode quality, resize filter, and lossless edge handling.
/// Create with [`LayoutConfig::new`] and customize with builder methods.
#[derive(Clone)]
pub struct LayoutConfig {
    quality: f32,
    subsampling: ChromaSubsampling,
    progressive: bool,
    auto_optimize: bool,
    filter: zenresize::Filter,
    edge_handling: EdgeHandling,
    pub(crate) fancy_upsampling: bool,
}

impl LayoutConfig {
    /// Create a layout config with the given encode quality (0-100).
    ///
    /// Defaults: 4:2:0 subsampling, progressive encoding, Robidoux filter,
    /// trim partial blocks for lossless, fancy upsampling for decode.
    pub fn new(quality: impl Into<f32>) -> Self {
        Self {
            quality: quality.into(),
            subsampling: ChromaSubsampling::Quarter,
            progressive: true,
            auto_optimize: true,
            filter: zenresize::Filter::default(),
            edge_handling: EdgeHandling::TrimPartialBlocks,
            fancy_upsampling: true,
        }
    }

    /// Set the resize filter (default: Robidoux).
    pub fn with_filter(mut self, f: zenresize::Filter) -> Self {
        self.filter = f;
        self
    }

    /// Set edge handling for lossless transforms (default: TrimPartialBlocks).
    pub fn with_edge_handling(mut self, eh: EdgeHandling) -> Self {
        self.edge_handling = eh;
        self
    }

    /// Set progressive encoding (default: true).
    pub fn with_progressive(mut self, p: bool) -> Self {
        self.progressive = p;
        self
    }

    /// Set chroma subsampling (default: Quarter / 4:2:0).
    pub fn with_subsampling(mut self, s: ChromaSubsampling) -> Self {
        self.subsampling = s;
        self
    }

    /// Set fancy (bilinear) upsampling for decode (default: true).
    /// Set to false for box-filter upsampling (faster, less quality).
    pub fn with_fancy_upsampling(mut self, f: bool) -> Self {
        self.fancy_upsampling = f;
        self
    }

    /// Enable hybrid trellis auto-optimization (default: true).
    ///
    /// When enabled, uses `auto_optimize()` on the encoder config for
    /// better rate-distortion tradeoffs at the cost of encode speed.
    pub fn with_auto_optimize(mut self, enable: bool) -> Self {
        self.auto_optimize = enable;
        self
    }

    /// Create a layout request for the given JPEG data.
    pub fn request<'a>(&'a self, jpeg_data: &'a [u8]) -> LayoutRequest<'a> {
        LayoutRequest {
            config: self,
            jpeg_data,
            commands: Vec::new(),
            optimize_for_decode: false,
        }
    }

    /// Build an `EncoderConfig` from layout settings.
    pub(crate) fn build_encoder_config(&self) -> EncoderConfig {
        let config =
            EncoderConfig::ycbcr(self.quality, self.subsampling).progressive(self.progressive);
        #[cfg(feature = "trellis")]
        let config = config.auto_optimize(self.auto_optimize);
        config
    }
}

/// A layout request for a specific JPEG image.
///
/// Collects layout commands (orient, crop, resize) and executes them
/// against the JPEG data, automatically selecting the optimal path.
pub struct LayoutRequest<'a> {
    config: &'a LayoutConfig,
    jpeg_data: &'a [u8],
    commands: Vec<Command>,
    optimize_for_decode: bool,
}

impl<'a> LayoutRequest<'a> {
    /// Apply EXIF orientation correction (1-8).
    pub fn auto_orient(mut self, exif_orientation: u8) -> Self {
        self.commands.push(Command::AutoOrient(exif_orientation));
        self
    }

    /// Rotate 90 degrees clockwise.
    pub fn rotate_90(mut self) -> Self {
        self.commands.push(Command::Rotate(Rotation::Rotate90));
        self
    }

    /// Rotate 180 degrees.
    pub fn rotate_180(mut self) -> Self {
        self.commands.push(Command::Rotate(Rotation::Rotate180));
        self
    }

    /// Rotate 270 degrees clockwise (90 counter-clockwise).
    pub fn rotate_270(mut self) -> Self {
        self.commands.push(Command::Rotate(Rotation::Rotate270));
        self
    }

    /// Flip horizontally.
    pub fn flip_h(mut self) -> Self {
        self.commands.push(Command::Flip(FlipAxis::Horizontal));
        self
    }

    /// Flip vertically.
    pub fn flip_v(mut self) -> Self {
        self.commands.push(Command::Flip(FlipAxis::Vertical));
        self
    }

    /// Crop to a pixel region.
    pub fn crop(mut self, crop: SourceCrop) -> Self {
        self.commands.push(Command::Crop(crop));
        self
    }

    /// Fit within the given dimensions (may upscale).
    pub fn fit(mut self, w: u32, h: u32) -> Self {
        self.commands.push(Command::Constrain(Constraint::new(
            ConstraintMode::Fit,
            w,
            h,
        )));
        self
    }

    /// Fit within the given dimensions (never upscale).
    pub fn within(mut self, w: u32, h: u32) -> Self {
        self.commands.push(Command::Constrain(Constraint::new(
            ConstraintMode::Within,
            w,
            h,
        )));
        self
    }

    /// Fill and crop to exact dimensions (may upscale).
    pub fn fit_crop(mut self, w: u32, h: u32) -> Self {
        self.commands.push(Command::Constrain(Constraint::new(
            ConstraintMode::FitCrop,
            w,
            h,
        )));
        self
    }

    /// Add an arbitrary layout command.
    pub fn command(mut self, cmd: Command) -> Self {
        self.commands.push(cmd);
        self
    }

    /// Optimize output for fast parallel decoding.
    ///
    /// Losslessly converts progressive JPEGs to baseline sequential and adds
    /// restart markers (every 4 MCU rows) for parallel decode. On the lossy
    /// path, forces baseline output with restart markers.
    ///
    /// Combined with orientation transforms, this is still lossless — the
    /// restructure handles both the transform and scan conversion in one pass.
    pub fn optimize_for_decode(mut self) -> Self {
        self.optimize_for_decode = true;
        self
    }

    /// Execute the layout pipeline.
    ///
    /// Automatically selects the lossless DCT-domain path when only orientation
    /// changes are requested (no resize, no crop). Falls back to lossy
    /// decode → resize → encode otherwise.
    ///
    /// For UltraHDR JPEGs with gain maps, the gain map is detected, extracted,
    /// transformed proportionally to the primary, and reassembled with MPF.
    pub fn execute(self, stop: &dyn Stop) -> Result<LayoutResult> {
        stop.check()?;

        // Parse JPEG header for dimensions and metadata
        let decoder = DecodeConfig::new();
        let info = decoder.read_info(self.jpeg_data)?;
        let src_w = info.dimensions.width;
        let src_h = info.dimensions.height;

        // Detect UltraHDR gain map
        let gain_map_jpeg = self.detect_and_extract_gainmap(&info);

        // Check EXIF for auto_orient commands that reference the source
        let mut commands = self.resolve_auto_orient(&info);

        // When optimizing for decode, auto-apply EXIF orientation if it can be
        // done without trimming any pixels. Only inject if the user didn't
        // already add an AutoOrient command.
        let auto_oriented = if self.optimize_for_decode {
            let has_explicit_orient = commands
                .iter()
                .any(|cmd| matches!(cmd, Command::AutoOrient(_)));
            if !has_explicit_orient {
                if let Some(exif_val) = lossless::safe_auto_orient(&info) {
                    commands.insert(0, Command::AutoOrient(exif_val));
                    true
                } else {
                    false
                }
            } else {
                // User explicitly requested orient — they handle EXIF reset
                false
            }
        } else {
            false
        };

        // Try lossless path first
        if let Some(transform) = lossless::detect_lossless(&commands) {
            let primary = if self.optimize_for_decode {
                // Early exit: if already baseline and decode-ready (small images
                // don't need DRI; larger ones need MCU-row-aligned DRI).
                if transform == crate::lossless::LosslessTransform::None
                    && info.mode == crate::types::JpegMode::Baseline
                    && lossless::is_decode_ready(self.jpeg_data, &info)
                {
                    self.jpeg_data.to_vec()
                } else {
                    // Use restructure: converts progressive→sequential, adds DRI,
                    // and optionally applies the spatial transform in one pass.
                    lossless::execute_restructure(
                        self.jpeg_data,
                        transform,
                        self.config.edge_handling,
                        stop,
                    )?
                }
            } else {
                lossless::execute_lossless(
                    self.jpeg_data,
                    transform,
                    self.config.edge_handling,
                    stop,
                )?
            };

            // Compute output dimensions
            let (out_w, out_h) = if transform.swaps_dimensions() {
                (src_h, src_w)
            } else {
                (src_w, src_h)
            };

            // Transform and reattach gain map if present.
            // Skip for identity (None) — the input already includes the gain map.
            let mut data = if transform != crate::lossless::LosslessTransform::None {
                if let Some(gm_bytes) = gain_map_jpeg {
                    let gm_fn = if self.optimize_for_decode {
                        lossless::execute_restructure
                    } else {
                        lossless::execute_lossless
                    };
                    let gm_transformed =
                        gm_fn(&gm_bytes, transform, self.config.edge_handling, stop)?;
                    gainmap::assemble_ultrahdr(primary, gm_transformed)
                } else {
                    primary
                }
            } else {
                primary
            };

            // Reset EXIF orientation to 1 when we auto-applied it on the lossless path.
            // The restructure preserves metadata as-is, so we patch the output.
            if auto_oriented && transform != crate::lossless::LosslessTransform::None {
                lossless::reset_exif_orientation_in_jpeg(&mut data);
            }

            return Ok(LayoutResult {
                data,
                lossless: true,
                width: out_w,
                height: out_h,
            });
        }

        // Lossy path: compute full layout plan via zenlayout
        let (ideal, request) = self.compute_layout(&commands, src_w, src_h)?;
        let offer = zenlayout::DecoderOffer::full_decode(src_w, src_h);
        let plan = ideal.finalize(&request, &offer);
        let target_w = plan.canvas.width;
        let target_h = plan.canvas.height;

        // Detect if any orientation commands are present (for EXIF orientation reset)
        let has_orientation = commands.iter().any(|cmd| {
            matches!(
                cmd,
                Command::AutoOrient(o) if *o != 1
            ) || matches!(cmd, Command::Rotate(_) | Command::Flip(_))
        });

        let primary = lossy::execute_lossy(
            self.jpeg_data,
            &info,
            self.config,
            &plan,
            has_orientation,
            self.optimize_for_decode,
            stop,
        )?;

        // Transform gain map proportionally if present
        let data = match gain_map_jpeg {
            Some(gm_bytes) => match self
                .transform_gainmap_lossy(&gm_bytes, src_w, src_h, target_w, target_h, stop)?
            {
                Some(gm_transformed) => gainmap::assemble_ultrahdr(primary, gm_transformed),
                None => primary,
            },
            None => primary,
        };

        Ok(LayoutResult {
            data,
            lossless: false,
            width: target_w,
            height: target_h,
        })
    }

    /// Detect UltraHDR content and extract the gain map JPEG if present.
    fn detect_and_extract_gainmap(&self, info: &crate::decode::JpegInfo) -> Option<Vec<u8>> {
        let xmp = info.xmp.as_deref()?;
        if !gainmap::is_ultrahdr_xmp(xmp) {
            return None;
        }
        gainmap::find_secondary_jpeg(self.jpeg_data)
    }

    /// Transform the gain map through the lossy path with proportional dimensions.
    fn transform_gainmap_lossy(
        &self,
        gm_bytes: &[u8],
        primary_src_w: u32,
        primary_src_h: u32,
        primary_dst_w: u32,
        primary_dst_h: u32,
        stop: &dyn Stop,
    ) -> Result<Option<Vec<u8>>> {
        let decoder = DecodeConfig::new();
        let gm_info = match decoder.read_info(gm_bytes) {
            Ok(info) => info,
            Err(_) => return Ok(None), // Corrupted gain map — skip silently
        };
        let gm_src_w = gm_info.dimensions.width;
        let gm_src_h = gm_info.dimensions.height;

        let (gm_dst_w, gm_dst_h) = gainmap::compute_gainmap_target(
            primary_src_w,
            primary_src_h,
            primary_dst_w,
            primary_dst_h,
            gm_src_w,
            gm_src_h,
        );

        let gm_transformed =
            lossy::resize_simple(gm_bytes, &gm_info, self.config, gm_dst_w, gm_dst_h, stop)?;

        Ok(Some(gm_transformed))
    }

    /// Resolve auto_orient commands that read from source EXIF.
    fn resolve_auto_orient(&self, info: &crate::decode::JpegInfo) -> Vec<Command> {
        self.commands
            .iter()
            .map(|cmd| {
                if let Command::AutoOrient(0) = cmd {
                    // Auto-orient with 0 means "read from EXIF"
                    let exif_orient = info
                        .exif
                        .as_ref()
                        .and_then(|e| crate::lossless::parse_exif_orientation(e))
                        .unwrap_or(1);
                    Command::AutoOrient(exif_orient)
                } else {
                    cmd.clone()
                }
            })
            .collect()
    }

    /// Compute full layout via zenlayout, returning the ideal layout and decoder request.
    fn compute_layout(
        &self,
        commands: &[Command],
        src_w: u32,
        src_h: u32,
    ) -> Result<(zenlayout::IdealLayout, zenlayout::DecoderRequest)> {
        zenlayout::compute_layout(commands, src_w, src_h, None)
            .map_err(|e| crate::error::Error::invalid_config(alloc::format!("layout error: {e}")))
    }
}

/// Result of a layout operation.
pub struct LayoutResult {
    /// Output JPEG bytes.
    pub data: Vec<u8>,
    /// Whether the lossless DCT-domain path was used.
    pub lossless: bool,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

// Re-export useful types from dependencies for ergonomics.
pub use zenlayout::Command as LayoutCommand;
pub use zenresize::Filter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encoder_config::EncoderConfig;
    use crate::encode::encoder_types::{ChromaSubsampling, PixelLayout};
    use enough::Unstoppable;

    /// Create a small test JPEG for layout tests.
    fn make_test_jpeg(width: u32, height: u32) -> Vec<u8> {
        let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
        let pixel_count = (width * height) as usize;
        let mut pixels = alloc::vec![0u8; pixel_count * 3];
        // Simple gradient pattern
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * 3;
                pixels[idx] = (x * 255 / width.max(1)) as u8;
                pixels[idx + 1] = (y * 255 / height.max(1)) as u8;
                pixels[idx + 2] = 128;
            }
        }
        let mut encoder = config
            .request()
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        encoder.push_packed(&pixels, Unstoppable).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn identity_is_lossless() {
        let jpeg = make_test_jpeg(64, 64);
        let result = LayoutConfig::new(85.0)
            .request(&jpeg)
            .execute(&Unstoppable)
            .unwrap();
        assert!(result.lossless);
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
    }

    #[test]
    fn rotate_90_is_lossless() {
        // Use MCU-aligned dimensions (divisible by 16 for 4:2:0)
        let jpeg = make_test_jpeg(64, 48);
        let result = LayoutConfig::new(85.0)
            .request(&jpeg)
            .rotate_90()
            .execute(&Unstoppable)
            .unwrap();
        assert!(result.lossless);
        // Rotate90 swaps dimensions
        assert_eq!(result.width, 48);
        assert_eq!(result.height, 64);
    }

    #[test]
    fn flip_h_is_lossless() {
        let jpeg = make_test_jpeg(64, 64);
        let result = LayoutConfig::new(85.0)
            .request(&jpeg)
            .flip_h()
            .execute(&Unstoppable)
            .unwrap();
        assert!(result.lossless);
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
    }

    #[test]
    fn fit_is_lossy() {
        let jpeg = make_test_jpeg(64, 64);
        let result = LayoutConfig::new(85.0)
            .request(&jpeg)
            .fit(32, 32)
            .execute(&Unstoppable)
            .unwrap();
        assert!(!result.lossless);
        assert_eq!(result.width, 32);
        assert_eq!(result.height, 32);
    }

    #[test]
    fn within_no_upscale() {
        let jpeg = make_test_jpeg(64, 64);
        // Target larger than source — Within mode should not upscale
        let result = LayoutConfig::new(85.0)
            .request(&jpeg)
            .within(256, 256)
            .execute(&Unstoppable)
            .unwrap();
        // Within with larger target still goes through lossy path (conservative detection)
        // but dimensions should match source
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
    }

    #[test]
    fn orient_plus_resize_is_lossy() {
        let jpeg = make_test_jpeg(64, 64);
        let result = LayoutConfig::new(85.0)
            .request(&jpeg)
            .auto_orient(6) // Rotate90
            .fit(32, 32)
            .execute(&Unstoppable)
            .unwrap();
        assert!(!result.lossless);
        assert_eq!(result.width, 32);
        assert_eq!(result.height, 32);
    }
}
