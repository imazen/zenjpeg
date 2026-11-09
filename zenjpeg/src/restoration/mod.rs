//! Codec adapter for the zensr artifact-removal engine.
//! JPEG parsing, limits, color management and orientation remain in zenjpeg.
use crate::decode::{DeblockMode, DecodeConfig, DecodeResult, DecodedCoefficients, OutputTarget};
use crate::error::{Error, Result};
use enough::Stop;
use zensr::RestoreError;
use zensr::source::{
    Category, Coefficients, Component, DecodeMode, Dimensions, EncodedRgb8, JpegInfo, Needs,
    Quality, Source,
};

pub use zensr::AdoptedModel;
pub use zensr::api::{Budget, Restorer, Summary};

pub(crate) struct JpegSource<'a> {
    data: &'a [u8],
    config: DecodeConfig,
    stop: &'a dyn Stop,
    info: JpegInfo,
    plain: Option<DecodeResult>,
    deblocked: Option<DecodeResult>,
    coefficients: Option<DecodedCoefficients>,
}
fn engine_error(error: impl core::fmt::Display) -> RestoreError {
    RestoreError::Decode(error.to_string())
}
impl<'a> JpegSource<'a> {
    pub(crate) fn new(data: &'a [u8], config: &DecodeConfig, stop: &'a dyn Stop) -> Result<Self> {
        let mut probe =
            crate::detect::probe(data).map_err(|e| Error::decode_error(e.to_string()))?;
        let recorded = correct_record(&mut probe);
        let jpegli_aq_off = probe
            .record
            .as_ref()
            .and_then(|r| r.current())
            .is_some_and(|entry| {
                !entry.flags.has(crate::detect::record::EncodeFlags::AQ)
                    && matches!(
                        entry.table_source,
                        crate::detect::record::TableSource::Jpegli
                            | crate::detect::record::TableSource::JpegliSharedChroma
                    )
                    && matches!(
                        probe.quality.scale,
                        crate::detect::QualityScale::ButteraugliDistance
                    )
            });
        let info = JpegInfo {
            dimensions: Dimensions {
                width: probe.dimensions.width,
                height: probe.dimensions.height,
            },
            encoder: Category(format!("{:?}", probe.encoder)),
            subsampling: Category(format!("{:?}", probe.subsampling)),
            quality: Quality {
                value: probe.quality.value,
                scale: Category(format!("{:?}", probe.quality.scale)),
            },
            num_components: probe.num_components,
            recorded,
            jpegli_aq_off,
            encoded_len: data.len(),
        };
        let mut config = config.clone();
        config.restorer = None;
        config.auto_orient = false;
        config.decode_transform = None;
        config.crop_region = None;
        config.correct_color = None;
        config.output_format = Some(crate::types::PixelFormat::Rgb);
        config.output_target = OutputTarget::default();
        config.deblock_mode = DeblockMode::Off;
        Ok(Self {
            data,
            config,
            stop,
            info,
            plain: None,
            deblocked: None,
            coefficients: None,
        })
    }
}
impl Source for JpegSource<'_> {
    fn info(&self) -> &JpegInfo {
        &self.info
    }
    fn prepare(&mut self, needs: Needs) -> core::result::Result<(), RestoreError> {
        self.stop.check().map_err(engine_error)?;
        if needs.coefficients && self.coefficients.is_none() {
            self.coefficients = Some(
                self.config
                    .decode_coefficients(self.data, self.stop)
                    .map_err(engine_error)?,
            );
        }
        if let Some(mode) = needs.pixels {
            match mode {
                DecodeMode::Plain if self.plain.is_none() => {
                    self.plain = Some(
                        self.config
                            .clone()
                            .deblock(DeblockMode::Off)
                            .decode(self.data, self.stop)
                            .map_err(engine_error)?,
                    );
                }
                DecodeMode::Deblocked if self.deblocked.is_none() => {
                    self.plain = None;
                    self.deblocked = Some(
                        self.config
                            .clone()
                            .deblock(DeblockMode::Auto)
                            .decode(self.data, self.stop)
                            .map_err(engine_error)?,
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn pixels(&self, mode: DecodeMode) -> core::result::Result<EncodedRgb8<'_>, RestoreError> {
        let decoded = match mode {
            DecodeMode::Plain => &self.plain,
            DecodeMode::Deblocked => &self.deblocked,
        }
        .as_ref()
        .ok_or(RestoreError::UnsupportedPixels(
            "codec pixels were not prepared",
        ))?;
        let data = decoded
            .pixels_u8()
            .ok_or(RestoreError::UnsupportedPixels("codec did not supply RGB8"))?;
        Ok(EncodedRgb8 {
            data,
            width: decoded.width(),
            height: decoded.height(),
            stride: decoded.width() as usize * 3,
        })
    }
    fn coefficients(&self) -> core::result::Result<Coefficients<'_>, RestoreError> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or(RestoreError::UnsupportedPixels(
                "codec coefficients were not prepared",
            ))?;
        Ok(Coefficients {
            quant_tables: &coefficients.quant_tables,
            components: coefficients
                .components
                .iter()
                .map(|c| Component {
                    coeffs: &c.coeffs,
                    blocks_wide: c.blocks_wide,
                    blocks_high: c.blocks_high,
                    quant_table_idx: c.quant_table_idx,
                    h_samp: c.h_samp,
                    v_samp: c.v_samp,
                })
                .collect(),
        })
    }
}

use crate::detect::record::{DialScale, TableSource};
use crate::detect::{Confidence, EncoderFamily, JpegProbe, QualityEstimate, QualityScale};
use crate::encoder::Quality as EncoderQuality;
/// Replace the fingerprint estimate with the recorded dial where the record
/// is verified and the table family tells us which scale the policies' curves
/// expect. Returns whether anything changed.
///
/// - Robidoux tables: the dial becomes the libjpeg quality that generated
///   them (`EncoderQuality::for_mozjpeg_tables`), family `Mozjpeg` — zenjpeg's
///   mozjpeg presets also trellis, so the mozjpeg slack is the right box.
/// - jpegli-lineage tables (`Jpegli`, `JpegliSharedChroma`): the dial becomes
///   the butteraugli distance the encoder targeted (`EncoderQuality::to_distance`),
///   family left as the fingerprint says (`Cjpegli*`, which it already does).
/// - `Custom`, `PiecewiseV4`, `GlassaLowBpp`, or a dial on the zensim scale:
///   the tables themselves remain the best estimate; untouched.
fn correct_record(probe: &mut JpegProbe) -> bool {
    let Some(e) = probe.record.as_ref().and_then(|r| r.current()).cloned() else {
        return false;
    };
    let dial = match e.quality_scale {
        DialScale::Jpegli => EncoderQuality::ApproxJpegli(e.quality),
        DialScale::Mozjpeg => {
            EncoderQuality::ApproxMozjpeg(e.quality.round().clamp(0.0, 100.0) as u8)
        }
        DialScale::Ssim2 => EncoderQuality::ApproxSsim2(e.quality),
        DialScale::Butteraugli => EncoderQuality::ApproxButteraugli(e.quality),
        _ => return false,
    };
    match e.table_source {
        TableSource::MozjpegRobidoux => {
            probe.encoder = EncoderFamily::Mozjpeg;
            probe.quality = QualityEstimate {
                value: dial.for_mozjpeg_tables() as f32,
                scale: QualityScale::MozjpegQuality,
                confidence: Confidence::Exact,
            };
            true
        }
        TableSource::Jpegli | TableSource::JpegliSharedChroma => {
            probe.quality = QualityEstimate {
                value: dial.to_distance(),
                scale: QualityScale::ButteraugliDistance,
                confidence: Confidence::Exact,
            };
            true
        }
        _ => false,
    }
}
