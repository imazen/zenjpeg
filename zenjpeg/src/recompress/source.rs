//! Source-JPEG analysis: encoder family, quality, structural facts, content
//! classification. This is the input every strategy and the router consume.

use crate::decoder::{JpegMode, Subsampling};
use crate::detect::{Confidence, EncoderFamily, JpegProbe, QualityEstimate, QualityScale};

use crate::recompress::error::Error;

/// What we know about the source JPEG without decoding its coefficients.
#[derive(Debug, Clone)]
pub struct SourceAnalysis {
    /// Original bytes length (for size-ratio computation).
    pub source_len: usize,
    /// Encoder family fingerprint from `crate::detect::probe`.
    pub encoder: EncoderFamily,
    /// Quality estimate + scale + confidence.
    pub quality: QualityEstimate,
    /// Image dimensions (no padding).
    pub width: u32,
    pub height: u32,
    /// Chroma subsampling mode.
    pub subsampling: Subsampling,
    /// Baseline or progressive.
    pub mode: JpegMode,
    /// Component count: 1 = grayscale, 3 = color, 4 = CMYK.
    pub num_components: u8,
}

impl SourceAnalysis {
    /// Coarse zensim-A estimate of the source against the (unknown)
    /// original. Pulled from the per-encoder calibration anchor table.
    ///
    /// This is a rough single-axis lookup; the real call lives in
    /// [`crate::recompress::target`] which conditions on encoder + subsampling +
    /// content class.
    pub fn estimated_zensim_a_vs_reference(&self) -> f32 {
        crate::recompress::target::encoder_quality_to_estimated_zensim_a(
            &self.encoder,
            &self.quality,
        )
    }
}

/// Run header-only analysis on `jpeg_bytes`.
pub fn analyze_source(jpeg_bytes: &[u8]) -> Result<SourceAnalysis, Error> {
    let probe: JpegProbe = crate::detect::probe(jpeg_bytes).map_err(|e| {
        use crate::detect::ProbeError;
        // Thread the concrete `ProbeError` kind through before flattening to
        // a `String`: `TooShort`/`Truncated` mean "not enough bytes yet" (the
        // caller can retry with more data), which is a different failure
        // mode from a structurally malformed header (`NotJpeg` /
        // `NoQuantTables`) — see `Error::category()`.
        match e {
            ProbeError::TooShort | ProbeError::Truncated => Error::ProbeTruncated(format!("{e}")),
            _ => Error::Probe(format!("{e}")),
        }
    })?;

    // We allow YCbCr (3) and grayscale (1). CMYK is parked behind a future
    // unsupported branch until we calibrate its strategy parameters.
    if probe.num_components == 4 {
        return Err(Error::Unsupported("CMYK source"));
    }

    // We currently support YCbCr 4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 and
    // grayscale (which the probe also returns as S444 with 1 component).
    // `Subsampling` is `#[non_exhaustive]`; reject any new variant.
    match probe.subsampling {
        Subsampling::S444 | Subsampling::S422 | Subsampling::S420 | Subsampling::S440 => {}
    }

    // For "Approximate" confidence on IJG / mozjpeg, we still return the
    // probe — the strategy router widens the cell's CI rather than
    // refusing the call.
    let _ = (Confidence::Exact, QualityScale::IjgQuality); // silence unused

    Ok(SourceAnalysis {
        source_len: jpeg_bytes.len(),
        encoder: probe.encoder,
        quality: probe.quality,
        width: probe.dimensions.width,
        height: probe.dimensions.height,
        subsampling: probe.subsampling,
        mode: probe.mode,
        num_components: probe.num_components,
    })
}
