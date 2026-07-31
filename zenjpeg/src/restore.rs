//! Optional ML restoration — feature **`restore`, DEFAULT OFF**.
//!
//! One call: JPEG bytes + a caller-supplied model -> probe -> measured decode
//! policy -> guarded x1 restoration -> quantization-consistency projection ->
//! planar RGB f32. The heavy science lives in the zensr repo (SYSTEMS.md);
//! this module is the thin, maintainable production wiring with every
//! constant carrying its measurement provenance.
//!
//! Dependency surface (kept shallow deliberately): `zensr-micro` only
//! (runtime + projection + guards; its own deps are archmage + magetypes).
//! Models are NOT embedded (repo 30KB rule): callers load the f16 weight
//! file (dejpeg-class quality tier 1.16MB, realtime tier 86KB — from the
//! zensr model releases) and hand bytes to [`RestoreModel::from_f16_bytes`].
//!
//! Policy provenance (zensr benchmarks/, 2026-07-25..28):
//! - Knusperli deblock only for Annex-K-family files at est-q <= 9.5
//!   (dejpeg_proj_lowq / SYSTEMS "S6-v2": coefficient-domain correction wins
//!   only there; AQ families never).
//! - High-q identity gate: model SKIPPED at IJG/Moz q >= 94.5 or
//!   Cjpegli-family distance <= 0.6 (model loses to identity on
//!   near-pristine input; dejpeg_proj_highq_slackabs).
//! - Projection slack per family: relative (slack_calibration: turbo 0.15,
//!   mozjpeg trellis 0.35, Cjpegli-family AQ 0.45) + absolute 1.5 for
//!   integer-sample encoders / 0.5 Cjpegli (u8-pre-FDCT noise, Q=1 bands;
//!   slack_calibration_highq).
//! - 4:2:0 chroma back-projected exactly on the half-res lattice; 4:2:2 /
//!   4:4:0 left unprojected (box2x2 would corrupt); CMYK skips projection.

use crate::decoder::{Decoder, DeblockMode};
use crate::detect::{self, EncoderFamily, JpegProbe, QualityScale};
use enough::Unstoppable;
use zensr_micro::adopted::AdoptedModel;
use zensr_micro::consist::{
    project_chroma_420, project_plane, rgb_to_ycbcr_planes, ycbcr_to_rgb_planes, CoeffOrder,
    CoeffView, ProjectionConfig, ProjectionReport,
};
use zensr_micro::guards::{guarded_merge, GuardConfig, GuardReport};

/// A loaded x1 restoration model (compact topology).
pub struct RestoreModel {
    inner: AdoptedModel,
}

impl RestoreModel {
    /// Load from the zensr f16 ship format. `nf`/`nc` come from the model's
    /// meta.json (quality tier: 64/16; realtime tier: 24/8).
    pub fn from_f16_bytes(bytes: &[u8], nf: usize, nc: usize) -> Result<Self, String> {
        let raw = zensr_micro::decode_all_f16(bytes);
        AdoptedModel::load_compact(&raw, nf, nc, 1).map(|inner| Self { inner })
    }
    /// Load from raw f32 little-endian weights (dev format).
    pub fn from_f32_bytes(bytes: &[u8], nf: usize, nc: usize) -> Result<Self, String> {
        let raw: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        AdoptedModel::load_compact(&raw, nf, nc, 1).map(|inner| Self { inner })
    }
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RestoreOptions {
    /// Worker threads for the tiled model runner.
    pub threads: usize,
    /// Tile size (0 = auto).
    pub tile: usize,
    /// Apply the measured deblock policy (see module docs). Off = always
    /// pixel-exact decode.
    pub deblock_policy: bool,
    /// Skip the model on near-pristine input (measured gate; module docs).
    pub high_q_identity: bool,
    /// IJG/Mozjpeg-scale quality at or above which the model is skipped.
    /// Default 94.5 = the QUALITY tier's measured crossover. The REALTIME
    /// tier is more aggressive and turns net-negative above ~q85 (zensr
    /// benchmarks/rt24g_high_*: q85 +0.29, q90 -0.23, q93 -0.51 ssim2), so
    /// realtime deployments should use [`RestoreOptions::realtime_tier`].
    pub high_q_threshold: f32,
    /// Cjpegli-family (Butteraugli-distance scale) counterpart: skip at or
    /// below this distance. Default 0.6; realtime uses 1.0.
    pub high_q_distance: f32,
    /// Apply the S10 quantization-consistency projection (output re-encodes
    /// to the file's own coefficients; never increases error vs the truth).
    pub projection: bool,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            tile: 0,
            deblock_policy: true,
            high_q_identity: true,
            high_q_threshold: 94.5,
            high_q_distance: 0.6,
            projection: true,
        }
    }
}

impl RestoreOptions {
    pub fn with_threads(mut self, t: usize) -> Self {
        self.threads = t;
        self
    }
    /// Preset for the 84KB realtime model: identity above q82 / distance 1.0.
    /// Measured crossover — the realtime tier trades gentleness for low-q
    /// power, so the quality-tier defaults let it run where it hurts.
    pub fn realtime_tier(mut self) -> Self {
        self.high_q_threshold = 82.0;
        self.high_q_distance = 1.0;
        self
    }
}

#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct RestoreReport {
    pub used_deblock_auto: bool,
    pub skipped_model_high_q: bool,
    pub guard: GuardReport,
    /// Per-projected-plane reports (Y, then chroma where projected).
    pub projection: Vec<ProjectionReport>,
}

/// Planar RGB f32 ([3, h, w], values in [0,1]) + provenance report.
#[non_exhaustive]
pub struct Restored {
    pub planes: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub report: RestoreReport,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum RestoreError {
    Probe(String),
    Decode(String),
    UnsupportedPixels(&'static str),
}
impl core::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RestoreError::Probe(e) => write!(f, "probe: {e}"),
            RestoreError::Decode(e) => write!(f, "decode: {e}"),
            RestoreError::UnsupportedPixels(e) => write!(f, "unsupported pixels: {e}"),
        }
    }
}
impl std::error::Error for RestoreError {}

fn is_cjpegli_family(p: &JpegProbe) -> bool {
    matches!(p.encoder, EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb)
}

fn wants_knusperli(p: &JpegProbe) -> bool {
    !is_cjpegli_family(p)
        && matches!(p.quality.scale, QualityScale::IjgQuality | QualityScale::MozjpegQuality)
        && p.quality.value <= 9.5
}

fn high_q_identity(p: &JpegProbe, opts: &RestoreOptions) -> bool {
    match p.quality.scale {
        QualityScale::IjgQuality | QualityScale::MozjpegQuality => {
            p.quality.value >= opts.high_q_threshold
        }
        QualityScale::ButteraugliDistance => p.quality.value <= opts.high_q_distance,
        _ => false,
    }
}

fn slack_for(p: &JpegProbe) -> ProjectionConfig {
    let (rel, abs) = if is_cjpegli_family(p) {
        (0.45, 0.5)
    } else if matches!(p.encoder, EncoderFamily::Mozjpeg) {
        (0.35, 1.5)
    } else {
        (0.15, 1.5)
    };
    ProjectionConfig::with_slack_q(rel).with_slack_abs(abs)
}

#[derive(PartialEq)]
enum PlaneGeom {
    Full,
    HalfBoth,
    Other,
}
fn plane_geom(bw: usize, bh: usize, w: usize, h: usize) -> PlaneGeom {
    let (hf, vf) = (bw * 8 >= w, bh * 8 >= h);
    if hf && vf {
        PlaneGeom::Full
    } else if !hf && !vf && bw * 16 >= w && bh * 16 >= h {
        PlaneGeom::HalfBoth
    } else {
        PlaneGeom::Other
    }
}

/// Full x1 restoration pipeline (see module docs for the measured policy).
pub fn restore(
    data: &[u8],
    model: &RestoreModel,
    opts: &RestoreOptions,
) -> Result<Restored, RestoreError> {
    let mut report = RestoreReport::default();
    let probe = detect::probe(data).map_err(|e| RestoreError::Probe(format!("{e:?}")))?;
    let want_auto = opts.deblock_policy && wants_knusperli(&probe);
    report.used_deblock_auto = want_auto;
    let mode = if want_auto { DeblockMode::Auto } else { DeblockMode::Off };

    let dec = Decoder::new()
        .deblock(mode)
        .decode(data, Unstoppable)
        .map_err(|e| RestoreError::Decode(format!("{e:?}")))?;
    let (w32, h32) = dec.dimensions();
    let (w, h) = (w32 as usize, h32 as usize);
    let px = dec.pixels_u8().ok_or(RestoreError::UnsupportedPixels("expected u8"))?;
    if px.len() != 3 * w * h {
        return Err(RestoreError::UnsupportedPixels("expected interleaved RGB8"));
    }
    let plane = w * h;
    let mut planes = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            planes[c * plane + i] = px[i * 3 + c] as f32 / 255.0;
        }
    }
    if opts.high_q_identity && high_q_identity(&probe, opts) {
        report.skipped_model_high_q = true;
        return Ok(Restored { planes, width: w, height: h, report });
    }

    let mut sr = model.inner.upscale_tiled(&planes, h, w, opts.threads, opts.tile);
    report.guard = guarded_merge(&mut sr, &planes, h, w, 1, &GuardConfig::default());

    if opts.projection {
        let coeffs = Decoder::new()
            .decode_coefficients(data, Unstoppable)
            .map_err(|e| RestoreError::Decode(format!("coefficients: {e:?}")))?;
        let ncomp = coeffs.components.len();
        if ncomp == 1 || ncomp == 3 {
            let pcfg = slack_for(&probe);
            let (mut y, mut cb, mut cr) =
                (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
            rgb_to_ycbcr_planes(&sr, plane, &mut y, &mut cb, &mut cr);
            for (ci, comp) in coeffs.components.iter().enumerate().take(3) {
                let Some(qt) = coeffs.quant_tables[comp.quant_table_idx as usize] else {
                    continue;
                };
                let cv = CoeffView {
                    coeffs: &comp.coeffs,
                    blocks_wide: comp.blocks_wide,
                    blocks_high: comp.blocks_high,
                    order: CoeffOrder::Zigzag,
                    quant: &qt,
                };
                let target = match ci {
                    0 => &mut y,
                    1 => &mut cb,
                    _ => &mut cr,
                };
                match plane_geom(comp.blocks_wide, comp.blocks_high, w, h) {
                    PlaneGeom::Full => {
                        report.projection.push(project_plane(target, w, h, &cv, &pcfg))
                    }
                    PlaneGeom::HalfBoth => {
                        report.projection.push(project_chroma_420(target, w, h, &cv, &pcfg))
                    }
                    PlaneGeom::Other => {}
                }
            }
            ycbcr_to_rgb_planes(&y, &cb, &cr, &mut sr, plane);
        }
    }
    Ok(Restored { planes: sr, width: w, height: h, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end smoke with a synthetic (random-weight) tiny model: encode
    /// with zenjpeg itself, restore, check dims/finiteness/projection ran.
    /// Real-model quality is validated in the zensr repo's eval suite.
    #[test]
    fn restore_smoke_synthetic_model() {
        let (nf, nc) = (8usize, 2usize);
        let n = (3 * nf * 9 + nf + nf) + nc * (nf * nf * 9 + nf + nf) + (nf * 3 * 9 + 3);
        let mut s = 7u32;
        let raw: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.05
            })
            .collect();
        let bytes: Vec<u8> = raw.iter().flat_map(|f| f.to_le_bytes()).collect();
        let model = RestoreModel::from_f32_bytes(&bytes, nf, nc).unwrap();
        let (w, h) = (48usize, 40usize);
        let rgb: Vec<u8> = (0..3 * w * h).map(|i| ((i * 7) % 251) as u8).collect();
        let mut enc = crate::encoder::EncoderConfig::ycbcr(
            35u8,
            crate::encoder::ChromaSubsampling::Quarter,
        )
        .encode_from_bytes(w as u32, h as u32, crate::encoder::PixelLayout::Rgb8Srgb)
        .expect("encoder");
        enc.push_packed(&rgb, enough::Unstoppable).expect("rows");
        let jpg = enc.finish().expect("encode");
        let r = restore(&jpg, &model, &RestoreOptions::default()).expect("restore");
        assert_eq!((r.width, r.height), (w, h));
        assert!(r.planes.iter().all(|v| v.is_finite()));
        assert_eq!(r.report.projection.len(), 3, "Y + both chroma projected");
    }
}
