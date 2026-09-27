//! Opt-in research binding for the existing Zq loop; no process-global model state.
use super::IterationContext;
use crate::error::{Error, Result};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use zenpredict_serving::Model;
use zensim::{BakeScorer, Fused944Session, PrecomputedReference, RgbSlice};

fn failure(error: impl core::fmt::Display) -> Error {
    Error::invalid_config(alloc::format!("Zq candidate: {error}"))
}

#[derive(Debug, Default)]
pub(super) struct ControllerUse {
    visited: AtomicUsize,
    non_neutral: AtomicUsize,
    changed: AtomicUsize,
}
impl ControllerUse {
    pub(super) fn record(&self, non_neutral: bool, changed: bool) {
        self.visited.fetch_add(1, Ordering::Relaxed);
        self.non_neutral
            .fetch_add(usize::from(non_neutral), Ordering::Relaxed);
        self.changed
            .fetch_add(usize::from(changed), Ordering::Relaxed);
    }
    fn take(&self) -> [usize; 3] {
        [&self.visited, &self.non_neutral, &self.changed].map(|v| v.swap(0, Ordering::Relaxed))
    }
}

pub(super) struct Config {
    model: Model,
    pub(super) seed_q: f32,
    mode: String,
    trace: Option<PathBuf>,
}
impl Config {
    pub(super) fn from_env(ctx: &IterationContext<'_>) -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("ZENJPEG_ZQ_BAKE") else {
            if [
                "ZENJPEG_ZQ_SEED_Q",
                "ZENJPEG_ZQ_SPATIAL",
                "ZENJPEG_ZQ_TRACE_DIR",
            ]
            .iter()
            .any(|key| std::env::var_os(key).is_some())
            {
                return Err(failure("research options require ZENJPEG_ZQ_BAKE"));
            }
            return Ok(None);
        };
        if ctx.layout != crate::encode::PixelLayout::Rgb8Srgb
            || ctx.width < 8
            || ctx.height < 8
            || ctx.target.block_artifact.is_some()
        {
            return Err(failure(
                "requires opaque packed sRGB8, dimensions >=8, and no legacy peak bound",
            ));
        }
        if !ctx.target.target.is_finite()
            || [ctx.target.max_overshoot, ctx.target.max_undershoot]
                .into_iter()
                .flatten()
                .any(|v| !v.is_finite() || v < 0.)
        {
            return Err(failure(
                "target/tolerances must be finite; tolerances nonnegative",
            ));
        }
        if std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("1") {
            return Err(failure("requires explicit ZENSIM_FORMULA_REV=1"));
        }
        let seed_q: f32 = std::env::var("ZENJPEG_ZQ_SEED_Q")
            .map_err(failure)?
            .parse()
            .map_err(failure)?;
        if !seed_q.is_finite() || !(1. ..=100.).contains(&seed_q) {
            return Err(failure("seed q must be finite in [1,100]"));
        }
        let mode = std::env::var("ZENJPEG_ZQ_SPATIAL").map_err(failure)?;
        if !matches!(mode.as_str(), "scalar" | "neutral" | "active") {
            return Err(failure("spatial mode must be scalar, neutral or active"));
        }
        let model = Model::from_bytes(&std::fs::read(path).map_err(failure)?).map_err(failure)?;
        Ok(Some(Self {
            model,
            seed_q,
            mode,
            trace: std::env::var_os("ZENJPEG_ZQ_TRACE_DIR").map(PathBuf::from),
        }))
    }
}

pub(super) struct Measurement<'a> {
    scorer: BakeScorer<'a>,
    source: &'a [[u8; 3]],
    width: usize,
    height: usize,
    pre: Option<PrecomputedReference>,
    session: Fused944Session,
    config: &'a Config,
    pass: usize,
    usage: Arc<ControllerUse>,
}
impl<'a> Measurement<'a> {
    pub(super) fn new(config: &'a Config, ctx: &'a IterationContext<'_>) -> Result<Self> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        let (source, rest) = ctx.pixels.as_chunks::<3>();
        if !rest.is_empty()
            || source.len()
                != width
                    .checked_mul(height)
                    .ok_or_else(|| failure("shape overflow"))?
        {
            return Err(failure("packed sRGB8 shape mismatch"));
        }
        let scorer = BakeScorer::new(&config.model).map_err(failure)?;
        let pre = if config.mode == "scalar" {
            None
        } else {
            Some(
                scorer
                    .precompute_reference(&RgbSlice::new(source, width, height))
                    .map_err(failure)?,
            )
        };
        if let Some(path) = &config.trace {
            std::fs::create_dir(path).map_err(failure)?;
        }
        Ok(Self {
            scorer,
            source,
            width,
            height,
            pre,
            session: Fused944Session::new(),
            config,
            pass: 0,
            usage: Arc::default(),
        })
    }

    /// `pixels` is the ordinary independent decoder's tightly packed sRGB8 output.
    pub(super) fn measure(&mut self, jpeg: &[u8], pixels: &[u8]) -> Result<(f32, Vec<f32>)> {
        let (decoded, rest) = pixels.as_chunks::<3>();
        if !rest.is_empty() || decoded.len() != self.source.len() {
            return Err(failure("decoded packed sRGB8 shape mismatch"));
        }
        let source = RgbSlice::new(self.source, self.width, self.height);
        let distorted = RgbSlice::new(decoded, self.width, self.height);
        let cols = self.width.div_ceil(8);
        let rows = self.height.div_ceil(8);
        let mut map = Vec::new();
        map.try_reserve_exact(cols * rows).map_err(failure)?;
        let score = if let Some(pre) = &self.pre {
            let spatial = self
                .scorer
                .compute_with_ref_and_attribution(
                    &source,
                    pre,
                    &distorted,
                    Some("jpeg"),
                    &mut self.session,
                    8,
                )
                .map_err(failure)?;
            if !spatial.unsupported_feature_ids().is_empty() || spatial.has_corruption_gate() {
                return Err(failure(
                    "unsupported spatial terms or discontinuous corruption gate",
                ));
            }
            for y in 0..rows {
                for x in 0..cols {
                    map.push(
                        spatial
                            .attribution()
                            .query_rect(x * 8, y * 8, (x + 1) * 8, (y + 1) * 8)
                            .abs() as f32,
                    );
                }
            }
            spatial.result().score() as f32
        } else {
            map.resize(cols * rows, 0.);
            self.scorer
                .compute(&source, &distorted, Some("jpeg"))
                .map_err(failure)?
                .score() as f32
        };
        if !score.is_finite() || map.iter().any(|v| !v.is_finite()) {
            return Err(failure("nonfinite score or block map"));
        }
        let [visited, non_neutral, changed] = self.usage.take();
        if let Some(path) = &self.config.trace {
            std::fs::write(path.join(format!("pass-{}.jpg", self.pass)), jpeg).map_err(failure)?;
            // Exact packed RGB8 pixels judged by this comparison; dimensions
            // are the fixed source dimensions in the command's input manifest.
            std::fs::write(path.join(format!("pass-{}.rgb8", self.pass)), pixels)
                .map_err(failure)?;
            if self.pre.is_some() {
                self.write_field("map", &map)?;
            }
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path.join("measurements.tsv"))
                .map_err(failure)?;
            if self.pass == 0 {
                writeln!(f, "pass\tscore\tbytes\tmap_evaluations\taq_blocks_visited\tnon_neutral_scales_consumed\taq_strengths_changed").map_err(failure)?;
            }
            writeln!(
                f,
                "{}\t{score:.9}\t{}\t{}\t{visited}\t{non_neutral}\t{changed}",
                self.pass,
                jpeg.len(),
                usize::from(self.pre.is_some())
            )
            .map_err(failure)?;
        }
        self.pass += 1;
        Ok((score, map))
    }
    pub(super) fn prepare_scales(&self, scales: &mut [f32]) -> Result<()> {
        if self.config.mode != "active" {
            scales.fill(1.);
        }
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.) {
            return Err(failure("invalid AQ scales"));
        }
        self.write_field("scales", scales)
    }
    pub(super) fn controller_use(&self) -> Arc<ControllerUse> {
        Arc::clone(&self.usage)
    }
    fn write_field(&self, kind: &str, values: &[f32]) -> Result<()> {
        if let Some(path) = &self.config.trace {
            let f = std::fs::File::create(path.join(format!("pass-{}.{kind}.f32", self.pass)))
                .map_err(failure)?;
            let mut f = std::io::BufWriter::new(f);
            for value in values {
                f.write_all(&value.to_le_bytes()).map_err(failure)?;
            }
            f.flush().map_err(failure)?;
        }
        Ok(())
    }
}
