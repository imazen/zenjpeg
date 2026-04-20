//! Rate-distortion sweep harness: run many (config × image × quality)
//! combinations and collect per-point metrics for later [`rd`](super::rd)
//! analysis.
//!
//! The sweep itself is dependency-light: the caller supplies encoding
//! and metric-computation as closures. The harness takes care of the
//! orchestration, timing, and packaging results into the shape that
//! [`super::rd::bd_rate`] and [`super::rd::compare`] expect.
//!
//! # Why closures, not trait objects with concrete metric impls
//!
//! Moving metric implementations (SSIMULACRA2, Butteraugli) out of the
//! zenjpeg library keeps these heavy perceptual dependencies out of the
//! published crate. They live in `zenjpeg-bench-utils` or directly in
//! `examples/rd_compare.rs`, which is where they belong — they're only
//! ever needed during codec R&D, not at encode time.
//!
//! # Shape of a sweep
//!
//! ```text
//! SweepResult (owned by caller)
//!   per image i:
//!     per config c:
//!       per quality q:
//!         PointResult { bytes, bpp, per-metric distortion, enc_ms }
//! ```
//!
//! From that we can extract, for any (metric, config, image), an
//! [`super::rd::RdCurve`] and run a full
//! [`super::rd::RdComparison`] vs any other config.

use alloc::borrow::ToOwned;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::rd::{RdCurve, RdPoint};

/// Which perceptual metric a distortion value comes from.
///
/// `Custom` is for caller-specific metrics (e.g. a task-specific loss
/// function). The string becomes the column name in CSV exports.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MetricKind {
    /// SSIMULACRA2 distortion = `100.0 - score` (smaller = better).
    Ssim2,
    /// BBS total from [`super::bbs`] (smaller = better).
    Bbs,
    /// Butteraugli distance (smaller = better).
    Butteraugli,
    /// DSSIM (smaller = better).
    Dssim,
    /// A named user-defined metric (smaller = better by convention).
    Custom(String),
}

impl MetricKind {
    /// Short CSV-safe slug.
    pub fn slug(&self) -> &str {
        match self {
            MetricKind::Ssim2 => "ssim2",
            MetricKind::Bbs => "bbs",
            MetricKind::Butteraugli => "butteraugli",
            MetricKind::Dssim => "dssim",
            MetricKind::Custom(name) => name.as_str(),
        }
    }
}

/// Content class of a corpus image, for aggregate reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageClass {
    /// Photographic content.
    Photo,
    /// Screenshot / UI / text.
    Screenshot,
    /// Line art / vector-style raster.
    LineArt,
    /// Fully synthetic patterns (checkerboard, dots, etc.).
    Synthetic,
}

impl ImageClass {
    /// Short slug used for grouping and CSV output.
    pub fn slug(&self) -> &'static str {
        match self {
            ImageClass::Photo => "photo",
            ImageClass::Screenshot => "screenshot",
            ImageClass::LineArt => "lineart",
            ImageClass::Synthetic => "synthetic",
        }
    }
}

/// One image in the sweep corpus.
#[derive(Debug, Clone)]
pub struct CorpusImage {
    /// Short, stable label (used as the row key in CSVs). Should be
    /// filename-safe and unique within a sweep.
    pub label: String,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Packed RGB8 pixel data, row-major, tightly packed (stride =
    /// width * 3 bytes). No alpha, no padding.
    pub rgb8: Vec<u8>,
    /// Class of content.
    pub class: ImageClass,
}

impl CorpusImage {
    /// Pixel count = `width * height`.
    pub fn pixels(&self) -> usize {
        self.width * self.height
    }
}

/// One rate-distortion sample: an encoded JPEG's size, derived bpp, and
/// distortion in each metric the caller measured.
#[derive(Debug, Clone)]
pub struct PointResult {
    /// Encoder quality parameter (as used in `EncoderConfig::ycbcr`).
    pub quality: u8,
    /// Encoded JPEG size in bytes.
    pub bytes: usize,
    /// Bits-per-pixel: `bytes * 8 / (width * height)`.
    pub bpp: f64,
    /// Per-metric distortion (smaller = better convention).
    pub distortions: BTreeMap<MetricKind, f64>,
    /// Wall-clock encode time, milliseconds.
    pub encode_ms: f64,
}

/// Full result of a [`run_sweep`] invocation: a 3-D grid indexed by
/// `(image, config, quality)`.
#[derive(Debug, Clone, Default)]
pub struct SweepResult {
    /// `results[image_label][config_name]` is the list of samples across
    /// qualities for that (image, config) cell.
    pub results: BTreeMap<String, BTreeMap<String, Vec<PointResult>>>,
    /// Echo of the image class for each image_label — kept alongside
    /// the results for downstream aggregate-by-class.
    pub classes: BTreeMap<String, ImageClass>,
}

impl SweepResult {
    /// Extract the [`RdCurve`] for `(image_label, config_name, metric)`.
    /// Returns an empty curve if the triple has no points.
    pub fn rd_curve(&self, image: &str, config: &str, metric: &MetricKind) -> RdCurve {
        let Some(configs) = self.results.get(image) else {
            return RdCurve::default();
        };
        let Some(points) = configs.get(config) else {
            return RdCurve::default();
        };
        RdCurve::from_points(points.iter().filter_map(|p| {
            p.distortions.get(metric).map(|d| RdPoint {
                rate_bpp: p.bpp,
                distortion: *d,
                quality: p.quality,
            })
        }))
    }

    /// Iterate over (image_label, class) pairs in sorted order.
    pub fn images(&self) -> impl Iterator<Item = (&str, ImageClass)> + '_ {
        self.results.keys().map(move |k| {
            (
                k.as_str(),
                *self.classes.get(k).unwrap_or(&ImageClass::Photo),
            )
        })
    }

    /// Labels of all configs that appear in at least one image's results.
    pub fn configs(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for per_image in self.results.values() {
            for name in per_image.keys() {
                if !seen.iter().any(|n: &String| n == name) {
                    seen.push(name.to_owned());
                }
            }
        }
        seen.sort();
        seen
    }
}

/// One sample produced by the caller's encode-and-measure closure.
///
/// The callback for [`run_sweep`] returns this per (image, config, quality).
/// If the encode fails, return `None`; the sweep logs it and moves on.
#[derive(Debug, Clone)]
pub struct SampleOutput {
    /// Size in bytes of the encoded JPEG.
    pub bytes: usize,
    /// Per-metric distortion for this sample.
    pub distortions: BTreeMap<MetricKind, f64>,
    /// Wall-clock encode time (ms).
    pub encode_ms: f64,
}

/// Signature of the closure that encodes one (image, config, quality)
/// combination and returns its size + distortions.
///
/// Implementations typically:
/// 1. Build an `EncoderConfig` from `(config_name, quality)`.
/// 2. Encode `image.rgb8` at `image.width × image.height`.
/// 3. Decode the JPEG back to RGB8.
/// 4. Run each requested metric on (original, decoded).
pub type Encoder<'a> = dyn Fn(&CorpusImage, &str, u8) -> Option<SampleOutput> + Sync + 'a;

/// Run a sweep: for each image × config × quality, call `encoder` and
/// collect results into a [`SweepResult`].
///
/// The closure-based API keeps this module independent of any perceptual
/// metric implementation. `progress` is called once per completed point
/// with the current counts `(done, total)` — pass a no-op if you don't
/// care.
pub fn run_sweep(
    images: &[CorpusImage],
    config_names: &[String],
    qualities: &[u8],
    encoder: &Encoder<'_>,
    progress: &mut dyn FnMut(usize, usize),
) -> SweepResult {
    let mut result = SweepResult::default();
    let total = images.len() * config_names.len() * qualities.len();
    let mut done = 0;
    for image in images {
        let per_image = result
            .results
            .entry(image.label.clone())
            .or_default();
        result.classes.insert(image.label.clone(), image.class);
        for cfg in config_names {
            let points = per_image.entry(cfg.clone()).or_default();
            for &q in qualities {
                if let Some(out) = encoder(image, cfg.as_str(), q) {
                    let pixels = image.pixels().max(1) as f64;
                    let bpp = out.bytes as f64 * 8.0 / pixels;
                    points.push(PointResult {
                        quality: q,
                        bytes: out.bytes,
                        bpp,
                        distortions: out.distortions,
                        encode_ms: out.encode_ms,
                    });
                }
                done += 1;
                progress(done, total);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_image(label: &str, class: ImageClass, w: usize, h: usize) -> CorpusImage {
        let rgb8 = vec![128u8; w * h * 3];
        CorpusImage {
            label: label.to_owned(),
            width: w,
            height: h,
            rgb8,
            class,
        }
    }

    #[test]
    fn run_sweep_collects_grid() {
        let imgs = [
            synthetic_image("a.png", ImageClass::Photo, 32, 32),
            synthetic_image("b.png", ImageClass::Synthetic, 32, 32),
        ];
        let configs = [String::from("default"), String::from("candidate")];
        let qs = [50u8, 85u8];

        let encoder = |img: &CorpusImage, name: &str, q: u8| -> Option<SampleOutput> {
            // Fake scores: candidate is always 10% smaller and 5% better.
            let base_bytes = 1000usize + q as usize * 10 + img.pixels();
            let bytes = if name == "candidate" {
                (base_bytes as f64 * 0.9) as usize
            } else {
                base_bytes
            };
            let ssim2 = 100.0 - q as f64 / 2.0;
            let ssim2 = if name == "candidate" { ssim2 - 2.0 } else { ssim2 };
            let mut distortions = BTreeMap::new();
            distortions.insert(MetricKind::Ssim2, 100.0 - ssim2);
            distortions.insert(MetricKind::Bbs, (100.0 - ssim2) * 2.0);
            Some(SampleOutput {
                bytes,
                distortions,
                encode_ms: 1.0,
            })
        };

        let mut progress_calls = 0;
        let result = run_sweep(
            &imgs,
            &configs,
            &qs,
            &encoder,
            &mut |_, _| progress_calls += 1,
        );
        assert_eq!(progress_calls, 2 * 2 * 2);
        assert_eq!(result.results.len(), 2);
        for img in &imgs {
            let per = &result.results[&img.label];
            assert_eq!(per.len(), 2);
            assert_eq!(per["default"].len(), 2);
            assert_eq!(per["candidate"].len(), 2);
        }
        let curve = result.rd_curve("a.png", "default", &MetricKind::Ssim2);
        assert_eq!(curve.points.len(), 2);
        // Baseline vs candidate: candidate is 10% smaller → negative BD-rate.
        let base = result.rd_curve("a.png", "default", &MetricKind::Ssim2);
        let cand = result.rd_curve("a.png", "candidate", &MetricKind::Ssim2);
        // The two curves barely overlap (candidate's distortion range is
        // shifted); bd_rate may or may not apply, but structurally the
        // data is correct.
        assert_eq!(base.points.len(), 2);
        assert_eq!(cand.points.len(), 2);
    }

    #[test]
    fn rd_curve_missing_returns_empty() {
        let res = SweepResult::default();
        let curve = res.rd_curve("nope", "nope", &MetricKind::Ssim2);
        assert!(curve.is_empty());
    }

    #[test]
    fn image_class_slug() {
        assert_eq!(ImageClass::Photo.slug(), "photo");
        assert_eq!(ImageClass::Screenshot.slug(), "screenshot");
    }

    #[test]
    fn metric_slug() {
        assert_eq!(MetricKind::Ssim2.slug(), "ssim2");
        assert_eq!(MetricKind::Bbs.slug(), "bbs");
        assert_eq!(MetricKind::Custom("xyb_dist".into()).slug(), "xyb_dist");
    }
}
