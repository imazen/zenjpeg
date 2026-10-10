//! The frame's colour model and sampling shape, decided once when SOF is parsed.
//!
//! Every decode path reads these two values. None of them asks "is this RGB?"
//! or derives a sampling ratio on its own: the RGB-as-RGB work showed that a
//! dozen independent answers drift apart (different parse points, floor-divided
//! ratios, component 1 checked but not component 2), and each drift was a
//! wrong-pixel bug or a panic. See `docs/DECODER_PATHS.md`, "Colour model and
//! sampling shape".

use super::IdctMethod;
use crate::color::planes::{Layout8, PlaneMatrix, RowKernel};
use crate::types::{ColorSpace, Component};

/// How the stored components map to colour: libjpeg's `jpeg_color_space`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColorModel {
    /// One component.
    Gray,
    /// Three components, BT.601 full-range YCbCr.
    YCbCr,
    /// Three components stored as R, G, B (no colour transform).
    Rgb,
    /// Three components stored as jpegli's scaled XYB (RGB-signalled
    /// components plus the XYB ICC profile).
    Xyb,
    /// Four components, Adobe-inverted CMYK.
    Cmyk,
    /// Four components, YCbCr + K.
    Ycck,
    /// Any other component count.
    Unknown,
}

/// The colour-relevant markers seen before SOF.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ColorMarkers {
    /// A JFIF APP0 (`"JFIF"`, payload of at least 14 bytes, as jpegli checks).
    pub saw_jfif: bool,
    /// The transform byte of the last Adobe APP14 (TN 5116), raw.
    pub adobe_transform: Option<u8>,
}

impl ColorModel {
    /// The colour model of a frame, from its components and the markers seen
    /// before its SOF.
    ///
    /// Follows jpegli's `ProcessSOF` (`internal/jpegli-cpp/lib/jpegli/
    /// decode_marker.cc`), which matches libjpeg-turbo's
    /// `default_decompress_parms` except that libjpeg-turbo decides at the
    /// first SOS and so also honours markers placed between SOF and SOS:
    ///
    /// - 3 components: a JFIF APP0 means YCbCr; else an Adobe APP14 means RGB
    ///   for transform 0 and YCbCr for any other value; else component IDs
    ///   `'R','G','B'` mean RGB and anything else YCbCr. An RGB frame whose
    ///   ICC profile is jpegli's XYB profile is XYB.
    /// - 4 components: transform 2 means YCCK, anything else Adobe-inverted
    ///   CMYK (zenjpeg's long-standing rule; jpegli and libjpeg-turbo read any
    ///   non-zero transform as YCCK).
    pub(crate) fn decide(
        components: &[Component],
        markers: ColorMarkers,
        icc_is_xyb: bool,
    ) -> Self {
        match components {
            [_] => Self::Gray,
            [c0, c1, c2] => {
                let rgb = if markers.saw_jfif {
                    false
                } else if let Some(transform) = markers.adobe_transform {
                    transform == 0
                } else {
                    [c0.id, c1.id, c2.id] == *b"RGB"
                };
                match (rgb, icc_is_xyb) {
                    (true, true) => Self::Xyb,
                    (true, false) => Self::Rgb,
                    (false, _) => Self::YCbCr,
                }
            }
            [_, _, _, _] if markers.adobe_transform == Some(2) => Self::Ycck,
            [_, _, _, _] => Self::Cmyk,
            _ => Self::Unknown,
        }
    }

    /// The public colour space reported for this model.
    pub(crate) fn color_space(self) -> ColorSpace {
        match self {
            Self::Gray => ColorSpace::Grayscale,
            Self::YCbCr => ColorSpace::YCbCr,
            Self::Rgb => ColorSpace::Rgb,
            Self::Xyb => ColorSpace::Xyb,
            // Every 4-component frame has always reported `Cmyk`.
            Self::Cmyk | Self::Ycck => ColorSpace::Cmyk,
            Self::Unknown => ColorSpace::Unknown,
        }
    }

    /// Three planes that the integer row kernels turn into RGB: YCbCr through
    /// the matrix, RGB as an identity. Every other model needs its own
    /// conversion and stays off the integer fast paths.
    pub(crate) fn has_row_kernel(self) -> bool {
        matches!(self, Self::YCbCr | Self::Rgb)
    }

    /// The plane conversion of this model for the integer row kernels: `None`
    /// for models those kernels do not serve.
    pub(crate) fn plane_matrix(self, idct_method: IdctMethod) -> Option<PlaneMatrix> {
        match self {
            Self::Gray => Some(PlaneMatrix::Gray),
            Self::YCbCr => Some(PlaneMatrix::YCbCr {
                turbo: idct_method == IdctMethod::Libjpeg,
            }),
            Self::Rgb => Some(PlaneMatrix::Identity),
            _ => None,
        }
    }

    /// The row kernel converting this model's planes into `layout`.
    pub(crate) fn row_kernel(self, layout: Layout8, idct_method: IdctMethod) -> Option<RowKernel> {
        self.plane_matrix(idct_method)
            .map(|m| RowKernel::new(m, layout))
    }
}

/// Chroma sampling relative to component 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChromaLayout {
    /// One component.
    Gray,
    /// All three components share one sampling.
    S444,
    /// Component 0 has twice the horizontal sampling of the chroma (4:2:2).
    H2V1,
    /// Twice in both directions (4:2:0).
    H2V2,
    /// Twice vertically (4:4:0).
    H1V2,
    /// Anything else: other ratios, fractional ratios, asymmetric chroma,
    /// chroma sampled above component 0, more than three components.
    Other,
}

/// The frame's sampling shape, classified once from the SOF components.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameShape {
    pub(crate) chroma: ChromaLayout,
    pub(crate) max_h: u8,
    pub(crate) max_v: u8,
    /// The chroma components (or the single component) are sampled 1x1, so an
    /// MCU is the canonical one every MCU-walking kernel (streaming baseline,
    /// fused parallel, wave parallel) is written for.
    pub(crate) unit_chroma: bool,
}

impl FrameShape {
    pub(crate) fn classify(components: &[Component]) -> Self {
        let f = |c: &Component| (c.h_samp_factor, c.v_samp_factor);
        let max_h = components
            .iter()
            .map(|c| c.h_samp_factor)
            .max()
            .unwrap_or(1);
        let max_v = components
            .iter()
            .map(|c| c.v_samp_factor)
            .max()
            .unwrap_or(1);
        let (chroma, unit_chroma) = match components {
            [g] => (ChromaLayout::Gray, f(g) == (1, 1)),
            [y, cb, cr] if f(cb) == f(cr) => {
                let (yh, yv) = f(y);
                let (ch, cv) = f(cb);
                // Exact ratios only: component 0 must carry the maxima and be
                // an integer multiple of the chroma.
                let layout = if yh % ch != 0 || yv % cv != 0 {
                    ChromaLayout::Other
                } else {
                    match (yh / ch, yv / cv) {
                        (1, 1) => ChromaLayout::S444,
                        (2, 1) => ChromaLayout::H2V1,
                        (2, 2) => ChromaLayout::H2V2,
                        (1, 2) => ChromaLayout::H1V2,
                        _ => ChromaLayout::Other,
                    }
                };
                (layout, (ch, cv) == (1, 1))
            }
            _ => (ChromaLayout::Other, false),
        };
        Self {
            chroma,
            max_h,
            max_v,
            unit_chroma,
        }
    }

    /// `(h, v)` chroma upsampling ratios (1 where the chroma is not subsampled).
    pub(crate) fn chroma_ratios(self) -> (usize, usize) {
        match self.chroma {
            ChromaLayout::H2V1 => (2, 1),
            ChromaLayout::H2V2 => (2, 2),
            ChromaLayout::H1V2 => (1, 2),
            _ => (1, 1),
        }
    }
}

/// Which decode paths may serve a frame, from its model and shape alone. The
/// gates in `parser::scan` (streaming), `fused_parallel` (fused, wave),
/// `parser::output` (fast integer) and `Decoder::classify_scanline_route`
/// (strip reader) call these and add only their own non-shape conditions
/// (progressive, DRI, thread count, output format).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PathPlan {
    pub(crate) model: ColorModel,
    pub(crate) shape: FrameShape,
    /// The first scan carries every component. A sequential frame whose first
    /// scan does not is non-interleaved (one scan per component, T.81 A.2.2):
    /// its MCUs cannot be walked as it is read, so only coefficient-buffered
    /// paths serve it.
    pub(crate) interleaved: bool,
}

impl PathPlan {
    /// The MCU-walking kernels (streaming baseline, fused parallel, wave
    /// parallel) handle grayscale, 4:4:4, 4:2:2 and 4:2:0 with 1x1 chroma,
    /// for models with a row kernel (and gray), when the first scan is
    /// interleaved.
    pub(crate) fn mcu_kernel(self) -> Option<ChromaLayout> {
        let model_ok = match self.shape.chroma {
            ChromaLayout::Gray => self.model == ColorModel::Gray,
            _ => self.model.has_row_kernel(),
        };
        let shape_ok = self.shape.unit_chroma
            && matches!(
                self.shape.chroma,
                ChromaLayout::Gray | ChromaLayout::S444 | ChromaLayout::H2V1 | ChromaLayout::H2V2
            );
        (self.interleaved && model_ok && shape_ok).then_some(self.shape.chroma)
    }

    /// The strip reader (`ScanlineReader` streaming and coefficient modes)
    /// handles grayscale 1x1 and every three-plane shape with sampling factors
    /// up to 2.
    pub(crate) fn strip(self) -> bool {
        match self.shape.chroma {
            ChromaLayout::Gray => self.model == ColorModel::Gray && self.shape.unit_chroma,
            ChromaLayout::Other => false,
            _ => self.model.has_row_kernel() && self.shape.max_h <= 2 && self.shape.max_v <= 2,
        }
    }

    /// The fast integer coefficient paths: `S444` (any shared sampling) and
    /// the three subsampled layouts, for models with a row kernel.
    pub(crate) fn fast_integer(self) -> Option<ChromaLayout> {
        (self.model.has_row_kernel()
            && matches!(
                self.shape.chroma,
                ChromaLayout::S444 | ChromaLayout::H2V1 | ChromaLayout::H2V2 | ChromaLayout::H1V2
            ))
        .then_some(self.shape.chroma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comps(ids: &[u8], samp: &[(u8, u8)]) -> Vec<Component> {
        ids.iter()
            .zip(samp)
            .map(|(&id, &(h, v))| Component {
                id,
                h_samp_factor: h,
                v_samp_factor: v,
                ..Component::default()
            })
            .collect()
    }

    #[test]
    fn decision_follows_jpegli_process_sof() {
        let ycc = comps(&[1, 2, 3], &[(1, 1); 3]);
        let rgb = comps(b"RGB", &[(1, 1); 3]);
        let m = |saw_jfif, adobe_transform| ColorMarkers {
            saw_jfif,
            adobe_transform,
        };
        assert_eq!(
            ColorModel::decide(&ycc, m(false, None), false),
            ColorModel::YCbCr
        );
        assert_eq!(
            ColorModel::decide(&rgb, m(false, None), false),
            ColorModel::Rgb
        );
        assert_eq!(
            ColorModel::decide(&ycc, m(false, Some(0)), false),
            ColorModel::Rgb
        );
        assert_eq!(
            ColorModel::decide(&rgb, m(false, Some(1)), false),
            ColorModel::YCbCr
        );
        for t in [3u8, 7, 255] {
            assert_eq!(
                ColorModel::decide(&ycc, m(false, Some(t)), false),
                ColorModel::YCbCr
            );
        }
        assert_eq!(
            ColorModel::decide(&ycc, m(true, Some(0)), false),
            ColorModel::YCbCr
        );
        assert_eq!(
            ColorModel::decide(&rgb, m(true, None), false),
            ColorModel::YCbCr
        );
        assert_eq!(
            ColorModel::decide(&rgb, m(false, None), true),
            ColorModel::Xyb
        );
        assert_eq!(
            ColorModel::decide(&ycc, m(false, None), true),
            ColorModel::YCbCr
        );
        let gray = comps(&[1], &[(1, 1)]);
        assert_eq!(
            ColorModel::decide(&gray, m(true, Some(0)), true),
            ColorModel::Gray
        );
        let four = comps(b"CMYK", &[(1, 1); 4]);
        assert_eq!(
            ColorModel::decide(&four, m(false, Some(2)), false),
            ColorModel::Ycck
        );
        assert_eq!(
            ColorModel::decide(&four, m(false, Some(0)), false),
            ColorModel::Cmyk
        );
        assert_eq!(
            ColorModel::decide(&four, m(false, None), false),
            ColorModel::Cmyk
        );
    }

    #[test]
    fn shape_classification_is_exact() {
        let s = |samp: &[(u8, u8)]| FrameShape::classify(&comps(&[1, 2, 3, 4][..samp.len()], samp));
        assert_eq!(s(&[(1, 1), (1, 1), (1, 1)]).chroma, ChromaLayout::S444);
        assert_eq!(s(&[(2, 2), (2, 2), (2, 2)]).chroma, ChromaLayout::S444);
        assert!(!s(&[(2, 2), (2, 2), (2, 2)]).unit_chroma);
        assert_eq!(s(&[(2, 1), (1, 1), (1, 1)]).chroma, ChromaLayout::H2V1);
        assert_eq!(s(&[(2, 2), (1, 1), (1, 1)]).chroma, ChromaLayout::H2V2);
        assert_eq!(s(&[(1, 2), (1, 1), (1, 1)]).chroma, ChromaLayout::H1V2);
        assert_eq!(s(&[(2, 2), (2, 1), (2, 1)]).chroma, ChromaLayout::H1V2);
        // Fractional, unsupported ratio, asymmetric, chroma above luma.
        assert_eq!(s(&[(3, 1), (2, 1), (2, 1)]).chroma, ChromaLayout::Other);
        assert_eq!(s(&[(3, 2), (2, 1), (2, 1)]).chroma, ChromaLayout::Other);
        assert_eq!(s(&[(4, 1), (1, 1), (1, 1)]).chroma, ChromaLayout::Other);
        assert_eq!(s(&[(2, 2), (2, 2), (1, 1)]).chroma, ChromaLayout::Other);
        assert_eq!(s(&[(1, 1), (2, 2), (2, 2)]).chroma, ChromaLayout::Other);
        assert_eq!(s(&[(1, 1); 4]).chroma, ChromaLayout::Other);
        assert!(s(&[(1, 1)]).unit_chroma);
        assert!(!s(&[(2, 2)]).unit_chroma);
    }

    #[test]
    fn path_plan_gates() {
        let shape =
            |samp: &[(u8, u8)]| FrameShape::classify(&comps(&[1, 2, 3][..samp.len()], samp));
        let plan = |model, samp: &[(u8, u8)]| PathPlan {
            interleaved: true,
            model,
            shape: shape(samp),
        };
        let h2v2 = [(2, 2), (1, 1), (1, 1)];
        assert_eq!(
            plan(ColorModel::Rgb, &h2v2).mcu_kernel(),
            Some(ChromaLayout::H2V2)
        );
        assert_eq!(plan(ColorModel::Xyb, &h2v2).mcu_kernel(), None);
        assert_eq!(plan(ColorModel::Xyb, &h2v2).fast_integer(), None);
        assert!(!plan(ColorModel::Xyb, &h2v2).strip());
        // 4:4:0 is a strip/fast-integer shape but not an MCU-kernel shape.
        let h1v2 = [(1, 2), (1, 1), (1, 1)];
        assert_eq!(plan(ColorModel::YCbCr, &h1v2).mcu_kernel(), None);
        assert!(plan(ColorModel::YCbCr, &h1v2).strip());
        assert_eq!(
            plan(ColorModel::YCbCr, &h1v2).fast_integer(),
            Some(ChromaLayout::H1V2)
        );
        // Oversampled grayscale is a non-interleaved scan: no MCU kernel, no strip.
        assert_eq!(plan(ColorModel::Gray, &[(2, 2)]).mcu_kernel(), None);
        assert!(!plan(ColorModel::Gray, &[(2, 2)]).strip());
        assert_eq!(
            plan(ColorModel::Gray, &[(1, 1)]).mcu_kernel(),
            Some(ChromaLayout::Gray)
        );
        // Fractional ratios reach nothing but the per-component path.
        let frac = PathPlan {
            interleaved: true,
            model: ColorModel::YCbCr,
            shape: shape(&[(3, 1), (2, 1), (2, 1)]),
        };
        assert_eq!(frac.mcu_kernel(), None);
        assert!(!frac.strip());
        assert_eq!(frac.fast_integer(), None);
    }
}
