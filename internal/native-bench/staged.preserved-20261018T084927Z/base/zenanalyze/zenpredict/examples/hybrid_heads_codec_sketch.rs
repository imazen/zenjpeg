//! Sketch of how a codec crate consumes a hybrid-heads picker model.
//!
//! The model emits a flat output vector. Codec compile-time
//! constants slice it into `bytes_log[0..n_cells]`,
//! `chroma_scale[n_cells..2*n_cells]`,
//! `lambda[2*n_cells..3*n_cells]`. Categorical pick is masked-argmin
//! over the bytes-log range; scalar predictions are read by index
//! and clamped to caller constraints.
//!
//! Run with a hybrid-heads `.bin` (e.g., one baked from
//! `zq_hybrid_heads.py`):
//!
//!   cargo run --release --example hybrid_heads_codec_sketch -- \
//!       benchmarks/zenjpeg_picker_hybrid_v1.bin
//!
//! This file is *just a sketch* — a real codec would have its own
//! ConfigSpec table, constraint type, and feature-extraction hook.

use std::env;
use std::fs;
use std::process::ExitCode;

/// Categorical cell layout for the zenjpeg hybrid model. Order MUST
/// match `hybrid_heads_manifest.cells[].id` from the bake JSON.
#[derive(Clone, Copy, Debug)]
struct CellSpec {
    label: &'static str,
    color: ColorMode,
    sub: Subsampling,
    trellis_on: bool,
    sa_piecewise: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ColorMode {
    YCbCr,
    Xyb,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Subsampling {
    Full,    // 4:4:4 / XYB Full
    Quarter, // 4:2:0 / XYB BQuarter
}

/// 12 cells in the order the hybrid-heads bake emits them.
/// (Sorted by (color, sub, trellis_on, sa).)
const CELLS: &[CellSpec] = &[
    CellSpec {
        label: "xyb_420_noT",
        color: ColorMode::Xyb,
        sub: Subsampling::Quarter,
        trellis_on: false,
        sa_piecewise: false,
    },
    CellSpec {
        label: "xyb_420_trellis",
        color: ColorMode::Xyb,
        sub: Subsampling::Quarter,
        trellis_on: true,
        sa_piecewise: false,
    },
    CellSpec {
        label: "xyb_444_noT",
        color: ColorMode::Xyb,
        sub: Subsampling::Full,
        trellis_on: false,
        sa_piecewise: false,
    },
    CellSpec {
        label: "xyb_444_trellis",
        color: ColorMode::Xyb,
        sub: Subsampling::Full,
        trellis_on: true,
        sa_piecewise: false,
    },
    CellSpec {
        label: "ycbcr_420_noT",
        color: ColorMode::YCbCr,
        sub: Subsampling::Quarter,
        trellis_on: false,
        sa_piecewise: false,
    },
    CellSpec {
        label: "ycbcr_420_noT_sa",
        color: ColorMode::YCbCr,
        sub: Subsampling::Quarter,
        trellis_on: false,
        sa_piecewise: true,
    },
    CellSpec {
        label: "ycbcr_420_trellis",
        color: ColorMode::YCbCr,
        sub: Subsampling::Quarter,
        trellis_on: true,
        sa_piecewise: false,
    },
    CellSpec {
        label: "ycbcr_420_trellis_sa",
        color: ColorMode::YCbCr,
        sub: Subsampling::Quarter,
        trellis_on: true,
        sa_piecewise: true,
    },
    CellSpec {
        label: "ycbcr_444_noT",
        color: ColorMode::YCbCr,
        sub: Subsampling::Full,
        trellis_on: false,
        sa_piecewise: false,
    },
    CellSpec {
        label: "ycbcr_444_noT_sa",
        color: ColorMode::YCbCr,
        sub: Subsampling::Full,
        trellis_on: false,
        sa_piecewise: true,
    },
    CellSpec {
        label: "ycbcr_444_trellis",
        color: ColorMode::YCbCr,
        sub: Subsampling::Full,
        trellis_on: true,
        sa_piecewise: false,
    },
    CellSpec {
        label: "ycbcr_444_trellis_sa",
        color: ColorMode::YCbCr,
        sub: Subsampling::Full,
        trellis_on: true,
        sa_piecewise: true,
    },
];
const N_CELLS: usize = 12;

/// Caller-supplied constraints on what the picker may produce.
#[derive(Clone, Copy, Debug, Default)]
struct ZqConstraints {
    require_color: Option<ColorMode>,
    forbid_xyb: bool,
    forbid_trellis: bool,
    forbid_progressive: bool, // not modeled in this picker but illustrative
    chroma_scale_range: Option<(f32, f32)>, // clamp
    lambda_range: Option<(f32, f32)>,
}

impl ZqConstraints {
    fn matches_cell(&self, c: &CellSpec) -> bool {
        if let Some(rc) = self.require_color
            && c.color != rc
        {
            return false;
        }
        if self.forbid_xyb && c.color == ColorMode::Xyb {
            return false;
        }
        if self.forbid_trellis && c.trellis_on {
            return false;
        }
        let _ = self.forbid_progressive;
        true
    }

    fn allowed_mask(&self) -> [bool; N_CELLS] {
        std::array::from_fn(|i| self.matches_cell(&CELLS[i]))
    }
}

/// Result of asking the picker for a config. Fields are read by
/// the `Debug` print at the bottom of `main`; allow `dead_code`
/// because this is a sketch and a real codec would consume them.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
struct EncoderConfig {
    color: ColorMode,
    sub: Subsampling,
    trellis_on: bool,
    sa_piecewise: bool,
    chroma_scale: f32,
    lambda: Option<f32>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: hybrid_heads_codec_sketch <hybrid.bin>");
        return ExitCode::from(2);
    }
    let bytes = match fs::read(&args[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("read {}: {e}", args[1]);
            return ExitCode::from(1);
        }
    };

    // Re-align via u64-backed buffer so the f32 sections are 4-aligned.
    let n_u64 = bytes.len().div_ceil(8);
    let mut storage: Vec<u64> = vec![0; n_u64];
    let view: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
    view[..bytes.len()].copy_from_slice(&bytes);
    let aligned: &[u8] = &bytemuck::cast_slice::<u64, u8>(&storage)[..bytes.len()];

    let model = match zenpredict::Model::from_bytes(aligned) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "loaded: n_inputs={} n_outputs={} schema_hash=0x{:016x}",
        model.n_inputs(),
        model.n_outputs(),
        model.schema_hash()
    );

    if model.n_outputs() != 3 * N_CELLS {
        eprintln!(
            "this sketch expects a hybrid-heads model with {} outputs (3 × {} cells), got {}",
            3 * N_CELLS,
            N_CELLS,
            model.n_outputs()
        );
        return ExitCode::from(1);
    }

    let n_in = model.n_inputs();
    let features: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.1).sin()).collect();

    let mut predictor = zenpredict::Predictor::new(&model);

    // Two demo runs:
    //   1. Unconstrained — pick the unconditional optimum.
    //   2. forbid_trellis + forbid_xyb — falls back to ycbcr no-trellis cells.
    for (label, constraints) in &[
        ("unconstrained", ZqConstraints::default()),
        (
            "forbid_xyb + forbid_trellis",
            ZqConstraints {
                forbid_xyb: true,
                forbid_trellis: true,
                ..Default::default()
            },
        ),
        (
            "chroma_scale clamp 0.85..1.15",
            ZqConstraints {
                chroma_scale_range: Some((0.85, 1.15)),
                ..Default::default()
            },
        ),
    ] {
        let mask_arr = constraints.allowed_mask();
        let mask = zenpredict::AllowedMask::new(&mask_arr);

        // Argmin over the categorical bytes head (output indices 0..N_CELLS).
        let cell_idx = match predictor
            .argmin_masked_in_range(
                &features,
                (0, N_CELLS),
                &mask,
                zenpredict::ScoreTransform::Exp,
                None,
            )
            .unwrap()
        {
            Some(i) => i,
            None => {
                println!("[{label}] no allowed cell");
                continue;
            }
        };
        let cell = CELLS[cell_idx];

        // Scalar predictions live at offsets [N_CELLS..2*N_CELLS] (chroma_scale)
        // and [2*N_CELLS..3*N_CELLS] (lambda). Re-fetch via predict() since
        // argmin_masked_in_range overwrites the scratch.
        let out = predictor.predict(&features).unwrap();
        let chroma_pred = out[N_CELLS + cell_idx];
        let lambda_pred = out[2 * N_CELLS + cell_idx];

        // Clamp scalar predictions to caller's range.
        let chroma_scale = match constraints.chroma_scale_range {
            Some((lo, hi)) => chroma_pred.clamp(lo, hi),
            None => chroma_pred,
        };
        let lambda = match constraints.lambda_range {
            Some((lo, hi)) => lambda_pred.clamp(lo, hi),
            None => lambda_pred,
        };

        let cfg = EncoderConfig {
            color: cell.color,
            sub: cell.sub,
            trellis_on: cell.trellis_on,
            sa_piecewise: cell.sa_piecewise,
            chroma_scale,
            lambda: cell.trellis_on.then_some(lambda),
        };

        println!("[{label}] cell {cell_idx:2} ({:24}): {cfg:?}", cell.label);
    }

    ExitCode::SUCCESS
}
