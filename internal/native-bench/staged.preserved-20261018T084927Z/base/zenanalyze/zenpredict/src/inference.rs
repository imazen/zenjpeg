//! Forward-pass kernel.
//!
//! For each layer:
//! 1. Initialize the accumulator with the layer's biases (broadcast).
//! 2. For each input element `x[i]`, add `x[i] * W[i, :]` to the
//!    accumulator. Embarrassingly parallel across the output dim.
//! 3. Apply activation in-place.
//!
//! The SAXPY-style inner loop is what `magetypes::f32x8` wants when
//! SIMD dispatch lands. The fixed-size `[f32; 8]` chunk loads let
//! LLVM auto-vectorize this to one `f32x8` FMA per iteration on
//! AVX2/AVX-512 and 2× `f32x4` on NEON/WASM today.

use crate::error::PredictError;
use crate::model::{Activation, LEAKY_RELU_ALPHA, LayerView, Model, WeightStorage};

/// Run the full forward pass: scale inputs, then layer-by-layer.
///
/// `scratch_a` and `scratch_b` are reused across layers. They must
/// each be at least [`Model::scratch_len`](crate::Model::scratch_len)
/// long. `output` must be exactly `n_outputs` long.
pub fn forward(
    model: &Model,
    features: &[f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    output: &mut [f32],
) -> Result<(), PredictError> {
    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    if features.len() != n_inputs {
        return Err(PredictError::FeatureLenMismatch {
            expected: n_inputs,
            got: features.len(),
        });
    }
    if output.len() != n_outputs {
        return Err(PredictError::FeatureLenMismatch {
            expected: n_outputs,
            got: output.len(),
        });
    }
    let need = model.scratch_len();
    if scratch_a.len() < need || scratch_b.len() < need {
        return Err(PredictError::FeatureLenMismatch {
            expected: need,
            got: scratch_a.len().min(scratch_b.len()),
        });
    }

    // Scale inputs: x' = (x - mean) / scale.
    //
    // Zero-variance columns: sklearn's `_handle_zeros_in_scale`
    // replaces `scale=0` with `1.0` so the column passes through as
    // `(x - mean)`. Mirror that defensively.
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    for i in 0..n_inputs {
        let s = scale[i];
        let safe_s = if s == 0.0 { 1.0 } else { s };
        scratch_a[i] = (features[i] - mean[i]) / safe_s;
    }

    let mut input_buf: &mut [f32] = scratch_a;
    let mut output_buf: &mut [f32] = scratch_b;

    let n_layers = model.n_layers();
    let last_idx = n_layers - 1;

    for (idx, layer) in model.layers().enumerate() {
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;
        let dst: &mut [f32] = if idx == last_idx {
            &mut output[..out_dim]
        } else {
            &mut output_buf[..out_dim]
        };
        let src = &input_buf[..in_dim];
        layer_forward(&layer, src, dst)?;
        if idx != last_idx {
            core::mem::swap(&mut input_buf, &mut output_buf);
        }
    }
    Ok(())
}

fn layer_forward(layer: &LayerView<'_>, src: &[f32], dst: &mut [f32]) -> Result<(), PredictError> {
    let out_dim = layer.out_dim;
    let in_dim = layer.in_dim;
    debug_assert_eq!(src.len(), in_dim);
    debug_assert_eq!(dst.len(), out_dim);
    debug_assert_eq!(layer.biases.len(), out_dim);

    match &layer.weights {
        WeightStorage::F32(w) => {
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f32(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::F16(w) => {
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f16(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::I8 { weights, scales } => {
            // Per-output `scales[o]` only applies to the SAXPY
            // accumulator, not the bias. Zero dst, accumulate raw,
            // then `dst[o] = bias[o] + scales[o] * dst[o]`.
            for v in dst.iter_mut() {
                *v = 0.0;
            }
            saxpy_matmul_i8(src, weights, dst, in_dim, out_dim);
            debug_assert_eq!(scales.len(), out_dim);
            for o in 0..out_dim {
                dst[o] = layer.biases[o] + scales[o] * dst[o];
            }
        }
    }

    apply_activation(dst, layer.activation);
    Ok(())
}

fn saxpy_matmul_f32(src: &[f32], w: &[f32], dst: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let weight_chunk: &[f32; 8] = row[base..base + 8].try_into().unwrap();
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                acc_chunk[k] = fma(s, weight_chunk[k], acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                dst[tail_start + k] = fma(s, row[tail_start + k], dst[tail_start + k]);
            }
        }
    }
}

fn saxpy_matmul_f16(src: &[f32], w: &[u16], dst: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                let wf = f16_bits_to_f32(row[base + k]);
                acc_chunk[k] = fma(s, wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = f16_bits_to_f32(row[tail_start + k]);
                dst[tail_start + k] = fma(s, wf, dst[tail_start + k]);
            }
        }
    }
}

fn saxpy_matmul_i8(src: &[f32], w: &[i8], dst: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let weight_chunk: &[i8; 8] = row[base..base + 8].try_into().unwrap();
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                let wf = weight_chunk[k] as f32;
                acc_chunk[k] = fma(s, wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = row[tail_start + k] as f32;
                dst[tail_start + k] = fma(s, wf, dst[tail_start + k]);
            }
        }
    }
}

/// IEEE-754 binary16 → binary32 converter. Pure integer bit math —
/// works in `no_std` and at compile time. Same answer as
/// `_mm256_cvtph_ps`, one element at a time.
#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h & 0x7c00) >> 10;
    let mant = h & 0x03ff;
    let bits = if exp == 0 {
        if mant == 0 {
            0
        } else {
            // Subnormal — promote to f32 normal.
            let k = 31 - mant.leading_zeros();
            let shift = 10 - k;
            let normalized_mant = (mant << shift) & 0x3ff;
            let f32_exp = k + 103;
            (f32_exp << 23) | (normalized_mant << 13)
        }
    } else if exp == 0x1f {
        0x7f80_0000 | (mant << 13)
    } else {
        ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(sign | bits)
}

/// `a * b + c`. Uses `f32::mul_add` (single-rounding fma) when the
/// `std` feature is on; falls back to `a * b + c` (two roundings)
/// for `no_std + alloc` builds. The numerical difference is in the
/// last bit and well below MLP training noise; the perf difference
/// is one fused instruction vs two.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        a * b + c
    }
}

fn apply_activation(buf: &mut [f32], act: Activation) {
    match act {
        Activation::Identity => {}
        Activation::Relu => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        Activation::LeakyRelu => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v *= LEAKY_RELU_ALPHA;
                }
            }
        }
    }
}

#[cfg(test)]
mod f16_tests {
    use super::f16_bits_to_f32;

    fn check(bits: u16, expected: f32, what: &str) {
        let got = f16_bits_to_f32(bits);
        if expected.is_nan() {
            assert!(
                got.is_nan(),
                "{what}: bits=0x{bits:04x} expected NaN, got {got}"
            );
        } else {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{what}: bits=0x{bits:04x} got {got} ({:08x}), expected {expected} ({:08x})",
                got.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn zeros_and_signs() {
        check(0x0000, 0.0, "+0");
        check(0x8000, -0.0, "-0");
    }

    #[test]
    fn ones() {
        check(0x3c00, 1.0, "+1.0");
        check(0xbc00, -1.0, "-1.0");
        check(0x4000, 2.0, "+2.0");
        check(0xc000, -2.0, "-2.0");
    }

    #[test]
    fn fractions() {
        // 0.5 = exp=14 (bias-15 → -1), mant=0
        check(0x3800, 0.5, "0.5");
        // 1/3 representable: 0x3555 = 0.333251953125
        check(0x3555, 0.333_251_95, "approx 1/3");
    }

    #[test]
    fn subnormals() {
        // Smallest positive subnormal: 0x0001 = 2^-24 ≈ 5.96e-8
        check(0x0001, 5.960_464_5e-8, "smallest +subnormal");
        // Largest subnormal: 0x03ff
        check(0x03ff, 6.097_555e-5, "largest +subnormal");
        check(0x8001, -5.960_464_5e-8, "smallest -subnormal");
    }

    #[test]
    fn extremes() {
        // Smallest positive normal: 0x0400 = 2^-14 ≈ 6.10e-5
        check(0x0400, 6.103_515_6e-5, "smallest +normal");
        // Largest normal: 0x7bff = 65504.0
        check(0x7bff, 65504.0, "largest +normal");
    }

    #[test]
    fn inf_nan() {
        check(0x7c00, f32::INFINITY, "+inf");
        check(0xfc00, f32::NEG_INFINITY, "-inf");
        let nan = f16_bits_to_f32(0x7e00);
        assert!(nan.is_nan(), "0x7e00 should be NaN");
    }
}
