//! Fuzz target: `Model::from_bytes` → `Predictor::new` → `predict`.
//! Covers the parser, load-time mutations (decompression,
//! feature_order / output_order permutation), and matmul kernels.
//! Must never panic.
//!
//! Two-stage: parse the byte stream as a model (per `parser_from_bytes`),
//! then if it parsed cleanly, construct a Predictor and call `predict`
//! with a zeroed feature vector sized to `n_inputs`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenpredict::{Model, Predictor};

fuzz_target!(|bytes: &[u8]| {
    let Ok(model) = Model::from_bytes(bytes) else {
        return;
    };
    let n_in = model.n_inputs();
    // Resource limit: matches the parser's MAX_DIM. Refuse to
    // allocate a gigabyte feature vector for fuzz inputs that the
    // parser somehow let through.
    if n_in > zenpredict::limits::MAX_DIM {
        return;
    }
    let features = vec![0.0f32; n_in];
    let mut predictor = Predictor::new(&model);
    let _ = predictor.predict(&features);
});
