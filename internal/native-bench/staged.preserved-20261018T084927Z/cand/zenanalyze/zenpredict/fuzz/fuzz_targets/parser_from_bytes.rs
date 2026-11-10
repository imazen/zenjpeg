//! Fuzz target: `Model::from_bytes` must never panic on arbitrary input.
//!
//! The parser must reject every malformed input with a typed
//! `PredictError`. Any panic — index OOB, arithmetic overflow,
//! `unwrap` on a `None`, alignment slip — is a fuzz hit.
//!
//! Resource limits (`MAX_BAKE_BYTES`, `MAX_DIM`, `MAX_LAYERS`) bound
//! the loop and reject pathologically large inputs O(1) without
//! allocating; this fuzz target verifies those guards stay tight.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenpredict::Model;

fuzz_target!(|bytes: &[u8]| {
    let _ = Model::from_bytes(bytes);
});
