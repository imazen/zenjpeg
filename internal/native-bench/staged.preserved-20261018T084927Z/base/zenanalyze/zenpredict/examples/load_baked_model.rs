//! Loads a ZNPR v3 `.bin` from argv[1], runs inference against a
//! deterministic input vector, and writes the output values to
//! stdout (one f32 per line, full precision). Used by the companion
//! `tools/bake_roundtrip_check.py` to compare Rust forward pass
//! output against the numpy reference.
//!
//! Usage:
//!   cargo run --release -p zenpredict --example load_baked_model -- <model.bin>
//!
//! Diagnostics on stderr; output values on stdout.

use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: load_baked_model <model.bin>");
        return ExitCode::from(2);
    }
    let bytes = match fs::read(&args[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("read {}: {e}", args[1]);
            return ExitCode::from(1);
        }
    };

    // Re-align into a u64-backed buffer so the loader's
    // bytemuck::try_cast_slice can zero-copy borrow the f32 / u16
    // sections. Production consumers wrap `include_bytes!` in
    // `#[repr(C, align(16))]` instead.
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
        "loaded: n_inputs={} n_outputs={} n_layers={} schema_hash=0x{:016x} metadata_entries={}",
        model.n_inputs(),
        model.n_outputs(),
        model.n_layers(),
        model.schema_hash(),
        model.metadata().len(),
    );

    // Deterministic input: feature_i = sin(i * 0.1) so the round-trip
    // check has content-rich values to compare against.
    let n_in = model.n_inputs();
    let features: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.1).sin()).collect();

    let mut predictor = zenpredict::Predictor::new(&model);
    let out = match predictor.predict(&features) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("predict: {e}");
            return ExitCode::from(1);
        }
    };

    for v in out {
        println!("{v}");
    }
    ExitCode::SUCCESS
}
