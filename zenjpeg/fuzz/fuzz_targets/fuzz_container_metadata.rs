//! Fuzz target for `zenjpeg::container::metadata` (feature
//! `extended-metadata`).
//!
//! The checks live in `fuzz/metadata_checks.rs`, which
//! `tests/fuzz_regression.rs` includes too, so the committed seeds replay
//! exactly these assertions on stable.

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../metadata_checks.rs"]
mod metadata_checks;

fuzz_target!(|data: &[u8]| {
    let data = if data.len() > 4 << 20 {
        &data[..4 << 20]
    } else {
        data
    };
    metadata_checks::check_all(data);
});
