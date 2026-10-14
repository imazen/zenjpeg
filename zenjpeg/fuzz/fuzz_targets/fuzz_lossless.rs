#![no_main]
use libfuzzer_sys::fuzz_target;
#[path = "../lossless_checks.rs"]
mod lossless_checks;
fuzz_target!(|data: &[u8]| lossless_checks::check(data));
