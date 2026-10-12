#![no_main]
use libfuzzer_sys::fuzz_target;
#[path = "../rewrite_checks.rs"]
mod rewrite_checks;
fuzz_target!(|data: &[u8]| rewrite_checks::check(data));
