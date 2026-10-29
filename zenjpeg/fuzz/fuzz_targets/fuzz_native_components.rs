#![no_main]
#[path = "../component_checks.rs"]
mod component_checks;
libfuzzer_sys::fuzz_target!(|data: &[u8]| component_checks::check(data));
