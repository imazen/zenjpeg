//! Limits fuzzer: verify resource limits are enforced under adversarial input.
//!
//! Decodes with strict resource limits — should never exceed them, OOM, or panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::decoder::Decoder;

fuzz_target!(|data: &[u8]| {
    let _ = Decoder::new()
        .max_pixels(4_000_000)
        .max_memory(64 * 1024 * 1024) // 64 MB
        .decode(data, enough::Unstoppable);
});
