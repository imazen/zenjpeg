//! Fuzz target for JPEG header parsing.
//!
//! Tests `Decoder::read_info()` which only parses headers without decoding
//! the full image. This is a lighter-weight target that focuses on marker
//! parsing and validation.

#![no_main]

use jpegli::decode::Decoder;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let decoder = Decoder::new();

    // Test header-only parsing
    if let Ok(info) = decoder.read_info(data) {
        // If we successfully read info, verify invariants
        assert!(info.dimensions.width > 0, "width should be positive");
        assert!(info.dimensions.height > 0, "height should be positive");
        assert!(info.num_components > 0, "num_components should be positive");
        assert!(info.num_components <= 4, "num_components should be <= 4");

        // Precision should be 8 or 12 for valid JPEG
        assert!(
            info.precision == 8 || info.precision == 12,
            "precision should be 8 or 12"
        );
    }
    // Errors are expected for malformed data - no assertion needed
});
