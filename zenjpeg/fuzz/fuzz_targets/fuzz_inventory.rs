//! Fuzz target for the structural inventory (`DecodeJob::inventory`).
//!
//! Runs the walker under the three job configurations that change its
//! dispositions (default, EXIF auto-orient, gain-map decode) and asserts:
//!
//! - No panic, no error: inputs this size never reach the part cap.
//! - The inventory covers exactly the input and passes `validate()`:
//!   top-level parts tile the input, children stay inside their parents,
//!   siblings never overlap, declared bodies are tiled.
//! - The configuration changes dispositions only, never the part layout.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zencodec::decode::{DecodeJob, DecoderConfig};
use zenjpeg::JpegDecoderConfig;

fuzz_target!(|data: &[u8]| {
    let data = if data.len() > 4 * 1024 * 1024 {
        &data[..4 * 1024 * 1024]
    } else {
        data
    };
    let jobs = [
        JpegDecoderConfig::new().job(),
        JpegDecoderConfig::new()
            .job()
            .with_orientation(zencodec::OrientationHint::Correct),
        JpegDecoderConfig::new()
            .job()
            .with_gain_map_render(zencodec::GainMapRender::Components),
    ];
    let mut layout: Option<Vec<core::ops::Range<u64>>> = None;
    for job in jobs {
        let inv = job
            .inventory(data)
            .expect("inventory must not fail below the part cap")
            .expect("zenjpeg declares the inventory capability");
        assert_eq!(inv.input_len(), data.len() as u64);
        if let Err(e) = inv.validate() {
            panic!("invalid inventory: {e}\n{inv}");
        }
        let ranges: Vec<_> = inv.parts().iter().map(|p| p.range.clone()).collect();
        match &layout {
            None => layout = Some(ranges),
            Some(first) => assert_eq!(first, &ranges, "job options changed the part layout"),
        }
    }
});
