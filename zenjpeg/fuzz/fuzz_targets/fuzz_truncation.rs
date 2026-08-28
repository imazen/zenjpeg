//! Truncation fuzzer (#92): the decoder must accept ANY prefix of a JPEG
//! byte stream without panicking, and whatever it returns must be
//! self-consistent — an `Ok` carries the header's dimensions, and once a
//! prefix decodes, a longer prefix must never fail with a *truncation*
//! error (more bytes can never turn a partial image back into "not enough
//! bytes"). When the whole input decodes — so every prefix is a pure
//! truncation of a clean stream — a longer prefix must not fail at all.
//! A longer prefix of a mutated stream MAY fail with a corruption error
//! (`Balanced` rejects bad Huffman data by design; see `Strictness`): the
//! extra bytes revealed the corruption, they did not un-decode the prefix.
//!
//! The differential half: whenever `Strict` accepts a prefix, the default
//! (`Balanced`) decode of that prefix yields the identical pixmap.
//!
//! Seed with real JPEGs (`corpus/`, `seeds/`): libFuzzer's mutations then
//! mostly land as truncations/corruptions of valid streams, which is the
//! streaming-consumer contract this target guards.
#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::decoder::{Decoder, ErrorKind, Strictness};

const MAX_PX: u64 = 1_000_000;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    // Truncation points: a handful spread over the stream plus one driven
    // by the last byte so the fuzzer can steer the cut precisely.
    let n = data.len();
    let steer = (data[n - 1] as usize * n) / 256;
    let cuts = [n, n - 1, n - 2, n / 2, n / 4, (n * 3) / 4, steer.max(2)];

    let mut first_ok: Option<usize> = None;
    let ok_at = |len: usize| -> bool {
        Decoder::new()
            .max_pixels(MAX_PX)
            .decode(&data[..len], enough::Unstoppable)
            .is_ok()
    };
    // Every prefix of a stream that decodes in full is a pure truncation.
    let clean_stream = ok_at(n);

    let mut sorted = cuts;
    sorted.sort_unstable();
    for &len in &sorted {
        let full = Decoder::new()
            .max_pixels(MAX_PX)
            .decode(&data[..len], enough::Unstoppable);
        let strict = Decoder::new()
            .max_pixels(MAX_PX)
            .strictness(Strictness::Strict)
            .decode(&data[..len], enough::Unstoppable);

        if let Ok(img) = &full {
            // Dimensions come from the header, never from how much scan
            // data survived.
            if let Ok(info) = Decoder::new().max_pixels(MAX_PX).read_info(&data[..len]) {
                assert_eq!(img.width(), info.dimensions.width);
                assert_eq!(img.height(), info.dimensions.height);
            }
            if first_ok.is_none() {
                first_ok = Some(len);
            }
            // Strict accepting the prefix means it is a complete, clean
            // stream: the tolerant decode must not differ.
            if let Ok(s) = &strict {
                assert_eq!(s.pixels_u8(), img.pixels_u8(), "Strict/Balanced pixel mismatch at {len}");
            }
        } else if let Some(f) = first_ok {
            // Monotone: a longer prefix of a stream that already decoded
            // must never come back as "not enough data", and on a clean
            // stream must still decode outright.
            let err = full.as_ref().err();
            let is_truncation = err
                .map(|e| matches!(e.kind(), ErrorKind::TruncatedData { .. }))
                .unwrap_or(false);
            assert!(
                !(is_truncation || clean_stream) || !ok_at(f),
                "prefix {f} decoded but longer prefix {len} errored (clean stream: {clean_stream}): {err:?}"
            );
        }
    }

    // Row-callback path must survive the same cuts.
    for &len in &sorted {
        let _ = Decoder::new().max_pixels(MAX_PX).decode_rows(
            &data[..len],
            zenjpeg::decoder::PixelFormat::Rgb,
            |_| Ok(()),
            enough::Unstoppable,
        );
        let _ = Decoder::new()
            .max_pixels(MAX_PX)
            .decode_coefficients(&data[..len], enough::Unstoppable);
    }
});
