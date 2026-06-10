//! ProgressiveScanMode::Smallest: exact min-bytes entropy selection.
//!
//! The contract: Smallest's output EQUALS the byte-minimum over the
//! explicit candidates (sequential, sequential+tiny, progressive), and
//! all candidates decode to identical pixels (it is a pure rate
//! decision). Not approximately — exactly.

use enough::Unstoppable;
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, PixelLayout, ProgressiveScanMode, TinyFileMode,
};

fn photo_ish_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    let mut state = 0x9E3779B97F4A7C15u64;
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let n = (state >> 32) as u32;
            let r = ((x * 2 + y) % 256) as u8 ^ (n & 0x3F) as u8;
            let g = ((x + y * 3) % 256) as u8 ^ ((n >> 6) & 0x3F) as u8;
            let b = ((x * 3 + y * 2) % 256) as u8 ^ ((n >> 12) & 0x3F) as u8;
            v.extend_from_slice(&[r, g, b]);
        }
    }
    v
}

fn encode(config: &EncoderConfig, rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn decode(jpeg: &[u8]) -> Vec<u8> {
    let decoder = zenjpeg::decoder::Decoder::new();
    let result = decoder.decode(jpeg, Unstoppable).expect("decodable");
    result.pixels_u8().expect("u8 pixels").to_vec()
}

fn candidates(base: &EncoderConfig, rgb: &[u8], w: u32, h: u32) -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "sequential",
            encode(
                &base
                    .clone()
                    .progressive(ProgressiveScanMode::Baseline)
                    .tiny_file_mode(TinyFileMode::Off),
                rgb,
                w,
                h,
            ),
        ),
        (
            "sequential+tiny",
            encode(
                &base
                    .clone()
                    .progressive(ProgressiveScanMode::Baseline)
                    .tiny_file_mode(TinyFileMode::Force),
                rgb,
                w,
                h,
            ),
        ),
        (
            "progressive",
            encode(
                &base.clone().progressive(ProgressiveScanMode::Progressive),
                rgb,
                w,
                h,
            ),
        ),
    ]
}

fn assert_smallest_is_exact_min(w: u32, h: u32, q: f32, sub: ChromaSubsampling) {
    let rgb = photo_ish_rgb(w, h);
    let base = EncoderConfig::ycbcr(q, sub);

    let smallest = encode(
        &base.clone().progressive(ProgressiveScanMode::Smallest),
        &rgb,
        w,
        h,
    );
    let cands = candidates(&base, &rgb, w, h);
    let min = cands.iter().map(|(_, b)| b.len()).min().unwrap();

    let sizes: Vec<_> = cands.iter().map(|(n, b)| (*n, b.len())).collect();
    assert_eq!(
        smallest.len(),
        min,
        "{w}x{h} q={q}: Smallest ({}) must EQUAL min of candidates {:?}",
        smallest.len(),
        sizes,
    );

    // Pure rate decision: every candidate decodes to the same pixels.
    let reference = decode(&smallest);
    for (name, bytes) in &cands {
        assert_eq!(
            decode(bytes),
            reference,
            "{name} must decode identically to Smallest (rate-only decision)"
        );
    }
}

#[test]
fn smallest_equals_min_on_normal_image() {
    // Progressive should win here — and Smallest must match it exactly.
    assert_smallest_is_exact_min(200, 160, 85.0, ChromaSubsampling::Quarter);
    assert_smallest_is_exact_min(200, 160, 10.0, ChromaSubsampling::Quarter);
}

#[test]
fn smallest_equals_min_on_tiny_image() {
    // The tiny bucket is where sequential(+tiny) can win — the case that
    // used to need a pixel-count heuristic.
    assert_smallest_is_exact_min(48, 48, 85.0, ChromaSubsampling::Quarter);
    assert_smallest_is_exact_min(32, 24, 50.0, ChromaSubsampling::None);
}

#[test]
fn smallest_respects_tiny_off() {
    // With TinyFileMode::Off the tiny candidate must not participate.
    let (w, h) = (48u32, 48u32);
    let rgb = photo_ish_rgb(w, h);
    let base =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).tiny_file_mode(TinyFileMode::Off);

    let smallest = encode(
        &base.clone().progressive(ProgressiveScanMode::Smallest),
        &rgb,
        w,
        h,
    );
    let seq = encode(
        &base.clone().progressive(ProgressiveScanMode::Baseline),
        &rgb,
        w,
        h,
    );
    let prog = encode(
        &base.clone().progressive(ProgressiveScanMode::Progressive),
        &rgb,
        w,
        h,
    );
    assert_eq!(smallest.len(), seq.len().min(prog.len()));
}
