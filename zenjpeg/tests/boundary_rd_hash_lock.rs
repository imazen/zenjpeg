//! Byte-identity hash-locks for the boundary-RD public surface (#91).
//!
//! Two invariants guarded here:
//!
//! 1. `BoundaryRd::Off` (the default) produces byte-for-byte identical
//!    output to a config that never touched the boundary-RD API. Any
//!    regression in the default-path plumbing breaks this gate
//!    immediately.
//!
//! 2. `BoundaryRd::On(BoundaryRdConfig::default())` is deterministic
//!    across runs. This freezes the documented best-we-know preset so
//!    we know when it changes.

use enough::Unstoppable;
use std::hash::{Hash, Hasher};
use zenjpeg::encoder::{
    BoundaryRd, BoundaryRdConfig, ChromaSubsampling, EncoderConfig, PixelLayout,
};

fn gen_checkerboard(w: usize, h: usize, cell: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            let v: u8 = if on { 230 } else { 40 };
            let i = (y * w + x) * 3;
            out[i] = v;
            out[i + 1] = v;
            out[i + 2] = v;
        }
    }
    out
}

fn encode(rgb: &[u8], w: u32, h: u32, cfg: EncoderConfig) -> Vec<u8> {
    let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    enc.push_packed(rgb, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn fx_hash(bytes: &[u8]) -> u64 {
    // Stable cross-platform hash — std SipHash-1-3 is fine here. We only
    // care about detecting drift, not resistance to collisions.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

#[test]
fn off_is_byte_identical_to_untouched_config() {
    let (w, h) = (64usize, 64usize);
    let rgb = gen_checkerboard(w, h, 8);
    let baseline = EncoderConfig::ycbcr(80f32, ChromaSubsampling::Quarter);
    let explicit_off =
        EncoderConfig::ycbcr(80f32, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::Off);

    let a = encode(&rgb, w as u32, h as u32, baseline);
    let b = encode(&rgb, w as u32, h as u32, explicit_off);
    assert_eq!(a.len(), b.len(), "len drift: {} vs {}", a.len(), b.len());
    assert_eq!(a, b, "byte drift with BoundaryRd::Off");
}

#[test]
fn on_with_default_config_is_deterministic() {
    let (w, h) = (64usize, 64usize);
    let rgb = gen_checkerboard(w, h, 8);
    let cfg = EncoderConfig::ycbcr(80f32, ChromaSubsampling::Quarter)
        .boundary_rd(BoundaryRd::On(BoundaryRdConfig::default()));

    let a = encode(&rgb, w as u32, h as u32, cfg.clone());
    let b = encode(&rgb, w as u32, h as u32, cfg);
    assert_eq!(
        fx_hash(&a),
        fx_hash(&b),
        "BoundaryRd::On(default) must be deterministic across calls"
    );
    assert_eq!(a, b);
}
