//! NEON-vs-forced-scalar for zenyuv's colour-conversion kernels.
//!
//! zenyuv had 35 dispatch sites and no tier benchmark — `rgb_to_yuv_bench.rs`
//! measures absolute throughput, which cannot reveal a kernel slower than the
//! scalar tier it dispatches away from. That gap was hiding real regressions
//! elsewhere in the 2026-07-29 aarch64 sweep (zenquant 0.58x, linear-srgb
//! 0.93x, zenresize 0.94x).
//!
//! Asymmetry worth measuring here specifically: ENCODE has hand-written AVX2
//! and NEON kernels (`neon_encode.rs`), while DECODE has neither — `decode.rs`
//! goes straight to the generic magetypes path on every architecture, and
//! `avx2_decode.rs` is not called from anywhere. The generic path still
//! instantiates a NEON tier, so whether that costs anything is a question for
//! measurement rather than inspection.
//!
//! NEON is BASELINE on aarch64, so the "scalar" arm is autovectorized too:
//! ~1.00x means LLVM already matched it, BELOW 1.00 is a bug.
//!
//! Run: `cargo bench -p zenyuv --bench kernel_tiers`

use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") { "neon" } else { "v3(avx2)" };

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool { false }

const W: usize = 1920;
const H: usize = 1080;

fn bench(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let n = W * H;
    let rgb: &'static [u8] =
        Box::leak((0..n * 3).map(|i| (i % 251) as u8).collect::<Vec<_>>().into_boxed_slice());

    // ---- encode: reachable from outside, and the paths with hand-written NEON ----
    use zenyuv::{Matrix, Range, SharpYuvConfig, YuvContext};

    suite.compare("encode_444_u8", |g| {
        g.throughput(Throughput::Bytes((n * 3) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || {
                    set_simd(simd);
                    (vec![0u8; n], vec![0u8; n], vec![0u8; n])
                })
                .run(move |(mut y, mut u, mut v)| {
                    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
                    ctx.encode_444_u8(rgb, &mut y, &mut u, &mut v, W, H);
                    (y, u, v)
                })
            });
        }
    });
    suite.compare("encode_420_u8", |g| {
        g.throughput(Throughput::Bytes((n * 3) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || {
                    set_simd(simd);
                    (vec![0u8; n], vec![0u8; n / 4], vec![0u8; n / 4])
                })
                .run(move |(mut y, mut u, mut v)| {
                    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
                    ctx.encode_420_u8(rgb, &mut y, &mut u, &mut v, W, H);
                    (y, u, v)
                })
            });
        }
    });
    suite.compare("encode_420_y_only_u8", |g| {
        g.throughput(Throughput::Bytes((n * 3) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || { set_simd(simd); vec![0u8; n] })
                    .run(move |mut y| {
                        let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
                        ctx.encode_420_y_only_u8(rgb, &mut y, W, H);
                        y
                    })
            });
        }
    });
    suite.compare("encode_sharp_420_u8", |g| {
        g.throughput(Throughput::Bytes((n * 3) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || {
                    set_simd(simd);
                    (vec![0u8; n], vec![0u8; n / 4], vec![0u8; n / 4])
                })
                .run(move |(mut y, mut u, mut v)| {
                    let mut ctx = YuvContext::new(Range::Full, Matrix::Bt601);
                    ctx.encode_sharp_420_u8(rgb, &mut y, &mut u, &mut v, W, H, &SharpYuvConfig::default());
                    (y, u, v)
                })
            });
        }
    });

    // DECODE is deliberately absent: `yuv*_to_rgb` are crate-internal (only
    // `YuvContext`'s encode methods are public), so a bench outside the crate
    // cannot reach them. That decode path is ALSO the one with no hand-written
    // kernel on any architecture — `decode.rs` calls the generic magetypes path
    // directly and `avx2_decode.rs` has no callers at all. Worth a look from
    // inside the crate; flagged rather than silently skipped.

    set_simd(true);
}

zenbench::main!(bench);
