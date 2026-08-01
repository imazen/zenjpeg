//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! zenjpeg has ~139 SIMD dispatch sites and ~20 files carrying NEON kernels,
//! but no bench could tell you what any of them are worth on ARM. The existing
//! SIMD benches either compare zenjpeg against mozjpeg/libjpeg-turbo (which
//! answers "are we competitive", not "is our SIMD earning its keep") or are
//! written `#[cfg(target_arch = "x86_64")]` and do not build on aarch64 at all
//! (`mage_simd.rs`). A kernel slower than its own scalar fallback is invisible
//! to both.
//!
//! This bench runs the identical encode and decode pipelines with the native
//! SIMD token disabled. (The same gap in linear-srgb was hiding a real
//! regression.)
//!
//! Run: `cargo bench -p zenjpeg --bench tier_isolation --features _dev`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use enough::Unstoppable;
use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

fn noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40));
            rgb[i + 1] = ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80));
            rgb[i + 2] = ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120));
        }
    }
    rgb
}

fn bench_tiers(suite: &mut Suite) {
    use zenjpeg::encode::EncoderConfig;
    use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};

    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, build with --features _dev). \
             Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    // 1 MP keeps a full SIMD-on/SIMD-off sweep to a sane wall time while still
    // being large enough that per-call overhead is not the story.
    let w = 1024usize;
    let h = 1024usize;
    let rgb: &'static [u8] = Box::leak(noise_patches(w, h).into_boxed_slice());

    let jpeg: &'static [u8] = Box::leak(
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .encode_bytes(rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .expect("encode for decode fixture")
            .into_boxed_slice(),
    );

    suite.compare("encode_q85_420_1MP", |g| {
        g.throughput(Throughput::Bytes((w * h * 3) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || set_simd(simd)).run(move |_| {
                    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
                    config
                        .encode_bytes(rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
                        .unwrap()
                })
            });
        }
    });

    suite.compare("decode_q85_420_1MP", |g| {
        g.throughput(Throughput::Bytes(jpeg.len() as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || set_simd(simd)).run(move |_| {
                    zenjpeg::decoder::Decoder::new()
                        .decode(jpeg, Unstoppable)
                        .unwrap()
                })
            });
        }
    });

    set_simd(true);

    // ---- per-kernel: the forward DCT ----
    //
    // The whole-image rows above cannot size one kernel: encode also does
    // colour conversion, quantization and entropy coding, so even a large DCT
    // win lands as a few percent end-to-end. This measures it directly.
    //
    // It exists because `forward_dct_8x8` had NO aarch64 arm. x86 and wasm each
    // dispatched to a vector path; ARM fell through to `forward_dct_8x8_scalar`,
    // so the `_neon` variant that `#[magetypes(v3, neon, wasm128, scalar)]`
    // already generates for `forward_dct_8x8_simd_chained_fallback` was
    // unreachable. Nothing failed — the vector code simply never ran.
    {
        let mut blk = [0.0f32; 64];
        let mut st = 0x9E37_79B9u32;
        for v in blk.iter_mut() {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((st >> 16) as f32 / 65535.0) * 510.0 - 255.0;
        }
        let blk: &'static [f32; 64] = Box::leak(Box::new(blk));
        suite.compare("forward_dct_8x8", |g| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.iter(move || {
                        set_simd(simd);
                        zenjpeg::encode::dct::forward_dct_8x8(blk)
                    })
                });
            }
        });
    }

    // ---- linear/HDR strip-encode kernels ----
    //
    // These four had an x86 fused-SIMD arm and NO aarch64 arm, so ARM ran the
    // scalar loop. They could not be fixed by re-attributing the x86 body: its
    // only sRGB step lives in `linear_srgb::tokens::x8`, and that whole module
    // is x86-only. The ARM arms are built on `tokens::x4` instead, running the
    // 4-wide transfer twice per 8 values.
    {
        let xf: &'static [f32; 8] =
            Box::leak(Box::new([0.0, 0.001, 0.0031308, 0.01, 0.2, 0.5, 1.0, 4.0]));
        let xu: &'static [u16; 8] =
            Box::leak(Box::new([0, 1, 205, 512, 13107, 32768, 60000, 65535]));
        macro_rules! k {
            ($name:expr, $call:expr) => {
                suite.compare($name, |g| {
                    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                        g.bench(arm, move |b| {
                            b.iter(move || {
                                set_simd(simd);
                                $call
                            })
                        });
                    }
                });
            };
        }
        k!("linear_to_srgb_255_x8",
           zenjpeg::encode::linear_lut::linear_to_srgb_255_x8(xf));
        k!("linear_u16_to_srgb_255_x8",
           zenjpeg::encode::linear_lut::linear_u16_to_srgb_255_x8(xu));
        k!("linear_rgb16_to_ycbcr_x8",
           zenjpeg::encode::linear_lut::linear_rgb16_to_ycbcr_x8(xu, xu, xu));
        k!("linear_rgbf32_to_ycbcr_x8",
           zenjpeg::encode::linear_lut::linear_rgbf32_to_ycbcr_x8(xf, xf, xf));
    }
}

zenbench::main!(bench_tiers);
