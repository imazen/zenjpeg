//! Baseline + comparison benchmark for JPEG container scanning.
//!
//! Locks in the "before" numbers for the public scanning behaviors we care
//! about, so any refactor of the internal scanner or introduction of a new
//! public `zenjpeg::container` module can be measured against this baseline
//! without ambiguity.
//!
//! Measured behaviors (all operate on raw JPEG bytes):
//!
//! 1. `find_jpeg_boundaries` — locate all top-level SOI..EOI pairs in a
//!    byte buffer. Critical for multi-image JPEGs (Ultra HDR, depth maps).
//!    Currently implemented in `ultrahdr_core::metadata::mpf`.
//!
//! 2. `primary_bounds` — locate the first (primary) SOI..EOI. Thin
//!    specialization of (1); bencjhed separately to catch any regression
//!    in the common single-image case.
//!
//! 3. `walk_markers_naive` — a naive byte-by-byte FF-walking baseline.
//!    Represents a lower bound (no memchr SIMD) and upper bound (no
//!    entropy-awareness).
//!
//! Once `zenjpeg::container::marker` exists, each of (1) and (2) will have
//! `old` + `new` variants in the same bench group so the A/B diff is
//! visible in every run.
//!
//! Inputs:
//! - `small`: a synthetic 256x256 noise JPEG (~8 KB). Tight measurement
//!   with many iterations — catches regressions in fixed-cost overhead.
//! - `medium`: synthetic 1024x1024 (~120 KB). Realistic web photo.
//! - `pixel_ultrahdr`: Pixel 6 Pro Ultra HDR sample from codec-corpus
//!   (~2.85 MB, includes APP1 EXIF with embedded thumbnail, APP2 MPF,
//!   XMP, ISO 21496-1 payload, primary JPEG, and gain map). THIS is the
//!   realistic worst case.
//!
//! Run:
//! ```sh
//! cargo bench -p zenjpeg --bench container_scan
//! ```

use enough::Unstoppable;
use std::path::PathBuf;
use zenbench::prelude::*;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

// ──────────────────────────────────────────────────────────────────────────
// Input generation
// ──────────────────────────────────────────────────────────────────────────

/// Deterministic noise-patches RGB buffer (realistic DCT distribution).
fn noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state: u32 = 0x9e37_79b9;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let r = ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40));
            let g = ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80));
            let b = ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120));
            let i = (y * w + x) * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }
    rgb
}

fn encode_jpeg(w: u32, h: u32) -> Vec<u8> {
    let rgb = noise_patches(w as usize, h as usize);
    let cfg = EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter);
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

/// Look up a representative Pixel Ultra HDR sample from the codec-corpus
/// cache (usually ~/.cache/codec-corpus). Returns `None` if not present so
/// the bench can fall back to only synthetic inputs in sandboxed CI.
fn pixel_fixture() -> Option<Vec<u8>> {
    let home = std::env::var_os("HOME")?;
    let p: PathBuf = PathBuf::from(home).join(
        ".cache/codec-corpus/v1/ultrahdr-conformance/valid/jpeg/pixel-ultrahdr/\
               Ultra_HDR_Samples_Originals_02.jpg",
    );
    std::fs::read(&p).ok()
}

// ──────────────────────────────────────────────────────────────────────────
// Naive baselines (for comparison)
// ──────────────────────────────────────────────────────────────────────────

/// Naive byte-by-byte SOI..EOI scan with no entropy awareness and no
/// length-bearing-segment skip. Historical implementation shape we're
/// replacing. Intentionally kept in this file to freeze its behavior as a
/// lower performance bound.
fn naive_find_boundaries(data: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 1 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] == 0xD8 {
            let start = pos;
            pos += 2;
            while pos + 1 < data.len() {
                if data[pos] == 0xFF && data[pos + 1] == 0xD9 {
                    out.push((start, pos + 2));
                    pos += 2;
                    break;
                }
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }
    out
}

/// Naive SOI..first-EOI for a single image (used in ultrahdr-rs
/// `primary_bounds` today — segment-aware but scalar).
fn naive_primary_bounds(data: &[u8]) -> Option<std::ops::Range<usize>> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut pos = 2;
    while pos < data.len() - 1 {
        if data[pos] == 0xFF && data[pos + 1] == 0xD9 {
            return Some(0..pos + 2);
        }
        if data[pos] == 0xFF {
            let marker = data[pos + 1];
            if marker == 0x00 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) || marker == 0xFF
            {
                pos += 2;
                continue;
            }
            if pos + 4 <= data.len() {
                let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                if len >= 2 {
                    pos += 2 + len;
                    continue;
                }
            }
        }
        pos += 1;
    }
    None
}

// ──────────────────────────────────────────────────────────────────────────
// Bench groups
// ──────────────────────────────────────────────────────────────────────────

fn bench_find_boundaries(suite: &mut Suite, label: &str, data: &'static [u8]) {
    suite.group(format!("find_jpeg_boundaries/{label}"), |g| {
        g.throughput(Throughput::Bytes(data.len() as u64));

        g.bench("naive_byte_scan", move |b| {
            b.iter(|| {
                let r = naive_find_boundaries(black_box(data));
                black_box(r);
            })
        });

        g.bench(
            "zenjpeg::container::marker::find_jpeg_boundaries",
            move |b| {
                b.iter(|| {
                    let r = zenjpeg::container::marker::find_jpeg_boundaries(black_box(data));
                    black_box(r);
                })
            },
        );

        g.bench("zenjpeg::container::find_jpeg_boundaries", move |b| {
            b.iter(|| {
                let r = zenjpeg::container::find_jpeg_boundaries(black_box(data));
                black_box(r);
            })
        });
    });
}

fn bench_primary_bounds(suite: &mut Suite, label: &str, data: &'static [u8]) {
    suite.group(format!("primary_bounds/{label}"), |g| {
        g.throughput(Throughput::Bytes(data.len() as u64));

        g.bench("naive_segment_walk", move |b| {
            b.iter(|| {
                let r = naive_primary_bounds(black_box(data));
                black_box(r);
            })
        });

        g.bench("zenjpeg::container::primary_bounds", move |b| {
            b.iter(|| {
                let r = zenjpeg::container::primary_bounds(black_box(data));
                black_box(r);
            })
        });
    });
}

fn bench_probe_workflow(suite: &mut Suite, label: &str, data: &'static [u8]) {
    use zenjpeg::container::{Wants, parse_mpf, primary_bounds, probe};
    suite.group(format!("probe_workflow/{label}"), |g| {
        g.throughput(Throughput::Bytes(data.len() as u64));

        // Sequential walks using new zenjpeg::container APIs as independent calls.
        // (The iso21496 envelope parser is now `pub(crate)` — covered implicitly
        // by the `single_probe_all` path via `Wants::ISO_GAINMAP`.)
        g.bench("sequential_unified", move |b| {
            b.iter(|| {
                let images = zenjpeg::container::find_jpeg_boundaries(black_box(data));
                let mpf = parse_mpf(black_box(data)).ok();
                let bounds = primary_bounds(black_box(data));
                black_box((images, mpf, bounds));
            })
        });

        // Single walk — capture everything via probe.
        g.bench("single_probe_all", move |b| {
            b.iter(|| {
                let p = probe(black_box(data), Wants::ALL);
                black_box(p);
            })
        });

        // Short-circuit "is this UltraHDR".
        g.bench("is_ultrahdr", move |b| {
            b.iter(|| {
                let r = zenjpeg::container::is_ultrahdr(black_box(data));
                black_box(r);
            })
        });
    });
}

fn bench_all(suite: &mut Suite) {
    // Eagerly encode synthetic inputs and leak as 'static so closures can
    // capture by shared reference across zenbench's interleaved runner.
    let small: &'static [u8] = Box::leak(encode_jpeg(256, 256).into_boxed_slice());
    let medium: &'static [u8] = Box::leak(encode_jpeg(1024, 1024).into_boxed_slice());
    let pixel_opt: Option<&'static [u8]> =
        pixel_fixture().map(|v| Box::leak(v.into_boxed_slice()) as &'static [u8]);

    bench_find_boundaries(suite, "synth_256", small);
    bench_find_boundaries(suite, "synth_1024", medium);
    if let Some(p) = pixel_opt {
        bench_find_boundaries(suite, "pixel_ultrahdr", p);
    } else {
        eprintln!("SKIP pixel_ultrahdr: fixture not present under ~/.cache/codec-corpus/");
    }

    bench_primary_bounds(suite, "synth_256", small);
    bench_primary_bounds(suite, "synth_1024", medium);
    if let Some(p) = pixel_opt {
        bench_primary_bounds(suite, "pixel_ultrahdr", p);
    }

    bench_probe_workflow(suite, "synth_256", small);
    bench_probe_workflow(suite, "synth_1024", medium);
    if let Some(p) = pixel_opt {
        bench_probe_workflow(suite, "pixel_ultrahdr", p);
    }
}

zenbench::main!(bench_all);
