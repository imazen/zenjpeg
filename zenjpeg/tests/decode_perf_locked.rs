//! Decode performance regression tests.
//!
//! These tests verify that decode performance doesn't regress beyond locked
//! instruction-count thresholds. They use wall-clock timing with generous
//! margins (2x) as a proxy — callgrind is more precise but too slow for CI.
//!
//! Locked values established 2026-02-15 (commit 0c6d6ba):
//!   Baseline 2048x2048:       252.5M instructions, ~17ms wall-clock
//!   Progressive 1024x1024:    105.6M instructions, ~7.7ms wall-clock
//!   Progressive 2048x2048:    415.4M instructions, ~32ms wall-clock
//!
//! For precise regression testing, use callgrind:
//!   valgrind --tool=callgrind ./target/release/examples/valgrind_decode jpegli <file>
//!
//! Progressive test images MUST NOT have restart markers (DRI=0) because
//! zune-jpeg 0.5.12 silently corrupts progressive output with DRI, making
//! cross-decoder comparisons invalid.

#![cfg(feature = "decoder")]

use enough::Unstoppable;
use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

fn create_test_jpeg(width: u32, height: u32, progressive: bool) -> Vec<u8> {
    // Deterministic noise+patches pattern — same as decode_compare benchmark.
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let noise = (h >> 24) as u8;
            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / width as usize) as u8;
                    data[idx + 1] = ((y * 255) / height as usize) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255 - edge;
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }
    let mut config =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(progressive);
    if progressive {
        // No DRI for progressive — zune-jpeg 0.5.12 corrupts output with restart markers
        config = config.restart_mcu_rows(0);
    }
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn decode_and_measure(jpeg_data: &[u8]) -> (std::time::Duration, usize) {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let start = std::time::Instant::now();
    let result = decoder
        .decode(jpeg_data, Unstoppable)
        .expect("decode failed");
    let elapsed = start.elapsed();
    let len = result.pixels_u8().map(|p| p.len()).unwrap_or(0);
    (elapsed, len)
}

/// Hash decode output for correctness verification.
fn hash_pixels(jpeg_data: &[u8]) -> u64 {
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder
        .decode(jpeg_data, Unstoppable)
        .expect("decode failed");
    let pixels = result.pixels_u8().unwrap();
    // FNV-1a hash
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in pixels {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// =============================================================================
// LOCKED VALUES (2026-02-15, commit 0c6d6ba)
// =============================================================================

// Instruction counts (callgrind, deterministic):
const LOCKED_IR_BASE_2048: u64 = 252_500_000;
const LOCKED_IR_PROG_1024: u64 = 105_600_000;
const LOCKED_IR_PROG_2048: u64 = 415_400_000;

// Wall-clock regression threshold: allow 3x overhead (covers noisy CI, system load).
// For precise checking, use callgrind via valgrind_decode example.
const WALL_CLOCK_MARGIN: f64 = 3.0;

// Locked wall-clock baselines (release build, AMD Zen 4):
const LOCKED_MS_BASE_2048: f64 = 17.0;
const LOCKED_MS_PROG_1024: f64 = 8.0;
const LOCKED_MS_PROG_2048: f64 = 32.0;

// Pixel output hashes — if these change, the decoder output changed.
// Progressive and baseline produce identical pixels (same quant tables, integer IDCT).
const HASH_BASE_2048: u64 = 0x54e96d7d25d89b9b;
const HASH_PROG_1024: u64 = 0x65db2dbe1148db64;
const HASH_PROG_2048: u64 = 0x54e96d7d25d89b9b;

// =============================================================================
// TESTS
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release --features decoder --test decode_perf_locked -- --ignored --nocapture
fn test_decode_perf_baseline_2048() {
    let jpeg = create_test_jpeg(2048, 2048, false);
    eprintln!("Baseline 2048x2048 JPEG: {} bytes", jpeg.len());

    // Warmup
    let _ = decode_and_measure(&jpeg);

    // Measure (best of 3)
    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let (elapsed, len) = decode_and_measure(&jpeg);
        assert_eq!(len, 2048 * 2048 * 3);
        if elapsed < best {
            best = elapsed;
        }
    }
    let ms = best.as_secs_f64() * 1000.0;
    eprintln!(
        "Baseline 2048: {:.2}ms (locked: {:.1}ms, threshold: {:.1}ms)",
        ms,
        LOCKED_MS_BASE_2048,
        LOCKED_MS_BASE_2048 * WALL_CLOCK_MARGIN
    );
    eprintln!(
        "Callgrind locked: {}M instructions",
        LOCKED_IR_BASE_2048 / 1_000_000
    );

    assert!(
        ms < LOCKED_MS_BASE_2048 * WALL_CLOCK_MARGIN,
        "Baseline 2048 regression: {:.2}ms > {:.1}ms threshold",
        ms,
        LOCKED_MS_BASE_2048 * WALL_CLOCK_MARGIN
    );

    let hash = hash_pixels(&jpeg);
    eprintln!("Output hash: {:#018x}", hash);
    if HASH_BASE_2048 != 0 {
        assert_eq!(hash, HASH_BASE_2048, "Baseline 2048 output changed!");
    }
}

#[test]
#[ignore]
fn test_decode_perf_progressive_1024() {
    let jpeg = create_test_jpeg(1024, 1024, true);
    eprintln!("Progressive 1024x1024 JPEG (no DRI): {} bytes", jpeg.len());

    let _ = decode_and_measure(&jpeg);

    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let (elapsed, len) = decode_and_measure(&jpeg);
        assert_eq!(len, 1024 * 1024 * 3);
        if elapsed < best {
            best = elapsed;
        }
    }
    let ms = best.as_secs_f64() * 1000.0;
    eprintln!(
        "Progressive 1024: {:.2}ms (locked: {:.1}ms, threshold: {:.1}ms)",
        ms,
        LOCKED_MS_PROG_1024,
        LOCKED_MS_PROG_1024 * WALL_CLOCK_MARGIN
    );
    eprintln!(
        "Callgrind locked: {}M instructions",
        LOCKED_IR_PROG_1024 / 1_000_000
    );

    assert!(
        ms < LOCKED_MS_PROG_1024 * WALL_CLOCK_MARGIN,
        "Progressive 1024 regression: {:.2}ms > {:.1}ms threshold",
        ms,
        LOCKED_MS_PROG_1024 * WALL_CLOCK_MARGIN
    );

    let hash = hash_pixels(&jpeg);
    eprintln!("Output hash: {:#018x}", hash);
    if HASH_PROG_1024 != 0 {
        assert_eq!(hash, HASH_PROG_1024, "Progressive 1024 output changed!");
    }
}

#[test]
#[ignore]
fn test_decode_perf_progressive_2048() {
    let jpeg = create_test_jpeg(2048, 2048, true);
    eprintln!("Progressive 2048x2048 JPEG (no DRI): {} bytes", jpeg.len());

    let _ = decode_and_measure(&jpeg);

    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let (elapsed, len) = decode_and_measure(&jpeg);
        assert_eq!(len, 2048 * 2048 * 3);
        if elapsed < best {
            best = elapsed;
        }
    }
    let ms = best.as_secs_f64() * 1000.0;
    eprintln!(
        "Progressive 2048: {:.2}ms (locked: {:.1}ms, threshold: {:.1}ms)",
        ms,
        LOCKED_MS_PROG_2048,
        LOCKED_MS_PROG_2048 * WALL_CLOCK_MARGIN
    );
    eprintln!(
        "Callgrind locked: {}M instructions",
        LOCKED_IR_PROG_2048 / 1_000_000
    );

    assert!(
        ms < LOCKED_MS_PROG_2048 * WALL_CLOCK_MARGIN,
        "Progressive 2048 regression: {:.2}ms > {:.1}ms threshold",
        ms,
        LOCKED_MS_PROG_2048 * WALL_CLOCK_MARGIN
    );

    let hash = hash_pixels(&jpeg);
    eprintln!("Output hash: {:#018x}", hash);
    if HASH_PROG_2048 != 0 {
        assert_eq!(hash, HASH_PROG_2048, "Progressive 2048 output changed!");
    }
}
