//! Round-trip test: zenjpeg encodes with DRI/RSTn restart markers, then decodes.
//!
//! A restart marker (0xFFD0..=0xFFD7) resets the entropy decoder state every
//! `restart_interval` MCUs. This lets a decoder:
//!   1. Recover from bitstream corruption at known re-sync points.
//!   2. Parallelize decode across independent MCU runs (zenjpeg's
//!      `fused_parallel_decode` / `to_pixels_fast_i16_*_parallel` paths).
//!
//! This test guarantees that the sequential (non-progressive) encode path
//! respects `EncoderConfig.restart_interval` and that the decoder produces
//! output within tolerance of the same image encoded *without* restart markers.
//!
//! Run: `cargo test --release --test restart_markers --features decoder`

use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::ProgressiveScanMode;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};
use zenjpeg::types::PixelFormat;

/// 256x256 deterministic RGB gradient with a little noise so compression is
/// non-trivial (flat colour encodes to ~200 bytes and hides MCU-count bugs).
fn make_rgb_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w as usize) * (h as usize) * 3];
    for y in 0..h {
        for x in 0..w {
            let i = ((y as usize) * (w as usize) + (x as usize)) * 3;
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = ((x.wrapping_add(y).wrapping_mul(3)) & 0xFF) as u8;
            // xorshift-ish per-pixel noise so AC coefficients aren't all zero.
            let mut hsh = x
                .wrapping_mul(374761393)
                .wrapping_add(y.wrapping_mul(668265263));
            hsh = (hsh ^ (hsh >> 13)).wrapping_mul(1274126177);
            let noise = (hsh >> 28) as u8;
            buf[i] = r.wrapping_add(noise);
            buf[i + 1] = g.wrapping_add(noise);
            buf[i + 2] = b.wrapping_add(noise);
        }
    }
    buf
}

/// Count `0xFF 0xDn` restart markers (n=0..=7) in an encoded JPEG stream.
///
/// Walks the full byte stream; real restart markers never carry a stuffing byte
/// (0xFF 0x00) so the raw-byte count is the true marker count. SOI (0xFFD8)
/// and EOI (0xFFD9) fall outside 0xD0..=0xD7 so they don't false-positive.
fn count_restart_markers(jpeg: &[u8]) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF {
            let next = jpeg[i + 1];
            if (0xD0..=0xD7).contains(&next) {
                n += 1;
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    n
}

/// Report whether a DRI marker (0xFF 0xDD) is present and the declared interval.
fn find_dri_interval(jpeg: &[u8]) -> Option<u16> {
    let mut i = 0;
    while i + 5 < jpeg.len() {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDD {
            // DRI segment: FF DD <len_hi> <len_lo> <ri_hi> <ri_lo>
            // len is always 4, ri is the restart interval.
            let ri = ((jpeg[i + 4] as u16) << 8) | (jpeg[i + 5] as u16);
            return Some(ri);
        }
        i += 1;
    }
    None
}

/// Encode at Q85 4:2:0 with the given `restart_mcu_rows` value.
///
/// `restart_mcu_rows=0` disables restart markers; any positive value emits a
/// DRI marker + RSTn markers every `restart_mcu_rows` MCU rows.
fn encode_q85_420(rgb: &[u8], w: u32, h: u32, restart_mcu_rows: u16) -> Vec<u8> {
    // Force baseline (sequential, single-scan) JPEG. zenjpeg defaults to
    // progressive, and progressive mode suppresses restart markers by design
    // (since they can't be used for parallel decode of progressive streams).
    // This test is specifically about the *sequential* DRI path.
    let cfg = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(ProgressiveScanMode::Baseline)
        .restart_mcu_rows(restart_mcu_rows);
    cfg.encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("zenjpeg encode failed")
}

fn decode_to_rgb(jpeg: &[u8]) -> (Vec<u8>, u32, u32) {
    let result = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(jpeg, Unstoppable)
        .expect("zenjpeg decode failed");
    let (w, h) = result.dimensions();
    let pixels = result.into_pixels_u8().expect("expected u8 pixels");
    (pixels, w, h)
}

fn mean_abs_err(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut s = 0u64;
    for i in 0..a.len() {
        s += a[i].abs_diff(b[i]) as u64;
    }
    s as f64 / a.len() as f64
}

fn max_abs_err(a: &[u8], b: &[u8]) -> u8 {
    let mut m = 0u8;
    for i in 0..a.len() {
        let d = a[i].abs_diff(b[i]);
        if d > m {
            m = d;
        }
    }
    m
}

/// MCU dimensions for a 4:2:0 JPEG: 16x16 luma pixels. Returns
/// `(mcu_cols, mcu_rows)` — the number of MCUs across the image.
fn mcu_grid_420(w: u32, h: u32) -> (u32, u32) {
    ((w + 15) / 16, (h + 15) / 16)
}

/// The main test: encode the same image twice (with and without restart
/// markers), decode both, and assert the restart-marked version:
///   1. Actually contains restart markers (0xFFDn byte pairs).
///   2. Declares its interval via DRI (0xFFDD).
///   3. Decodes to a near-identical image as the no-restart baseline.
///
/// Uses 512x512 rather than 256x256 because zenjpeg's
/// `resolve_restart_rows` guard refuses to emit markers when the estimated
/// file size is below the overhead budget — for 256² that budget caps the
/// RST marker count at 0. See `zenjpeg/src/encode/config.rs:204-226`.
#[test]
fn sequential_baseline_with_restart_markers_roundtrips() {
    let (w, h) = (512u32, 512u32);
    let rgb = make_rgb_gradient(w, h);

    // `restart_mcu_rows = 2` → DRI every 2 MCU rows.
    // At 512x512 4:2:0 the MCU grid is 32x32, so 2 rows = 64 MCUs between
    // restarts → expect 32 rows / 2 = 16 segments → 15 RSTn markers.
    let rows: u16 = 2;
    let jpeg_with = encode_q85_420(&rgb, w, h, rows);
    let jpeg_without = encode_q85_420(&rgb, w, h, 0);

    // === 1. JPEG actually contains restart markers ===
    let rst_count = count_restart_markers(&jpeg_with);
    let rst_count_plain = count_restart_markers(&jpeg_without);

    assert!(
        rst_count > 0,
        "expected restart markers in encoded stream; found 0 (restart_mcu_rows={rows})"
    );
    assert_eq!(
        rst_count_plain, 0,
        "control case (no DRI) must not emit RSTn markers; found {rst_count_plain}"
    );

    // === 2. DRI marker is present; read the ACTUAL interval the encoder chose ===
    //
    // Note: zenjpeg's `resolve_restart_rows` treats our `rows` request as a
    // MINIMUM, not an exact value. It can be bumped up by two guards:
    //   - `MIN_MCUS_PER_RESTART` (=64): ensures each segment has enough MCUs
    //     to benefit parallel decode.
    //   - Overhead budget (0.3% file-size cap): prevents RST bloat on small
    //     images.
    // At Q85 the output is roughly 0.6-1.2 bpp, and the heuristic
    // conservatively models 0.5 bpp — so small / simple images see the
    // effective interval grow. We read the DRI back to learn what the
    // encoder picked and count markers against that, rather than asserting
    // exact user-requested rows.
    let dri = find_dri_interval(&jpeg_with)
        .expect("encoded JPEG with restart_mcu_rows>0 must contain DRI marker");
    assert!(dri > 0, "DRI interval must be > 0 when restart_mcu_rows>0");
    assert!(
        find_dri_interval(&jpeg_without).is_none(),
        "encoded JPEG without restart markers must not contain a DRI marker"
    );

    // Count of segments implied by the ACTUAL DRI = ceil(total_mcus / dri).
    // For an exact integer split we expect (total / dri) - 1 RSTn markers;
    // for a partial last segment we expect ceil(total / dri) - 1. Allow
    // +/- 2 markers for end-of-scan conventions.
    let (mcu_cols, mcu_rows_) = mcu_grid_420(w, h);
    let total_mcus = (mcu_cols as usize) * (mcu_rows_ as usize);
    let expected_segments = total_mcus.div_ceil(dri as usize);
    let expected_rst = expected_segments.saturating_sub(1);
    assert!(
        (expected_rst.saturating_sub(2)..=expected_rst + 2).contains(&rst_count),
        "restart marker count {rst_count} outside plausible range \
         {}..={} for {w}² 4:2:0 dri={dri} (requested rows={rows})",
        expected_rst.saturating_sub(2),
        expected_rst + 2
    );

    // === 3. Decoded output within tolerance of baseline ===
    let (pixels_with, dw, dh) = decode_to_rgb(&jpeg_with);
    let (pixels_without, bw, bh) = decode_to_rgb(&jpeg_without);

    assert_eq!((dw, dh), (w, h), "decoded size mismatch (restart path)");
    assert_eq!((bw, bh), (w, h), "decoded size mismatch (baseline path)");
    assert_eq!(
        pixels_with.len(),
        pixels_without.len(),
        "decoded buffer length mismatch"
    );

    // Restart markers reset the entropy state and the DC predictor at each
    // RSTn. They don't change quantization, so the decoded output is almost
    // identical to the no-restart version — small rounding differences are
    // possible where DC predictor resets interact with the quant grid, but
    // the perceptual delta should be well under 1 LSB on average.
    let mean = mean_abs_err(&pixels_with, &pixels_without);
    let max = max_abs_err(&pixels_with, &pixels_without);

    eprintln!(
        "restart-markers roundtrip ({w}x{h}): rows={rows} rst_count={rst_count} \
         dri_mcus={dri} mean_abs_err={mean:.3} max_abs_err={max}"
    );

    assert!(
        mean < 1.0,
        "mean |Δ| {mean:.3} between restart-marked and plain decode is too high"
    );
    assert!(
        max <= 4,
        "max |Δ| {max} between restart-marked and plain decode is too high \
         (restart markers shouldn't change pixel values beyond DC-predictor rounding)"
    );
}

/// Larger image — exercises more MCU rows and picks up any per-row boundary
/// bugs in the restart-marker emit path.
#[test]
fn sequential_baseline_restart_markers_1024() {
    let (w, h) = (1024u32, 1024u32);
    let rgb = make_rgb_gradient(w, h);

    let rows: u16 = 4;
    let jpeg = encode_q85_420(&rgb, w, h, rows);

    let rst_count = count_restart_markers(&jpeg);
    let dri = find_dri_interval(&jpeg).expect("DRI must be emitted");

    // 1024x1024 4:2:0 is large enough that overhead budget doesn't bump the
    // interval; the encoder should honor rows=4 directly → DRI = 4 * 64 = 256.
    let (mcu_cols, mcu_rows_) = mcu_grid_420(w, h);
    let expected_dri = (rows as u32) * mcu_cols;
    assert_eq!(
        dri as u32, expected_dri,
        "DRI mismatch on 1024² (expected exact match)"
    );

    let total_mcus = (mcu_cols as usize) * (mcu_rows_ as usize);
    let expected_segments = total_mcus.div_ceil(dri as usize);
    let expected_rst = expected_segments.saturating_sub(1);
    assert!(
        (expected_rst.saturating_sub(2)..=expected_rst + 2).contains(&rst_count),
        "1024² rows={rows} produced {rst_count} RSTn; expected ~{expected_rst}"
    );

    // Decode sanity check.
    let (pixels, dw, dh) = decode_to_rgb(&jpeg);
    assert_eq!((dw, dh), (w, h));
    assert_eq!(pixels.len(), (w as usize) * (h as usize) * 3);
}
