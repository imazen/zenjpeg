//! #92: the "decode a growing prefix" contract, checked at EVERY byte offset
//! of small baseline / progressive / restart-interval / grayscale fixtures.
//!
//! For each prefix `data[..n]` under the default (`Balanced`) strictness:
//! - decoding never panics (the whole test is the assertion);
//! - an `Ok` carries the header dimensions;
//! - **monotone acceptance**: once some prefix decodes, every longer prefix
//!   decodes too — a consumer re-decoding on each network chunk must never
//!   see a partial image turn back into an error;
//! - **differential**: if `Strict` accepts the prefix, the `Balanced` pixmap
//!   is identical to Strict's;
//! - a prefix that decodes but is shorter than the last scan's end reports
//!   at least one `Truncated*` warning (the signal a streaming consumer keys
//!   off), and the complete stream reports none of them;
//! - **zero fill is zero fill**: for a baseline prefix, every pixel row
//!   that starts two MCU rows or more below the cut MCU is the exact
//!   neutral grey a zero block decodes to. The streaming paths used to skip
//!   the IDCT for truncated blocks and ship whatever the previous MCU row
//!   had left in the strip buffer (#92);
//! - **no phantom data**: in the coefficient domain every prefix is at least
//!   as close to the complete decode as the previous prefix was, coefficient
//!   by coefficient. A decoder that keeps "decoding" past the cut against
//!   synthetic zero bits fails this — that is how 66k `-1 << al` phantom
//!   coefficients per truncated progressive scan were found (#92).

use enough::Unstoppable;
use zenjpeg::decoder::{DecodeWarning, Decoder, Strictness};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn noise_rgb(w: u32, h: u32, seed0: u32) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 3) as usize];
    let mut seed = seed0;
    for (i, b) in v.iter_mut().enumerate() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        // Blocky low-frequency structure + noise so every scan carries data.
        let x = (i / 3) % w as usize;
        let y = (i / 3) / w as usize;
        *b = (((x / 8 + y / 8) * 37) as u8).wrapping_add((seed >> 27) as u8);
    }
    v
}

fn fixture(name: &str, w: u32, h: u32, cfg: EncoderConfig, gray: bool) -> (String, Vec<u8>) {
    let layout = if gray {
        PixelLayout::Gray8Srgb
    } else {
        PixelLayout::Rgb8Srgb
    };
    let rgb = noise_rgb(w, h, 0xC0FFEE);
    let px: Vec<u8> = if gray {
        rgb.as_chunks::<3>().0.iter().map(|p| p[1]).collect()
    } else {
        rgb
    };
    let mut enc = cfg.encode_from_bytes(w, h, layout).expect("encoder");
    enc.push_packed(&px, Unstoppable).expect("push");
    (name.to_string(), enc.finish().expect("finish"))
}

fn fixtures() -> Vec<(String, Vec<u8>)> {
    let (w, h) = (72u32, 56u32); // non-MCU-aligned on both axes
    vec![
        fixture(
            "baseline-420",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter).progressive(false),
            false,
        ),
        fixture(
            "baseline-444-dri",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::None)
                .progressive(false)
                .restart_mcu_rows(1),
            false,
        ),
        fixture(
            "progressive-420",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter).progressive(true),
            false,
        ),
        fixture(
            "progressive-420-dri",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter)
                .progressive(true)
                .restart_mcu_rows(1),
            false,
        ),
        fixture(
            "gray-baseline",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::None).progressive(false),
            true,
        ),
    ]
}

/// MCU size in pixels for a fixture, by name.
fn mcu_size(name: &str) -> (u32, u32) {
    if name.contains("-420") {
        (16, 16)
    } else {
        (8, 8)
    }
}

/// For a truncated **baseline** decode, every pixel row that starts at least
/// two MCU rows below the MCU the cut landed in must be the neutral grey of
/// a zero block (Y = Cb = Cr = 128 → RGB 128). Two MCU rows of margin keep
/// fancy chroma upsampling's vertical blend against the last real row out of
/// the checked region.
fn assert_zero_fill_below_cut(name: &str, n: usize, img: &zenjpeg::decoder::DecodeResult) {
    let Some(DecodeWarning::TruncatedScan { blocks_decoded, .. }) = img
        .warnings()
        .iter()
        .find(|w| matches!(w, DecodeWarning::TruncatedScan { .. }))
    else {
        return;
    };
    let (mcu_w, mcu_h) = mcu_size(name);
    let mcu_cols = img.width().div_ceil(mcu_w);
    let cut_mcu_row = blocks_decoded / mcu_cols;
    let first_checked_row = (cut_mcu_row + 2) * mcu_h;
    let px = img.pixels_u8().expect("u8 pixels");
    let bpp = img.bytes_per_pixel();
    let stride = img.stride();
    for y in first_checked_row..img.height() {
        let row = &px[y as usize * stride..y as usize * stride + img.width() as usize * bpp];
        if let Some(i) = row.iter().position(|&v| v != 128) {
            panic!(
                "{name}@{n}: pixel row {y} (cut in MCU row {cut_mcu_row}) is not zero-filled: byte {i} = {} (expected 128 everywhere); the strip carried stale pixels from an earlier MCU row",
                row[i]
            );
        }
    }
}

fn is_truncation_warning(w: &DecodeWarning) -> bool {
    matches!(
        w,
        DecodeWarning::TruncatedScan { .. }
            | DecodeWarning::TruncatedBetweenScans { .. }
            | DecodeWarning::TruncatedProgressiveScan
    )
}

/// Offset one past the last entropy-coded byte: everything from here on is
/// trailer (the EOI marker), so a prefix at or beyond it lost no image data.
fn end_of_last_scan(jpeg: &[u8]) -> usize {
    assert_eq!(
        &jpeg[jpeg.len() - 2..],
        &[0xFF, 0xD9],
        "fixture must end in EOI"
    );
    jpeg.len() - 2
}

#[test]
fn every_prefix_decodes_consistently() {
    for (name, jpeg) in fixtures() {
        let full = Decoder::new()
            .decode(&jpeg, Unstoppable)
            .expect("full decode");
        let (fw, fh) = (full.width(), full.height());
        assert!(
            !full.warnings().iter().any(is_truncation_warning),
            "{name}: complete stream reports a truncation warning: {:?}",
            full.warnings()
        );
        let scan_end = end_of_last_scan(&jpeg);

        let final_coeffs = coefficient_snapshot(&jpeg).expect("full coefficient decode");
        let mut prev_coeffs: Option<Vec<i16>> = None;
        let mut first_ok: Option<usize> = None;
        let mut strict_ok_count = 0usize;
        for n in 0..=jpeg.len() {
            let prefix = &jpeg[..n];
            let balanced = Decoder::new().decode(prefix, Unstoppable);
            let strict = Decoder::new()
                .strictness(Strictness::Strict)
                .decode(prefix, Unstoppable);

            match &balanced {
                Ok(img) => {
                    assert_eq!((img.width(), img.height()), (fw, fh), "{name}@{n}: dims");
                    if first_ok.is_none() {
                        first_ok = Some(n);
                    }
                    let coeffs = coefficient_snapshot(prefix).unwrap_or_else(|| {
                        panic!("{name}@{n}: pixels decode but coefficients don't")
                    });
                    assert_eq!(
                        coeffs.len(),
                        final_coeffs.len(),
                        "{name}@{n}: coefficient count"
                    );
                    if let Some(prev) = &prev_coeffs {
                        let regressed = final_coeffs
                            .iter()
                            .zip(&coeffs)
                            .zip(prev)
                            .enumerate()
                            .filter(|(_, ((f, c), p))| {
                                (*f - *c).unsigned_abs() > (*f - *p).unsigned_abs()
                            })
                            .map(|(k, ((f, c), p))| format!("#{k}: final {f}, was {p}, now {c}"))
                            .collect::<Vec<_>>();
                        assert!(
                            regressed.is_empty(),
                            "{name}@{n}: {} coefficient(s) moved away from the final value (phantom data decoded past the cut); first: {:?}",
                            regressed.len(),
                            &regressed[..regressed.len().min(6)]
                        );
                    }
                    prev_coeffs = Some(coeffs);
                    if n < scan_end {
                        assert!(
                            img.warnings().iter().any(is_truncation_warning),
                            "{name}@{n}: prefix lost scan data but reports no Truncated* warning: {:?}",
                            img.warnings()
                        );
                    }
                    if name.contains("baseline") {
                        assert_zero_fill_below_cut(&name, n, img);
                    }
                }
                Err(e) => {
                    if let Some(f) = first_ok {
                        panic!("{name}: prefix {f} decoded but longer prefix {n} errored: {e}");
                    }
                }
            }
            if let Ok(s) = &strict {
                strict_ok_count += 1;
                let b = balanced
                    .as_ref()
                    .unwrap_or_else(|e| panic!("{name}@{n}: Strict ok but Balanced err: {e}"));
                assert_eq!(
                    s.pixels_u8(),
                    b.pixels_u8(),
                    "{name}@{n}: Strict/Balanced pixels"
                );
            }
        }
        let f = first_ok.unwrap_or_else(|| panic!("{name}: no prefix ever decoded"));
        assert!(f < jpeg.len(), "{name}: only the complete stream decodes");
        assert!(
            strict_ok_count >= 1,
            "{name}: Strict never accepted even the full stream"
        );
        eprintln!(
            "{name}: {} bytes, first decodable prefix {f} ({:.0}%), strict-ok prefixes {strict_ok_count}",
            jpeg.len(),
            100.0 * f as f64 / jpeg.len() as f64
        );
    }
}

/// Quantized coefficients of every component, flattened, for one decode of
/// `prefix`. `None` when the prefix does not decode yet.
fn coefficient_snapshot(prefix: &[u8]) -> Option<Vec<i16>> {
    let c = Decoder::new()
        .decode_coefficients(prefix, Unstoppable)
        .ok()?;
    Some(
        c.components
            .iter()
            .flat_map(|comp| comp.coeffs.iter().copied())
            .collect(),
    )
}

/// Root-mean-square distance between two equally sized u8 pixmaps.
fn rms(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum();
    (sse / a.len() as f64).sqrt()
}

/// #92 test-plan item "simulated 8-chunk progressive arrival": a ~200 KB
/// progressive JPEG delivered in 8 equal network chunks, re-decoded from
/// byte 0 after each arrival (Chromium's cache-miss / low-end-device path).
///
/// Asserted per arrival:
/// - every cumulative prefix decodes (the header is inside chunk 1);
/// - each partial pixmap carries a `Truncated*` warning, the final one none;
/// - **coefficient-domain convergence is exact and monotone**: progressive
///   scans only ever add magnitude bits, so for every quantized coefficient
///   `|final - partial[i+1]| <= |final - partial[i]|`, and each chunk
///   strictly improves at least one coefficient (no chunk is wasted);
/// - the pixmap's RMS distance to the final image is non-increasing;
/// - the last arrival is byte-identical to a one-shot decode of the file.
///
/// The coefficient check is the strong one: it would catch phantom
/// coefficients decoded from zero-padding past the cut (a wrong pixel, not a
/// missing one), which a pixel-domain metric could hide inside rounding.
#[test]
fn chunked_progressive_arrival_converges() {
    const CHUNKS: usize = 8;
    let (name, jpeg) = fixture(
        "progressive-420-large",
        800,
        600,
        EncoderConfig::ycbcr(92.0, ChromaSubsampling::Quarter).progressive(true),
        false,
    );
    assert!(
        jpeg.len() >= 150_000,
        "{name}: fixture is only {} bytes; the 8-chunk simulation wants ~200 KB",
        jpeg.len()
    );
    let chunk = jpeg.len().div_ceil(CHUNKS);

    let one_shot = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("one-shot decode");
    let final_px = one_shot.pixels_u8().expect("u8 pixels");
    let final_coeffs = coefficient_snapshot(&jpeg).expect("one-shot coefficients");

    let mut prev_coeffs: Option<Vec<i16>> = None;
    let mut prev_rms = f64::INFINITY;
    for i in 1..=CHUNKS {
        let n = (i * chunk).min(jpeg.len());
        let prefix = &jpeg[..n];
        let img = Decoder::new()
            .decode(prefix, Unstoppable)
            .unwrap_or_else(|e| panic!("{name}: arrival {i}/{CHUNKS} ({n} bytes) errored: {e}"));
        assert_eq!(
            (img.width(), img.height()),
            (one_shot.width(), one_shot.height()),
            "{name}: arrival {i} dims"
        );
        let truncated = img.warnings().iter().any(is_truncation_warning);
        assert_eq!(
            truncated,
            i < CHUNKS,
            "{name}: arrival {i}/{CHUNKS} truncation signal wrong: {:?}",
            img.warnings()
        );

        let coeffs = coefficient_snapshot(prefix)
            .unwrap_or_else(|| panic!("{name}: arrival {i} coefficient decode failed"));
        assert_eq!(
            coeffs.len(),
            final_coeffs.len(),
            "{name}: arrival {i} coefficient count"
        );
        if let Some(prev) = &prev_coeffs {
            let mut improved = 0usize;
            let mut regressed: Vec<String> = Vec::new();
            for (k, ((f, c), p)) in final_coeffs.iter().zip(&coeffs).zip(prev).enumerate() {
                let (err_now, err_prev) = ((f - c).unsigned_abs(), (f - p).unsigned_abs());
                if err_now > err_prev {
                    regressed.push(format!("#{k}: final {f}, was {p}, now {c}"));
                }
                improved += usize::from(err_now < err_prev);
            }
            assert!(
                regressed.is_empty(),
                "{name}: arrival {i}: {} coefficient(s) moved AWAY from final (phantom values decoded past the cut?); first: {:?}",
                regressed.len(),
                &regressed[..regressed.len().min(8)]
            );
            assert!(improved > 0, "{name}: arrival {i} improved no coefficient");
        }

        let px = img.pixels_u8().expect("u8 pixels");
        let d = rms(px, final_px);
        assert!(
            d <= prev_rms,
            "{name}: arrival {i} RMS to final rose {prev_rms:.4} -> {d:.4}"
        );
        eprintln!("{name}: arrival {i}/{CHUNKS} {n} bytes, rms-to-final {d:.4}");
        prev_rms = d;
        prev_coeffs = Some(coeffs);
    }
    assert_eq!(
        prev_rms, 0.0,
        "{name}: final arrival differs from one-shot decode"
    );
    let last = Decoder::new().decode(&jpeg, Unstoppable).expect("final");
    assert_eq!(
        last.pixels_u8(),
        Some(final_px),
        "{name}: final arrival pixels"
    );
}

/// The fused parallel baseline path (`--features parallel`, DRI, >= 1024
/// MCUs) carries its own copy of the block loop, including the speculative
/// padding-block arm that used to rewind on `Truncated`. For a spread of cuts
/// through two large restart-interval fixtures, its output must be identical
/// to the sequential path's — same Ok/Err, same pixels, same warnings — and
/// both must zero-fill below the cut.
#[cfg(feature = "parallel")]
#[test]
fn fused_parallel_truncation_matches_sequential() {
    let fixtures = [
        fixture(
            "baseline-420-dri-large",
            632, // 4:2:0 with a partial MCU column AND a partial block column
            472,
            EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                .progressive(false)
                .restart_mcu_rows(1),
            false,
        ),
        fixture(
            "baseline-444-dri-large",
            516,
            508,
            EncoderConfig::ycbcr(85.0, ChromaSubsampling::None)
                .progressive(false)
                .restart_mcu_rows(1),
            false,
        ),
    ];
    for (name, jpeg) in fixtures {
        let scan_end = end_of_last_scan(&jpeg);
        // 257 cuts spread over the stream, plus every byte of one late
        // restart interval so cuts land in padding blocks and mid-symbol.
        let mut cuts: Vec<usize> = (0..=256).map(|i| i * jpeg.len() / 256).collect();
        cuts.extend((scan_end * 7 / 10)..(scan_end * 7 / 10 + 512).min(jpeg.len()));
        let mut ok_seen = false;
        for n in cuts {
            let prefix = &jpeg[..n];
            let par = Decoder::new().decode(prefix, Unstoppable);
            let seq = Decoder::new().num_threads(1).decode(prefix, Unstoppable);
            match (&par, &seq) {
                (Ok(p), Ok(s)) => {
                    ok_seen = true;
                    assert_eq!(p.warnings(), s.warnings(), "{name}@{n}: warnings differ");
                    assert_eq!(
                        p.pixels_u8(),
                        s.pixels_u8(),
                        "{name}@{n}: fused-parallel pixels differ from sequential"
                    );
                    if n < scan_end {
                        assert!(
                            p.warnings().iter().any(is_truncation_warning),
                            "{name}@{n}: no Truncated* warning: {:?}",
                            p.warnings()
                        );
                    }
                    assert_zero_fill_below_cut(&name, n, p);
                }
                (Err(_), Err(_)) => assert!(
                    !ok_seen,
                    "{name}@{n}: decoded prefix turned back into an error"
                ),
                (p, s) => panic!(
                    "{name}@{n}: fused-parallel {} but sequential {}",
                    p.as_ref()
                        .map(|i| format!("Ok({:?})", i.warnings()))
                        .unwrap_or_else(|e| format!("Err({e})")),
                    s.as_ref()
                        .map(|i| format!("Ok({:?})", i.warnings()))
                        .unwrap_or_else(|e| format!("Err({e})")),
                ),
            }
        }
        assert!(ok_seen, "{name}: no cut ever decoded");
    }
}
