//! Regression tests for sweep issue #197: frequency-count vs emission
//! divergence and incomplete-table acceptance in the main encoder.
//!
//! Class-B mechanism (same as #194): if the counting pass walks blocks with
//! different DC-prediction resets than emission, a post-restart DC category
//! can appear at emit time that was never counted, gets no Huffman code, and
//! is written as ZERO bits — an undecodable stream returned as Ok.
//!
//! Every encode here is decode-validated; the content deliberately slams DC
//! between extremes across restart boundaries so an uncounted post-restart
//! category actually occurs.

use enough::Unstoppable;
use zenjpeg::decode::DecodeConfig;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

/// Alternating black/white horizontal bands, one MCU-row tall — maximal DC
/// swings exactly at restart-row boundaries.
fn dc_slam_rgb(w: u32, h: u32, band: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        let v = if (y / band) % 2 == 0 { 255u8 } else { 0 };
        for x in 0..w {
            // Slight horizontal texture so AC tables see symbols too.
            let t = ((x * 7 + y) % 13) as u8;
            px.extend_from_slice(&[v.saturating_sub(t), v, v.saturating_sub(t / 2)]);
        }
    }
    px
}

fn assert_stream_valid(jpeg: &[u8], ctx: &str) {
    // Structural entropy decode: a zero-bit (codeless-symbol) write desyncs
    // the scan and fails here. Works for YCbCr and XYB/SOF1 alike.
    let coeffs = DecodeConfig::new()
        .decode_coefficients(jpeg, Unstoppable)
        .unwrap_or_else(|e| panic!("{ctx}: encoded stream does not decode: {e}"));
    assert!(!coeffs.components.is_empty(), "{ctx}: no components");
}

#[test]
fn xyb_restart_intervals_roundtrip() {
    let (w, h) = (256u32, 256u32);
    for sub in [XybSubsampling::BQuarter, XybSubsampling::Full] {
        for q in [10.0f32, 30.0, 75.0] {
            for rows in [0u16, 1, 2] {
                let ctx = format!("xyb/{sub:?}/q{q}/restart_rows{rows}");
                // Band height = one MCU row so every restart boundary lands on
                // a black<->white transition (max DC diff right after reset).
                let band = if matches!(sub, XybSubsampling::BQuarter) {
                    16
                } else {
                    8
                };
                let mut enc = EncoderConfig::xyb(q, sub)
                    .progressive(false)
                    .optimize_huffman(true)
                    .restart_mcu_rows(rows)
                    .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(&dc_slam_rgb(w, h, band), Unstoppable)
                    .unwrap();
                let jpeg = enc.finish().unwrap_or_else(|e| panic!("{ctx}: {e}"));
                assert_stream_valid(&jpeg, &ctx);
            }
        }
    }
}

#[test]
fn ycbcr_restart_intervals_roundtrip() {
    let (w, h) = (256u32, 256u32);
    for ss in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        for opt in [true, false] {
            for rows in [0u16, 1, 3] {
                let ctx = format!("ycbcr/{ss:?}/opt{opt}/restart_rows{rows}");
                let mut enc = EncoderConfig::ycbcr(30.0, ss)
                    .progressive(false)
                    .optimize_huffman(opt)
                    .restart_mcu_rows(rows)
                    .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(&dc_slam_rgb(w, h, 16), Unstoppable)
                    .unwrap();
                let jpeg = enc.finish().unwrap_or_else(|e| panic!("{ctx}: {e}"));
                assert_stream_valid(&jpeg, &ctx);

                // Cross-decoder for the baseline YCbCr rows.
                let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
                dec.decode()
                    .unwrap_or_else(|e| panic!("{ctx}: jpeg-decoder rejected: {e}"));
            }
        }
    }
}

/// Parallel entropy encoding needs restart markers; with restart_mcu_rows(0)
/// the interval is auto-resolved ONCE at config computation (per the parallel()
/// docs), so the DRI header, the frequency count, and emission all agree.
/// Pre-fix, the emitter alone substituted interval 64: RST markers appeared
/// with NO DRI header and with DC resets the histogram never counted.
#[cfg(feature = "parallel")]
#[test]
fn parallel_with_restart_zero_stays_consistent() {
    use zenjpeg::encoder::ParallelEncoding;
    let (w, h) = (512u32, 512u32);
    let mut enc = EncoderConfig::ycbcr(50.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .optimize_huffman(true)
        .restart_mcu_rows(0)
        .parallel(ParallelEncoding::Auto)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&dc_slam_rgb(w, h, 16), Unstoppable)
        .unwrap();
    let jpeg = enc.finish().unwrap();
    assert_stream_valid(&jpeg, "parallel/restart0");
    // RST markers may only appear together with a DRI header.
    let has_dri = jpeg.windows(2).any(|w| w == [0xFF, 0xDD]);
    let has_rst = jpeg
        .windows(2)
        .any(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]));
    assert!(
        has_dri || !has_rst,
        "RST markers without a DRI header (pre-fix parallel corruption)"
    );
    let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
    dec.decode()
        .expect("parallel restart-0 output must decode in jpeg-decoder");
}

/// Custom Huffman tables must be rejected at build time when they cannot
/// cover the mode's symbol range: Annex K is complete for baseline YCbCr but
/// lacks the SOF1/XYB extended DC categories 12-15.
#[test]
fn custom_tables_validated_for_mode_coverage() {
    use zenjpeg::huffman::optimize::HuffmanTableSet;
    let annex_k = HuffmanTableSet::from_standard().unwrap();

    // YCbCr baseline: Annex K covers everything the mode can emit.
    let ok = EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .huffman(annex_k.clone())
        .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb);
    assert!(ok.is_ok(), "Annex K must remain valid for baseline YCbCr");

    // XYB: DC categories 12-15 are reachable; Annex K has no codes for them
    // and would previously encode them as ZERO bits (silent corruption).
    let err = EncoderConfig::xyb(75.0, XybSubsampling::BQuarter)
        .progressive(false)
        .huffman(annex_k)
        .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb);
    match err {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("no code"),
                "error should explain the missing coverage, got: {msg}"
            );
        }
        Ok(_) => panic!("incomplete custom tables must be rejected for XYB"),
    }
}
