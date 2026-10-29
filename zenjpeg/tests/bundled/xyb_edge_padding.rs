//! Regression test for issue #186: XYB bottom-partial-strip vertical
//! padding used the wrong stride.
//!
//! `pad_strips_vertically` replicated cb/cr rows at packed `width`
//! stride, but under XYB `convert_strip_to_xyb` had already rearranged
//! cb_strip (the perceptual-Y plane) to padded stride — so when
//! `width % 8 != 0` AND the bottom strip was partial, the replicated
//! rows landed at shifted offsets (corrupting the tail of the last real
//! row and filling pad rows with phase-shifted data). The B plane
//! (cr_down) additionally kept the previous strip's stale rows.
//!
//! Detection: a vertical-stripe image makes every 8-row band identical,
//! so any padding corruption shows up as elevated error in the last
//! band relative to the interior. Pre-fix, XYB-Full at 130x67 measured
//! a last-band/interior ratio of 1.15; post-fix all cases are ~1.00.

use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};
use zenjpeg::types::PixelFormat;

/// High-contrast vertical stripes: every row identical, content varies
/// horizontally, so a horizontally-shifted or stale pad row changes the
/// bottom blocks' coefficients and rings into the real rows.
fn vstripe_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let v = if (x / 2) % 2 == 0 { 230u8 } else { 25u8 };
            let i = (y * width + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = 255 - v;
            rgb[i + 2] = v / 2 + 60;
        }
    }
    rgb
}

/// Mean absolute error per 8-row band.
fn band_errors(src: &[u8], dec: &[u8], w: usize, h: usize) -> Vec<f64> {
    let mut bands = Vec::new();
    let mut y = 0;
    while y < h {
        let y_end = (y + 8).min(h);
        let (mut sum, mut n) = (0u64, 0u64);
        for row in y..y_end {
            for x in 0..w * 3 {
                let i = row * w * 3 + x;
                sum += (src[i] as i32 - dec[i] as i32).unsigned_abs() as u64;
                n += 1;
            }
        }
        bands.push(sum as f64 / n as f64);
        y = y_end;
    }
    bands
}

#[test]
fn xyb_bottom_strip_padding_no_edge_error() {
    // Trigger sizes: width not a multiple of 8 AND height leaving a
    // partial bottom strip (strip height is 16 for BQuarter, 8 for
    // Full). The pre-fix defect measured ratio 1.15 at 130x67 Full.
    for (w, h) in [(130usize, 67usize), (131, 71)] {
        let src = vstripe_image(w, h);
        for (mode, sub) in [
            ("BQuarter", XybSubsampling::BQuarter),
            ("Full", XybSubsampling::Full),
        ] {
            let encoded = EncoderConfig::xyb(90, sub)
                .encode_bytes(&src, w as u32, h as u32, PixelLayout::Rgb8Srgb)
                .unwrap_or_else(|e| panic!("{w}x{h} {mode} encode failed: {e}"));
            let decoded = Decoder::new()
                .output_format(PixelFormat::Rgb)
                .decode(&encoded, enough::Unstoppable)
                .unwrap_or_else(|e| panic!("{w}x{h} {mode} decode failed: {e}"));
            let pixels = decoded.pixels_u8().expect("u8 pixels");

            let bands = band_errors(&src, pixels, w, h);
            let interior: f64 =
                bands[..bands.len() - 1].iter().sum::<f64>() / (bands.len() - 1) as f64;
            let last = *bands.last().unwrap();
            let ratio = last / interior.max(0.01);
            assert!(
                ratio <= 1.05,
                "{w}x{h} XYB {mode}: bottom-band error ratio {ratio:.3} \
                 (last={last:.2}, interior={interior:.2}) — bottom-strip \
                 padding regression (issue #186)"
            );
        }
    }
}
