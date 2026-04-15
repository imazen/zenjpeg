//! Diagnostic: compare XYB decode with and without `correct_color(Srgb)`.
//! Confirms that calling `correct_color(Srgb)` on an XYB JPEG corrupts the
//! output (because moxcms applies the XYB ICC to already-converted sRGB).

use enough::Unstoppable;
use zenjpeg::color::icc::TargetColorSpace;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{PixelLayout, XybSubsampling};

fn main() {
    let w = 128u32;
    let h = 128u32;
    let rgb: Vec<u8> = (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let top = y < h / 2;
                let left = x < w / 2;
                match (top, left) {
                    (true, true) => [220u8, 40, 40],
                    (true, false) => [40u8, 220, 40],
                    (false, true) => [40u8, 40, 220],
                    (false, false) => [220u8, 220, 40],
                }
            })
        })
        .collect();

    let cfg = EncoderConfig::xyb(85.0, XybSubsampling::Full).progressive(false);
    let jpeg = cfg
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode");

    let probe = |pixels: &[u8], x: u32, y: u32| -> (u8, u8, u8) {
        let i = (y as usize * w as usize + x as usize) * 3;
        (pixels[i], pixels[i + 1], pixels[i + 2])
    };
    let probes = |pixels: &[u8], label: &str| {
        let q = w / 4;
        eprintln!(
            "{label}:\n  TL={:?}\n  TR={:?}\n  BL={:?}\n  BR={:?}",
            probe(pixels, q, q),
            probe(pixels, 3 * q, q),
            probe(pixels, q, 3 * q),
            probe(pixels, 3 * q, 3 * q),
        );
    };

    let no_cc = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("decode no_cc");
    probes(no_cc.pixels_u8().unwrap(), "default (no correct_color)");

    let with_srgb = Decoder::new()
        .correct_color(Some(TargetColorSpace::Srgb))
        .decode(&jpeg, Unstoppable)
        .expect("decode with_srgb");
    probes(with_srgb.pixels_u8().unwrap(), "correct_color(Some(Srgb))");
}
