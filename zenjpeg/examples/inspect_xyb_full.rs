//! Hex-dump SOF marker + roundtrip a tiny XYB Full encode to verify pixel
//! correctness. Used to triage whether `XybSubsampling::Full` is currently
//! producing a valid bitstream.

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{PixelLayout, XybSubsampling};

fn main() {
    let w: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let h = w;
    // Patchy non-uniform image: 4 colored quadrants. If encoder/decoder
    // disagree on MCU geometry, colors will smear across quadrant edges.
    let rgb: Vec<u8> = (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                let top = y < h / 2;
                let left = x < w / 2;
                match (top, left) {
                    (true, true) => [220u8, 40, 40],    // TL red
                    (true, false) => [40u8, 220, 40],   // TR green
                    (false, true) => [40u8, 40, 220],   // BL blue
                    (false, false) => [220u8, 220, 40], // BR yellow
                }
            })
        })
        .collect();

    let progressive: bool = std::env::args()
        .nth(2)
        .map(|s| s == "p" || s == "progressive")
        .unwrap_or(false);
    eprintln!("progressive={progressive}");

    for sub in [XybSubsampling::BQuarter, XybSubsampling::Full] {
        let cfg = EncoderConfig::xyb(85.0, sub).progressive(progressive);
        let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        e.push_packed(&rgb, Unstoppable).unwrap();
        let jpeg = e.finish().unwrap();

        // Find SOF marker (0xFF 0xC0 / 0xC1 / 0xC2)
        let mut sof = None;
        for i in 0..jpeg.len() - 1 {
            if jpeg[i] == 0xFF
                && (jpeg[i + 1] == 0xC0 || jpeg[i + 1] == 0xC1 || jpeg[i + 1] == 0xC2)
            {
                sof = Some(i);
                break;
            }
        }
        let sof_idx = sof.expect("no SOF found");
        let sof_marker = jpeg[sof_idx + 1];
        let length = u16::from_be_bytes([jpeg[sof_idx + 2], jpeg[sof_idx + 3]]);
        let nf = jpeg[sof_idx + 9];
        eprintln!(
            "\n=== {:?}: SOF{}={:02X}, length={}, nf={} ===",
            sub,
            sof_marker - 0xC0,
            sof_marker,
            length,
            nf
        );
        for c in 0..(nf as usize) {
            let off = sof_idx + 10 + c * 3;
            let id = jpeg[off];
            let samp = jpeg[off + 1];
            let qt = jpeg[off + 2];
            let id_char = if (b'A'..=b'Z').contains(&id) {
                id as char
            } else {
                '?'
            };
            eprintln!(
                "  comp[{c}] id={id} ({id_char}) samp={samp:02X} (h={} v={}) qt={qt}",
                samp >> 4,
                samp & 0xF
            );
        }
        eprintln!("  bytes={}", jpeg.len());

        // Write to disk so djpegli can decode it independently
        let path = format!("/tmp/xyb_{:?}.jpg", sub);
        std::fs::write(&path, &jpeg).unwrap();
        eprintln!("  wrote {path}");

        // Decode with our decoder and probe the 4 quadrant centers
        match Decoder::new().decode(&jpeg, Unstoppable) {
            Ok(decoded) => {
                let pixels = decoded.pixels_u8().unwrap();
                let probe = |x: u32, y: u32| -> (u8, u8, u8) {
                    let i = (y as usize * w as usize + x as usize) * 3;
                    (pixels[i], pixels[i + 1], pixels[i + 2])
                };
                let q = w / 4; // probe at center of each quadrant
                eprintln!(
                    "  decoded quadrants (probe at quadrant centers, w={w}):\n    TL={:?} expect~(220,40,40)\n    TR={:?} expect~(40,220,40)\n    BL={:?} expect~(40,40,220)\n    BR={:?} expect~(220,220,40)",
                    probe(q, q),
                    probe(3 * q, q),
                    probe(q, 3 * q),
                    probe(3 * q, 3 * q),
                );
            }
            Err(e) => eprintln!("  DECODE FAILED: {e:?}"),
        }
    }
}
