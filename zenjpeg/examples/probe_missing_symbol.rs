//! Debug-build probe for the latent missing-Huffman-symbol bug caught by the
//! `HuffmanEncodeTable::encode` debug_assert (Coverage CI, locked_values
//! matrix on frymire). Runs the same encoder config matrix; in a debug build
//! any zero-length-code emission panics with the symbol id.
//!
//! Run: cargo run --profile dev --example probe_missing_symbol

use std::panic;

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

fn load_frymire() -> (Vec<u8>, u32, u32) {
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire.png");
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(png_path).expect("open frymire.png"),
    ));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png too large")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width, info.height);
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        other => panic!("unhandled png color type {other:?}"),
    };
    (rgb, w, h)
}

fn main() {
    let (pixels, width, height) = load_frymire();
    println!("frymire {width}x{height}");

    let modes = ["baseline", "progressive"];
    let subsamplings = ["444", "422", "420", "440", "xyb"];
    let huffmans = ["std", "opt"];
    let qualities = [10u8, 30, 50, 75, 85, 90, 95];

    let mut failures = 0usize;
    for mode in modes {
        for ss in subsamplings {
            for huff in huffmans {
                for q in qualities {
                    let pixels = pixels.clone();
                    let result = panic::catch_unwind(move || {
                        let subsamp = match ss {
                            "444" | "xyb" => ChromaSubsampling::None,
                            "422" => ChromaSubsampling::HalfHorizontal,
                            "420" => ChromaSubsampling::Quarter,
                            "440" => ChromaSubsampling::HalfVertical,
                            _ => unreachable!(),
                        };
                        let progressive = mode == "progressive";
                        let optimize = huff == "opt";
                        let config = if ss == "xyb" {
                            EncoderConfig::xyb(q as f32, XybSubsampling::BQuarter)
                                .progressive(progressive)
                                .optimize_huffman(optimize)
                                .restart_mcu_rows(0)
                        } else {
                            EncoderConfig::ycbcr(q as f32, subsamp)
                                .progressive(progressive)
                                .optimize_huffman(optimize)
                                .restart_mcu_rows(0)
                        };
                        let mut enc = config
                            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                            .expect("setup");
                        enc.push_packed(&pixels, Unstoppable).expect("push");
                        enc.finish().expect("finish").len()
                    });
                    match result {
                        Ok(len) => println!("{mode}/{ss}/{huff}/q{q}: ok ({len} bytes)"),
                        Err(_) => {
                            println!("{mode}/{ss}/{huff}/q{q}: PANIC (see stderr above)");
                            failures += 1;
                        }
                    }
                }
            }
        }
    }
    println!("panics: {failures}");
}
