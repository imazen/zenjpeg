//! Diagnostic probe for issues #194/#195 root-cause verification.
//!
//! 1. Prints decoder block-grid dims for aligned and non-aligned images.
//! 2. For failing #194 cells, checks whether the encode-order symbol stream
//!    contains symbols absent from the optimized Huffman tables (zero-length
//!    codes -> silent bitstream corruption).

use enough::Unstoppable;
use zenjpeg::decode::DecodeConfig;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::lossless::{transform, EdgeHandling, LosslessTransform, TransformConfig};

fn gen_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            rgb[i] = (x % 256) as u8;
            rgb[i + 1] = (y % 256) as u8;
            rgb[i + 2] = ((x ^ y) % 256) as u8;
        }
    }
    rgb
}

fn encode(w: u32, h: u32, ss: ChromaSubsampling) -> Vec<u8> {
    let rgb = gen_rgb(w, h);
    let mut enc = EncoderConfig::ycbcr(90.0, ss)
        .progressive(false)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn print_grids(label: &str, jpeg: &[u8]) {
    let coeffs = DecodeConfig::new()
        .decode_coefficients(jpeg, Unstoppable)
        .unwrap();
    print!("{label}: {}x{}", coeffs.width, coeffs.height);
    for c in &coeffs.components {
        print!(
            "  [id{} {}x{} samp {}x{}]",
            c.id, c.blocks_wide, c.blocks_high, c.h_samp, c.v_samp
        );
    }
    println!();
}

fn main() {
    println!("=== Grid padding conventions ===");
    for (w, h, ss, name) in [
        (640u32, 480u32, ChromaSubsampling::Quarter, "640x480 4:2:0"),
        (2000, 1333, ChromaSubsampling::Quarter, "2000x1333 4:2:0"),
        (66, 50, ChromaSubsampling::Quarter, "66x50 4:2:0"),
        (66, 50, ChromaSubsampling::HalfHorizontal, "66x50 4:2:2"),
        (66, 50, ChromaSubsampling::None, "66x50 4:4:4"),
    ] {
        print_grids(name, &encode(w, h, ss));
    }

    println!();
    println!("=== Transformed-output stream self-consistency (decode our own output) ===");
    // For each #194 failing cell: transform, then strict-decode our own output and
    // report warnings/errors; also compare coefficient totals.
    for (ss, ss_name) in [
        (ChromaSubsampling::HalfHorizontal, "4:2:2"),
        (ChromaSubsampling::Quarter, "4:2:0"),
    ] {
        let src = encode(640, 480, ss);
        for t in [LosslessTransform::Transpose, LosslessTransform::Transverse] {
            let out = transform(
                &src,
                &TransformConfig {
                    transform: t,
                    edge_handling: EdgeHandling::RejectPartialBlocks,
                },
                Unstoppable,
            )
            .unwrap();
            match DecodeConfig::new().decode_coefficients(&out, Unstoppable) {
                Ok(c) => {
                    // Compare against source coefficient energy as a sanity signal
                    let sum_abs: i64 = c
                        .components
                        .iter()
                        .flat_map(|comp| comp.coeffs.iter())
                        .map(|&v| i64::from(v).abs())
                        .sum();
                    let src_c = DecodeConfig::new()
                        .decode_coefficients(&src, Unstoppable)
                        .unwrap();
                    let src_sum: i64 = src_c
                        .components
                        .iter()
                        .flat_map(|comp| comp.coeffs.iter())
                        .map(|&v| i64::from(v).abs())
                        .sum();
                    println!(
                        "{ss_name} {t:?}: our decode Ok, |coeff| sum {sum_abs} vs src {src_sum} ({})",
                        if sum_abs == src_sum { "EQUAL" } else { "DIFFERS" }
                    );
                }
                Err(e) => println!("{ss_name} {t:?}: our own decoder REJECTS output: {e}"),
            }
        }
    }
}
