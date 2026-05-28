//! Localize the fine→coarse Preserve crater. Encodes a turbo-style
//! source (zenjpeg ApproxJpegli stands in if turbo unavailable; we pass
//! a real turbo source path as arg[1] when given), decodes coefficients,
//! emits via UniformScale / TargetQuality / identity at a target, and
//! dumps the quant tables + first luma block before/after + decoded
//! pixel diff. Run with a turbo source JPEG to reproduce the crater.
//!
//! Usage: preserve_requant_diag <source.jpg> <original.png> <target_q>

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use enough::Unstoppable;
use zenjpeg::decode::DecodeConfig;
use zenjpeg::decoder::{OutputTarget, Subsampling};
use zenjpeg::recompress::expert::{
    EmitConfig, QuantScale, QuantStrategy, emit_preserved, target_zensim_a_to_ijg_q,
};

fn read_png_rgb8(path: &Path) -> (u32, u32, Vec<u8>) {
    let f = File::open(path).unwrap();
    let dec = png::Decoder::new(BufReader::new(f));
    let mut r = dec.read_info().unwrap();
    let info = r.info().clone();
    let mut buf = vec![0u8; r.output_buffer_size().unwrap_or(0)];
    let fr = r.next_frame(&mut buf).unwrap();
    buf.truncate(fr.buffer_size());
    let mut rgb = Vec::new();
    match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => rgb = buf,
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            for px in buf.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        _ => panic!("png fmt"),
    }
    (info.width, info.height, rgb)
}

fn score(orig: &[u8], jpeg: &[u8]) -> f32 {
    zenjpeg::recompress::expert::score_against_reference(
        orig,
        decode_dims(jpeg).0,
        decode_dims(jpeg).1,
        jpeg,
    )
    .unwrap_or(f32::NAN)
}
fn decode_dims(jpeg: &[u8]) -> (u32, u32) {
    let d = DecodeConfig::new()
        .output_target(OutputTarget::Srgb8)
        .decode(jpeg, Unstoppable)
        .unwrap();
    (d.width, d.height)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src = std::fs::read(&a[1]).unwrap();
    let (w, h, orig) = read_png_rgb8(Path::new(&a[2]));
    let target: f32 = a[3].parse().unwrap();
    let tq = target_zensim_a_to_ijg_q(target);

    let coeffs = DecodeConfig::new()
        .decode_coefficients(&src, Unstoppable)
        .unwrap();
    println!(
        "source: {} bytes, {}x{}, target zensim {} → encoder q {}",
        src.len(),
        w,
        h,
        target,
        tq
    );
    println!(
        "num quant tables: {}",
        coeffs.quant_tables.iter().filter(|t| t.is_some()).count()
    );
    for (i, c) in coeffs.components.iter().enumerate() {
        println!(
            "  comp[{}] id={} blocks={}x{} h_samp={} v_samp={} q_idx={} dc(block0)={}",
            i,
            c.id,
            c.blocks_wide,
            c.blocks_high,
            c.h_samp,
            c.v_samp,
            c.quant_table_idx,
            c.block(0)[0]
        );
    }
    println!("source vs orig zensim: {:.2}", score(&orig, &src));

    for (name, strat) in [
        (
            "identity",
            QuantStrategy::UniformScale(QuantScale::IDENTITY),
        ),
        (
            "uniform",
            QuantStrategy::UniformScale(QuantScale {
                luma: 2.0,
                chroma: 2.0,
            }),
        ),
        (
            "target_q",
            QuantStrategy::TargetQuality { target_ijg_q: tq },
        ),
    ] {
        let cfg = match strat {
            QuantStrategy::UniformScale(s) => EmitConfig::uniform_scale(s),
            QuantStrategy::TargetQuality { target_ijg_q } => {
                EmitConfig::target_quality(target_ijg_q)
            }
            QuantStrategy::RobidouxTargetQuality { target_quality } => {
                EmitConfig::robidoux_target_quality(target_quality)
            }
        };
        match emit_preserved(&coeffs, Subsampling::S420, &cfg) {
            Ok(bytes) => {
                let rq = DecodeConfig::new()
                    .decode_coefficients(&bytes, Unstoppable)
                    .unwrap();
                let s = score(&orig, &bytes);
                println!(
                    "\n[{name}] {} bytes (ratio {:.3}), zensim vs orig {:.2}",
                    bytes.len(),
                    bytes.len() as f32 / src.len() as f32,
                    s
                );
                for (i, c) in rq.components.iter().enumerate() {
                    println!(
                        "    comp[{}] q_idx={} dc(block0)={} dc(block1)={}",
                        i,
                        c.quant_table_idx,
                        c.block(0)[0],
                        if c.num_blocks() > 1 { c.block(1)[0] } else { 0 }
                    );
                }
                std::fs::write(format!("/tmp/requant_{name}.jpg"), &bytes).ok();
            }
            Err(e) => println!("\n[{name}] emit error: {e:?}"),
        }
    }
}
