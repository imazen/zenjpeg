//! Head-to-head: at the same source + target, does Preserve deliver
//! higher zensim-A vs the reference than Tuned (avoiding an IDCT/FDCT
//! round-trip)? Output is a TSV that surfaces the gen-loss delta.

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::PathBuf;

use enough::Unstoppable;
use zenjpeg::decoder::{DecodeConfig, OutputTarget, Subsampling as DecodeSubsampling};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

use zenjpeg::recompress::expert::{
    EmitConfig, QuantScale, QuantStrategy, analyze_source, build_aq_mask, emit_preserved,
    score_against_reference,
};

fn read_png_rgb8(path: &PathBuf) -> (u32, u32, Vec<u8>) {
    let f = File::open(path).unwrap();
    let decoder = png::Decoder::new(BufReader::new(f));
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info().clone();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let frame = reader.next_frame(&mut buf).unwrap();
    buf.truncate(frame.buffer_size());
    let w = info.width as usize;
    let h = info.height as usize;
    let mut rgb = Vec::with_capacity(w * h * 3);
    match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => rgb = buf,
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            for px in buf.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        _ => panic!("unsupported PNG format"),
    }
    (info.width, info.height, rgb)
}

fn encode_source(rgb: &[u8], w: u32, h: u32, q: f32) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(Quality::ApproxJpegli(q), ChromaSubsampling::Quarter)
        .progressive(true);
    let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    enc.push_packed(rgb, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn tuned_recompress(source: &[u8], target_ijg_q: u8) -> Vec<u8> {
    let decode_cfg = DecodeConfig::new().output_target(OutputTarget::Srgb8);
    let decoded = decode_cfg.decode(source, Unstoppable).unwrap();
    let pixels = decoded.pixels_u8().unwrap();
    let cfg = EncoderConfig::ycbcr(target_ijg_q, ChromaSubsampling::Quarter)
        .progressive(true)
        .auto_optimize(true);
    let mut enc = cfg
        .encode_from_bytes(decoded.width, decoded.height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn preserve_recompress(source: &[u8], strategy: QuantStrategy, with_aq: bool) -> Vec<u8> {
    let coeffs = DecodeConfig::new()
        .decode_coefficients(source, Unstoppable)
        .unwrap();
    let aq_mask = if with_aq {
        build_aq_mask(&coeffs)
    } else {
        None
    };
    let cfg = match strategy {
        QuantStrategy::UniformScale(s) => EmitConfig::uniform_scale(s).with_aq_mask(aq_mask),
        QuantStrategy::TargetQuality { target_ijg_q } => {
            EmitConfig::target_quality(target_ijg_q).with_aq_mask(aq_mask)
        }
        QuantStrategy::RobidouxTargetQuality { target_quality } => {
            EmitConfig::robidoux_target_quality(target_quality).with_aq_mask(aq_mask)
        }
    };
    emit_preserved(&coeffs, DecodeSubsampling::S420, &cfg).unwrap()
}

/// Convert target-IJG-Q (the value zenjpeg's encoder takes) into the
/// equivalent uniform quant-table scale-from-source. For a zenjpeg
/// source at jpegli quality 90 (BA distance ~0.5), scaling by (95/Q)
/// roughly tracks the JPEG quality-vs-quant curve.
fn approx_scale(source_q: f32, target_ijg_q: u8) -> f32 {
    let tq = target_ijg_q as f32;
    let sq = source_q;
    let f_source = if sq < 50.0 {
        5000.0 / sq
    } else {
        200.0 - 2.0 * sq
    };
    let f_target = if tq < 50.0 {
        5000.0 / tq
    } else {
        200.0 - 2.0 * tq
    };
    (f_target / f_source).max(1.0)
}

fn main() {
    let ref_dir = PathBuf::from("/tmp/zjr-real-refs-big");
    let mut refs: Vec<PathBuf> = std::fs::read_dir(&ref_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()?.to_str()? == "png").then_some(p)
        })
        .collect();
    refs.sort();
    refs.truncate(8); // keep runtime manageable

    let source_qs = [90.0f32, 75.0, 60.0];
    let target_qs = [70u8, 60, 50];

    println!("ref\tsource_q\ttarget_q\tstrategy\toutput_len\tsize_ratio\tzensim_a_vs_ref");

    for ref_path in &refs {
        let (w, h, rgb) = read_png_rgb8(ref_path);
        let ref_basename = ref_path.file_stem().unwrap().to_str().unwrap();
        for &src_q in &source_qs {
            let source = encode_source(&rgb, w, h, src_q);
            let _src_analysis = analyze_source(&source).unwrap();
            for &target_q in &target_qs {
                let scale = approx_scale(src_q, target_q);
                let uniform = QuantStrategy::UniformScale(QuantScale {
                    luma: scale,
                    chroma: scale,
                });
                let target_q_strat = QuantStrategy::TargetQuality {
                    target_ijg_q: target_q,
                };

                // Tuned
                let tuned_out = tuned_recompress(&source, target_q);
                let tuned_score = score_against_reference(&rgb, w, h, &tuned_out).unwrap();
                let row = |label: &str, out: &[u8], score: f32| {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}",
                        ref_basename,
                        src_q,
                        target_q,
                        label,
                        out.len(),
                        out.len() as f32 / source.len() as f32,
                        score,
                    );
                };
                row("tuned", &tuned_out, tuned_score);

                // Preserve variants: uniform-scale vs target-quality, with/without AQ.
                for (label, strat) in [
                    ("preserve_uniform_noaq", uniform),
                    ("preserve_target_noaq", target_q_strat),
                ] {
                    let o = preserve_recompress(&source, strat, false);
                    let s = score_against_reference(&rgb, w, h, &o).unwrap();
                    row(label, &o, s);
                }
                for (label, strat) in [
                    ("preserve_uniform_aq", uniform),
                    ("preserve_target_aq", target_q_strat),
                ] {
                    let o = preserve_recompress(&source, strat, true);
                    let s = score_against_reference(&rgb, w, h, &o).unwrap();
                    row(label, &o, s);
                }
            }
        }
    }
}
