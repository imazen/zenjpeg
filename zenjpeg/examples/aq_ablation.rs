//! AQ ablation: for a grid of (source_q, target_q) cells, emit Preserve
//! WITH and WITHOUT the AQ mask and measure both the size ratio and the
//! cumulative zensim-A vs the reference. The question we answer:
//!
//!   "When does zeroing high-AC in flat blocks (AQ) help — smaller file
//!    at equal-or-better quality — and when does it hurt?"
//!
//! Output TSV columns:
//!   ref  source_q  target_q  variant  output_len  size_ratio  zensim_a
//!   tier_veryflat  tier_flat  tier_middetail  tier_detailed
//!
//! The tier_* columns are the activity-classifier histogram for the
//! source, so we can correlate AQ benefit with flat-region fraction.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use enough::Unstoppable;
use zenjpeg::decoder::{DecodeConfig, Subsampling as DecodeSubsampling};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

use zenjpeg::recompress::expert::{
    EmitConfig, QuantScale, QuantStrategy, build_aq_mask, emit_preserved, score_against_reference,
    tier_histogram,
};

fn read_png_rgb8(path: &PathBuf) -> (u32, u32, Vec<u8>) {
    let f = File::open(path).unwrap();
    let decoder = png::Decoder::new(BufReader::new(f));
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info().clone();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let frame = reader.next_frame(&mut buf).unwrap();
    buf.truncate(frame.buffer_size());
    let mut rgb = Vec::new();
    match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => rgb = buf,
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            for px in buf.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        _ => panic!("unsupported PNG"),
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
    refs.truncate(10);

    let source_qs = [90.0f32, 75.0, 60.0];
    let target_qs = [70u8, 60, 50];

    println!(
        "ref\tsource_q\ttarget_q\tvariant\toutput_len\tsize_ratio\tzensim_a\t\
         tier_veryflat\ttier_flat\ttier_middetail\ttier_detailed"
    );

    for ref_path in &refs {
        let (w, h, rgb) = read_png_rgb8(ref_path);
        let ref_name = ref_path.file_stem().unwrap().to_str().unwrap();
        for &src_q in &source_qs {
            let source = encode_source(&rgb, w, h, src_q);
            let coeffs = DecodeConfig::new()
                .decode_coefficients(&source, Unstoppable)
                .unwrap();
            let hist = tier_histogram(&coeffs);
            let aq_mask = build_aq_mask(&coeffs);

            for &target_q in &target_qs {
                let scale = approx_scale(src_q, target_q);
                let strat = QuantStrategy::UniformScale(QuantScale {
                    luma: scale,
                    chroma: scale,
                });

                for (variant, mask) in [("noaq", None), ("aq", aq_mask.clone())] {
                    let QuantStrategy::UniformScale(s) = strat else {
                        unreachable!()
                    };
                    let cfg = EmitConfig::uniform_scale(s).with_aq_mask(mask);
                    let out = emit_preserved(&coeffs, DecodeSubsampling::S420, &cfg).unwrap();
                    let score = score_against_reference(&rgb, w, h, &out).unwrap();
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{}\t{}\t{}\t{}",
                        ref_name,
                        src_q,
                        target_q,
                        variant,
                        out.len(),
                        out.len() as f32 / source.len() as f32,
                        score,
                        hist[0],
                        hist[1],
                        hist[2],
                        hist[3],
                    );
                }
            }
        }
    }
}
