//! AQ direction study: does zeroing high-AC in FLAT blocks or in BUSY
//! blocks give the better size/quality trade?
//!
//! Perceptual-coding theory (HVS contrast masking, exploited by
//! jpegli/mozjpeg AQ) says BUSY blocks tolerate more distortion — so
//! zeroing their high-freq tail should save more bytes per unit
//! quality loss. The current production AQ targets FLAT blocks. This
//! experiment settles which the zensim metric actually rewards.
//!
//! Variants per cell (all at the SAME uniform quant scale, so only the
//! AQ block-selection differs):
//!   - none:        no AQ (control)
//!   - flat_t48:    zero AC 48..64 in flat blocks (ratio <= 0.08)
//!   - busy_t48:    zero AC 48..64 in busy blocks (ratio > 0.25)
//!   - busy_t32:    zero AC 32..64 in busy blocks (more aggressive)
//!
//! Output TSV: ref source_q target_q variant output_len size_ratio zensim_a

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use enough::Unstoppable;
use zenjpeg::decode::DecodedCoefficients;
use zenjpeg::decoder::{DecodeConfig, Subsampling as DecodeSubsampling};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

use zenjpeg::recompress::expert::{
    AqMask, EmitConfig, QuantScale, build_aq_mask, build_aq_mask_busy, classify_block,
    emit_preserved, score_against_reference,
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

/// Flat-targeting at a fixed tail (zero AC tail..64 in flat blocks,
/// ratio <= 0.08). Built manually so the band matches the busy variant
/// for a fair comparison.
fn flat_mask(coeffs: &DecodedCoefficients, tail_from: usize) -> Option<AqMask> {
    use zenjpeg::recompress::expert::ActivityTier;
    let luma = coeffs.components.first()?;
    let mut mask: AqMask = Vec::with_capacity(luma.num_blocks());
    for b in 0..luma.num_blocks() {
        let block: &[i16; 64] = luma.block(b).try_into().unwrap();
        let is_flat = matches!(
            classify_block(block),
            ActivityTier::VeryFlat | ActivityTier::Flat
        );
        let mut m = 0u64;
        if is_flat {
            for i in tail_from..64 {
                m |= 1u64 << i;
            }
        }
        mask.push(m);
    }
    Some(mask)
}

fn emit(coeffs: &DecodedCoefficients, scale: f32, mask: Option<AqMask>) -> Vec<u8> {
    let cfg = EmitConfig::uniform_scale(QuantScale {
        luma: scale,
        chroma: scale,
    })
    .with_aq_mask(mask);
    emit_preserved(coeffs, DecodeSubsampling::S420, &cfg).unwrap()
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

    println!("ref\tsource_q\ttarget_q\tvariant\toutput_len\tsize_ratio\tzensim_a");

    let _ = build_aq_mask; // reference to production mask for parity if needed

    for ref_path in &refs {
        let (w, h, rgb) = read_png_rgb8(ref_path);
        let ref_name = ref_path.file_stem().unwrap().to_str().unwrap();
        for &src_q in &source_qs {
            let source = encode_source(&rgb, w, h, src_q);
            let coeffs = DecodeConfig::new()
                .decode_coefficients(&source, Unstoppable)
                .unwrap();
            for &target_q in &target_qs {
                let scale = approx_scale(src_q, target_q);
                let variants: [(&str, Option<AqMask>); 4] = [
                    ("none", None),
                    ("flat_t48", flat_mask(&coeffs, 48)),
                    ("busy_t48", build_aq_mask_busy(&coeffs, 48)),
                    ("busy_t32", build_aq_mask_busy(&coeffs, 32)),
                ];
                for (variant, mask) in variants {
                    let out = emit(&coeffs, scale, mask);
                    let score = score_against_reference(&rgb, w, h, &out).unwrap();
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}",
                        ref_name,
                        src_q,
                        target_q,
                        variant,
                        out.len(),
                        out.len() as f32 / source.len() as f32,
                        score,
                    );
                }
            }
        }
    }
}
