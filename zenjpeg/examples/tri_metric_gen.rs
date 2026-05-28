//! Tri-metric cross-check generator.
//!
//! Emits the recompressed-JPEG variants for two experiments into an
//! output dir, plus a manifest TSV. A driver script then scores every
//! variant with `zen-metrics` under butteraugli / cvvdp / zensim (one
//! shared decode per metric), and joins on the variant path. This keeps
//! the perceptual scoring outside the harness so all three metrics see
//! identical decodes.
//!
//! Experiments:
//!   - `aqdir`: AQ block-selection direction. All at the same uniform
//!     quant scale, so only the AQ mask differs:
//!       none / flat_t48 / busy_t48 / busy_t32
//!   - `pvt`: Preserve vs Tuned (generation-loss). At each target:
//!       tuned / preserve_uniform / preserve_target
//!
//! Usage: tri_metric_gen <ref_dir> <out_dir> <manifest.tsv>
//! Manifest columns:
//!   variant_path  ref_path  ref  source_q  target_q  experiment  variant  output_len  size_ratio

use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use enough::Unstoppable;
use zenjpeg::decode::DecodedCoefficients;
use zenjpeg::decoder::{DecodeConfig, OutputTarget, Subsampling as DecodeSubsampling};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

use zenjpeg::recompress::expert::{
    ActivityTier, AqMask, EmitConfig, QuantScale, build_aq_mask_busy, classify_block,
    emit_preserved,
};

fn read_png_rgb8(path: &Path) -> (u32, u32, Vec<u8>) {
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

fn tuned_recompress(source: &[u8], target_ijg_q: u8) -> Vec<u8> {
    let decoded = DecodeConfig::new()
        .output_target(OutputTarget::Srgb8)
        .decode(source, Unstoppable)
        .unwrap();
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

fn flat_mask(coeffs: &DecodedCoefficients, tail_from: usize) -> Option<AqMask> {
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

fn preserve_uniform(coeffs: &DecodedCoefficients, scale: f32, mask: Option<AqMask>) -> Vec<u8> {
    let cfg = EmitConfig::uniform_scale(QuantScale {
        luma: scale,
        chroma: scale,
    })
    .with_aq_mask(mask);
    emit_preserved(coeffs, DecodeSubsampling::S420, &cfg).unwrap()
}

fn preserve_target(coeffs: &DecodedCoefficients, target_ijg_q: u8) -> Vec<u8> {
    let cfg = EmitConfig::target_quality(target_ijg_q);
    emit_preserved(coeffs, DecodeSubsampling::S420, &cfg).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: tri_metric_gen <ref_dir> <out_dir> <manifest.tsv>");
        std::process::exit(1);
    }
    let ref_dir = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let manifest_path = PathBuf::from(&args[3]);
    fs::create_dir_all(&out_dir).unwrap();

    let mut refs: Vec<PathBuf> = fs::read_dir(&ref_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()?.to_str()? == "png").then_some(p)
        })
        .collect();
    refs.sort();
    refs.truncate(6);

    let source_qs = [90.0f32, 75.0, 60.0];
    let target_qs = [70u8, 60, 50];

    let mut manifest = File::create(&manifest_path).unwrap();
    writeln!(
        manifest,
        "variant_path\tref_path\tref\tsource_q\ttarget_q\texperiment\tvariant\toutput_len\tsize_ratio"
    )
    .unwrap();

    let mut emit = |manifest: &mut File,
                    ref_path: &Path,
                    ref_name: &str,
                    src_q: f32,
                    tgt_q: u8,
                    experiment: &str,
                    variant: &str,
                    bytes: &[u8],
                    source_len: usize| {
        let fname = format!(
            "{ref_name}__src{}__tgt{tgt_q}__{experiment}__{variant}.jpg",
            src_q as u32
        );
        let vpath = out_dir.join(&fname);
        fs::write(&vpath, bytes).unwrap();
        writeln!(
            manifest,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}",
            vpath.display(),
            ref_path.display(),
            ref_name,
            src_q,
            tgt_q,
            experiment,
            variant,
            bytes.len(),
            bytes.len() as f32 / source_len as f32,
        )
        .unwrap();
    };

    for ref_path in &refs {
        let (w, h, rgb) = read_png_rgb8(ref_path);
        let ref_name = ref_path.file_stem().unwrap().to_str().unwrap();
        for &src_q in &source_qs {
            let source = encode_source(&rgb, w, h, src_q);
            let coeffs = DecodeConfig::new()
                .decode_coefficients(&source, Unstoppable)
                .unwrap();
            let slen = source.len();
            for &tgt_q in &target_qs {
                let scale = approx_scale(src_q, tgt_q);

                // --- aqdir experiment ---
                let aq_variants: [(&str, Option<AqMask>); 4] = [
                    ("none", None),
                    ("flat_t48", flat_mask(&coeffs, 48)),
                    ("busy_t48", build_aq_mask_busy(&coeffs, 48)),
                    ("busy_t32", build_aq_mask_busy(&coeffs, 32)),
                ];
                for (variant, mask) in aq_variants {
                    let out = preserve_uniform(&coeffs, scale, mask);
                    emit(
                        &mut manifest,
                        ref_path,
                        ref_name,
                        src_q,
                        tgt_q,
                        "aqdir",
                        variant,
                        &out,
                        slen,
                    );
                }

                // --- pvt experiment ---
                let tuned = tuned_recompress(&source, tgt_q);
                emit(
                    &mut manifest,
                    ref_path,
                    ref_name,
                    src_q,
                    tgt_q,
                    "pvt",
                    "tuned",
                    &tuned,
                    slen,
                );
                let pu = preserve_uniform(&coeffs, scale, None);
                emit(
                    &mut manifest,
                    ref_path,
                    ref_name,
                    src_q,
                    tgt_q,
                    "pvt",
                    "preserve_uniform",
                    &pu,
                    slen,
                );
                let pt = preserve_target(&coeffs, tgt_q);
                emit(
                    &mut manifest,
                    ref_path,
                    ref_name,
                    src_q,
                    tgt_q,
                    "pvt",
                    "preserve_target",
                    &pt,
                    slen,
                );
            }
        }
    }
    eprintln!("tri_metric_gen: manifest at {}", manifest_path.display());
}
