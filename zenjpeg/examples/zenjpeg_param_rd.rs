//! zenjpeg encoder param RD ablation: which encoding param set gives the
//! best rate-distortion for the re-encode (Tuned/Deblock) strategies?
//!
//! "auto_optimize is not it" — this finds what is. Encodes pristine CID22
//! originals with each candidate param set across a quality range, and
//! emits (param_set, quality, size_ratio_vs_q90, zensim_vs_original).
//! The Pareto-best param set (highest zensim at equal size, or smallest
//! size at equal zensim) is the one the strategies should use.
//!
//! Output is scored externally by zen-metrics for a fair single decode;
//! this harness writes the encoded JPEGs + a manifest.
//!
//! Usage: zenjpeg_param_rd <ref_dir> <out_dir> <manifest.tsv>

use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use enough::Unstoppable;
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, ProgressiveScanMode,
    Quality, XybSubsampling,
};

fn read_png_rgb8(path: &Path) -> (u32, u32, Vec<u8>) {
    let f = File::open(path).unwrap();
    let dec = png::Decoder::new(BufReader::new(f));
    let mut r = dec.read_info().unwrap();
    let info = r.info().clone();
    let mut buf = vec![0u8; r.output_buffer_size().unwrap_or(0)];
    let frame = r.next_frame(&mut buf).unwrap();
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

/// Build an encoder config for the named param set at quality `q`.
fn encode(rgb: &[u8], w: u32, h: u32, set: &str, q: i32) -> Option<Vec<u8>> {
    let cfg = match set {
        // baseline reference points
        "jpegli_prog" => EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliProgressive),
        "auto_optimize" => EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .progressive(true)
            .auto_optimize(true),
        "hybrid_prog" => EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridProgressive),
        "hybrid_max" => EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridMaxCompression),
        "mozjpeg_max" => EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::MozjpegMaxCompression),
        "prog_search" => EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .progressive(ProgressiveScanMode::ProgressiveSearch)
            .optimize_huffman(true),
        // XYB perceptual color space (uses butteraugli-distance quality semantics)
        "xyb" => EncoderConfig::xyb(
            Quality::ApproxButteraugli(ijg_to_distance(q)),
            XybSubsampling::BQuarter,
        ),
        _ => return None,
    };
    let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).ok()?;
    enc.push_packed(rgb, Unstoppable).ok()?;
    enc.finish().ok()
}

/// Rough IJG-q → butteraugli distance for the XYB path.
fn ijg_to_distance(q: i32) -> f32 {
    let q = q.clamp(1, 100) as f32;
    // jpegli-ish: q90≈1.0, q70≈2.3, q50≈3.5
    (1.0 + (90.0 - q) * 0.06).max(0.3)
}

const PARAM_SETS: &[&str] = &[
    "jpegli_prog",
    "auto_optimize",
    "hybrid_prog",
    "hybrid_max",
    "mozjpeg_max",
    "prog_search",
    "xyb",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
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

    let qualities = [50i32, 60, 70, 80, 90];

    let mut manifest = File::create(&manifest_path).unwrap();
    writeln!(
        manifest,
        "variant_path\tref_path\tref\tparam_set\tquality\toutput_len"
    )
    .unwrap();

    for ref_path in &refs {
        let (w, h, rgb) = read_png_rgb8(ref_path);
        let stem = ref_path.file_stem().unwrap().to_str().unwrap();
        for set in PARAM_SETS {
            for &q in &qualities {
                let Some(bytes) = encode(&rgb, w, h, set, q) else {
                    eprintln!("skip {stem} {set} q{q} (encode failed)");
                    continue;
                };
                let fname = format!("{stem}__{set}__q{q}.jpg");
                let vpath = out_dir.join(&fname);
                fs::write(&vpath, &bytes).unwrap();
                writeln!(
                    manifest,
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    vpath.display(),
                    ref_path.display(),
                    stem,
                    set,
                    q,
                    bytes.len()
                )
                .unwrap();
            }
        }
    }
    eprintln!("zenjpeg_param_rd: manifest at {}", manifest_path.display());
}
