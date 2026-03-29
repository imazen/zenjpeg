//! zensim regression testing on real photographic corpora: CID22, CLIC, gb82.
//!
//! **No synthetic images, no gradients, no test patterns.** Every image is a
//! real photograph from a quality assessment corpus.
//!
//! Four encoder modes compared (all 4:2:0):
//!   - **mozjpeg**: mozjpeg-rs | ProgressiveSmallest | trellis
//!   - **zen-moz**: zenjpeg | ApproxMozjpeg(Q) + MozjpegProgressive | trellis, no AQ
//!   - **zen-jpegli**: zenjpeg | native Q + JpegliProgressive | AQ, no trellis | jpegli tables
//!   - **zen-auto**: zenjpeg | native Q + auto_optimize | AQ + hybrid trellis | jpegli tables (SOTA)
//!
//! Four tests:
//!
//! 1. **photo_encoder_quality_vs_original** — All 4 encoder modes vs original
//!    - Decoder: zenjpeg | Jpegli IDCT (12-bit) | Triangle upsampling | no ICC
//!    - Metric:  zensim(decoded, original) for each mode at same Q
//!
//! 2. **photo_decoder_parity** — Decoder accuracy on mozjpeg files (PRIMARY)
//!    - Encoder:   mozjpeg-rs | ProgressiveSmallest | 4:2:0
//!    - Decoder A: mozjpeg-sys | libjpeg-turbo FFI | islow IDCT | fancy upsample
//!    - Decoder B: zenjpeg    | Jpegli IDCT (12-bit) | Triangle upsampling
//!    - Decoder C: zenjpeg    | Libjpeg IDCT (13-bit Loeffler) | LibjpegCompat upsample
//!    - Metric:    zensim(B vs A) and zensim(C vs A) + max pixel diff
//!
//! 3. **photo_encoder_cross_comparison** — zen-moz vs mozjpeg decoded similarity
//!    - Encoder A: mozjpeg-rs | ProgressiveSmallest | 4:2:0
//!    - Encoder B: zenjpeg   | ApproxMozjpeg(Q) + MozjpegProgressive | 4:2:0
//!    - Decoder:   zenjpeg   | Jpegli IDCT (12-bit) | Triangle upsampling | no ICC
//!    - Metric:    zensim(decoded-A, decoded-B) + max pixel diff
//!
//! 4. **photo_size_matched_quality** — All modes at matched file size
//!    - Each zenjpeg mode's Q bisected to match mozjpeg's output size ±2%
//!    - Decoder: zenjpeg | Jpegli IDCT (12-bit) | Triangle upsampling | no ICC
//!    - Metric:  zensim(decoded, original) — same bits, who wins?
//!
//! Corpus: CID22-512 (250 photos, 512x512) + CLIC 2025 (62 photos, ~2048px) + gb82 (25 photos)
//!
//! Run:
//! ```bash
//! cargo test --release -p zenjpeg --test photo_corpus_zensim \
//!     --features "trellis decoder" -- --nocapture --ignored
//! ```

use enough::Unstoppable;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use zensim::{RgbSlice, Zensim, ZensimProfile};

use butteraugli::ButteraugliParams;
use zenjpeg::decode::ChromaUpsampling;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, Quality};

// ── Constants ───────────────────────────────────────────────────────────────

const QUALITY_LEVELS: [u8; 6] = [50, 70, 80, 85, 90, 95];

/// Extended quality range for reconstruction test — includes low Q.
const RECON_QUALITY_LEVELS: [u8; 11] = [5, 10, 20, 30, 40, 50, 70, 80, 85, 90, 95];

/// Maximum pixel count — CLIC images are ~2048px, all fit comfortably.
const MAX_PIXELS: u64 = 6_000_000;

// ── Shared infrastructure ───────────────────────────────────────────────────

thread_local! {
    static ZENSIM: Zensim = Zensim::new(ZensimProfile::latest());
}

/// Load corpus, or return empty vec (test will skip).
fn load_corpus_or_skip() -> Vec<LoadedImage> {
    let images = load_all_corpora();
    if images.is_empty() {
        println!("No corpora found, skipping");
    }
    images
}

/// Run a function over (image, quality) pairs in parallel with progress reporting.
fn run_par<T: Send>(
    images: &[LoadedImage],
    quality_levels: &[u8],
    desc: &str,
    f: impl Fn(&LoadedImage, u8) -> T + Sync,
) -> Vec<T> {
    let work: Vec<(usize, u8)> = images
        .iter()
        .enumerate()
        .flat_map(|(i, _)| quality_levels.iter().map(move |&q| (i, q)))
        .collect();
    let total = work.len() as u32;
    let progress = AtomicU32::new(0);
    println!("Running {total} {desc}...\n");

    work.par_iter()
        .map(|&(idx, q)| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 200 == 0 {
                eprintln!("  ... {done}/{total}");
            }
            f(&images[idx], q)
        })
        .collect()
}

// ── Image loading ───────────────────────────────────────────────────────────

struct LoadedImage {
    corpus: &'static str,
    pixels: Vec<u8>, // flat RGB
    width: u32,
    height: u32,
}

fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for chunk in src.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for &g in src {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for chunk in src.chunks_exact(2) {
                rgb.extend_from_slice(&[chunk[0], chunk[0], chunk[0]]);
            }
            rgb
        }
        _ => return None,
    };
    Some((rgb, w, h))
}

/// Collect all PNG files from a directory (non-recursive).
fn collect_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    files
}

/// Load all corpora via `codec_corpus` crate (auto-downloads if not cached).
fn load_all_corpora() -> Vec<LoadedImage> {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("codec-corpus init failed: {e}");
            return Vec::new();
        }
    };

    let mut images = Vec::new();
    let mut load_sub = |subpath: &str, tag: &'static str| {
        let dir = match corpus.get(subpath) {
            Ok(d) => d,
            Err(_) => return,
        };
        for path in collect_pngs(&dir) {
            if let Some((pixels, w, h)) = load_png_rgb(&path) {
                if (w as u64) * (h as u64) <= MAX_PIXELS && pixels.len() == (w * h * 3) as usize {
                    images.push(LoadedImage {
                        corpus: tag,
                        pixels,
                        width: w,
                        height: h,
                    });
                }
            }
        }
    };

    load_sub("CID22/CID22-512/training", "CID22");
    load_sub("CID22/CID22-512/validation", "CID22");
    load_sub("clic2025/final-test", "CLIC");
    load_sub("clic2025/training", "CLIC");
    load_sub("gb82", "gb82");

    images
}

// ── Encoder configs ──────────────────────────────────────────────────────────

/// Named encoder configuration for comparison.
#[derive(Clone)]
struct EncoderMode {
    /// Short label for output tables.
    label: &'static str,
    /// Full description for banner.
    desc: &'static str,
}

const MODE_MOZJPEG: EncoderMode = EncoderMode {
    label: "mozjpeg",
    desc: "mozjpeg-rs | ProgressiveSmallest | trellis | 4:2:0",
};

const MODE_ZEN_MOZ: EncoderMode = EncoderMode {
    label: "zen-moz",
    desc: "zenjpeg | ApproxMozjpeg(Q) + MozjpegProgressive | trellis, no AQ | 4:2:0",
};

const MODE_ZEN_JPEGLI: EncoderMode = EncoderMode {
    label: "zen-jpegli",
    desc: "zenjpeg | native Q + JpegliProgressive | AQ, no trellis | jpegli tables | 4:2:0",
};

const MODE_ZEN_AUTO: EncoderMode = EncoderMode {
    label: "zen-auto",
    desc: "zenjpeg | native Q + auto_optimize | AQ + hybrid trellis | jpegli tables | 4:2:0",
};

/// Encode with **mozjpeg-rs** (C library via Rust wrapper).
/// ProgressiveSmallest, 4:2:0, mozjpeg trellis, mozjpeg quality scale.
fn encode_mozjpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg-rs encode failed")
}

/// Encode with **zenjpeg in mozjpeg-mimicry mode**.
/// ApproxMozjpeg(q) quality + MozjpegProgressive + trellis, no AQ + 4:2:0.
fn encode_zen_moz(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::MozjpegProgressive);
    encode_with_config(pixels, w, h, config)
}

/// Encode with **zenjpeg JpegliProgressive mode**.
/// Native quality + JpegliProgressive + AQ (no trellis) + jpegli quant tables + 4:2:0.
fn encode_zen_jpegli(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::JpegliProgressive);
    encode_with_config(pixels, w, h, config)
}

/// Encode with **zenjpeg auto_optimize mode** (SOTA).
/// Native quality + auto_optimize + AQ + hybrid trellis + jpegli tables + 4:2:0.
fn encode_zen_auto(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).auto_optimize(true);
    encode_with_config(pixels, w, h, config)
}

fn encode_with_config(pixels: &[u8], w: u32, h: u32, config: EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// Decode with **zenjpeg default mode**.
/// IDCT: Jpegli (12-bit fixed-point). Upsampling: Triangle (jpegli-style). No ICC.
fn decode_zen_default(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let dec = Decoder::new().apply_icc(false);
    let img = dec.decode(jpeg, Unstoppable).expect("decode failed");
    (img.width, img.height, img.into_pixels_u8().unwrap())
}

/// Decode with **zenjpeg LibjpegCompat mode** (closest to mozjpeg/libjpeg-turbo).
/// IDCT: Libjpeg (13-bit Loeffler, auto-selected). Upsampling: LibjpegCompat. No ICC.
fn decode_zen_compat(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let dec = Decoder::new()
        .apply_icc(false)
        .chroma_upsampling(ChromaUpsampling::LibjpegCompat);
    let img = dec.decode(jpeg, Unstoppable).expect("decode failed");
    (img.width, img.height, img.into_pixels_u8().unwrap())
}

/// Decode with **mozjpeg-sys** (libjpeg-turbo C library via FFI).
/// IDCT: islow (13-bit Loeffler). Upsampling: fancy. Output: JCS_RGB.
fn decode_mozjpeg_sys(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);
        jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);
        if jpeg_read_header(&mut cinfo, true as boolean) != 1 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("bad header".into());
        }
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);
        let (w, h) = (cinfo.output_width, cinfo.output_height);
        let stride = w as usize * cinfo.output_components as usize;
        let mut output = vec![0u8; h as usize * stride];
        while cinfo.output_scanline < h {
            let offset = cinfo.output_scanline as usize * stride;
            let mut row_ptr = output[offset..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
        Ok((w, h, output))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn as_rgb(rgb: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(rgb)
}

fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn zensim_score(z: &Zensim, a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
    let sa = RgbSlice::new(as_rgb(a), w, h);
    let sb = RgbSlice::new(as_rgb(b), w, h);
    z.compute(&sa, &sb).map(|r| r.score()).unwrap_or(-1.0)
}

/// Butteraugli distance (lower = better, <1.0 = good).
fn butteraugli_score(original: &[u8], decoded: &[u8], w: usize, h: usize) -> f64 {
    let to_img = |data: &[u8]| -> imgref::ImgVec<rgb::RGB8> {
        let pixels: Vec<rgb::RGB8> = data
            .chunks_exact(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect();
        imgref::ImgVec::new(pixels, w, h)
    };
    butteraugli::butteraugli(
        to_img(original).as_ref(),
        to_img(decoded).as_ref(),
        &ButteraugliParams::default(),
    )
    .expect("butteraugli failed")
    .score
}

/// Print corpus composition.
fn print_corpus_info(images: &[LoadedImage]) {
    let cid22 = images.iter().filter(|i| i.corpus == "CID22").count();
    let clic = images.iter().filter(|i| i.corpus == "CLIC").count();
    let gb82 = images.iter().filter(|i| i.corpus == "gb82").count();
    println!(
        "Loaded {} photos: {} CID22 (512x512) + {} CLIC (~2048px) + {} gb82",
        images.len(),
        cid22,
        clic,
        gb82
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: ALL ENCODER MODES vs ORIGINAL (same Q parameter)
// ═══════════════════════════════════════════════════════════════════════════

struct QualityResult {
    quality: u8,
    /// (label, bytes, zensim_vs_orig) for each encoder mode
    scores: Vec<(&'static str, usize, f64)>,
}

fn process_quality(img: &LoadedImage, quality: u8) -> QualityResult {
    let (w, h, px) = (img.width, img.height, &img.pixels);

    // Encode with all 4 configs
    let jpegs: Vec<(&str, Vec<u8>)> = vec![
        (MODE_MOZJPEG.label, encode_mozjpeg(px, w, h, quality)),
        (MODE_ZEN_MOZ.label, encode_zen_moz(px, w, h, quality)),
        (MODE_ZEN_JPEGLI.label, encode_zen_jpegli(px, w, h, quality)),
        (MODE_ZEN_AUTO.label, encode_zen_auto(px, w, h, quality)),
    ];

    // Decode ALL with zenjpeg default (Jpegli IDCT, Triangle upsampling)

    let scores: Vec<_> = jpegs
        .into_iter()
        .map(|(label, jpeg)| {
            let sz = jpeg.len();
            let (_, _, dec) = decode_zen_default(&jpeg);
            let score = ZENSIM.with(|z| zensim_score(z, px, &dec, w as usize, h as usize));
            (label, sz, score)
        })
        .collect();

    QualityResult { quality, scores }
}

#[test]
#[ignore = "requires photo corpora and decoder/trellis features"]
fn photo_encoder_quality_vs_original() {
    let images = load_corpus_or_skip();
    if images.is_empty() {
        return;
    }

    println!("=== All Encoder Modes vs Original (same Q parameter) ===");
    println!("  {}: {}", MODE_MOZJPEG.label, MODE_MOZJPEG.desc);
    println!("  {}: {}", MODE_ZEN_MOZ.label, MODE_ZEN_MOZ.desc);
    println!("  {}: {}", MODE_ZEN_JPEGLI.label, MODE_ZEN_JPEGLI.desc);
    println!("  {}: {}", MODE_ZEN_AUTO.label, MODE_ZEN_AUTO.desc);
    println!("  Decoder: zenjpeg | Jpegli IDCT (12-bit) | Triangle upsampling | no ICC");
    println!("  Metric:  zensim(decoded, original) for each encoder mode");
    println!();
    print_corpus_info(&images);

    let results = run_par(
        &images,
        &QUALITY_LEVELS,
        "images × 4 modes",
        process_quality,
    );

    let labels = [
        MODE_MOZJPEG.label,
        MODE_ZEN_MOZ.label,
        MODE_ZEN_JPEGLI.label,
        MODE_ZEN_AUTO.label,
    ];

    // Per-quality summary: mean zensim and mean kb for each mode
    println!(
        "  {:>3}  {:>10} {:>6}  {:>10} {:>6}  {:>10} {:>6}  {:>10} {:>6}",
        "Q", labels[0], "kb", labels[1], "kb", labels[2], "kb", labels[3], "kb"
    );
    println!("  {}", "-".repeat(92));

    for &q in &QUALITY_LEVELS {
        let qr: Vec<&QualityResult> = results.iter().filter(|r| r.quality == q).collect();
        let n = qr.len() as f64;
        let mut means: Vec<(f64, f64)> = Vec::new(); // (mean_score, mean_kb)
        for (li, _label) in labels.iter().enumerate() {
            let score_sum: f64 = qr.iter().map(|r| r.scores[li].2).sum();
            let kb_sum: f64 = qr.iter().map(|r| r.scores[li].1 as f64 / 1024.0).sum();
            means.push((score_sum / n, kb_sum / n));
        }
        println!(
            "  Q{q:<2}  {:>10.2} {:>5.0}k  {:>10.2} {:>5.0}k  {:>10.2} {:>5.0}k  {:>10.2} {:>5.0}k",
            means[0].0,
            means[0].1,
            means[1].0,
            means[1].1,
            means[2].0,
            means[2].1,
            means[3].0,
            means[3].1,
        );
    }

    // Win counts: how often each mode beats mozjpeg
    println!();
    for (li, label) in labels.iter().enumerate().skip(1) {
        let wins = results
            .iter()
            .filter(|r| r.scores[li].2 - r.scores[0].2 > 0.1)
            .count();
        let losses = results
            .iter()
            .filter(|r| r.scores[li].2 - r.scores[0].2 < -0.1)
            .count();
        let ties = results.len() - wins - losses;
        let mean_d: f64 = results
            .iter()
            .map(|r| r.scores[li].2 - r.scores[0].2)
            .sum::<f64>()
            / results.len() as f64;
        let mean_sz: f64 = results
            .iter()
            .map(|r| r.scores[li].1 as f64 / r.scores[0].1 as f64)
            .sum::<f64>()
            / results.len() as f64;
        println!(
            "  {label} vs mozjpeg: Δ={mean_d:>+.3} zensim, size={mean_sz:.3}x, {wins}w/{ties}t/{losses}l"
        );
    }

    // zen-auto must not catastrophically regress vs mozjpeg
    let auto_mean_delta: f64 = results
        .iter()
        .map(|r| r.scores[3].2 - r.scores[0].2)
        .sum::<f64>()
        / results.len() as f64;
    assert!(
        auto_mean_delta > -2.0,
        "zen-auto mean delta {auto_mean_delta:+.3} vs mozjpeg — regression"
    );
    println!("\nPASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: DECODER PARITY (PRIMARY)
// ═══════════════════════════════════════════════════════════════════════════

struct ParityResult {
    corpus: &'static str,
    quality: u8,
    default_score: f64,
    compat_score: f64,
    default_max_diff: u8,
    compat_max_diff: u8,
}

fn process_parity(img: &LoadedImage, quality: u8) -> ParityResult {
    let (w, h) = (img.width, img.height);
    let moz_jpeg = encode_mozjpeg(&img.pixels, w, h, quality);

    // Three decoders on the same JPEG
    let (_, _, moz_dec) = decode_mozjpeg_sys(&moz_jpeg).expect("mozjpeg decode");
    let (_, _, zen_def) = decode_zen_default(&moz_jpeg);
    let (_, _, zen_compat) = decode_zen_compat(&moz_jpeg);

    let (ds, cs) = ZENSIM.with(|z| {
        (
            zensim_score(z, &moz_dec, &zen_def, w as usize, h as usize),
            zensim_score(z, &moz_dec, &zen_compat, w as usize, h as usize),
        )
    });

    ParityResult {
        corpus: img.corpus,
        quality,
        default_score: ds,
        compat_score: cs,
        default_max_diff: max_pixel_diff(&moz_dec, &zen_def),
        compat_max_diff: max_pixel_diff(&moz_dec, &zen_compat),
    }
}

#[test]
#[ignore = "requires photo corpora and decoder/trellis features"]
fn photo_decoder_parity() {
    let images = load_corpus_or_skip();
    if images.is_empty() {
        return;
    }

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  DECODER PARITY: Can zenjpeg correctly read mozjpeg files?          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("  Encoder:   mozjpeg-rs  | ProgressiveSmallest | 4:2:0");
    println!("  Decoder A: mozjpeg-sys | libjpeg-turbo FFI | islow IDCT | fancy upsample");
    println!("  Decoder B: zenjpeg     | Jpegli IDCT (12-bit) | Triangle upsampling");
    println!("  Decoder C: zenjpeg     | Libjpeg IDCT (13-bit Loeffler) | LibjpegCompat upsample");
    println!("  Metric:    zensim(B vs A) and zensim(C vs A) + max pixel diff");
    println!();
    print_corpus_info(&images);

    let results = run_par(
        &images,
        &QUALITY_LEVELS,
        "decoder parity checks",
        process_parity,
    );

    // Per-quality summary
    println!("  Q   default_zensim  max_diff  compat_zensim  max_diff");
    println!("  {}", "-".repeat(58));
    for &q in &QUALITY_LEVELS {
        let qr: Vec<&ParityResult> = results.iter().filter(|r| r.quality == q).collect();
        let n = qr.len() as f64;
        let def_m = qr.iter().map(|r| r.default_score).sum::<f64>() / n;
        let compat_m = qr.iter().map(|r| r.compat_score).sum::<f64>() / n;
        let def_max = qr.iter().map(|r| r.default_max_diff).max().unwrap_or(0);
        let compat_max = qr.iter().map(|r| r.compat_max_diff).max().unwrap_or(0);
        println!("  Q{q:<2}  {def_m:>13.2}  {def_max:>8}  {compat_m:>13.2}  {compat_max:>8}");
    }

    // Per-corpus
    for corpus in ["CID22", "CLIC", "gb82"] {
        let cr: Vec<&ParityResult> = results.iter().filter(|r| r.corpus == corpus).collect();
        if cr.is_empty() {
            continue;
        }
        let n = cr.len() as f64;
        let def_m = cr.iter().map(|r| r.default_score).sum::<f64>() / n;
        let compat_m = cr.iter().map(|r| r.compat_score).sum::<f64>() / n;
        let compat_max = cr.iter().map(|r| r.compat_max_diff).max().unwrap_or(0);
        println!(
            "  {corpus:<5}: default={def_m:.2}, compat={compat_m:.2}, compat_max_diff={compat_max}"
        );
    }

    let compat_mean = results.iter().map(|r| r.compat_score).sum::<f64>() / results.len() as f64;
    let compat_min = results
        .iter()
        .map(|r| r.compat_score)
        .fold(f64::INFINITY, f64::min);
    let compat_worst_max = results.iter().map(|r| r.compat_max_diff).max().unwrap_or(0);
    let default_mean = results.iter().map(|r| r.default_score).sum::<f64>() / results.len() as f64;
    let default_worst_max = results
        .iter()
        .map(|r| r.default_max_diff)
        .max()
        .unwrap_or(0);

    println!(
        "\nLibjpegCompat: mean={compat_mean:.2}, min={compat_min:.2}, worst max_diff={compat_worst_max}"
    );
    println!("Default Jpegli: mean={default_mean:.2}, worst max_diff={default_worst_max}");

    assert!(
        compat_worst_max <= 3,
        "LibjpegCompat max_diff {compat_worst_max} > 3"
    );
    assert!(
        compat_min > 85.0,
        "LibjpegCompat min zensim {compat_min:.2} < 85"
    );
    assert!(
        default_worst_max <= 5,
        "Default max_diff {default_worst_max} > 5"
    );
    println!("PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2.5: DECODER RECONSTRUCTION QUALITY vs ORIGINAL (all encoders)
// ═══════════════════════════════════════════════════════════════════════════
//
// For each (encoder, quality), encode the source, then decode with all 3
// decoders, and measure zensim of each decoded output vs the original.
// Shows which (encoder, decoder) combination best reconstructs the source.
//
// Encoders:
//   mozjpeg:    mozjpeg-rs | ProgressiveSmallest | 4:2:0
//   zen-jpegli: zenjpeg | native Q + JpegliProgressive | AQ | jpegli tables | 4:2:0
//   zen-auto:   zenjpeg | native Q + auto_optimize | AQ + hybrid trellis | jpegli tables | 4:2:0
// Decoders:
//   mozjpeg-sys: libjpeg-turbo FFI | islow IDCT (13-bit) | fancy upsample
//   zen-default: zenjpeg | Jpegli IDCT (12-bit) | Triangle upsampling
//   zen-compat:  zenjpeg | Libjpeg IDCT (13-bit Loeffler) | LibjpegCompat upsample
// Metric: zensim(decoded, original) for each (encoder × decoder) cell

struct ReconRow {
    quality: u8,
    /// For each encoder: (label, [(decoder_label, zensim_vs_orig)])
    /// 3 encoders × 3 decoders = 9 scores per row
    cells: Vec<(&'static str, Vec<(&'static str, f64)>)>,
}

fn process_recon(img: &LoadedImage, quality: u8) -> ReconRow {
    let (w, h, px) = (img.width, img.height, &img.pixels);

    let encoders: Vec<(&str, Vec<u8>)> = vec![
        (MODE_MOZJPEG.label, encode_mozjpeg(px, w, h, quality)),
        (MODE_ZEN_JPEGLI.label, encode_zen_jpegli(px, w, h, quality)),
        (MODE_ZEN_AUTO.label, encode_zen_auto(px, w, h, quality)),
    ];

    let cells: Vec<_> = encoders
        .into_iter()
        .map(|(enc_label, jpeg)| {
            let (_, _, dec_moz) = decode_mozjpeg_sys(&jpeg).expect("mozjpeg decode");
            let (_, _, dec_def) = decode_zen_default(&jpeg);
            let (_, _, dec_compat) = decode_zen_compat(&jpeg);

            let scores = ZENSIM.with(|z| {
                vec![
                    (
                        "moz-sys",
                        zensim_score(z, px, &dec_moz, w as usize, h as usize),
                    ),
                    (
                        "zen-def",
                        zensim_score(z, px, &dec_def, w as usize, h as usize),
                    ),
                    (
                        "zen-cmp",
                        zensim_score(z, px, &dec_compat, w as usize, h as usize),
                    ),
                ]
            });
            (enc_label, scores)
        })
        .collect();

    ReconRow { quality, cells }
}

#[test]
#[ignore = "requires photo corpora and decoder/trellis features"]
fn photo_decoder_reconstruction_vs_original() {
    let images = load_corpus_or_skip();
    if images.is_empty() {
        return;
    }

    println!("=== Decoder Reconstruction vs Original (all encoders × all decoders) ===");
    println!("  Encoders:");
    println!("    {}: {}", MODE_MOZJPEG.label, MODE_MOZJPEG.desc);
    println!("    {}: {}", MODE_ZEN_JPEGLI.label, MODE_ZEN_JPEGLI.desc);
    println!("    {}: {}", MODE_ZEN_AUTO.label, MODE_ZEN_AUTO.desc);
    println!("  Decoders:");
    println!("    moz-sys: mozjpeg-sys | libjpeg-turbo FFI | islow IDCT (13-bit) | fancy upsample");
    println!("    zen-def: zenjpeg | Jpegli IDCT (12-bit) | Triangle upsampling");
    println!("    zen-cmp: zenjpeg | Libjpeg IDCT (13-bit Loeffler) | LibjpegCompat upsample");
    println!("  Metric: zensim(decoded, original) — higher = better reconstruction");
    println!("  Quality range: Q5–Q95 (11 levels)");
    println!();
    print_corpus_info(&images);

    let results = run_par(
        &images,
        &RECON_QUALITY_LEVELS,
        "images × 3 encoders × 3 decoders",
        process_recon,
    );

    let enc_labels = [
        MODE_MOZJPEG.label,
        MODE_ZEN_JPEGLI.label,
        MODE_ZEN_AUTO.label,
    ];
    let dec_labels = ["moz-sys", "zen-def", "zen-cmp"];

    // Print per-encoder table: rows = Q, columns = decoder
    for (ei, enc_label) in enc_labels.iter().enumerate() {
        println!("\n  Encoder: {enc_label}");
        println!(
            "  {:>3}  {:>10}  {:>10}  {:>10}  {:>7}  {:>7}",
            "Q", dec_labels[0], dec_labels[1], dec_labels[2], "Δ def", "Δ cmp"
        );
        println!("  {}", "-".repeat(60));

        for &q in &RECON_QUALITY_LEVELS {
            let qr: Vec<&ReconRow> = results.iter().filter(|r| r.quality == q).collect();
            let n = qr.len() as f64;

            // Mean score for each decoder under this encoder
            let mut means = [0.0f64; 3];
            for r in &qr {
                for (di, &(_, score)) in r.cells[ei].1.iter().enumerate() {
                    means[di] += score;
                }
            }
            for m in &mut means {
                *m /= n;
            }

            // Δ = zen-decoder minus moz-sys
            println!(
                "  Q{q:<2}  {:.2}  {:.2}  {:.2}  {:>+6.2}  {:>+6.2}",
                means[0],
                means[1],
                means[2],
                means[1] - means[0], // zen-def vs moz-sys
                means[2] - means[0], // zen-cmp vs moz-sys
            );
        }
    }

    // Best decoder per encoder (across all Q and images)
    println!();
    for (ei, enc_label) in enc_labels.iter().enumerate() {
        let mut dec_wins = [0usize; 3];
        for r in &results {
            let scores: Vec<f64> = r.cells[ei].1.iter().map(|&(_, s)| s).collect();
            let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for (di, &s) in scores.iter().enumerate() {
                if (s - best).abs() < 0.001 {
                    dec_wins[di] += 1;
                }
            }
        }
        println!(
            "  {enc_label}: best decoder wins — moz-sys:{} zen-def:{} zen-cmp:{}",
            dec_wins[0], dec_wins[1], dec_wins[2]
        );
    }

    println!("PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: ENCODER CROSS-COMPARISON
// ═══════════════════════════════════════════════════════════════════════════

struct CrossResult {
    corpus: &'static str,
    quality: u8,
    cross_score: f64,
    cross_max_diff: u8,
}

fn process_cross(img: &LoadedImage, quality: u8) -> CrossResult {
    let (w, h) = (img.width, img.height);
    let moz = encode_mozjpeg(&img.pixels, w, h, quality);
    let zen = encode_zen_moz(&img.pixels, w, h, quality);
    // Decode both with zenjpeg default (Jpegli IDCT, Triangle upsampling)
    let (_, _, moz_dec) = decode_zen_default(&moz);
    let (_, _, zen_dec) = decode_zen_default(&zen);

    let score = ZENSIM.with(|z| zensim_score(z, &moz_dec, &zen_dec, w as usize, h as usize));

    CrossResult {
        corpus: img.corpus,
        quality,
        cross_score: score,
        cross_max_diff: max_pixel_diff(&moz_dec, &zen_dec),
    }
}

#[test]
#[ignore = "requires photo corpora and decoder/trellis features"]
fn photo_encoder_cross_comparison() {
    let images = load_corpus_or_skip();
    if images.is_empty() {
        return;
    }

    println!("=== Encoder Cross-Comparison: decoded outputs compared to each other ===");
    println!("  Encoder A: mozjpeg-rs | ProgressiveSmallest | 4:2:0");
    println!("  Encoder B: zenjpeg   | ApproxMozjpeg(Q) + MozjpegProgressive | 4:2:0");
    println!("  Decoder:   zenjpeg   | Jpegli IDCT (12-bit) | Triangle upsampling | no ICC");
    println!("  Metric:    zensim(decoded-A, decoded-B) + max pixel diff");
    println!("  Same decoder for both → differences are purely encoder-side.");
    println!();
    print_corpus_info(&images);

    let results = run_par(&images, &QUALITY_LEVELS, "cross-comparisons", process_cross);

    println!("  Q   mean_zensim  min_zensim  worst_max_diff");
    println!("  {}", "-".repeat(48));
    for &q in &QUALITY_LEVELS {
        let qr: Vec<&CrossResult> = results.iter().filter(|r| r.quality == q).collect();
        let n = qr.len() as f64;
        let mean = qr.iter().map(|r| r.cross_score).sum::<f64>() / n;
        let min = qr
            .iter()
            .map(|r| r.cross_score)
            .fold(f64::INFINITY, f64::min);
        let max_diff = qr.iter().map(|r| r.cross_max_diff).max().unwrap_or(0);
        println!("  Q{q:<2}  {mean:>10.2}  {min:>10.2}  {max_diff:>14}");
    }

    for corpus in ["CID22", "CLIC", "gb82"] {
        let cr: Vec<&CrossResult> = results.iter().filter(|r| r.corpus == corpus).collect();
        if cr.is_empty() {
            continue;
        }
        let n = cr.len() as f64;
        let mean = cr.iter().map(|r| r.cross_score).sum::<f64>() / n;
        println!("  {corpus:<5}: mean cross zensim={mean:.2}");
    }

    let overall_mean = results.iter().map(|r| r.cross_score).sum::<f64>() / results.len() as f64;
    let overall_min = results
        .iter()
        .map(|r| r.cross_score)
        .fold(f64::INFINITY, f64::min);
    println!("\nOverall: mean={overall_mean:.2}, min={overall_min:.2}");

    assert!(overall_mean > 75.0, "cross mean {overall_mean:.2} < 75");
    assert!(overall_min > 40.0, "cross min {overall_min:.2} < 40");
    println!("PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: SIZE-MATCHED QUALITY (all modes)
// ═══════════════════════════════════════════════════════════════════════════

/// Binary-search an encoder function to match target size within ±2%.
fn bisect_quality(
    pixels: &[u8],
    w: u32,
    h: u32,
    target: usize,
    enc_fn: fn(&[u8], u32, u32, u8) -> Vec<u8>,
) -> Option<(u8, Vec<u8>)> {
    let lo_bound = (target as f64 * 0.98) as usize;
    let hi_bound = (target as f64 * 1.02) as usize;
    let mut lo: u8 = 1;
    let mut hi: u8 = 100;
    let mut best: Option<(u8, Vec<u8>)> = None;
    let mut best_dist = usize::MAX;

    for _ in 0..15 {
        if lo > hi {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        let jpeg = enc_fn(pixels, w, h, mid);
        let sz = jpeg.len();
        let dist = sz.abs_diff(target);
        if dist < best_dist {
            best_dist = dist;
            best = Some((mid, jpeg));
        }
        if sz >= lo_bound && sz <= hi_bound {
            return best;
        }
        if sz < target {
            lo = mid.saturating_add(1);
        } else {
            hi = mid.saturating_sub(1);
        }
    }
    if let Some((_, ref d)) = best {
        if d.len() >= (target as f64 * 0.95) as usize && d.len() <= (target as f64 * 1.05) as usize
        {
            return best;
        }
    }
    None
}

struct SizeMatchRow {
    quality: u8,
    moz_zensim: f64,
    moz_bfly: f64,
    /// (label, quality_used, bytes, zensim_vs_orig, butteraugli_vs_orig)
    zen_modes: Vec<(&'static str, u8, usize, f64, f64)>,
}

fn process_size_match_all(img: &LoadedImage, moz_quality: u8) -> SizeMatchRow {
    let (w, h, px) = (img.width, img.height, &img.pixels);
    let moz_jpeg = encode_mozjpeg(px, w, h, moz_quality);
    let target = moz_jpeg.len();

    // Decode mozjpeg, score vs original (zensim + butteraugli)
    let (_, _, moz_dec) = decode_zen_default(&moz_jpeg);

    let moz_zensim = ZENSIM.with(|z| zensim_score(z, px, &moz_dec, w as usize, h as usize));
    let moz_bfly = butteraugli_score(px, &moz_dec, w as usize, h as usize);

    // Bisect each zen mode to match mozjpeg's size
    let modes: Vec<(&str, fn(&[u8], u32, u32, u8) -> Vec<u8>)> = vec![
        (
            MODE_ZEN_MOZ.label,
            encode_zen_moz as fn(&[u8], u32, u32, u8) -> Vec<u8>,
        ),
        (
            MODE_ZEN_JPEGLI.label,
            encode_zen_jpegli as fn(&[u8], u32, u32, u8) -> Vec<u8>,
        ),
        (
            MODE_ZEN_AUTO.label,
            encode_zen_auto as fn(&[u8], u32, u32, u8) -> Vec<u8>,
        ),
    ];

    let mut zen_modes = Vec::new();
    for (label, enc_fn) in modes {
        if let Some((q, jpeg)) = bisect_quality(px, w, h, target, enc_fn) {
            let (_, _, dec) = decode_zen_default(&jpeg);
            let zscore = ZENSIM.with(|z| zensim_score(z, px, &dec, w as usize, h as usize));
            let bfly = butteraugli_score(px, &dec, w as usize, h as usize);
            zen_modes.push((label, q, jpeg.len(), zscore, bfly));
        }
    }

    SizeMatchRow {
        quality: moz_quality,
        moz_zensim,
        moz_bfly,
        zen_modes,
    }
}

#[test]
#[ignore = "requires photo corpora and decoder/trellis features"]
fn photo_size_matched_quality() {
    let images = load_corpus_or_skip();
    if images.is_empty() {
        return;
    }

    println!("=== Size-Matched Quality: all modes at equal file size ===");
    println!("  Reference: mozjpeg-rs | ProgressiveSmallest | 4:2:0 | Q as given");
    println!("  Each zenjpeg mode's Q bisected to match mozjpeg's output size ±2%");
    println!("  {}: {}", MODE_ZEN_MOZ.label, MODE_ZEN_MOZ.desc);
    println!("  {}: {}", MODE_ZEN_JPEGLI.label, MODE_ZEN_JPEGLI.desc);
    println!("  {}: {}", MODE_ZEN_AUTO.label, MODE_ZEN_AUTO.desc);
    println!("  Decoder: zenjpeg | Jpegli IDCT (12-bit) | Triangle upsampling | no ICC");
    println!("  Metric:  zensim(decoded, original) — same bits, who wins?");
    println!();
    print_corpus_info(&images);

    let results = run_par(
        &images,
        &QUALITY_LEVELS,
        "images × 3 modes bisection",
        process_size_match_all,
    );

    let labels = [
        MODE_ZEN_MOZ.label,
        MODE_ZEN_JPEGLI.label,
        MODE_ZEN_AUTO.label,
    ];

    // Per-quality summary: zensim table
    println!("  ZENSIM (higher = better):");
    println!(
        "  {:>3}  {:>8}  {:>10} {:>6}  {:>10} {:>6}  {:>10} {:>6}",
        "Q", "mozjpeg", labels[0], "Δ", labels[1], "Δ", labels[2], "Δ"
    );
    println!("  {}", "-".repeat(78));

    for &q in &QUALITY_LEVELS {
        let qr: Vec<&SizeMatchRow> = results.iter().filter(|r| r.quality == q).collect();
        let n = qr.len() as f64;
        let moz_m = qr.iter().map(|r| r.moz_zensim).sum::<f64>() / n;

        let mut cols: Vec<String> = Vec::new();
        for (li, label) in labels.iter().enumerate() {
            let matched: Vec<f64> = qr
                .iter()
                .filter_map(|r| {
                    r.zen_modes.get(li).and_then(
                        |&(l, _, _, s, _)| {
                            if l == *label { Some(s) } else { None }
                        },
                    )
                })
                .collect();
            if matched.is_empty() {
                cols.push(format!("{:>10} {:>+6}", "—", "—"));
            } else {
                let m = matched.iter().sum::<f64>() / matched.len() as f64;
                cols.push(format!("{m:>10.2} {:>+6.2}", m - moz_m));
            }
        }
        println!("  Q{q:<2}  {moz_m:>8.2}  {}", cols.join("  "));
    }

    // Per-quality summary: butteraugli table
    println!();
    println!("  BUTTERAUGLI (lower = better):");
    println!(
        "  {:>3}  {:>8}  {:>10} {:>6}  {:>10} {:>6}  {:>10} {:>6}",
        "Q", "mozjpeg", labels[0], "Δ", labels[1], "Δ", labels[2], "Δ"
    );
    println!("  {}", "-".repeat(78));

    for &q in &QUALITY_LEVELS {
        let qr: Vec<&SizeMatchRow> = results.iter().filter(|r| r.quality == q).collect();
        let n = qr.len() as f64;
        let moz_b = qr.iter().map(|r| r.moz_bfly).sum::<f64>() / n;

        let mut cols: Vec<String> = Vec::new();
        for (li, label) in labels.iter().enumerate() {
            let matched: Vec<f64> = qr
                .iter()
                .filter_map(|r| {
                    r.zen_modes.get(li).and_then(
                        |&(l, _, _, _, b)| {
                            if l == *label { Some(b) } else { None }
                        },
                    )
                })
                .collect();
            if matched.is_empty() {
                cols.push(format!("{:>10} {:>+6}", "—", "—"));
            } else {
                let m = matched.iter().sum::<f64>() / matched.len() as f64;
                // For butteraugli, negative delta = better (lower distance)
                cols.push(format!("{m:>10.3} {:>+6.3}", m - moz_b));
            }
        }
        println!("  Q{q:<2}  {moz_b:>8.3}  {}", cols.join("  "));
    }

    // Overall win/loss for each mode (zensim)
    println!();
    println!("  Size-matched win/loss (zensim, higher=better):");
    for (li, label) in labels.iter().enumerate() {
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut deltas: Vec<f64> = Vec::new();
        for r in &results {
            if let Some(&(l, _, _, s, _)) = r.zen_modes.get(li) {
                if l == *label {
                    let d = s - r.moz_zensim;
                    deltas.push(d);
                    if d > 0.1 {
                        wins += 1;
                    }
                    if d < -0.1 {
                        losses += 1;
                    }
                }
            }
        }
        if !deltas.is_empty() {
            let mean_d = deltas.iter().sum::<f64>() / deltas.len() as f64;
            let ties = deltas.len() - wins - losses;
            println!("  {label}: Δ={mean_d:>+.3} zensim, {wins}w/{ties}t/{losses}l");
        }
    }

    // Overall win/loss for each mode (butteraugli)
    println!();
    println!("  Size-matched win/loss (butteraugli, lower=better):");
    for (li, label) in labels.iter().enumerate() {
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut deltas: Vec<f64> = Vec::new();
        for r in &results {
            if let Some(&(l, _, _, _, b)) = r.zen_modes.get(li) {
                if l == *label {
                    // Negative delta = zen is better (lower butteraugli)
                    let d = b - r.moz_bfly;
                    deltas.push(d);
                    if d < -0.01 {
                        wins += 1;
                    } // zen lower = zen wins
                    if d > 0.01 {
                        losses += 1;
                    } // zen higher = zen loses
                }
            }
        }
        if !deltas.is_empty() {
            let mean_d = deltas.iter().sum::<f64>() / deltas.len() as f64;
            let ties = deltas.len() - wins - losses;
            println!("  {label}: Δ={mean_d:>+.4} bfly, {wins}w/{ties}t/{losses}l");
        }
    }

    // zen-auto at matched size must be competitive
    let auto_zensim_deltas: Vec<f64> = results
        .iter()
        .filter_map(|r| {
            r.zen_modes.get(2).and_then(|&(l, _, _, s, _)| {
                if l == MODE_ZEN_AUTO.label {
                    Some(s - r.moz_zensim)
                } else {
                    None
                }
            })
        })
        .collect();
    let auto_mean = if auto_zensim_deltas.is_empty() {
        0.0
    } else {
        auto_zensim_deltas.iter().sum::<f64>() / auto_zensim_deltas.len() as f64
    };
    println!("\nzen-auto size-matched mean Δ vs mozjpeg: {auto_mean:+.3} zensim");
    assert!(
        auto_mean > -2.0,
        "zen-auto size-matched mean delta {auto_mean:+.3} — regression"
    );
    println!("PASS");
}
