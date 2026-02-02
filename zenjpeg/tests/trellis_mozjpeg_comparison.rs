//! Comparison: zenjpeg internalized trellis vs mozjpeg-rs trellis.
//!
//! Encodes CID22 corpus images across quality levels, subsampling modes,
//! trellis presets, and progressive settings, comparing file size and
//! SSIMULACRA2 quality between zenjpeg and mozjpeg-rs.
//!
//! Run: cargo test --release -p zenjpeg --test trellis_mozjpeg_comparison -- --nocapture --ignored

use std::path::{Path, PathBuf};

// zenjpeg
use zenjpeg::encode::mozjpeg_compat::TrellisConfig;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

// mozjpeg-rs (re-exported at crate root)
use mozjpeg_rs::Encoder as MozEncoder;
use mozjpeg_rs::Preset as MozPreset;
use mozjpeg_rs::Subsampling as MozSubsampling;
use mozjpeg_rs::TrellisConfig as MozTrellisConfig;

// Quality metrics
use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

// ============================================================================
// Image loading
// ============================================================================

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    let (rgb, w, h) = match info.color_type {
        png::ColorType::Rgb => (buf, info.width, info.height),
        png::ColorType::Rgba => {
            let rgb: Vec<u8> = buf.chunks(4).flat_map(|c| &c[..3]).copied().collect();
            (rgb, info.width, info.height)
        }
        _ => return None,
    };
    Some((rgb, w, h))
}

fn find_cid22_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("/home/lilith/work/codec-eval/codec-corpus/CID22/CID22-512/validation"),
        PathBuf::from("../codec-eval/codec-corpus/CID22/CID22-512/validation"),
        PathBuf::from("../../codec-eval/codec-corpus/CID22/CID22-512/validation"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

fn load_cid22_images(max_images: usize) -> Vec<(String, Vec<u8>, u32, u32)> {
    let dir = match find_cid22_dir() {
        Some(d) => d,
        None => {
            eprintln!("CID22 corpus not found, skipping");
            return vec![];
        }
    };

    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    entries.truncate(max_images);

    entries
        .iter()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let (rgb, w, h) = load_png(&entry.path())?;
            Some((name, rgb, w, h))
        })
        .collect()
}

// ============================================================================
// Encoding helpers
// ============================================================================

fn encode_zenjpeg(
    pixels: &[u8],
    w: u32,
    h: u32,
    quality: f32,
    subsamp: ChromaSubsampling,
    progressive: bool,
    trellis: Option<TrellisConfig>,
) -> Vec<u8> {
    let mut config = EncoderConfig::ycbcr(quality, subsamp).progressive(progressive);
    if let Some(t) = trellis {
        config = config.trellis(t);
    }
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(pixels, enough::Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn encode_mozjpeg(
    pixels: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    subsamp: MozSubsampling,
    progressive: bool,
    trellis: MozTrellisConfig,
) -> Vec<u8> {
    let preset = if progressive {
        MozPreset::ProgressiveBalanced
    } else {
        MozPreset::BaselineBalanced
    };
    MozEncoder::new(preset)
        .quality(quality)
        .subsampling(subsamp)
        .trellis(trellis)
        .encode_rgb(pixels, w, h)
        .unwrap()
}

// ============================================================================
// Quality measurement
// ============================================================================

fn decode_jpeg_to_rgb(jpeg: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(jpeg);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("jpeg decode failed")
}

fn compute_ssim2(original: &[u8], jpeg_bytes: &[u8], width: usize, height: usize) -> f64 {
    let decoded = decode_jpeg_to_rgb(jpeg_bytes);

    let orig_rgb = Rgb::new(
        original
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let dec_rgb = Rgb::new(
        decoded
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_rgb, dec_rgb).unwrap_or(-999.0)
}

// ============================================================================
// Configuration definitions
// ============================================================================

#[derive(Clone)]
struct TestConfig {
    label: &'static str,
    quality: u8,
    subsamp_zen: ChromaSubsampling,
    subsamp_moz: MozSubsampling,
    progressive: bool,
    trellis_zen: Option<TrellisConfig>,
    trellis_moz: MozTrellisConfig,
}

fn build_configs() -> Vec<TestConfig> {
    let qualities = [50, 75, 90];
    let subsampling_pairs: Vec<(&str, ChromaSubsampling, MozSubsampling)> = vec![
        ("420", ChromaSubsampling::Quarter, MozSubsampling::S420),
        ("444", ChromaSubsampling::None, MozSubsampling::S444),
    ];
    let progressive_modes = [(false, "base"), (true, "prog")];
    let trellis_presets: Vec<(&str, Option<TrellisConfig>, MozTrellisConfig)> = vec![
        ("no-trellis", None, MozTrellisConfig::disabled()),
        (
            "trellis-default",
            Some(TrellisConfig::default()),
            MozTrellisConfig::default(),
        ),
        (
            "trellis-favor-size",
            Some(TrellisConfig::favor_size()),
            MozTrellisConfig::favor_size(),
        ),
        (
            "trellis-thorough",
            Some(TrellisConfig::thorough()),
            MozTrellisConfig::thorough(),
        ),
    ];

    let mut configs = Vec::new();
    for &quality in &qualities {
        for (ss_name, ss_zen, ss_moz) in &subsampling_pairs {
            for (prog, prog_name) in &progressive_modes {
                for (trellis_name, t_zen, t_moz) in &trellis_presets {
                    let label_str = format!("q{quality}-{ss_name}-{prog_name}-{trellis_name}");
                    configs.push(TestConfig {
                        label: Box::leak(label_str.into_boxed_str()),
                        quality,
                        subsamp_zen: *ss_zen,
                        subsamp_moz: *ss_moz,
                        progressive: *prog,
                        trellis_zen: *t_zen,
                        trellis_moz: *t_moz,
                    });
                }
            }
        }
    }
    configs
}

// ============================================================================
// The test
// ============================================================================

#[test]
#[ignore] // Requires CID22 corpus
fn trellis_comparison_cid22_sweep() {
    let images = load_cid22_images(15);
    if images.is_empty() {
        eprintln!("No CID22 images found, skipping test");
        return;
    }

    let configs = build_configs();
    println!(
        "\nComparing zenjpeg vs mozjpeg-rs: {} images × {} configs = {} encodes per encoder\n",
        images.len(),
        configs.len(),
        images.len() * configs.len()
    );

    // Header
    println!(
        "{:<45} {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7}",
        "Config", "zen_KB", "moz_KB", "Δ%", "zen_s2", "moz_s2", "Δs2"
    );
    println!("{}", "-".repeat(100));

    let mut total_zen_bytes: u64 = 0;
    let mut total_moz_bytes: u64 = 0;
    let mut total_zen_ssim2: f64 = 0.0;
    let mut total_moz_ssim2: f64 = 0.0;
    let mut count: u64 = 0;

    // Track worst regressions
    let mut worst_size_regression: f64 = f64::NEG_INFINITY;
    let mut worst_size_label = String::new();
    let mut worst_quality_regression: f64 = f64::INFINITY;
    let mut worst_quality_label = String::new();

    for config in &configs {
        let mut cfg_zen_bytes: u64 = 0;
        let mut cfg_moz_bytes: u64 = 0;
        let mut cfg_zen_ssim2: f64 = 0.0;
        let mut cfg_moz_ssim2: f64 = 0.0;
        let mut cfg_count = 0u32;

        for (_name, pixels, w, h) in &images {
            let wu = *w as usize;
            let hu = *h as usize;

            // Encode with zenjpeg
            let zen_jpeg = encode_zenjpeg(
                pixels,
                *w,
                *h,
                config.quality as f32,
                config.subsamp_zen,
                config.progressive,
                config.trellis_zen,
            );

            // Encode with mozjpeg-rs
            let moz_jpeg = encode_mozjpeg(
                pixels,
                *w,
                *h,
                config.quality,
                config.subsamp_moz,
                config.progressive,
                config.trellis_moz,
            );

            // Measure quality
            let zen_ssim2 = compute_ssim2(pixels, &zen_jpeg, wu, hu);
            let moz_ssim2 = compute_ssim2(pixels, &moz_jpeg, wu, hu);

            cfg_zen_bytes += zen_jpeg.len() as u64;
            cfg_moz_bytes += moz_jpeg.len() as u64;
            cfg_zen_ssim2 += zen_ssim2;
            cfg_moz_ssim2 += moz_ssim2;
            cfg_count += 1;

            // Track per-image regressions
            let size_pct = (zen_jpeg.len() as f64 / moz_jpeg.len() as f64 - 1.0) * 100.0;
            let ssim2_delta = zen_ssim2 - moz_ssim2;

            if size_pct > worst_size_regression {
                worst_size_regression = size_pct;
                worst_size_label = format!("{} / {}", config.label, _name);
            }
            if ssim2_delta < worst_quality_regression {
                worst_quality_regression = ssim2_delta;
                worst_quality_label = format!("{} / {}", config.label, _name);
            }
        }

        let avg_zen_ssim2 = cfg_zen_ssim2 / cfg_count as f64;
        let avg_moz_ssim2 = cfg_moz_ssim2 / cfg_count as f64;
        let size_pct = (cfg_zen_bytes as f64 / cfg_moz_bytes as f64 - 1.0) * 100.0;
        let ssim2_delta = avg_zen_ssim2 - avg_moz_ssim2;

        println!(
            "{:<45} {:>7.1} {:>7.1} {:>+6.1}%  {:>7.2} {:>7.2} {:>+6.2}",
            config.label,
            cfg_zen_bytes as f64 / 1024.0,
            cfg_moz_bytes as f64 / 1024.0,
            size_pct,
            avg_zen_ssim2,
            avg_moz_ssim2,
            ssim2_delta,
        );

        total_zen_bytes += cfg_zen_bytes;
        total_moz_bytes += cfg_moz_bytes;
        total_zen_ssim2 += cfg_zen_ssim2;
        total_moz_ssim2 += cfg_moz_ssim2;
        count += cfg_count as u64;
    }

    println!("{}", "-".repeat(100));
    let overall_size_pct = (total_zen_bytes as f64 / total_moz_bytes as f64 - 1.0) * 100.0;
    let overall_ssim2_delta = total_zen_ssim2 / count as f64 - total_moz_ssim2 / count as f64;
    println!(
        "{:<45} {:>7.1} {:>7.1} {:>+6.1}%  {:>7.2} {:>7.2} {:>+6.2}",
        "OVERALL",
        total_zen_bytes as f64 / 1024.0,
        total_moz_bytes as f64 / 1024.0,
        overall_size_pct,
        total_zen_ssim2 / count as f64,
        total_moz_ssim2 / count as f64,
        overall_ssim2_delta,
    );

    println!(
        "\nWorst size regression:    {:>+.1}% ({})",
        worst_size_regression, worst_size_label
    );
    println!(
        "Worst quality regression: {:>+.2} SSIM2 ({})",
        worst_quality_regression, worst_quality_label
    );
    println!("\nPositive Δ% = zenjpeg larger; Positive Δs2 = zenjpeg better quality");

    // Sanity: no catastrophic regressions.
    // Note: these are different encoders (jpegli-based vs mozjpeg-based) with
    // different quant tables, AQ, and DCT implementations, so per-image size
    // differences up to ~40% are expected. We only catch genuine breakage.
    assert!(
        worst_size_regression < 50.0,
        "Catastrophic size regression: {worst_size_regression:.1}% on {worst_size_label}"
    );
}
