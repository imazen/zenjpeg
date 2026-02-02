//! Comparison: zenjpeg internalized trellis vs mozjpeg-rs trellis.
//!
//! Two levels of testing:
//!
//! 1. **Block-level parity**: Feed identical DCT coefficients and quant tables
//!    to both trellis implementations, verify quantized output matches exactly.
//!    This proves the internalized algorithm is correct.
//!
//! 2. **Full-encode comparison**: Encode CID22 corpus images with both encoders
//!    using matched settings (Robidoux quant tables, same trellis config),
//!    comparing file size and SSIMULACRA2 quality.
//!
//! Run block-level:
//!   cargo test --release -p zenjpeg --test trellis_mozjpeg_comparison -- --nocapture trellis_block_parity
//!
//! Run full-encode (requires CID22 corpus + mozjpeg-tables feature):
//!   cargo test --release -p zenjpeg --features mozjpeg-tables --test trellis_mozjpeg_comparison -- --nocapture --ignored

#[cfg(feature = "mozjpeg-tables")]
use std::path::{Path, PathBuf};

// zenjpeg internalized trellis
use zenjpeg::encode::mozjpeg_compat::TrellisConfig;
use zenjpeg::trellis::{trellis_quantize_block, RateTable};

// mozjpeg-rs trellis (the original we ported from)
use mozjpeg_rs::consts::{
    AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES, DCTSIZE2,
};
use mozjpeg_rs::huffman::{DerivedTable, HuffTable};
use mozjpeg_rs::trellis::trellis_quantize_block as moz_trellis_quantize_block;
use mozjpeg_rs::TrellisConfig as MozTrellisConfig;

// ============================================================================
// Huffman table builders (standard tables in both formats)
// ============================================================================

/// Build mozjpeg-rs DerivedTable from standard AC luminance Huffman.
fn moz_standard_luma_ac() -> DerivedTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_LUMINANCE_BITS);
    for (i, &v) in AC_LUMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    DerivedTable::from_huff_table(&htbl, false).unwrap()
}

/// Build mozjpeg-rs DerivedTable from standard AC chrominance Huffman.
fn moz_standard_chroma_ac() -> DerivedTable {
    let mut htbl = HuffTable::default();
    htbl.bits.copy_from_slice(&AC_CHROMINANCE_BITS);
    for (i, &v) in AC_CHROMINANCE_VALUES.iter().enumerate() {
        htbl.huffval[i] = v;
    }
    DerivedTable::from_huff_table(&htbl, false).unwrap()
}

/// Verify RateTable and DerivedTable produce the same code lengths.
fn verify_table_parity(rate: &RateTable, derived: &DerivedTable, name: &str) {
    for symbol in 0..=255u8 {
        let rate_len = rate.get_code_length(symbol);
        let (_, derived_len) = derived.get_code(symbol);
        assert_eq!(
            rate_len, derived_len,
            "{name}: code length mismatch for symbol {symbol:#04x}: rate={rate_len}, derived={derived_len}"
        );
    }
}

// ============================================================================
// Matched config pairs (zenjpeg + mozjpeg-rs constructors produce equivalent state)
// ============================================================================

/// Build matching (zenjpeg, mozjpeg-rs) TrellisConfig pairs.
///
/// Uses named constructors on both sides since they share the same parameter values.
/// Fields are pub(crate) in zenjpeg, so we can't convert field-by-field.
fn matched_configs() -> Vec<(&'static str, TrellisConfig, MozTrellisConfig)> {
    vec![
        ("default", TrellisConfig::default(), MozTrellisConfig::default()),
        ("favor_size", TrellisConfig::favor_size(), MozTrellisConfig::favor_size()),
        ("thorough", TrellisConfig::thorough(), MozTrellisConfig::thorough()),
    ]
}

// ============================================================================
// Block-level parity tests
// ============================================================================

/// Standard JPEG Annex K luminance quant table
const ANNEX_K_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
    56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
    104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Robidoux/ImageMagick luminance quant table (mozjpeg default)
const ROBIDOUX_LUMA: [u16; 64] = [
    16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
    135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74, 94,
    124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311, 418,
];

/// Compare a single block through both trellis implementations.
fn compare_block(
    src: &[i32; DCTSIZE2],
    qtable: &[u16; DCTSIZE2],
    zen_table: &RateTable,
    moz_table: &DerivedTable,
    zen_config: &TrellisConfig,
    moz_config: &MozTrellisConfig,
    label: &str,
) -> bool {
    let mut zen_out = [0i16; DCTSIZE2];
    trellis_quantize_block(src, &mut zen_out, qtable, zen_table, zen_config);

    let mut moz_out = [0i16; DCTSIZE2];
    moz_trellis_quantize_block(src, &mut moz_out, qtable, moz_table, moz_config);

    if zen_out != moz_out {
        let mut diffs = 0;
        for i in 0..DCTSIZE2 {
            if zen_out[i] != moz_out[i] {
                diffs += 1;
                if diffs <= 5 {
                    eprintln!(
                        "  [{label}] coeff[{i}]: zen={}, moz={}",
                        zen_out[i], moz_out[i]
                    );
                }
            }
        }
        eprintln!("  [{label}] total mismatches: {diffs}/64");
        return false;
    }
    true
}

/// Generate a deterministic pseudo-random block of DCT coefficients.
///
/// Uses a simple LCG for reproducibility without depending on rand.
fn make_test_block(seed: u64, scale: i32) -> [i32; DCTSIZE2] {
    let mut block = [0i32; DCTSIZE2];
    let mut state = seed;
    for coeff in &mut block {
        // LCG: next = (a * state + c) mod m
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Map to range [-scale, +scale]
        let raw = ((state >> 33) as i64) as i32;
        *coeff = raw % (scale + 1);
    }
    block
}

#[test]
fn trellis_block_parity_table_lengths() {
    // Verify that RateTable and DerivedTable produce identical code lengths
    let zen_luma = RateTable::standard_luma_ac();
    let zen_chroma = RateTable::standard_chroma_ac();
    let moz_luma = moz_standard_luma_ac();
    let moz_chroma = moz_standard_chroma_ac();

    verify_table_parity(&zen_luma, &moz_luma, "luma_ac");
    verify_table_parity(&zen_chroma, &moz_chroma, "chroma_ac");
    println!("All 512 Huffman code lengths match between RateTable and DerivedTable");
}

#[test]
fn trellis_block_parity_default_config() {
    let zen_table = RateTable::standard_luma_ac();
    let moz_table = moz_standard_luma_ac();
    let zen_config = TrellisConfig::default();
    let moz_config = MozTrellisConfig::default();

    let mut pass = 0u32;
    let mut fail = 0u32;

    for seed in 0..1000 {
        // Scale coefficients by 8*quant to simulate realistic DCT values
        // (trellis expects raw DCT coefficients before quantization)
        let src = make_test_block(seed, 8000);
        if compare_block(&src, &ANNEX_K_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, &format!("seed={seed}")) {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    println!("TrellisConfig::default() — Annex K luma: {pass} pass, {fail} fail out of 1000 blocks");
    assert_eq!(fail, 0, "Block-level parity failures with default config");
}

#[test]
fn trellis_block_parity_all_configs() {
    let configs = matched_configs();

    let tables: Vec<(&str, &[u16; 64], RateTable, DerivedTable)> = vec![
        ("annex_k", &ANNEX_K_LUMA, RateTable::standard_luma_ac(), moz_standard_luma_ac()),
        ("robidoux", &ROBIDOUX_LUMA, RateTable::standard_luma_ac(), moz_standard_luma_ac()),
        ("chroma", &ANNEX_K_LUMA, RateTable::standard_chroma_ac(), moz_standard_chroma_ac()),
    ];

    let mut total_pass = 0u32;
    let mut total_fail = 0u32;

    for (config_name, zen_config, moz_config) in &configs {
        for (table_name, qtable, zen_table, moz_table) in &tables {
            let mut pass = 0u32;
            let mut fail = 0u32;

            for seed in 0..200 {
                let src = make_test_block(seed, 8000);
                let label = format!("{config_name}/{table_name}/seed={seed}");
                if compare_block(&src, qtable, zen_table, moz_table, zen_config, moz_config, &label) {
                    pass += 1;
                } else {
                    fail += 1;
                }
            }

            println!("  {config_name:>12} / {table_name:<10}: {pass} pass, {fail} fail");
            total_pass += pass;
            total_fail += fail;
        }
    }

    println!("\nTotal: {total_pass} pass, {total_fail} fail out of {} blocks",
             total_pass + total_fail);
    assert_eq!(total_fail, 0, "Block-level parity failures across configs");
}

#[test]
fn trellis_block_parity_edge_cases() {
    let zen_table = RateTable::standard_luma_ac();
    let moz_table = moz_standard_luma_ac();
    let zen_config = TrellisConfig::default();
    let moz_config = MozTrellisConfig::default();
    let mut fail = 0u32;

    // All zeros
    {
        let src = [0i32; DCTSIZE2];
        if !compare_block(&src, &ANNEX_K_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, "zeros") {
            fail += 1;
        }
    }

    // DC only
    {
        let mut src = [0i32; DCTSIZE2];
        src[0] = 5000;
        if !compare_block(&src, &ANNEX_K_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, "dc_only") {
            fail += 1;
        }
    }

    // Single AC coefficient
    for pos in 1..DCTSIZE2 {
        let mut src = [0i32; DCTSIZE2];
        src[pos] = 500;
        let label = format!("single_ac_{pos}");
        if !compare_block(&src, &ANNEX_K_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, &label) {
            fail += 1;
        }
    }

    // Near-max values
    {
        let src = [7000i32; DCTSIZE2];
        if !compare_block(&src, &ANNEX_K_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, "near_max") {
            fail += 1;
        }
    }

    // All negative
    {
        let src = [-3000i32; DCTSIZE2];
        if !compare_block(&src, &ANNEX_K_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, "all_negative") {
            fail += 1;
        }
    }

    // Alternating sign
    {
        let mut src = [0i32; DCTSIZE2];
        for i in 0..DCTSIZE2 {
            src[i] = if i % 2 == 0 { 2000 } else { -2000 };
        }
        if !compare_block(&src, &ANNEX_K_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, "alternating") {
            fail += 1;
        }
    }

    // With Robidoux tables (high quant values)
    {
        let src = make_test_block(42, 50000);
        if !compare_block(&src, &ROBIDOUX_LUMA, &zen_table, &moz_table, &zen_config, &moz_config, "robidoux_large") {
            fail += 1;
        }
    }

    println!("Edge cases: {} tested, {} failed", 6 + 63, fail);
    assert_eq!(fail, 0, "Block-level parity edge case failures");
}

// ============================================================================
// Full-encode comparison (requires CID22 corpus + mozjpeg-tables)
// ============================================================================

#[cfg(feature = "mozjpeg-tables")]
mod full_encode {
    use super::*;
    use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, MozjpegTables, PixelLayout, QuantTablePreset};

    use mozjpeg_rs::Encoder as MozEncoder;
    use mozjpeg_rs::Preset as MozPreset;
    use mozjpeg_rs::Subsampling as MozSubsampling;

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

    /// Encode with zenjpeg using Robidoux quant tables + trellis.
    ///
    /// Uses mozjpeg-compatible quant tables to isolate trellis behavior.
    /// Note: zenjpeg still applies jpegli AQ to trellis lambda, which
    /// is an expected difference from mozjpeg-rs.
    fn encode_zenjpeg_robidoux(
        pixels: &[u8],
        w: u32,
        h: u32,
        quality: u8,
        subsamp: ChromaSubsampling,
        progressive: bool,
        trellis: Option<TrellisConfig>,
    ) -> Vec<u8> {
        let tables = MozjpegTables::generate(quality, QuantTablePreset::Robidoux);
        let mut config = EncoderConfig::ycbcr(quality as f32, subsamp)
            .tables(tables)
            .progressive(progressive)
            .separate_chroma_tables(false) // mozjpeg uses shared Cb/Cr table
            .allow_16bit_quant_tables(false); // match mozjpeg baseline clamping
        if let Some(t) = trellis {
            config = config.trellis(t);
        }
        let mut enc = config
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, enough::Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    /// Encode with mozjpeg-rs using matching settings.
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
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
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
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        compute_frame_ssimulacra2(orig_rgb, dec_rgb).unwrap_or(-999.0)
    }

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
        ];

        let mut configs = Vec::new();
        for &quality in &qualities {
            for (ss_name, ss_zen, ss_moz) in &subsampling_pairs {
                for (prog, prog_name) in &progressive_modes {
                    for (trellis_name, t_zen, t_moz) in &trellis_presets {
                        let label_str =
                            format!("q{quality}-{ss_name}-{prog_name}-{trellis_name}");
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

    /// Full-encode comparison with matched Robidoux quant tables.
    ///
    /// Expected remaining differences:
    /// - DCT implementation (jpegli vs mozjpeg produce ±1 coefficient differences)
    /// - AQ: zenjpeg modulates trellis lambda by AQ strength; mozjpeg-rs does not
    /// - Color conversion: slightly different RGB→YCbCr implementations
    /// - Huffman optimization: different algorithms for table construction
    /// - Deringing: different overshoot deringing implementations
    #[test]
    #[ignore] // Requires CID22 corpus
    fn full_encode_robidoux_comparison() {
        let images = load_cid22_images(15);
        if images.is_empty() {
            eprintln!("No CID22 images found, skipping test");
            return;
        }

        let configs = build_configs();
        println!(
            "\nMatched Robidoux tables: {} images × {} configs = {} encodes per encoder",
            images.len(),
            configs.len(),
            images.len() * configs.len()
        );
        println!("Known differences: AQ-modulated lambda, DCT ±1, color conversion, Huffman opt\n");

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
        let mut worst_size_regression: f64 = f64::NEG_INFINITY;
        let mut worst_size_label = String::new();

        for config in &configs {
            let mut cfg_zen_bytes: u64 = 0;
            let mut cfg_moz_bytes: u64 = 0;
            let mut cfg_zen_ssim2: f64 = 0.0;
            let mut cfg_moz_ssim2: f64 = 0.0;
            let mut cfg_count = 0u32;

            for (_name, pixels, w, h) in &images {
                let wu = *w as usize;
                let hu = *h as usize;

                let zen_jpeg = encode_zenjpeg_robidoux(
                    pixels,
                    *w,
                    *h,
                    config.quality,
                    config.subsamp_zen,
                    config.progressive,
                    config.trellis_zen,
                );

                let moz_jpeg = encode_mozjpeg(
                    pixels,
                    *w,
                    *h,
                    config.quality,
                    config.subsamp_moz,
                    config.progressive,
                    config.trellis_moz,
                );

                let zen_ssim2 = compute_ssim2(pixels, &zen_jpeg, wu, hu);
                let moz_ssim2 = compute_ssim2(pixels, &moz_jpeg, wu, hu);

                cfg_zen_bytes += zen_jpeg.len() as u64;
                cfg_moz_bytes += moz_jpeg.len() as u64;
                cfg_zen_ssim2 += zen_ssim2;
                cfg_moz_ssim2 += moz_ssim2;
                cfg_count += 1;

                let size_pct =
                    (zen_jpeg.len() as f64 / moz_jpeg.len() as f64 - 1.0) * 100.0;
                if size_pct > worst_size_regression {
                    worst_size_regression = size_pct;
                    worst_size_label = format!("{} / {}", config.label, _name);
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
        let overall_size_pct =
            (total_zen_bytes as f64 / total_moz_bytes as f64 - 1.0) * 100.0;
        let overall_ssim2_delta =
            total_zen_ssim2 / count as f64 - total_moz_ssim2 / count as f64;
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
            "\nWorst size regression: {:>+.1}% ({})",
            worst_size_regression, worst_size_label
        );
        println!("Positive Δ% = zenjpeg larger; Positive Δs2 = zenjpeg better quality");

        // With matched quant tables, differences should be smaller than
        // cross-encoder comparison. Still allow headroom for AQ/DCT differences.
        assert!(
            worst_size_regression < 40.0,
            "Size regression too large with matched tables: {worst_size_regression:.1}% on {worst_size_label}"
        );
    }
}
