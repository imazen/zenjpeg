//! Investigation test for progressive3.jpg decode accuracy.
//!
//! On macOS Intel (macos-26-intel CI runner), progressive3.jpg produces
//! completely wrong decode output (interior max_diff=255 vs djpeg reference).
//! This test decodes the file with multiple decoders and IDCT methods to
//! isolate the issue.
//!
//! progressive3.jpg characteristics:
//! - 650x470 SOF2 (progressive) 4:4:4 JPEG with 11 scans
//! - All components 1x1 sampling (MCU = 8x8)
//! - 650 mod 8 = 2 (partial right edge), 470 mod 8 = 6 (partial bottom edge)
//! - Y component: DC first, AC [1..8] first Al=2, AC [9..63] first Al=2,
//!   AC [1..63] refine Ah=2->Al=1, AC [1..63] refine Ah=1->Al=0
//! - Cb/Cr: DC first, AC [1..2] first, AC [3..63] first
//! - No restart markers (DRI=0)
//!
//! Run: cargo test --release -p zenjpeg --features decoder --test progressive3_decode -- --nocapture

use archmage::SimdToken;
use enough::Unstoppable;
use std::fs;

/// Find progressive3.jpg in the codec-corpus cache.
fn find_progressive3() -> Option<Vec<u8>> {
    if let Ok(corpus) = codec_corpus::Corpus::new()
        && let Ok(dir) = corpus.get("jpeg-conformance/valid")
    {
        let path = dir.join("progressive3.jpg");
        if path.exists() {
            return fs::read(&path).ok();
        }
    }
    let cache_path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/codec-corpus/v1/jpeg-conformance/valid/progressive3.jpg");
    if cache_path.exists() {
        return fs::read(&cache_path).ok();
    }
    None
}

/// Decode with zenjpeg (default Jpegli IDCT).
fn decode_zenjpeg_default(data: &[u8]) -> (Vec<u8>, usize, usize, usize) {
    let img = zenjpeg::decoder::Decoder::new()
        .decode(data, Unstoppable)
        .expect("zenjpeg decode failed");
    let w = img.width as usize;
    let h = img.height as usize;
    let pixels = img.into_pixels_u8().unwrap();
    let ch = if pixels.len() == w * h { 1 } else { 3 };
    (pixels, w, h, ch)
}

/// Decode with zenjpeg using Libjpeg IDCT method (Loeffler algorithm, i64).
fn decode_zenjpeg_libjpeg_compat(data: &[u8]) -> (Vec<u8>, usize, usize, usize) {
    use zenjpeg::decode::IdctMethod;
    let img = zenjpeg::decoder::Decoder::new()
        .idct_method(IdctMethod::Libjpeg)
        .decode(data, Unstoppable)
        .expect("zenjpeg libjpeg-compat decode failed");
    let w = img.width as usize;
    let h = img.height as usize;
    let pixels = img.into_pixels_u8().unwrap();
    let ch = if pixels.len() == w * h { 1 } else { 3 };
    (pixels, w, h, ch)
}

/// Decode with zune-jpeg (pure Rust, integer IDCT).
fn decode_zune(data: &[u8]) -> (Vec<u8>, usize, usize, usize) {
    use zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let mut decoder = JpegDecoder::new(ZCursor::new(data));
    let pixels = decoder.decode().expect("zune decode failed");
    let info = decoder.info().unwrap();
    let ch = info.components as usize;
    (pixels, info.width as usize, info.height as usize, ch)
}

/// Decode with jpeg-decoder crate (pure Rust reference decoder).
fn decode_jpeg_decoder_crate(data: &[u8]) -> (Vec<u8>, usize, usize, usize) {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    let pixels = decoder.decode().expect("jpeg-decoder decode failed");
    let info = decoder.info().unwrap();
    let channels = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 | jpeg_decoder::PixelFormat::L16 => 1,
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::CMYK32 => 4,
    };
    (pixels, info.width as usize, info.height as usize, channels)
}

/// Compute max pixel diff between two decoded images.
fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Compute max pixel diff in different regions.
fn region_diffs(
    a: &[u8],
    b: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> (u8, u8, u8, u8) {
    let mcu_w = 8;
    let mcu_h = 8;
    let edge_w = width % mcu_w;
    let edge_h = height % mcu_h;
    let right_start = if edge_w > 0 { width - edge_w } else { width };
    let bottom_start = if edge_h > 0 { height - edge_h } else { height };

    let mut interior_max = 0u8;
    let mut right_max = 0u8;
    let mut bottom_max = 0u8;
    let mut corner_max = 0u8;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            let mut pixel_max = 0u8;
            for c in 0..channels {
                let diff = (a[idx + c] as i16 - b[idx + c] as i16).unsigned_abs() as u8;
                pixel_max = pixel_max.max(diff);
            }
            let in_right = x >= right_start && edge_w > 0;
            let in_bottom = y >= bottom_start && edge_h > 0;
            if in_right && in_bottom {
                corner_max = corner_max.max(pixel_max);
            } else if in_right {
                right_max = right_max.max(pixel_max);
            } else if in_bottom {
                bottom_max = bottom_max.max(pixel_max);
            } else {
                interior_max = interior_max.max(pixel_max);
            }
        }
    }

    (interior_max, right_max, bottom_max, corner_max)
}

/// Decode progressive3.jpg with multiple decoders and IDCT methods.
///
/// Tests both the default (Jpegli) IDCT path and the LibjpegCompat path.
/// Compares against zune-jpeg and jpeg-decoder as ground truth.
///
/// On macOS Intel, this was producing Interior=255 max_diff vs djpeg,
/// indicating completely wrong progressive decode output.
#[test]
fn progressive3_decode_comparison() {
    let data = match find_progressive3() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: progressive3.jpg not found in codec-corpus");
            return;
        }
    };

    eprintln!("=== progressive3.jpg decode comparison ===");
    eprintln!("File size: {} bytes", data.len());

    // Decode with all decoders
    let (zen_px, zen_w, zen_h, zen_ch) = decode_zenjpeg_default(&data);
    eprintln!(
        "zenjpeg (default):    {}x{} {}ch {} bytes",
        zen_w,
        zen_h,
        zen_ch,
        zen_px.len()
    );

    let (zen_lj_px, zen_lj_w, zen_lj_h, zen_lj_ch) = decode_zenjpeg_libjpeg_compat(&data);
    eprintln!(
        "zenjpeg (libjpeg):    {}x{} {}ch {} bytes",
        zen_lj_w,
        zen_lj_h,
        zen_lj_ch,
        zen_lj_px.len()
    );

    let (zune_px, zune_w, zune_h, zune_ch) = decode_zune(&data);
    eprintln!(
        "zune-jpeg:            {}x{} {}ch {} bytes",
        zune_w,
        zune_h,
        zune_ch,
        zune_px.len()
    );

    let (jd_px, jd_w, jd_h, jd_ch) = decode_jpeg_decoder_crate(&data);
    eprintln!(
        "jpeg-decoder:         {}x{} {}ch {} bytes",
        jd_w,
        jd_h,
        jd_ch,
        jd_px.len()
    );

    // Verify dimensions
    assert_eq!((zen_w, zen_h), (650, 470), "unexpected zenjpeg dimensions");
    assert_eq!(
        (zen_w, zen_h, zen_ch),
        (zune_w, zune_h, zune_ch),
        "dimension mismatch zen vs zune"
    );
    assert_eq!(
        (zen_w, zen_h, zen_ch),
        (jd_w, jd_h, jd_ch),
        "dimension mismatch zen vs jpeg-decoder"
    );

    // Pairwise comparisons
    let zen_vs_zune = max_pixel_diff(&zen_px, &zune_px);
    let zen_vs_jd = max_pixel_diff(&zen_px, &jd_px);
    let zen_lj_vs_jd = max_pixel_diff(&zen_lj_px, &jd_px);
    let zen_lj_vs_zune = max_pixel_diff(&zen_lj_px, &zune_px);
    let zen_vs_zen_lj = max_pixel_diff(&zen_px, &zen_lj_px);
    let zune_vs_jd = max_pixel_diff(&zune_px, &jd_px);

    eprintln!();
    eprintln!("Pairwise max pixel diffs:");
    eprintln!("  zen(default) vs zune:         {}", zen_vs_zune);
    eprintln!("  zen(default) vs jpeg-decoder:  {}", zen_vs_jd);
    eprintln!("  zen(libjpeg) vs jpeg-decoder:  {}", zen_lj_vs_jd);
    eprintln!("  zen(libjpeg) vs zune:         {}", zen_lj_vs_zune);
    eprintln!("  zen(default) vs zen(libjpeg): {}", zen_vs_zen_lj);
    eprintln!("  zune vs jpeg-decoder:          {}", zune_vs_jd);

    // Region analysis for each pair
    let (int_z, rt_z, bot_z, cor_z) = region_diffs(&zen_px, &zune_px, zen_w, zen_h, zen_ch);
    eprintln!();
    eprintln!(
        "zen(default) vs zune regions: Interior={} Right={} Bottom={} Corner={}",
        int_z, rt_z, bot_z, cor_z
    );

    let (int_jd, rt_jd, bot_jd, cor_jd) = region_diffs(&zen_px, &jd_px, zen_w, zen_h, zen_ch);
    eprintln!(
        "zen(default) vs jd regions:   Interior={} Right={} Bottom={} Corner={}",
        int_jd, rt_jd, bot_jd, cor_jd
    );

    let (int_lj, rt_lj, bot_lj, cor_lj) = region_diffs(&zen_lj_px, &jd_px, zen_w, zen_h, zen_ch);
    eprintln!(
        "zen(libjpeg) vs jd regions:   Interior={} Right={} Bottom={} Corner={}",
        int_lj, rt_lj, bot_lj, cor_lj
    );

    // Sample pixels at key locations
    eprintln!();
    eprintln!("Sample pixels (RGB) at key locations:");
    let locations = [
        (0, 0, "top-left"),
        (zen_w / 2, zen_h / 2, "center"),
        (zen_w - 1, 0, "top-right edge"),
        (0, zen_h - 1, "bottom-left edge"),
        (zen_w - 1, zen_h - 1, "corner"),
    ];
    for (x, y, label) in locations {
        let i = (y * zen_w + x) * zen_ch;
        eprintln!(
            "  ({:>3},{:>3}) {:<20} zen=({:>3},{:>3},{:>3}) zune=({:>3},{:>3},{:>3}) jd=({:>3},{:>3},{:>3})",
            x,
            y,
            label,
            zen_px[i],
            zen_px[i + 1],
            zen_px[i + 2],
            zune_px[i],
            zune_px[i + 1],
            zune_px[i + 2],
            jd_px[i],
            jd_px[i + 1],
            jd_px[i + 2],
        );
    }

    // Report SIMD capability for diagnosis
    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = archmage::X64V3Token::summon().is_some();
        let has_avx512 = archmage::X64V4Token::summon().is_some();
        eprintln!();
        eprintln!("SIMD: AVX2={} AVX-512={}", has_avx2, has_avx512);
    }
    #[cfg(target_arch = "aarch64")]
    {
        eprintln!();
        eprintln!("SIMD: NEON (always available on aarch64)");
    }

    // Assertions: zenjpeg must match pure-Rust reference decoders closely
    // Both zune-jpeg and jpeg-decoder are pure Rust with well-tested progressive
    // support. If zenjpeg diverges from BOTH, it has a bug.
    assert!(
        zen_vs_zune <= 2,
        "zenjpeg(default) vs zune-jpeg max_diff={} is too large (expected <= 2). \
         Both use integer IDCT, so any large difference indicates a progressive \
         decode bug in zenjpeg (coefficient accumulation, IDCT, or color conversion).",
        zen_vs_zune
    );
    assert!(
        zen_vs_jd <= 5,
        "zenjpeg(default) vs jpeg-decoder max_diff={} is too large (expected <= 5). \
         This indicates a decode bug. On macOS Intel, this was producing max_diff=255.",
        zen_vs_jd
    );
    assert!(
        zen_lj_vs_jd <= 3,
        "zenjpeg(libjpeg-compat) vs jpeg-decoder max_diff={} (expected <= 3). \
         Both use Loeffler IDCT, so differences should be minimal.",
        zen_lj_vs_jd
    );
}

/// Test that both IDCT paths produce equivalent output for progressive3.jpg.
///
/// If this test fails but progressive3_decode_comparison passes, the bug is
/// specifically in one IDCT implementation (not in progressive coefficient
/// accumulation or color conversion).
#[test]
fn progressive3_idct_path_equivalence() {
    let data = match find_progressive3() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: progressive3.jpg not found in codec-corpus");
            return;
        }
    };

    let (default_px, _, _, _) = decode_zenjpeg_default(&data);
    let (libjpeg_px, _, _, _) = decode_zenjpeg_libjpeg_compat(&data);

    let max_diff = max_pixel_diff(&default_px, &libjpeg_px);
    eprintln!(
        "progressive3.jpg: default vs libjpeg IDCT max_diff = {}",
        max_diff
    );

    // The two IDCT methods should produce nearly identical output.
    // The Jpegli IDCT uses 12-bit fixed-point constants while Libjpeg uses
    // 13-bit Loeffler constants, so small differences (up to 3) are normal.
    // A large divergence (>5) would indicate a platform-specific IDCT bug.
    assert!(
        max_diff <= 5,
        "IDCT path divergence: default vs libjpeg max_diff={}. \
         If this fails on macOS Intel but not Linux, the bug is in a \
         platform-specific IDCT code path (likely AVX2 vs scalar dispatch).",
        max_diff
    );
}
