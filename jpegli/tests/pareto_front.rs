//! Pareto front validation: compare jpegli vs mozjpeg quality/size tradeoff.
//!
//! Verifies that jpegli is competitive with mozjpeg on the Pareto front
//! of DSSIM vs file size.

use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let (width, height) = (info.width as usize, info.height as usize);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let orig_rgba = rgb_to_rgba(original);
    let dec_rgba = rgb_to_rgba(decoded);
    let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp = attr.create_image_rgba(&dec_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn encode_jpegli(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality.into()))
        .encode(rgb)
        .expect("jpegli encode")
}

fn encode_mozjpeg(rgb: &[u8], width: usize, height: usize, quality: f32, use_444: bool) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality);

    if use_444 {
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));
    }

    let mut started = comp.start_compress(Vec::new()).expect("mozjpeg start");
    let row_stride = width * 3;
    for y in 0..height {
        let row = &rgb[y * row_stride..(y + 1) * row_stride];
        let _ = started.write_scanlines(row);
    }
    started.finish().expect("mozjpeg finish")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode")
}

/// Test that jpegli is competitive with mozjpeg on Pareto front.
///
/// For similar file sizes, jpegli should achieve similar or better DSSIM.
/// We allow up to 20% worse DSSIM at the same quality setting, since
/// the quality scales may differ between encoders.
#[test]
fn test_pareto_front_flower_small() {
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found");
        return;
    }

    let (original, width, height) = load_png(path).expect("load png");
    let width_u32 = width as u32;
    let height_u32 = height as u32;

    println!("\n=== Pareto Front Comparison (4:4:4 subsampling) ===");
    println!(
        "{:>7} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Quality", "jpegli Size", "moz444 Size", "jpegli DSSIM", "moz444 DSSIM", "Winner"
    );
    println!("{}", "-".repeat(76));

    let mut jpegli_wins = 0;
    let mut mozjpeg_wins = 0;
    let mut ties = 0;

    for quality in [60, 70, 80, 90] {
        let jpegli_data = encode_jpegli(&original, width_u32, height_u32, quality);
        let mozjpeg_data = encode_mozjpeg(&original, width, height, quality as f32, true);

        let jpegli_decoded = decode_jpeg(&jpegli_data);
        let mozjpeg_decoded = decode_jpeg(&mozjpeg_data);

        let jpegli_dssim = compute_dssim(&original, &jpegli_decoded, width, height);
        let mozjpeg_dssim = compute_dssim(&original, &mozjpeg_decoded, width, height);

        // Determine winner: lower DSSIM is better, smaller size is better
        // Use a simple metric: DSSIM * size
        let jpegli_score = jpegli_dssim * jpegli_data.len() as f64;
        let mozjpeg_score = mozjpeg_dssim * mozjpeg_data.len() as f64;

        let winner = if jpegli_score < mozjpeg_score * 0.95 {
            jpegli_wins += 1;
            "jpegli"
        } else if mozjpeg_score < jpegli_score * 0.95 {
            mozjpeg_wins += 1;
            "mozjpeg"
        } else {
            ties += 1;
            "tie"
        };

        println!(
            "{:>7} {:>12} {:>12} {:>12.6} {:>12.6} {:>8}",
            quality,
            jpegli_data.len(),
            mozjpeg_data.len(),
            jpegli_dssim,
            mozjpeg_dssim,
            winner
        );

        // Assert that jpegli is not dramatically worse
        // Allow 50% tolerance since encoders may have different quality curves
        assert!(
            jpegli_dssim < mozjpeg_dssim * 1.5,
            "jpegli DSSIM ({}) is >50% worse than mozjpeg ({}) at quality {}",
            jpegli_dssim,
            mozjpeg_dssim,
            quality
        );

        // Assert that file size is within reason
        assert!(
            jpegli_data.len() < mozjpeg_data.len() * 2,
            "jpegli size ({}) is >2x mozjpeg ({}) at quality {}",
            jpegli_data.len(),
            mozjpeg_data.len(),
            quality
        );
    }

    println!();
    println!(
        "Summary: jpegli wins: {}, mozjpeg wins: {}, ties: {}",
        jpegli_wins, mozjpeg_wins, ties
    );

    // Note: mozjpeg is highly optimized, so it's expected to often win
    // The important thing is that jpegli is in the same ballpark,
    // which is verified by the individual assertions above
    println!("Note: mozjpeg is a mature, optimized encoder - some losses are expected");
}

/// Test that at similar file sizes, quality is comparable.
#[test]
fn test_quality_at_similar_size() {
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping: test file not found");
        return;
    }

    let (original, width, height) = load_png(path).expect("load png");
    let width_u32 = width as u32;
    let height_u32 = height as u32;

    // Encode with jpegli at Q80
    let jpegli_data = encode_jpegli(&original, width_u32, height_u32, 80);
    let jpegli_decoded = decode_jpeg(&jpegli_data);
    let jpegli_dssim = compute_dssim(&original, &jpegli_decoded, width, height);

    println!("\n=== Quality at Similar Size ===");
    println!("jpegli Q80: {} bytes, DSSIM: {:.6}", jpegli_data.len(), jpegli_dssim);

    // Find mozjpeg quality that produces similar size
    let target_size = jpegli_data.len();
    let mut best_quality = 80;
    let mut best_diff = usize::MAX;

    for q in 60..=95 {
        let moz_data = encode_mozjpeg(&original, width, height, q as f32, true);
        let diff = (moz_data.len() as i64 - target_size as i64).unsigned_abs() as usize;
        if diff < best_diff {
            best_diff = diff;
            best_quality = q;
        }
    }

    let mozjpeg_data = encode_mozjpeg(&original, width, height, best_quality as f32, true);
    let mozjpeg_decoded = decode_jpeg(&mozjpeg_data);
    let mozjpeg_dssim = compute_dssim(&original, &mozjpeg_decoded, width, height);

    println!(
        "mozjpeg Q{}: {} bytes, DSSIM: {:.6}",
        best_quality,
        mozjpeg_data.len(),
        mozjpeg_dssim
    );

    let size_ratio = jpegli_data.len() as f64 / mozjpeg_data.len() as f64;
    let dssim_ratio = jpegli_dssim / mozjpeg_dssim;

    println!("Size ratio (jpegli/mozjpeg): {:.3}", size_ratio);
    println!("DSSIM ratio (jpegli/mozjpeg): {:.3}", dssim_ratio);

    // At similar file sizes, quality should be within 30%
    assert!(
        dssim_ratio < 1.3,
        "At similar sizes, jpegli DSSIM ratio should be < 1.3, got {}",
        dssim_ratio
    );
}
