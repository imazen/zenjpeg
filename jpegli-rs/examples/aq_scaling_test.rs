//! Test AQ scaling to match hybrid file sizes with baseline jpegli.
//!
//! Run with:
//! ```
//! cargo run --release --example aq_scaling_test --features experimental-hybrid-trellis
//! ```

use std::env;
use std::path::PathBuf;

fn main() {
    let image_dir = PathBuf::from(
        env::args()
            .nth(1)
            .unwrap_or_else(|| "/home/lilith/work/codec-eval/codec-corpus/kodak".to_string()),
    );

    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    // Collect PNG files
    let mut files: Vec<PathBuf> = std::fs::read_dir(&image_dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();
    files.truncate(max_files);

    if files.is_empty() {
        eprintln!("No PNG files found in {}", image_dir.display());
        return;
    }

    eprintln!("Testing AQ scaling on {} images\n", files.len());

    // Quality levels to test
    let qualities = [50, 70, 80];

    println!("image,quality,encoder,bytes,bpp,dssim,size_vs_baseline");

    for path in &files {
        if let Some(img) = load_image(path) {
            for &q in &qualities {
                // 1. Baseline jpegli (no hybrid)
                let baseline = encode_baseline(&img, q);
                println!(
                    "{},{},baseline,{},{:.4},{:.6},0.0%",
                    img.name, q, baseline.bytes, baseline.bpp, baseline.dssim
                );

                // 2. Hybrid without scaling (expect ~16% larger)
                let hybrid_raw = encode_hybrid(&img, q, None);
                let size_diff_raw = 100.0 * (hybrid_raw.bytes as f64 - baseline.bytes as f64)
                    / baseline.bytes as f64;
                println!(
                    "{},{},hybrid_raw,{},{:.4},{:.6},{:+.1}%",
                    img.name, q, hybrid_raw.bytes, hybrid_raw.bpp, hybrid_raw.dssim, size_diff_raw
                );

                // 3. Hybrid with AQ scaling to match baseline size
                let hybrid_scaled = encode_hybrid(&img, q, Some(size_diff_raw as f32));
                let size_diff_scaled = 100.0 * (hybrid_scaled.bytes as f64 - baseline.bytes as f64)
                    / baseline.bytes as f64;
                println!(
                    "{},{},hybrid_scaled,{},{:.4},{:.6},{:+.1}%",
                    img.name,
                    q,
                    hybrid_scaled.bytes,
                    hybrid_scaled.bpp,
                    hybrid_scaled.dssim,
                    size_diff_scaled
                );

                // 4. Summary line
                let quality_gain = 100.0 * (baseline.dssim - hybrid_scaled.dssim) / baseline.dssim;
                eprintln!(
                    "  {} Q{}: baseline={} bytes, scaled={} bytes ({:+.1}%), quality gain: {:.1}%",
                    img.name,
                    q,
                    baseline.bytes,
                    hybrid_scaled.bytes,
                    size_diff_scaled,
                    quality_gain
                );
            }
            eprintln!();
        }
    }
}

struct ImageData {
    name: String,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

struct EncodingResult {
    bytes: usize,
    bpp: f64,
    dssim: f64,
}

fn load_image(path: &PathBuf) -> Option<ImageData> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    if info.color_type != png::ColorType::Rgb {
        return None;
    }

    Some(ImageData {
        name: path.file_name()?.to_string_lossy().to_string(),
        pixels: buf[..info.buffer_size()].to_vec(),
        width: info.width as usize,
        height: info.height as usize,
    })
}

fn encode_baseline(img: &ImageData, quality: u8) -> EncodingResult {
    let jpeg_data = jpegli::Encoder::new()
        .width(img.width as u32)
        .height(img.height as u32)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
        .encode(&img.pixels)
        .expect("baseline encode");

    let decoded = decode_jpeg(&jpeg_data);
    let dssim = compute_dssim(&img.pixels, &decoded, img.width, img.height);
    let bpp = 8.0 * jpeg_data.len() as f64 / (img.width * img.height) as f64;

    EncodingResult {
        bytes: jpeg_data.len(),
        bpp,
        dssim,
    }
}

#[cfg(feature = "experimental-hybrid-trellis")]
fn encode_hybrid(
    img: &ImageData,
    quality: u8,
    target_size_reduction: Option<f32>,
) -> EncodingResult {
    use jpegli::adaptive_quant::compute_aq_strength_map;
    use jpegli::hybrid_config::HybridConfig;

    // Extract Y plane for AQ computation
    let y_plane: Vec<f32> = img
        .pixels
        .chunks(3)
        .map(|c| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32)
        .collect();

    // Compute AQ map
    let mut aq_map = compute_aq_strength_map(&y_plane, img.width, img.height, 8);
    let original_mean = aq_map.mean();

    // Apply scaling if requested
    if let Some(size_reduction) = target_size_reduction {
        let scale = aq_map.scale_for_size_reduction(size_reduction);
        aq_map.scale(scale);
        eprintln!(
            "    AQ scaling: mean {:.3} -> {:.3} (scale={:.2}x) to reduce size by {:.1}%",
            original_mean,
            aq_map.mean(),
            scale,
            size_reduction
        );
    }

    // Encode with hybrid config and custom AQ map
    let jpeg_data = jpegli::Encoder::new()
        .width(img.width as u32)
        .height(img.height as u32)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
        .hybrid_config(HybridConfig::default())
        .aq_map(aq_map)
        .encode(&img.pixels)
        .expect("hybrid encode");

    let decoded = decode_jpeg(&jpeg_data);
    let dssim = compute_dssim(&img.pixels, &decoded, img.width, img.height);
    let bpp = 8.0 * jpeg_data.len() as f64 / (img.width * img.height) as f64;

    EncodingResult {
        bytes: jpeg_data.len(),
        bpp,
        dssim,
    }
}

#[cfg(not(feature = "experimental-hybrid-trellis"))]
fn encode_hybrid(
    img: &ImageData,
    quality: u8,
    _target_size_reduction: Option<f32>,
) -> EncodingResult {
    encode_baseline(img, quality)
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
    decoder.decode().expect("decode")
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;

    let attr = Dssim::new();

    let orig_rgba: Vec<rgb::RGBA<u8>> = original
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr
        .create_image_rgba(&decoded_rgba, width, height)
        .unwrap();

    let (dssim, _) = attr.compare(&orig_img, decoded_img);
    dssim.into()
}
