//! Pareto curve comparison: jpegli vs mozjpeg vs adaptive hybrid
//!
//! **DEPRECATED**: Use `quality_compare` instead:
//!   cargo run --release --example quality_compare -- --pareto image.png
//!
//! Run with:
//! ```
//! cargo run --release --example pareto_comparison --features experimental-hybrid-trellis
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

    eprintln!(
        "Loading {} images from {}",
        files.len(),
        image_dir.display()
    );

    // Quality levels to test (Q40-Q80)
    let qualities = [40, 50, 60, 70, 75, 80];

    // CSV header
    println!("encoder,quality,image,width,height,bytes,bpp,dssim,ssim2,butteraugli,combined_score");

    for path in &files {
        if let Some(img) = load_image(path) {
            for &q in &qualities {
                // 1. jpegli baseline
                let jpegli_result = encode_jpegli(&img, q);
                print_result("jpegli", q, &img, &jpegli_result);

                // 2. mozjpeg
                let mozjpeg_result = encode_mozjpeg(&img, q);
                print_result("mozjpeg", q, &img, &mozjpeg_result);

                // 3. Adaptive hybrid (use hybrid if aq_mean > 0.25)
                let adaptive_result = encode_adaptive_hybrid(&img, q);
                print_result("adaptive", q, &img, &adaptive_result);
            }
        }
    }
}

struct ImageData {
    name: String,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    aq_mean: f32,
}

struct EncodingResult {
    bytes: usize,
    dssim: f64,
    ssim2: f64,
    butteraugli: f64,
}

impl EncodingResult {
    /// Combined score: normalize each metric and average with equal weights
    /// Lower is better for all (we invert SSIM2)
    fn combined_score(&self) -> f64 {
        // Normalize to similar scales:
        // DSSIM: typically 0.001-0.01, multiply by 100 to get 0.1-1.0
        // SSIM2: typically 60-90, invert: (100-ssim2)/100 to get 0.1-0.4
        // Butteraugli: typically 1-4, divide by 4 to get 0.25-1.0
        let norm_dssim = self.dssim * 100.0;
        let norm_ssim2 = (100.0 - self.ssim2) / 100.0;
        let norm_butter = self.butteraugli / 4.0;

        // Equal weights (1/3 each)
        (norm_dssim + norm_ssim2 + norm_butter) / 3.0
    }
}

fn print_result(encoder: &str, quality: u8, img: &ImageData, result: &EncodingResult) {
    let pixels = img.width * img.height;
    let bpp = 8.0 * result.bytes as f64 / pixels as f64;
    println!(
        "{},{},{},{},{},{},{:.4},{:.6},{:.2},{:.4},{:.6}",
        encoder,
        quality,
        img.name,
        img.width,
        img.height,
        result.bytes,
        bpp,
        result.dssim,
        result.ssim2,
        result.butteraugli,
        result.combined_score()
    );
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

    let width = info.width as usize;
    let height = info.height as usize;
    let pixels = buf[..info.buffer_size()].to_vec();

    // Compute AQ mean for adaptive decision
    let y_plane: Vec<f32> = pixels
        .chunks(3)
        .map(|c| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32)
        .collect();
    let aq_map = jpegli::adaptive_quant::compute_aq_strength_map(&y_plane, width, height, 8);
    let aq_mean = aq_map.mean();

    Some(ImageData {
        name: path.file_name()?.to_string_lossy().to_string(),
        pixels,
        width,
        height,
        aq_mean,
    })
}

fn encode_jpegli(img: &ImageData, quality: u8) -> EncodingResult {
    let jpeg_data = jpegli::Encoder::new()
        .width(img.width as u32)
        .height(img.height as u32)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
        .encode(&img.pixels)
        .expect("jpegli encode");

    let decoded = decode_jpeg(&jpeg_data);
    compute_metrics(
        &img.pixels,
        &decoded,
        img.width,
        img.height,
        jpeg_data.len(),
    )
}

fn encode_mozjpeg(img: &ImageData, quality: u8) -> EncodingResult {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(img.width, img.height);
    comp.set_quality(quality as f32);
    // NOTE: Do NOT use set_scan_optimization_mode - it breaks quality setting!
    // Use optimize_coding only
    comp.set_optimize_coding(true);

    let mut comp = comp.start_compress(Vec::new()).expect("mozjpeg start");
    comp.write_scanlines(&img.pixels).expect("mozjpeg write");
    let jpeg_data = comp.finish().expect("mozjpeg finish");

    let decoded = decode_jpeg(&jpeg_data);
    compute_metrics(
        &img.pixels,
        &decoded,
        img.width,
        img.height,
        jpeg_data.len(),
    )
}

#[cfg(feature = "experimental-hybrid-trellis")]
fn encode_adaptive_hybrid(img: &ImageData, quality: u8) -> EncodingResult {
    use jpegli::hybrid_config::{should_use_hybrid, HybridConfig};

    let mut encoder = jpegli::Encoder::new()
        .width(img.width as u32)
        .height(img.height as u32)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32));

    // Adaptive: use hybrid only for complex images
    if should_use_hybrid(img.aq_mean) {
        encoder = encoder.hybrid_config(HybridConfig::default());
    }

    let jpeg_data = encoder.encode(&img.pixels).expect("adaptive encode");
    let decoded = decode_jpeg(&jpeg_data);
    compute_metrics(
        &img.pixels,
        &decoded,
        img.width,
        img.height,
        jpeg_data.len(),
    )
}

#[cfg(not(feature = "experimental-hybrid-trellis"))]
fn encode_adaptive_hybrid(img: &ImageData, quality: u8) -> EncodingResult {
    // Fallback to jpegli if experimental-hybrid-trellis not enabled
    encode_jpegli(img, quality)
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
    decoder.decode().expect("decode")
}

fn compute_metrics(
    original: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
    bytes: usize,
) -> EncodingResult {
    EncodingResult {
        bytes,
        dssim: compute_dssim(original, decoded, width, height),
        ssim2: compute_ssim2(original, decoded, width, height),
        butteraugli: compute_butteraugli(original, decoded, width, height),
    }
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

fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

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

    let decoded_rgb = Rgb::new(
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

    compute_frame_ssimulacra2(orig_rgb, decoded_rgb).unwrap_or(0.0)
}

fn compute_butteraugli(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use butteraugli::{compute_butteraugli, ButteraugliParams};

    let params = ButteraugliParams::default();
    match compute_butteraugli(original, decoded, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => 99.0,
    }
}
