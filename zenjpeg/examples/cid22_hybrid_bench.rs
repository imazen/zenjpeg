//! Quick CID22 hybrid vs jpegli comparison using butteraugli

use enough::Unstoppable;
use zenjpeg::encode::trellis::{adaptive_config, detect_image_type, ImageType};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn main() {
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";

    // Get all directories with original.png
    let mut images: Vec<_> = std::fs::read_dir(base_dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("original.png").exists())
        .map(|e| e.path().join("original.png"))
        .collect();
    images.sort();

    if images.is_empty() {
        eprintln!("No CID22 images found in {}", base_dir);
        return;
    }

    println!("=== CID22 Hybrid vs Jpegli Comparison (Butteraugli) ===\n");
    println!(
        "{:>30}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Image",
        "Mean",
        "CV",
        "Type",
        "JpegliB",
        "HybridB",
        "ΔSize%",
        "JpegliBA",
        "HybridBA",
        "ΔButter%"
    );
    println!(
        "{:-<30}  {:-<6}  {:-<6}  {:-<8}  {:-<8}  {:-<8}  {:-<8}  {:-<8}  {:-<8}  {:-<8}",
        "", "", "", "", "", "", "", "", "", ""
    );

    let mut total_jpegli_bytes = 0usize;
    let mut total_hybrid_bytes = 0usize;
    let mut total_jpegli_butter = 0.0f64;
    let mut total_hybrid_butter = 0.0f64;
    let mut count = 0;

    for img_path in images.iter().take(20) {
        // Limit to 20 for speed
        let img = match ImageData::from_path(img_path) {
            Some(i) => i,
            None => continue,
        };

        let name = img_path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy();

        // Compute AQ stats for adaptive config
        let y_plane = extract_y_plane(&img);
        let aq_map =
            match zenjpeg::quant::aq::compute_aq_strength_map(&y_plane, img.width, img.height, 1) {
                Ok(m) => m,
                Err(_) => continue,
            };
        let (_, _, aq_mean, aq_std) = aq_map.stats();

        // Jpegli baseline (no trellis)
        let jpegli_config =
            EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).optimize_huffman(true);
        let jpegli_jpeg = match encode_image(&jpegli_config, &img) {
            Some(j) => j,
            None => continue,
        };
        let jpegli_bytes = jpegli_jpeg.len();

        // Hybrid with adaptive config
        let hybrid = adaptive_config(aq_mean, aq_std);
        let hybrid_config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .optimize_huffman(true)
            .hybrid_config(hybrid);
        let hybrid_jpeg = match encode_image(&hybrid_config, &img) {
            Some(j) => j,
            None => continue,
        };
        let hybrid_bytes = hybrid_jpeg.len();

        // Compute butteraugli
        let orig_rgb = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
        let jpegli_decoded: RgbImage = match decode_jpeg_to_rgb(&jpegli_jpeg) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let hybrid_decoded: RgbImage = match decode_jpeg_to_rgb(&hybrid_jpeg) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let jpegli_butter = QualityMetrics::butteraugli(orig_rgb.as_ref(), jpegli_decoded.as_ref());
        let hybrid_butter = QualityMetrics::butteraugli(orig_rgb.as_ref(), hybrid_decoded.as_ref());

        let size_delta = (hybrid_bytes as f64 / jpegli_bytes as f64 - 1.0) * 100.0;
        let butter_delta = (hybrid_butter / jpegli_butter - 1.0) * 100.0;

        let cv = if aq_mean > 0.001 {
            aq_std / aq_mean
        } else {
            0.0
        };
        let img_type = detect_image_type(aq_mean, aq_std);
        let type_str = match img_type {
            ImageType::Photo => "Photo",
            ImageType::Screenshot => "Screen",
            ImageType::Mixed => "Mixed",
        };

        println!(
            "{:>30}  {:>6.3}  {:>6.2}  {:>8}  {:>8}  {:>8}  {:>+8.1}  {:>8.3}  {:>8.3}  {:>+8.1}",
            &name[..name.len().min(30)],
            aq_mean,
            cv,
            type_str,
            jpegli_bytes,
            hybrid_bytes,
            size_delta,
            jpegli_butter,
            hybrid_butter,
            butter_delta
        );

        total_jpegli_bytes += jpegli_bytes;
        total_hybrid_bytes += hybrid_bytes;
        total_jpegli_butter += jpegli_butter;
        total_hybrid_butter += hybrid_butter;
        count += 1;
    }

    if count > 0 {
        let avg_size_delta = (total_hybrid_bytes as f64 / total_jpegli_bytes as f64 - 1.0) * 100.0;
        let avg_butter_delta = (total_hybrid_butter / total_jpegli_butter - 1.0) * 100.0;

        println!(
            "{:-<30}  {:-<6}  {:-<6}  {:-<8}  {:-<8}  {:-<8}  {:-<8}  {:-<8}  {:-<8}  {:-<8}",
            "", "", "", "", "", "", "", "", "", ""
        );
        println!(
            "{:>30}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>+8.1}  {:>8.3}  {:>8.3}  {:>+8.1}",
            "AVERAGE",
            "",
            "",
            "",
            total_jpegli_bytes,
            total_hybrid_bytes,
            avg_size_delta,
            total_jpegli_butter / count as f64,
            total_hybrid_butter / count as f64,
            avg_butter_delta
        );
        println!("\n{} images processed", count);
    }
}

fn extract_y_plane(img: &ImageData) -> Vec<f32> {
    let mut y = Vec::with_capacity(img.width * img.height);
    for chunk in img.pixels.chunks(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        y.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    y
}

fn encode_image(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut encoder = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    encoder.push_packed(&img.pixels, Unstoppable).ok()?;
    encoder.finish().ok()
}
