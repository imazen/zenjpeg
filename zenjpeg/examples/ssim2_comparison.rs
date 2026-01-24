//! Compare SSIMULACRA2 quality scores between Rust jpegli and C++ jpegli FFI
//!
//! Usage:
//! ```bash
//! cargo run --release --example ssim2_comparison
//! ```
//!
//! IMPORTANT: Uses distance-based encoding for fair comparison.
//! C++ jpegli's `jpeg_set_quality()` uses 2 chroma tables, while
//! `jpegli_set_distance()` uses 3 tables matching Rust's behavior.

use enough::Unstoppable;
use fast_ssim2::{compute_frame_ssimulacra2, srgb_u8_to_linear, LinearRgbImage};
use zenjpeg::encoder::{
    ChromaSubsampling as JpegliSubsampling, EncoderConfig as JpegliEncoderConfig, PixelLayout,
    Quality,
};
use zenjpeg::types::Subsampling;
use jpegli_bench_utils::{
    ChromaSubsampling as BenchSubsampling, ColorMode, EncoderConfig, EncoderImpl, ImageData,
    ScanMode, SyntheticPattern,
};

/// Convert quality (0-100) to butteraugli distance.
/// Same formula as C++ jpegli_quality_to_distance.
fn quality_to_distance(q: f32) -> f32 {
    if q >= 100.0 {
        0.01
    } else if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
    }
}

fn create_test_image(width: u32, height: u32) -> ImageData {
    let pattern = SyntheticPattern::Complex;
    let img = pattern.generate(width, height);
    ImageData {
        name: format!("complex_{}x{}", width, height),
        pixels: img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect(),
        width: width as usize,
        height: height as usize,
    }
}

fn encode_rust(
    image: &ImageData,
    distance: f32,
    subsampling: Subsampling,
    progressive: bool,
) -> Vec<u8> {
    let sub = match subsampling {
        Subsampling::S444 => JpegliSubsampling::None,
        Subsampling::S422 => JpegliSubsampling::HalfHorizontal,
        Subsampling::S420 => JpegliSubsampling::Quarter,
        Subsampling::S440 => JpegliSubsampling::HalfVertical,
        _ => JpegliSubsampling::Quarter,
    };
    let config = JpegliEncoderConfig::ycbcr(Quality::ApproxButteraugli(distance), sub)
        .progressive(progressive)
        .optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(
            image.width as u32,
            image.height as u32,
            PixelLayout::Rgb8Srgb,
        )
        .expect("encoder setup");
    enc.push_packed(&image.pixels, Unstoppable).expect("push");
    enc.finish().expect("Rust encode failed")
}

fn encode_cpp_ffi(
    image: &ImageData,
    distance: f32,
    subsampling: Subsampling,
    progressive: bool,
) -> Vec<u8> {
    let config = EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(if progressive {
            ScanMode::Progressive
        } else {
            ScanMode::Baseline
        })
        .subsampling(match subsampling {
            Subsampling::S444 => BenchSubsampling::S444,
            Subsampling::S422 => BenchSubsampling::S422,
            Subsampling::S420 => BenchSubsampling::S420,
            Subsampling::S440 => BenchSubsampling::S440,
            _ => BenchSubsampling::S420,
        })
        .distance(distance); // Use distance for 3-table parity

    config.encode(image).expect("C++ FFI encode failed")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("JPEG decode failed")
}

/// Convert sRGB u8 bytes to LinearRgbImage (linear f32)
fn bytes_to_linear_rgb(bytes: &[u8], width: usize, height: usize) -> LinearRgbImage {
    let pixels: Vec<[f32; 3]> = bytes
        .chunks_exact(3)
        .map(|c| {
            [
                srgb_u8_to_linear(c[0]),
                srgb_u8_to_linear(c[1]),
                srgb_u8_to_linear(c[2]),
            ]
        })
        .collect();
    LinearRgbImage::new(pixels, width, height)
}

fn compute_ssim2(original: &ImageData, decoded: &[u8]) -> f64 {
    let orig_linear = bytes_to_linear_rgb(&original.pixels, original.width, original.height);
    let dec_linear = bytes_to_linear_rgb(decoded, original.width, original.height);

    compute_frame_ssimulacra2(orig_linear, dec_linear).expect("SSIMULACRA2 computation failed")
}

fn main() {
    println!("=== SSIMULACRA2 Quality Comparison: Rust vs C++ jpegli ===\n");

    // Test sizes: 512x512, 2K (2048x2048), 4K (4096x4096)
    let sizes = [
        (512, 512, "512x512"),
        (2048, 2048, "2K"),
        (4096, 4096, "4K"),
    ];

    let qualities = [75, 90, 95];
    let subsampling = Subsampling::S420;
    let progressive = true;

    println!(
        "{:<10} {:<8} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "Size", "Quality", "Rust Size", "C++ Size", "Size Δ%", "Rust SSIM2", "C++ SSIM2"
    );
    println!("{}", "-".repeat(82));

    for (width, height, size_name) in sizes {
        println!("\nGenerating {} image...", size_name);
        let image = create_test_image(width, height);

        for quality in qualities {
            // Convert quality to distance for fair comparison (both use 3 quant tables)
            let distance = quality_to_distance(quality as f32);

            // Encode with both
            let rust_jpeg = encode_rust(&image, distance, subsampling, progressive);
            let cpp_jpeg = encode_cpp_ffi(&image, distance, subsampling, progressive);

            // Decode both
            let rust_decoded = decode_jpeg(&rust_jpeg);
            let cpp_decoded = decode_jpeg(&cpp_jpeg);

            // Compute SSIM2 scores (comparing decoded to original)
            let rust_ssim2 = compute_ssim2(&image, &rust_decoded);
            let cpp_ssim2 = compute_ssim2(&image, &cpp_decoded);

            // Size difference
            let size_diff_pct =
                (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64 * 100.0;

            println!(
                "{:<10} {:<8} {:>12} {:>12} {:>+9.2}% {:>10.2} {:>10.2}",
                size_name,
                format!("q{}", quality),
                format!("{} B", rust_jpeg.len()),
                format!("{} B", cpp_jpeg.len()),
                size_diff_pct,
                rust_ssim2,
                cpp_ssim2
            );
        }
    }

    println!("\n=== Summary ===");
    println!("SSIMULACRA2: Higher is better (100 = identical to original)");
    println!("Size Δ%: Negative means Rust produces smaller files");
}
