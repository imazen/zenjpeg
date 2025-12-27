//! Compare jpegli and mozjpeg quality/size at various quality levels.
//!
//! Uses DSSIM to measure perceptual quality against the original image.

use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::path::Path;

fn load_png(path: &Path) -> (Vec<RGBA8>, usize, usize) {
    let file = fs::File::open(path).expect("Failed to open PNG");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let (width, height) = (info.width as usize, info.height as usize);

    // Convert to RGBA8
    let pixels: Vec<RGBA8> = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3]
            .chunks(3)
            .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
            .collect(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .map(|c| RGBA8::new(c[0], c[1], c[2], c[3]))
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .map(|&g| RGBA8::new(g, g, g, 255))
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .map(|c| RGBA8::new(c[0], c[0], c[0], c[1]))
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    (pixels, width, height)
}

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(original: &[RGBA8], decoded: &[RGBA8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let orig = attr
        .create_image_rgba(original, width, height)
        .expect("Failed to create dssim image");
    let comp = attr
        .create_image_rgba(decoded, width, height)
        .expect("Failed to create dssim image");
    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn encode_jpegli(pixels: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality.into()))
        .encode(pixels)
        .expect("jpegli encoding failed")
}

fn encode_mozjpeg(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: f32,
    use_444: bool,
) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality);

    if use_444 {
        // Set Cb and Cr to 1x1 sampling (4:4:4, no chroma subsampling)
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));
    }

    let mut started = comp
        .start_compress(Vec::new())
        .expect("mozjpeg start error");

    let row_stride = width * 3;
    for y in 0..height {
        let row_start = y * row_stride;
        let row = &pixels[row_start..row_start + row_stride];
        let _ = started.write_scanlines(row);
    }

    started.finish().expect("mozjpeg finish error")
}

fn decode_jpeg_to_rgb(data: &[u8]) -> (Vec<u8>, u32, u32) {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    let pixels = decoder.decode().expect("JPEG decode failed");
    let info = decoder.info().unwrap();
    (pixels, info.width as u32, info.height as u32)
}

fn main() {
    let png_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/lilith/work/jpegli-rs/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png".to_string()
    });

    let (original, width, height) = load_png(Path::new(&png_path));
    let width_u32 = width as u32;
    let height_u32 = height as u32;

    // Convert RGBA8 to RGB for encoding
    let rgb_pixels: Vec<u8> = original.iter().flat_map(|p| [p.r, p.g, p.b]).collect();

    println!("Image: {} ({}x{})", png_path, width, height);
    println!();

    // First comparison: jpegli (4:4:4) vs mozjpeg (4:4:4) - same subsampling
    println!("=== Comparison 1: Both using 4:4:4 subsampling (no chroma subsampling) ===");
    println!(
        "{:>7} {:>12} {:>14} {:>12} {:>14}",
        "Quality", "jpegli Size", "mozjpeg444 Size", "jpegli DSSIM", "mozjpeg444 DSSIM"
    );
    println!("{}", "-".repeat(75));

    for quality in [60, 70, 75, 80, 85, 90, 95] {
        let jpegli_data = encode_jpegli(&rgb_pixels, width_u32, height_u32, quality);
        let mozjpeg444_data = encode_mozjpeg(&rgb_pixels, width, height, quality as f32, true);

        let (jpegli_decoded, _, _) = decode_jpeg_to_rgb(&jpegli_data);
        let (mozjpeg444_decoded, _, _) = decode_jpeg_to_rgb(&mozjpeg444_data);

        let jpegli_rgba = rgb_to_rgba(&jpegli_decoded);
        let mozjpeg444_rgba = rgb_to_rgba(&mozjpeg444_decoded);

        let jpegli_dssim = compute_dssim(&original, &jpegli_rgba, width, height);
        let mozjpeg444_dssim = compute_dssim(&original, &mozjpeg444_rgba, width, height);

        println!(
            "{:>7} {:>12} {:>14} {:>12.6} {:>14.6}",
            quality,
            jpegli_data.len(),
            mozjpeg444_data.len(),
            jpegli_dssim,
            mozjpeg444_dssim
        );
    }

    println!();
    println!("=== Comparison 2: jpegli (4:4:4) vs mozjpeg (default 4:2:0) ===");
    println!(
        "{:>7} {:>12} {:>14} {:>12} {:>14}",
        "Quality", "jpegli Size", "mozjpeg420 Size", "jpegli DSSIM", "mozjpeg420 DSSIM"
    );
    println!("{}", "-".repeat(75));

    for quality in [60, 70, 75, 80, 85, 90, 95] {
        let jpegli_data = encode_jpegli(&rgb_pixels, width_u32, height_u32, quality);
        let mozjpeg420_data = encode_mozjpeg(&rgb_pixels, width, height, quality as f32, false);

        let (jpegli_decoded, _, _) = decode_jpeg_to_rgb(&jpegli_data);
        let (mozjpeg420_decoded, _, _) = decode_jpeg_to_rgb(&mozjpeg420_data);

        let jpegli_rgba = rgb_to_rgba(&jpegli_decoded);
        let mozjpeg420_rgba = rgb_to_rgba(&mozjpeg420_decoded);

        let jpegli_dssim = compute_dssim(&original, &jpegli_rgba, width, height);
        let mozjpeg420_dssim = compute_dssim(&original, &mozjpeg420_rgba, width, height);

        println!(
            "{:>7} {:>12} {:>14} {:>12.6} {:>14.6}",
            quality,
            jpegli_data.len(),
            mozjpeg420_data.len(),
            jpegli_dssim,
            mozjpeg420_dssim
        );
    }

    println!();
    println!("Note: DSSIM closer to 0 is better quality.");
    println!("      Compare DSSIM at similar file sizes to judge quality/size efficiency.");
}
