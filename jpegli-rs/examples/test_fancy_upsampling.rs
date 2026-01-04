//! Test to compare fancy upsampling vs box filter upsampling quality

use dssim::Dssim;
use rgb::RGBA8;

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

fn create_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 255 / width) as u8);
            rgb.push((y * 255 / height) as u8);
            rgb.push(((x + y) * 255 / (width + height)) as u8);
        }
    }
    rgb
}

fn create_checkerboard(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    let block_size = 16;
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / block_size) + (y / block_size)) % 2 == 0;
            let val = if checker { 200u8 } else { 50u8 };
            rgb.push(val);
            rgb.push(val + 30);
            rgb.push(val + 50);
        }
    }
    rgb
}

fn create_photo_like(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            // Simulate organic photo-like content with smooth variations
            let r = ((x as f32 / width as f32 * 6.28).sin() * 40.0 + 120.0) as u8;
            let g = ((y as f32 / height as f32 * 6.28).cos() * 50.0 + 100.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32 * 6.28).sin() * 30.0 + 90.0) as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }
    rgb
}

fn test_image(name: &str, rgb: &[u8], width: usize, height: usize, quality: u8) {
    // Encode with 4:2:0 chroma subsampling
    let jpeg_data = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .subsampling(jpegli::Subsampling::S420)
        .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
        .encode(rgb)
        .unwrap();

    // Decode with fancy upsampling (default)
    let decoded_fancy = jpegli::Decoder::new()
        .fancy_upsampling(true)
        .decode(&jpeg_data)
        .unwrap();

    // Decode with box filter upsampling
    let decoded_box = jpegli::Decoder::new()
        .fancy_upsampling(false)
        .decode(&jpeg_data)
        .unwrap();

    let dssim_fancy = compute_dssim(rgb, &decoded_fancy.data, width, height);
    let dssim_box = compute_dssim(rgb, &decoded_box.data, width, height);

    let improvement = ((dssim_box - dssim_fancy) / dssim_box) * 100.0;

    println!(
        "  {:<15} Q{}: DSSIM fancy={:.6}, box={:.6}, improvement={:>5.1}%",
        name, quality, dssim_fancy, dssim_box, improvement
    );
}

fn main() {
    let width = 512;
    let height = 512;

    println!("Testing fancy upsampling impact with 4:2:0 chroma subsampling");
    println!("(Lower DSSIM = better quality)\n");

    let gradient = create_gradient(width, height);
    let checkerboard = create_checkerboard(width, height);
    let photo_like = create_photo_like(width, height);

    for quality in [75, 85, 95] {
        println!("Quality {}:", quality);
        test_image("Gradient", &gradient, width, height, quality);
        test_image("Checkerboard", &checkerboard, width, height, quality);
        test_image("Photo-like", &photo_like, width, height, quality);
        println!();
    }

    println!("\nComparing 4:4:4 vs 4:2:0 at Q90 (gradient image):");

    // 4:4:4 (no chroma subsampling)
    let jpeg_444 = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .subsampling(jpegli::Subsampling::S444)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .encode(&gradient)
        .unwrap();

    let decoded_444 = jpegli::Decoder::new().decode(&jpeg_444).unwrap();
    let dssim_444 = compute_dssim(&gradient, &decoded_444.data, width, height);

    // 4:2:0 with fancy upsampling
    let jpeg_420 = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .subsampling(jpegli::Subsampling::S420)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .encode(&gradient)
        .unwrap();

    let decoded_420_fancy = jpegli::Decoder::new()
        .fancy_upsampling(true)
        .decode(&jpeg_420)
        .unwrap();

    let decoded_420_box = jpegli::Decoder::new()
        .fancy_upsampling(false)
        .decode(&jpeg_420)
        .unwrap();

    let dssim_420_fancy = compute_dssim(&gradient, &decoded_420_fancy.data, width, height);
    let dssim_420_box = compute_dssim(&gradient, &decoded_420_box.data, width, height);

    println!(
        "  4:4:4 (no subsampling):    {} bytes, DSSIM: {:.6}",
        jpeg_444.len(),
        dssim_444
    );
    println!(
        "  4:2:0 + fancy upsampling:  {} bytes, DSSIM: {:.6}",
        jpeg_420.len(),
        dssim_420_fancy
    );
    println!(
        "  4:2:0 + box filter:        {} bytes, DSSIM: {:.6}",
        jpeg_420.len(),
        dssim_420_box
    );
    println!(
        "  Size reduction: {:.1}%",
        (1.0 - jpeg_420.len() as f64 / jpeg_444.len() as f64) * 100.0
    );
    println!(
        "  Quality impact (fancy): {:.1}%",
        ((dssim_420_fancy - dssim_444) / dssim_444) * 100.0
    );
    println!(
        "  Quality impact (box):   {:.1}%",
        ((dssim_420_box - dssim_444) / dssim_444) * 100.0
    );
}
