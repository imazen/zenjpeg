//! Find quality level where zenjpeg matches C mozjpeg's DSSIM

use dssim::Dssim;
use rgb::RGBA8;
use zenjpeg::{Encoder, Quality};

fn main() {
    let width = 256;
    let height = 256;

    // Create test image
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let r = ((fx * 0.7 + fy * 0.3) * 255.0) as u8;
            let g = ((fy * 0.8) * 200.0 + 30.0) as u8;
            let b = (((fx + fy) * 0.5).sin() * 64.0 + 128.0) as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }

    // Reference: C mozjpeg at Q75 produces DSSIM ~0.000886
    let target_dssim = 0.000886;

    println!("Finding quality level to match C mozjpeg DSSIM {:.6}", target_dssim);
    println!();

    // Try different quality levels
    for q in (40..=85).step_by(5) {
        let jpeg = Encoder::new()
            .quality(Quality::Standard(q))
            .encode_rgb(&rgb, width, height)
            .expect("encoding failed");

        // Decode
        let decoded = jpeg_decoder::Decoder::new(&jpeg[..])
            .decode()
            .expect("decode failed");

        let dssim = compute_dssim(&rgb, &decoded, width, height);

        let status = if dssim <= target_dssim { "✓ matches" } else { "" };
        println!("Q{:2}: {} bytes, DSSIM={:.6} {}", q, jpeg.len(), dssim, status);
    }
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();

    let orig_rgba: Vec<RGBA8> = original.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let dec_rgba: Vec<RGBA8> = decoded.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let dec_img = attr.create_image_rgba(&dec_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, dec_img);
    dssim.into()
}
