//! Compare XYB output quality and size against C jpegli.

use enough::Unstoppable;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::process::Command;

fn main() {
    let images = [
        "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/5.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/13.png",
    ];
    let qualities = [70, 80, 90];

    println!("XYB Parity Test: Rust vs C jpegli");
    println!("{}", "=".repeat(90));
    println!();
    println!(
        "{:<20} {:<5} {:<10} {:<10} {:<8} {:<10} {:<10} {:<8}",
        "Image", "Q", "C bytes", "Rust bytes", "Size %", "C DSSIM", "Rust DSSIM", "Δ DSSIM"
    );
    println!("{}", "-".repeat(90));

    for img_path in &images {
        let img_name = std::path::Path::new(img_path)
            .file_name()
            .unwrap()
            .to_string_lossy();

        // Load image once
        let file = std::fs::File::open(img_path).expect("open");
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info().expect("info");
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("decode");
        let pixels = &buf[..info.buffer_size()];
        let width = info.width;
        let height = info.height;

        for &q in &qualities {
            // C jpegli XYB
            let cpp_path = format!("/tmp/cpp_xyb_{}_{}.jpg", img_name, q);
            Command::new("cjpegli")
                .args(&[
                    img_path.to_string(),
                    cpp_path.clone(),
                    "-q".to_string(),
                    q.to_string(),
                    "--xyb".to_string(),
                ])
                .output()
                .expect("cjpegli");
            let cpp_bytes = std::fs::metadata(&cpp_path).map(|m| m.len()).unwrap_or(0);

            // Rust XYB with hybrid trellis (matches C jpegli's AQ)
            #[cfg(feature = "experimental-hybrid-trellis")]
            let rust_jpeg = {
                use jpegli::hybrid::HybridConfig;
                let config = EncoderConfig::new(90.0, ChromaSubsampling::Quarter)
                    .quality(q as f32)
                    .xyb()
                    .hybrid_config(HybridConfig::default());
                let mut enc = config
                    .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                    .expect("encoder setup");
                enc.push_packed(pixels, Unstoppable).expect("push");
                enc.finish().expect("encode")
            };
            #[cfg(not(feature = "experimental-hybrid-trellis"))]
            let rust_jpeg = {
                let config = EncoderConfig::new(q as f32, ChromaSubsampling::Quarter).xyb();
                let mut enc = config
                    .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                    .expect("encoder setup");
                enc.push_packed(pixels, Unstoppable).expect("push");
                enc.finish().expect("encode")
            };
            let rust_bytes = rust_jpeg.len();
            let rust_path = format!("/tmp/rust_xyb_{}_{}.jpg", img_name, q);
            std::fs::write(&rust_path, &rust_jpeg).expect("write");

            // Decode and compute DSSIM
            let cpp_decoded = decode_jpeg(&std::fs::read(&cpp_path).unwrap());
            let rust_decoded = decode_jpeg(&rust_jpeg);

            let cpp_dssim = compute_dssim(pixels, &cpp_decoded, width as usize, height as usize);
            let rust_dssim = compute_dssim(pixels, &rust_decoded, width as usize, height as usize);

            let size_diff = 100.0 * (rust_bytes as f64 - cpp_bytes as f64) / cpp_bytes as f64;
            let dssim_diff = rust_dssim - cpp_dssim;

            println!(
                "{:<20} {:<5} {:<10} {:<10} {:+.1}%{:<3} {:<10.6} {:<10.6} {:+.6}",
                img_name,
                q,
                cpp_bytes,
                rust_bytes,
                size_diff,
                "",
                cpp_dssim,
                rust_dssim,
                dssim_diff
            );
        }
    }
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
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
