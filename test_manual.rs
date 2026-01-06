// Test at HEAD with explicit settings to match fe0baee behavior
use jpegli::{Encoder, PixelFormat, Quality, types::Subsampling, ChromaConversion};
use jpegli::decode::Decoder;

fn main() {
    let png_data = std::fs::read("/home/lilith/work/jpegli-rs-simd/jpegli-rs/tests/images/1.png").expect("read png");
    let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()].chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => panic!("Unsupported"),
    };
    let width = info.width;
    let height = info.height;
    
    println!("Image: {}x{}", width, height);
    
    // Encode with EXPLICIT Intrinsic setting
    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(70.0))
        .subsampling(Subsampling::S420)
        .chroma_conversion(ChromaConversion::Intrinsic) // Explicit, not Auto
        .optimize_huffman(true)
        .encode(&rgb)
        .expect("encode");
    
    println!("JPEG size: {} bytes (expected around 45699)", jpeg.len());
    
    // Decode and compute butteraugli
    let decoded = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg)
        .expect("decode");
    
    let params = butteraugli::ButteraugliParams::default();
    let bfly = butteraugli::compute_butteraugli(
        &rgb, &decoded.data, width as usize, height as usize, &params
    ).expect("bfly").score;
    
    println!("Butteraugli: {:.8} (expected 3.76503134)", bfly);
}
