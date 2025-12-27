use jpegli::{
    decode::Decoder,
    encode::Encoder,
    types::{JpegMode, PixelFormat, Subsampling},
    Quality,
};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .mode(JpegMode::Progressive);

    let encoded = encoder.encode(&pixels).expect("encode");
    println!("Encoded {} bytes", encoded.len());

    // Test with OUR decoder
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    match decoder.decode(&encoded) {
        Ok(img) => println!("jpegli-rs decoder: OK ({}x{})", img.width, img.height),
        Err(e) => println!("jpegli-rs decoder: FAIL - {:?}", e),
    }
}
