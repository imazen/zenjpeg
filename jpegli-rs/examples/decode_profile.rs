use jpegli::{Decoder, Encoder, PixelFormat, Quality};
use std::time::Instant;

fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = 128;
        }
    }
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .encode(&data)
        .unwrap()
}

fn main() {
    let jpeg_data = create_test_jpeg(2048, 2048);
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    
    // Warmup
    for _ in 0..3 {
        let _ = decoder.decode(&jpeg_data);
    }
    
    let start = Instant::now();
    for _ in 0..10 {
        let _ = decoder.decode(&jpeg_data);
    }
    let elapsed = start.elapsed();
    
    let pixels = 2048.0 * 2048.0;
    let mpps = (pixels * 10.0) / elapsed.as_secs_f64() / 1_000_000.0;
    eprintln!("Decoded 10x 2048x2048 in {:?} ({:.1} MP/s)", elapsed, mpps);
}
