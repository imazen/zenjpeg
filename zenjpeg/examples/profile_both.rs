use std::hint::black_box;
use std::io::Cursor;
use std::time::Instant;

fn main() {
    // Create test image
    let width = 2048u32;
    let height = 2048u32;
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            data[idx] = ((x * 255) / width as usize) as u8;
            data[idx + 1] = ((y * 255) / height as usize) as u8;
            data[idx + 2] = (((x + y) * 128) / (width + height) as usize) as u8;
        }
    }

    use enough::Unstoppable;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();
    eprintln!("JPEG size: {} bytes", jpeg.len());

    let iterations = 50;

    // Warmup
    for _ in 0..5 {
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::colorspace::ColorSpace;
        use zune_jpeg::zune_core::options::DecoderOptions;
        let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let cursor = Cursor::new(jpeg.as_slice());
        let mut dec = JpegDecoder::new_with_options(cursor, options);
        let _ = dec.decode().unwrap();

        use zenjpeg::decode::Decoder;
        use zenjpeg::decoder::PixelFormat;
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let _ = decoder.decode(&jpeg, Unstoppable).unwrap();
    }

    // Profile zune-jpeg
    eprintln!("=== Profiling zune-jpeg ({} iterations) ===", iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::colorspace::ColorSpace;
        use zune_jpeg::zune_core::options::DecoderOptions;
        let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let cursor = Cursor::new(black_box(jpeg.as_slice()));
        let mut dec = JpegDecoder::new_with_options(cursor, options);
        let _ = black_box(dec.decode().unwrap());
    }
    let zune_elapsed = start.elapsed();
    let zune_per_iter = zune_elapsed / iterations;
    eprintln!(
        "  Total: {:?}, Per-iteration: {:?}",
        zune_elapsed, zune_per_iter
    );

    // Profile zenjpeg
    eprintln!("=== Profiling zenjpeg ({} iterations) ===", iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        use zenjpeg::decode::Decoder;
        use zenjpeg::decoder::PixelFormat;
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let _ = black_box(decoder.decode(&jpeg, Unstoppable).unwrap());
    }
    let jpegli_elapsed = start.elapsed();
    let jpegli_per_iter = jpegli_elapsed / iterations;
    eprintln!(
        "  Total: {:?}, Per-iteration: {:?}",
        jpegli_elapsed, jpegli_per_iter
    );

    eprintln!("\n=== Summary ===");
    eprintln!("zune-jpeg:  {:?}/decode", zune_per_iter);
    eprintln!("zenjpeg:  {:?}/decode", jpegli_per_iter);
    let ratio = jpegli_elapsed.as_secs_f64() / zune_elapsed.as_secs_f64();
    eprintln!("Ratio: {:.2}x (zenjpeg / zune-jpeg)", ratio);
}
