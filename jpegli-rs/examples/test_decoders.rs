use jpegli::quant::Quality;
use jpegli::types::JpegMode;
use jpegli::{Encoder, PixelFormat};
use std::io::Cursor;
use std::process::Command;

fn main() {
    // Test case that fails: photo-like pattern at Q50
    let width = 128u32;
    let height = 96u32;

    let data: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let r = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8;
                let g = ((x.wrapping_mul(13) ^ y.wrapping_mul(23)) % 256) as u8;
                let b = ((x.wrapping_mul(11) ^ y.wrapping_mul(19)) % 256) as u8;
                [r, g, b]
            })
        })
        .collect();

    let jpeg_data = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(50.0))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("Encoding should succeed");

    println!("Encoded {} bytes of progressive JPEG", jpeg_data.len());
    std::fs::write("/tmp/test_prog.jpg", &jpeg_data).unwrap();

    // Test zune-jpeg
    let cursor = Cursor::new(&jpeg_data);
    match zune_jpeg::JpegDecoder::new(cursor).decode() {
        Ok(pixels) => println!("zune-jpeg: OK, {} pixels", pixels.len()),
        Err(e) => println!("zune-jpeg: FAIL - {:?}", e),
    }

    // Test mozjpeg decompress
    match mozjpeg::Decompress::new_mem(&jpeg_data) {
        Ok(d) => match d.rgb() {
            Ok(mut rgb) => match rgb.read_scanlines::<[u8; 3]>() {
                Ok(lines) => {
                    let pixel_count: usize = lines.len();
                    println!("mozjpeg: OK, {} RGB triplets", pixel_count);
                }
                Err(e) => println!("mozjpeg: decode FAIL - {:?}", e),
            },
            Err(e) => println!("mozjpeg: rgb conversion FAIL - {:?}", e),
        },
        Err(e) => println!("mozjpeg: init FAIL - {:?}", e),
    }

    // Test jpeg-decoder
    match decode_zune(&jpeg_data[..]) {
        Ok(pixels) => println!("jpeg-decoder: OK, {} pixels", pixels.len()),
        Err(e) => println!("jpeg-decoder: FAIL - {}", e),
    }

    // Also test djpeg command-line
    let output = Command::new("djpeg")
        .arg("-outfile")
        .arg("/tmp/test_prog.ppm")
        .arg("/tmp/test_prog.jpg")
        .output();
    match output {
        Ok(o) if o.status.success() => println!("djpeg: OK"),
        Ok(o) => println!("djpeg: FAIL - {}", String::from_utf8_lossy(&o.stderr)),
        Err(e) => println!("djpeg: not available - {}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
