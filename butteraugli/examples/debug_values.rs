//! Debug butteraugli intermediate values

use butteraugli::{compute_butteraugli, ButteraugliParams};

fn main() {
    // Load the actual test image and encode/decode
    let test_image = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";
    
    if let Ok(file) = std::fs::File::open(test_image) {
        let decoder = png::Decoder::new(file);
        if let Ok(mut reader) = decoder.read_info() {
            let mut buf = vec![0; reader.output_buffer_size()];
            if let Ok(info) = reader.next_frame(&mut buf) {
                let (width, height) = (info.width as usize, info.height as usize);
                let rgb: Vec<u8> = buf[..width * height * 3].to_vec();
                
                println!("Image: {}x{}", width, height);
                
                // Encode at Q90 with mozjpeg
                let jpeg_data = encode_jpeg(&rgb, width as u32, height as u32, 90);
                let decoded = decode_jpeg(&jpeg_data);
                
                if decoded.len() == rgb.len() {
                    // Compute pixel-level difference
                    let mut max_diff = 0u8;
                    let mut total_diff = 0u64;
                    for i in 0..rgb.len() {
                        let diff = (rgb[i] as i16 - decoded[i] as i16).abs() as u8;
                        max_diff = max_diff.max(diff);
                        total_diff += diff as u64;
                    }
                    println!("Pixel diff: max={}, mean={:.2}", max_diff, total_diff as f64 / rgb.len() as f64);
                    
                    // Compute butteraugli
                    let params = ButteraugliParams::default();
                    let result = compute_butteraugli(&rgb, &decoded, width, height, &params);
                    
                    println!("Butteraugli score: {:.4}", result.score);
                }
            }
        }
    } else {
        println!("Could not open test image");
    }
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::io::Cursor;
    let mut output = Vec::new();
    let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
    comp.set_size(width as usize, height as usize);
    comp.set_quality(quality as f32);
    let mut started = comp.start_compress(Cursor::new(&mut output)).expect("start compress");
    let row_stride = width as usize * 3;
    for row in rgb.chunks(row_stride) {
        started.write_scanlines(row).expect("write scanline");
    }
    started.finish().expect("finish compress");
    output
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().unwrap_or_default()
}
