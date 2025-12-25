//! Trace butteraugli intermediate values for debugging.

use butteraugli::{compute_butteraugli, ButteraugliParams};
use std::fs;
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    
    let (width, height) = (info.width as usize, info.height as usize);
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };
    Some((rgb, width, height))
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::io::Cursor;
    let mut output = Vec::new();
    let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
    comp.set_size(width as usize, height as usize);
    comp.set_quality(quality as f32);
    let mut started = comp.start_compress(Cursor::new(&mut output)).expect("start");
    let row_stride = width as usize * 3;
    for row in rgb.chunks(row_stride) {
        started.write_scanlines(row).expect("write");
    }
    started.finish().expect("finish");
    output
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().unwrap_or_default()
}

fn main() {
    let path = Path::new("/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png");
    
    if let Some((original, width, height)) = load_png(path) {
        println!("Image: {}x{}", width, height);
        
        // Test various qualities
        for quality in [50, 70, 85, 90] {
            let jpeg_data = encode_jpeg(&original, width as u32, height as u32, quality);
            let decoded = decode_jpeg(&jpeg_data);
            
            if decoded.len() != original.len() {
                println!("Q{}: size mismatch", quality);
                continue;
            }
            
            // Compute pixel-level stats
            let mut max_diff = 0u8;
            let mut sum_diff = 0u64;
            for i in 0..original.len() {
                let diff = (original[i] as i16 - decoded[i] as i16).unsigned_abs() as u8;
                max_diff = max_diff.max(diff);
                sum_diff += diff as u64;
            }
            let mean_diff = sum_diff as f64 / original.len() as f64;
            
            // Compute butteraugli
            let params = ButteraugliParams::default();
            let result = compute_butteraugli(&original, &decoded, width, height, &params);
            
            // Analyze diffmap
            let diffmap = result.diffmap.as_ref().unwrap();
            let mut max_diffmap = 0.0f32;
            let mut sum_diffmap = 0.0f64;
            for y in 0..diffmap.height() {
                for x in 0..diffmap.width() {
                    let v = diffmap.get(x, y);
                    max_diffmap = max_diffmap.max(v);
                    sum_diffmap += v as f64;
                }
            }
            let mean_diffmap = sum_diffmap / (diffmap.width() * diffmap.height()) as f64;
            
            println!("Q{}: butteraugli={:.4}, max_diffmap={:.4}, mean_diffmap={:.4}, pixel_max={}, pixel_mean={:.2}",
                quality, result.score, max_diffmap, mean_diffmap, max_diff, mean_diff);
        }
    } else {
        println!("Could not load image");
    }
}
