use zenjpeg::{Encoder, Quality};

fn main() {
    // Simple 16x16 gradient
    let width = 16;
    let height = 16;
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = 128u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    for q in [30, 50, 70, 95] {
        let encoder = Encoder::new().quality(Quality::Standard(q));
        let jpeg_data = encoder.encode_rgb(&pixels, width, height).unwrap();
        
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
        let decoded = decoder.decode().unwrap();
        
        // Compute max and average difference
        let mut max_diff = 0i32;
        let mut sum_diff = 0i64;
        for i in 0..pixels.len() {
            let diff = (pixels[i] as i32 - decoded[i] as i32).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff as i64;
        }
        let avg_diff = sum_diff as f64 / pixels.len() as f64;
        
        println!("Q{}: size={} bytes, max_diff={}, avg_diff={:.2}", 
            q, jpeg_data.len(), max_diff, avg_diff);
    }
}
