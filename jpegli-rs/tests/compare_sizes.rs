use jpegli::{decode::Decoder, types::Subsampling, JpegEncoder, Quality};

fn create_gradient(size: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (size * size * 3) as usize];
    for y in 0..size {
        for x in 0..size {
            let val = ((x + y) * 255 / (size * 2 - 2)) as u8;
            let idx = ((y * size + x) * 3) as usize;
            pixels[idx] = val;
            pixels[idx + 1] = val;
            pixels[idx + 2] = val;
        }
    }
    pixels
}

fn test_size(size: u32) {
    let pixels = create_gradient(size);

    let encoder = JpegEncoder::new(size, size)
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S420);
    let jpeg = match encoder.encode(&pixels) {
        Ok(j) => j,
        Err(e) => {
            println!("  {}x{}: ENCODE FAILED: {:?}", size, size, e);
            return;
        }
    };

    // Save for inspection
    std::fs::write(format!("/tmp/test_420_{}.jpg", size), &jpeg).ok();

    // Try our decoder
    let decoder = Decoder::new();
    match decoder.decode(&jpeg) {
        Ok(decoded) => {
            let mut max_err = 0i32;
            for i in 0..pixels.len().min(decoded.data.len()) {
                let err = (pixels[i] as i32 - decoded.data[i] as i32).abs();
                if err > max_err {
                    max_err = err;
                }
            }
            println!(
                "  {}x{}: OK, {} bytes, max_err={}",
                size,
                size,
                jpeg.len(),
                max_err
            );
        }
        Err(e) => {
            println!(
                "  {}x{}: DECODE FAILED: {:?}, {} bytes",
                size,
                size,
                e,
                jpeg.len()
            );
            // Try with djpeg
        }
    }
}

#[test]
fn compare_420_sizes() {
    println!("Testing 4:2:0 at various sizes:");
    for size in [16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256] {
        test_size(size);
    }
}
