// Check if specific coefficient patterns cause the failure
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn solid_gray(w: u32, h: u32, val: u8) -> Vec<u8> {
    vec![val; (w * h) as usize]
}

fn gradient_h(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|_| (0..w).map(|x| (x % 256) as u8))
        .collect()
}

fn gradient_v(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |_| (y % 256) as u8))
        .collect()
}

fn checkerboard(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| if (x + y) % 2 == 0 { 0 } else { 255 }))
        .collect()
}

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn test_pattern(name: &str, data: &[u8], w: u32, h: u32) -> bool {
    let jpeg = Encoder::new()
        .width(w)
        .height(h)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(data)
        .expect("encode failed");
    jpeg_decoder::Decoder::new(&jpeg[..]).decode().is_ok()
}

fn main() {
    let sizes = [49u32, 50, 51, 52, 53];

    println!("Pattern tests for failing sizes:\n");

    for &size in &sizes {
        println!("=== {}x{} ===", size, size);

        let solid = solid_gray(size, size, 128);
        let gradh = gradient_h(size, size);
        let gradv = gradient_v(size, size);
        let check = checkerboard(size, size);
        let photo = photo_like(size, size);

        println!(
            "  solid_128:   {}",
            if test_pattern("solid", &solid, size, size) {
                "OK"
            } else {
                "FAIL"
            }
        );
        println!(
            "  gradient_h:  {}",
            if test_pattern("gradh", &gradh, size, size) {
                "OK"
            } else {
                "FAIL"
            }
        );
        println!(
            "  gradient_v:  {}",
            if test_pattern("gradv", &gradv, size, size) {
                "OK"
            } else {
                "FAIL"
            }
        );
        println!(
            "  checkerboard:{}",
            if test_pattern("check", &check, size, size) {
                "OK"
            } else {
                "FAIL"
            }
        );
        println!(
            "  photo_like:  {}",
            if test_pattern("photo", &photo, size, size) {
                "OK"
            } else {
                "FAIL"
            }
        );
        println!();
    }
}
