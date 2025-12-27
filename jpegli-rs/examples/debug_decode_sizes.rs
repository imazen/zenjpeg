use jpegli::decode::Decoder;
use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn test_size(width: u32, height: u32) {
    let mut rgb = vec![128u8; (width * height * 3) as usize];
    for i in 0..rgb.len() {
        rgb[i] = (i % 256) as u8;
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true);

    let jpeg = encoder.encode(&rgb).expect("encode");

    let result = Decoder::new().decode(&jpeg);
    let status = match result {
        Ok(_) => "OK",
        Err(_) => "FAIL",
    };

    println!("{}x{}: {} ({} bytes)", width, height, status, jpeg.len());
}

fn main() {
    println!("Testing XYB decode at various sizes...\n");

    // Test various sizes
    for size in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128] {
        test_size(size, size);
    }

    println!("\nTesting non-square sizes:");
    test_size(64, 32);
    test_size(32, 64);
    test_size(48, 32);
    test_size(80, 64);
}
