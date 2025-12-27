//! Use jpeg-decoder internals to understand the correct decoding

use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn main() {
    let width = 64u32;
    let height = 64u32;

    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = ((x * 4) % 256) as u8;
            rgb[i + 1] = ((y * 4) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    let jpeg = Encoder::new()
        .width(width)
        .height(height)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true)
        .encode(&rgb)
        .expect("encode");

    std::fs::write("/tmp/test_xyb.jpg", &jpeg).unwrap();
    println!("Wrote {} bytes to /tmp/test_xyb.jpg", jpeg.len());

    // Decode with jpeg-decoder
    let mut decoder = jpeg_decoder::Decoder::new(&jpeg[..]);
    match decoder.decode() {
        Ok(pixels) => {
            println!("jpeg-decoder: decoded {} bytes of pixels", pixels.len());
            let info = decoder.info().unwrap();
            println!("  Size: {}x{}", info.width, info.height);
            println!("  Components: {:?}", info.pixel_format);
        }
        Err(e) => println!("jpeg-decoder failed: {:?}", e),
    }

    // Also try cjpegli (djpegli) via command line if available
    println!("\nTrying djpegli:");
    match std::process::Command::new("djpegli")
        .args(["/tmp/test_xyb.jpg", "/tmp/test_xyb.png"])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                println!("  djpegli: SUCCESS");
            } else {
                println!("  djpegli: FAILED");
                println!("  stderr: {}", String::from_utf8_lossy(&output.stderr));
            }
        }
        Err(e) => println!("  djpegli not available: {}", e),
    }

    // Now test our native decoder
    println!("\nTrying native decoder:");
    match jpegli::decode::Decoder::new().decode(&jpeg) {
        Ok(img) => println!("  Native: SUCCESS {}x{}", img.width, img.height),
        Err(e) => println!("  Native: FAILED - {:?}", e),
    }
}
