use jpegli::decode::Decoder;
use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn main() {
    // Create the 64x64 gradient image that fails
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

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .quality(Quality::from_quality(90.0))
        .use_xyb(true);

    let jpeg = encoder.encode(&rgb).expect("encode");
    println!("Encoded {} bytes", jpeg.len());

    // Print JPEG structure
    println!("\nJPEG markers:");
    let mut i = 0;
    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF {
            let marker = jpeg[i + 1];
            match marker {
                0x00 => {
                    i += 1;
                    continue;
                } // Stuffed byte
                0xD8 => println!("  {:04x}: SOI", i),
                0xD9 => println!("  {:04x}: EOI", i),
                0xC0 => println!("  {:04x}: SOF0 (baseline)", i),
                0xC2 => println!("  {:04x}: SOF2 (progressive)", i),
                0xC4 => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    println!("  {:04x}: DHT (len={})", i, len);
                    i += 2 + len;
                    continue;
                }
                0xDB => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    println!("  {:04x}: DQT (len={})", i, len);
                    i += 2 + len;
                    continue;
                }
                0xDA => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    println!("  {:04x}: SOS (len={})", i, len);
                    // After SOS header, entropy data follows until next marker
                    i += 2 + len;
                    // Count entropy data bytes
                    let ecs_start = i;
                    while i < jpeg.len() - 1 {
                        if jpeg[i] == 0xFF && jpeg[i + 1] != 0x00 && jpeg[i + 1] != 0xFF {
                            break;
                        }
                        i += 1;
                    }
                    println!("    -> entropy data: {} bytes", i - ecs_start);
                    continue;
                }
                0xE0..=0xEF => {
                    let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                    println!("  {:04x}: APP{} (len={})", i, marker - 0xE0, len);
                    i += 2 + len;
                    continue;
                }
                0xDD => {
                    println!("  {:04x}: DRI", i);
                    i += 6;
                    continue;
                }
                _ => println!("  {:04x}: marker 0x{:02X}", i, marker),
            }
        }
        i += 1;
    }

    // Try with jpeg-decoder crate
    println!("\n--- Testing with jpeg-decoder crate ---");
    match jpeg_decoder::Decoder::new(&jpeg[..]).decode() {
        Ok(pixels) => {
            println!("jpeg-decoder SUCCESS: {} bytes output", pixels.len());
        }
        Err(e) => {
            println!("jpeg-decoder FAILED: {:?}", e);
        }
    }

    // Try with native decoder
    println!("\n--- Testing with native decoder ---");
    match Decoder::new().decode(&jpeg) {
        Ok(img) => {
            println!("Native decode SUCCESS: {}x{}", img.width, img.height);
        }
        Err(e) => {
            println!("Native decode FAILED: {:?}", e);
        }
    }

    // Save the JPEG for external analysis
    std::fs::write("/tmp/failing_gradient.jpg", &jpeg).expect("write file");
    println!("\nSaved to /tmp/failing_gradient.jpg");
}
