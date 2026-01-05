//! Debug progressive JPEG decoding issue

use jpegli::{
    decode::Decoder,
    encode::Encoder,
    types::{JpegMode, PixelFormat, Subsampling},
    Quality,
};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .mode(JpegMode::Progressive);

    let encoded = encoder.encode(&pixels).expect("encode");
    println!("Encoded {} bytes", encoded.len());

    // Dump first 200 bytes of JPEG to inspect structure
    println!("\nJPEG header bytes:");
    for (i, chunk) in encoded
        .iter()
        .take(300)
        .collect::<Vec<_>>()
        .chunks(16)
        .enumerate()
    {
        print!("{:04x}: ", i * 16);
        for b in chunk {
            print!("{:02x} ", b);
        }
        println!();
    }

    // Find all markers
    println!("\nMarkers found:");
    let mut i = 0;
    while i < encoded.len() - 1 {
        if encoded[i] == 0xFF && encoded[i + 1] != 0x00 && encoded[i + 1] != 0xFF {
            let marker = encoded[i + 1];
            let name = match marker {
                0xD8 => "SOI",
                0xD9 => "EOI",
                0xC0 => "SOF0 (Baseline)",
                0xC2 => "SOF2 (Progressive)",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xDB => "DQT",
                0xDD => "DRI",
                0xE0..=0xEF => "APPn",
                _ => "???",
            };

            if marker == 0xDA {
                // SOS - parse details
                if i + 5 < encoded.len() {
                    let len = u16::from_be_bytes([encoded[i + 2], encoded[i + 3]]) as usize;
                    let num_components = encoded[i + 4];
                    print!("  {:04x}: SOS len={} comps={}", i, len, num_components);

                    if i + 4 + (num_components as usize * 2) + 3 < encoded.len() {
                        let base = i + 5 + (num_components as usize * 2);
                        let ss = encoded[base];
                        let se = encoded[base + 1];
                        let ah_al = encoded[base + 2];
                        let ah = ah_al >> 4;
                        let al = ah_al & 0x0F;
                        println!(" Ss={} Se={} Ah={} Al={}", ss, se, ah, al);
                    } else {
                        println!();
                    }
                } else {
                    println!("  {:04x}: {}", i, name);
                }
            } else if marker == 0xC4 {
                // DHT - parse table class and ID
                if i + 4 < encoded.len() {
                    let len = u16::from_be_bytes([encoded[i + 2], encoded[i + 3]]) as usize;
                    let tc_th = encoded[i + 4];
                    let tc = tc_th >> 4;
                    let th = tc_th & 0x0F;
                    println!("  {:04x}: DHT len={} class={} id={}", i, len, tc, th);
                } else {
                    println!("  {:04x}: {}", i, name);
                }
            } else {
                println!("  {:04x}: {} (0x{:02x})", i, name, marker);
            }

            // Skip marker payload if it has length
            if marker >= 0xC0
                && marker != 0xD8
                && marker != 0xD9
                && marker != 0xD0
                && marker != 0xFF
                && i + 3 < encoded.len()
            {
                let len = u16::from_be_bytes([encoded[i + 2], encoded[i + 3]]) as usize;
                i += 2 + len;
            } else {
                i += 2;
            }
        } else {
            i += 1;
        }
    }

    // Try decoding with our decoder
    println!("\n=== Attempting jpegli-rs decode ===");
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    match decoder.decode(&encoded) {
        Ok(img) => println!("jpegli-rs decoder: OK ({}x{})", img.width, img.height),
        Err(e) => println!("jpegli-rs decoder: FAIL - {:?}", e),
    }

    // Try with jpeg-decoder for comparison
    println!("\n=== Attempting jpeg-decoder decode ===");
    match decode_zune(&encoded[..]) {
        Ok(pixels) => println!("jpeg-decoder: OK ({} bytes)", pixels.len()),
        Err(e) => println!("jpeg-decoder: FAIL - {}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
