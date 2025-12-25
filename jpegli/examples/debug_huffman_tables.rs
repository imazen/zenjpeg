//! Debug Huffman table generation

use jpegli::{Encoder, Quality};
use std::fs::File;

fn load_png(path: &str) -> (u32, u32, Vec<u8>) {
    let file = File::open(path).unwrap();
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());

    let pixels = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buf.chunks(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        _ => panic!("Unsupported color type"),
    };
    (info.width, info.height, pixels)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/v/work/corpus/CID22-512/1044329.png".to_string());

    println!("Loading: {}", path);
    let (w, h, pixels) = load_png(&path);
    println!("Image: {}x{}", w, h);

    // Encode with optimized tables
    let jpeg = Encoder::new()
        .width(w)
        .height(h)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(true)
        .encode(&pixels)
        .unwrap();

    // Parse DHT markers from the JPEG to see what was written
    println!("\nDHT markers in output:");
    let mut i = 0;
    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xC4 {
            // DHT marker
            let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let tc_th = jpeg[i + 4];
            let tc = tc_th >> 4; // table class (0=DC, 1=AC)
            let th = tc_th & 0x0F; // table ID

            let bits = &jpeg[i + 5..i + 5 + 16];
            let num_symbols: usize = bits.iter().map(|&b| b as usize).sum();
            let values = &jpeg[i + 21..i + 21 + num_symbols];

            println!(
                "\n  {} table {} (len={}):",
                if tc == 0 { "DC" } else { "AC" },
                th,
                len
            );
            println!("    bits: {:?}", bits);
            println!("    num_symbols: {}", num_symbols);
            if num_symbols <= 20 {
                println!("    values: {:?}", values);
            } else {
                println!("    values: {:?}... ({} total)", &values[..10], num_symbols);
            }

            // Validate: sum of bits should not exceed max codes per length
            let mut total_codes = 0u32;
            let mut valid = true;
            for (len_idx, &count) in bits.iter().enumerate() {
                let length = len_idx + 1;
                total_codes = (total_codes + count as u32) << 1;
                if total_codes > (1u32 << length) {
                    println!(
                        "    ERROR: overflow at length {}: {} codes but max is {}",
                        length,
                        total_codes >> 1,
                        1u32 << (length - 1)
                    );
                    valid = false;
                }
            }
            if valid {
                println!("    Huffman code count: valid");
            }

            i += 2 + len;
        } else {
            i += 1;
        }
    }

    // Try to decode
    println!("\nDecoding...");
    match jpeg_decoder::Decoder::new(&jpeg[..]).decode() {
        Ok(_) => println!("Success!"),
        Err(e) => println!("Failed: {:?}", e),
    }
}
