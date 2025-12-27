//! Debug histogram symbols for progressive encoding

use jpegli::consts::DCT_BLOCK_SIZE;
use jpegli::huffman_opt::ProgressiveTokenBuffer;
use jpegli::{
    encode::Encoder,
    types::{JpegMode, PixelFormat, Subsampling},
    Quality,
};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

    // We need to inspect the tokenization process
    // For now, let's just analyze what symbols would be generated

    // Create encoder
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .mode(JpegMode::Progressive);

    // Get the encoded JPEG to see what tables are in it
    let encoded = encoder.encode(&pixels).expect("encode");

    // Parse the DHT markers to see what symbols are in the tables
    println!("Analyzing DHT markers in encoded JPEG...");

    let mut i = 0;
    while i < encoded.len() - 1 {
        if encoded[i] == 0xFF && encoded[i + 1] == 0xC4 {
            // Found DHT marker
            let len = u16::from_be_bytes([encoded[i + 2], encoded[i + 3]]) as usize;
            let mut pos = i + 4;

            while pos < i + 2 + len {
                let tc_th = encoded[pos];
                let tc = tc_th >> 4;
                let th = tc_th & 0x0F;
                pos += 1;

                // Read bits array (16 bytes)
                let mut bits = [0u8; 16];
                for j in 0..16 {
                    bits[j] = encoded[pos + j];
                }
                pos += 16;

                // Count total symbols
                let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();

                // Read symbol values
                let mut symbols = Vec::with_capacity(total_symbols);
                for _ in 0..total_symbols {
                    if pos < encoded.len() {
                        symbols.push(encoded[pos]);
                        pos += 1;
                    }
                }

                let class = if tc == 0 { "DC" } else { "AC" };
                println!("\n{} Table {} ({} symbols):", class, th, total_symbols);
                println!("  bits: {:?}", bits);

                // For AC tables, show symbol breakdown
                if tc == 1 {
                    // Categorize symbols
                    let mut eob_symbols = Vec::new();
                    let mut zrl_found = false;
                    let mut size_1_symbols = Vec::new();
                    let mut other_symbols = Vec::new();

                    for &sym in &symbols {
                        let run = sym >> 4;
                        let size = sym & 0x0F;

                        if sym == 0xF0 {
                            zrl_found = true;
                        } else if size == 0 {
                            eob_symbols.push(sym);
                        } else if size == 1 {
                            size_1_symbols.push(sym);
                        } else {
                            other_symbols.push(sym);
                        }
                    }

                    println!("  EOB symbols (size=0): {:02X?}", eob_symbols);
                    println!(
                        "  ZRL (0xF0): {}",
                        if zrl_found { "present" } else { "absent" }
                    );
                    println!("  Size-1 symbols: {:02X?}", size_1_symbols);
                    println!("  Other symbols: {:02X?}", other_symbols);

                    // Check for refinement-only symbols
                    // In refinement, we need: EOB variants, ZRL, and size-1 symbols for newly nonzero
                    println!("\n  Refinement-compatible check:");
                    println!("    - Has EOB (0x00): {}", eob_symbols.contains(&0x00));
                    println!("    - Has ZRL (0xF0): {}", zrl_found);
                    println!(
                        "    - Has newly-nonzero symbols: {}",
                        !size_1_symbols.is_empty()
                    );
                }
            }

            i += 2 + len;
        } else {
            i += 1;
        }
    }

    // Now try to decode with our decoder
    println!("\n=== Testing decode ===");
    let decoder = jpegli::decode::Decoder::new().output_format(PixelFormat::Rgb);
    match decoder.decode(&encoded) {
        Ok(img) => println!("jpegli-rs decoder: OK ({}x{})", img.width, img.height),
        Err(e) => println!("jpegli-rs decoder: FAIL - {:?}", e),
    }
}
