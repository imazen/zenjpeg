//! Check what AC symbols are in the Huffman table

use jpegli::{
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

    // Parse DHT and extract AC table 0
    let mut i = 0;
    while i < encoded.len() - 1 {
        if encoded[i] == 0xFF && encoded[i + 1] == 0xC4 {
            let len = u16::from_be_bytes([encoded[i + 2], encoded[i + 3]]) as usize;
            let mut pos = i + 4;

            while pos < i + 2 + len {
                let tc_th = encoded[pos];
                let tc = tc_th >> 4;
                let th = tc_th & 0x0F;
                pos += 1;

                let mut bits = [0u8; 16];
                for j in 0..16 {
                    bits[j] = encoded[pos + j];
                }
                pos += 16;

                let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();
                let mut symbols = Vec::with_capacity(total_symbols);
                for _ in 0..total_symbols {
                    if pos < encoded.len() {
                        symbols.push(encoded[pos]);
                        pos += 1;
                    }
                }

                if tc == 1 && th == 0 {
                    // This is AC Table 0
                    println!("AC Table 0 bits: {:?}", bits);
                    println!("AC Table 0 symbols ({} total):", symbols.len());

                    // Generate Huffman codes for each symbol
                    let mut code: u32 = 0;
                    let mut symbol_idx = 0;

                    println!("\nCode assignments:");
                    for (length_minus_1, &count) in bits.iter().enumerate() {
                        let length = length_minus_1 + 1;
                        for _ in 0..count {
                            let sym = symbols[symbol_idx];
                            let run = sym >> 4;
                            let size = sym & 0x0F;
                            let sym_type = if sym == 0xF0 {
                                "ZRL"
                            } else if size == 0 {
                                if run == 0 {
                                    "EOB"
                                } else {
                                    "EOBn"
                                }
                            } else if size == 1 {
                                "NEW_NZ"
                            } else {
                                "FIRST_AC"
                            };
                            println!(
                                "  len={:2} code={:0width$b} → sym=0x{:02X} ({})",
                                length,
                                code,
                                sym,
                                sym_type,
                                width = length
                            );
                            code += 1;
                            symbol_idx += 1;
                        }
                        code <<= 1;
                    }

                    // Check for missing refinement symbols
                    println!("\nMissing refinement symbols check:");
                    let has_eob = symbols.contains(&0x00);
                    let has_zrl = symbols.contains(&0xF0);
                    println!(
                        "  EOB (0x00): {}",
                        if has_eob { "present" } else { "MISSING" }
                    );
                    println!(
                        "  ZRL (0xF0): {}",
                        if has_zrl { "present" } else { "MISSING" }
                    );

                    // Check EOB run symbols 0x10-0xE0
                    for run in 1..=14 {
                        let sym = (run << 4) as u8;
                        let present = symbols.contains(&sym);
                        if !present {
                            println!("  EOB{} (0x{:02X}): MISSING", 1 << run, sym);
                        }
                    }

                    // Check newly-nonzero symbols 0x01-0xF1
                    for run in 0..=15 {
                        let sym = (run << 4) | 1;
                        if !symbols.contains(&sym) {
                            println!("  Newly-NZ run={} (0x{:02X}): MISSING", run, sym);
                        }
                    }
                }
            }
            i += 2 + len;
        } else {
            i += 1;
        }
    }
}
