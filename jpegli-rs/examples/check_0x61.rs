use jpegli::huffman::HuffmanEncodeTable;

fn main() {
    let ac_luma = HuffmanEncodeTable::std_ac_luminance();
    let ac_chroma = HuffmanEncodeTable::std_ac_chrominance();

    // Check 0x61 (run=6, cat=1)
    let (code_luma, len_luma) = ac_luma.encode(0x61);
    let (code_chroma, len_chroma) = ac_chroma.encode(0x61);
    println!("Symbol 0x61 (run=6, cat=1):");
    println!("  Luma:   code={:0>16b}, len={}", code_luma, len_luma);
    println!("  Chroma: code={:0>16b}, len={}", code_chroma, len_chroma);

    // Parse bits 1111001000111111
    let bits = 0b1111001000111111u16;
    println!("\nBits to parse: {:016b}", bits);

    // Try chroma codes at start
    for symbol in [0x00, 0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0xF0] {
        let (code, len) = ac_chroma.encode(symbol);
        if len > 0 {
            let mask = (1u16 << len) - 1;
            let shifted = (bits >> (16 - len)) as u32;
            if shifted == code {
                println!(
                    "  Matched symbol 0x{:02X} at start (code={:0>16b}, len={})",
                    symbol, code, len
                );
            }
        }
    }
}
