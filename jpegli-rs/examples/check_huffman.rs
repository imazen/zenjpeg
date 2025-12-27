// Check Huffman table for EOB symbol and refinement symbols
use jpegli::huffman::HuffmanEncodeTable;

fn check_table(name: &str, table: &HuffmanEncodeTable) {
    println!("\n{} table:", name);

    // Check EOB (0x00)
    let (code, len) = table.encode(0x00);
    println!("  EOB (0x00): code={:0>16b}, len={}", code, len);

    // Check refinement symbols (run << 4 | 1)
    let mut missing = 0;
    for run in 0..=15 {
        let symbol = (run << 4) | 1;
        let (_, len) = table.encode(symbol);
        if len == 0 {
            println!("  Symbol 0x{:02X} (run={:2}): MISSING!", symbol, run);
            missing += 1;
        }
    }
    if missing == 0 {
        println!("  All refinement symbols present");
    }

    // Check ZRL
    let (_, len) = table.encode(0xF0);
    println!("  ZRL (0xF0): len={}", len);
}

fn main() {
    let ac_luma = HuffmanEncodeTable::std_ac_luminance();
    let ac_chroma = HuffmanEncodeTable::std_ac_chrominance();

    check_table("AC Luminance", &ac_luma);
    check_table("AC Chrominance", &ac_chroma);

    // Compare specific symbols
    println!("\nComparing EOB codes:");
    let (code_luma, len_luma) = ac_luma.encode(0x00);
    let (code_chroma, len_chroma) = ac_chroma.encode(0x00);
    println!("  Luma:   code={:0>16b}, len={}", code_luma, len_luma);
    println!("  Chroma: code={:0>16b}, len={}", code_chroma, len_chroma);
}
