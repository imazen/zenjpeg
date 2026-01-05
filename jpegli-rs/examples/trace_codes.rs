use jpegli::huffman_opt::FrequencyCounter;
use jpegli::types::HuffmanMethod;

fn main() {
    // Simulate the merged histogram from the failing case
    let mut counter = FrequencyCounter::new();

    // From AC first scans
    for s in [0x01, 0x02, 0x03, 0x04, 0x12, 0x14, 0x50] {
        counter.count(s);
    }
    for s in [0x00, 0x12, 0x50] {
        counter.count(s);
    }
    for s in [0x00, 0x02, 0x12, 0x13, 0x20, 0x30] {
        counter.count(s);
    }
    for s in [0x60] {
        for _ in 0..3 {
            counter.count(s);
        }
    }

    // From refinement scans (masked symbols)
    for s in [
        0x00, 0x01, 0x11, 0x21, 0x30, 0x31, 0x40, 0x51, 0x61, 0x81, 0x91, 0xa1, 0xd1, 0xf0, 0xf1,
    ] {
        for _ in 0..5 {
            counter.count(s);
        }
    }
    for s in [0x00, 0x20, 0x30, 0x40, 0xf1] {
        for _ in 0..3 {
            counter.count(s);
        }
    }
    for s in [
        0x00, 0x01, 0x10, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x81, 0x91, 0xa1, 0xc1, 0xf0, 0xf1,
    ] {
        for _ in 0..10 {
            counter.count(s);
        }
    }
    for s in [
        0x00, 0x01, 0x11, 0x20, 0x21, 0x31, 0x41, 0x51, 0x61, 0x81, 0x91, 0xa1, 0xb1, 0xc1, 0xd1,
        0xe1, 0xf0, 0xf1,
    ] {
        for _ in 0..5 {
            counter.count(s);
        }
    }

    // Generate table
    let result = counter.generate_table_with_method(HuffmanMethod::MozjpegClassic);
    match result {
        Ok(table) => {
            println!("Generated table:");
            println!("bits: {:?}", table.bits);
            println!("values: {:02x?}", table.values);

            // Check specific symbols
            for s in [0x00, 0x01, 0xf1, 0x30, 0x60] {
                let (code, len) = table.table.encode(s);
                println!("Symbol 0x{:02x}: code=0x{:04x} len={}", s, code, len);
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
