//! Trace scan 10 decoding to find where djpeg fails.

use std::fs;

fn main() {
    let rust_jpg = fs::read("/tmp/binary_compare_rust.jpg").unwrap();

    // Find scan 10 (second Y refinement, Ah=1 Al=0)
    // From djpeg output: Component 1, dc=0 ac=2, Ss=3 Se=63 Ah=1 Al=0
    // This is the 11th SOS marker (0-indexed: scan 10)

    // Build Huffman decode table for AC table at slot 2
    let ac_table_2 = find_ac_table(&rust_jpg, 2);
    println!("AC table 2 has {} symbols", ac_table_2.symbols.len());

    // Find scan 10 data
    let scan_10_start = find_nth_sos(&rust_jpg, 10);
    let scan_10_end = find_nth_sos(&rust_jpg, 11);

    println!("Scan 10: {:04X} - {:04X} ({} bytes)",
             scan_10_start, scan_10_end, scan_10_end - scan_10_start);

    // Skip SOS header (8 bytes for single-component scan)
    let data_start = scan_10_start + 10; // 2 (marker) + 2 (length) + 6 (params)

    // Decode the scan data
    let scan_data = &rust_jpg[data_start..scan_10_end];
    decode_ac_refinement_scan(scan_data, &ac_table_2);
}

struct HuffmanTable {
    symbols: Vec<u8>,
    codes: Vec<u32>,
    lengths: Vec<u8>,
}

fn find_ac_table(data: &[u8], table_id: u8) -> HuffmanTable {
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == 0xC4 {
            let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            let mut pos = i + 4;
            let end = i + 2 + len;

            while pos < end {
                let table_spec = data[pos];
                let tc = (table_spec >> 4) & 0x0F;
                let tid = table_spec & 0x0F;
                pos += 1;

                let counts = &data[pos..pos + 16];
                let total: usize = counts.iter().map(|&c| c as usize).sum();
                pos += 16;

                let symbols = data[pos..pos + total].to_vec();
                pos += total;

                if tc == 1 && tid == table_id {
                    // Build codes
                    let mut codes = Vec::new();
                    let mut lengths = Vec::new();
                    let mut code: u32 = 0;
                    let mut sym_idx = 0;

                    for (bit_len, &count) in counts.iter().enumerate() {
                        let bits = bit_len + 1;
                        for _ in 0..count {
                            codes.push(code);
                            lengths.push(bits as u8);
                            sym_idx += 1;
                            code += 1;
                        }
                        code <<= 1;
                    }

                    return HuffmanTable { symbols, codes, lengths };
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
    panic!("AC table {} not found", table_id);
}

fn find_nth_sos(data: &[u8], n: usize) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            if count == n {
                return i;
            }
            count += 1;
            // Skip SOS header
            let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            i += 2 + len;
            // Skip scan data until next marker
            while i < data.len() - 1 {
                if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
                    if data[i + 1] >= 0xD0 && data[i + 1] <= 0xD7 {
                        i += 2; // Skip RST marker
                        continue;
                    }
                    break;
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    data.len() // End of file
}

fn decode_ac_refinement_scan(data: &[u8], table: &HuffmanTable) {
    let mut reader = BitReader::new(data);
    let mut block_num = 0;
    let mut symbol_num = 0;
    let mut eob_run = 0u16;

    // 64 blocks (8x8 = 64 pixels) in a 64x64 image with 4:4:4
    while block_num < 64 {
        if eob_run > 0 {
            eob_run -= 1;
            block_num += 1;
            continue;
        }

        // Process coefficients 3-63 for this block
        let mut k = 3;
        while k <= 63 {
            // Read Huffman symbol
            match decode_symbol(&mut reader, table) {
                Ok((sym, code_bits)) => {
                    let rrrr = (sym >> 4) & 0x0F;
                    let ssss = sym & 0x0F;

                    if ssss == 0 {
                        if rrrr == 0x0F {
                            // ZRL - skip 16 zeros plus refbits
                            // In refinement, we need to read correction bits for nonzero coeffs
                            symbol_num += 1;
                            k += 16;
                        } else if rrrr == 0 {
                            // EOB (1 block)
                            symbol_num += 1;
                            block_num += 1;
                            break;
                        } else {
                            // EOB run
                            let run_bits = rrrr;
                            let base = 1u16 << run_bits;
                            let extra = reader.read_bits(run_bits as usize).unwrap_or(0);
                            eob_run = base + extra as u16 - 1; // -1 because current block counts
                            symbol_num += 1;
                            block_num += 1;
                            break;
                        }
                    } else if ssss == 1 {
                        // Newly nonzero coefficient
                        // Read sign bit
                        let sign_bit = reader.read_bits(1).unwrap_or(0);
                        symbol_num += 1;
                        k += rrrr as usize + 1;
                    } else {
                        println!("ERROR at block {} symbol {}: unexpected ssss={} (sym=0x{:02X})",
                                 block_num, symbol_num, ssss, sym);
                        println!("  Bit position: {}", reader.bit_pos);
                        println!("  Last {} bits decoded: {:0width$b}", code_bits,
                                 reader.last_code, width = reader.last_len as usize);
                        return;
                    }
                }
                Err(msg) => {
                    println!("ERROR at block {} symbol {} k={}: {}",
                             block_num, symbol_num, k, msg);
                    println!("  Byte position: {}", reader.byte_pos);
                    println!("  Bit position: {}", reader.bit_pos);
                    // Show context
                    let start = reader.byte_pos.saturating_sub(4);
                    let end = (reader.byte_pos + 4).min(data.len());
                    print!("  Bytes around error: ");
                    for i in start..end {
                        if i == reader.byte_pos {
                            print!("[{:02X}]", data[i]);
                        } else {
                            print!("{:02X} ", data[i]);
                        }
                    }
                    println!();
                    return;
                }
            }
        }
    }

    println!("Successfully decoded {} blocks, {} symbols", block_num, symbol_num);
}

fn decode_symbol(reader: &mut BitReader, table: &HuffmanTable) -> Result<(u8, u32), String> {
    let mut code: u32 = 0;

    for len in 1..=16u8 {
        let bit = reader.read_bits(1).ok_or("EOF reading bit")?;
        code = (code << 1) | bit;

        // Check if this matches any code of this length
        for (i, &sym_len) in table.lengths.iter().enumerate() {
            if sym_len == len && table.codes[i] == code {
                reader.last_code = code;
                reader.last_len = len;
                return Ok((table.symbols[i], code));
            }
        }
    }

    Err(format!("No match for code {:016b} ({} bits read)", code, 16))
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: usize,
    current_byte: u8,
    bits_left: u8,
    last_code: u32,
    last_len: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut reader = BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
            current_byte: 0,
            bits_left: 0,
            last_code: 0,
            last_len: 0,
        };
        reader.refill();
        reader
    }

    fn refill(&mut self) {
        if self.byte_pos >= self.data.len() {
            return;
        }

        self.current_byte = self.data[self.byte_pos];
        self.byte_pos += 1;
        self.bits_left = 8;

        // Handle byte stuffing (0xFF 0x00)
        if self.current_byte == 0xFF && self.byte_pos < self.data.len() {
            if self.data[self.byte_pos] == 0x00 {
                self.byte_pos += 1; // Skip the 0x00
            }
        }
    }

    fn read_bits(&mut self, n: usize) -> Option<u32> {
        let mut result: u32 = 0;

        for _ in 0..n {
            if self.bits_left == 0 {
                self.refill();
                if self.bits_left == 0 {
                    return None;
                }
            }

            let bit = (self.current_byte >> (self.bits_left - 1)) & 1;
            result = (result << 1) | (bit as u32);
            self.bits_left -= 1;
            self.bit_pos += 1;
        }

        Some(result)
    }
}
