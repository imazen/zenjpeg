//! Step-by-step debug of the decoder failure

use jpegli::encode::Encoder;
use jpegli::quant::Quality;

// Minimal bitstream reader
struct DebugBitReader<'a> {
    data: &'a [u8],
    position: usize,
    bit_buffer: u32,
    bits_in_buffer: u8,
    total_bits_read: u64,
}

impl<'a> DebugBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            bit_buffer: 0,
            bits_in_buffer: 0,
            total_bits_read: 0,
        }
    }

    fn read_byte(&mut self) -> Option<u8> {
        if self.position >= self.data.len() {
            return None;
        }
        let byte = self.data[self.position];
        self.position += 1;

        if byte == 0xFF {
            if self.position >= self.data.len() {
                return None;
            }
            let next = self.data[self.position];
            if next == 0x00 {
                self.position += 1;
            } else if (0xD0..=0xD7).contains(&next) {
                self.position += 1;
            } else {
                self.position -= 1;
                return None;
            }
        }
        Some(byte)
    }

    fn fill_buffer(&mut self, count: u8) -> bool {
        while self.bits_in_buffer < count {
            match self.read_byte() {
                Some(byte) => {
                    self.bit_buffer = (self.bit_buffer << 8) | (byte as u32);
                    self.bits_in_buffer += 8;
                }
                None => return false,
            }
        }
        true
    }

    fn read_bits(&mut self, count: u8) -> Option<u32> {
        if !self.fill_buffer(count) {
            if self.bits_in_buffer < count {
                return None;
            }
        }
        self.total_bits_read += count as u64;
        let bits = (self.bit_buffer >> (self.bits_in_buffer - count)) & ((1 << count) - 1);
        self.bits_in_buffer -= count;
        Some(bits)
    }

    fn peek_bits(&mut self, count: u8) -> Option<u32> {
        if !self.fill_buffer(count) {
            if self.bits_in_buffer < count {
                return None;
            }
        }
        let bits = (self.bit_buffer >> (self.bits_in_buffer - count)) & ((1 << count) - 1);
        Some(bits)
    }

    fn skip_bits(&mut self, count: u8) {
        if count <= self.bits_in_buffer {
            self.bits_in_buffer -= count;
            self.total_bits_read += count as u64;
        }
    }
}

// Minimal Huffman decoder
struct HuffTable {
    fast_lookup: [i16; 512], // 9 bits
    maxcode: [i32; 17],
    valoffset: [i32; 17],
    values: Vec<u8>,
}

impl HuffTable {
    fn from_bits_values(bits: &[u8; 16], values: &[u8]) -> Self {
        let mut table = Self {
            fast_lookup: [-1; 512],
            maxcode: [-1; 17],
            valoffset: [0; 17],
            values: values.to_vec(),
        };

        // Build size and code arrays
        let mut huffsize = vec![0u8; values.len() + 1];
        let mut huffcode = vec![0u32; values.len()];

        let mut k = 0;
        for (i, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                huffsize[k] = (i + 1) as u8;
                k += 1;
            }
        }
        huffsize[k] = 0;

        let mut code: u32 = 0;
        let mut si = huffsize[0] as usize;
        k = 0;
        while huffsize[k] != 0 {
            while (huffsize[k] as usize) == si {
                huffcode[k] = code;
                code += 1;
                k += 1;
            }
            code <<= 1;
            si += 1;
        }

        // Build maxcode and valoffset
        let mut j = 0;
        for i in 1..=16 {
            if bits[i - 1] == 0 {
                table.maxcode[i] = -1;
            } else {
                table.valoffset[i] = j as i32 - huffcode[j] as i32;
                j += bits[i - 1] as usize;
                table.maxcode[i] = huffcode[j - 1] as i32;
            }
        }

        // Build fast lookup
        for (k, &code) in huffcode.iter().enumerate() {
            let length = huffsize[k] as usize;
            if length <= 9 && length > 0 {
                let fast_code = (code as usize) << (9 - length);
                let count = 1 << (9 - length);
                for m in 0..count {
                    let idx = fast_code + m;
                    if idx < 512 {
                        table.fast_lookup[idx] = (values[k] as i16) | ((length as i16) << 8);
                    }
                }
            }
        }

        table
    }

    fn decode(&self, reader: &mut DebugBitReader) -> Option<u8> {
        // Fast path
        if let Some(bits) = reader.peek_bits(9) {
            let lookup = self.fast_lookup[bits as usize];
            if lookup >= 0 {
                let length = (lookup >> 8) as u8;
                reader.skip_bits(length);
                return Some((lookup & 0xFF) as u8);
            }
        }

        // Slow path
        let mut code = 0u32;
        for len in 1..=16 {
            code = (code << 1) | reader.read_bits(1)?;
            if (code as i32) <= self.maxcode[len] {
                let idx = (code as i32 + self.valoffset[len]) as usize;
                if idx < self.values.len() {
                    return Some(self.values[idx]);
                }
            }
        }
        None
    }
}

fn decode_value(category: u8, bits: u16) -> i16 {
    if category == 0 {
        return 0;
    }
    let half = 1u16 << (category - 1);
    if bits >= half {
        bits as i16
    } else {
        ((bits as i32) - ((1i32 << category) - 1)) as i16
    }
}

fn parse_dht(jpeg: &[u8]) -> (HuffTable, HuffTable) {
    let mut pos = 2;
    let mut dc_bits = [0u8; 16];
    let mut dc_values = Vec::new();
    let mut ac_bits = [0u8; 16];
    let mut ac_values = Vec::new();

    while pos < jpeg.len() - 1 {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = jpeg[pos + 1];
        pos += 2;

        if marker == 0xC4 {
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            let mut dht_pos = pos + 2;
            let dht_end = pos + len;

            while dht_pos < dht_end {
                let info = jpeg[dht_pos];
                let table_class = info >> 4;
                dht_pos += 1;

                let mut bits = [0u8; 16];
                let mut num_symbols = 0;
                for i in 0..16 {
                    bits[i] = jpeg[dht_pos + i];
                    num_symbols += bits[i] as usize;
                }
                dht_pos += 16;

                let values: Vec<u8> = jpeg[dht_pos..dht_pos + num_symbols].to_vec();
                dht_pos += num_symbols;

                if table_class == 0 {
                    dc_bits = bits;
                    dc_values = values;
                } else {
                    ac_bits = bits;
                    ac_values = values;
                }
            }
            pos += len;
        } else if marker == 0xD9 {
            break;
        } else if marker >= 0xC0 && marker <= 0xFE && marker != 0xD8 {
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            pos += len;
        }
    }

    (
        HuffTable::from_bits_values(&dc_bits, &dc_values),
        HuffTable::from_bits_values(&ac_bits, &ac_values),
    )
}

fn find_entropy_data(jpeg: &[u8]) -> (usize, usize) {
    let mut pos = 2;
    while pos < jpeg.len() - 1 {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = jpeg[pos + 1];
        pos += 2;

        if marker == 0xDA {
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            let start = pos + len;
            let mut end = start;
            while end < jpeg.len() - 1 {
                if jpeg[end] == 0xFF && jpeg[end + 1] != 0x00 && jpeg[end + 1] != 0xFF {
                    if jpeg[end + 1] < 0xD0 || jpeg[end + 1] > 0xD7 {
                        break;
                    }
                }
                end += 1;
            }
            return (start, end);
        } else if marker >= 0xC0 && marker <= 0xFE && marker != 0xD8 {
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            pos += len;
        }
    }
    (0, 0)
}

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

    println!("JPEG: {} bytes", jpeg.len());

    let (dc_table, ac_table) = parse_dht(&jpeg);
    println!("DC table: {} values", dc_table.values.len());
    println!("AC table: {} values", ac_table.values.len());

    let (ecs_start, ecs_end) = find_entropy_data(&jpeg);
    println!(
        "Entropy data: bytes {}..{} ({} bytes)",
        ecs_start,
        ecs_end,
        ecs_end - ecs_start
    );

    // Dump last few bytes
    println!("Last 10 bytes of entropy data:");
    let last_start = ecs_end.saturating_sub(10);
    for i in last_start..ecs_end {
        print!("{:02X} ", jpeg[i]);
    }
    println!();

    // Check for stuffed bytes in last portion
    println!("Checking for byte stuffing in last 20 bytes:");
    let check_start = ecs_end.saturating_sub(20);
    for i in check_start..ecs_end {
        if jpeg[i] == 0xFF && i + 1 < jpeg.len() && jpeg[i + 1] == 0x00 {
            println!("  Found stuffed FF 00 at position {}", i);
        }
    }

    // What's after entropy data?
    println!(
        "Bytes after entropy data: {:02X} {:02X}",
        jpeg[ecs_end],
        jpeg[ecs_end + 1]
    );

    // XYB has 3 components: 2x2, 2x2, 1x1
    // MCU covers 16x16 pixels, so 4x4 = 16 MCUs for 64x64 image
    // Each MCU has 4+4+1 = 9 blocks
    // Total: 144 blocks

    let mut reader = DebugBitReader::new(&jpeg[ecs_start..ecs_end]);
    let mut prev_dc = [0i16; 3];
    let mut blocks_decoded = 0;
    let mut bits_per_block = Vec::new();

    let mcu_cols = 4;
    let mcu_rows = 4;
    let h_samp = [2, 2, 1];
    let v_samp = [2, 2, 1];

    'outer: for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            for comp in 0..3 {
                for v in 0..v_samp[comp] {
                    for h in 0..h_samp[comp] {
                        let bits_before = reader.total_bits_read;

                        // Decode DC
                        let dc_cat = match dc_table.decode(&mut reader) {
                            Some(c) => c,
                            None => {
                                println!(
                                    "FAIL at block {}: DC decode failed at bit {}",
                                    blocks_decoded, reader.total_bits_read
                                );
                                println!(
                                    "  MCU ({}, {}), comp {}, block ({}, {})",
                                    mcu_x, mcu_y, comp, h, v
                                );
                                break 'outer;
                            }
                        };

                        let dc_diff = if dc_cat == 0 {
                            0i16
                        } else {
                            match reader.read_bits(dc_cat) {
                                Some(bits) => decode_value(dc_cat, bits as u16),
                                None => {
                                    println!(
                                        "FAIL at block {}: DC bits read failed",
                                        blocks_decoded
                                    );
                                    break 'outer;
                                }
                            }
                        };

                        let dc = prev_dc[comp] + dc_diff;
                        prev_dc[comp] = dc;

                        // Decode AC
                        let mut ac_idx = 1;
                        while ac_idx < 64 {
                            let symbol = match ac_table.decode(&mut reader) {
                                Some(s) => s,
                                None => {
                                    println!(
                                        "FAIL at block {}: AC decode failed at coeff {}, bit {}",
                                        blocks_decoded, ac_idx, reader.total_bits_read
                                    );
                                    println!(
                                        "  MCU ({}, {}), comp {}, block ({}, {})",
                                        mcu_x, mcu_y, comp, h, v
                                    );
                                    println!(
                                        "  Byte pos: {}, bits in buffer: {}",
                                        reader.position, reader.bits_in_buffer
                                    );
                                    break 'outer;
                                }
                            };

                            if symbol == 0 {
                                break; // EOB
                            }

                            let run = symbol >> 4;
                            let size = symbol & 0x0F;

                            if size == 0 {
                                if run == 15 {
                                    ac_idx += 16;
                                } else {
                                    break; // Invalid
                                }
                            } else {
                                ac_idx += run as usize;
                                if ac_idx >= 64 {
                                    println!(
                                        "FAIL: AC index out of bounds at block {}",
                                        blocks_decoded
                                    );
                                    break 'outer;
                                }

                                if size > 0 {
                                    match reader.read_bits(size) {
                                        Some(_) => {}
                                        None => {
                                            println!(
                                                "FAIL: AC bits read failed at block {}",
                                                blocks_decoded
                                            );
                                            break 'outer;
                                        }
                                    }
                                }
                                ac_idx += 1;
                            }
                        }

                        let bits_used = reader.total_bits_read - bits_before;
                        bits_per_block.push((blocks_decoded, comp, bits_used));
                        blocks_decoded += 1;
                    }
                }
            }
        }
    }

    println!("\nDecoded {} blocks successfully", blocks_decoded);

    // Print bits per component
    let mut comp_bits = [0u64; 3];
    let mut comp_blocks = [0u32; 3];
    for &(_, comp, bits) in &bits_per_block {
        comp_bits[comp] += bits;
        comp_blocks[comp] += 1;
    }
    println!("\nBits per component:");
    for c in 0..3 {
        println!(
            "  Comp {}: {} blocks, {} bits total, {:.1} bits/block avg",
            c,
            comp_blocks[c],
            comp_bits[c],
            comp_bits[c] as f64 / comp_blocks[c] as f64
        );
    }

    // Print the last 10 blocks
    println!("\nLast 10 blocks decoded:");
    let start = bits_per_block.len().saturating_sub(10);
    for &(idx, comp, bits) in &bits_per_block[start..] {
        println!("  Block {}: comp {}, {} bits", idx, comp, bits);
    }
    println!("Total bits read: {}", reader.total_bits_read);
    println!("Expected: 144 blocks");
}
