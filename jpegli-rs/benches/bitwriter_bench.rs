//! Microbenchmark for BitWriter performance
//!
//! Compares different 0xFF detection strategies.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

/// Original 4-byte comparison approach
mod original {
    pub struct BitWriter {
        buffer: Vec<u8>,
        bit_buffer: u64,
        bits_in_buffer: u8,
    }

    impl BitWriter {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                buffer: Vec::with_capacity(capacity),
                bit_buffer: 0,
                bits_in_buffer: 0,
            }
        }

        #[inline]
        pub fn write_bits(&mut self, bits: u32, count: u8) {
            self.bit_buffer = (self.bit_buffer << count) | (bits as u64);
            self.bits_in_buffer += count;
            if self.bits_in_buffer >= 32 {
                self.flush_bytes();
            }
        }

        #[inline(never)]
        #[cold]
        fn flush_bytes(&mut self) {
            while self.bits_in_buffer >= 32 {
                self.bits_in_buffer -= 32;
                let word = (self.bit_buffer >> self.bits_in_buffer) as u32;
                let bytes = word.to_be_bytes();

                // Original: 4 separate comparisons
                let has_ff = (bytes[0] == 0xFF)
                    | (bytes[1] == 0xFF)
                    | (bytes[2] == 0xFF)
                    | (bytes[3] == 0xFF);

                if !has_ff {
                    self.buffer.extend_from_slice(&bytes);
                } else {
                    for &b in &bytes {
                        self.buffer.push(b);
                        if b == 0xFF {
                            self.buffer.push(0x00);
                        }
                    }
                }
            }
        }

        pub fn len(&self) -> usize {
            self.buffer.len()
        }
    }
}

/// SWAR approach (matching C++ jpegli)
mod swar {
    #[inline(always)]
    const fn has_byte_0xff_u64(v: u64) -> bool {
        let x = v ^ 0xFFFF_FFFF_FFFF_FFFF;
        (((x.wrapping_sub(0x0101_0101_0101_0101)) & !x) & 0x8080_8080_8080_8080) != 0
    }

    #[inline(always)]
    const fn has_byte_0xff_u32(v: u32) -> bool {
        let x = v ^ 0xFFFF_FFFF;
        (((x.wrapping_sub(0x0101_0101)) & !x) & 0x8080_8080) != 0
    }

    pub struct BitWriter {
        buffer: Vec<u8>,
        bit_buffer: u64,
        bits_in_buffer: u8,
    }

    impl BitWriter {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                buffer: Vec::with_capacity(capacity),
                bit_buffer: 0,
                bits_in_buffer: 0,
            }
        }

        #[inline]
        pub fn write_bits(&mut self, bits: u32, count: u8) {
            self.bit_buffer = (self.bit_buffer << count) | (bits as u64);
            self.bits_in_buffer += count;
            if self.bits_in_buffer >= 32 {
                self.flush_bytes();
            }
        }

        #[inline(always)]
        fn emit_byte(&mut self, byte: u8) {
            self.buffer.push(byte);
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
        }

        #[inline(always)]
        fn emit_4_bytes(&mut self, word: u32) {
            if !has_byte_0xff_u32(word) {
                self.buffer.extend_from_slice(&word.to_be_bytes());
            } else {
                self.emit_byte((word >> 24) as u8);
                self.emit_byte((word >> 16) as u8);
                self.emit_byte((word >> 8) as u8);
                self.emit_byte(word as u8);
            }
        }

        #[inline(always)]
        fn emit_8_bytes(&mut self, word: u64) {
            if !has_byte_0xff_u64(word) {
                self.buffer.extend_from_slice(&word.to_be_bytes());
            } else {
                self.emit_byte((word >> 56) as u8);
                self.emit_byte((word >> 48) as u8);
                self.emit_byte((word >> 40) as u8);
                self.emit_byte((word >> 32) as u8);
                self.emit_byte((word >> 24) as u8);
                self.emit_byte((word >> 16) as u8);
                self.emit_byte((word >> 8) as u8);
                self.emit_byte(word as u8);
            }
        }

        #[inline(never)]
        #[cold]
        fn flush_bytes(&mut self) {
            while self.bits_in_buffer >= 64 {
                self.bits_in_buffer -= 64;
                self.emit_8_bytes(self.bit_buffer);
            }
            while self.bits_in_buffer >= 32 {
                self.bits_in_buffer -= 32;
                let word = (self.bit_buffer >> self.bits_in_buffer) as u32;
                self.emit_4_bytes(word);
            }
        }

        pub fn len(&self) -> usize {
            self.buffer.len()
        }
    }
}

/// Unsafe approach with direct pointer writes
mod unsafe_ptr {
    #[inline(always)]
    const fn has_byte_0xff_u64(v: u64) -> bool {
        let x = v ^ 0xFFFF_FFFF_FFFF_FFFF;
        (((x.wrapping_sub(0x0101_0101_0101_0101)) & !x) & 0x8080_8080_8080_8080) != 0
    }

    pub struct BitWriter {
        buffer: Vec<u8>,
        bit_buffer: u64,
        bits_in_buffer: u8,
    }

    impl BitWriter {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                buffer: Vec::with_capacity(capacity),
                bit_buffer: 0,
                bits_in_buffer: 0,
            }
        }

        #[inline]
        pub fn write_bits(&mut self, bits: u32, count: u8) {
            self.bit_buffer = (self.bit_buffer << count) | (bits as u64);
            self.bits_in_buffer += count;
            if self.bits_in_buffer >= 32 {
                self.flush_bytes();
            }
        }

        /// Unsafe fast path: emit 8 bytes when no 0xFF present
        /// Uses direct pointer write to avoid Vec overhead
        #[inline(always)]
        unsafe fn emit_8_bytes_fast(&mut self, word: u64) {
            // Ensure we have space for 8 bytes
            self.buffer.reserve(8);
            let len = self.buffer.len();
            let ptr = self.buffer.as_mut_ptr().add(len);
            // Write big-endian 8 bytes
            std::ptr::write_unaligned(ptr as *mut u64, word.to_be());
            self.buffer.set_len(len + 8);
        }

        #[inline(always)]
        fn emit_byte(&mut self, byte: u8) {
            self.buffer.push(byte);
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
        }

        #[inline(never)]
        #[cold]
        fn flush_bytes(&mut self) {
            while self.bits_in_buffer >= 64 {
                self.bits_in_buffer -= 64;
                let word = self.bit_buffer;
                if !has_byte_0xff_u64(word) {
                    // SAFETY: We checked no 0xFF bytes, and reserve is called inside
                    unsafe { self.emit_8_bytes_fast(word) };
                } else {
                    // Slow path with stuffing
                    self.emit_byte((word >> 56) as u8);
                    self.emit_byte((word >> 48) as u8);
                    self.emit_byte((word >> 40) as u8);
                    self.emit_byte((word >> 32) as u8);
                    self.emit_byte((word >> 24) as u8);
                    self.emit_byte((word >> 16) as u8);
                    self.emit_byte((word >> 8) as u8);
                    self.emit_byte(word as u8);
                }
            }
            while self.bits_in_buffer >= 32 {
                self.bits_in_buffer -= 32;
                let word = (self.bit_buffer >> self.bits_in_buffer) as u32;
                let bytes = word.to_be_bytes();
                for &b in &bytes {
                    self.emit_byte(b);
                }
            }
        }

        pub fn len(&self) -> usize {
            self.buffer.len()
        }
    }
}

fn bench_bitwriter(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitwriter");

    // Simulate encoding ~10000 blocks worth of Huffman codes
    // Mix of short (4-8 bit) and longer (12-16 bit) codes
    let codes: Vec<(u32, u8)> = (0..100000)
        .map(|i| {
            // Simulated Huffman codes with typical distribution
            match i % 10 {
                0..=3 => (i as u32 & 0xF, 4),    // Short codes (common)
                4..=6 => (i as u32 & 0x3F, 6),   // Medium codes
                7..=8 => (i as u32 & 0x3FF, 10), // Longer codes
                _ => (i as u32 & 0xFFFF, 16),    // Rare long codes
            }
        })
        .collect();

    // Codes that will generate lots of 0xFF bytes (worst case for stuffing)
    let ff_heavy_codes: Vec<(u32, u8)> = (0..100000)
        .map(|i| {
            // Force 0xFF patterns
            match i % 4 {
                0 => (0xFF, 8),
                1 => (0xFFFF, 16),
                2 => (0xFFF, 12),
                _ => (i as u32 & 0xFF, 8),
            }
        })
        .collect();

    group.throughput(Throughput::Elements(codes.len() as u64));

    group.bench_function("original_4cmp", |b| {
        b.iter(|| {
            let mut writer = original::BitWriter::with_capacity(200000);
            for &(bits, count) in &codes {
                writer.write_bits(black_box(bits), black_box(count));
            }
            black_box(writer.len())
        })
    });

    group.bench_function("swar_8byte", |b| {
        b.iter(|| {
            let mut writer = swar::BitWriter::with_capacity(200000);
            for &(bits, count) in &codes {
                writer.write_bits(black_box(bits), black_box(count));
            }
            black_box(writer.len())
        })
    });

    group.bench_function("unsafe_ptr", |b| {
        b.iter(|| {
            let mut writer = unsafe_ptr::BitWriter::with_capacity(200000);
            for &(bits, count) in &codes {
                writer.write_bits(black_box(bits), black_box(count));
            }
            black_box(writer.len())
        })
    });

    // 0xFF-heavy benchmarks (worst case for byte stuffing)
    group.bench_function("ff_heavy_original", |b| {
        b.iter(|| {
            let mut writer = original::BitWriter::with_capacity(300000);
            for &(bits, count) in &ff_heavy_codes {
                writer.write_bits(black_box(bits), black_box(count));
            }
            black_box(writer.len())
        })
    });

    group.bench_function("ff_heavy_swar", |b| {
        b.iter(|| {
            let mut writer = swar::BitWriter::with_capacity(300000);
            for &(bits, count) in &ff_heavy_codes {
                writer.write_bits(black_box(bits), black_box(count));
            }
            black_box(writer.len())
        })
    });

    group.bench_function("ff_heavy_unsafe", |b| {
        b.iter(|| {
            let mut writer = unsafe_ptr::BitWriter::with_capacity(300000);
            for &(bits, count) in &ff_heavy_codes {
                writer.write_bits(black_box(bits), black_box(count));
            }
            black_box(writer.len())
        })
    });

    group.finish();
}

criterion_group!(benches, bench_bitwriter);
criterion_main!(benches);
