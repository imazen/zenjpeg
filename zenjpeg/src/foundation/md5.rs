//! Minimal MD5 (RFC 1321) for Extended XMP GUIDs (XMP Specification Part 3).
//!
//! The encoder names an extended XMP packet by its digest, and the metadata
//! reader verifies a reassembled packet against its GUID. [`Md5`] digests
//! input in pieces with a fixed 64-byte buffer, so a packet split across
//! segments is hashed in place, without copying or allocating. Replaces the
//! `md-5` crate dependency (8 transitive deps).

const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

const K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// Incremental MD5: [`update`](Self::update) any number of times, then
/// [`finish`](Self::finish). No allocation.
#[derive(Clone)]
pub(crate) struct Md5 {
    state: [u32; 4],
    block: [u8; 64],
    /// Bytes buffered in `block` (always < 64 between calls).
    buffered: usize,
    /// Total message length in bytes.
    length: u64,
}

impl Md5 {
    pub(crate) fn new() -> Self {
        Self {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            block: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.block[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered < 64 {
                return;
            }
            let block = self.block;
            self.compress(&block);
            self.buffered = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            self.compress(block);
        }
        self.block[..rest.len()].copy_from_slice(rest);
        self.buffered = rest.len();
    }

    pub(crate) fn finish(mut self) -> [u8; 16] {
        let bit_len = self.length.wrapping_mul(8);
        // Padding: 0x80, zeros to 56 mod 64, then the bit length.
        let pad_len = if self.buffered < 56 {
            56 - self.buffered
        } else {
            120 - self.buffered
        };
        let mut pad = [0u8; 72];
        pad[0] = 0x80;
        self.update(&pad[..pad_len]);
        self.update(&bit_len.to_le_bytes());
        debug_assert_eq!(self.buffered, 0);
        let mut out = [0u8; 16];
        for (dst, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *dst = word.to_le_bytes();
        }
        out
    }

    fn compress(&mut self, chunk: &[u8; 64]) {
        let mut m = [0u32; 16];
        for (word, bytes) in m.iter_mut().zip(chunk.as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*bytes);
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i {
                0..16 => ((b & c) | (!b & d), i),
                16..32 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..48 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// MD5 of `data`, returning 16 bytes.
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut h = Md5::new();
    h.update(data);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5_empty() {
        let hash = md5(b"");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn test_md5_abc() {
        let hash = md5(b"abc");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "900150983cd24fb0d6963f7d28e17f72");
    }

    /// Splitting the input at any point gives the one-shot digest, across
    /// every padding boundary (lengths around 55/56/63/64 bytes).
    #[test]
    fn test_md5_incremental_matches_one_shot() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 128, 300] {
            let whole = md5(&data[..len]);
            for split in 0..=len {
                let mut h = Md5::new();
                h.update(&data[..split]);
                h.update(&data[split..len]);
                assert_eq!(h.finish(), whole, "len {len} split {split}");
            }
            let mut bytewise = Md5::new();
            for b in &data[..len] {
                bytewise.update(core::slice::from_ref(b));
            }
            assert_eq!(bytewise.finish(), whole, "len {len} bytewise");
        }
    }

    #[test]
    fn test_md5_rfc1321_suite() {
        // RFC 1321 A.5 test suite.
        let cases: [(&[u8], &str); 4] = [
            (b"a", "0cc175b9c0f1b6a831c399e269772661"),
            (b"message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ];
        for (input, expected) in cases {
            let hex: String = md5(input).iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, expected);
        }
    }

    #[test]
    fn test_md5_long() {
        let hash = md5(b"The quick brown fox jumps over the lazy dog");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "9e107d9d372bb6826bd81d3542a419d6");
    }
}
