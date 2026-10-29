//! Test-only baseline (SOF0) JPEG writer with full control over the frame.
//!
//! zenjpeg's own encoder writes only the four standard chroma layouts, so the
//! sampling shapes a decoder must survive on crafted input — 4:1:1, fractional
//! ratios (3x1 luma over 2x1 chroma), asymmetric interleaved chroma (R 2x2, G
//! 2x2, B 1x1), oversampled single-component frames — cannot come from it. This
//! writer takes the coefficients directly: the caller supplies every block in
//! zigzag order, so DC-only content is DC-only by construction and pixel values
//! are known exactly (with every quantizer 8, a DC-only block decodes to
//! `128 + dc` on every IDCT).
//!
//! It also places marker segments anywhere before or after SOF, which is how
//! the colour-model signalling tests build APP14-after-SOF and JFIF+APP14
//! variants. Huffman coding uses the ITU T.81 Annex K luminance tables (copied
//! from `zenjpeg/src/huffman/encode.rs`) for every component.

const STD_DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const STD_AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
const STD_AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// One frame component: SOF id and sampling factors.
#[derive(Clone, Copy, Debug)]
pub struct Comp {
    pub id: u8,
    pub h: u8,
    pub v: u8,
}

/// Where a marker segment goes relative to SOF.
#[derive(Clone, Debug)]
pub enum Segment {
    /// Before SOF (right after SOI, in order).
    PreFrame(Vec<u8>),
    /// After SOF, before the tables and SOS.
    PostFrame(Vec<u8>),
}

/// Frame description. `block(c, bx, by)` returns component `c`'s block at
/// `(bx, by)` on its MCU-padded block grid, in zigzag order.
pub struct Spec<'a> {
    pub width: u32,
    pub height: u32,
    pub comps: Vec<Comp>,
    pub block: &'a dyn Fn(usize, usize, usize) -> [i16; 64],
    /// Restart interval in MCUs (0 = none).
    pub restart_interval: u16,
    /// Every quantizer entry (one table, shared by all components).
    pub quant: u8,
    pub segments: Vec<Segment>,
}

impl Spec<'_> {
    pub fn max_h(&self) -> usize {
        self.comps.iter().map(|c| c.h as usize).max().unwrap()
    }
    pub fn max_v(&self) -> usize {
        self.comps.iter().map(|c| c.v as usize).max().unwrap()
    }
    /// A one-component scan is non-interleaved (T.81 A.2.2): its MCU is one
    /// data unit whatever the sampling factors say.
    fn interleaved(&self) -> bool {
        self.comps.len() > 1
    }
    pub fn mcu_cols(&self) -> usize {
        let mcu_w = if self.interleaved() {
            8 * self.max_h()
        } else {
            8
        };
        (self.width as usize).div_ceil(mcu_w)
    }
    pub fn mcu_rows(&self) -> usize {
        let mcu_h = if self.interleaved() {
            8 * self.max_v()
        } else {
            8
        };
        (self.height as usize).div_ceil(mcu_h)
    }
}

/// Adobe APP14 segment with the given transform byte (TN 5116).
pub fn app14(transform: u8) -> Vec<u8> {
    let mut seg = vec![0xFF, 0xEE, 0x00, 0x0E];
    seg.extend_from_slice(b"Adobe");
    seg.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, transform]);
    seg
}

/// Minimal JFIF 1.01 APP0 segment.
pub fn jfif() -> Vec<u8> {
    let mut seg = vec![0xFF, 0xE0, 0x00, 0x10];
    seg.extend_from_slice(b"JFIF\0");
    seg.extend_from_slice(&[0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]);
    seg
}

struct Huff {
    code: [u16; 256],
    len: [u8; 256],
}

fn build_huff(bits: &[u8; 16], vals: &[u8]) -> Huff {
    let mut h = Huff {
        code: [0; 256],
        len: [0; 256],
    };
    let mut code = 0u16;
    let mut k = 0;
    for (l, &n) in bits.iter().enumerate() {
        for _ in 0..n {
            h.code[vals[k] as usize] = code;
            h.len[vals[k] as usize] = (l + 1) as u8;
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    h
}

struct Bits {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl Bits {
    fn put(&mut self, value: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.acc = (self.acc << len) | (value & ((1 << len) - 1));
        self.n += len;
        while self.n >= 8 {
            let byte = (self.acc >> (self.n - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.n -= 8;
        }
        self.acc &= (1 << self.n) - 1;
    }

    /// Pad the final partial byte with 1-bits (T.81 F.1.2.3).
    fn flush(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1 << pad) - 1, pad);
        }
    }
}

fn category(v: i32) -> u32 {
    32 - (v.unsigned_abs()).leading_zeros()
}

fn put_value(bits: &mut Bits, v: i32, cat: u32) {
    let raw = if v < 0 { v - 1 } else { v };
    bits.put(raw as u32, cat);
}

fn encode_block(bits: &mut Bits, block: &[i16; 64], pred: &mut i32, dc: &Huff, ac: &Huff) {
    let diff = block[0] as i32 - *pred;
    *pred = block[0] as i32;
    let cat = category(diff);
    bits.put(dc.code[cat as usize] as u32, dc.len[cat as usize] as u32);
    put_value(bits, diff, cat);

    let mut run = 0u32;
    for &z in &block[1..] {
        if z == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            bits.put(ac.code[0xF0] as u32, ac.len[0xF0] as u32);
            run -= 16;
        }
        let cat = category(z as i32);
        let sym = ((run << 4) | cat) as usize;
        bits.put(ac.code[sym] as u32, ac.len[sym] as u32);
        put_value(bits, z as i32, cat);
        run = 0;
    }
    if run > 0 {
        bits.put(ac.code[0] as u32, ac.len[0] as u32);
    }
}

/// Panic unless `seg` is one complete marker segment that may sit between SOI
/// and SOS without changing the frame: its length field matches, and it is not
/// SOI, EOI, SOS or a frame header (SOFn). Inserting a second SOF or an SOS
/// would make the fixture a different file from the one the test describes.
fn assert_plain_segment(seg: &[u8]) {
    assert!(
        seg.len() >= 4 && seg[0] == 0xFF,
        "not a marker segment: {seg:02X?}"
    );
    let m = seg[1];
    let sof = matches!(m, 0xC0..=0xCF) && !matches!(m, 0xC4 | 0xC8 | 0xCC);
    assert!(
        !sof && !matches!(m, 0xD8..=0xDA),
        "marker FF{m:02X} cannot be inserted as a plain segment"
    );
    assert_eq!(2 + be16(seg, 2), seg.len(), "FF{m:02X} length field");
}

/// Write the frame described by `spec` as a baseline interleaved JPEG.
pub fn write(spec: &Spec<'_>) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8];
    for seg in &spec.segments {
        let (Segment::PreFrame(s) | Segment::PostFrame(s)) = seg;
        assert_plain_segment(s);
    }
    for seg in &spec.segments {
        if let Segment::PreFrame(s) = seg {
            out.extend_from_slice(s);
        }
    }

    // DQT: one 8-bit table, every entry `quant`.
    out.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    out.extend(std::iter::repeat_n(spec.quant, 64));

    // SOF0
    let nf = spec.comps.len();
    assert!((1..=4).contains(&nf), "{nf} components");
    let len = 8 + 3 * nf;
    out.extend_from_slice(&[0xFF, 0xC0, 0x00, len as u8, 0x08]);
    let dim =
        |v: u32, what: &str| u16::try_from(v).unwrap_or_else(|_| panic!("{what} {v} > 65535"));
    out.extend_from_slice(&dim(spec.height, "height").to_be_bytes());
    out.extend_from_slice(&dim(spec.width, "width").to_be_bytes());
    out.push(nf as u8);
    for c in &spec.comps {
        out.extend_from_slice(&[c.id, (c.h << 4) | c.v, 0]);
    }

    for seg in &spec.segments {
        if let Segment::PostFrame(s) = seg {
            out.extend_from_slice(s);
        }
    }

    // DHT: DC table 0 and AC table 0.
    let dht_len = 2 + 17 + STD_DC_VALUES.len() + 17 + STD_AC_VALUES.len();
    out.extend_from_slice(&[0xFF, 0xC4]);
    out.extend_from_slice(&u16::try_from(dht_len).expect("DHT length").to_be_bytes());
    out.push(0x00);
    out.extend_from_slice(&STD_DC_BITS);
    out.extend_from_slice(&STD_DC_VALUES);
    out.push(0x10);
    out.extend_from_slice(&STD_AC_BITS);
    out.extend_from_slice(&STD_AC_VALUES);

    if spec.restart_interval > 0 {
        out.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04]);
        out.extend_from_slice(&spec.restart_interval.to_be_bytes());
    }

    // SOS: all components interleaved, full spectral range.
    let sos_len = 6 + 2 * nf;
    out.extend_from_slice(&[0xFF, 0xDA, 0x00, sos_len as u8, nf as u8]);
    for c in &spec.comps {
        out.extend_from_slice(&[c.id, 0x00]);
    }
    out.extend_from_slice(&[0x00, 0x3F, 0x00]);

    let dc = build_huff(&STD_DC_BITS, &STD_DC_VALUES);
    let ac = build_huff(&STD_AC_BITS, &STD_AC_VALUES);
    let mut bits = Bits {
        out: Vec::new(),
        acc: 0,
        n: 0,
    };
    let mut preds = vec![0i32; nf];
    let (mcu_cols, mcu_rows) = (spec.mcu_cols(), spec.mcu_rows());
    let ri = spec.restart_interval as usize;
    let mut rst = 0u8;
    for m in 0..mcu_cols * mcu_rows {
        if ri > 0 && m > 0 && m % ri == 0 {
            bits.flush();
            bits.out.extend_from_slice(&[0xFF, 0xD0 + rst]);
            rst = (rst + 1) % 8;
            preds.iter_mut().for_each(|p| *p = 0);
        }
        let (mx, my) = (m % mcu_cols, m / mcu_cols);
        if !spec.interleaved() {
            let block = (spec.block)(0, mx, my);
            encode_block(&mut bits, &block, &mut preds[0], &dc, &ac);
            continue;
        }
        for (ci, c) in spec.comps.iter().enumerate() {
            for v in 0..c.v as usize {
                for h in 0..c.h as usize {
                    let bx = mx * c.h as usize + h;
                    let by = my * c.v as usize + v;
                    let block = (spec.block)(ci, bx, by);
                    encode_block(&mut bits, &block, &mut preds[ci], &dc, &ac);
                }
            }
        }
    }
    bits.flush();
    out.extend_from_slice(&bits.out);
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

/// A DC-only block whose every pixel decodes to `value` when the quantizer is 8.
pub fn dc_block(value: u8) -> [i16; 64] {
    let mut b = [0i16; 64];
    b[0] = value as i16 - 128;
    b
}

// ---------------------------------------------------------------------------
// Header surgery on existing files
// ---------------------------------------------------------------------------

pub fn be16(b: &[u8], p: usize) -> usize {
    ((b[p] as usize) << 8) | b[p + 1] as usize
}

/// Walk marker segments up to the first SOS: (offset, marker, total length).
pub fn segments(jpeg: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut out = Vec::new();
    let mut p = 2;
    while p + 4 <= jpeg.len() && jpeg[p] == 0xFF {
        let m = jpeg[p + 1];
        let len = 2 + be16(jpeg, p + 2);
        out.push((p, m, len));
        if m == 0xDA {
            break;
        }
        p += len;
    }
    out
}

/// Rebuild a JPEG's header: drop every APP0/APP14 before SOS, rewrite the
/// component IDs in SOF and in every SOS, and insert `pre` after SOI and `post`
/// right after SOF. The entropy-coded data is untouched, so the result decodes
/// the same component planes under a different colour interpretation.
pub fn resignal(jpeg: &[u8], ids: [u8; 3], pre: &[Vec<u8>], post: &[Vec<u8>]) -> Vec<u8> {
    for seg in pre.iter().chain(post) {
        assert_plain_segment(seg);
    }
    let segs = segments(jpeg);
    let mut out = vec![0xFF, 0xD8];
    out.extend(pre.iter().flatten());
    let mut old_ids = [0u8; 3];
    let mut body_start = None;
    let mut frames = 0;
    for &(p, m, len) in &segs {
        if m == 0xDA {
            body_start = Some(p);
            break;
        }
        if m == 0xE0 || m == 0xEE {
            continue;
        }
        let mut seg = jpeg[p..p + len].to_vec();
        let sof = matches!(m, 0xC0..=0xCF) && !matches!(m, 0xC4 | 0xC8 | 0xCC);
        assert!(
            !sof || matches!(m, 0xC0..=0xC2),
            "FF{m:02X}: only SOF0..SOF2 frames are resignalled"
        );
        if sof {
            frames += 1;
            assert_eq!(seg[9], 3, "3-component frame");
            for i in 0..3 {
                old_ids[i] = seg[10 + 3 * i];
                seg[10 + 3 * i] = ids[i];
            }
            out.extend_from_slice(&seg);
            out.extend(post.iter().flatten());
            continue;
        }
        out.extend_from_slice(&seg);
    }
    assert_eq!(frames, 1, "one frame header before the first SOS");
    let body_start = body_start.expect("no SOS before the end of the header");
    let map = |id: u8| (0..3).find(|&i| old_ids[i] == id).map_or(id, |i| ids[i]);
    // Copy the rest, remapping the component IDs of every SOS (progressive
    // files have one per scan). Entropy data is skipped marker to marker.
    let mut rest = jpeg[body_start..].to_vec();
    let mut p = 0;
    while p + 4 <= rest.len() {
        if rest[p] != 0xFF {
            p += 1;
            continue;
        }
        let m = rest[p + 1];
        if m == 0x00 || (0xD0..=0xD7).contains(&m) || m == 0xFF {
            p += if m == 0xFF { 1 } else { 2 };
            continue;
        }
        if m == 0xD9 {
            break;
        }
        if m == 0xDA {
            let ns = rest[p + 4] as usize;
            for i in 0..ns {
                let o = p + 5 + 2 * i;
                rest[o] = map(rest[o]);
            }
        }
        p += 2 + be16(&rest, p + 2);
    }
    out.extend_from_slice(&rest);
    out
}
