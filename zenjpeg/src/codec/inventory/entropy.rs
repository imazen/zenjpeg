//! Count-only entropy pass: where a Huffman scan's entropy-coded data
//! really ends, and where each restart interval's does, as the decoder
//! consumes it. It decodes symbols and skips their value bits; it stores no
//! coefficients and produces no pixels, so it runs in O(scan bytes) time
//! with O(1) state (one byte of bits, the EOB run, the RSTn count).
//!
//! It mirrors the decoder's rules (entropy/decoder.rs, foundation/bitstream.rs):
//!
//! - bytes are read with unstuffing: `FF 00` is a data `FF`, and fill
//!   bytes before the `00` belong to it; `FF` followed by anything else
//!   (after fills) is a marker, and the real bits end there (the decoder
//!   pads the rest of the scan with zero bits);
//! - a symbol is the canonical code `HuffmanDecodeTable` assigns, read bit
//!   by bit (the decoder's fast paths give the same symbol for a valid
//!   code); a code that matches nothing within 16 real bits is invalid;
//! - sequential blocks: a DC category (above 16 is an error) and its value
//!   bits, then AC symbols until EOB or 63 coefficients, ZRL skipping 16;
//!   a run past the block consumes its value bits and ends the block
//!   (`AcIndexOverflow`); any other `r/0` symbol ends the block;
//! - progressive DC first scans: a DC symbol and value bits per block; DC
//!   refinement: one bit per block; AC first scans: symbols in `Ss..=Se`
//!   per block, with `EOBn` runs covering whole blocks;
//! - with a restart interval, after every interval but the last the
//!   decoder skips whatever is left before the next marker and expects
//!   `RSTn` there (n counting from 0, mod 8).
//!
//! AC refinement scans need per-coefficient history and arithmetic-coded
//! scans the arithmetic decoder's state; those are not counted.

use alloc::vec::Vec;
use core::ops::Range;

use crate::huffman::HuffmanDecodeTable;

/// One component of a scan.
#[derive(Clone, Copy)]
pub(super) struct ScanComp<'t> {
    /// Sampling factors.
    pub h: u8,
    pub v: u8,
    pub dc: &'t HuffmanDecodeTable,
    pub ac: &'t HuffmanDecodeTable,
}

/// A Huffman scan to count.
pub(super) struct Scan<'a, 't> {
    pub comps: &'a [ScanComp<'t>],
    /// The frame's largest sampling factors.
    pub hmax: u8,
    pub vmax: u8,
    pub width: u32,
    pub height: u32,
    pub progressive: bool,
    pub ss: u8,
    pub se: u8,
    pub ah: u8,
    /// MCUs per restart interval, 0 = none.
    pub restart: u16,
}

/// How the counted scan ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum End {
    /// Every MCU decoded from real bits.
    Complete,
    /// The real bits ran out (a marker, or the end of the data) before the
    /// last MCU; the decoder pads the rest with zero bits.
    Exhausted,
    /// An interval ended where the marker at `at` is not its `RSTn`: a
    /// non-Strict decoder scans forward for any RSTn (`resync_to_restart`),
    /// a Strict one fails.
    Resync { at: usize },
    /// A Huffman code at `at` matches no symbol, more than 16 bytes before
    /// the next marker: Balanced and Strict decoders fail, Lenient and
    /// Permissive end the block there.
    InvalidCode { at: usize },
    /// A DC category above 16 at `at`: every decoder fails.
    BadDcCategory { at: usize },
    /// Not counted: AC refinement scans, geometry the decoder rejects, or
    /// an invalid code within 16 bytes of a marker (whether the decoder
    /// fails or ends the scan depends on how far it read ahead).
    Unsupported,
}

/// What the pass found.
pub(super) struct Count {
    pub end: End,
    /// Bytes after an interval's (or the scan's) last data byte that the
    /// decoder never decodes, before the next RSTn or marker, and whether
    /// they follow the scan's last MCU. Bit padding in the last data byte,
    /// and the `00` stuffed after a final `FF`, are part of the data.
    pub tails: Vec<(Range<usize>, bool)>,
    /// When the real bits ran out at a marker inside the scan's range (an
    /// RSTn the decoder does not expect): where its fill run starts. The
    /// decoder ends the scan there, ignores the RSTn and skips what
    /// follows as stray bytes.
    pub stop_at: Option<usize>,
    /// A run past the end of a block (`AcIndexOverflow`, an error when
    /// Strict).
    pub ac_overflow: bool,
    /// An RSTn with the wrong number (`RestartMarkerResync`, an error when
    /// Strict; other levels accept it).
    pub rst_mismatch: bool,
}

/// Why decoding stopped early.
enum Stop {
    /// No real bit is left: a marker at `marker` (`None`: the data ended).
    Out,
    Invalid(usize),
    BadDc(usize),
}

/// Reads bits a byte at a time, the way `BitReader::read_byte_slow`
/// unstuffs them, and remembers where the last bit it handed out came from.
struct Bits<'a> {
    data: &'a [u8],
    /// Next byte to load.
    pos: usize,
    /// Where the scan's own range ends (its terminating marker, or the end
    /// of the data).
    limit: usize,
    byte: u8,
    left: u8,
    /// End (exclusive) of the bytes the loaded data byte came from: after
    /// the stuffed `00` for a data `FF`.
    byte_end: usize,
    /// `byte_end` of the byte the last consumed bit came from.
    consumed_end: usize,
    /// The marker the bits ran into: `(first FF of its fill run, last FF,
    /// code)`.
    marker: Option<(usize, usize, u8)>,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], start: usize, limit: usize) -> Self {
        Self {
            data,
            pos: start,
            limit,
            byte: 0,
            left: 0,
            byte_end: start,
            consumed_end: start,
            marker: None,
        }
    }

    fn load(&mut self) -> bool {
        if self.marker.is_some() {
            return false;
        }
        if self.pos >= self.limit {
            self.end_marker();
            return false;
        }
        let b = self.data[self.pos];
        if b != 0xFF {
            self.byte = b;
            self.left = 8;
            self.pos += 1;
            self.byte_end = self.pos;
            return true;
        }
        let mut q = self.pos + 1;
        while q < self.limit && self.data[q] == 0xFF {
            q += 1;
        }
        if q >= self.limit {
            // Fill bytes run into the scan's terminating marker.
            self.end_marker();
            return false;
        }
        if self.data[q] == 0x00 {
            self.byte = 0xFF;
            self.left = 8;
            self.pos = q + 1;
            self.byte_end = self.pos;
            return true;
        }
        self.marker = Some((self.pos, q - 1, self.data[q]));
        false
    }

    /// The scan's own terminating marker at `limit`, unless the data ends
    /// there.
    fn end_marker(&mut self) {
        if self.limit < self.data.len() {
            let code = self.data.get(self.limit + 1).copied().unwrap_or(0);
            self.marker = Some((self.limit, self.limit, code));
        }
    }

    fn bit(&mut self) -> Result<u32, Stop> {
        if self.left == 0 && !self.load() {
            return Err(Stop::Out);
        }
        self.left -= 1;
        self.consumed_end = self.byte_end;
        Ok(u32::from(self.byte >> self.left) & 1)
    }

    fn bits(&mut self, n: u8) -> Result<(), Stop> {
        for _ in 0..n {
            self.bit()?;
        }
        Ok(())
    }

    fn value(&mut self, n: u8) -> Result<u32, Stop> {
        let mut v = 0;
        for _ in 0..n {
            v = v << 1 | self.bit()?;
        }
        Ok(v)
    }

    /// A symbol: the decoder's bit-by-bit fallback, which the fast paths
    /// agree with on every valid code.
    fn symbol(&mut self, t: &HuffmanDecodeTable) -> Result<u8, Stop> {
        let at = self.consumed_end;
        let mut code: i32 = 0;
        for len in 1..=16 {
            code = code << 1 | self.bit()? as i32;
            if code <= t.maxcode[len] {
                let idx = code + t.valoffset[len];
                if let Some(&s) = usize::try_from(idx).ok().and_then(|i| t.values.get(i)) {
                    return Ok(s);
                }
            }
        }
        Err(Stop::Invalid(at))
    }

    /// Forget the partial byte and any marker state at a restart boundary,
    /// and continue at `pos`.
    fn restart_at(&mut self, pos: usize) {
        self.pos = pos;
        self.left = 0;
        self.marker = None;
        self.byte_end = pos;
        self.consumed_end = pos;
    }
}

/// The next marker at or after `from` in `data[..limit]`: `(first FF of its
/// fill run, last FF, code)`, skipping stuffed `FF 00`. `None` when the
/// scan's range ends first (its terminating marker is at `limit`).
fn next_marker(data: &[u8], from: usize, limit: usize) -> Option<(usize, usize, u8)> {
    let mut pos = from;
    while pos < limit {
        let ff = pos + memchr::memchr(0xFF, &data[pos..limit])?;
        let mut q = ff + 1;
        while q < limit && data[q] == 0xFF {
            q += 1;
        }
        if q >= limit {
            return None;
        }
        if data[q] == 0x00 {
            pos = q + 1;
            continue;
        }
        return Some((ff, q - 1, data[q]));
    }
    None
}

/// Count `scan`'s entropy-coded data, which starts at `start`; `limit` is
/// where the walker ends it (its terminating marker, or the end of the
/// data).
pub(super) fn count(data: &[u8], start: usize, limit: usize, scan: &Scan<'_, '_>) -> Count {
    let mut out = Count {
        end: End::Unsupported,
        tails: Vec::new(),
        stop_at: None,
        ac_overflow: false,
        rst_mismatch: false,
    };
    // `decode_progressive_scan`: a DC scan has Ss = Se = 0, every other
    // scan is an AC scan of one component.
    let ac_scan = scan.progressive && !(scan.ss == 0 && scan.se == 0);
    if (ac_scan && (scan.ah > 0 || scan.comps.len() != 1))
        || scan.comps.is_empty()
        || scan.width == 0
        || scan.height == 0
        || scan.hmax == 0
        || scan.vmax == 0
    {
        return out;
    }
    let div = |a: u64, b: u64| a.div_ceil(b);
    let (w, h) = (u64::from(scan.width), u64::from(scan.height));
    let (hmax, vmax) = (u64::from(scan.hmax), u64::from(scan.vmax));
    // MCUs, and blocks per MCU per component (A.2.2 / A.2.3).
    let (mcus, per_mcu): (u64, [u64; 4]) = if scan.comps.len() == 1 {
        let c = scan.comps[0];
        let cw = div(w * u64::from(c.h), hmax);
        let ch = div(h * u64::from(c.v), vmax);
        (div(cw, 8) * div(ch, 8), [1, 0, 0, 0])
    } else {
        let mut per = [0u64; 4];
        for (k, c) in scan.comps.iter().enumerate().take(4) {
            per[k] = u64::from(c.h) * u64::from(c.v);
        }
        (div(w, 8 * hmax) * div(h, 8 * vmax), per)
    };
    let restart = u64::from(scan.restart);
    let mut bits = Bits::new(data, start, limit);
    let mut eobrun: u64 = 0;
    let mut rst = 0u8;
    let mut mcu = 0u64;
    while mcu < mcus {
        let interval_end = if restart > 0 {
            ((mcu / restart) + 1) * restart
        } else {
            mcus
        }
        .min(mcus);
        // One interval.
        let mut stopped = None;
        'interval: while mcu < interval_end {
            for (k, c) in scan.comps.iter().enumerate() {
                for _ in 0..per_mcu[k.min(3)] {
                    let r = block(&mut bits, scan, c, &mut eobrun, &mut out.ac_overflow);
                    if let Err(s) = r {
                        stopped = Some(s);
                        break 'interval;
                    }
                }
            }
            mcu += 1;
        }
        let last = interval_end >= mcus;
        match stopped {
            Some(Stop::Invalid(at)) => {
                // `decode_huffman_symbol_lenient` takes an invalid code as
                // the end of the scan once its read-ahead (up to 8 bytes
                // in the bit buffer) has reached a marker. Near a marker
                // the outcome depends on how far it read, so claim nothing.
                let next = next_marker(data, bits.pos, limit).map_or(limit, |(first, _, _)| first);
                out.end = if next.saturating_sub(bits.pos) <= 16 {
                    End::Unsupported
                } else {
                    End::InvalidCode { at }
                };
                return out;
            }
            Some(Stop::BadDc(at)) => {
                out.end = End::BadDcCategory { at };
                return out;
            }
            Some(Stop::Out) => {
                // The real bits ran out inside this interval: the decoder
                // pads it with zero bits. No tail here.
                if last {
                    out.end = End::Exhausted;
                    out.stop_at = bits
                        .marker
                        .map(|(first, _, _)| first)
                        .filter(|&m| m < limit);
                    return out;
                }
                match bits.marker {
                    Some((_, at, code)) if (0xD0..=0xD7).contains(&code) => {
                        out.rst_mismatch |= code != 0xD0 + rst;
                        rst = (rst + 1) & 7;
                        mcu = interval_end;
                        eobrun = 0;
                        bits.restart_at(at + 2);
                        continue;
                    }
                    Some((at, _, _)) => {
                        out.end = End::Resync { at };
                        return out;
                    }
                    None => {
                        out.end = End::Exhausted;
                        return out;
                    }
                }
            }
            None => {}
        }
        // The interval's data is complete: what follows up to the next
        // marker is never decoded. After the last MCU that is everything
        // up to the scan's terminating marker: the decoder skips it as
        // stray bytes, a trailing RSTn included.
        let data_end = bits.consumed_end;
        if last {
            if limit > data_end {
                out.tails.push((data_end..limit, true));
            }
            out.end = End::Complete;
            return out;
        }
        let next = next_marker(data, bits.pos.max(data_end), limit);
        let tail_end = next.map_or(limit, |(first, _, _)| first);
        if tail_end > data_end {
            out.tails.push((data_end..tail_end, false));
        }
        match next {
            Some((_, at, code)) if (0xD0..=0xD7).contains(&code) => {
                out.rst_mismatch |= code != 0xD0 + rst;
                rst = (rst + 1) & 7;
                eobrun = 0;
                bits.restart_at(at + 2);
            }
            _ => {
                out.end = End::Resync { at: tail_end };
                return out;
            }
        }
    }
    out.end = End::Complete;
    out
}

/// One block (one data unit) of `scan`.
fn block(
    bits: &mut Bits<'_>,
    scan: &Scan<'_, '_>,
    c: &ScanComp<'_>,
    eobrun: &mut u64,
    ac_overflow: &mut bool,
) -> Result<(), Stop> {
    if scan.progressive {
        if scan.ss == 0 && scan.se == 0 {
            if scan.ah > 0 {
                // DC refinement: one bit.
                return bits.bits(1);
            }
            return dc(bits, c.dc);
        }
        // AC first scan.
        if *eobrun > 0 {
            *eobrun -= 1;
            return Ok(());
        }
        let mut k = u32::from(scan.ss);
        while k <= u32::from(scan.se) {
            let rs = bits.symbol(c.ac)?;
            let (r, s) = (rs >> 4, rs & 0x0F);
            if s == 0 {
                if r < 15 {
                    // EOBn: this block and 2^r - 1 + extra bits more.
                    *eobrun = (1u64 << r) - 1 + u64::from(bits.value(r)?);
                    break;
                }
                k += 16;
                continue;
            }
            k += u32::from(r);
            if k > u32::from(scan.se) {
                // Run past the band: the value bits are read, the block ends.
                *ac_overflow = true;
                return bits.bits(s);
            }
            bits.bits(s)?;
            k += 1;
        }
        return Ok(());
    }
    dc(bits, c.dc)?;
    let mut k = 1u32;
    while k < 64 {
        let rs = bits.symbol(c.ac)?;
        let (r, s) = (rs >> 4, rs & 0x0F);
        if s == 0 {
            if r == 15 {
                k += 16;
                continue;
            }
            // EOB, or an r/0 symbol the decoder takes as one.
            break;
        }
        k += u32::from(r);
        if k >= 64 {
            *ac_overflow = true;
            bits.bits(s)?;
            break;
        }
        bits.bits(s)?;
        k += 1;
    }
    Ok(())
}

fn dc(bits: &mut Bits<'_>, t: &HuffmanDecodeTable) -> Result<(), Stop> {
    let at = bits.consumed_end;
    let cat = bits.symbol(t)?;
    if cat > 16 {
        return Err(Stop::BadDc(at));
    }
    bits.bits(cat)
}
