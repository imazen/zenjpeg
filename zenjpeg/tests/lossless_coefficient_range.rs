//! Lossless re-encoding of coefficients a crafted stream decodes to.
//!
//! The decoder keeps DCT coefficients modulo 2^16 (the DC prediction wraps,
//! as libjpeg's `(JCOEF)` store does), so DC values can sit anywhere in `i16`
//! and neighbouring blocks can differ by more than `i16` holds. The
//! re-encoders take DC differences modulo 2^16 too, which reproduces every
//! stored coefficient, and refuse -32768 (an AC value or a DC difference),
//! the one value no Huffman category holds and the one a flip cannot negate.
//! Found by `fuzz_lossless` (`fuzz/regression/lossless-dc-difference-overflow`,
//! `fuzz/regression/lossless-flip-negates-minus-32768`).

use enough::Unstoppable;
use zenjpeg::decoder::{DecodeError, Decoder, ErrorKind};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestructureConfig, TransformConfig, restructure,
    transform,
};

/// MSB-first entropy-coded segment writer with 0xFF stuffing.
#[derive(Default)]
struct Bits {
    out: Vec<u8>,
    acc: u8,
    n: u8,
}

impl Bits {
    fn put(&mut self, value: u32, len: u32) {
        for i in (0..len).rev() {
            self.acc = (self.acc << 1) | ((value >> i) & 1) as u8;
            self.n += 1;
            if self.n == 8 {
                self.out.push(self.acc);
                if self.acc == 0xFF {
                    self.out.push(0);
                }
                self.acc = 0;
                self.n = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        while self.n != 0 {
            self.put(1, 1);
        }
        self.out
    }
}

fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let len = (payload.len() + 2) as u16;
    let mut out = vec![0xFF, marker, (len >> 8) as u8, len as u8];
    out.extend_from_slice(payload);
    out
}

fn category(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// `(category, extra bits)` of a difference or AC value.
fn magnitude(v: i32) -> (u32, u32) {
    let cat = category(v);
    let extra = if v >= 0 { v } else { v - 1 } as u32 & ((1u32 << cat) - 1);
    (cat, extra)
}

/// DC table: symbols 0..=15, each a 5-bit code equal to the symbol.
fn dc_table() -> Vec<u8> {
    let mut p = vec![0x00];
    let mut counts = [0u8; 16];
    counts[4] = 16;
    p.extend_from_slice(&counts);
    p.extend(0..16u8);
    segment(0xC4, &p)
}

/// AC table `0x10 | id` with the given symbols, each a code of `len` bits.
fn ac_table(symbols: &[u8], len: usize) -> Vec<u8> {
    let mut p = vec![0x10];
    let mut counts = [0u8; 16];
    counts[len - 1] = symbols.len() as u8;
    p.extend_from_slice(&counts);
    p.extend_from_slice(symbols);
    segment(0xC4, &p)
}

fn header(sof: u8, width: u16, height: u16) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8];
    let mut dqt = vec![0x00];
    dqt.extend_from_slice(&[1u8; 64]);
    out.extend(segment(0xDB, &dqt));
    let [wh, wl] = width.to_be_bytes();
    let [hh, hl] = height.to_be_bytes();
    out.extend(segment(sof, &[8, hh, hl, wh, wl, 1, 1, 0x11, 0]));
    out
}

fn sos(ss: u8, se: u8, al: u8) -> Vec<u8> {
    segment(0xDA, &[1, 1, 0x00, ss, se, al])
}

/// Baseline 8-bit grayscale, one block per DC difference (raster order), all
/// AC zero.
fn baseline_gray(width: u16, height: u16, dc_diffs: &[i32]) -> Vec<u8> {
    let mut out = header(0xC0, width, height);
    out.extend(dc_table());
    out.extend(ac_table(&[0x00], 1));
    out.extend(sos(0, 63, 0));
    let mut bits = Bits::default();
    for &d in dc_diffs {
        let (cat, extra) = magnitude(d);
        bits.put(cat, 5);
        bits.put(extra, cat);
        bits.put(0, 1); // EOB
    }
    out.extend(bits.finish());
    out.extend([0xFF, 0xD9]);
    out
}

/// Progressive 8x8 grayscale whose zigzag coefficient `k` decodes to
/// `-4 << 13 == -32768`: a DC scan, then an AC first scan over band `k..=k`
/// with Al = 13.
fn progressive_ac_minus_32768(k: u8) -> Vec<u8> {
    let mut out = header(0xC2, 8, 8);
    out.extend(dc_table());
    out.extend(ac_table(&[0x00, 0x03], 2));
    out.extend(sos(0, 0, 0));
    let mut bits = Bits::default();
    bits.put(0, 5); // DC difference 0
    out.extend(bits.finish());
    out.extend(sos(k, k, 13));
    let mut bits = Bits::default();
    let (cat, extra) = magnitude(-4);
    bits.put(0b01, 2); // symbol 0x03: run 0, size 3
    bits.put(extra, cat);
    out.extend(bits.finish());
    out.extend([0xFF, 0xD9]);
    out
}

fn coefficients(jpeg: &[u8]) -> (usize, usize, Vec<i16>) {
    let c = Decoder::new()
        .decode_coefficients(jpeg, Unstoppable)
        .expect("decode coefficients");
    let comp = &c.components[0];
    (comp.blocks_wide, comp.blocks_high, comp.coeffs.clone())
}

fn dcs(coeffs: &[i16]) -> Vec<i16> {
    coeffs.as_chunks::<64>().0.iter().map(|b| b[0]).collect()
}

fn config(transform: LosslessTransform) -> TransformConfig {
    TransformConfig {
        transform,
        edge_handling: EdgeHandling::RejectPartialBlocks,
    }
}

fn restructured(
    jpeg: &[u8],
    output_mode: OutputMode,
    transform: LosslessTransform,
) -> Result<Vec<u8>, DecodeError> {
    let rc = RestructureConfig {
        output_mode,
        transform: Some(config(transform)),
        ..RestructureConfig::default()
    };
    restructure(jpeg, &rc, Unstoppable)
}

fn assert_out_of_range(result: Result<Vec<u8>, DecodeError>, what: &str) {
    let err = match result {
        Ok(out) => panic!(
            "{what}: re-encoded an uncodable coefficient ({} bytes) instead of refusing it",
            out.len()
        ),
        Err(err) => err,
    };
    assert!(
        matches!(err.kind(), ErrorKind::InvalidJpegData { .. }),
        "{what}: {err:?}"
    );
}

/// 24 blocks in a row, each DC 2047 above the last: the decoded DC passes
/// 32767 at block 16 and wraps. Every re-encode reproduces the coefficients.
#[test]
fn dc_differences_wrap_like_the_decoder() {
    let jpeg = baseline_gray(24 * 8, 8, &[2047; 24]);
    let (bw, bh, input) = coefficients(&jpeg);
    assert_eq!((bw, bh), (24, 1));
    let expected: Vec<i16> = (1..=24).map(|k| (2047 * k) as i16).collect();
    assert_eq!(dcs(&input), expected, "the decoder wraps the DC prediction");

    let cases: [(&str, Result<Vec<u8>, DecodeError>); 5] = [
        (
            "transform(None)",
            transform(&jpeg, &config(LosslessTransform::None), Unstoppable),
        ),
        (
            "transform(Rotate90)",
            transform(&jpeg, &config(LosslessTransform::Rotate90), Unstoppable),
        ),
        (
            "transform(Rotate180)",
            transform(&jpeg, &config(LosslessTransform::Rotate180), Unstoppable),
        ),
        (
            "restructure(Sequential)",
            restructured(&jpeg, OutputMode::Sequential, LosslessTransform::None),
        ),
        (
            "restructure(Progressive)",
            restructured(&jpeg, OutputMode::Progressive, LosslessTransform::None),
        ),
    ];
    for (what, result) in cases {
        let out = result.unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, _, got) = coefficients(&out);
        let want: Vec<i16> = if what == "transform(Rotate180)" {
            expected.iter().rev().copied().collect()
        } else {
            // Rotate90 turns the row into a column in the same order.
            expected.clone()
        };
        assert_eq!(dcs(&got), want, "{what}");
        assert!(
            got.as_chunks::<64>()
                .0
                .iter()
                .all(|b| b[1..].iter().all(|&c| c == 0)),
            "{what}"
        );
    }
}

/// 2x2 blocks with DC -16384, 0 / 0, 16384: every source difference fits in
/// 15 bits, but rotating 90 degrees orders the blocks C A / D B and puts
/// -16384 right before 16384, a difference of 32768, which wraps to -32768.
#[test]
fn a_dc_difference_of_32768_is_refused() {
    let jpeg = baseline_gray(16, 16, &[-16384, 16384, 0, 16384]);
    let (_, _, input) = coefficients(&jpeg);
    assert_eq!(dcs(&input), [-16384, 0, 0, 16384]);

    // The source's own order still codes, baseline and progressive.
    let same = transform(&jpeg, &config(LosslessTransform::None), Unstoppable).unwrap();
    assert_eq!(coefficients(&same).2, input);
    let same = restructured(&jpeg, OutputMode::Progressive, LosslessTransform::None).unwrap();
    assert_eq!(coefficients(&same).2, input);

    assert_out_of_range(
        transform(&jpeg, &config(LosslessTransform::Rotate90), Unstoppable),
        "transform(Rotate90)",
    );
    assert_out_of_range(
        restructured(&jpeg, OutputMode::Sequential, LosslessTransform::Rotate90),
        "restructure(Sequential, Rotate90)",
    );
    assert_out_of_range(
        restructured(&jpeg, OutputMode::Progressive, LosslessTransform::Rotate90),
        "restructure(Progressive, Rotate90)",
    );
}

/// An AC coefficient of -32768 has no category; baseline and progressive
/// re-encodes both refuse it instead of writing symbol `run << 4 | 16`.
#[test]
fn an_ac_value_of_minus_32768_is_refused() {
    let jpeg = progressive_ac_minus_32768(1);
    let (_, _, input) = coefficients(&jpeg);
    assert_eq!(input[1], i16::MIN, "the decoder shifts -4 by Al = 13");
    assert_out_of_range(
        transform(&jpeg, &config(LosslessTransform::None), Unstoppable),
        "transform(None)",
    );
    assert_out_of_range(
        restructured(&jpeg, OutputMode::Progressive, LosslessTransform::None),
        "restructure(Progressive)",
    );
}

/// Zigzag 6 is row 0, column 3, which a horizontal flip negates. -32768 has
/// no negation in `i16`, so the flip fails; it used to wrap back to -32768
/// (a panic with overflow checks), and the progressive writer, whose band
/// 3..=63 starts at Al = 2, wrote it into a file that decodes it as 32767.
/// Untransformed, the progressive writer refuses it as the baseline writer
/// does.
#[test]
fn a_flipped_minus_32768_is_refused() {
    let jpeg = progressive_ac_minus_32768(6);
    let (_, _, input) = coefficients(&jpeg);
    assert_eq!(input[6], i16::MIN);
    for (what, result) in [
        (
            "transform(FlipHorizontal)",
            transform(
                &jpeg,
                &config(LosslessTransform::FlipHorizontal),
                Unstoppable,
            ),
        ),
        (
            "restructure(Progressive, FlipHorizontal)",
            restructured(
                &jpeg,
                OutputMode::Progressive,
                LosslessTransform::FlipHorizontal,
            ),
        ),
        (
            "restructure(Progressive, None)",
            restructured(&jpeg, OutputMode::Progressive, LosslessTransform::None),
        ),
    ] {
        assert_out_of_range(result, what);
    }
}
