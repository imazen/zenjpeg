//! `Decoder::max_memory` actually bounds header-driven allocation.
//!
//! A JPEG header is ~1 KB regardless of the frame size it declares, but the
//! decoder's coefficient storage and pixel output are both sized from those
//! declared dimensions. A 15000x8000 SOF costs ~1 KB of input and asks for
//! hundreds of megabytes of output — the classic amplification shape.
//!
//! Every test here builds such a header **from scratch** (no fixture: the file
//! is a few hundred bytes of markers) and asserts the decode is refused with a
//! typed limit error rather than served.
//!
//! Run: `cargo test -p zenjpeg --test decode_memory_limit`

use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, ErrorKind};

// ---------------------------------------------------------------------------
// Minimal JPEG header construction
// ---------------------------------------------------------------------------

/// Frame type to emit.
#[derive(Clone, Copy, PartialEq)]
enum Frame {
    /// SOF0 — baseline sequential.
    Baseline,
    /// SOF2 — progressive (always uses buffered coefficient storage).
    Progressive,
}

/// How the frame's components are split across scans.
#[derive(Clone, Copy, PartialEq)]
enum Scans {
    /// One `Ns=3` interleaved scan.
    Interleaved,
    /// Three `Ns=1` non-interleaved scans (T.81 A.2.2).
    NonInterleaved,
}

fn seg(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    let len = (payload.len() + 2) as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
}

/// A syntactically valid JPEG whose SOF declares `width x height`, carrying
/// almost no entropy data. 4:2:0 sampling (2x2, 1x1, 1x1).
///
/// The entropy segments are deliberately empty: the decoder allocates its
/// frame-sized buffers from the header *before* it discovers there is nothing
/// to decode, which is exactly the amplification being tested. Truncated
/// entropy data is zero-filled by the default `Balanced` strictness, so a
/// decode that is not refused by a limit succeeds and returns the full frame.
fn header_jpeg(frame: Frame, scans: Scans, width: u16, height: u16) -> Vec<u8> {
    let mut j = vec![0xFF, 0xD8]; // SOI

    // Two quantization tables, all-ones (valid; zero entries would be clamped).
    for tq in 0..2u8 {
        let mut p = vec![tq];
        p.extend_from_slice(&[1u8; 64]);
        seg(&mut j, 0xDB, &p);
    }

    // SOF: 8-bit, 3 components, Y=2x2 Cb=1x1 Cr=1x1, quant slots 0/1/1.
    let sof_marker = match frame {
        Frame::Baseline => 0xC0,
        Frame::Progressive => 0xC2,
    };
    let mut sof = vec![8];
    sof.extend_from_slice(&height.to_be_bytes());
    sof.extend_from_slice(&width.to_be_bytes());
    sof.push(3);
    sof.extend_from_slice(&[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    seg(&mut j, sof_marker, &sof);

    // Two trivial Huffman tables (class 0 = DC, class 1 = AC), one code each.
    // Enough to be well-formed; the scans carry no data worth decoding.
    for (class, id) in [(0u8, 0u8), (1, 0), (0, 1), (1, 1)] {
        let mut p = vec![(class << 4) | id];
        let mut counts = [0u8; 16];
        counts[0] = 1; // one 1-bit code
        p.extend_from_slice(&counts);
        p.push(0x00); // symbol 0
        seg(&mut j, 0xC4, &p);
    }

    let sos = |comps: &[(u8, u8)], ss: u8, se: u8, ah_al: u8, j: &mut Vec<u8>| {
        let mut p = vec![comps.len() as u8];
        for &(id, tables) in comps {
            p.push(id);
            p.push(tables);
        }
        p.extend_from_slice(&[ss, se, ah_al]);
        seg(j, 0xDA, &p);
        // A couple of entropy bytes so the scan is not empty at the marker level.
        j.extend_from_slice(&[0x00, 0x00]);
    };

    match (frame, scans) {
        (Frame::Baseline, Scans::Interleaved) => {
            sos(&[(1, 0x00), (2, 0x11), (3, 0x11)], 0, 63, 0x00, &mut j);
        }
        (Frame::Baseline, Scans::NonInterleaved) => {
            sos(&[(1, 0x00)], 0, 63, 0x00, &mut j);
            sos(&[(2, 0x11)], 0, 63, 0x00, &mut j);
            sos(&[(3, 0x11)], 0, 63, 0x00, &mut j);
        }
        (Frame::Progressive, _) => {
            // DC-first, interleaved: the shape every progressive file starts with.
            sos(&[(1, 0x00), (2, 0x11), (3, 0x11)], 0, 0, 0x00, &mut j);
        }
    }

    j.extend_from_slice(&[0xFF, 0xD9]); // EOI
    j
}

/// Assert the decode was refused by the configured memory cap.
#[track_caller]
fn assert_memory_limited(err: &zenjpeg::decoder::Error, what: &str) {
    match err.kind() {
        ErrorKind::ResourceLimitExceeded {
            kind,
            actual,
            limit,
        } => {
            assert_eq!(
                *kind,
                zencodec::LimitKind::Memory,
                "{what}: wrong limit kind ({kind:?}); expected Memory"
            );
            assert!(
                actual > limit,
                "{what}: reported actual {actual} is not above limit {limit}"
            );
        }
        other => panic!("{what}: expected ResourceLimitExceeded(Memory), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The amplification
// ---------------------------------------------------------------------------

/// 15000x8000 is 120 MP — right at the default `max_pixels`, so the pixel cap
/// does not catch it. The header is under a kilobyte.
const BIG_W: u16 = 15000;
const BIG_H: u16 = 8000;

/// The whole point: a tiny file must not be able to buy an unbounded decode.
#[test]
fn tiny_header_declaring_120mp_is_refused_by_a_small_memory_cap() {
    for (frame, scans, label) in [
        (Frame::Baseline, Scans::Interleaved, "baseline interleaved"),
        (
            Frame::Baseline,
            Scans::NonInterleaved,
            "baseline non-interleaved",
        ),
        (Frame::Progressive, Scans::Interleaved, "progressive"),
    ] {
        let data = header_jpeg(frame, scans, BIG_W, BIG_H);
        assert!(
            data.len() < 1024,
            "{label}: fixture should be a ~1 KB header, got {} bytes",
            data.len()
        );

        let err = Decoder::new()
            .max_memory(16 * 1024 * 1024)
            .decode(&data, Unstoppable)
            .err()
            .unwrap_or_else(|| {
                panic!(
                    "{label}: {} bytes of header declaring {BIG_W}x{BIG_H} decoded under a \
                     16 MB cap — max_memory is not enforced on this path",
                    data.len()
                )
            });
        assert_memory_limited(&err, label);
    }
}

/// The default cap is 512 MB, so the amplification is refused even when the
/// caller never touches `max_memory`.
#[test]
fn default_memory_cap_refuses_the_amplification() {
    let data = header_jpeg(Frame::Progressive, Scans::Interleaved, BIG_W, BIG_H);
    let err = Decoder::new()
        .decode(&data, Unstoppable)
        .expect_err("120 MP progressive header should exceed the 512 MB default cap");
    assert_memory_limited(&err, "default cap");
}

/// The cap must be a *ceiling*, not a constant: raising it past the requirement
/// lets the same file through, which proves the error above came from the cap
/// and not from something incidental about the fixture.
#[test]
fn raising_the_cap_admits_the_same_file() {
    // 1000x1000 4:2:0: ~23.4k blocks over three components at ~137 B/block for
    // the progressive side tables, plus 3 MB of RGB output. ~7 MB total.
    let data = header_jpeg(Frame::Progressive, Scans::Interleaved, 1000, 1000);

    let err = Decoder::new()
        .max_memory(1024 * 1024)
        .decode(&data, Unstoppable)
        .expect_err("1 MB cap must refuse a 1000x1000 frame");
    assert_memory_limited(&err, "1 MB cap");

    let img = Decoder::new()
        .max_memory(64 * 1024 * 1024)
        .decode(&data, Unstoppable)
        .expect("64 MB cap must admit a 1000x1000 frame");
    assert_eq!((img.width, img.height), (1000, 1000));
}

/// `u64::MAX` and `0` both mean unlimited (matching `max_pixels`).
#[test]
fn sentinel_values_mean_unlimited() {
    let data = header_jpeg(Frame::Progressive, Scans::Interleaved, 2000, 2000);
    for cap in [0u64, u64::MAX] {
        let img = Decoder::new()
            .max_memory(cap)
            .decode(&data, Unstoppable)
            .unwrap_or_else(|e| panic!("max_memory({cap}) should be unlimited, got {e:?}"));
        assert_eq!((img.width, img.height), (2000, 2000));
    }
}

/// A real encode/decode round trip must not trip the default cap. This is the
/// false-positive guard: a limit that rejects ordinary images is worse than one
/// that rejects nothing.
#[test]
fn ordinary_images_are_unaffected_by_the_default_cap() {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

    let (w, h) = (512u32, 384u32);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut state = 0x9E3779B97F4A7C15u64;
    for px in rgb.as_chunks_mut::<3>().0 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        px[0] = (state >> 32) as u8;
        px[1] = (state >> 40) as u8;
        px[2] = (state >> 48) as u8;
    }

    for subsampling in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        let mut enc = EncoderConfig::ycbcr(85.0, subsampling)
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .expect("encoder");
        enc.push_packed(&rgb, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");

        let img = Decoder::new()
            .decode(&jpeg, Unstoppable)
            .expect("default cap must admit an ordinary 512x384 image");
        assert_eq!((img.width, img.height), (w, h));

        // And through the coefficient path, which allocates the most.
        Decoder::new()
            .decode_coefficients(&jpeg, Unstoppable)
            .expect("default cap must admit coefficient decode of an ordinary image");
    }
}

/// The charge accumulates across a decode, so a multi-scan file cannot spend
/// the same budget three times over.
#[test]
fn budget_accumulates_across_scans() {
    // Sized so that ONE component's coefficient storage fits under the cap but
    // all three together do not: 1200x1200 4:2:0 is 22500 luma blocks +
    // 2 x 5625 chroma blocks; at 129 B/block that is ~2.9 MB luma and ~0.7 MB
    // per chroma plane, so a 3 MB cap admits luma alone and must refuse the
    // frame.
    let data = header_jpeg(Frame::Baseline, Scans::NonInterleaved, 1200, 1200);
    let err = Decoder::new()
        .max_memory(3 * 1024 * 1024)
        .decode(&data, Unstoppable)
        .expect_err("per-scan budgets must not reset");
    assert_memory_limited(&err, "accumulation");
}
