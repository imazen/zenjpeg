//! Baseline (SOF0) JPEGs written as one non-interleaved scan per component.
//!
//! ISO/IEC 10918-1 A.2 lets a sequential frame be split into several scans.
//! When a scan carries a single component (`Ns=1`) it is *non-interleaved*:
//! per A.2.2 the MCU is one data unit, and the scan contains
//! `ceil(x_i/8) * ceil(y_i/8)` data units in raster order over that
//! component's own block grid — **not** the frame's interleaved MCU grid.
//! Real encoders emit this: `cjpeg -scans` with a `0;\n1;\n2;` script does,
//! and `internal/jpegli-cpp/testdata/jxl/flower/flower_small.q85_420_non_interleaved.jpg`
//! is exactly this shape.
//!
//! # Oracle
//!
//! The fixtures come in pairs. Each pair is the *same source image* encoded by
//! the same libjpeg-turbo `cjpeg` invocation at the same quality and sampling
//! factors — the only difference is whether the coefficients were serialised as
//! one interleaved scan or three non-interleaved scans. Same DCT, same quant
//! tables, therefore identical coefficients, therefore **decoding them must
//! produce byte-identical pixels**. `djpeg` confirms this holds for the
//! reference decoder on all three pairs.
//!
//! Provenance (libjpeg-turbo 3.x `cjpeg`, source `src88.ppm` = deterministic
//! 88x54 xorshift noise + saturated colour patches + flat-chroma bands):
//!
//! ```text
//! printf '0: 0 63 0 0;\n1: 0 63 0 0;\n2: 0 63 0 0;\n' > ni.scan
//! cjpeg -quality 88 -sample 2x2,1x1,1x1 -scans ni.scan -outfile ni420_88x54.jpg  src88.ppm
//! cjpeg -quality 88 -sample 2x2,1x1,1x1             -outfile int420_88x54.jpg src88.ppm
//! ...likewise for -sample 1x1,1x1,1x1 (444) and 2x1,1x1,1x1 (422)
//! ```
//!
//! 88x54 is deliberate: for 4:2:0 the luma *true* grid is
//! `ceil(88/8) x ceil(54/8) = 11 x 7` while the padded interleaved MCU grid is
//! `2*ceil(88/16) x 2*ceil(54/16) = 12 x 8`. A decoder that walks the
//! interleaved grid for a non-interleaved scan reads the wrong number of data
//! units in the wrong order.
//!
//! Run: `cargo test -p zenjpeg --test non_interleaved_scans`

use enough::Unstoppable;
use std::path::PathBuf;
use zenjpeg::decode::Decoder;

fn fixture(name: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata/non_interleaved")
        .join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read fixture {}: {e}", p.display()))
}

fn decode_rgb(data: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = Decoder::new().output_format(zenjpeg::types::PixelFormat::Rgb);
    let img = decoder.decode(data, Unstoppable).expect("decode");
    let px = img.pixels_u8().expect("u8 pixels").to_vec();
    (img.width, img.height, px)
}

/// Largest absolute per-channel difference, and the count of differing bytes.
fn diff_stats(a: &[u8], b: &[u8]) -> (u8, usize) {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    let mut max = 0u8;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x.abs_diff(*y);
        if d != 0 {
            n += 1;
        }
        max = max.max(d);
    }
    (max, n)
}

/// Mean absolute deviation of a channel from its own mean, over the whole image.
///
/// Chroma that has been dropped entirely leaves R==G==B on every pixel; this is
/// the check that the *colour* survived, independent of the pixel-exact
/// comparison below (which would also pass if both files decoded to grey).
fn chroma_energy(rgb: &[u8]) -> f64 {
    let mut acc = 0f64;
    for px in rgb.as_chunks::<3>().0 {
        let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
        let mean = (r + g + b) / 3.0;
        acc += (r - mean).abs() + (g - mean).abs() + (b - mean).abs();
    }
    acc / (rgb.len() as f64 / 3.0)
}

fn check_pair(ni: &str, int: &str, label: &str) {
    let ni_data = fixture(ni);
    let int_data = fixture(int);

    let (nw, nh, ni_px) = decode_rgb(&ni_data);
    let (iw, ih, int_px) = decode_rgb(&int_data);

    assert_eq!((nw, nh), (88, 54), "{label}: non-interleaved dimensions");
    assert_eq!((iw, ih), (88, 54), "{label}: interleaved dimensions");

    // The interleaved member of the pair is the well-tested path; if IT has no
    // chroma the fixture is wrong, not the decoder.
    let int_chroma = chroma_energy(&int_px);
    assert!(
        int_chroma > 10.0,
        "{label}: interleaved reference has no chroma ({int_chroma:.2}) — bad fixture"
    );

    // The actual P0: a non-interleaved baseline file must not decode to a
    // greyscale image. Dropping the Cb/Cr scans leaves R==G==B everywhere.
    let ni_chroma = chroma_energy(&ni_px);
    assert!(
        ni_chroma > 10.0,
        "{label}: non-interleaved decode has no chroma (mean |channel-luma| = {ni_chroma:.2}, \
         interleaved reference = {int_chroma:.2}) — the Cb/Cr scans were dropped"
    );

    // Same coefficients, so same pixels. Exact.
    let (max, ndiff) = diff_stats(&ni_px, &int_px);
    assert_eq!(
        max,
        0,
        "{label}: non-interleaved decode differs from the interleaved encoding of the \
         same coefficients — max channel delta {max}, {ndiff} of {} bytes differ",
        ni_px.len()
    );
}

#[test]
fn non_interleaved_baseline_420_matches_interleaved() {
    check_pair("ni420_88x54.jpg", "int420_88x54.jpg", "4:2:0");
}

#[test]
fn non_interleaved_baseline_422_matches_interleaved() {
    check_pair("ni422_88x54.jpg", "int422_88x54.jpg", "4:2:2");
}

#[test]
fn non_interleaved_baseline_444_matches_interleaved() {
    check_pair("ni444_88x54.jpg", "int444_88x54.jpg", "4:4:4");
}

/// The same file decoded through every public entry point must agree.
///
/// `decode()` may take the streaming path while `decode_coefficients()` always
/// buffers; a dispatch bug that only mis-routes one of them shows up here.
#[test]
fn non_interleaved_420_coefficient_path_agrees_with_decode() {
    let ni = fixture("ni420_88x54.jpg");
    let (_, _, streamed) = decode_rgb(&ni);

    // Force the buffered/coefficient path by asking for coefficients and
    // rendering them through the same decoder.
    let decoder = Decoder::new().output_format(zenjpeg::types::PixelFormat::Rgb);
    let coeffs = decoder
        .decode_coefficients(&ni, Unstoppable)
        .expect("decode_coefficients");
    assert_eq!(coeffs.components.len(), 3, "three components expected");

    // Non-zero chroma coefficients must exist — this is the coefficient-level
    // statement of "the Cb/Cr scans were decoded".
    for (i, comp) in coeffs.components.iter().enumerate().skip(1) {
        let nonzero = comp.coeffs.iter().filter(|&&c| c != 0).count();
        assert!(
            nonzero > 0,
            "component {i} has no non-zero coefficients — its scan was not decoded"
        );
    }

    let ic = chroma_energy(&streamed);
    assert!(ic > 10.0, "streamed decode lost chroma ({ic:.2})");
}

/// Partially interleaved: scan 1 is `Ns=1` {Y}, scan 2 is `Ns=2` {Cb,Cr}.
///
/// The `Ns=2` scan is interleaved (A.2.3) over the *frame's* MCU grid, so it
/// uses the ordinary interleaved traversal — but the frame still has more than
/// one scan, so the whole-frame single-scan paths must not claim it.
///
/// `cjpeg -scans` script: `0;\n1,2;`
#[test]
fn partially_interleaved_baseline_420_matches_interleaved() {
    check_pair(
        "pi420_88x54.jpg",
        "int420_88x54.jpg",
        "4:2:0 partially interleaved",
    );
}

/// Arithmetic-coded (SOF9) sequential frame split into three `Ns=1` scans.
///
/// Same defect class as the Huffman case: `decode_arithmetic_scan` walked the
/// interleaved MCU grid unconditionally. The oracle is the *Huffman*
/// interleaved encoding of the same source at the same quality and sampling —
/// arithmetic coding is a different entropy coder over identical coefficients,
/// so the decoded pixels match exactly (verified with `djpeg`).
///
/// `cjpeg -quality 88 -arithmetic -sample 2x2,1x1,1x1 -scans ni.scan`
#[test]
fn non_interleaved_arithmetic_420_matches_interleaved() {
    check_pair(
        "ni420_arith_88x54.jpg",
        "int420_88x54.jpg",
        "4:2:0 arithmetic",
    );
}

/// Single-component frame with `Hi=Vi=2`.
///
/// A grayscale frame's only component *is* the maximum, so `x_i == X` and
/// `y_i == Y` (A.1.1): the scan holds `ceil(88/8) * ceil(54/8) = 77` data
/// units, exactly as at 1x1. libjpeg-turbo emits byte-identical entropy data
/// for both files — only the SOF sampling byte differs — so they must decode
/// to the same pixels. Walking the MCU-padded grid instead reads
/// `ceil(88/16) * ceil(54/16) * 4 = 96` data units in 2x2 MCU order.
#[test]
fn grayscale_2x2_sampling_matches_1x1() {
    let two = fixture("gray2x2_88x54.jpg");
    let one = fixture("gray1x1_88x54.jpg");

    let (w2, h2, p2) = decode_rgb(&two);
    let (w1, h1, p1) = decode_rgb(&one);
    assert_eq!((w2, h2), (88, 54));
    assert_eq!((w1, h1), (88, 54));

    let (max, ndiff) = diff_stats(&p2, &p1);
    assert_eq!(
        max,
        0,
        "grayscale Hi=Vi=2 decode differs from the Hi=Vi=1 encoding of the same \
         entropy data — max channel delta {max}, {ndiff} of {} bytes differ",
        p2.len()
    );
}

/// Large enough, and restart-marked, to reach the fused parallel decoder.
///
/// `try_fused_parallel_decode` also renders the whole frame from one scan by
/// walking the interleaved MCU grid, and it is gated on MCU count and
/// MCU-row-aligned DRI rather than on scan shape. The small fixtures above
/// cannot reach it: it needs `>= 1024` MCUs, `restart_interval != 0`, and
/// `--features parallel` with more than one thread.
///
/// 264x264 4:4:4 gives a 33x33 = 1089 MCU grid, and `cjpeg -restart 1` writes
/// `DRI Ri=33` (exactly one MCU row). Under `--features parallel` this test
/// exercises the fused path; under default features it exercises the
/// sequential one. Both must match the interleaved encoding.
#[test]
fn non_interleaved_with_dri_matches_interleaved_at_parallel_scale() {
    let ni = fixture("ni444_264x264_dri.jpg");
    let int = fixture("int444_264x264_dri.jpg");

    let (nw, nh, ni_px) = decode_rgb(&ni);
    let (iw, ih, int_px) = decode_rgb(&int);
    assert_eq!((nw, nh), (264, 264));
    assert_eq!((iw, ih), (264, 264));

    let ni_chroma = chroma_energy(&ni_px);
    assert!(
        ni_chroma > 10.0,
        "non-interleaved decode has no chroma ({ni_chroma:.2}) — the Cb/Cr scans were dropped"
    );

    let (max, ndiff) = diff_stats(&ni_px, &int_px);
    assert_eq!(
        max,
        0,
        "non-interleaved 264x264+DRI decode differs from the interleaved encoding of \
         the same coefficients — max channel delta {max}, {ndiff} of {} bytes differ",
        ni_px.len()
    );
}
