//! Fuzz crash regression suite — replays every committed seed on stable.
//!
//! `fuzz/regression/` holds minimized inputs that once crashed (or hung) a
//! fuzz target and have since been fixed. Replaying them needs neither
//! nightly nor `cargo-fuzz`: this is a plain `cargo test`, so the seeds gate
//! every CI run instead of only the hand-run fuzzing sessions. `cargo test -p
//! zenjpeg --lib --tests` (what `.github/workflows/ci.yml` already runs on all
//! six platforms) picks it up for free; `.github/workflows/fuzz.yml` also runs
//! it as its own job.
//!
//! Every seed goes through EVERY entry point, not just the target that found
//! it. The seeds are raw bytes, not target-specific fixtures, and a bug
//! reached from one dispatch path is usually reachable from the others — the
//! `truncation-*` seeds are the standing example: they were found by
//! `fuzz_truncation`, but the same prefixes flow through `decode_rows`,
//! `decode_coefficients` and the container scanners.
//!
//! To add a seed: drop the (preferably `cargo fuzz tmin`-minimized) file into
//! `fuzz/regression/` — a `crash-<sha>` / `timeout-<name>` / `<class>-<name>`
//! name, no extension. Nothing else. Subdirectories are walked too.
//!
//! The limits below mirror `fuzz/fuzz_targets/*.rs` exactly — they are
//! deliberately tighter than production defaults so a decompression bomb
//! reports as a rejected input rather than as a timeout. Keep them in sync.
//!
//! Contract under test is "no panic, no hang, no broken invariant" — NOT
//! "decodes successfully". Every helper deliberately ignores `Err`.

use std::fs;
use std::path::{Path, PathBuf};

use enough::Unstoppable;
use zenjpeg::container::{
    GainMapPresence, MAX_XMP_LENGTH, MpfEntry, Wants, create_mpf_header, find_jpeg_boundaries,
    is_ultrahdr, iter, parse_mpf, parse_mpf_segment, parse_xmp, parse_xmp_full, primary_bounds,
    probe,
};
use zenjpeg::decoder::{
    ChromaUpsampling, Decoder, ErrorKind, OutputTarget, PixelFormat, Strictness,
};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

/// `fuzz_decode`, `fuzz_decode_paths`, `fuzz_push_decode`, `fuzz_differential`.
const MAX_PX: u64 = 4_000_000;
/// `fuzz_decode_limits` is the only target that also caps memory.
const MAX_MEM: u64 = 64 * 1024 * 1024;
/// `fuzz_truncation` runs a tighter pixel cap than the rest.
const TRUNCATION_MAX_PX: u64 = 1_000_000;

fn regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

/// Every file under `fuzz/regression/`, paired with the directory that names
/// the target it was found by (kept only for failure messages).
fn seeds() -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, target: &str, out: &mut Vec<(String, PathBuf)>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(target)
                    .to_string();
                walk(&path, &name, out);
            } else if path.is_file() {
                out.push((target.to_string(), path));
            }
        }
    }

    let mut out = Vec::new();
    walk(&regression_dir(), "regression", &mut out);
    out.sort();
    out
}

// ── decode entry points ──────────────────────────────────────────────────

/// `fuzz_decode_limits` — the only target that sets `max_memory`.
fn run_decode_limits(data: &[u8]) {
    let _ = Decoder::new()
        .max_pixels(MAX_PX)
        .max_memory(MAX_MEM)
        .decode(data, Unstoppable);
}

/// `fuzz_decode` steps 1-8 plus `fuzz_decode_paths`' config matrix.
fn run_decode_matrix(data: &[u8]) {
    let _ = Decoder::new().max_pixels(MAX_PX).decode(data, Unstoppable);

    for format in [
        PixelFormat::Gray,
        PixelFormat::Rgb,
        PixelFormat::Rgba,
        PixelFormat::Bgr,
        PixelFormat::Bgra,
    ] {
        let _ = Decoder::new()
            .output_format(format)
            .max_pixels(MAX_PX)
            .decode(data, Unstoppable);
    }

    for target in [
        OutputTarget::Srgb8,
        OutputTarget::SrgbF32,
        OutputTarget::LinearF32,
        OutputTarget::SrgbF32Precise,
        OutputTarget::LinearF32Precise,
    ] {
        let _ = Decoder::new()
            .output_target(target)
            .max_pixels(MAX_PX)
            .decode(data, Unstoppable);
    }

    for upsampling in [
        ChromaUpsampling::Triangle,
        ChromaUpsampling::NearestNeighbor,
    ] {
        let _ = Decoder::new()
            .chroma_upsampling(upsampling)
            .max_pixels(MAX_PX)
            .decode(data, Unstoppable);
    }

    for strictness in [
        Strictness::Strict,
        Strictness::Balanced,
        Strictness::Lenient,
        Strictness::Permissive,
    ] {
        let _ = Decoder::new()
            .strictness(strictness)
            .max_pixels(MAX_PX)
            .decode(data, Unstoppable);
    }

    // Coefficient extraction (DCT domain, no IDCT) and YCbCr f32 planes are
    // separate code paths from the pixel decoders above.
    let _ = Decoder::new()
        .max_pixels(MAX_PX)
        .decode_coefficients(data, Unstoppable);
    let _ = Decoder::new()
        .max_pixels(MAX_PX)
        .decode_to_ycbcr_f32(data, Unstoppable);
    let _ = Decoder::new()
        .auto_orient(true)
        .max_pixels(MAX_PX)
        .decode(data, Unstoppable);
}

/// `fuzz_push_decode` (and `fuzz_decode` step 9) — the streaming row paths.
fn run_push_decode(data: &[u8]) {
    for format in [
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Gray,
    ] {
        let _ =
            Decoder::new()
                .max_pixels(MAX_PX)
                .decode_rows(data, format, |_row| Ok(()), Unstoppable);
    }
    for format in [PixelFormat::RgbaF32, PixelFormat::GrayF32] {
        let _ = Decoder::new().max_pixels(MAX_PX).decode_rows_f32(
            data,
            format,
            |_row| Ok(()),
            Unstoppable,
        );
    }
}

/// `fuzz_read_info` — header-only parse, with the target's own invariants.
fn run_read_info(data: &[u8]) {
    if let Ok(info) = Decoder::new().read_info(data) {
        assert!(info.dimensions.width > 0, "width should be positive");
        assert!(info.dimensions.height > 0, "height should be positive");
        assert!(info.num_components > 0, "num_components should be positive");
        assert!(info.num_components <= 4, "num_components should be <= 4");
        assert!(
            info.precision == 8 || info.precision == 12,
            "precision should be 8 or 12, got {}",
            info.precision
        );
    }
}

/// `fuzz_truncation` — every prefix must decode-or-fail self-consistently.
///
/// This is the contract four of the committed seeds (`truncation-*`) were
/// minimized against, so it is replayed in full rather than summarized.
fn run_truncation(data: &[u8]) {
    if data.len() < 4 {
        return;
    }
    let n = data.len();
    let steer = (data[n - 1] as usize * n) / 256;
    let cuts = [n, n - 1, n - 2, n / 2, n / 4, (n * 3) / 4, steer.max(2)];

    let ok_at = |len: usize| -> bool {
        Decoder::new()
            .max_pixels(TRUNCATION_MAX_PX)
            .decode(&data[..len], Unstoppable)
            .is_ok()
    };
    // Every prefix of a stream that decodes in full is a pure truncation.
    let clean_stream = ok_at(n);

    let mut sorted = cuts;
    sorted.sort_unstable();

    let mut first_ok: Option<usize> = None;
    for &len in &sorted {
        let full = Decoder::new()
            .max_pixels(TRUNCATION_MAX_PX)
            .decode(&data[..len], Unstoppable);
        let strict = Decoder::new()
            .max_pixels(TRUNCATION_MAX_PX)
            .strictness(Strictness::Strict)
            .decode(&data[..len], Unstoppable);

        if let Ok(img) = &full {
            // Dimensions come from the header, never from how much scan data
            // survived.
            if let Ok(info) = Decoder::new()
                .max_pixels(TRUNCATION_MAX_PX)
                .read_info(&data[..len])
            {
                assert_eq!(img.width(), info.dimensions.width);
                assert_eq!(img.height(), info.dimensions.height);
            }
            if first_ok.is_none() {
                first_ok = Some(len);
            }
            // Strict accepting the prefix means it is a complete, clean
            // stream: the tolerant decode must not differ.
            if let Ok(s) = &strict {
                assert_eq!(
                    s.pixels_u8(),
                    img.pixels_u8(),
                    "Strict/Balanced pixel mismatch at {len}"
                );
            }
        } else if let Some(f) = first_ok {
            // Monotone: a longer prefix of a stream that already decoded must
            // never come back as "not enough data", and on a clean stream must
            // still decode outright.
            let err = full.as_ref().err();
            let is_truncation = err
                .map(|e| matches!(e.kind(), ErrorKind::TruncatedData { .. }))
                .unwrap_or(false);
            assert!(
                !(is_truncation || clean_stream) || !ok_at(f),
                "prefix {f} decoded but longer prefix {len} errored \
                 (clean stream: {clean_stream}): {err:?}"
            );
        }
    }

    // Row-callback and coefficient paths must survive the same cuts.
    for &len in &sorted {
        let _ = Decoder::new().max_pixels(TRUNCATION_MAX_PX).decode_rows(
            &data[..len],
            PixelFormat::Rgb,
            |_| Ok(()),
            Unstoppable,
        );
        let _ = Decoder::new()
            .max_pixels(TRUNCATION_MAX_PX)
            .decode_coefficients(&data[..len], Unstoppable);
    }
}

/// `fuzz_differential` — zenjpeg vs zune-jpeg on the same bytes.
///
/// Dimensions are compared whenever both decoders produce an image: those come
/// from the SOF header, so a disagreement is a real bug either way.
///
/// The PIXEL comparison is gated on `Strictness::Strict` accepting the input,
/// and that precondition is load-bearing rather than a convenience. Its bound
/// (`max_diff <= 4`) only describes IDCT rounding, which is the *only*
/// legitimate difference between two decoders reading the same well-formed
/// stream. On a malformed stream the two are not decoding, they are
/// *recovering*, and recovery policy is deliberately per-decoder — zenjpeg's
/// is a documented four-level ladder (pad-zeros on truncation, EOB on an AC
/// run past the block, RST resync, …; see `docs/strictness.md`) that no other
/// decoder promises to match. Comparing recovered pixels compares two
/// different policies, not two implementations of one contract. Borrowing
/// `fuzz_truncation`'s own test for "complete, clean stream" — Strict accepted
/// it — keeps the assertion at full strength exactly where it is well-defined.
///
/// This matters concretely: `truncation-ac-overflow-slow-path` is rejected by
/// Strict ("extraneous bytes between markers"), recovered by both Balanced and
/// zune, and the two recoveries differ by 27. Ungated, that fails here. NOTE
/// that `fuzz/fuzz_targets/fuzz_differential.rs` carries the ungated form and
/// so has the same latent false positive — it just has not drawn a
/// malformed-but-both-recoverable input yet. Left as-is deliberately: changing
/// a fuzz target's assertion is a separate call from authoring this harness.
fn run_differential(data: &[u8]) {
    use zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;

    let ours = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .max_pixels(MAX_PX)
        .decode(data, Unstoppable);

    let mut zune = JpegDecoder::new(ZCursor::new(data));
    let Ok(zune_pixels) = zune.decode() else {
        return;
    };
    let Some(zune_info) = zune.info() else {
        return;
    };
    let Ok(img) = &ours else {
        // Differing strictness between decoders is expected and not a bug.
        return;
    };
    let Some(our_pixels) = img.pixels_u8() else {
        return;
    };

    assert_eq!(
        img.width(),
        u32::from(zune_info.width),
        "Width mismatch vs zune-jpeg"
    );
    assert_eq!(
        img.height(),
        u32::from(zune_info.height),
        "Height mismatch vs zune-jpeg"
    );

    let clean_stream = Decoder::new()
        .max_pixels(MAX_PX)
        .strictness(Strictness::Strict)
        .decode(data, Unstoppable)
        .is_ok();
    if !clean_stream {
        return;
    }

    // zune-jpeg returns grayscale as 1 byte/pixel even when RGB is requested,
    // so expand before comparing.
    let pixel_count = img.width() as usize * img.height() as usize;
    let zune_rgb: Vec<u8> = if zune_pixels.len() == pixel_count {
        zune_pixels.iter().flat_map(|&y| [y, y, y]).collect()
    } else if zune_pixels.len() == pixel_count * 3 {
        zune_pixels
    } else {
        // CMYK / YCbCr / some other layout — a format mismatch, not a bug.
        return;
    };

    if our_pixels.len() != zune_rgb.len() {
        return;
    }
    let max_diff = our_pixels
        .iter()
        .zip(zune_rgb.iter())
        .map(|(&x, &y)| (i16::from(x) - i16::from(y)).unsigned_abs())
        .max()
        .unwrap_or(0);
    // IDCT implementations differ by a rounding step; 4 is the target's bound.
    assert!(
        max_diff <= 4,
        "Pixel values differ too much on a Strict-clean stream: {max_diff}"
    );
}

// ── container entry points ───────────────────────────────────────────────

/// `fuzz_container_marker`.
fn run_container_marker(data: &[u8]) {
    let mut last_end = 0usize;
    let mut iterations = 0usize;
    for span in iter(data) {
        iterations += 1;
        assert!(
            iterations <= data.len() + 2,
            "MarkerIter yielded more than data.len()+2 spans — non-terminating?"
        );
        assert!(
            span.offset + span.length <= data.len(),
            "span beyond buffer: offset={}, length={}, data.len()={}",
            span.offset,
            span.length,
            data.len()
        );
        assert!(
            span.offset >= last_end,
            "span overlaps previous: offset={}, last_end={last_end}",
            span.offset
        );
        last_end = span.offset + span.length;

        let payload_start = (span.payload.as_ptr() as usize).wrapping_sub(data.as_ptr() as usize);
        if !span.payload.is_empty() {
            assert!(
                payload_start >= span.offset && payload_start < span.offset + span.length,
                "payload pointer not inside span"
            );
            assert!(
                payload_start + span.payload.len() <= span.offset + span.length,
                "payload extends past span"
            );
        }
    }

    let primary = primary_bounds(data);
    if let Some(r) = &primary {
        assert!(r.start < r.end, "primary_bounds returned empty range");
        assert!(r.end <= data.len(), "primary_bounds exceeds buffer");
        assert_eq!(r.start, 0, "primary_bounds must start at 0");
    }

    let all = find_jpeg_boundaries(data);
    for r in &all {
        assert!(r.start < r.end, "find_jpeg_boundaries returned empty range");
        assert!(r.end <= data.len(), "find_jpeg_boundaries exceeds buffer");
    }
    for pair in all.windows(2) {
        assert!(
            pair[0].end <= pair[1].start,
            "find_jpeg_boundaries ranges overlap: {:?} vs {:?}",
            pair[0],
            pair[1]
        );
    }
    if let (Some(p), Some(first)) = (primary, all.first()) {
        assert_eq!(
            p, *first,
            "primary_bounds disagrees with find_jpeg_boundaries first"
        );
    }
}

/// `fuzz_container_mpf`.
fn run_container_mpf(data: &[u8]) {
    fn validate(entries: &[MpfEntry]) {
        assert!(
            entries.len() <= 1000,
            "parse returned more than the hard cap: {}",
            entries.len()
        );
        for e in entries {
            let _ = e.offset.checked_add(e.size).expect("offset+size overflow");
        }
    }

    if let Ok(entries) = parse_mpf(data) {
        validate(&entries);
    }
    let tiff_pos = data.len().saturating_mul(3);
    if let Ok(entries) = parse_mpf_segment(data, tiff_pos) {
        validate(&entries);
    }

    if data.len() >= 8 {
        // Sizes of 0 are not round-trippable by contract (the issue #148
        // foreign-file resync heuristic treats a size-0 first entry as "not a
        // real entry"), so clamp exactly as the fuzz target does.
        let primary =
            (u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize).clamp(1, 1_000_000);
        let gainmap =
            (u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize).clamp(1, 1_000_000);
        let built = create_mpf_header(primary, gainmap, None);
        // Strip APP2 marker + length + "MPF\0" (4 + 4 bytes).
        let tiff_start = 8;
        if built.len() > tiff_start {
            let parsed =
                parse_mpf_segment(&built[tiff_start..], tiff_start).expect("roundtrip parse");
            assert_eq!(parsed.len(), 2);
            assert_eq!(parsed[0].size, primary);
            assert_eq!(parsed[1].size, gainmap);
        }
    }
}

/// `fuzz_container_probe`.
fn run_container_probe(data: &[u8]) {
    let len = data.len() as u32;
    let p = probe(data, Wants::ALL);

    let check = |r: Option<&core::ops::Range<u32>>| {
        if let Some(r) = r {
            assert!(r.start <= r.end, "inverted range {r:?}");
            assert!(r.end <= len, "range {r:?} exceeds input len {len}");
        }
    };
    check(p.icc_profile());
    check(p.exif());
    check(p.xmp());
    check(p.mpf());
    check(p.iso_gainmap());
    for r in p.image_ranges() {
        check(Some(r));
    }
    for r in p.extended_xmp() {
        check(Some(r));
    }

    let mut prev_end: Option<u32> = None;
    for r in p.image_ranges() {
        if let Some(pe) = prev_end {
            assert!(r.start >= pe, "image ranges out of order");
        }
        prev_end = Some(r.end);
    }

    let iso = p.iso_gainmap().is_some();
    let hdrgm = p.has_xmp_hdrgm();
    let gcontainer = p.has_xmp_gcontainer_gainmap();
    match p.gainmap_presence() {
        GainMapPresence::None => {
            assert!(
                !iso && !hdrgm && !gcontainer,
                "None presence but signal(s) present"
            );
        }
        GainMapPresence::Iso21496 => assert!(iso && !hdrgm && !gcontainer),
        GainMapPresence::XmpHdrgmLegacy => assert!(!iso && hdrgm),
        GainMapPresence::GContainerOnly => assert!(!iso && !hdrgm && gcontainer),
        GainMapPresence::IsoAndXmp => assert!(iso && hdrgm),
        GainMapPresence::IsoAndGContainer => assert!(iso && !hdrgm && gcontainer),
        // `#[non_exhaustive]` forces a wildcard arm. A variant we do not know
        // about means the invariants here are undefined — flag it.
        other => panic!(
            "harness missing GainMapPresence arm for {other:?} — update this \
             match (and fuzz_container_probe.rs) when adding a variant"
        ),
    }

    if !(iso || hdrgm || gcontainer || p.mpf().is_some()) {
        assert!(
            !is_ultrahdr(data),
            "is_ultrahdr claims true but full probe found no signal"
        );
    }
}

/// `fuzz_container_xmp` — text-only, so non-UTF-8 seeds skip it (as the
/// target does).
fn run_container_xmp(data: &[u8]) {
    let data = if data.len() > MAX_XMP_LENGTH + 1024 {
        &data[..MAX_XMP_LENGTH + 1024]
    } else {
        data
    };
    let Ok(as_str) = core::str::from_utf8(data) else {
        return;
    };
    if let Ok((params, _len)) = parse_xmp(as_str) {
        // Non-finite values are legal here — `f64::from_str("5e555")` is
        // `Ok(inf)`. Finiteness is downstream validation's job, not the
        // parser's, so only "does not panic" is asserted.
        core::hint::black_box(&params);
    }
    let (_meta, items) = parse_xmp_full(as_str);
    core::hint::black_box(&items);
}

// ── encode entry points ──────────────────────────────────────────────────

/// `fuzz_encode` and `fuzz_roundtrip`.
///
/// Those two targets take `arbitrary`-derived structs rather than raw bytes,
/// so a seed file is not a JPEG to them — it is an `Unstructured` byte stream.
/// Rather than take an `arbitrary` dev-dependency just to reproduce the exact
/// byte-to-struct mapping, the seed drives the same knobs deterministically
/// and supplies its own bytes as the pixel payload. Same encoder entry points,
/// same config matrix, seed-derived inputs.
fn run_encode_and_roundtrip(data: &[u8]) {
    // Small, cheap, and deterministic: dimensions from the first bytes,
    // clamped exactly like the targets clamp theirs.
    let w = u32::from(*data.first().unwrap_or(&1)).clamp(1, 64);
    let h = u32::from(*data.get(1).unwrap_or(&1)).clamp(1, 64);
    let quality = f32::from(*data.get(2).unwrap_or(&50)).clamp(1.0, 100.0);
    let selector = usize::from(*data.get(3).unwrap_or(&0));
    let progressive = data.get(4).is_some_and(|b| b % 2 == 0);
    let optimize_huffman = data.get(5).is_some_and(|b| b % 2 == 0);

    let subsampling = match selector % 4 {
        0 => ChromaSubsampling::None,
        1 => ChromaSubsampling::HalfHorizontal,
        2 => ChromaSubsampling::Quarter,
        _ => ChromaSubsampling::HalfVertical,
    };
    let (layout, bpp) = match selector % 5 {
        0 => (PixelLayout::Gray8Srgb, 1),
        1 => (PixelLayout::Rgb8Srgb, 3),
        2 => (PixelLayout::Rgbx8Srgb, 4),
        3 => (PixelLayout::Bgr8Srgb, 3),
        _ => (PixelLayout::Bgrx8Srgb, 4),
    };

    let mut pixels = data.to_vec();
    pixels.resize(w as usize * h as usize * bpp, 128);

    // `fuzz_encode`: grayscale wins, then XYB (RGB only), then YCbCr.
    for use_xyb in [false, true] {
        let config = if matches!(layout, PixelLayout::Gray8Srgb) {
            EncoderConfig::grayscale(quality)
        } else if use_xyb && matches!(layout, PixelLayout::Rgb8Srgb | PixelLayout::Rgbx8Srgb) {
            EncoderConfig::xyb(quality, XybSubsampling::BQuarter)
        } else {
            EncoderConfig::ycbcr(quality, subsampling)
        }
        .progressive(progressive)
        .optimize_huffman(optimize_huffman);

        if let Ok(mut enc) = config.encode_from_bytes(w, h, layout) {
            let _ = enc.push_packed(&pixels, Unstoppable);
            let _ = enc.finish();
        }
    }

    // `fuzz_roundtrip`: RGB in, decode back out, dimensions must survive.
    let mut rgb = data.to_vec();
    rgb.resize(w as usize * h as usize * 3, 128);
    let config = EncoderConfig::ycbcr(quality, subsampling).progressive(progressive);
    let Ok(mut enc) = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb) else {
        return;
    };
    if enc.push_packed(&rgb, Unstoppable).is_err() {
        return;
    }
    let Ok(encoded) = enc.finish() else {
        return;
    };
    let Ok(decoded) = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .max_pixels(MAX_PX)
        .decode(&encoded, Unstoppable)
    else {
        return;
    };
    let Some(decoded_pixels) = decoded.pixels_u8() else {
        return;
    };
    assert_eq!(decoded.width(), w, "roundtrip width mismatch");
    assert_eq!(decoded.height(), h, "roundtrip height mismatch");
    assert_eq!(
        decoded_pixels.len(),
        w as usize * h as usize * 3,
        "roundtrip pixel buffer size mismatch"
    );
}

// ── the gate ─────────────────────────────────────────────────────────────

/// The suite must never silently pass because the seed directory moved,
/// emptied out, or filled with zero-byte placeholders. There is no
/// `|| true`-shaped escape hatch anywhere in this file.
#[test]
fn regression_corpus_is_present() {
    let dir = regression_dir();
    assert!(
        dir.is_dir(),
        "{} is missing — the regression seeds are committed, not generated",
        dir.display()
    );
    let seeds = seeds();
    assert!(
        !seeds.is_empty(),
        "{} contains no seed files; a suite that tests nothing is worse than \
         one that fails",
        dir.display()
    );
    let total_bytes: u64 = seeds
        .iter()
        .map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    assert!(
        total_bytes > 0,
        "{} holds {} seed file(s) but 0 bytes total — the corpus was \
         truncated, not populated",
        dir.display(),
        seeds.len()
    );
}

#[test]
fn regression_seeds_do_not_panic() {
    let seeds = seeds();
    let mut replayed = 0usize;
    for (target, path) in &seeds {
        let data = fs::read(path)
            .unwrap_or_else(|e| panic!("failed to read seed {}: {e}", path.display()));
        // A panic aborts the test with the seed already named in the output.
        eprintln!("replaying {target}: {}", path.display());

        run_decode_limits(&data);
        run_decode_matrix(&data);
        run_push_decode(&data);
        run_read_info(&data);
        run_truncation(&data);
        run_differential(&data);
        run_container_marker(&data);
        run_container_mpf(&data);
        run_container_probe(&data);
        run_container_xmp(&data);
        run_encode_and_roundtrip(&data);

        replayed += 1;
    }
    // Belt and braces against a future refactor that walks the loop zero
    // times while both tests still report success.
    assert_eq!(
        replayed,
        seeds.len(),
        "replay loop skipped seeds: {replayed} of {}",
        seeds.len()
    );
    assert!(replayed > 0, "replayed no seeds at all");
}
