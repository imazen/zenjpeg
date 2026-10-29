//! perf_manifest — bit-exact A/B output manifest for performance work.
//!
//! Emits one TSV row per (case, path, SIMD tier): `key<TAB>sha256<TAB>bytes`.
//! `emit` iterates every archmage token permutation (dev builds carry
//! `testable_dispatch`), so each row is tagged with the effective dispatch
//! tier: `all enabled` is default dispatch; the fully-disabled permutation is
//! the forced-scalar arm. Compare is `candidate == baseline` **within** each
//! tier — encoder bytes legitimately differ *between* tiers (a few ULPs of FP
//! difference can flip a DCT coefficient; see CLAUDE.md "Cross-backend dispatch
//! parity tolerance"), so the manifest never asserts cross-tier equality.
//!
//! Usage:
//!   perf_manifest emit [--out FILE] [--allow-missing-corpus]
//!   perf_manifest compare BASE.tsv CAND.tsv
//!
//! Exit status: compare exits 1 on any differing/missing row or scope
//! mismatch; exits 2 on unreadable or malformed manifests. emit exits nonzero
//! without writing output when the codec corpus is unavailable unless
//! `--allow-missing-corpus` is passed — a reduced scope is always an explicit
//! caller choice and is recorded in the manifest's `#` provenance lines.
//!
//! Manifest files begin with `#key<TAB>value` provenance lines (format
//! version, feature set, host arch, filter scope, corpus state, row count).
//! `compare` requires identical provenance — a reduced-scope manifest cannot
//! silently compare equal to a full one.
//!
//! Rows that cannot run (API rejection, panic) emit a stable
//! `SKIP:`/`ERR:`/`PANIC:` tag instead of a hash, so the key still exists on
//! both sides of the comparison and still diffs if only one side errors.
//! `MANIFEST_FILTER=<substr>` restricts to matching encode keys (smoke runs);
//! the filter is recorded in `#scope`.
//!
//! Actual coverage (what a passing compare proves identical):
//!   encode: sizes 1x1..8200x8 over noise/flatchroma/screen + two corpus
//!     photo slots (clic2025/final-test, imageflow/test_inputs); quality
//!     5..98 plus mozjpeg80/ssim2-80 approximations; subsampling
//!     444/422/420/440; baseline and progressive (progressive both with and
//!     without restart markers); restart 0/1/4/16; optimize-huffman on/off;
//!     all 17 PixelLayout variants; strided input; XYB bquarter/full;
//!     gamma-aware(+iterative) downsampling; st plus mt (--features
//!     parallel) rows.
//!   decode of each produced JPEG: full-image u8 (rgb/bgr/rgba/bgra/gray/
//!     rgb16; fancy+box upsampling and libjpeg+jpegli IDCT are crossed on
//!     rgb only — the upsampler x IDCT product is not fully crossed for
//!     other formats); full-image f32 (srgb, srgb-precise, linear) with an
//!     explicit threads=1/threads=0 split; decode_into; ScanlineReader in
//!     batches of 1/4/7 rows with a stride pad; coefficient-domain decode;
//!     planar YCbCr f32.
//!   corpus decode: every *.jpg under jpeg-conformance/{valid,
//!     non-conformant,crash-repro}, zune/{test-images,fuzz-corpus}/jpeg,
//!     imageflow/test_inputs and mozjpeg, through a reduced path set.
//! Known stable ERR rows: the YCbCrF32 encode case is rejected with
//!   `ERR:InvalidBufferSize` — the layout declares 12 bytes/pixel while an
//!   inner validator expects 3. That is a pre-existing library
//!   inconsistency recorded as a stable tag, not exercised pixels.
//!
//! `cargo test -p zenjpeg --example perf_manifest` (optionally
//! `--features parallel`) runs the permanent strictness/coverage suite:
//! malformed/empty/duplicate/invalid-length/truncated manifests are
//! rejected, compare detects hash flips + missing keys + provenance drift,
//! all layouts appear in enc cases, and the f32 st/mt rows select distinct
//! decoder thread counts.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use zenjpeg::decode::{ChromaUpsampling, Decoder, IdctMethod, OutputTarget};
use zenjpeg::decoder::PixelFormat;
#[cfg(feature = "parallel")]
use zenjpeg::encoder::ParallelEncoding;
use zenjpeg::encoder::Unstoppable;
use zenjpeg::encoder::{
    ChromaSubsampling, DownsamplingMethod, EncoderConfig, PixelLayout, Quality, XybSubsampling,
};

// ============================================================================
// Content generators (deterministic; no I/O)
// ============================================================================

/// Noise+patches: 4 block types per 8x8 (textured, ramp, sharp edges, noise).
/// Same generator shape as `valgrind_decode`'s fixture — realistic DCT
/// coefficient distributions without the degenerate all-zero blocks that
/// smooth gradients produce.
fn gen_noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let (bx, by) = ((x / 8) as u32, (y / 8) as u32);
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let mut hh = (x as u32)
                .wrapping_mul(374761393)
                .wrapping_add((y as u32).wrapping_mul(668265263));
            hh = (hh ^ (hh >> 13)).wrapping_mul(1274126177);
            let noise = (hh >> 24) as u8;
            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / w.max(1)) as u8;
                    data[idx + 1] = ((y * 255) / h.max(1)) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255 - edge;
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }
    data
}

/// Flat-chroma (DC-only) blocks: chroma constant inside each 8x8 chroma cell,
/// changing every 16 rows so it differs across MCU boundaries. This is the
/// only content class that reaches the decoder's DC-only block paths; mirrors
/// `smooth_chroma_bands`/`smooth_chroma_image` in the test suite.
fn gen_flat_chroma(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = match (y / 16) % 4 {
                0 => (200i32, 90, 70),
                1 => (70, 180, 90),
                2 => (80, 90, 210),
                _ => (190, 190, 70),
            };
            // Equal luma delta on all channels moves Y but keeps Cb/Cr flat.
            let luma = ((x * 7 + y * 3) % 90) as i32 / 3;
            rgb[i] = (r + luma).clamp(0, 255) as u8;
            rgb[i + 1] = (g + luma).clamp(0, 255) as u8;
            rgb[i + 2] = (b + luma).clamp(0, 255) as u8;
        }
    }
    rgb
}

/// Screenshot-like content: flat fills, 1px black/white lines, checkerboard
/// text-ish blocks. Exercises hard edges + flat regions together.
fn gen_screen(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let cell = ((x / 64) + (y / 48)) % 3;
            match cell {
                0 => {
                    // flat window fill with a hard 1px border
                    let border = x % 64 == 0 || y % 48 == 0 || x % 64 == 63 || y % 48 == 47;
                    let v = if border { 30u8 } else { 235u8 };
                    rgb[i] = v;
                    rgb[i + 1] = v;
                    rgb[i + 2] = v.wrapping_sub(5);
                }
                1 => {
                    // checkerboard "text" on white
                    let on = (x % 4 < 2) ^ (y % 4 < 2);
                    let v = if on { 0u8 } else { 255u8 };
                    rgb[i] = v;
                    rgb[i + 1] = v;
                    rgb[i + 2] = v;
                }
                _ => {
                    // saturated fills with sharp diagonal edge
                    let d = (x + y) % 96 < 48;
                    rgb[i] = if d { 220 } else { 40 };
                    rgb[i + 1] = if d { 60 } else { 200 };
                    rgb[i + 2] = 130;
                }
            }
        }
    }
    rgb
}

// ============================================================================
// Corpus inputs. An unavailable corpus is a hard error unless the caller
// passes --allow-missing-corpus; a partially-populated corpus is recorded in
// the manifest's #corpus provenance so scope reduction can never be silent.
// ============================================================================

struct Corpus {
    inner: Option<codec_corpus::Corpus>,
}

/// Where the corpus resolve failed, for provenance. `Disabled` is the
/// explicit --allow-missing-corpus state.
#[derive(Clone, Copy, PartialEq, Debug)]
enum CorpusState {
    Ok,
    /// Corpus opened but this many expected entries were missing.
    Partial(usize),
    Disabled,
}

/// Corpus directories the manifest depends on: JPEGs decoded directly.
const CORPUS_JPEG_DIRS: &[&str] = &[
    "jpeg-conformance/valid",
    "jpeg-conformance/non-conformant",
    "jpeg-conformance/crash-repro",
    "zune/test-images/jpeg",
    "zune/fuzz-corpus/jpeg",
    "imageflow/test_inputs",
    "mozjpeg",
];
/// Corpus directories supplying the photo encode slots (in slot order).
const CORPUS_PHOTO_DIRS: &[&str] = &["clic2025/final-test", "imageflow/test_inputs"];

impl Corpus {
    /// Present-or-absent view used when the caller opted into reduced scope.
    fn try_open() -> Self {
        Self {
            inner: codec_corpus::Corpus::new().ok(),
        }
    }
    fn get(&self, rel: &str) -> Option<std::path::PathBuf> {
        self.inner.as_ref().and_then(|c| c.get(rel).ok())
    }
    fn available(&self) -> bool {
        self.inner.is_some()
    }
    /// Expected corpus inputs that resolve to nothing usable: unresolvable
    /// or empty JPEG dirs, unreadable subdirectories/entries/files, plus
    /// photo slots with no qualifying PNG. The count comes from the same
    /// traversal that loads the rows — a preflight that stops at the first
    /// success cannot certify the collection.
    fn missing_expected(&self) -> usize {
        let (photos, photo_failures) = self.scan_photos();
        self.scan_jpegs().1 + photo_failures + photos.iter().filter(|s| s.is_none()).count()
    }
    /// Same count with pre-resolved photo slots.
    fn missing_with(&self, photo_slots: &[Option<(Vec<u8>, usize, usize)>]) -> usize {
        if self.inner.is_none() {
            return CORPUS_JPEG_DIRS.len() + photo_slots.len();
        }
        self.scan_jpegs().1
            + self.scan_photos().1
            + photo_slots.iter().filter(|s| s.is_none()).count()
    }
    /// All corpus JPEGs plus the count of coverage failures the traversal
    /// hit (unresolved dir, unreadable subdirectory/entry/file, or a dir
    /// yielding zero usable inputs). Failures are returned, never silently
    /// swallowed — the coverage gate consumes this same result.
    fn scan_jpegs(&self) -> (Vec<(String, Vec<u8>)>, usize) {
        let mut out = Vec::new();
        let mut failures = 0usize;
        for d in CORPUS_JPEG_DIRS {
            let Some(dir) = self.get(d) else {
                failures += 1;
                continue;
            };
            let (mut v, f) = scan_jpeg_dir(d, &dir);
            out.append(&mut v);
            failures += f;
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out.dedup_by(|a, b| a.0 == b.0);
        (out, failures)
    }
    /// Photo slots + traversal failures (unresolved dir, unreadable
    /// subdirectory/entry, or a *.png that cannot be opened). Every PNG is
    /// checked for readability; decode stops at the first qualifying photo
    /// per slot, so slot stability and `SKIP:no-src` semantics are
    /// unchanged.
    fn scan_photos(&self) -> (Vec<Option<(Vec<u8>, usize, usize)>>, usize) {
        let mut out = Vec::new();
        let mut failures = 0usize;
        for dir in CORPUS_PHOTO_DIRS {
            let mut found = None;
            let Some(d) = self.get(dir) else {
                failures += 1;
                out.push(found);
                continue;
            };
            let mut stack = vec![(d, 0usize)];
            let mut pngs = Vec::new();
            while let Some((dir, depth)) = stack.pop() {
                let rd = match std::fs::read_dir(&dir) {
                    Ok(rd) => rd,
                    Err(_) => {
                        failures += 1;
                        continue;
                    }
                };
                for ent in rd {
                    let ent = match ent {
                        Ok(e) => e,
                        Err(_) => {
                            failures += 1;
                            continue;
                        }
                    };
                    let p = ent.path();
                    match p.metadata() {
                        Ok(m) if m.is_dir() => {
                            if depth < MAX_SCAN_DEPTH {
                                stack.push((p, depth + 1));
                            } else {
                                failures += 1;
                            }
                        }
                        Ok(_) => {
                            if p.extension().is_some_and(|e| e == "png") {
                                pngs.push(p);
                            }
                        }
                        // Same rule as the JPEG scan: an entry whose
                        // metadata cannot be read may hide coverage.
                        Err(_) => failures += 1,
                    }
                }
            }
            pngs.sort();
            let mut readable = Vec::with_capacity(pngs.len());
            for p in pngs {
                if std::fs::File::open(&p).is_ok() {
                    readable.push(p);
                } else {
                    failures += 1;
                }
            }
            for p in readable {
                if let Ok((rgb, w, h)) = load_png_rgb8(&p)
                    && w >= 2048
                    && h >= 1536
                {
                    eprintln!("corpus photo slot {}: {}", out.len(), p.display());
                    found = Some((rgb, w as usize, h as usize));
                    break;
                }
            }
            out.push(found);
        }
        (out, failures)
    }
}

/// Traverse one expected JPEG dir: every readable `*.jpg`/`*.jpeg` file as
/// (`top`-relative name, bytes) plus the count of traversal failures —
/// unreadable subdirectory, unreadable directory entry, unreadable file,
/// or a dir that yields zero usable inputs.
/// Deepest directory nesting a corpus scan will follow. Real corpus trees
/// are a few levels deep; this bound exists to break symlink cycles — an
/// entry past the cap counts as a failure, never a silent skip.
const MAX_SCAN_DEPTH: usize = 32;

fn scan_jpeg_dir(top: &str, root: &std::path::Path) -> (Vec<(String, Vec<u8>)>, usize) {
    let mut out = Vec::new();
    let mut failures = 0usize;
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => {
                failures += 1;
                continue;
            }
        };
        for ent in rd {
            let ent = match ent {
                Ok(e) => e,
                Err(_) => {
                    failures += 1;
                    continue;
                }
            };
            let p = ent.path();
            match p.metadata() {
                Ok(m) if m.is_dir() => {
                    if depth < MAX_SCAN_DEPTH {
                        stack.push((p, depth + 1));
                    } else {
                        failures += 1;
                    }
                }
                Ok(_) => {
                    if p.extension().and_then(|e| e.to_str()).is_some_and(|s| {
                        s.eq_ignore_ascii_case("jpg") || s.eq_ignore_ascii_case("jpeg")
                    }) {
                        match std::fs::read(&p) {
                            Ok(bytes) => {
                                let rel = p.strip_prefix(root).unwrap_or(&p);
                                out.push((format!("{top}/{}", rel.display()), bytes));
                            }
                            Err(_) => failures += 1,
                        }
                    }
                }
                // Entry metadata unreadable — it may hide a subtree or a
                // JPEG, so it counts as a coverage failure, not a skip.
                Err(_) => failures += 1,
            }
        }
    }
    if out.is_empty() && failures == 0 {
        // Resolved dir yields zero usable inputs — missing coverage.
        failures = 1;
    }
    (out, failures)
}

/// Decode a corpus PNG to RGB8 (8-bit RGB/RGBA/gray only; others -> Err).
fn load_png_rgb8(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut dec = png::Decoder::new(std::io::BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let out_size = reader.output_buffer_size().ok_or("png size")?;
    let mut buf = vec![0u8; out_size];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width, info.height);
    let n = (w as usize) * (h as usize);
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..n * 3].to_vec(),
        png::ColorType::Rgba => buf[..n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..n].iter().flat_map(|&v| [v, v, v]).collect(),
        other => return Err(format!("unsupported png color {other:?}")),
    };
    Ok((rgb, w, h))
}

/// Crop a large RGB8 image to (w,h) taking the top-left region.
fn crop_rgb8(src: &[u8], sw: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        let s = y * sw * 3;
        let d = y * w * 3;
        out[d..d + w * 3].copy_from_slice(&src[s..s + w * 3]);
    }
    out
}

// ============================================================================
// Case definitions
// ============================================================================

#[derive(Clone, Copy)]
enum SrcKind {
    Noise,
    FlatChroma,
    Screen,
    /// Index into the corpus photo table (loaded once at startup).
    Photo(usize),
}

#[derive(Clone, Copy)]
enum ColorSpec {
    Ycbcr(ChromaSubsampling),
    Xyb(XybSubsampling),
    Grayscale,
    Rgb,
}

#[derive(Clone, Copy)]
enum QualSpec {
    Q(f32),
    Mozjpeg(u8),
    Ssim2(f32),
}

#[derive(Clone)]
struct EncCase {
    key: String,
    w: u32,
    h: u32,
    src: SrcKind,
    layout: PixelLayout,
    color: ColorSpec,
    q: QualSpec,
    prog: bool,
    restart: u16,
    opt_huff: Option<bool>,
    allow16: Option<bool>,
    strided: bool,
    mt: bool,
    ds: Option<DownsamplingMethod>,
}

fn sub_name(s: ChromaSubsampling) -> &'static str {
    match s {
        ChromaSubsampling::None => "444",
        ChromaSubsampling::HalfHorizontal => "422",
        ChromaSubsampling::Quarter => "420",
        ChromaSubsampling::HalfVertical => "440",
        _ => "sub?",
    }
}

fn layout_name(l: PixelLayout) -> &'static str {
    match l {
        PixelLayout::Rgb8Srgb => "rgb8",
        PixelLayout::Bgr8Srgb => "bgr8",
        PixelLayout::Rgbx8Srgb => "rgbx8",
        PixelLayout::Rgba8Srgb => "rgba8",
        PixelLayout::Bgrx8Srgb => "bgrx8",
        PixelLayout::Bgra8Srgb => "bgra8",
        PixelLayout::Gray8Srgb => "gray8",
        PixelLayout::Rgb16Linear => "rgb16",
        PixelLayout::Rgbx16Linear => "rgbx16",
        PixelLayout::Rgba16Linear => "rgba16",
        PixelLayout::Gray16Linear => "gray16",
        PixelLayout::RgbF32Linear => "rgbf32",
        PixelLayout::RgbxF32Linear => "rgbxf32",
        PixelLayout::RgbaF32Linear => "rgbaf32",
        PixelLayout::GrayF32Linear => "grayf32",
        PixelLayout::YCbCr8 => "ycbcr8",
        PixelLayout::YCbCrF32 => "ycbcrf32",
        _ => "layout?",
    }
}

fn color_name(c: ColorSpec) -> String {
    match c {
        ColorSpec::Ycbcr(s) => format!("ycbcr-{}", sub_name(s)),
        ColorSpec::Xyb(s) => format!(
            "xyb-{}",
            match s {
                XybSubsampling::Full => "full",
                XybSubsampling::BQuarter => "bquarter",
                _ => "other",
            }
        ),
        ColorSpec::Grayscale => "gray".to_string(),
        ColorSpec::Rgb => "rgb".to_string(),
    }
}

fn qual_name(q: QualSpec) -> String {
    match q {
        QualSpec::Q(q) => format!("q{q}"),
        QualSpec::Mozjpeg(q) => format!("qmoz{q}"),
        QualSpec::Ssim2(q) => format!("qssim2-{q}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn mk_case(
    w: u32,
    h: u32,
    src: SrcKind,
    layout: PixelLayout,
    color: ColorSpec,
    q: QualSpec,
    prog: bool,
    restart: u16,
    opt_huff: Option<bool>,
    allow16: Option<bool>,
    strided: bool,
    mt: bool,
    ds: Option<DownsamplingMethod>,
) -> EncCase {
    let sname = match src {
        SrcKind::Noise => "noise".to_string(),
        SrcKind::FlatChroma => "flatchroma".to_string(),
        SrcKind::Screen => "screen".to_string(),
        SrcKind::Photo(i) => format!("photo{i}"),
    };
    let key = format!(
        "{}x{}|{}|{}|{}|{}|{}|rst{}|opt{}|a16{}|str{}|mt{}|ds{}",
        w,
        h,
        sname,
        layout_name(layout),
        color_name(color),
        if prog { "prog" } else { "base" },
        qual_name(q),
        restart,
        opt_huff.map_or("-".into(), |b| (b as u8).to_string()),
        allow16.map_or("-".into(), |b| (b as u8).to_string()),
        strided as u8,
        mt as u8,
        ds.map_or("-".into(), |d| format!("{d:?}")),
    );
    EncCase {
        key,
        w,
        h,
        src,
        layout,
        color,
        q,
        prog,
        restart,
        opt_huff,
        allow16,
        strided,
        mt,
        ds,
    }
}

fn enc_cases() -> Vec<EncCase> {
    let mut v = Vec::new();
    let base = QualSpec::Q(75.0);
    let ycbcr420 = ColorSpec::Ycbcr(ChromaSubsampling::Quarter);

    // ---- size sweep on noise, default config -------------------------------
    for &(w, h) in &[
        (1u32, 1u32),
        (7, 7),
        (17, 9),
        (64, 64),
        (257, 131),
        (1024, 1024),
        (2048, 1536),
        (8200, 8), // wider than the historic 8192px fixup row buffer
    ] {
        v.push(mk_case(
            w,
            h,
            SrcKind::Noise,
            PixelLayout::Rgb8Srgb,
            ycbcr420,
            base,
            false,
            4,
            None,
            None,
            false,
            false,
            None,
        ));
    }
    // ---- content sweep ------------------------------------------------------
    for &src in &[SrcKind::FlatChroma, SrcKind::Screen] {
        for &(w, h) in &[(257u32, 131u32), (1024, 1024)] {
            v.push(mk_case(
                w,
                h,
                src,
                PixelLayout::Rgb8Srgb,
                ycbcr420,
                base,
                false,
                4,
                None,
                None,
                false,
                false,
                None,
            ));
        }
    }
    // ---- quality sweep ------------------------------------------------------
    for &q in &[5.0f32, 20.0, 40.0, 60.0, 90.0, 98.0] {
        for &src in &[SrcKind::Noise, SrcKind::FlatChroma] {
            v.push(mk_case(
                257,
                131,
                src,
                PixelLayout::Rgb8Srgb,
                ycbcr420,
                QualSpec::Q(q),
                false,
                4,
                None,
                None,
                false,
                false,
                None,
            ));
        }
    }
    // ---- subsampling sweep --------------------------------------------------
    for &s in &[
        ChromaSubsampling::None,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::HalfVertical,
    ] {
        for &src in &[SrcKind::Noise, SrcKind::FlatChroma] {
            v.push(mk_case(
                257,
                131,
                src,
                PixelLayout::Rgb8Srgb,
                ColorSpec::Ycbcr(s),
                base,
                false,
                4,
                None,
                None,
                false,
                false,
                None,
            ));
        }
    }
    // ---- progressive --------------------------------------------------------
    for &src in &[SrcKind::Noise, SrcKind::FlatChroma] {
        for &q in &[20.0f32, 75.0, 90.0] {
            v.push(mk_case(
                257,
                131,
                src,
                PixelLayout::Rgb8Srgb,
                ycbcr420,
                QualSpec::Q(q),
                true,
                4,
                None,
                None,
                false,
                false,
                None,
            ));
        }
    }
    v.push(mk_case(
        1024,
        1024,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        true,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    // progressive without restart markers exercises the serial path
    v.push(mk_case(
        1024,
        1024,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        true,
        0,
        None,
        None,
        false,
        false,
        None,
    ));
    // ---- pixel layouts ------------------------------------------------------
    for &l in &[
        PixelLayout::Bgr8Srgb,
        PixelLayout::Rgbx8Srgb,
        PixelLayout::Rgba8Srgb,
        PixelLayout::Bgrx8Srgb,
        PixelLayout::Bgra8Srgb,
        PixelLayout::Gray8Srgb,
        PixelLayout::Rgb16Linear,
        PixelLayout::Rgbx16Linear,
        PixelLayout::Rgba16Linear,
        PixelLayout::Gray16Linear,
        PixelLayout::RgbF32Linear,
        PixelLayout::RgbxF32Linear,
        PixelLayout::RgbaF32Linear,
        PixelLayout::GrayF32Linear,
        PixelLayout::YCbCr8,
        PixelLayout::YCbCrF32,
    ] {
        v.push(mk_case(
            257,
            131,
            SrcKind::Noise,
            l,
            ycbcr420,
            base,
            false,
            4,
            None,
            None,
            false,
            false,
            None,
        ));
    }
    // ---- strided input (stride > width * bpp) -------------------------------
    for &l in &[
        PixelLayout::Rgb8Srgb,
        PixelLayout::Rgba8Srgb,
        PixelLayout::Gray8Srgb,
    ] {
        v.push(mk_case(
            257,
            131,
            SrcKind::Noise,
            l,
            ycbcr420,
            base,
            false,
            4,
            None,
            None,
            true,
            false,
            None,
        ));
    }
    // ---- XYB ----------------------------------------------------------------
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ColorSpec::Xyb(XybSubsampling::BQuarter),
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ColorSpec::Xyb(XybSubsampling::Full),
        QualSpec::Q(90.0),
        true,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    // ---- restart intervals --------------------------------------------------
    for &r in &[0u16, 1, 16] {
        v.push(mk_case(
            257,
            131,
            SrcKind::Noise,
            PixelLayout::Rgb8Srgb,
            ycbcr420,
            base,
            false,
            r,
            None,
            None,
            false,
            false,
            None,
        ));
    }
    // ---- Huffman strategy / quant tables / quality variants ------------------
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        false,
        4,
        Some(false),
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        QualSpec::Q(50.0),
        false,
        4,
        None,
        Some(true),
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        QualSpec::Q(50.0),
        false,
        4,
        None,
        Some(false),
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        QualSpec::Mozjpeg(80),
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        QualSpec::Ssim2(80.0),
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    // ---- color modes ---------------------------------------------------------
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Gray8Srgb,
        ColorSpec::Grayscale,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ColorSpec::Grayscale,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ColorSpec::Rgb,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    // ---- downsampling methods ------------------------------------------------
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        Some(DownsamplingMethod::GammaAware),
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Noise,
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        Some(DownsamplingMethod::GammaAwareIterative),
    ));
    // ---- dense small grid (64x64) -------------------------------------------
    for &sub in &[ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        for &prog in &[false, true] {
            for &q in &[20.0f32, 75.0, 90.0] {
                v.push(mk_case(
                    64,
                    64,
                    SrcKind::Noise,
                    PixelLayout::Rgb8Srgb,
                    ColorSpec::Ycbcr(sub),
                    QualSpec::Q(q),
                    prog,
                    4,
                    None,
                    None,
                    false,
                    false,
                    None,
                ));
            }
        }
    }
    // ---- flat-chroma DC-only dense grid -------------------------------------
    for &sub in &[
        ChromaSubsampling::None,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfVertical,
    ] {
        for &prog in &[false, true] {
            for &q in &[60.0f32, 90.0] {
                v.push(mk_case(
                    64,
                    64,
                    SrcKind::FlatChroma,
                    PixelLayout::Rgb8Srgb,
                    ColorSpec::Ycbcr(sub),
                    QualSpec::Q(q),
                    prog,
                    4,
                    None,
                    None,
                    false,
                    false,
                    None,
                ));
            }
        }
    }
    // ---- photo content (corpus; resolved at emit) ----------------------------
    v.push(mk_case(
        1024,
        1024,
        SrcKind::Photo(0),
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        2048,
        1536,
        SrcKind::Photo(0),
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        257,
        131,
        SrcKind::Photo(0),
        PixelLayout::Rgb8Srgb,
        ColorSpec::Ycbcr(ChromaSubsampling::None),
        QualSpec::Q(90.0),
        true,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    v.push(mk_case(
        1024,
        1024,
        SrcKind::Photo(1),
        PixelLayout::Rgb8Srgb,
        ycbcr420,
        base,
        false,
        4,
        None,
        None,
        false,
        false,
        None,
    ));
    // ---- MT encodes (SKIP rows when parallel feature is off) -----------------
    for &(w, h) in &[(257u32, 131u32), (1024, 1024), (2048, 1536)] {
        v.push(mk_case(
            w,
            h,
            SrcKind::Noise,
            PixelLayout::Rgb8Srgb,
            ycbcr420,
            base,
            false,
            4,
            None,
            None,
            false,
            true,
            None,
        ));
    }
    v
}

// ============================================================================
// Encode/decode execution
// ============================================================================

/// Convert the RGB8 source into `layout`'s byte stream (deterministic; alpha
/// channels get 0xFF, f32/16-bit get widened values).
fn src_as_layout(rgb8: &[u8], w: usize, h: usize, layout: PixelLayout) -> Vec<u8> {
    let n = w * h;
    let px = |i: usize| [rgb8[i * 3], rgb8[i * 3 + 1], rgb8[i * 3 + 2]];
    match layout {
        PixelLayout::Rgb8Srgb | PixelLayout::YCbCr8 => rgb8.to_vec(),
        PixelLayout::Bgr8Srgb => (0..n)
            .flat_map(|i| {
                let p = px(i);
                [p[2], p[1], p[0]]
            })
            .collect(),
        PixelLayout::Gray8Srgb => (0..n).map(|i| rgb8[i * 3]).collect(),
        PixelLayout::Rgbx8Srgb | PixelLayout::Rgba8Srgb => (0..n)
            .flat_map(|i| {
                let p = px(i);
                [p[0], p[1], p[2], 255]
            })
            .collect(),
        PixelLayout::Bgrx8Srgb | PixelLayout::Bgra8Srgb => (0..n)
            .flat_map(|i| {
                let p = px(i);
                [p[2], p[1], p[0], 255]
            })
            .collect(),
        PixelLayout::Rgb16Linear => (0..n)
            .flat_map(|i| {
                let p = px(i);
                [p[0], p[1], p[2]]
            })
            .flat_map(|v| (v as u16 * 257).to_le_bytes())
            .collect(),
        PixelLayout::Rgbx16Linear | PixelLayout::Rgba16Linear => (0..n)
            .flat_map(|i| {
                let p = px(i);
                [p[0], p[1], p[2], 255]
            })
            .flat_map(|v| (v as u16 * 257).to_le_bytes())
            .collect(),
        PixelLayout::Gray16Linear => (0..n)
            .map(|i| rgb8[i * 3])
            .flat_map(|v| (v as u16 * 257).to_le_bytes())
            .collect(),
        PixelLayout::RgbF32Linear | PixelLayout::YCbCrF32 => (0..n)
            .flat_map(|i| {
                let p = px(i);
                [p[0], p[1], p[2]]
            })
            .flat_map(|v| (v as f32 / 255.0).to_le_bytes())
            .collect(),
        PixelLayout::RgbxF32Linear | PixelLayout::RgbaF32Linear => (0..n)
            .flat_map(|i| {
                let p = px(i);
                [p[0], p[1], p[2], 255]
            })
            .flat_map(|v| (v as f32 / 255.0).to_le_bytes())
            .collect(),
        PixelLayout::GrayF32Linear => (0..n)
            .map(|i| rgb8[i * 3])
            .flat_map(|v| (v as f32 / 255.0).to_le_bytes())
            .collect(),
        _ => rgb8.to_vec(),
    }
}

fn run_encode(case: &EncCase, src_rgb: &[u8]) -> Result<Vec<u8>, String> {
    let q: Quality = match case.q {
        QualSpec::Q(q) => q.into(),
        QualSpec::Mozjpeg(q) => Quality::ApproxMozjpeg(q),
        QualSpec::Ssim2(q) => Quality::ApproxSsim2(q),
    };
    let mut cfg = match case.color {
        ColorSpec::Ycbcr(s) => EncoderConfig::ycbcr(q, s),
        ColorSpec::Xyb(s) => EncoderConfig::xyb(q, s),
        ColorSpec::Grayscale => EncoderConfig::grayscale(q),
        ColorSpec::Rgb => EncoderConfig::rgb(q),
    }
    .progressive(case.prog)
    .restart_mcu_rows(case.restart);
    if let Some(v) = case.opt_huff {
        cfg = cfg.optimize_huffman(v);
    }
    if let Some(v) = case.allow16 {
        cfg = cfg.allow_16bit_quant_tables(v);
    }
    if let Some(m) = case.ds {
        cfg = cfg.downsampling_method(m);
    }
    #[cfg(feature = "parallel")]
    let cfg = if case.mt {
        cfg.parallel(ParallelEncoding::Auto)
    } else {
        cfg
    };
    #[cfg(not(feature = "parallel"))]
    let cfg = if case.mt {
        return Err("parallel feature off".into());
    } else {
        cfg
    };

    let (w, h) = (case.w as usize, case.h as usize);
    let data = src_as_layout(src_rgb, w, h, case.layout);
    let mut enc = cfg
        .encode_from_bytes(case.w, case.h, case.layout)
        .map_err(|e| format!("{:?}", e.0.error()))?;
    if case.strided {
        let bpp = case.layout.bytes_per_pixel();
        let stride = w * bpp + 13; // odd byte pad exercises stride handling
        let mut padded = vec![0xABu8; stride * h];
        let row_bytes = w * bpp;
        for y in 0..h {
            padded[y * stride..y * stride + row_bytes]
                .copy_from_slice(&data[y * row_bytes..(y + 1) * row_bytes]);
        }
        enc.push(&padded, h, stride, Unstoppable)
            .map_err(|e| format!("{:?}", e.0.error()))?;
    } else {
        enc.push_packed(&data, Unstoppable)
            .map_err(|e| format!("{:?}", e.0.error()))?;
    }
    enc.finish().map_err(|e| format!("{:?}", e.0.error()))
}

// ---- decode paths -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ups {
    Fancy,
    Box,
}

#[derive(Debug, Clone, PartialEq)]
enum DecKind {
    /// Full-image decode to u8 pixels.
    Full {
        fmt: PixelFormat,
        up: Ups,
        idct: IdctMethod,
        threads: usize,
    },
    /// Full-image decode to an f32 OutputTarget.
    FullF32 {
        target: OutputTarget,
        up: Ups,
        threads: usize,
    },
    /// decode_into caller-provided buffer.
    Into { fmt: PixelFormat },
    /// ScanlineReader in batches of `batch` rows; stride has 5-byte pad.
    Scan {
        batch: usize,
        fmt: PixelFormat,
        up: Ups,
        threads: usize,
    },
    /// Coefficient-domain decode; hash coeffs+quant tables.
    Coeffs,
    /// Full-image decode to planar YCbCr f32.
    YcbcrF32,
}

fn dec_paths() -> Vec<(&'static str, DecKind)> {
    #[allow(unused_mut)]
    let mut v: Vec<(&'static str, DecKind)> = vec![
        (
            "full-rgb-fancy-st",
            DecKind::Full {
                fmt: PixelFormat::Rgb,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-rgb-box-st",
            DecKind::Full {
                fmt: PixelFormat::Rgb,
                up: Ups::Box,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-rgb-fancy-jpegli",
            DecKind::Full {
                fmt: PixelFormat::Rgb,
                up: Ups::Fancy,
                idct: IdctMethod::Jpegli,
                threads: 1,
            },
        ),
        (
            "full-rgba-st",
            DecKind::Full {
                fmt: PixelFormat::Rgba,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-bgr-st",
            DecKind::Full {
                fmt: PixelFormat::Bgr,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-bgra-st",
            DecKind::Full {
                fmt: PixelFormat::Bgra,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-gray-st",
            DecKind::Full {
                fmt: PixelFormat::Gray,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-rgb16-st",
            DecKind::Full {
                fmt: PixelFormat::Rgb16,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-f32-srgb-st",
            DecKind::FullF32 {
                target: OutputTarget::SrgbF32,
                up: Ups::Fancy,
                threads: 1,
            },
        ),
        (
            "full-f32-linear-st",
            DecKind::FullF32 {
                target: OutputTarget::LinearF32,
                up: Ups::Fancy,
                threads: 1,
            },
        ),
        (
            "full-f32-srgb-precise-st",
            DecKind::FullF32 {
                target: OutputTarget::SrgbF32Precise,
                up: Ups::Fancy,
                threads: 1,
            },
        ),
        (
            "into-rgb-st",
            DecKind::Into {
                fmt: PixelFormat::Rgb,
            },
        ),
        (
            "into-rgba-st",
            DecKind::Into {
                fmt: PixelFormat::Rgba,
            },
        ),
        (
            "scan-rgb-b7-st",
            DecKind::Scan {
                batch: 7,
                fmt: PixelFormat::Rgb,
                up: Ups::Fancy,
                threads: 1,
            },
        ),
        (
            "scan-rgb-b1-st",
            DecKind::Scan {
                batch: 1,
                fmt: PixelFormat::Rgb,
                up: Ups::Fancy,
                threads: 1,
            },
        ),
        (
            "scan-rgba-b4-st",
            DecKind::Scan {
                batch: 4,
                fmt: PixelFormat::Rgba,
                up: Ups::Fancy,
                threads: 1,
            },
        ),
        (
            "scan-rgb-box-b7-st",
            DecKind::Scan {
                batch: 7,
                fmt: PixelFormat::Rgb,
                up: Ups::Box,
                threads: 1,
            },
        ),
        ("coeffs", DecKind::Coeffs),
        ("ycbcr-f32", DecKind::YcbcrF32),
    ];
    #[cfg(feature = "parallel")]
    {
        v.push((
            "full-rgb-fancy-mt",
            DecKind::Full {
                fmt: PixelFormat::Rgb,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 0,
            },
        ));
        v.push((
            "full-rgb-box-mt",
            DecKind::Full {
                fmt: PixelFormat::Rgb,
                up: Ups::Box,
                idct: IdctMethod::Libjpeg,
                threads: 0,
            },
        ));
        v.push((
            "full-f32-srgb-mt",
            DecKind::FullF32 {
                target: OutputTarget::SrgbF32,
                up: Ups::Fancy,
                threads: 0,
            },
        ));
        v.push((
            "scan-rgb-box-mt",
            DecKind::Scan {
                batch: 7,
                fmt: PixelFormat::Rgb,
                up: Ups::Box,
                threads: 0,
            },
        ));
    }
    v
}

/// Reduced path set for corpus JPEGs (keeps runtime sane across ~14 tiers).
fn corpus_dec_paths() -> Vec<(&'static str, DecKind)> {
    #[allow(unused_mut)]
    let mut v: Vec<(&'static str, DecKind)> = vec![
        (
            "full-rgb-fancy-st",
            DecKind::Full {
                fmt: PixelFormat::Rgb,
                up: Ups::Fancy,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "full-rgb-box-st",
            DecKind::Full {
                fmt: PixelFormat::Rgb,
                up: Ups::Box,
                idct: IdctMethod::Libjpeg,
                threads: 1,
            },
        ),
        (
            "scan-rgb-b7-st",
            DecKind::Scan {
                batch: 7,
                fmt: PixelFormat::Rgb,
                up: Ups::Fancy,
                threads: 1,
            },
        ),
        ("coeffs", DecKind::Coeffs),
    ];
    #[cfg(feature = "parallel")]
    v.push((
        "full-rgb-fancy-mt",
        DecKind::Full {
            fmt: PixelFormat::Rgb,
            up: Ups::Fancy,
            idct: IdctMethod::Libjpeg,
            threads: 0,
        },
    ));
    v
}

fn run_decode(jpeg: &[u8], kind: &DecKind) -> Result<Vec<u8>, String> {
    match kind {
        DecKind::Full {
            fmt,
            up,
            idct,
            threads,
        } => {
            let r = Decoder::new()
                .output_format(*fmt)
                .num_threads(*threads)
                .chroma_upsampling(match up {
                    Ups::Fancy => ChromaUpsampling::Triangle,
                    Ups::Box => ChromaUpsampling::NearestNeighbor,
                })
                .idct_method(*idct)
                .decode(jpeg, Unstoppable)
                .map_err(|e| format!("{:?}", e.0.error()))?;
            match r.pixels_u8() {
                Some(px) => Ok(px.to_vec()),
                None => Err("no u8 pixels".into()),
            }
        }
        DecKind::FullF32 {
            target,
            up,
            threads,
        } => {
            let r = Decoder::new()
                .num_threads(*threads)
                .output_target(*target)
                .chroma_upsampling(match up {
                    Ups::Fancy => ChromaUpsampling::Triangle,
                    Ups::Box => ChromaUpsampling::NearestNeighbor,
                })
                .decode(jpeg, Unstoppable)
                .map_err(|e| format!("{:?}", e.0.error()))?;
            match r.pixels_f32() {
                Some(px) => Ok(px.iter().flat_map(|v| v.to_le_bytes()).collect()),
                None => Err("no f32 pixels".into()),
            }
        }
        DecKind::Into { fmt } => {
            let d = Decoder::new().num_threads(1);
            let info = d
                .read_info(jpeg)
                .map_err(|e| format!("{:?}", e.0.error()))?;
            let (w, h) = (
                info.dimensions.width as usize,
                info.dimensions.height as usize,
            );
            let bpp = fmt.bytes_per_pixel();
            let mut dst = vec![0xCDu8; w * h * bpp];
            let n = d
                .decode_into(jpeg, *fmt, &mut dst, Unstoppable)
                .map_err(|e| format!("{:?}", e.0.error()))?;
            dst.truncate(n);
            Ok(dst)
        }
        DecKind::Scan {
            batch,
            fmt,
            up,
            threads,
        } => {
            let d = Decoder::new()
                .num_threads(*threads)
                .chroma_upsampling(match up {
                    Ups::Fancy => ChromaUpsampling::Triangle,
                    Ups::Box => ChromaUpsampling::NearestNeighbor,
                });
            let mut r = d
                .scanline_reader(jpeg)
                .map_err(|e| format!("{:?}", e.0.error()))?;
            let (w, h) = (r.width() as usize, r.height() as usize);
            if w == 0 || h == 0 {
                return Err("zero dims".into());
            }
            let bpp = fmt.bytes_per_pixel();
            let row_bytes = w * bpp;
            let stride = row_bytes + 5; // sentinel padding catches wrong-pitch writes
            let mut buf = vec![0x77u8; stride * h];
            let mut row = 0usize;
            while row < h {
                let take = (*batch).min(h - row);
                let out = imgref::ImgRefMut::new_stride(
                    &mut buf[row * stride..],
                    row_bytes,
                    take,
                    stride,
                );
                let got = match fmt {
                    PixelFormat::Rgb => r.read_rows_rgb8(out),
                    PixelFormat::Rgba => r.read_rows_rgba8(out),
                    PixelFormat::Bgr => r.read_rows_bgr8(out),
                    PixelFormat::Gray => r.read_rows_gray8(out),
                    _ => return Err("scan fmt unsupported".into()),
                }
                .map_err(|e| format!("{:?}", e.0.error()))?;
                if got == 0 {
                    break;
                }
                row += got;
            }
            Ok(buf)
        }
        DecKind::Coeffs => {
            let c = Decoder::new()
                .num_threads(1)
                .decode_coefficients(jpeg, Unstoppable)
                .map_err(|e| format!("{:?}", e.0.error()))?;
            let mut out = Vec::new();
            out.extend_from_slice(&c.width.to_le_bytes());
            out.extend_from_slice(&c.height.to_le_bytes());
            for comp in &c.components {
                out.push(comp.id);
                out.extend_from_slice(&(comp.blocks_wide as u32).to_le_bytes());
                out.extend_from_slice(&(comp.blocks_high as u32).to_le_bytes());
                for &v in &comp.coeffs {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            for t in &c.quant_tables {
                match t {
                    Some(tab) => {
                        out.push(1);
                        for &v in tab.iter() {
                            out.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    None => out.push(0),
                }
            }
            Ok(out)
        }
        DecKind::YcbcrF32 => {
            let y = Decoder::new()
                .num_threads(1)
                .decode_to_ycbcr_f32(jpeg, Unstoppable)
                .map_err(|e| format!("{:?}", e.0.error()))?;
            let mut out = Vec::with_capacity(y.plane_size() * 12);
            for v in y.y.iter().chain(y.cb.iter()).chain(y.cr.iter()) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
    }
}

// ============================================================================
// Emit / compare
// ============================================================================

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    let mut s = String::with_capacity(64);
    for b in h.finalize() {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Run one case catching panics; Ok bytes -> (bytes), Err -> tagged string.
fn capture(f: impl FnOnce() -> Result<Vec<u8>, String>) -> Result<Vec<u8>, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err("PANIC".into()),
    }
}

/// All corpus JPEGs (name -> bytes). Traversal failures are counted inside
/// `Corpus::scan_jpegs` and drive the `#corpus` provenance/gate — they are
/// never silently swallowed here.
fn corpus_jpegs(corpus: &Corpus) -> Vec<(String, Vec<u8>)> {
    corpus.scan_jpegs().0
}

/// Photo sources for `SrcKind::Photo(i)`: RGB8 pixels + dims. Slot-stable —
/// `Some` at index `i` iff that source resolved, so a missing file yields a
/// `SKIP:` row at its own key instead of shifting a later photo into the slot.
/// `i=0` is the first CLIC final-test PNG >=2048x1536 (photograph); `i=1` is
/// the first qualifying imageflow PNG (screenshot-like/synthetic).
fn corpus_photos(corpus: &Corpus) -> Vec<Option<(Vec<u8>, usize, usize)>> {
    corpus.scan_photos().0
}

/// Collected manifest: rows + the report string + provenance metadata.
/// Corpus state after collection. The traversal that loaded the rows is
/// authoritative — a preflight `Ok` downgrades to `Partial` when collection
/// itself hit failures, and a stale `Partial` count is replaced by what the
/// collection actually observed. `Disabled` is terminal.
fn collected_corpus_state(initial: CorpusState, missing: usize) -> CorpusState {
    match initial {
        CorpusState::Disabled => CorpusState::Disabled,
        _ if missing > 0 => CorpusState::Partial(missing),
        _ => CorpusState::Ok,
    }
}

struct Collected {
    rows: BTreeMap<String, (String, usize)>,
    report: String,
    corpus_state: CorpusState,
    filter: Option<String>,
}

fn collect_rows(corpus: &Corpus, corpus_state: CorpusState) -> Collected {
    let cases = enc_cases();
    let paths = dec_paths();
    let cpaths = corpus_dec_paths();
    let (photo_imgs, photo_failures) = corpus.scan_photos();
    let (jpegs, jpeg_failures) = corpus.scan_jpegs();
    // Corpus present but incomplete: fold the failure count from THIS
    // traversal into provenance — the gate and the rows share one result.
    let missing =
        jpeg_failures + photo_failures + photo_imgs.iter().filter(|s| s.is_none()).count();
    let corpus_state = collected_corpus_state(corpus_state, missing);
    let filter = std::env::var("MANIFEST_FILTER").ok();

    // Resolve each generated case's RGB8 source once (identical across tiers);
    // photos are cropped per-case inside the loop.
    let srcs: Vec<Option<Vec<u8>>> = cases
        .iter()
        .map(|c| match c.src {
            SrcKind::Noise => Some(gen_noise_patches(c.w as usize, c.h as usize)),
            SrcKind::FlatChroma => Some(gen_flat_chroma(c.w as usize, c.h as usize)),
            SrcKind::Screen => Some(gen_screen(c.w as usize, c.h as usize)),
            SrcKind::Photo(_) => None,
        })
        .collect();

    let mut rows: BTreeMap<String, (String, usize)> = BTreeMap::new();

    let report = archmage::testing::for_each_token_permutation(
        archmage::testing::CompileTimePolicy::Warn,
        |perm| {
            for (case, src) in cases.iter().zip(srcs.iter()) {
                if let Some(f) = &filter
                    && !case.key.contains(f.as_str())
                {
                    continue;
                }
                let ekey = format!("enc|{}|{}", case.key, perm.label);
                // Resolve source pixels (photos cropped to case dims).
                let src_rgb: Option<std::borrow::Cow<'_, [u8]>> = match case.src {
                    SrcKind::Photo(i) => {
                        photo_imgs
                            .get(i)
                            .and_then(|s| s.as_ref())
                            .and_then(|(rgb, sw, sh)| {
                                let (w, h) = (case.w as usize, case.h as usize);
                                (*sw >= w && *sh >= h)
                                    .then(|| std::borrow::Cow::Owned(crop_rgb8(rgb, *sw, w, h)))
                            })
                    }
                    _ => src.as_deref().map(std::borrow::Cow::Borrowed),
                };
                let jpeg = match &src_rgb {
                    Some(px) => match capture(|| run_encode(case, px.as_ref())) {
                        Ok(bytes) => {
                            rows.insert(ekey, (sha256_hex(&bytes), bytes.len()));
                            Some(bytes)
                        }
                        Err(e) => {
                            let tag = if e == "PANIC" {
                                "PANIC".into()
                            } else {
                                format!("ERR:{e}")
                            };
                            rows.insert(ekey, (tag, 0));
                            None
                        }
                    },
                    None => {
                        rows.insert(ekey, ("SKIP:no-src".into(), 0));
                        None
                    }
                };
                for (frag, kind) in &paths {
                    let dkey = format!("dec|self/{}|{}|{}", case.key, frag, perm.label);
                    match &jpeg {
                        Some(j) => match capture(|| run_decode(j, kind)) {
                            Ok(bytes) => {
                                rows.insert(dkey, (sha256_hex(&bytes), bytes.len()));
                            }
                            Err(e) => {
                                let tag = if e == "PANIC" {
                                    "PANIC".into()
                                } else {
                                    format!("ERR:{e}")
                                };
                                rows.insert(dkey, (tag, 0));
                            }
                        },
                        None => {
                            rows.insert(dkey, ("SKIP:enc".into(), 0));
                        }
                    }
                }
            }
            for (name, bytes) in &jpegs {
                if let Some(f) = &filter
                    && !name.contains(f.as_str())
                {
                    continue;
                }
                for (frag, kind) in &cpaths {
                    let dkey = format!("dec|corpus/{}|{}|{}", name, frag, perm.label);
                    match capture(|| run_decode(bytes, kind)) {
                        Ok(px) => {
                            rows.insert(dkey, (sha256_hex(&px), px.len()));
                        }
                        Err(e) => {
                            let tag = if e == "PANIC" {
                                "PANIC".into()
                            } else {
                                format!("ERR:{e}")
                            };
                            rows.insert(dkey, (tag, 0));
                        }
                    }
                }
            }
        },
    );
    Collected {
        rows,
        report: format!("{report}"),
        corpus_state,
        filter,
    }
}

/// `#`-prefixed provenance lines. Compared verbatim by `compare` — two
/// manifests only verify the same scope when all provenance keys match.
fn provenance_lines(c: &Collected) -> Vec<(String, String)> {
    let corpus = match c.corpus_state {
        CorpusState::Ok => "ok".to_string(),
        CorpusState::Partial(n) => format!("partial:{n}"),
        CorpusState::Disabled => "disabled-explicit".to_string(),
    };
    vec![
        ("manifest".into(), "v2".into()),
        (
            "features".into(),
            format!("parallel={}", cfg!(feature = "parallel")),
        ),
        ("host".into(), std::env::consts::ARCH.to_string()),
        (
            "scope".into(),
            match &c.filter {
                Some(f) => format!("filter={f}"),
                None => "all".into(),
            },
        ),
        ("corpus".into(), corpus),
        ("rows".into(), c.rows.len().to_string()),
    ]
}

fn emit(out_path: Option<&str>, allow_missing_corpus: bool) -> Result<(), String> {
    let corpus = Corpus::try_open();
    // Scope must be caller-explicit: emit only proceeds when the corpus is
    // fully present, or the caller opted into a reduced corpus scope. A
    // resolvable cache with missing dirs is reduced scope too — never emit
    // it silently.
    let initial = if !corpus.available() {
        CorpusState::Disabled
    } else {
        let missing = corpus.missing_expected();
        if missing > 0 {
            CorpusState::Partial(missing)
        } else {
            CorpusState::Ok
        }
    };
    if initial != CorpusState::Ok && !allow_missing_corpus {
        return Err(format!(
            "codec corpus incomplete or unavailable (state={initial:?}); pass \
             --allow-missing-corpus to emit a reduced-scope manifest (recorded \
             in #corpus provenance)"
        ));
    }
    let c = collect_rows(&corpus, initial);
    eprintln!("permutations: {}", c.report);
    eprintln!("rows: {}", c.rows.len());
    if c.rows.is_empty() {
        return Err(format!(
            "empty effective scope: zero rows collected (filter={:?} corpus={:?})",
            c.filter, c.corpus_state
        ));
    }
    let mut buf = String::new();
    for (k, v) in provenance_lines(&c) {
        let _ = writeln!(buf, "#{k}\t{v}");
    }
    for (k, (v, n)) in &c.rows {
        let _ = writeln!(buf, "{k}\t{v}\t{n}");
    }
    let written = match out_path {
        Some(p) => std::fs::write(p, buf).map_err(|e| e.to_string()),
        None => {
            let stdout = std::io::stdout();
            let mut h = stdout.lock();
            h.write_all(buf.as_bytes()).map_err(|e| e.to_string())
        }
    };
    // The traversal that loaded the rows is authoritative: coverage lost
    // between the preflight scan and collection still counts as reduced
    // scope. The manifest is written first — it is the honest record of
    // what was collected — but emitting it without explicit opt-in fails.
    if c.corpus_state != CorpusState::Ok && !allow_missing_corpus {
        return Err(format!(
            "codec corpus incomplete or unavailable (state={:?}); pass \
             --allow-missing-corpus to emit a reduced-scope manifest (recorded \
             in #corpus provenance)",
            c.corpus_state
        ));
    }
    written
}

/// A strictly-parsed manifest: `#key<TAB>value` provenance plus
/// `key<TAB>hash_or_tag<TAB>len` rows. Malformed input is a hard error —
/// a verification gate must never fail open on truncated or edited files.
struct ParsedManifest {
    provenance: BTreeMap<String, String>,
    rows: BTreeMap<String, (String, usize)>,
}

/// A manifest row value is either a 64-hex SHA-256 or a documented status
/// tag: `PANIC`, `SKIP:<reason>`, or `ERR:<error-kind>`. Anything else is
/// not a value this tool could have emitted — corrupt evidence must not
/// certify equality.
fn valid_digest(v: &str) -> bool {
    if v.len() == 64
        && v.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return true;
    }
    if v == "PANIC" {
        return true;
    }
    for pre in ["ERR:", "SKIP:"] {
        if let Some(rest) = v.strip_prefix(pre) {
            return !rest.is_empty() && rest.bytes().all(|b| (32..127).contains(&b));
        }
    }
    false
}

/// Provenance keys a v2 manifest must carry (in addition to #manifest=v2).
const V2_REQUIRED_PROV: &[&str] = &["features", "host", "scope", "corpus", "rows"];

fn parse_manifest(path: &str) -> Result<ParsedManifest, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut prov: BTreeMap<String, String> = BTreeMap::new();
    let mut rows: BTreeMap<String, (String, usize)> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let lno = i + 1;
        if line.is_empty() {
            return Err(format!("{path}:{lno}: blank line"));
        }
        if let Some(rest) = line.strip_prefix('#') {
            let mut it = rest.splitn(2, '\t');
            let (Some(k), Some(v)) = (it.next(), it.next()) else {
                return Err(format!("{path}:{lno}: malformed provenance line"));
            };
            if k.is_empty() {
                return Err(format!("{path}:{lno}: empty provenance key"));
            }
            if prov.insert(k.to_string(), v.to_string()).is_some() {
                return Err(format!("{path}:{lno}: duplicate provenance key {k:?}"));
            }
            continue;
        }
        let mut it = line.split('\t');
        let (Some(k), Some(v), Some(n)) = (it.next(), it.next(), it.next()) else {
            return Err(format!(
                "{path}:{lno}: malformed row (expected key<TAB>hash<TAB>len)"
            ));
        };
        if it.next().is_some() {
            return Err(format!("{path}:{lno}: too many columns"));
        }
        if k.is_empty() {
            return Err(format!("{path}:{lno}: empty case key"));
        }
        if !valid_digest(v) {
            return Err(format!("{path}:{lno}: invalid digest {v:?}"));
        }
        let len: usize = n
            .parse()
            .map_err(|_| format!("{path}:{lno}: invalid length {n:?}"))?;
        if rows.insert(k.to_string(), (v.to_string(), len)).is_some() {
            return Err(format!("{path}:{lno}: duplicate key {k:?}"));
        }
    }
    // Schema gate: the file must declare this tool's manifest version and
    // carry the full v2 provenance set. Headerless files, unknown versions
    // and incomplete v2 headers are rejected outright — there is no silent
    // "legacy" mode that would let truncated or foreign evidence pass.
    match prov.get("manifest").map(String::as_str) {
        Some("v2") => {}
        Some(v) => return Err(format!("{path}: unsupported manifest version {v:?}")),
        None => {
            return Err(format!(
                "{path}: no #manifest provenance — not a v2 manifest"
            ));
        }
    }
    for req in V2_REQUIRED_PROV {
        if !prov.contains_key(*req) {
            return Err(format!("{path}: missing required #{req} provenance"));
        }
    }
    // Truncation guard: the declared #rows must equal the parsed count —
    // a mismatch means the file was cut mid-write.
    match prov
        .get("rows")
        .expect("required-prov gate passed")
        .parse::<usize>()
    {
        Ok(declared) if declared == rows.len() => {}
        Ok(declared) => {
            return Err(format!(
                "{path}: #rows declares {declared} but {} rows parsed",
                rows.len()
            ));
        }
        Err(_) => return Err(format!("{path}: invalid #rows value")),
    }
    if rows.is_empty() {
        return Err(format!("{path}: zero data rows — nothing was verified"));
    }
    Ok(ParsedManifest {
        provenance: prov,
        rows,
    })
}

fn compare(a_path: &str, b_path: &str) -> Result<i32, String> {
    let a = parse_manifest(a_path)?;
    let b = parse_manifest(b_path)?;
    let mut diffs = 0usize;
    // Scope/provenance must match: a corpus-disabled or differently-filtered
    // manifest cannot pass as evidence for the full matrix.
    let mut prov_keys: std::collections::BTreeSet<&String> = a.provenance.keys().collect();
    prov_keys.extend(b.provenance.keys());
    let mut prov_diffs = 0usize;
    for k in prov_keys {
        match (a.provenance.get(k), b.provenance.get(k)) {
            (Some(x), Some(y)) if x == y => {}
            (x, y) => {
                prov_diffs += 1;
                println!(
                    "PROV-DIFF\t#{k}\t{}\t{}",
                    x.map_or("<absent>", String::as_str),
                    y.map_or("<absent>", String::as_str)
                );
            }
        }
    }
    if a.provenance.get("manifest") != b.provenance.get("manifest") {
        // already reported via the loop above; note kept for clarity
    }
    let mut missing_b = 0usize;
    let mut missing_a = 0usize;
    for (k, v) in &a.rows {
        match b.rows.get(k) {
            None => {
                missing_b += 1;
                println!("MISSING-IN-B\t{k}");
            }
            Some(vb) if vb != v => {
                diffs += 1;
                println!("DIFF\t{k}\t{}\t{}", v.0, vb.0);
            }
            _ => {}
        }
    }
    for k in b.rows.keys() {
        if !a.rows.contains_key(k) {
            missing_a += 1;
            println!("MISSING-IN-A\t{k}");
        }
    }
    println!(
        "compare: {} rows A, {} rows B, {} diff, {} missing-in-B, {} missing-in-A, {} provenance diffs",
        a.rows.len(),
        b.rows.len(),
        diffs,
        missing_b,
        missing_a,
        prov_diffs
    );
    Ok((diffs + missing_a + missing_b + prov_diffs > 0) as i32)
}

/// Full v2 provenance block shared by the check fixtures — emit writes the
/// same six keys, so fixtures exercise the real required-schema.
const TEST_PROV: &str =
    "#manifest\tv2\n#features\tparallel=false\n#host\tx86_64\n#scope\tall\n#corpus\tok\n";

/// One strictness/coverage assertion group, callable from both the
/// `self-test` subcommand and `cargo test --example perf_manifest`.
fn check_parser_strict() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("perf-manifest-selftest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let write = |name: &str, body: &str| -> Result<String, String> {
        let p = dir.join(name);
        std::fs::write(&p, body).map_err(|e| e.to_string())?;
        Ok(p.to_string_lossy().into_owned())
    };
    let sha1 = "a".repeat(64);
    let sha2 = "b".repeat(64);
    // Valid v2 manifest: full required provenance + hex rows + tag rows.
    let a = write(
        "a.tsv",
        &format!(
            "{TEST_PROV}#rows\t5\nk1\t{sha1}\t4\nk2\t{sha2}\t3\n\
             k3\tPANIC\t0\nk4\tSKIP:no-src\t0\nk5\tERR:InvalidBufferSize\t0\n"
        ),
    )?;
    parse_manifest(&a).map_err(|e| format!("valid manifest rejected: {e}"))?;
    for (name, body) in [
        ("empty.tsv", String::new()),
        (
            "blank-line.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\t{sha1}\t4\n\n"),
        ),
        ("headerless.tsv", format!("k1\t{sha1}\t4\n")),
        (
            "short-row.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\t{sha1}\n"),
        ),
        (
            "extra-col.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\t{sha1}\t4\tx\n"),
        ),
        (
            "bad-len.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\t{sha1}\tnan\n"),
        ),
        (
            "dup-key.tsv",
            format!("{TEST_PROV}#rows\t2\nk1\t{sha1}\t4\nk1\t{sha2}\t4\n"),
        ),
        (
            "dup-prov.tsv",
            format!("#manifest\tv2\n#manifest\tv2\n{TEST_PROV}#rows\t1\nk1\t{sha1}\t4\n"),
        ),
        (
            "bad-prov.tsv",
            format!("#notabvalue\n{TEST_PROV}#rows\t1\nk1\t{sha1}\t4\n"),
        ),
        (
            "empty-prov-key.tsv",
            format!("#\tv\n{TEST_PROV}#rows\t1\nk1\t{sha1}\t4\n"),
        ),
        ("v2-missing-rows.tsv", format!("{TEST_PROV}k1\t{sha1}\t4\n")),
        (
            "missing-scope.tsv",
            TEST_PROV.replace("#scope\tall\n", "") + &format!("#rows\t1\nk1\t{sha1}\t4\n"),
        ),
        (
            "unknown-version.tsv",
            format!("#manifest\tv99\n{TEST_PROV}#rows\t1\nk1\t{sha1}\t4\n")
                .replace("#manifest\tv2\n", ""),
        ),
        (
            "empty-key.tsv",
            format!("{TEST_PROV}#rows\t1\n\t{sha1}\t4\n"),
        ),
        (
            "empty-digest.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\t\t4\n"),
        ),
        (
            "bad-digest.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\tnot-a-sha-or-tag\t4\n"),
        ),
        (
            "short-digest.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\tdeadbeef\t4\n"),
        ),
        (
            "upper-digest.tsv",
            format!("{TEST_PROV}#rows\t1\nk1\t{}\t4\n", "A".repeat(64)),
        ),
        (
            "rows-mismatch.tsv",
            format!("{TEST_PROV}#rows\t7\nk1\t{sha1}\t4\n"),
        ),
        (
            "rows-badnum.tsv",
            format!("{TEST_PROV}#rows\tnan\nk1\t{sha1}\t4\n"),
        ),
    ] {
        let p = write(name, &body)?;
        if parse_manifest(&p).is_ok() {
            return Err(format!("{name} parsed OK but must be rejected"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// compare must catch hash flips, missing keys, and provenance/scope drift —
/// and must pass identical manifests.
fn check_compare_semantics() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("perf-manifest-cmptest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let write = |name: &str, body: &str| -> Result<String, String> {
        let p = dir.join(name);
        std::fs::write(&p, body).map_err(|e| e.to_string())?;
        Ok(p.to_string_lossy().into_owned())
    };
    let sha1 = "a".repeat(64);
    let sha2 = "b".repeat(64);
    let a = write(
        "a.tsv",
        &format!("{TEST_PROV}#rows\t2\nk1\t{sha1}\t4\nk2\t{sha2}\t3\n"),
    )?;
    let same = write(
        "b-same.tsv",
        &format!("{TEST_PROV}#rows\t2\nk1\t{sha1}\t4\nk2\t{sha2}\t3\n"),
    )?;
    let hashflip = write(
        "b-hash.tsv",
        &format!(
            "{TEST_PROV}#rows\t2\nk1\t{}\t4\nk2\t{sha2}\t3\n",
            "c".repeat(64)
        ),
    )?;
    let missing = write(
        "b-miss.tsv",
        &format!("{TEST_PROV}#rows\t1\nk1\t{sha1}\t4\n"),
    )?;
    let prov = write(
        "b-prov.tsv",
        &format!("{TEST_PROV}#rows\t2\nk1\t{sha1}\t4\nk2\t{sha2}\t3\n")
            .replace("parallel=false", "parallel=true"),
    )?;
    if compare(&a, &same)? != 0 {
        return Err("identical manifests flagged".into());
    }
    if compare(&a, &hashflip)? == 0 {
        return Err("hash flip undetected".into());
    }
    if compare(&a, &missing)? == 0 {
        return Err("missing key undetected".into());
    }
    if compare(&a, &prov)? == 0 {
        return Err("provenance mismatch undetected".into());
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// Empty-but-resolved corpus dirs count as missing coverage, not Ok —
/// the r2 reviewer repro: all eight expected dirs exist, all empty.
/// Directory presence alone can never satisfy the coverage gate.
fn check_corpus_gate_counts_contents() -> Result<(), String> {
    let base =
        std::env::temp_dir().join(format!("perf-manifest-emptycorpus-{}", std::process::id()));
    let inner = codec_corpus::Corpus::with_cache_root(&base).map_err(|e| e.to_string())?;
    let root = base.join("codec-corpus/v1");
    for rel in CORPUS_JPEG_DIRS.iter().chain(CORPUS_PHOTO_DIRS) {
        std::fs::create_dir_all(root.join(rel)).map_err(|e| e.to_string())?;
    }
    std::fs::write(root.join(".version"), "1.1.0\n").map_err(|e| e.to_string())?;
    let corpus = Corpus { inner: Some(inner) };
    if !corpus_jpegs(&corpus).is_empty() {
        return Err("empty resolved cache yielded JPEGs".into());
    }
    if corpus_photos(&corpus).iter().any(Option::is_some) {
        return Err("empty resolved cache yielded photo slots".into());
    }
    if corpus.missing_expected() == 0 {
        return Err("empty resolved corpus passes the coverage gate".into());
    }
    let _ = std::fs::remove_dir_all(&base);
    Ok(())
}

/// An unreadable required JPEG inside an otherwise-complete corpus dir must
/// still count as missing coverage — the r3 reviewer repro. The gate cannot
/// certify collection while the load traversal drops errors.
#[cfg(unix)]
fn check_corpus_gate_counts_unreadable() -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let base = std::env::temp_dir().join(format!("perf-manifest-unread-{}", std::process::id()));
    let inner = codec_corpus::Corpus::with_cache_root(&base).map_err(|e| e.to_string())?;
    let root = base.join("codec-corpus/v1");
    for rel in CORPUS_JPEG_DIRS.iter().chain(CORPUS_PHOTO_DIRS) {
        std::fs::create_dir_all(root.join(rel)).map_err(|e| e.to_string())?;
    }
    std::fs::write(root.join(".version"), "1.1.0\n").map_err(|e| e.to_string())?;
    for rel in CORPUS_JPEG_DIRS {
        std::fs::write(
            root.join(rel).join("readable.jpg"),
            [0xff, 0xd8, 0xff, 0xd9],
        )
        .map_err(|e| e.to_string())?;
    }
    let bad = root.join(CORPUS_JPEG_DIRS[0]).join("unreadable.jpg");
    std::fs::write(&bad, [0xff, 0xd8, 0xff, 0xd9]).map_err(|e| e.to_string())?;
    std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o0))
        .map_err(|e| e.to_string())?;
    if std::fs::read(&bad).is_ok() {
        return Err("fixture failed to become unreadable".into());
    }
    let corpus = Corpus { inner: Some(inner) };
    let photos: Vec<Option<(Vec<u8>, usize, usize)>> =
        vec![Some((vec![0u8; 3], 1, 1)); CORPUS_PHOTO_DIRS.len()];
    let missing = corpus.missing_with(&photos);
    let _ = std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o600));
    let _ = std::fs::remove_dir_all(&base);
    if missing == 0 {
        return Err("unreadable JPEG dropped but corpus certified complete".into());
    }
    Ok(())
}

/// An unreadable subdirectory inside an expected dir hides its whole
/// subtree — that failure must also reach the gate.
#[cfg(unix)]
fn check_corpus_gate_counts_unreadable_subdir() -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let base = std::env::temp_dir().join(format!("perf-manifest-unrdir-{}", std::process::id()));
    let inner = codec_corpus::Corpus::with_cache_root(&base).map_err(|e| e.to_string())?;
    let root = base.join("codec-corpus/v1");
    for rel in CORPUS_JPEG_DIRS.iter().chain(CORPUS_PHOTO_DIRS) {
        std::fs::create_dir_all(root.join(rel)).map_err(|e| e.to_string())?;
    }
    std::fs::write(root.join(".version"), "1.1.0\n").map_err(|e| e.to_string())?;
    for rel in CORPUS_JPEG_DIRS {
        std::fs::write(
            root.join(rel).join("readable.jpg"),
            [0xff, 0xd8, 0xff, 0xd9],
        )
        .map_err(|e| e.to_string())?;
    }
    let bad = root.join(CORPUS_JPEG_DIRS[0]).join("sub");
    std::fs::create_dir_all(&bad).map_err(|e| e.to_string())?;
    std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o0))
        .map_err(|e| e.to_string())?;
    if std::fs::read_dir(&bad).is_ok() {
        return Err("fixture subdir failed to become unreadable".into());
    }
    let corpus = Corpus { inner: Some(inner) };
    let photos: Vec<Option<(Vec<u8>, usize, usize)>> =
        vec![Some((vec![0u8; 3], 1, 1)); CORPUS_PHOTO_DIRS.len()];
    let missing = corpus.missing_with(&photos);
    let _ = std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o700));
    let _ = std::fs::remove_dir_all(&base);
    if missing == 0 {
        return Err("unreadable subtree dropped but corpus certified complete".into());
    }
    Ok(())
}

/// A directory whose listing is readable but whose children's metadata is
/// denied (read-without-search, mode 0o400) must not silently hide a
/// subtree — the entry counts as a coverage failure.
#[cfg(unix)]
fn check_metadata_denied_is_failure() -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("perf-manifest-meta-{}", std::process::id()));
    let parent = root.join("metadata-parent");
    let child = parent.join("child");
    std::fs::create_dir_all(&child).map_err(|e| e.to_string())?;
    std::fs::write(root.join("readable.jpg"), [0xff, 0xd8, 0xff, 0xd9])
        .map_err(|e| e.to_string())?;
    std::fs::write(child.join("hidden.jpg"), [0xff, 0xd8, 0xff, 0xd9])
        .map_err(|e| e.to_string())?;
    let (all, failures) = scan_jpeg_dir("test", &root);
    if (all.len(), failures) != (2, 0) {
        let _ = std::fs::remove_dir_all(&root);
        return Err(format!(
            "fixture scan got ({}, {failures}), want (2, 0)",
            all.len()
        ));
    }
    std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o400))
        .map_err(|e| e.to_string())?;
    if std::fs::read_dir(&parent).is_err() || std::fs::metadata(&child).is_ok() {
        let _ = std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o700));
        let _ = std::fs::remove_dir_all(&root);
        return Err("fixture failed to produce listable-dir/denied-stat".into());
    }
    let (loaded, failures) = scan_jpeg_dir("test", &root);
    let _ = std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o700));
    let _ = std::fs::remove_dir_all(&root);
    if loaded.len() != 1 {
        return Err(format!(
            "metadata-denied subtree should drop out, got {}",
            loaded.len()
        ));
    }
    if failures == 0 {
        return Err("metadata error omitted a subtree but recorded zero coverage failures".into());
    }
    Ok(())
}

/// Every PixelLayout the encoder accepts must appear in an EncCase — a gap
/// means a layout-specific regression cannot alter the manifest.
fn check_layout_coverage() -> Result<(), String> {
    let all = [
        PixelLayout::Rgb8Srgb,
        PixelLayout::Bgr8Srgb,
        PixelLayout::Rgbx8Srgb,
        PixelLayout::Rgba8Srgb,
        PixelLayout::Bgrx8Srgb,
        PixelLayout::Bgra8Srgb,
        PixelLayout::Gray8Srgb,
        PixelLayout::Rgb16Linear,
        PixelLayout::Rgbx16Linear,
        PixelLayout::Rgba16Linear,
        PixelLayout::Gray16Linear,
        PixelLayout::RgbF32Linear,
        PixelLayout::RgbxF32Linear,
        PixelLayout::RgbaF32Linear,
        PixelLayout::GrayF32Linear,
        PixelLayout::YCbCr8,
        PixelLayout::YCbCrF32,
    ];
    let cases = enc_cases();
    let missing: Vec<_> = all
        .iter()
        .filter(|l| !cases.iter().any(|c| c.layout == **l))
        .collect();
    if !missing.is_empty() {
        return Err(format!("pixel layouts not exercised: {missing:?}"));
    }
    Ok(())
}

/// FullF32 rows must carry an explicit thread count: 1 for st, 0 (auto) for
/// mt — otherwise "st" f32 rows silently run the auto pool under parallel.
fn check_f32_thread_split() -> Result<(), String> {
    let paths = dec_paths();
    let get = |label: &str| paths.iter().find(|(l, _)| *l == label).map(|(_, k)| k);
    let (Some(st), ..) = (get("full-f32-srgb-st"),) else {
        return Err("full-f32-srgb-st decode path missing".into());
    };
    let DecKind::FullF32 { threads: st_t, .. } = st else {
        return Err("full-f32-srgb-st is not a FullF32 path".into());
    };
    if *st_t != 1 {
        return Err(format!("full-f32-srgb-st uses threads={st_t}, expected 1"));
    }
    if cfg!(feature = "parallel") {
        let Some(mt) = get("full-f32-srgb-mt") else {
            return Err("parallel build lacks full-f32-srgb-mt".into());
        };
        if mt == st {
            return Err("st/mt FullF32 rows select identical decoder configs".into());
        }
        let DecKind::FullF32 { threads: mt_t, .. } = mt else {
            return Err("full-f32-srgb-mt is not a FullF32 path".into());
        };
        if *mt_t != 0 {
            return Err(format!("full-f32-srgb-mt uses threads={mt_t}, expected 0"));
        }
    }
    Ok(())
}

/// Photo slots must remain stable: both Photo(0) and Photo(1) cases exist and
/// keep their indices even when a source is absent (the row becomes
/// SKIP:no-src at its own key instead of shifting a later photo up).
fn check_photo_slots() -> Result<(), String> {
    let cases = enc_cases();
    for i in [0usize, 1] {
        if !cases
            .iter()
            .any(|c| matches!(c.src, SrcKind::Photo(j) if j == i))
        {
            return Err(format!("no enc case for photo slot {i}"));
        }
    }
    Ok(())
}

/// Runs the whole suite; `cargo run --example perf_manifest -- self-test`.
/// The collection traversal, not the earlier preflight, decides the
/// emitted corpus state: coverage observed while loading the rows must
/// downgrade `Ok`, and the freshest count wins.
fn check_collected_state_is_authoritative() -> Result<(), String> {
    use CorpusState::{Disabled, Ok as Cok, Partial};
    let cases = [
        (Cok, 0, Cok),
        (Cok, 2, Partial(2)),
        (Partial(1), 3, Partial(3)),
        (Partial(9), 0, Cok),
        (Disabled, 0, Disabled),
        (Disabled, 5, Disabled),
    ];
    for (initial, missing, want) in cases {
        let got = collected_corpus_state(initial, missing);
        if got != want {
            return Err(format!(
                "collected_corpus_state({initial:?}, {missing}) = {got:?}, want {want:?}"
            ));
        }
    }
    Ok(())
}

fn self_test() -> Result<(), String> {
    check_parser_strict()?;
    check_compare_semantics()?;
    check_corpus_gate_counts_contents()?;
    #[cfg(unix)]
    {
        check_corpus_gate_counts_unreadable()?;
        check_corpus_gate_counts_unreadable_subdir()?;
        check_metadata_denied_is_failure()?;
    }
    check_layout_coverage()?;
    check_f32_thread_split()?;
    check_photo_slots()?;
    check_collected_state_is_authoritative()?;
    println!("self-test: OK");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn parser_strict() {
        super::check_parser_strict().unwrap()
    }
    #[test]
    fn compare_semantics() {
        super::check_compare_semantics().unwrap()
    }
    #[test]
    fn corpus_gate_counts_contents() {
        super::check_corpus_gate_counts_contents().unwrap()
    }
    #[cfg(unix)]
    #[test]
    fn corpus_gate_counts_unreadable() {
        super::check_corpus_gate_counts_unreadable().unwrap()
    }
    #[cfg(unix)]
    #[test]
    fn corpus_gate_counts_unreadable_subdir() {
        super::check_corpus_gate_counts_unreadable_subdir().unwrap()
    }
    #[cfg(unix)]
    #[test]
    fn metadata_denied_is_failure() {
        super::check_metadata_denied_is_failure().unwrap()
    }
    #[test]
    fn layout_coverage() {
        super::check_layout_coverage().unwrap()
    }
    #[test]
    fn f32_thread_split() {
        super::check_f32_thread_split().unwrap()
    }
    #[test]
    fn photo_slots() {
        super::check_photo_slots().unwrap()
    }
    #[test]
    fn collected_state_is_authoritative() {
        super::check_collected_state_is_authoritative().unwrap()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("self-test") => match self_test() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("self-test failed: {e}");
                1
            }
        },
        Some("emit") => {
            let out = args
                .iter()
                .position(|a| a == "--out")
                .and_then(|i| args.get(i + 1))
                .map(String::as_str);
            let allow_missing = args.iter().any(|a| a == "--allow-missing-corpus");
            match emit(out, allow_missing) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("emit failed: {e}");
                    2
                }
            }
        }
        Some("compare") => {
            let (Some(a), Some(b)) = (args.get(2), args.get(3)) else {
                eprintln!("usage: perf_manifest compare A.tsv B.tsv");
                std::process::exit(2);
            };
            match compare(a, b) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("compare failed: {e}");
                    2
                }
            }
        }
        _ => {
            eprintln!(
                "usage: perf_manifest emit [--out FILE] [--allow-missing-corpus] | compare A.tsv B.tsv"
            );
            2
        }
    };
    std::process::exit(code);
}
