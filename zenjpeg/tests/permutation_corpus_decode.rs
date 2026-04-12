//! Permutation corpus decode validator.
//!
//! Reads the manifest produced by `gen_permutation_corpus`, decodes each JPEG
//! with zenjpeg and mozjpeg (libjpeg-turbo), records a per-file row with:
//!
//!   - both decoders' status (ok, error, bad dims)
//!   - byte-identical flag
//!   - max pixel diff, mean abs diff, diff histogram buckets
//!   - zensim regression score + category (only when pixels differ)
//!
//! Results are written to a detailed TSV so later stages can shrink the corpus
//! using any criteria (worst-max-diff, worst-category, expected failures, etc).
//!
//! Run (after generating corpus):
//!   cargo test --release -p zenjpeg --features decoder \
//!     --test permutation_corpus_decode \
//!     -- --nocapture --ignored
//!
//! Env:
//!   ZENJPEG_PERM_OUT    corpus root (same as generator)
//!   ZENJPEG_PERM_LIMIT  max files to process (for smoke runs)
//!   ZENJPEG_PERM_REPORT output TSV path (default: <out>/validation.tsv)

#![cfg(all(feature = "decoder", not(target_arch = "wasm32")))]

use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const DEFAULT_OUT: &str = "/home/lilith/work/zen/zenjpeg-perm-corpus";

// ── Decoders (copied pattern from corpus_decoder_comparison.rs) ────────────

#[derive(Debug)]
struct Decoded {
    width: u32,
    height: u32,
    channels: u8, // 1 = gray, 3 = RGB
    pixels: Vec<u8>,
}

fn decode_zenjpeg(data: &[u8]) -> Result<Decoded, String> {
    use enough::Unstoppable;
    use zenjpeg::decoder::Decoder;

    let decoder = Decoder::new();
    let img = decoder
        .decode(data, Unstoppable)
        .map_err(|e| format!("{e}"))?;
    let w = img.width as usize;
    let h = img.height as usize;
    let raw = img.pixels_u8().ok_or("no pixel data")?.to_vec();
    // Normalize to RGB u8 (3 channels) so grayscale and color compare
    // apples-to-apples against mozjpeg's forced RGB output.
    let pixels = if raw.len() == w * h * 3 {
        raw
    } else if raw.len() == w * h {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for &v in &raw {
            rgb.push(v);
            rgb.push(v);
            rgb.push(v);
        }
        rgb
    } else {
        return Err(format!("unexpected pixel length {} for {}x{}", raw.len(), w, h));
    };
    Ok(Decoded {
        width: w as u32,
        height: h as u32,
        channels: 3,
        pixels,
    })
}

fn decode_mozjpeg(data: &[u8]) -> Result<Decoded, String> {
    use mozjpeg_sys::*;
    use std::mem;

    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        // Install an error handler that does NOT longjmp; mozjpeg's default
        // exit_on_error will call exit() on fatal. We want a graceful return.
        // mozjpeg-sys handles the setjmp machinery inside jpeg_create_decompress
        // with its own hook; errors come back via returning zero from header.

        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);

        jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);

        let header_ok = jpeg_read_header(&mut cinfo, true as boolean);
        if header_ok != 1 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("mozjpeg: bad header".into());
        }

        // Always request RGB output so grayscale JPEGs come out as replicated
        // RGB, matching the normalized zenjpeg output. This lets grayscale and
        // color files share the same diff pipeline.
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;

        if jpeg_start_decompress(&mut cinfo) == 0 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("mozjpeg: start_decompress failed".into());
        }

        let width = cinfo.output_width;
        let height = cinfo.output_height;
        let components = cinfo.output_components as usize;
        let row_stride = width as usize * components;
        let mut output = vec![0u8; height as usize * row_stride];
        while cinfo.output_scanline < height {
            let offset = cinfo.output_scanline as usize * row_stride;
            let mut row_ptr = output[offset..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);

        Ok(Decoded {
            width,
            height,
            channels: components as u8,
            pixels: output,
        })
    }
}

// ── Diff metrics ───────────────────────────────────────────────────────────

#[derive(Default)]
struct DiffStats {
    max_diff: u8,
    mean_abs: f64,
    // Bucket counts: diff == 0, 1, 2, 3, 4, 5, 6..10, 11..20, 21..50, >50
    b_0: u64,
    b_1: u64,
    b_2: u64,
    b_3: u64,
    b_4: u64,
    b_5: u64,
    b_6_10: u64,
    b_11_20: u64,
    b_21_50: u64,
    b_gt_50: u64,
}

fn compute_diff_stats(a: &[u8], b: &[u8]) -> DiffStats {
    assert_eq!(a.len(), b.len());
    let mut s = DiffStats::default();
    let mut sum: u64 = 0;
    let mut max: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i16 - *y as i16).unsigned_abs() as u8;
        sum += d as u64;
        if d > max {
            max = d;
        }
        match d {
            0 => s.b_0 += 1,
            1 => s.b_1 += 1,
            2 => s.b_2 += 1,
            3 => s.b_3 += 1,
            4 => s.b_4 += 1,
            5 => s.b_5 += 1,
            6..=10 => s.b_6_10 += 1,
            11..=20 => s.b_11_20 += 1,
            21..=50 => s.b_21_50 += 1,
            _ => s.b_gt_50 += 1,
        }
    }
    s.max_diff = max;
    s.mean_abs = sum as f64 / a.len().max(1) as f64;
    s
}

// ── Zensim regression (only when pixels differ) ────────────────────────────

fn zensim_regress(a: &Decoded, b: &Decoded) -> Option<(f64, String)> {
    // Requires RGB input ≥ 8×8.
    if a.width < 8 || a.height < 8 {
        return None;
    }
    if a.channels != 3 || b.channels != 3 {
        return None;
    }
    if (a.width, a.height) != (b.width, b.height) {
        return None;
    }
    use zensim::{PixelFormat, StridedBytes, Zensim, ZensimProfile};
    use zensim_regress::{RegressionTolerance, check_regression};

    let zsim = Zensim::new(ZensimProfile::latest());
    let w = a.width as usize;
    let h = a.height as usize;
    let stride = w * 3;
    let expected =
        StridedBytes::try_new(&a.pixels, w, h, stride, PixelFormat::Srgb8Rgb).ok()?;
    let actual = StridedBytes::try_new(&b.pixels, w, h, stride, PixelFormat::Srgb8Rgb).ok()?;
    let tol = RegressionTolerance::off_by_one();
    let report = check_regression(&zsim, &expected, &actual, &tol).ok()?;
    Some((report.score(), format!("{:?}", report.category())))
}

// ── SOFn marker scan ───────────────────────────────────────────────────────
//
// libjpeg-turbo (mozjpeg-sys) calls exit() on arithmetic-coded JPEGs because
// it has no built-in handler and its default error_exit is fatal. We detect
// those (and any non-standard SOFn) before calling mozjpeg so the whole test
// process doesn't die on the first arithmetic file.

/// Scan a JPEG header for the SOFn marker byte (the lower byte after 0xFF).
/// Returns None if SOI is missing, the marker wasn't found before SOS, or
/// segment lengths don't parse.
fn jpeg_sof_marker(data: &[u8]) -> Option<u8> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 3 < data.len() {
        if data[i] != 0xFF {
            return None;
        }
        // Skip fill bytes (0xFF 0xFF ...)
        while i + 1 < data.len() && data[i + 1] == 0xFF {
            i += 1;
        }
        if i + 1 >= data.len() {
            return None;
        }
        let marker = data[i + 1];
        // SOFn: 0xC0..=0xCF except 0xC4 (DHT), 0xC8 (JPG reserved), 0xCC (DAC)
        if (0xC0..=0xCF).contains(&marker)
            && marker != 0xC4
            && marker != 0xC8
            && marker != 0xCC
        {
            return Some(marker);
        }
        // Reached entropy-coded data or end without SOF.
        if marker == 0xDA || marker == 0xD9 {
            return None;
        }
        // Standalone markers (no length payload): 0xD0..=0xD8.
        if (0xD0..=0xD8).contains(&marker) {
            i += 2;
            continue;
        }
        if i + 3 >= data.len() {
            return None;
        }
        let seg_len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        if seg_len < 2 {
            return None;
        }
        i += 2 + seg_len;
    }
    None
}

/// True if the JPEG uses arithmetic coding (SOF9/SOFA/SOFB).
fn is_arithmetic_jpeg(data: &[u8]) -> bool {
    matches!(jpeg_sof_marker(data), Some(0xC9) | Some(0xCA) | Some(0xCB))
}

// ── Manifest parsing ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ManifestRow {
    hash: String,
    rel_path: String,
    tool: String,
    source: String,
    params: String,
    bytes: u64,
    expect_zenjpeg_fail: bool,
}

fn read_manifest(manifest_path: &Path) -> Result<Vec<ManifestRow>, String> {
    let f = File::open(manifest_path).map_err(|e| format!("open manifest: {e}"))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    let mut lines = reader.lines();
    // Skip header
    if let Some(Ok(hdr)) = lines.next() {
        if !hdr.starts_with("hash\t") {
            return Err(format!("unexpected header: {hdr}"));
        }
    }
    for line in lines.flatten() {
        let mut it = line.splitn(7, '\t');
        let hash = it.next().unwrap_or("").to_string();
        let rel_path = it.next().unwrap_or("").to_string();
        let tool = it.next().unwrap_or("").to_string();
        let source = it.next().unwrap_or("").to_string();
        let params = it.next().unwrap_or("").to_string();
        let bytes: u64 = it.next().unwrap_or("0").parse().unwrap_or(0);
        let expect_zenjpeg_fail = it.next().unwrap_or("0") == "1";
        if hash.is_empty() {
            continue;
        }
        out.push(ManifestRow {
            hash,
            rel_path,
            tool,
            source,
            params,
            bytes,
            expect_zenjpeg_fail,
        });
    }
    Ok(out)
}

// ── Per-file result ────────────────────────────────────────────────────────

struct FileResult {
    row: ManifestRow,
    zen_ok: bool,
    zen_err: String,
    moz_ok: bool,
    moz_err: String,
    dim_match: bool,
    byte_identical: bool,
    diff: Option<DiffStats>,
    zensim_score: Option<f64>,
    zensim_category: Option<String>,
    // "exact", "byte_equal", "diff_small", "diff_large",
    // "zen_err_moz_ok", "moz_err_zen_ok", "both_err", "dim_mismatch"
    status: &'static str,
}

fn process_file(row: ManifestRow, corpus_root: &Path) -> FileResult {
    let path = corpus_root.join(&row.rel_path);
    let data = match fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            return FileResult {
                row,
                zen_ok: false,
                zen_err: format!("read: {e}"),
                moz_ok: false,
                moz_err: format!("read: {e}"),
                dim_match: false,
                byte_identical: false,
                diff: None,
                zensim_score: None,
                zensim_category: None,
                status: "read_err",
            };
        }
    };

    // mozjpeg-sys aborts the process on arithmetic-coded files (libjpeg-turbo
    // has no implementation and its default error_exit calls exit()). Detect
    // those via SOFn scan and skip both decoders. Zenjpeg also rejects them
    // and the mismatch is recorded in the status column.
    if is_arithmetic_jpeg(&data) {
        return FileResult {
            row,
            zen_ok: false,
            zen_err: "arithmetic coding (not supported)".into(),
            moz_ok: false,
            moz_err: "arithmetic coding (not supported)".into(),
            dim_match: false,
            byte_identical: false,
            diff: None,
            zensim_score: None,
            zensim_category: None,
            status: "arith_skipped",
        };
    }

    // XYB files: mozjpeg has no XYB awareness and decodes the coefficients as
    // YCbCr, producing wrong colors. Comparing those bytes against zenjpeg's
    // XYB→sRGB conversion is meaningless. Record zen status only.
    let is_xyb = row.params.contains("xyb");
    if is_xyb {
        let zen = decode_zenjpeg(&data);
        return match zen {
            Ok(_) => FileResult {
                row,
                zen_ok: true,
                zen_err: String::new(),
                moz_ok: false,
                moz_err: "skipped (xyb)".into(),
                dim_match: false,
                byte_identical: false,
                diff: None,
                zensim_score: None,
                zensim_category: None,
                status: "xyb_zen_ok",
            },
            Err(e) => FileResult {
                row,
                zen_ok: false,
                zen_err: e,
                moz_ok: false,
                moz_err: "skipped (xyb)".into(),
                dim_match: false,
                byte_identical: false,
                diff: None,
                zensim_score: None,
                zensim_category: None,
                status: "xyb_zen_err",
            },
        };
    }

    let zen = decode_zenjpeg(&data);
    let moz = decode_mozjpeg(&data);

    match (zen, moz) {
        (Ok(z), Ok(m)) => {
            let dim_match = z.width == m.width && z.height == m.height && z.channels == m.channels;
            if !dim_match {
                return FileResult {
                    row,
                    zen_ok: true,
                    zen_err: String::new(),
                    moz_ok: true,
                    moz_err: String::new(),
                    dim_match: false,
                    byte_identical: false,
                    diff: None,
                    zensim_score: None,
                    zensim_category: None,
                    status: "dim_mismatch",
                };
            }
            // Fast path: byte-compare pixels first.
            let byte_eq = z.pixels == m.pixels;
            if byte_eq {
                return FileResult {
                    row,
                    zen_ok: true,
                    zen_err: String::new(),
                    moz_ok: true,
                    moz_err: String::new(),
                    dim_match: true,
                    byte_identical: true,
                    diff: None,
                    zensim_score: None,
                    zensim_category: None,
                    status: "byte_equal",
                };
            }
            // Slow path: per-pixel stats + zensim (RGB only).
            let stats = compute_diff_stats(&z.pixels, &m.pixels);
            let (score, cat) = zensim_regress(&z, &m)
                .map(|(s, c)| (Some(s), Some(c)))
                .unwrap_or((None, None));
            let status = if stats.max_diff <= 2 {
                "diff_small"
            } else if stats.max_diff <= 8 {
                "diff_moderate"
            } else {
                "diff_large"
            };
            FileResult {
                row,
                zen_ok: true,
                zen_err: String::new(),
                moz_ok: true,
                moz_err: String::new(),
                dim_match: true,
                byte_identical: false,
                diff: Some(stats),
                zensim_score: score,
                zensim_category: cat,
                status,
            }
        }
        (Err(e), Ok(_)) => FileResult {
            row,
            zen_ok: false,
            zen_err: e,
            moz_ok: true,
            moz_err: String::new(),
            dim_match: false,
            byte_identical: false,
            diff: None,
            zensim_score: None,
            zensim_category: None,
            status: "zen_err_moz_ok",
        },
        (Ok(_), Err(e)) => FileResult {
            row,
            zen_ok: true,
            zen_err: String::new(),
            moz_ok: false,
            moz_err: e,
            dim_match: false,
            byte_identical: false,
            diff: None,
            zensim_score: None,
            zensim_category: None,
            status: "moz_err_zen_ok",
        },
        (Err(ze), Err(me)) => FileResult {
            row,
            zen_ok: false,
            zen_err: ze,
            moz_ok: false,
            moz_err: me,
            dim_match: false,
            byte_identical: false,
            diff: None,
            zensim_score: None,
            zensim_category: None,
            status: "both_err",
        },
    }
}

fn write_result_row(w: &mut BufWriter<File>, r: &FileResult) -> std::io::Result<()> {
    let (max_diff, mean_abs, b0, b1, b2, b3, b4, b5, b6_10, b11_20, b21_50, b_gt50) =
        if let Some(d) = &r.diff {
            (
                d.max_diff, d.mean_abs, d.b_0, d.b_1, d.b_2, d.b_3, d.b_4, d.b_5, d.b_6_10,
                d.b_11_20, d.b_21_50, d.b_gt_50,
            )
        } else {
            (0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        };
    let zen_err = r.zen_err.replace('\t', " ").replace('\n', " ");
    let moz_err = r.moz_err.replace('\t', " ").replace('\n', " ");
    let score = r.zensim_score.map(|s| format!("{s:.3}")).unwrap_or_default();
    let cat = r.zensim_category.clone().unwrap_or_default();
    writeln!(
        w,
        "{hash}\t{rel}\t{tool}\t{src}\t{params}\t{bytes}\t{expect}\t\
         {status}\t{zen_ok}\t{moz_ok}\t{dim_match}\t{byte_eq}\t\
         {max_diff}\t{mean_abs:.4}\t\
         {b0}\t{b1}\t{b2}\t{b3}\t{b4}\t{b5}\t{b6_10}\t{b11_20}\t{b21_50}\t{b_gt50}\t\
         {score}\t{cat}\t{zen_err}\t{moz_err}",
        hash = r.row.hash,
        rel = r.row.rel_path,
        tool = r.row.tool,
        src = r.row.source,
        params = r.row.params,
        bytes = r.row.bytes,
        expect = r.row.expect_zenjpeg_fail as u8,
        status = r.status,
        zen_ok = r.zen_ok as u8,
        moz_ok = r.moz_ok as u8,
        dim_match = r.dim_match as u8,
        byte_eq = r.byte_identical as u8,
    )
}

fn write_header(w: &mut BufWriter<File>) -> std::io::Result<()> {
    writeln!(
        w,
        "hash\trel_path\ttool\tsource\tparams\tbytes\texpect_zenjpeg_fail\t\
         status\tzen_ok\tmoz_ok\tdim_match\tbyte_equal\t\
         max_diff\tmean_abs\t\
         b_0\tb_1\tb_2\tb_3\tb_4\tb_5\tb_6_10\tb_11_20\tb_21_50\tb_gt_50\t\
         zensim_score\tzensim_category\tzen_err\tmoz_err"
    )
}

// ── Test entry point ───────────────────────────────────────────────────────

#[test]
#[ignore = "requires pre-generated corpus via `cargo run --release --example gen_permutation_corpus`"]
fn validate_permutation_corpus() {
    let out_dir = PathBuf::from(
        std::env::var("ZENJPEG_PERM_OUT").unwrap_or_else(|_| DEFAULT_OUT.into()),
    );
    let manifest_path = out_dir.join("manifest.tsv");
    if !manifest_path.exists() {
        panic!(
            "manifest not found at {} — run the generator first:\n  \
             cargo run --release --example gen_permutation_corpus",
            manifest_path.display()
        );
    }

    let mut rows = read_manifest(&manifest_path).expect("read manifest");
    if let Ok(limit_str) = std::env::var("ZENJPEG_PERM_LIMIT") {
        if let Ok(n) = limit_str.parse::<usize>() {
            rows.truncate(n);
        }
    }

    let report_path = std::env::var("ZENJPEG_PERM_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| out_dir.join("validation.tsv"));

    println!("=== zenjpeg permutation corpus validator ===");
    println!("corpus:    {}", out_dir.display());
    println!("manifest:  {} rows", rows.len());
    println!("report:    {}", report_path.display());

    let writer = Mutex::new(BufWriter::with_capacity(
        64 * 1024,
        File::create(&report_path).expect("create report"),
    ));
    write_header(&mut writer.lock().unwrap()).expect("write header");

    let n = rows.len();
    let done = AtomicU64::new(0);
    let n_byte_eq = AtomicU64::new(0);
    let n_small = AtomicU64::new(0);
    let n_mod = AtomicU64::new(0);
    let n_large = AtomicU64::new(0);
    let n_zen_err = AtomicU64::new(0);
    let n_moz_err = AtomicU64::new(0);
    let n_both_err = AtomicU64::new(0);
    let n_dim_mismatch = AtomicU64::new(0);
    let n_arith = AtomicU64::new(0);
    let n_xyb_ok = AtomicU64::new(0);
    let n_xyb_err = AtomicU64::new(0);
    let max_diff_seen = AtomicU64::new(0);

    let t0 = Instant::now();
    rows.into_par_iter().for_each(|row| {
        let result = process_file(row, &out_dir);
        match result.status {
            "byte_equal" => {
                n_byte_eq.fetch_add(1, Ordering::Relaxed);
            }
            "diff_small" => {
                n_small.fetch_add(1, Ordering::Relaxed);
            }
            "diff_moderate" => {
                n_mod.fetch_add(1, Ordering::Relaxed);
            }
            "diff_large" => {
                n_large.fetch_add(1, Ordering::Relaxed);
            }
            "zen_err_moz_ok" => {
                n_zen_err.fetch_add(1, Ordering::Relaxed);
            }
            "moz_err_zen_ok" => {
                n_moz_err.fetch_add(1, Ordering::Relaxed);
            }
            "both_err" => {
                n_both_err.fetch_add(1, Ordering::Relaxed);
            }
            "dim_mismatch" => {
                n_dim_mismatch.fetch_add(1, Ordering::Relaxed);
            }
            "arith_skipped" => {
                n_arith.fetch_add(1, Ordering::Relaxed);
            }
            "xyb_zen_ok" => {
                n_xyb_ok.fetch_add(1, Ordering::Relaxed);
            }
            "xyb_zen_err" => {
                n_xyb_err.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        if let Some(d) = &result.diff {
            let cur = max_diff_seen.load(Ordering::Relaxed);
            if d.max_diff as u64 > cur {
                max_diff_seen.fetch_max(d.max_diff as u64, Ordering::Relaxed);
            }
        }
        {
            let mut w = writer.lock().unwrap();
            let _ = write_result_row(&mut w, &result);
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 2000 == 0 {
            println!(
                "  processed {d}/{n}  ({:.0}/s)  max_diff_so_far={}",
                d as f64 / t0.elapsed().as_secs_f64(),
                max_diff_seen.load(Ordering::Relaxed),
            );
        }
    });

    writer.lock().unwrap().flush().expect("flush report");
    let elapsed = t0.elapsed();

    println!();
    println!("=== validation summary ===");
    println!("elapsed:        {:.1}s", elapsed.as_secs_f64());
    println!("processed:      {n}");
    println!("byte_equal:     {}", n_byte_eq.load(Ordering::Relaxed));
    println!("diff_small ≤2:  {}", n_small.load(Ordering::Relaxed));
    println!("diff_mod 3-8:   {}", n_mod.load(Ordering::Relaxed));
    println!("diff_large >8:  {}", n_large.load(Ordering::Relaxed));
    println!("zen_err+moz_ok: {}", n_zen_err.load(Ordering::Relaxed));
    println!("moz_err+zen_ok: {}", n_moz_err.load(Ordering::Relaxed));
    println!("both_err:       {}", n_both_err.load(Ordering::Relaxed));
    println!("dim_mismatch:   {}", n_dim_mismatch.load(Ordering::Relaxed));
    println!("arith_skipped:  {}", n_arith.load(Ordering::Relaxed));
    println!("xyb_zen_ok:     {}", n_xyb_ok.load(Ordering::Relaxed));
    println!("xyb_zen_err:    {}", n_xyb_err.load(Ordering::Relaxed));
    println!("max_diff seen:  {}", max_diff_seen.load(Ordering::Relaxed));
    println!("report:         {}", report_path.display());

    // Non-assertive test: this is an investigation tool. It records what
    // happened; it does NOT fail the build. Analysis happens on the TSV.
}
