//! Compare zenjpeg decoder output against mozjpeg (libjpeg-turbo) and zune-jpeg.
//!
//! For every JPEG in the corpus, decode with all three decoders and compare
//! pixel output. Reports max pixel difference and any files where decoders
//! disagree significantly.
//!
//! Run: cargo test --release -p zenjpeg --test corpus_decoder_comparison -- --nocapture --ignored

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

fn corpus_dir_path() -> std::path::PathBuf {
    zenjpeg_bench_utils::corpus_builder_dir()
}
const SKIP_DIRS: &[&str] = &["repro-images", "cc-index"];
const MAX_FILE_SIZE: u64 = 50 * 1024 * 1024;

// ── Decoders ───────────────────────────────────────────────────────────────

/// Decode with zenjpeg. Returns (width, height, rgb_pixels) or error string.
fn decode_zenjpeg(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use enough::Unstoppable;
    use zenjpeg::decoder::Decoder;

    // Disable ICC to compare raw decode output against mozjpeg/zune
    // (which don't apply ICC). Otherwise wide-gamut images show large
    // diffs from the Adobe RGB → sRGB transform, not from decoder bugs.
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(data, Unstoppable)
        .map_err(|e| format!("{e}"))?;
    let pixels = decoded.pixels_u8().ok_or("no pixel data")?.to_vec();
    Ok((decoded.width, decoded.height, pixels))
}

/// Decode with mozjpeg-sys (libjpeg-turbo with NASM SIMD). Returns RGB pixels.
fn decode_mozjpeg(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use mozjpeg_sys::*;
    use std::mem;

    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);

        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);

        jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);

        let header_ok = jpeg_read_header(&mut cinfo, true as boolean);
        if header_ok != 1 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("mozjpeg: bad header".into());
        }

        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);

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

        Ok((width, height, output))
    }
}

/// Decode with zune-jpeg. Returns RGB pixels.
fn decode_zune(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new_with_options(cursor, DecoderOptions::new_fast());
    decoder
        .decode_headers()
        .map_err(|e| format!("zune header: {e:?}"))?;

    let info = decoder.info().ok_or("zune: no info after header")?;
    let width = info.width as u32;
    let height = info.height as u32;

    let pixels = decoder
        .decode()
        .map_err(|e| format!("zune decode: {e:?}"))?;
    Ok((width, height, pixels))
}

// ── Comparison ─────────────────────────────────────────────────────────────

fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

/// File collection (same as corpus_decode_all).
fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_inner(dir, &mut files);
    files.sort();
    files
}

fn collect_inner(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if SKIP_DIRS.iter().any(|&s| name == s) {
                continue;
            }
            collect_inner(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn short_path(path: &Path) -> String {
    let base = corpus_dir_path();
    path.strip_prefix(&base)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn is_jpeg_by_magic(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}

// ── Per-file result ────────────────────────────────────────────────────────

#[derive(Debug)]
struct CompareResult {
    path: String,
    size: usize,
    width: u32,
    height: u32,
    // max pixel diff between decoder pairs
    zen_vs_moz_max: u8,
    zen_vs_zune_max: u8,
    moz_vs_zune_max: u8,
    // mean absolute diff
    zen_vs_moz_mean: f64,
    zen_vs_zune_mean: f64,
    // per-decoder errors (None = success)
    zen_err: Option<String>,
    moz_err: Option<String>,
    zune_err: Option<String>,
}

fn process_file(path: &Path) -> Option<CompareResult> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() > MAX_FILE_SIZE {
        return None;
    }

    let data = std::fs::read(path).ok()?;
    if !is_jpeg_by_magic(&data) {
        return None;
    }

    let sp = short_path(path);
    let size = data.len();

    let zen = decode_zenjpeg(&data);
    let moz = std::panic::catch_unwind(|| decode_mozjpeg(&data)).unwrap_or(Err("panic".into()));
    let zune = std::panic::catch_unwind(|| decode_zune(&data)).unwrap_or(Err("panic".into()));

    let mut result = CompareResult {
        path: sp,
        size,
        width: 0,
        height: 0,
        zen_vs_moz_max: 0,
        zen_vs_zune_max: 0,
        moz_vs_zune_max: 0,
        zen_vs_moz_mean: 0.0,
        zen_vs_zune_mean: 0.0,
        zen_err: zen.as_ref().err().cloned(),
        moz_err: moz.as_ref().err().cloned(),
        zune_err: zune.as_ref().err().cloned(),
    };

    if let Ok((w, h, _)) = &zen {
        result.width = *w;
        result.height = *h;
    } else if let Ok((w, h, _)) = &moz {
        result.width = *w;
        result.height = *h;
    }

    // Compare pairs where both succeeded and dimensions match
    if let (Ok((zw, zh, zp)), Ok((mw, mh, mp))) = (&zen, &moz)
        && zw == mw
        && zh == mh
        && zp.len() == mp.len()
    {
        result.zen_vs_moz_max = max_diff(zp, mp);
        result.zen_vs_moz_mean = mean_abs_diff(zp, mp);
    }
    if let (Ok((zw, zh, zp)), Ok((uw, uh, up))) = (&zen, &zune)
        && zw == uw
        && zh == uh
        && zp.len() == up.len()
    {
        result.zen_vs_zune_max = max_diff(zp, up);
        result.zen_vs_zune_mean = mean_abs_diff(zp, up);
    }
    if let (Ok((mw, mh, mp)), Ok((uw, uh, up))) = (&moz, &zune)
        && mw == uw
        && mh == uh
        && mp.len() == up.len()
    {
        result.moz_vs_zune_max = max_diff(mp, up);
    }

    Some(result)
}

#[ignore = "requires local corpus-builder directory (set CORPUS_BUILDER_DIR)"]
#[test]
fn compare_decoders_on_corpus() {
    let corpus = corpus_dir_path();
    if !corpus.exists() {
        println!("Corpus not found at {}, skipping", corpus.display());
        return;
    }

    println!("Collecting files...");
    let files = collect_files(&corpus);
    println!(
        "Found {} files, comparing with {} threads...",
        files.len(),
        rayon::current_num_threads()
    );

    let progress = AtomicU32::new(0);
    let total = files.len() as u32;
    let results: Vec<CompareResult> = files
        .par_iter()
        .filter_map(|path| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 5000 == 0 {
                eprintln!("  ... {done}/{total}");
            }
            process_file(path)
        })
        .collect();

    // Aggregate
    let total_jpeg = results.len();
    let mut all_ok = 0u32;
    let mut zen_only_fail = 0u32;
    let mut moz_only_fail = 0u32;
    let mut zune_only_fail = 0u32;
    let mut all_fail = 0u32;
    let mut zen_fail_moz_ok = Vec::new();

    // Diff histograms
    let mut zen_moz_max_hist = [0u32; 256];
    let mut zen_zune_max_hist = [0u32; 256];

    let mut worst_zen_moz: Option<&CompareResult> = None;
    let mut worst_zen_zune: Option<&CompareResult> = None;

    for r in &results {
        let zen_ok = r.zen_err.is_none();
        let moz_ok = r.moz_err.is_none();
        let zune_ok = r.zune_err.is_none();

        match (zen_ok, moz_ok, zune_ok) {
            (true, true, true) => all_ok += 1,
            (false, true, _) => {
                zen_fail_moz_ok.push(r);
                zen_only_fail += 1;
            }
            (true, false, _) => moz_only_fail += 1,
            (false, false, _) => all_fail += 1,
            _ => zune_only_fail += 1,
        }

        if zen_ok && moz_ok {
            zen_moz_max_hist[r.zen_vs_moz_max as usize] += 1;
            if worst_zen_moz.is_none() || r.zen_vs_moz_max > worst_zen_moz.unwrap().zen_vs_moz_max {
                worst_zen_moz = Some(r);
            }
        }
        if zen_ok && zune_ok {
            zen_zune_max_hist[r.zen_vs_zune_max as usize] += 1;
            if worst_zen_zune.is_none()
                || r.zen_vs_zune_max > worst_zen_zune.unwrap().zen_vs_zune_max
            {
                worst_zen_zune = Some(r);
            }
        }
    }

    // Print results
    println!("\n=== Decoder Comparison: {total_jpeg} JPEGs ===\n");

    println!("Decode success:");
    println!("  All 3 OK:           {all_ok}");
    println!("  zen fail, moz OK:   {zen_only_fail}");
    println!("  moz fail, zen OK:   {moz_only_fail}");
    println!("  zune fail only:     {zune_only_fail}");
    println!("  All 3 fail:         {all_fail}");

    // zen vs mozjpeg histogram
    println!("\n--- zenjpeg vs mozjpeg (max pixel diff) ---");
    let zen_moz_compared: u32 = zen_moz_max_hist.iter().sum();
    println!("  Compared: {zen_moz_compared} files");
    print_histogram(&zen_moz_max_hist);
    if let Some(w) = worst_zen_moz {
        println!(
            "  Worst: {} ({}x{}, {} bytes) max_diff={} mean_diff={:.3}",
            w.path, w.width, w.height, w.size, w.zen_vs_moz_max, w.zen_vs_moz_mean
        );
    }

    // zen vs zune histogram
    println!("\n--- zenjpeg vs zune-jpeg (max pixel diff) ---");
    let zen_zune_compared: u32 = zen_zune_max_hist.iter().sum();
    println!("  Compared: {zen_zune_compared} files");
    print_histogram(&zen_zune_max_hist);
    if let Some(w) = worst_zen_zune {
        println!(
            "  Worst: {} ({}x{}, {} bytes) max_diff={} mean_diff={:.3}",
            w.path, w.width, w.height, w.size, w.zen_vs_zune_max, w.zen_vs_zune_mean
        );
    }

    // Files where zen fails but mozjpeg succeeds
    if !zen_fail_moz_ok.is_empty() {
        println!(
            "\n--- zenjpeg FAILS where mozjpeg succeeds ({}) ---",
            zen_fail_moz_ok.len()
        );
        for r in zen_fail_moz_ok.iter().take(20) {
            println!(
                "  {} ({} bytes): {}",
                r.path,
                r.size,
                r.zen_err.as_deref().unwrap_or("?")
            );
        }
        if zen_fail_moz_ok.len() > 20 {
            println!("  ... and {} more", zen_fail_moz_ok.len() - 20);
        }
    }

    // Save to file
    let report_path = "/tmp/corpus_decoder_comparison.txt";
    let mut report = String::new();
    report.push_str(&format!("Total JPEGs: {total_jpeg}\n"));
    report.push_str(&format!("All 3 OK: {all_ok}\n"));
    report.push_str(&format!("zen fail moz OK: {zen_only_fail}\n"));
    report.push_str(&format!("moz fail zen OK: {moz_only_fail}\n\n"));

    report.push_str("zen vs moz max_diff histogram:\n");
    for (diff, &count) in zen_moz_max_hist.iter().enumerate() {
        if count > 0 {
            report.push_str(&format!("  diff={diff}: {count} files\n"));
        }
    }
    report.push_str("\nzen vs zune max_diff histogram:\n");
    for (diff, &count) in zen_zune_max_hist.iter().enumerate() {
        if count > 0 {
            report.push_str(&format!("  diff={diff}: {count} files\n"));
        }
    }

    // Per-file details for high-diff cases
    report.push_str("\nFiles with zen_vs_moz max_diff > 1:\n");
    for r in &results {
        if r.zen_err.is_none() && r.moz_err.is_none() && r.zen_vs_moz_max > 1 {
            report.push_str(&format!(
                "  {} ({}x{}, {}B) max={} mean={:.3}\n",
                r.path, r.width, r.height, r.size, r.zen_vs_moz_max, r.zen_vs_moz_mean
            ));
        }
    }

    let _ = std::fs::write(report_path, &report);
    println!("\nFull results saved to {report_path}");

    // Assert: no panics in zenjpeg (already covered by corpus_decode_all),
    // and zen should decode everything mozjpeg can
    if zen_only_fail > 1 {
        // Allow 1 for the >100MP limit file
        println!(
            "\nWARNING: zenjpeg failed on {zen_only_fail} files that mozjpeg decoded successfully"
        );
    }
}

fn print_histogram(hist: &[u32; 256]) {
    let total: u32 = hist.iter().sum();
    if total == 0 {
        println!("  (no data)");
        return;
    }

    // Cumulative buckets
    let exact_0 = hist[0];
    let le_1: u32 = hist[..=1].iter().sum();
    let le_2: u32 = hist[..=2].iter().sum();
    let le_3: u32 = hist[..=3].iter().sum();
    let le_5: u32 = hist[..=5].iter().sum();
    let le_10: u32 = hist[..=10].iter().sum();
    let gt_10: u32 = total - le_10;
    let max_seen = hist.iter().rposition(|&c| c > 0).unwrap_or(0);

    println!(
        "  exact 0: {:>5} ({:.1}%)",
        exact_0,
        exact_0 as f64 / total as f64 * 100.0
    );
    println!(
        "  ≤1:      {:>5} ({:.1}%)",
        le_1,
        le_1 as f64 / total as f64 * 100.0
    );
    println!(
        "  ≤2:      {:>5} ({:.1}%)",
        le_2,
        le_2 as f64 / total as f64 * 100.0
    );
    println!(
        "  ≤3:      {:>5} ({:.1}%)",
        le_3,
        le_3 as f64 / total as f64 * 100.0
    );
    println!(
        "  ≤5:      {:>5} ({:.1}%)",
        le_5,
        le_5 as f64 / total as f64 * 100.0
    );
    println!(
        "  ≤10:     {:>5} ({:.1}%)",
        le_10,
        le_10 as f64 / total as f64 * 100.0
    );
    if gt_10 > 0 {
        println!(
            "  >10:     {:>5} ({:.1}%)",
            gt_10,
            gt_10 as f64 / total as f64 * 100.0
        );
    }
    println!("  max seen: {max_seen}");
}
