//! Corpus-level decode accuracy comparison.
//!
//! Decodes the JPEG conformance suite with multiple decoders and compares
//! pixel-level output to measure decode accuracy.
//!
//! Decoders tested:
//! - zenjpeg (our decoder)
//! - zune-jpeg (fast pure Rust, integer IDCT)
//! - jpeg-decoder (pure Rust reference)
//! - libjpeg-turbo via djpeg CLI (C reference)
//!
//! Run with: cargo test --release --features decoder -p zenjpeg --test decode_accuracy_corpus -- --nocapture --ignored

use dssim_core::Dssim;
use enough::Unstoppable;
use rgb::RGBA8;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus() -> Option<codec_corpus::Corpus> {
    codec_corpus::Corpus::new().ok()
}

fn collect_jpgs(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "jpg" || ext == "jpeg" {
                        files.push(path);
                    }
                }
            }
        }
    }
    files.sort();
    files
}

struct DecodeResult {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    channels: usize, // 1 = gray, 3 = RGB
}

fn decode_zenjpeg(data: &[u8]) -> Option<DecodeResult> {
    use zenjpeg::decoder::Decoder;
    let img = Decoder::new().decode(data, Unstoppable).ok()?;
    let w = img.width as usize;
    let h = img.height as usize;
    let pixels = img.into_pixels_u8().unwrap();
    let channels = if pixels.len() == (w * h) { 1 } else { 3 };
    Some(DecodeResult {
        pixels,
        width: w,
        height: h,
        channels,
    })
}

fn decode_zenjpeg_libjpeg_compat(data: &[u8]) -> Option<DecodeResult> {
    use zenjpeg::decode::ChromaUpsampling;
    use zenjpeg::decoder::Decoder;
    let img = Decoder::new()
        .chroma_upsampling(ChromaUpsampling::LibjpegCompat)
        .decode(data, Unstoppable)
        .ok()?;
    let w = img.width as usize;
    let h = img.height as usize;
    let pixels = img.into_pixels_u8().unwrap();
    let channels = if pixels.len() == (w * h) { 1 } else { 3 };
    Some(DecodeResult {
        pixels,
        width: w,
        height: h,
        channels,
    })
}

fn decode_zune(data: &[u8]) -> Option<DecodeResult> {
    use zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let mut decoder = JpegDecoder::new(ZCursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let channels = info.components as usize;
    // zune-jpeg may output 1 or 3 channels
    Some(DecodeResult {
        pixels,
        width: info.width as usize,
        height: info.height as usize,
        channels,
    })
}

fn decode_jpeg_decoder_crate(data: &[u8]) -> Option<DecodeResult> {
    use jpeg_decoder::Decoder;
    let mut decoder = Decoder::new(data);
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let channels = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 | jpeg_decoder::PixelFormat::L16 => 1,
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::CMYK32 => 4,
    };
    Some(DecodeResult {
        pixels,
        width: info.width as usize,
        height: info.height as usize,
        channels,
    })
}

/// Find libjpeg-turbo djpeg binary. Prefers the locally-built static binary
/// over the system djpeg (which may be IJG libjpeg 9d, NOT libjpeg-turbo).
fn find_turbo_djpeg() -> &'static str {
    const TURBO_DJPEG: &str = "/tmp/libjpeg-turbo-build/djpeg-static";
    if std::path::Path::new(TURBO_DJPEG).exists() {
        TURBO_DJPEG
    } else {
        // Fall back to system djpeg (may be IJG, not turbo!)
        "djpeg"
    }
}

fn decode_djpeg(path: &Path) -> Option<DecodeResult> {
    let djpeg_bin = find_turbo_djpeg();
    // Write to temp file to avoid binary data issues with stdout pipe
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("_zenjpeg_djpeg_out.pnm");
    let status = Command::new(djpeg_bin)
        .arg("-pnm")
        .arg("-outfile")
        .arg(&tmp_path)
        .arg(path)
        .output()
        .ok()?;
    if !status.status.success() {
        let _ = fs::remove_file(&tmp_path);
        return None;
    }
    let data = fs::read(&tmp_path).ok()?;
    let _ = fs::remove_file(&tmp_path);
    parse_pnm(&data)
}

fn parse_pnm(data: &[u8]) -> Option<DecodeResult> {
    // Parse binary PNM (P5=PGM, P6=PPM)
    // Header is ASCII text, body is raw binary
    let header_end = find_pnm_data_start(data)?;

    // Parse header as ASCII
    let header = std::str::from_utf8(&data[..header_end]).ok()?;
    let mut tokens = Vec::new();
    for line in header.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        for tok in line.split_whitespace() {
            tokens.push(tok.to_string());
        }
    }

    if tokens.len() < 4 {
        return None;
    }

    let channels = match tokens[0].as_str() {
        "P5" => 1, // PGM
        "P6" => 3, // PPM
        _ => return None,
    };
    let width: usize = tokens[1].parse().ok()?;
    let height: usize = tokens[2].parse().ok()?;
    let _maxval: usize = tokens[3].parse().ok()?;

    let pixels = data[header_end..].to_vec();
    let expected = width * height * channels;
    if pixels.len() < expected {
        return None;
    }

    Some(DecodeResult {
        pixels: pixels[..expected].to_vec(),
        width,
        height,
        channels,
    })
}

fn find_pnm_data_start(data: &[u8]) -> Option<usize> {
    // PNM header has: magic \n [comments] \n width height \n maxval \n <binary data>
    // We count 3 non-comment header lines (magic, dims, maxval)
    let mut header_lines = 0;
    let mut i = 0;
    while i < data.len() {
        if data[i] == b'\n' {
            // Check if next line is a comment
            if i + 1 < data.len() && data[i + 1] == b'#' {
                // skip comment line entirely
                i += 1;
                while i < data.len() && data[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            header_lines += 1;
            if header_lines == 3 {
                return Some(i + 1);
            }
        }
        i += 1;
    }
    None
}

fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
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

fn compute_dssim(a: &[u8], b: &[u8], width: usize, height: usize, channels: usize) -> f64 {
    if a.len() != b.len() {
        return 99.0;
    }
    let attr = Dssim::new();
    let (a_rgba, b_rgba): (Vec<RGBA8>, Vec<RGBA8>) = match channels {
        1 => {
            let a_r: Vec<RGBA8> = a.iter().map(|&g| RGBA8::new(g, g, g, 255)).collect();
            let b_r: Vec<RGBA8> = b.iter().map(|&g| RGBA8::new(g, g, g, 255)).collect();
            (a_r, b_r)
        }
        3 => {
            let a_r: Vec<RGBA8> = a
                .chunks(3)
                .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
                .collect();
            let b_r: Vec<RGBA8> = b
                .chunks(3)
                .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
                .collect();
            (a_r, b_r)
        }
        _ => return 99.0,
    };
    let a_img = attr.create_image_rgba(&a_rgba, width, height).unwrap();
    let b_img = attr.create_image_rgba(&b_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&a_img, b_img);
    dssim.into()
}

#[derive(Default)]
struct PairStats {
    count: usize,
    max_pixel_diffs: Vec<u8>,
    mean_abs_diffs: Vec<f64>,
    dssim_values: Vec<f64>,
    skipped: Vec<String>,
}

impl PairStats {
    fn add(&mut self, max_diff: u8, mean_diff: f64, dssim: f64) {
        self.count += 1;
        self.max_pixel_diffs.push(max_diff);
        self.mean_abs_diffs.push(mean_diff);
        self.dssim_values.push(dssim);
    }

    fn skip(&mut self, reason: String) {
        self.skipped.push(reason);
    }

    fn summary(&self) -> String {
        if self.count == 0 {
            return format!("(no comparable images, {} skipped)", self.skipped.len());
        }
        let max_diff = *self.max_pixel_diffs.iter().max().unwrap();
        let p50_diff = percentile(&self.max_pixel_diffs, 50);
        let mean_mad: f64 = self.mean_abs_diffs.iter().sum::<f64>() / self.count as f64;
        let max_dssim = self.dssim_values.iter().cloned().fold(0.0f64, f64::max);
        let mean_dssim = self.dssim_values.iter().sum::<f64>() / self.count as f64;
        format!(
            "n={:2} | max_pixel_diff: p50={} max={} | mean_abs_diff: {:.3} | DSSIM: mean={:.6} max={:.6} | skipped={}",
            self.count, p50_diff, max_diff, mean_mad, mean_dssim, max_dssim, self.skipped.len()
        )
    }
}

fn percentile(values: &[u8], pct: usize) -> u8 {
    let mut sorted: Vec<u8> = values.to_vec();
    sorted.sort();
    let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

#[test]
#[ignore]
fn corpus_decode_accuracy() {
    let c = match corpus() {
        Some(c) => c,
        None => { eprintln!("Skipping: corpus unavailable"); return; }
    };
    let corpus = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => { eprintln!("Skipping: {e}"); return; }
    };

    let files = collect_jpgs(&corpus);
    eprintln!("Found {} JPEG files in conformance corpus\n", files.len());

    // Decode all files with each decoder
    struct Decoded {
        name: String,
        zenjpeg: Option<DecodeResult>,
        zune: Option<DecodeResult>,
        jpeg_decoder: Option<DecodeResult>,
        djpeg: Option<DecodeResult>,
    }

    let mut decoded_files: Vec<Decoded> = Vec::new();
    let mut success_counts = [0usize; 4]; // zenjpeg, zune, jpeg-decoder, djpeg

    for path in &files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let data = fs::read(path).expect("read file");

        let zen = decode_zenjpeg(&data);
        let zune = decode_zune(&data);
        let jd = decode_jpeg_decoder_crate(&data);
        let dj = decode_djpeg(path);

        if zen.is_some() {
            success_counts[0] += 1;
        }
        if zune.is_some() {
            success_counts[1] += 1;
        }
        if jd.is_some() {
            success_counts[2] += 1;
        }
        if dj.is_some() {
            success_counts[3] += 1;
        }

        decoded_files.push(Decoded {
            name: fname,
            zenjpeg: zen,
            zune,
            jpeg_decoder: jd,
            djpeg: dj,
        });
    }

    eprintln!(
        "Decode success: zenjpeg={}/{} zune={}/{} jpeg-decoder={}/{} djpeg(libjpeg-turbo)={}/{}",
        success_counts[0],
        files.len(),
        success_counts[1],
        files.len(),
        success_counts[2],
        files.len(),
        success_counts[3],
        files.len(),
    );
    eprintln!();

    // Compare pairs: zenjpeg vs each other decoder
    let pairs: &[(&str, &str)] = &[
        ("zenjpeg", "djpeg(libjpeg-turbo)"),
        ("zenjpeg", "zune-jpeg"),
        ("zenjpeg", "jpeg-decoder"),
        ("zune-jpeg", "djpeg(libjpeg-turbo)"),
        ("jpeg-decoder", "djpeg(libjpeg-turbo)"),
    ];

    for &(name_a, name_b) in pairs {
        let mut stats = PairStats::default();

        for d in &decoded_files {
            let (a, b) = match (name_a, name_b) {
                ("zenjpeg", "djpeg(libjpeg-turbo)") => (&d.zenjpeg, &d.djpeg),
                ("zenjpeg", "zune-jpeg") => (&d.zenjpeg, &d.zune),
                ("zenjpeg", "jpeg-decoder") => (&d.zenjpeg, &d.jpeg_decoder),
                ("zune-jpeg", "djpeg(libjpeg-turbo)") => (&d.zune, &d.djpeg),
                ("jpeg-decoder", "djpeg(libjpeg-turbo)") => (&d.jpeg_decoder, &d.djpeg),
                _ => continue,
            };

            let (Some(a), Some(b)) = (a, b) else {
                stats.skip(format!("{}: one or both decoders failed", d.name));
                continue;
            };

            if a.width != b.width || a.height != b.height {
                stats.skip(format!(
                    "{}: dimension mismatch {}x{} vs {}x{}",
                    d.name, a.width, a.height, b.width, b.height
                ));
                continue;
            }

            // Handle channel mismatch (e.g., CMYK→RGB conversion differences)
            if a.channels != b.channels {
                stats.skip(format!(
                    "{}: channel mismatch {} vs {}",
                    d.name, a.channels, b.channels
                ));
                continue;
            }

            if a.pixels.len() != b.pixels.len() {
                stats.skip(format!(
                    "{}: pixel data length mismatch {} vs {}",
                    d.name,
                    a.pixels.len(),
                    b.pixels.len()
                ));
                continue;
            }

            let max_diff = max_pixel_diff(&a.pixels, &b.pixels);
            let mean_diff = mean_abs_diff(&a.pixels, &b.pixels);
            let dssim = compute_dssim(&a.pixels, &b.pixels, a.width, a.height, a.channels);
            stats.add(max_diff, mean_diff, dssim);
        }

        eprintln!("{:40} vs {:25} {}", name_a, name_b, stats.summary());
    }

    // Per-file details for zenjpeg vs djpeg (the C reference)
    eprintln!("\n=== Per-file: zenjpeg vs djpeg(libjpeg-turbo) ===");
    eprintln!(
        "{:<45} {:>8} {:>10} {:>12}",
        "File", "MaxDiff", "MeanDiff", "DSSIM"
    );
    eprintln!("{}", "-".repeat(80));

    let mut good_count = 0;
    let mut good_max_diff = 0u8;
    let mut good_dssim_sum = 0.0f64;
    let mut outliers = Vec::new();

    for d in &decoded_files {
        let (Some(a), Some(b)) = (&d.zenjpeg, &d.djpeg) else {
            continue;
        };
        if a.width != b.width
            || a.height != b.height
            || a.channels != b.channels
            || a.pixels.len() != b.pixels.len()
        {
            continue;
        }

        let max_diff = max_pixel_diff(&a.pixels, &b.pixels);
        let mean_diff = mean_abs_diff(&a.pixels, &b.pixels);
        let dssim = compute_dssim(&a.pixels, &b.pixels, a.width, a.height, a.channels);
        eprintln!(
            "{:<45} {:>8} {:>10.4} {:>12.8}",
            d.name, max_diff, mean_diff, dssim
        );

        // Track "normal" images (max_diff <= 30, DSSIM < 0.001) vs outliers
        if max_diff <= 30 && dssim < 0.001 {
            good_count += 1;
            good_max_diff = good_max_diff.max(max_diff);
            good_dssim_sum += dssim;
        } else {
            outliers.push(format!(
                "  {} (max_diff={}, dssim={:.6})",
                d.name, max_diff, dssim
            ));
        }
    }

    eprintln!("\n=== Summary (excluding outliers) ===");
    eprintln!(
        "Normal images: {}/{} with max_pixel_diff <= {} and mean DSSIM = {:.8}",
        good_count,
        good_count + outliers.len(),
        good_max_diff,
        if good_count > 0 {
            good_dssim_sum / good_count as f64
        } else {
            0.0
        }
    );
    if !outliers.is_empty() {
        eprintln!("Outliers ({}):", outliers.len());
        for o in &outliers {
            eprintln!("{}", o);
        }
    }

    // Now compare zenjpeg vs zune-jpeg per-file
    eprintln!("\n=== Per-file: zenjpeg vs zune-jpeg ===");
    eprintln!(
        "{:<45} {:>8} {:>10} {:>12}",
        "File", "MaxDiff", "MeanDiff", "DSSIM"
    );
    eprintln!("{}", "-".repeat(80));

    for d in &decoded_files {
        let (Some(a), Some(b)) = (&d.zenjpeg, &d.zune) else {
            continue;
        };
        if a.width != b.width
            || a.height != b.height
            || a.channels != b.channels
            || a.pixels.len() != b.pixels.len()
        {
            continue;
        }

        let max_diff = max_pixel_diff(&a.pixels, &b.pixels);
        let mean_diff = mean_abs_diff(&a.pixels, &b.pixels);
        let dssim = compute_dssim(&a.pixels, &b.pixels, a.width, a.height, a.channels);
        eprintln!(
            "{:<45} {:>8} {:>10.4} {:>12.8}",
            d.name, max_diff, mean_diff, dssim
        );
    }

    // Also compare zune vs djpeg per-file to establish the "normal" inter-decoder variance
    eprintln!("\n=== Per-file: zune-jpeg vs djpeg(libjpeg-turbo) ===");
    eprintln!(
        "{:<45} {:>8} {:>10} {:>12}",
        "File", "MaxDiff", "MeanDiff", "DSSIM"
    );
    eprintln!("{}", "-".repeat(80));

    for d in &decoded_files {
        let (Some(a), Some(b)) = (&d.zune, &d.djpeg) else {
            continue;
        };
        if a.width != b.width
            || a.height != b.height
            || a.channels != b.channels
            || a.pixels.len() != b.pixels.len()
        {
            continue;
        }

        let max_diff = max_pixel_diff(&a.pixels, &b.pixels);
        let mean_diff = mean_abs_diff(&a.pixels, &b.pixels);
        let dssim = compute_dssim(&a.pixels, &b.pixels, a.width, a.height, a.channels);
        eprintln!(
            "{:<45} {:>8} {:>10.4} {:>12.8}",
            d.name, max_diff, mean_diff, dssim
        );
    }
}

#[test]
#[ignore]
fn corpus_libjpeg_compat_vs_djpeg() {
    let c = match corpus() {
        Some(c) => c,
        None => { eprintln!("Skipping: corpus unavailable"); return; }
    };
    let corpus = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => { eprintln!("Skipping: {e}"); return; }
    };

    let files = collect_jpgs(&corpus);
    eprintln!(
        "Found {} JPEG files — comparing Triangle vs LibjpegCompat vs djpeg\n",
        files.len()
    );

    eprintln!(
        "{:<45} {:>10} {:>10} {:>10} {:>10}",
        "File", "Tri→djpeg", "LJC→djpeg", "Tri→LJC", "Tri→Zune"
    );
    eprintln!("{}", "-".repeat(95));

    let mut tri_vs_djpeg_diffs = Vec::new();
    let mut ljc_vs_djpeg_diffs = Vec::new();
    let mut tri_vs_ljc_diffs = Vec::new();

    for path in &files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let data = fs::read(path).expect("read file");

        let tri = decode_zenjpeg(&data);
        let ljc = decode_zenjpeg_libjpeg_compat(&data);
        let dj = decode_djpeg(path);
        let zune = decode_zune(&data);

        // Need all four for a useful comparison
        let (Some(tri), Some(ljc), Some(dj)) = (&tri, &ljc, &dj) else {
            eprintln!("{:<45} SKIPPED (decode failure)", fname);
            continue;
        };

        if tri.width != dj.width
            || tri.height != dj.height
            || tri.channels != dj.channels
            || tri.pixels.len() != dj.pixels.len()
        {
            eprintln!("{:<45} SKIPPED (dimension/channel mismatch)", fname);
            continue;
        }

        let tri_dj = max_pixel_diff(&tri.pixels, &dj.pixels);
        let ljc_dj = max_pixel_diff(&ljc.pixels, &dj.pixels);
        let tri_ljc = max_pixel_diff(&tri.pixels, &ljc.pixels);

        let tri_zune_str = if let Some(zune) = &zune {
            if zune.width == tri.width
                && zune.height == tri.height
                && zune.channels == tri.channels
                && zune.pixels.len() == tri.pixels.len()
            {
                let d = max_pixel_diff(&tri.pixels, &zune.pixels);
                format!("{}", d)
            } else {
                "dim?".to_string()
            }
        } else {
            "fail".to_string()
        };

        // Highlight improvements
        let marker = if ljc_dj < tri_dj {
            " <<"
        } else if ljc_dj > tri_dj {
            " !!"
        } else {
            ""
        };

        eprintln!(
            "{:<45} {:>10} {:>10} {:>10} {:>10}{}",
            fname, tri_dj, ljc_dj, tri_ljc, tri_zune_str, marker
        );

        tri_vs_djpeg_diffs.push((fname.clone(), tri_dj));
        ljc_vs_djpeg_diffs.push((fname.clone(), ljc_dj));
        tri_vs_ljc_diffs.push((fname, tri_ljc));
    }

    // Summary
    eprintln!("\n=== Summary ===");
    if !tri_vs_djpeg_diffs.is_empty() {
        let tri_max: u8 = tri_vs_djpeg_diffs.iter().map(|(_, d)| *d).max().unwrap();
        let ljc_max: u8 = ljc_vs_djpeg_diffs.iter().map(|(_, d)| *d).max().unwrap();
        let tri_mean: f64 = tri_vs_djpeg_diffs
            .iter()
            .map(|(_, d)| *d as f64)
            .sum::<f64>()
            / tri_vs_djpeg_diffs.len() as f64;
        let ljc_mean: f64 = ljc_vs_djpeg_diffs
            .iter()
            .map(|(_, d)| *d as f64)
            .sum::<f64>()
            / ljc_vs_djpeg_diffs.len() as f64;

        eprintln!(
            "Triangle   vs djpeg: max_pixel_diff max={:3}, mean={:.1}",
            tri_max, tri_mean
        );
        eprintln!(
            "LibjpegCompat vs djpeg: max_pixel_diff max={:3}, mean={:.1}",
            ljc_max, ljc_mean
        );

        // Show worst files for each
        let mut tri_sorted = tri_vs_djpeg_diffs.clone();
        tri_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("\nWorst Triangle vs djpeg:");
        for (name, diff) in tri_sorted.iter().take(5) {
            eprintln!("  {:>3} {}", diff, name);
        }

        let mut ljc_sorted = ljc_vs_djpeg_diffs.clone();
        ljc_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("\nWorst LibjpegCompat vs djpeg:");
        for (name, diff) in ljc_sorted.iter().take(5) {
            eprintln!("  {:>3} {}", diff, name);
        }

        // Count exact matches (0 diff)
        let tri_exact = tri_vs_djpeg_diffs.iter().filter(|(_, d)| *d == 0).count();
        let ljc_exact = ljc_vs_djpeg_diffs.iter().filter(|(_, d)| *d == 0).count();
        eprintln!(
            "\nExact matches: Triangle={}/{}, LibjpegCompat={}/{}",
            tri_exact,
            tri_vs_djpeg_diffs.len(),
            ljc_exact,
            ljc_vs_djpeg_diffs.len()
        );
    }
}

#[test]
#[ignore]
fn investigate_rst_diff() {
    use zenjpeg::decode::ChromaUpsampling;
    use zenjpeg::decoder::Decoder;

    let c = match corpus() {
        Some(c) => c,
        None => { eprintln!("Skipping: corpus unavailable"); return; }
    };
    let corpus_dir = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => { eprintln!("Skipping: {e}"); return; }
    };
    let path = corpus_dir.join("rst_1block.jpg");
    if !path.exists() {
        eprintln!("File not found");
        return;
    }
    let data = fs::read(&path).expect("read");

    // Decode with libjpeg-compat mode
    let ljc = Decoder::new()
        .chroma_upsampling(ChromaUpsampling::LibjpegCompat)
        .decode(&data, Unstoppable)
        .expect("decode");

    let dj = decode_djpeg(&path).expect("djpeg failed");

    let w = ljc.width as usize;
    let h = ljc.height as usize;
    let ljc_data = ljc.into_pixels_u8().unwrap();
    assert_eq!(w, dj.width);
    assert_eq!(h, dj.height);

    eprintln!(
        "Image {}x{}, {} bytes vs {} bytes",
        w,
        h,
        ljc_data.len(),
        dj.pixels.len()
    );

    // Find top-N worst diffs with location and channel info
    let mut diffs: Vec<(usize, usize, &str, i16, u8, u8)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for (c, ch_name) in [(0, "R"), (1, "G"), (2, "B")] {
                let idx = (y * w + x) * 3 + c;
                if idx >= ljc_data.len() || idx >= dj.pixels.len() {
                    continue;
                }
                let z = ljc_data[idx];
                let d = dj.pixels[idx];
                let diff = z as i16 - d as i16;
                if diff.unsigned_abs() > 20 {
                    diffs.push((x, y, ch_name, diff, z, d));
                }
            }
        }
    }

    diffs.sort_by(|a, b| b.3.unsigned_abs().cmp(&a.3.unsigned_abs()));

    eprintln!("\nTop 30 worst pixel diffs (|diff|>20):");
    eprintln!(
        "{:>4} {:>4} {:>2} {:>6} {:>4} {:>4}",
        "x", "y", "ch", "diff", "zen", "dj"
    );
    for &(x, y, ch, diff, z, d) in diffs.iter().take(30) {
        eprintln!("{:>4} {:>4} {:>2} {:>6} {:>4} {:>4}", x, y, ch, diff, z, d);
    }

    // Per-channel max diffs
    let mut r_max = 0i16;
    let mut g_max = 0i16;
    let mut b_max = 0i16;
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            if idx + 2 >= ljc_data.len() || idx + 2 >= dj.pixels.len() {
                continue;
            }
            r_max = r_max.max((ljc_data[idx] as i16 - dj.pixels[idx] as i16).abs());
            g_max = g_max.max((ljc_data[idx + 1] as i16 - dj.pixels[idx + 1] as i16).abs());
            b_max = b_max.max((ljc_data[idx + 2] as i16 - dj.pixels[idx + 2] as i16).abs());
        }
    }
    eprintln!(
        "\nPer-channel max diff: R={} G={} B={}",
        r_max, g_max, b_max
    );

    // Check if worst diffs cluster in certain MCU block positions
    eprintln!("\nDiffs by block position (block-relative x,y):");
    let mut block_hist = std::collections::HashMap::new();
    for &(x, y, _, diff, _, _) in &diffs {
        if diff.unsigned_abs() > 40 {
            let bx = x % 8;
            let by = y % 8;
            *block_hist.entry((bx, by)).or_insert(0u32) += 1;
        }
    }
    let mut block_sorted: Vec<_> = block_hist.into_iter().collect();
    block_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for ((bx, by), count) in block_sorted.iter().take(10) {
        eprintln!("  block_pos ({},{}) : {} high diffs", bx, by, count);
    }
}

/// Test that border pixels at partial MCU boundaries are as accurate as interior pixels.
/// For 4:2:0 images with non-aligned dimensions, the rightmost columns and bottom rows
/// sit at partial MCU boundaries where chroma upsampling and IDCT clipping interact.
/// This test verifies those specific pixels aren't worse than interior pixels.
#[test]
#[ignore]
fn border_pixel_accuracy() {
    let c = match corpus() {
        Some(c) => c,
        None => { eprintln!("Skipping: corpus unavailable"); return; }
    };
    let corpus = match c.get("jpeg-conformance/valid") {
        Ok(p) => p,
        Err(e) => { eprintln!("Skipping: {e}"); return; }
    };

    let files = collect_jpgs(&corpus);

    eprintln!(
        "{:<45} {:>5} {:>5} {:>4} {:>5} {:>10} {:>10} {:>10} {:>10}",
        "File", "W", "H", "Samp", "Edge", "Interior", "RightEdge", "BottomEdge", "Corner"
    );
    eprintln!("{}", "-".repeat(120));

    let mut any_border_worse = false;

    for path in &files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let data = fs::read(path).expect("read file");

        let zen = match decode_zenjpeg(&data) {
            Some(d) => d,
            None => continue,
        };
        let dj = match decode_djpeg(path) {
            Some(d) => d,
            None => continue,
        };

        if zen.width != dj.width
            || zen.height != dj.height
            || zen.channels != dj.channels
            || zen.pixels.len() != dj.pixels.len()
        {
            continue;
        }

        let w = zen.width;
        let h = zen.height;
        let ch = zen.channels;

        // Determine MCU size from subsampling (read SOF marker)
        // For now, detect from image: if 4:2:0, MCU=16x16; if 4:4:4, MCU=8x8
        // We'll use the JPEG parser to get actual sampling factors
        let (mcu_w, mcu_h) = detect_mcu_size(&data);

        let edge_w = w % mcu_w; // partial columns at right edge (0 = aligned)
        let edge_h = h % mcu_h; // partial rows at bottom edge (0 = aligned)

        // Skip images that are fully MCU-aligned (no partial boundary to test)
        if edge_w == 0 && edge_h == 0 {
            continue;
        }

        // Compute max pixel diff in regions:
        // Interior: not touching any edge MCU boundary
        // RightEdge: rightmost edge_w columns (or last mcu_w columns if edge_w == 0)
        // BottomEdge: bottom edge_h rows (or last mcu_h rows if edge_h == 0)
        // Corner: intersection of right and bottom edges

        let right_start = if edge_w > 0 { w - edge_w } else { w };
        let bottom_start = if edge_h > 0 { h - edge_h } else { h };

        let mut interior_max = 0u8;
        let mut right_max = 0u8;
        let mut bottom_max = 0u8;
        let mut corner_max = 0u8;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * ch;
                let mut pixel_max = 0u8;
                for c in 0..ch {
                    let diff = (zen.pixels[idx + c] as i16 - dj.pixels[idx + c] as i16)
                        .unsigned_abs() as u8;
                    pixel_max = pixel_max.max(diff);
                }

                let in_right = x >= right_start && edge_w > 0;
                let in_bottom = y >= bottom_start && edge_h > 0;

                if in_right && in_bottom {
                    corner_max = corner_max.max(pixel_max);
                } else if in_right {
                    right_max = right_max.max(pixel_max);
                } else if in_bottom {
                    bottom_max = bottom_max.max(pixel_max);
                } else {
                    interior_max = interior_max.max(pixel_max);
                }
            }
        }

        let samp_str = format!("{}x{}", mcu_w / 8, mcu_h / 8);
        let edge_str = if edge_w > 0 && edge_h > 0 {
            format!("{}x{}", edge_w, edge_h)
        } else if edge_w > 0 {
            format!("{}xOK", edge_w)
        } else {
            format!("OKx{}", edge_h)
        };

        // Flag if border is worse than interior
        let worst_border = right_max.max(bottom_max).max(corner_max);
        let flag = if worst_border > interior_max + 1 {
            " !!!"
        } else {
            ""
        };
        if worst_border > interior_max + 1 {
            any_border_worse = true;
        }

        eprintln!(
            "{:<45} {:>5} {:>5} {:>4} {:>5} {:>10} {:>10} {:>10} {:>10}{}",
            fname, w, h, samp_str, edge_str, interior_max, right_max, bottom_max, corner_max, flag
        );
    }

    eprintln!();
    if any_border_worse {
        eprintln!("WARNING: Some border regions have significantly worse accuracy than interior!");
    } else {
        eprintln!("OK: Border pixel accuracy is comparable to interior pixels.");
    }

    assert!(
        !any_border_worse,
        "Border pixels should not be significantly worse than interior"
    );
}

/// Detect MCU dimensions from JPEG sampling factors
fn detect_mcu_size(data: &[u8]) -> (usize, usize) {
    // Quick parse: find SOF marker and read sampling factors
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF {
            let marker = data[i + 1];
            match marker {
                0xC0..=0xC2 => {
                    // SOF0/SOF1/SOF2
                    if i + 11 < data.len() {
                        let nf = data[i + 9] as usize; // number of components
                        if nf >= 1 && i + 10 + nf * 3 <= data.len() {
                            let mut max_h = 1usize;
                            let mut max_v = 1usize;
                            for c in 0..nf {
                                let sampling = data[i + 11 + c * 3];
                                let h = (sampling >> 4) as usize;
                                let v = (sampling & 0x0F) as usize;
                                max_h = max_h.max(h);
                                max_v = max_v.max(v);
                            }
                            return (max_h * 8, max_v * 8);
                        }
                    }
                    return (8, 8); // fallback
                }
                0xD8 | 0xD9 | 0x00 => {
                    i += 2;
                    continue;
                }
                _ => {
                    // Skip marker segment
                    if i + 3 < data.len() {
                        let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                        i += 2 + len;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    (8, 8) // default
}
