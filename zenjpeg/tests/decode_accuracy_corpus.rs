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

use dssim::Dssim;
use enough::Unstoppable;
use rgb::RGBA8;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const CORPUS_DIR: &str = "/home/lilith/work/codec-eval/codec-corpus/jpeg-conformance/valid";

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
    let channels = if img.data.len() == (img.width as usize * img.height as usize) {
        1
    } else {
        3
    };
    Some(DecodeResult {
        pixels: img.data,
        width: img.width as usize,
        height: img.height as usize,
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

fn decode_djpeg(path: &Path) -> Option<DecodeResult> {
    // Write to temp file to avoid binary data issues with stdout pipe
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("_zenjpeg_djpeg_out.pnm");
    let status = Command::new("djpeg")
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
    let corpus = Path::new(CORPUS_DIR);
    if !corpus.exists() {
        eprintln!("Corpus not found at {CORPUS_DIR}");
        return;
    }

    let files = collect_jpgs(corpus);
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
