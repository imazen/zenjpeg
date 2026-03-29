//! Decode all JPEGs in a directory with every decoder path, compare to jpeg-decoder.
//!
//! Tests: buffered (Srgb8), buffered (SrgbF32Precise), scanline (Srgb8), and
//! compares each against jpeg-decoder (libjpeg-compatible) pixel output.
//!
//! Usage:
//!   cargo run --release --features decoder --example fuzz_corpus_decode -- <input_dir> <fail_dir>

use std::fs;
use std::panic;
use std::path::PathBuf;
use std::time::Instant;

use imgref::ImgRefMut;
use zenjpeg::decode::{ChromaUpsampling, DecodeConfig, OutputTarget};

/// Decode modes to test
#[derive(Debug, Clone, Copy)]
enum Mode {
    /// Buffered decode, integer IDCT, u8 sRGB, libjpeg-compat upsampling
    BufferedInt,
    /// Buffered decode, f32 IDCT + dequant bias, libjpeg-compat upsampling
    BufferedF32,
    /// Buffered decode, integer IDCT, jpegli-style triangle upsampling
    BufferedTriangle,
    /// Scanline decode, integer IDCT, libjpeg-compat upsampling
    Scanline,
    /// Scanline decode, box filter upsampling (fast mode)
    ScanlineFast,
}

impl Mode {
    const ALL: &[Mode] = &[
        Mode::BufferedInt,
        Mode::BufferedF32,
        Mode::BufferedTriangle,
        Mode::Scanline,
        Mode::ScanlineFast,
    ];

    fn name(self) -> &'static str {
        match self {
            Mode::BufferedInt => "buf-int",
            Mode::BufferedF32 => "buf-f32",
            Mode::BufferedTriangle => "buf-tri",
            Mode::Scanline => "scanline",
            Mode::ScanlineFast => "scan-fast",
        }
    }
}

fn decode_buffered(
    data: &[u8],
    target: OutputTarget,
    upsampling: ChromaUpsampling,
) -> Result<(Vec<u8>, u32, u32), String> {
    let config = DecodeConfig::new()
        .chroma_upsampling(upsampling)
        .output_target(target);

    let result = config
        .decode(data, enough::Unstoppable)
        .map_err(|e| format!("{e}"))?;

    let w = result.width();
    let h = result.height();

    let pixels = match target {
        OutputTarget::Srgb8 => result.into_pixels_u8().ok_or("no u8 pixels")?,
        OutputTarget::SrgbF32Precise => {
            let f32_pixels = result.into_pixels_f32().ok_or("no f32 pixels")?;
            f32_pixels
                .iter()
                .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
                .collect()
        }
        _ => return Err("unsupported target".into()),
    };

    Ok((pixels, w, h))
}

fn decode_scanline(
    data: &[u8],
    upsampling: ChromaUpsampling,
) -> Result<(Vec<u8>, u32, u32), String> {
    let config = DecodeConfig::new().chroma_upsampling(upsampling);

    let mut reader = config.scanline_reader(data).map_err(|e| format!("{e}"))?;

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let is_gray = reader.info().color_space == zenjpeg::decode::ColorSpace::Grayscale;

    if is_gray {
        let mut pixels = vec![0u8; w * h];
        let mut rows_read = 0;
        while rows_read < h {
            let remaining = h - rows_read;
            let slice = &mut pixels[rows_read * w..];
            let output = ImgRefMut::new(slice, w, remaining);
            rows_read += reader.read_rows_gray8(output).map_err(|e| format!("{e}"))?;
        }
        let rgb: Vec<u8> = pixels.iter().flat_map(|&g| [g, g, g]).collect();
        Ok((rgb, w as u32, h as u32))
    } else {
        let stride = w * 3;
        let mut pixels = vec![0u8; stride * h];
        let mut rows_read = 0;
        while rows_read < h {
            let remaining = h - rows_read;
            let slice = &mut pixels[rows_read * stride..];
            let output = ImgRefMut::new(slice, stride, remaining);
            rows_read += reader.read_rows_rgb8(output).map_err(|e| format!("{e}"))?;
        }
        Ok((pixels, w as u32, h as u32))
    }
}

fn decode_jpeg_decoder(data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    use jpeg_decoder::Decoder;
    let mut decoder = Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().map_err(|e| format!("jpeg-decoder: {e}"))?;
    let info = decoder.info().ok_or("jpeg-decoder: no info")?;
    let w = info.width as u32;
    let h = info.height as u32;

    // Convert to RGB if needed
    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&g| [g, g, g]).collect(),
        jpeg_decoder::PixelFormat::L16 => {
            // 16-bit grayscale → 8-bit RGB
            pixels
                .chunks_exact(2)
                .flat_map(|pair| {
                    let v = (u16::from_ne_bytes([pair[0], pair[1]]) >> 8) as u8;
                    [v, v, v]
                })
                .collect()
        }
        jpeg_decoder::PixelFormat::CMYK32 => {
            // CMYK → RGB approximation
            pixels
                .chunks_exact(4)
                .flat_map(|cmyk| {
                    let c = cmyk[0] as f32 / 255.0;
                    let m = cmyk[1] as f32 / 255.0;
                    let y = cmyk[2] as f32 / 255.0;
                    let k = cmyk[3] as f32 / 255.0;
                    let r = (255.0 * (1.0 - c) * (1.0 - k)) as u8;
                    let g = (255.0 * (1.0 - m) * (1.0 - k)) as u8;
                    let b = (255.0 * (1.0 - y) * (1.0 - k)) as u8;
                    [r, g, b]
                })
                .collect()
        }
    };

    Ok((rgb, w, h))
}

fn pixel_diff_stats(a: &[u8], b: &[u8]) -> (u8, f64) {
    if a.len() != b.len() {
        return (255, 255.0);
    }
    let mut max_diff: u8 = 0;
    let mut sum_diff: u64 = 0;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let d = av.abs_diff(bv);
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let mean = sum_diff as f64 / a.len() as f64;
    (max_diff, mean)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_dir> <fail_dir>", args[0]);
        std::process::exit(1);
    }

    let input_dir = PathBuf::from(&args[1]);
    let fail_dir = PathBuf::from(&args[2]);

    if !input_dir.is_dir() {
        eprintln!("Input directory does not exist: {}", input_dir.display());
        std::process::exit(1);
    }

    fs::create_dir_all(&fail_dir).expect("Failed to create fail directory");

    let mut files: Vec<PathBuf> = fs::read_dir(&input_dir)
        .expect("Failed to read input directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| {
                    let s = ext.to_ascii_lowercase();
                    s == "jpg" || s == "jpeg"
                })
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    println!(
        "Found {} JPEG files in {}",
        files.len(),
        input_dir.display()
    );
    println!("Failures will be copied to {}", fail_dir.display());
    println!(
        "Modes: {}",
        Mode::ALL
            .iter()
            .map(|m| m.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Reference: jpeg-decoder (libjpeg-compatible)");
    println!();

    let start = Instant::now();
    let mut total_bytes = 0u64;

    struct ModeStats {
        ok: u32,
        fail: u32,
        max_diff_vs_ref: u8,
        sum_mean_diff: f64,
        count_compared: u32,
    }
    let mut stats: Vec<ModeStats> = Mode::ALL
        .iter()
        .map(|_| ModeStats {
            ok: 0,
            fail: 0,
            max_diff_vs_ref: 0,
            sum_mean_diff: 0.0,
            count_compared: 0,
        })
        .collect();
    let mut ref_fails = 0u32;
    let mut failures: Vec<(PathBuf, String)> = Vec::new();
    let mut worst_files: Vec<Vec<(String, u8, f64)>> =
        Mode::ALL.iter().map(|_| Vec::new()).collect();

    for (i, path) in files.iter().enumerate() {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                let msg = format!("IO: {e}");
                eprintln!("[{}/{}] IO FAIL {fname}: {msg}", i + 1, files.len());
                failures.push((path.clone(), msg));
                continue;
            }
        };
        total_bytes += data.len() as u64;

        // Decode with jpeg-decoder (reference)
        let ref_result = decode_jpeg_decoder(&data);
        let ref_pixels = match &ref_result {
            Ok((pixels, _w, _h)) => Some(pixels.as_slice()),
            Err(e) => {
                ref_fails += 1;
                if !e.contains("dimensions") {
                    eprintln!("[{}/{}] ref FAIL {fname}: {e}", i + 1, files.len());
                }
                None
            }
        };

        // Test each mode
        let mut any_fail = false;
        for (mi, mode) in Mode::ALL.iter().enumerate() {
            let data_ref = &data;
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| match mode {
                Mode::BufferedInt => {
                    decode_buffered(data_ref, OutputTarget::Srgb8, ChromaUpsampling::Triangle)
                }
                Mode::BufferedF32 => decode_buffered(
                    data_ref,
                    OutputTarget::SrgbF32Precise,
                    ChromaUpsampling::Triangle,
                ),
                Mode::BufferedTriangle => {
                    decode_buffered(data_ref, OutputTarget::Srgb8, ChromaUpsampling::Triangle)
                }
                Mode::Scanline => decode_scanline(data_ref, ChromaUpsampling::Triangle),
                Mode::ScanlineFast => decode_scanline(data_ref, ChromaUpsampling::NearestNeighbor),
            }));

            let result = match result {
                Ok(r) => r,
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<String>() {
                        format!("PANIC: {s}")
                    } else if let Some(s) = e.downcast_ref::<&str>() {
                        format!("PANIC: {s}")
                    } else {
                        "PANIC: unknown".into()
                    };
                    Err(msg)
                }
            };

            match result {
                Ok((pixels, _w, _h)) => {
                    stats[mi].ok += 1;
                    if let Some(ref_px) = ref_pixels {
                        let (max_d, mean_d) = pixel_diff_stats(&pixels, ref_px);
                        stats[mi].max_diff_vs_ref = stats[mi].max_diff_vs_ref.max(max_d);
                        stats[mi].sum_mean_diff += mean_d;
                        stats[mi].count_compared += 1;

                        if max_d > 4 {
                            worst_files[mi].push((fname.clone(), max_d, mean_d));
                        }
                    }
                }
                Err(e) => {
                    stats[mi].fail += 1;
                    any_fail = true;
                    eprintln!(
                        "[{}/{}] {} FAIL {fname}: {e}",
                        i + 1,
                        files.len(),
                        mode.name()
                    );
                }
            }
        }

        if any_fail {
            failures.push((path.clone(), "decode failure".into()));
        }

        if (i + 1) % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!(
                "  ... {}/{} ({:.0} files/sec, {:.1} MB)",
                i + 1,
                files.len(),
                rate,
                total_bytes as f64 / 1_048_576.0
            );
        }
    }

    // Copy failures
    for (path, _) in &failures {
        let dest = fail_dir.join(path.file_name().unwrap());
        if let Err(e) = fs::copy(path, &dest) {
            eprintln!("Failed to copy {}: {e}", path.display());
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!(
        "=== Results ({:.1}s, {:.0} files/sec, {:.1} MB) ===",
        elapsed.as_secs_f64(),
        files.len() as f64 / elapsed.as_secs_f64(),
        total_bytes as f64 / 1_048_576.0
    );
    println!();

    println!(
        "{:<12} {:>6} {:>6} {:>10} {:>12}",
        "Mode", "OK", "Fail", "MaxDiff", "AvgMeanDiff"
    );
    println!("{}", "-".repeat(52));
    for (mi, mode) in Mode::ALL.iter().enumerate() {
        let s = &stats[mi];
        let mean = if s.count_compared > 0 {
            s.sum_mean_diff / s.count_compared as f64
        } else {
            0.0
        };
        println!(
            "{:<12} {:>6} {:>6} {:>10} {:>12.4}",
            mode.name(),
            s.ok,
            s.fail,
            s.max_diff_vs_ref,
            mean
        );
    }
    println!();
    println!("jpeg-decoder failures: {ref_fails}");

    for (mi, mode) in Mode::ALL.iter().enumerate() {
        let mut w = worst_files[mi].clone();
        if w.is_empty() {
            continue;
        }
        w.sort_by(|a, b| b.1.cmp(&a.1));
        println!();
        println!(
            "=== {} files with max_diff > 4 vs jpeg-decoder ({}) ===",
            mode.name(),
            w.len()
        );
        for (fname, max_d, mean_d) in w.iter().take(20) {
            println!("  {fname}: max={max_d} mean={mean_d:.4}");
        }
        if w.len() > 20 {
            println!("  ... and {} more", w.len() - 20);
        }
    }

    if !failures.is_empty() {
        println!();
        println!("=== Decode failures ({}) ===", failures.len());
        for (path, msg) in &failures {
            println!("  {}: {msg}", path.file_name().unwrap().to_string_lossy());
        }
        println!();
        println!("Failed files copied to: {}", fail_dir.display());
    }
}
