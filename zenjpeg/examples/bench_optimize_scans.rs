//! Benchmark optimize_scans on a corpus of images.
//!
//! Usage: cargo run --release -p zenjpeg --example bench_optimize_scans -- [corpus_dir]

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn encode_image(config: &EncoderConfig, width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable)
        .expect("push pixels");
    enc.finish().expect("finish encoding")
}

fn try_encode_image(
    config: &EncoderConfig,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<Vec<u8>, String> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| format!("setup: {}", e))?;
    enc.push_packed(pixels, enough::Unstoppable)
        .map_err(|e| format!("push: {}", e))?;
    enc.finish().map_err(|e| format!("finish: {}", e))
}

fn load_png(path: &std::path::Path) -> Option<(u32, u32, Vec<u8>)> {
    let data = std::fs::read(path).ok()?;
    let decoder = png::Decoder::new(std::io::Cursor::new(&data));
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info().clone();

    if info.color_type != png::ColorType::Rgb && info.color_type != png::ColorType::Rgba {
        eprintln!(
            "  Skipping {} (color type {:?})",
            path.file_name().unwrap().to_string_lossy(),
            info.color_type
        );
        return None;
    }
    if info.bit_depth != png::BitDepth::Eight {
        eprintln!(
            "  Skipping {} (bit depth {:?})",
            path.file_name().unwrap().to_string_lossy(),
            info.bit_depth
        );
        return None;
    }

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).ok()?;
    let raw = &buf[..frame.buffer_size()];

    let width = info.width;
    let height = info.height;

    let rgb = if info.color_type == png::ColorType::Rgba {
        raw.chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect()
    } else {
        raw.to_vec()
    };

    Some((width, height, rgb))
}

fn decode_jpeg_zune(jpeg_data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(jpeg_data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("zune-jpeg decode")
}

fn main() {
    let corpus_dir = std::env::args().nth(1).unwrap_or_else(|| {
        codec_corpus::Corpus::new()
            .ok()
            .and_then(|c| c.get("clic2025/final-test").ok())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| {
                let default = std::path::Path::new(
                    &std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()),
                )
                .join("work/codec-eval/codec-corpus/clic2025/final-test");
                default.to_string_lossy().to_string()
            })
    });

    let dir = std::path::Path::new(&corpus_dir);
    if !dir.exists() {
        eprintln!("Directory not found: {}", corpus_dir);
        std::process::exit(1);
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    eprintln!("Found {} PNG images in {}", entries.len(), corpus_dir);

    let qualities = [5.0, 10.0, 25.0, 50.0, 75.0, 85.0, 90.0];

    for quality in &qualities {
        eprintln!("\n=== Quality {} ===", quality);
        eprintln!(
            "{:<12} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "Image", "Size", "Normal", "Optimized", "Savings", "Savings%", "MaxDiff"
        );
        eprintln!("{}", "-".repeat(78));

        let mut total_normal = 0usize;
        let mut total_opt = 0usize;
        let mut count = 0usize;
        let mut encode_failures = 0usize;

        for entry in &entries {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy();
            let short_name: String = name.chars().take(10).collect();

            let (width, height, pixels) = match load_png(&path) {
                Some(v) => v,
                None => continue,
            };

            let size_str = format!("{}x{}", width, height);

            let config_normal =
                EncoderConfig::ycbcr(*quality, ChromaSubsampling::Quarter).progressive(true);
            let config_opt =
                EncoderConfig::ycbcr(*quality, ChromaSubsampling::Quarter).optimize_scans(true);

            let jpeg_normal = encode_image(&config_normal, width, height, &pixels);

            let jpeg_opt = match try_encode_image(&config_opt, width, height, &pixels) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "{:<12} {:>6} {:>10}    ENCODE FAILED: {}",
                        short_name,
                        size_str,
                        jpeg_normal.len(),
                        e
                    );
                    encode_failures += 1;
                    continue;
                }
            };

            // Full decode both to verify correctness
            let decoded_normal = decode_jpeg_zune(&jpeg_normal);
            let decoded_opt = decode_jpeg_zune(&jpeg_opt);

            let expected_size = (width * height * 3) as usize;
            if decoded_normal.len() != expected_size || decoded_opt.len() != expected_size {
                eprintln!("  {} DECODE SIZE MISMATCH!", short_name);
                encode_failures += 1;
                continue;
            }

            // Check pixel differences
            let max_diff: u8 = decoded_normal
                .iter()
                .zip(decoded_opt.iter())
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap_or(0);

            let savings = jpeg_normal.len() as i64 - jpeg_opt.len() as i64;
            let savings_pct = (1.0 - jpeg_opt.len() as f64 / jpeg_normal.len() as f64) * 100.0;

            eprintln!(
                "{:<12} {:>6} {:>10} {:>10} {:>10} {:>7.2}% {:>8}",
                short_name,
                size_str,
                jpeg_normal.len(),
                jpeg_opt.len(),
                savings,
                savings_pct,
                max_diff
            );

            if max_diff > 1 {
                eprintln!("  WARNING: max pixel diff {} > 1!", max_diff);
            }

            total_normal += jpeg_normal.len();
            total_opt += jpeg_opt.len();
            count += 1;
        }

        let total_savings_pct = (1.0 - total_opt as f64 / total_normal.max(1) as f64) * 100.0;

        eprintln!("{}", "-".repeat(78));
        eprintln!(
            "{:<12} {:>6} {:>10} {:>10} {:>10} {:>7.2}%",
            "TOTAL",
            "",
            total_normal,
            total_opt,
            total_normal as i64 - total_opt as i64,
            total_savings_pct
        );
        eprintln!("{} images encoded, {} failures", count, encode_failures);
    }
}
