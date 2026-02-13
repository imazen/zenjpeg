//! Spatial Huffman analysis: measure potential compression gains from
//! per-region Huffman table optimization.
//!
//! Run with:
//!   cargo test --release -p zenjpeg --test spatial_huffman_analysis --features test-utils -- --nocapture --ignored

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use zenjpeg::encode::spatial_huffman::analyze_spatial_huffman;

// ---- Minimal JPEG encoder for getting quantized blocks ----

/// Standard JPEG luminance quantization table (Annex K)
const STD_LUMA_QT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
    56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
    104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table (Annex K)
const STD_CHROMA_QT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99,
    99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// JPEG zigzag order
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Scale quantization table by quality factor (libjpeg algorithm)
fn scale_qt(base: &[u16; 64], quality: u32) -> [u16; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };

    let mut result = [0u16; 64];
    for i in 0..64 {
        let val = (base[i] as u32 * scale + 50) / 100;
        result[i] = val.clamp(1, 255) as u16;
    }
    result
}

/// Simple forward DCT (not optimized — just for analysis)
fn fdct_8x8(block: &[f32; 64]) -> [f32; 64] {
    let mut result = [0.0f32; 64];

    // Row transform
    let mut temp = [0.0f32; 64];
    for row in 0..8 {
        for u in 0..8 {
            let cu = if u == 0 {
                1.0 / f32::sqrt(2.0)
            } else {
                1.0
            };
            let mut sum = 0.0;
            for x in 0..8 {
                sum +=
                    block[row * 8 + x] * f32::cos((2.0 * x as f32 + 1.0) * u as f32 * std::f32::consts::PI / 16.0);
            }
            temp[row * 8 + u] = cu * sum / 2.0;
        }
    }

    // Column transform
    for col in 0..8 {
        for v in 0..8 {
            let cv = if v == 0 {
                1.0 / f32::sqrt(2.0)
            } else {
                1.0
            };
            let mut sum = 0.0;
            for y in 0..8 {
                sum +=
                    temp[y * 8 + col] * f32::cos((2.0 * y as f32 + 1.0) * v as f32 * std::f32::consts::PI / 16.0);
            }
            result[v * 8 + col] = cv * sum / 2.0;
        }
    }

    result
}

/// Quantize a DCT block and output in zigzag order
fn quantize_zigzag(dct: &[f32; 64], qt: &[u16; 64]) -> [i16; 64] {
    let mut result = [0i16; 64];
    for i in 0..64 {
        let zigzag_pos = ZIGZAG[i];
        let val = dct[zigzag_pos] / qt[zigzag_pos] as f32;
        result[i] = val.round() as i16;
    }
    result
}

/// RGB to YCbCr conversion (BT.601)
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = 128.0 + (-0.168736 * rf - 0.331264 * gf + 0.5 * bf);
    let cr = 128.0 + (0.5 * rf - 0.418688 * gf - 0.081312 * bf);
    (y, cb, cr)
}

/// Extract and encode all 8x8 blocks from an image.
/// Returns (y_blocks, cb_blocks, cr_blocks) in raster order.
fn extract_blocks(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u32,
) -> (Vec<[i16; 64]>, Vec<[i16; 64]>, Vec<[i16; 64]>) {
    let luma_qt = scale_qt(&STD_LUMA_QT, quality);
    let chroma_qt = scale_qt(&STD_CHROMA_QT, quality);

    let blocks_w = (width + 7) / 8;
    let blocks_h = (height + 7) / 8;
    let total_blocks = blocks_w * blocks_h;

    let mut y_blocks = Vec::with_capacity(total_blocks);
    let mut cb_blocks = Vec::with_capacity(total_blocks);
    let mut cr_blocks = Vec::with_capacity(total_blocks);

    // Convert entire image to YCbCr planes
    let mut y_plane = vec![0.0f32; width * height];
    let mut cb_plane = vec![0.0f32; width * height];
    let mut cr_plane = vec![0.0f32; width * height];

    for py in 0..height {
        for px in 0..width {
            let idx = (py * width + px) * 3;
            let (y, cb, cr) = rgb_to_ycbcr(pixels[idx], pixels[idx + 1], pixels[idx + 2]);
            y_plane[py * width + px] = y;
            cb_plane[py * width + px] = cb;
            cr_plane[py * width + px] = cr;
        }
    }

    // Extract 8x8 blocks
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let mut y_block = [0.0f32; 64];
            let mut cb_block = [0.0f32; 64];
            let mut cr_block = [0.0f32; 64];

            for dy in 0..8 {
                for dx in 0..8 {
                    let py = (by * 8 + dy).min(height - 1);
                    let px = (bx * 8 + dx).min(width - 1);
                    let plane_idx = py * width + px;
                    let block_idx = dy * 8 + dx;

                    y_block[block_idx] = y_plane[plane_idx] - 128.0; // Level shift
                    cb_block[block_idx] = cb_plane[plane_idx] - 128.0;
                    cr_block[block_idx] = cr_plane[plane_idx] - 128.0;
                }
            }

            // Forward DCT
            let y_dct = fdct_8x8(&y_block);
            let cb_dct = fdct_8x8(&cb_block);
            let cr_dct = fdct_8x8(&cr_block);

            // Quantize (output in zigzag order)
            y_blocks.push(quantize_zigzag(&y_dct, &luma_qt));
            cb_blocks.push(quantize_zigzag(&cb_dct, &chroma_qt));
            cr_blocks.push(quantize_zigzag(&cr_dct, &chroma_qt));
        }
    }

    (y_blocks, cb_blocks, cr_blocks)
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = File::open(path).ok()?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    if info.color_type == png::ColorType::Rgba {
        let mut rgb = Vec::with_capacity(info.width as usize * info.height as usize * 3);
        for chunk in buf.chunks(4) {
            rgb.push(chunk[0]);
            rgb.push(chunk[1]);
            rgb.push(chunk[2]);
        }
        Some((rgb, info.width, info.height))
    } else if info.color_type == png::ColorType::Rgb {
        Some((buf, info.width, info.height))
    } else {
        None
    }
}

fn find_corpus_images() -> Vec<(PathBuf, &'static str)> {
    let mut images = Vec::new();
    let home = std::env::var("HOME").unwrap_or_default();

    let dirs: &[(&str, &str)] = &[
        (
            &format!("{home}/work/codec-eval/codec-corpus/CID22/CID22-512/validation"),
            "CID22",
        ),
        (
            &format!("{home}/work/codec-eval/codec-corpus/gb82-sc"),
            "Screenshots",
        ),
        (
            &format!("{home}/work/codec-eval/codec-corpus/clic2025/final-test"),
            "CLIC2025",
        ),
    ];

    for &(dir, corpus) in dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut paths: Vec<_> = entries
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
                .map(|e| (e.path(), corpus))
                .collect();
            paths.sort_by(|a, b| a.0.cmp(&b.0));
            images.extend(paths);
        }
    }
    images
}

#[test]
#[ignore]
fn spatial_huffman_ceiling() {
    let images = find_corpus_images();
    if images.is_empty() {
        eprintln!("No corpus images found at ~/work/codec-eval/codec-corpus/");
        return;
    }

    let quality = 85u32;

    eprintln!("\n=== Spatial Huffman Table Analysis (Q{quality}, 4:4:4) ===");
    eprintln!("Theoretical ceiling: per-band optimal Huffman vs global optimal Huffman\n");

    eprintln!(
        "{:<8} {:<30} {:>5}x{:<5} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Corpus", "Image", "W", "H", "GlblBits", "B=1row", "B=2row", "B=4row", "B=8row", "B=16row"
    );
    eprintln!("{}", "-".repeat(120));

    let band_sizes = [1usize, 2, 4, 8, 16, 32];
    // Accumulate per-corpus and global
    let mut corpus_stats: std::collections::HashMap<&str, (u64, [i64; 6])> =
        std::collections::HashMap::new();
    let mut global_total_bits = 0u64;
    let mut global_savings = [0i64; 6];
    let mut image_count = 0;

    for (path, corpus) in &images {
        let Some((pixels, width, height)) = load_png(path) else {
            continue;
        };
        if width < 64 || height < 64 {
            continue;
        }

        let (y_blocks, cb_blocks, cr_blocks) =
            extract_blocks(&pixels, width as usize, height as usize, quality);

        let mcu_cols = (width as usize + 7) / 8;
        let mcu_rows = (height as usize + 7) / 8;

        let results = analyze_spatial_huffman(
            &y_blocks,
            &cb_blocks,
            &cr_blocks,
            mcu_cols,
            mcu_rows,
            true,
        );

        let global_bits = results.first().map(|r| r.global_bits).unwrap_or(0);
        global_total_bits += global_bits;

        let filename = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let name = if filename.len() > 28 {
            format!("{}...", &filename[..25])
        } else {
            filename.to_string()
        };

        let mut savings_strs = Vec::new();
        for (i, r) in results.iter().enumerate() {
            if i < 6 {
                global_savings[i] += r.net_savings_bits;
                let entry = corpus_stats.entry(corpus).or_insert((0, [0i64; 6]));
                entry.1[i] += r.net_savings_bits;
            }
            savings_strs.push(format!("{:>7.3}%", r.savings_pct));
        }
        while savings_strs.len() < 6 {
            savings_strs.push("    N/A".to_string());
        }

        corpus_stats.entry(corpus).or_insert((0, [0i64; 6])).0 += global_bits;

        eprintln!(
            "{:<8} {:<30} {:>5}x{:<5} {:>10} {} {} {} {} {}",
            corpus,
            name,
            width,
            height,
            global_bits,
            savings_strs[0],
            savings_strs[1],
            savings_strs[2],
            savings_strs[3],
            savings_strs[4],
        );

        image_count += 1;
    }

    eprintln!("{}", "-".repeat(120));

    // Per-corpus summary
    eprintln!("\n=== Per-Corpus Summary ===");
    for (corpus, (total_bits, savings)) in &corpus_stats {
        eprintln!("\n  {corpus}:");
        for (i, &bs) in band_sizes.iter().enumerate() {
            if *total_bits > 0 {
                let pct = savings[i] as f64 / *total_bits as f64 * 100.0;
                eprintln!(
                    "    Band {:>2} MCU rows: {:>+10} bits ({:>+.4}%)",
                    bs, savings[i], pct
                );
            }
        }
    }

    // Overall summary
    eprintln!("\n=== Overall Summary ({image_count} images, Q{quality}) ===");
    eprintln!("  Total Huffman+extra bits (global tables): {global_total_bits}");
    for (i, &bs) in band_sizes.iter().enumerate() {
        if global_total_bits > 0 {
            let pct = global_savings[i] as f64 / global_total_bits as f64 * 100.0;
            eprintln!(
                "  Band {:>2} MCU rows: {:>+10} bits ({:>+.4}%)",
                bs, global_savings[i], pct
            );
        }
    }

    eprintln!("\nPositive = spatial tables would save bits (theoretical ceiling)");
    eprintln!("Negative = DHT overhead exceeds any per-band Huffman gains");
    eprintln!("This is the THEORETICAL CEILING — actual implementation would achieve less");
}

/// Also measure: what if we just used the best single band size, per-image?
#[test]
#[ignore]
fn spatial_huffman_best_band_per_image() {
    let images = find_corpus_images();
    if images.is_empty() {
        return;
    }

    let quality = 85u32;
    eprintln!("\n=== Best Band Size Per Image (Q{quality}) ===\n");

    let mut total_global = 0u64;
    let mut total_best_savings = 0i64;
    let mut best_band_counts = [0u32; 6];
    let band_sizes = [1usize, 2, 4, 8, 16, 32];

    for (path, _corpus) in &images {
        let Some((pixels, width, height)) = load_png(path) else {
            continue;
        };
        if width < 64 || height < 64 {
            continue;
        }

        let (y_blocks, cb_blocks, cr_blocks) =
            extract_blocks(&pixels, width as usize, height as usize, quality);
        let mcu_cols = (width as usize + 7) / 8;
        let mcu_rows = (height as usize + 7) / 8;

        let results = analyze_spatial_huffman(
            &y_blocks, &cb_blocks, &cr_blocks, mcu_cols, mcu_rows, true,
        );

        if let Some(first) = results.first() {
            total_global += first.global_bits;
        }

        // Find best band size for this image
        let mut best_savings = i64::MIN;
        let mut best_idx = 0;
        for (i, r) in results.iter().enumerate() {
            if r.net_savings_bits > best_savings {
                best_savings = r.net_savings_bits;
                best_idx = i;
            }
        }
        total_best_savings += best_savings;
        if best_idx < best_band_counts.len() {
            best_band_counts[best_idx] += 1;
        }
    }

    eprintln!("If each image picks its best band size:");
    eprintln!(
        "  Total savings: {:>+10} bits ({:>+.4}%)",
        total_best_savings,
        if total_global > 0 {
            total_best_savings as f64 / total_global as f64 * 100.0
        } else {
            0.0
        }
    );
    eprintln!("\nBest band size distribution:");
    for (i, &bs) in band_sizes.iter().enumerate() {
        eprintln!("  {:>2} MCU rows: {} images", bs, best_band_counts[i]);
    }
}
