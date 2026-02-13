//! Measure the actual file size cost of restart markers at various intervals.
//!
//! Uses the real encoder to produce JPEGs with and without restart markers,
//! then measures the byte-level cost.
//!
//! Run with:
//!   cargo test --release -p zenjpeg --test restart_cost_analysis -- --nocapture --ignored

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

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

/// Encode an image with a given restart interval and return the JPEG size.
fn encode_with_restart(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: i32,
    restart_interval: u16,
    baseline: bool,
) -> Option<usize> {
    let config = if baseline {
        EncoderConfig::ycbcr(quality, ChromaSubsampling::None)
            .progressive(false)
            .restart_interval(restart_interval)
    } else {
        EncoderConfig::ycbcr(quality, ChromaSubsampling::None)
            .restart_interval(restart_interval)
    };

    match config.encode_bytes(pixels, width, height, PixelLayout::Rgb8Srgb) {
        Ok(jpeg) => Some(Vec::len(&jpeg)),
        Err(e) => {
            eprintln!("  Encode error: {e}");
            None
        }
    }
}

/// Restart interval in MCU rows (convert to MCUs)
fn rows_to_mcus(mcu_rows: u16, mcu_cols: u16) -> u16 {
    mcu_rows.saturating_mul(mcu_cols)
}

#[test]
#[ignore]
fn restart_marker_file_size_cost() {
    let images = find_corpus_images();
    if images.is_empty() {
        eprintln!("No corpus images found");
        return;
    }

    let quality = 85;

    // Test both baseline and progressive
    for &(mode_name, baseline) in &[("Baseline", true), ("Progressive", false)] {
        eprintln!("\n================================================================================");
        eprintln!("=== Restart Marker Cost: {mode_name} Q{quality} 4:4:4 ===");
        eprintln!("================================================================================\n");

        eprintln!(
            "{:<8} {:<25} {:>5}x{:<5} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "Corpus", "Image", "W", "H", "NoRST",
            "1row", "2row", "4row", "8row", "16row", "32row"
        );
        eprintln!("{}", "-".repeat(120));

        let mut corpus_stats: std::collections::HashMap<&str, CorpusStats> =
            std::collections::HashMap::new();
        let mut global = CorpusStats::default();

        for (path, corpus) in &images {
            let Some((pixels, width, height)) = load_png(path) else {
                continue;
            };
            if width < 64 || height < 64 {
                continue;
            }

            let mcu_cols = ((width + 7) / 8) as u16;

            // Baseline: no restart markers
            let Some(base_size) = encode_with_restart(&pixels, width, height, quality, 0, baseline)
            else {
                continue;
            };

            let filename = path.file_stem().unwrap_or_default().to_string_lossy();
            let name = if filename.len() > 23 {
                format!("{}...", &filename[..20])
            } else {
                filename.to_string()
            };

            // Test restart intervals in MCU rows
            let row_intervals = [1u16, 2, 4, 8, 16, 32];
            let mut cost_strs = Vec::new();

            for &rows in &row_intervals {
                let mcu_interval = rows_to_mcus(rows, mcu_cols);
                if let Some(rst_size) =
                    encode_with_restart(&pixels, width, height, quality, mcu_interval, baseline)
                {
                    let delta_pct =
                        (rst_size as f64 - base_size as f64) / base_size as f64 * 100.0;
                    cost_strs.push(format!("{:>+8.3}%", delta_pct));

                    let entry = corpus_stats.entry(corpus).or_default();
                    entry.add(rows as usize, base_size, rst_size);
                    global.add(rows as usize, base_size, rst_size);
                } else {
                    cost_strs.push("     ERR".to_string());
                }
            }

            eprintln!(
                "{:<8} {:<25} {:>5}x{:<5} {:>8} {} {} {} {} {} {}",
                corpus,
                name,
                width,
                height,
                base_size,
                cost_strs[0],
                cost_strs[1],
                cost_strs[2],
                cost_strs[3],
                cost_strs[4],
                cost_strs[5],
            );
        }

        eprintln!("{}", "-".repeat(120));

        // Summaries
        let row_intervals = [1usize, 2, 4, 8, 16, 32];

        eprintln!("\n=== Per-Corpus Summary ({mode_name}) ===");
        for (corpus, stats) in &corpus_stats {
            eprintln!("\n  {corpus} ({} images):", stats.image_count);
            for &rows in &row_intervals {
                if let Some((total_base, total_rst)) = stats.totals.get(&rows) {
                    let delta_pct =
                        (*total_rst as f64 - *total_base as f64) / *total_base as f64 * 100.0;
                    let delta_bytes = *total_rst as i64 - *total_base as i64;
                    eprintln!(
                        "    DRI={:>2} rows: {:>+.4}% ({:>+} bytes)",
                        rows, delta_pct, delta_bytes
                    );
                }
            }
        }

        eprintln!(
            "\n=== Overall Summary ({mode_name}, {} images) ===",
            global.image_count
        );
        for &rows in &row_intervals {
            if let Some((total_base, total_rst)) = global.totals.get(&rows) {
                let delta_pct =
                    (*total_rst as f64 - *total_base as f64) / *total_base as f64 * 100.0;
                let delta_bytes = *total_rst as i64 - *total_base as i64;
                let avg_bytes = delta_bytes as f64 / global.image_count as f64;
                eprintln!(
                    "  DRI={:>2} rows: {:>+.4}% ({:>+} bytes total, {:>+.0} bytes/image avg)",
                    rows, delta_pct, delta_bytes, avg_bytes
                );
            }
        }
    }
}

#[derive(Default)]
struct CorpusStats {
    image_count: usize,
    // interval_rows -> (total_base_bytes, total_rst_bytes)
    totals: std::collections::HashMap<usize, (u64, u64)>,
}

impl CorpusStats {
    fn add(&mut self, interval_rows: usize, base_size: usize, rst_size: usize) {
        // Only count the image once (use the first interval to count)
        if interval_rows == 1 {
            self.image_count += 1;
        }
        let entry = self.totals.entry(interval_rows).or_insert((0, 0));
        entry.0 += base_size as u64;
        entry.1 += rst_size as u64;
    }
}

/// Also measure: what's the cost at fixed MCU intervals (not row-based)?
/// This is what the parallel encoder actually uses.
#[test]
#[ignore]
fn restart_marker_mcu_intervals() {
    let images = find_corpus_images();
    if images.is_empty() {
        eprintln!("No corpus images found");
        return;
    }

    let quality = 85;

    eprintln!("\n=== Restart Marker Cost: Fixed MCU Intervals, Baseline Q{quality} 4:4:4 ===");
    eprintln!("(This is what parallel encoding actually uses)\n");

    eprintln!(
        "{:<8} {:<25} {:>5}x{:<5} {:>4} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "Corpus", "Image", "W", "H", "MCUs", "NoRST",
        "DRI=64", "DRI=128", "DRI=256", "DRI=512", "DRI=1024"
    );
    eprintln!("{}", "-".repeat(110));

    let mcu_intervals = [64u16, 128, 256, 512, 1024];
    let mut global_base: u64 = 0;
    let mut global_rst: [u64; 5] = [0; 5];
    let mut count = 0usize;

    for (path, corpus) in &images {
        let Some((pixels, width, height)) = load_png(path) else {
            continue;
        };
        if width < 64 || height < 64 {
            continue;
        }

        let mcu_cols = (width + 7) / 8;
        let mcu_rows = (height + 7) / 8;
        let total_mcus = mcu_cols * mcu_rows;

        let Some(base_size) = encode_with_restart(&pixels, width, height, quality, 0, true) else {
            continue;
        };

        let filename = path.file_stem().unwrap_or_default().to_string_lossy();
        let name = if filename.len() > 23 {
            format!("{}...", &filename[..20])
        } else {
            filename.to_string()
        };

        let mut cost_strs = Vec::new();
        for (i, &interval) in mcu_intervals.iter().enumerate() {
            if let Some(rst_size) =
                encode_with_restart(&pixels, width, height, quality, interval, true)
            {
                let delta_pct =
                    (rst_size as f64 - base_size as f64) / base_size as f64 * 100.0;
                cost_strs.push(format!("{:>+8.3}%", delta_pct));
                global_rst[i] += rst_size as u64;
            } else {
                cost_strs.push("     ERR".to_string());
            }
        }

        global_base += base_size as u64;
        count += 1;

        eprintln!(
            "{:<8} {:<25} {:>5}x{:<5} {:>4} {:>8} {} {} {} {} {}",
            corpus,
            name,
            width,
            height,
            total_mcus,
            base_size,
            cost_strs[0],
            cost_strs[1],
            cost_strs[2],
            cost_strs[3],
            cost_strs[4],
        );
    }

    eprintln!("{}", "-".repeat(110));
    eprintln!("\n=== Overall ({count} images, Baseline Q{quality}) ===");
    for (i, &interval) in mcu_intervals.iter().enumerate() {
        if global_base > 0 {
            let delta_pct =
                (global_rst[i] as f64 - global_base as f64) / global_base as f64 * 100.0;
            let delta_bytes = global_rst[i] as i64 - global_base as i64;
            let _num_restarts_approx = global_base / (interval as u64 * 50); // very rough
            eprintln!(
                "  DRI={:>4} MCUs: {:>+.4}% ({:>+} bytes total, {:>+.0} bytes/image avg)",
                interval,
                delta_pct,
                delta_bytes,
                delta_bytes as f64 / count as f64
            );
        }
    }

    eprintln!("\nNote: Positive = restart markers increase file size.");
    eprintln!("The cost includes: RST marker bytes (2 each) + DRI header (6) + DC prediction reset + byte alignment padding.");
}
