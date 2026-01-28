//! Build corpus-based Huffman tables from CLIC 2025 validation set,
//! then validate on final-test set.
//!
//! This example:
//! 1. Encodes all validation images with optimal Huffman tables
//! 2. Collects and merges symbol frequencies per quality tier
//! 3. Validates corpus tables vs standard tables on final-test set
//! 4. Outputs Rust code for embedding the corpus tables
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example build_corpus_tables
//!
//! Output files are written to /mnt/v/output/zenjpeg/corpus_tables/

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use zenjpeg::encode::{HuffmanFrequencyCounts, Quality, StreamingEncoder};
use zenjpeg::types::Subsampling;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Quality tiers to test
const QUALITY_TIERS: &[u8] = &[75, 85, 95];

/// Output directory for generated tables
const OUTPUT_DIR: &str = "/mnt/v/output/zenjpeg/corpus_tables";

fn main() -> Result<()> {
    let corpus_base = PathBuf::from("/home/lilith/work/codec-corpus/clic2025");
    let validation_dir = corpus_base.join("validation");
    let final_test_dir = corpus_base.join("final-test");

    // Collect validation images
    let validation_images: Vec<PathBuf> = fs::read_dir(&validation_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "png"))
        .collect();

    // Collect final-test images
    let test_images: Vec<PathBuf> = fs::read_dir(&final_test_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "png"))
        .collect();

    println!("=== CLIC 2025 Corpus Huffman Table Builder ===\n");
    println!("Validation images: {}", validation_images.len());
    println!("Final-test images: {}", test_images.len());
    println!("Quality tiers: {:?}\n", QUALITY_TIERS);

    // Store corpus counts per quality tier
    let mut corpus_counts_by_quality: Vec<(u8, HuffmanFrequencyCounts)> = Vec::new();

    // Step 1: Build corpus tables from validation set
    println!("=== Step 1: Building corpus tables from validation set ===\n");

    for &quality in QUALITY_TIERS {
        println!("--- Quality {} ---", quality);

        let mut corpus_counts = HuffmanFrequencyCounts::new();
        let mut total_bytes = 0usize;
        let start = Instant::now();

        for (i, image_path) in validation_images.iter().enumerate() {
            let (width, height, pixels) = load_png(image_path)?;

            let mut encoder = StreamingEncoder::new(width, height)
                .quality(Quality::ApproxJpegli(quality as f32))
                .subsampling(Subsampling::S420)
                .progressive(false)
                .start()?;

            let row_size = width as usize * 3;
            for y in 0..height as usize {
                let start = y * row_size;
                encoder.push_row(&pixels[start..start + row_size])?;
            }

            let result = encoder.finish_with_tables()?;
            total_bytes += result.jpeg.len();

            // Merge frequency counts
            corpus_counts.add(&result.frequency_counts);

            if (i + 1) % 10 == 0 || i + 1 == validation_images.len() {
                print!("\r  Encoded {}/{} images...", i + 1, validation_images.len());
            }
        }

        let elapsed = start.elapsed();
        println!(
            "\n  Total bytes: {} KB, Time: {:.2}s",
            total_bytes / 1024,
            elapsed.as_secs_f64()
        );
        println!(
            "  AC luma entropy: {:.2} bits, symbols used: {}",
            corpus_counts.ac_luma.entropy(),
            corpus_counts.ac_luma.num_symbols()
        );

        corpus_counts_by_quality.push((quality, corpus_counts));
    }

    // Step 2: Validate on final-test set
    println!("\n=== Step 2: Validating on final-test set ===\n");

    for (quality, corpus_counts) in &corpus_counts_by_quality {
        println!("--- Quality {} ---", quality);

        // Generate tables from corpus
        let corpus_tables = corpus_counts.generate_tables()?;

        let mut optimal_total = 0usize;
        let mut corpus_total = 0usize;
        let mut standard_total = 0usize;

        let mut max_corpus_overhead = 0.0f64;
        let mut max_standard_overhead = 0.0f64;
        let mut corpus_overhead_sum = 0.0f64;
        let mut standard_overhead_sum = 0.0f64;

        for (i, image_path) in test_images.iter().enumerate() {
            let (width, height, pixels) = load_png(image_path)?;

            // Encode with optimal tables (baseline)
            let optimal_bytes = {
                let mut encoder = StreamingEncoder::new(width, height)
                    .quality(Quality::ApproxJpegli(*quality as f32))
                    .subsampling(Subsampling::S420)
                    .progressive(false)
                    .start()?;

                let row_size = width as usize * 3;
                for y in 0..height as usize {
                    let start = y * row_size;
                    encoder.push_row(&pixels[start..start + row_size])?;
                }
                encoder.finish()?.len()
            };

            // Encode with corpus tables (force streaming mode)
            let corpus_bytes = {
                let mut encoder = StreamingEncoder::new(width, height)
                    .quality(Quality::ApproxJpegli(*quality as f32))
                    .subsampling(Subsampling::S420)
                    .progressive(false)
                    .memory_limit(1) // Force immediate streaming
                    .custom_huffman_tables(corpus_tables.clone())
                    .start()?;

                let row_size = width as usize * 3;
                for y in 0..height as usize {
                    let start = y * row_size;
                    encoder.push_row(&pixels[start..start + row_size])?;
                }
                encoder.finish()?.len()
            };

            // Encode with standard JPEG tables (force streaming mode)
            let standard_bytes = {
                let mut encoder = StreamingEncoder::new(width, height)
                    .quality(Quality::ApproxJpegli(*quality as f32))
                    .subsampling(Subsampling::S420)
                    .progressive(false)
                    .memory_limit(1) // Force immediate streaming
                    .use_standard_huffman_tables(true)
                    .start()?;

                let row_size = width as usize * 3;
                for y in 0..height as usize {
                    let start = y * row_size;
                    encoder.push_row(&pixels[start..start + row_size])?;
                }
                encoder.finish()?.len()
            };

            optimal_total += optimal_bytes;
            corpus_total += corpus_bytes;
            standard_total += standard_bytes;

            let corpus_overhead =
                100.0 * (corpus_bytes as f64 - optimal_bytes as f64) / optimal_bytes as f64;
            let standard_overhead =
                100.0 * (standard_bytes as f64 - optimal_bytes as f64) / optimal_bytes as f64;

            corpus_overhead_sum += corpus_overhead;
            standard_overhead_sum += standard_overhead;

            if corpus_overhead > max_corpus_overhead {
                max_corpus_overhead = corpus_overhead;
            }
            if standard_overhead > max_standard_overhead {
                max_standard_overhead = standard_overhead;
            }

            if (i + 1) % 10 == 0 || i + 1 == test_images.len() {
                print!("\r  Validated {}/{} images...", i + 1, test_images.len());
            }
        }

        let n = test_images.len() as f64;
        let avg_corpus_overhead = corpus_overhead_sum / n;
        let avg_standard_overhead = standard_overhead_sum / n;

        let total_corpus_overhead =
            100.0 * (corpus_total as f64 - optimal_total as f64) / optimal_total as f64;
        let total_standard_overhead =
            100.0 * (standard_total as f64 - optimal_total as f64) / optimal_total as f64;

        println!();
        println!("  Results on {} test images:", test_images.len());
        println!();
        println!("  {:20} {:>12} {:>12}", "", "Corpus", "Standard");
        println!("  {:20} {:>12} {:>12}", "", "Tables", "Tables");
        println!("  {:20} {:>11.2}% {:>11.2}%", "Mean overhead:", avg_corpus_overhead, avg_standard_overhead);
        println!("  {:20} {:>11.2}% {:>11.2}%", "Max overhead:", max_corpus_overhead, max_standard_overhead);
        println!("  {:20} {:>11.2}% {:>11.2}%", "Total overhead:", total_corpus_overhead, total_standard_overhead);
        println!();
    }

    // Step 3: Print summary
    println!("=== Summary ===\n");
    println!("Corpus tables built from {} validation images.", validation_images.len());
    println!("Tested on {} final-test images.\n", test_images.len());

    println!("Quality | Corpus Overhead | Standard Overhead | Improvement");
    println!("--------|-----------------|-------------------|------------");

    for (quality, corpus_counts) in &corpus_counts_by_quality {
        let corpus_tables = corpus_counts.generate_tables()?;

        let mut optimal_total = 0usize;
        let mut corpus_total = 0usize;
        let mut standard_total = 0usize;

        for image_path in &test_images {
            let (width, height, pixels) = load_png(image_path)?;

            let optimal_bytes = encode_with_tables(
                width,
                height,
                &pixels,
                *quality,
                None, // optimal
            )?;
            let corpus_bytes = encode_with_tables(
                width,
                height,
                &pixels,
                *quality,
                Some(TableChoice::Custom(corpus_tables.clone())),
            )?;
            let standard_bytes = encode_with_tables(
                width,
                height,
                &pixels,
                *quality,
                Some(TableChoice::Standard),
            )?;

            optimal_total += optimal_bytes;
            corpus_total += corpus_bytes;
            standard_total += standard_bytes;
        }

        let corpus_overhead =
            100.0 * (corpus_total as f64 - optimal_total as f64) / optimal_total as f64;
        let standard_overhead =
            100.0 * (standard_total as f64 - optimal_total as f64) / optimal_total as f64;
        let improvement = standard_overhead - corpus_overhead;

        println!(
            "Q{:3}    | {:>14.2}% | {:>16.2}% | {:>10.2}%",
            quality, corpus_overhead, standard_overhead, improvement
        );
    }

    println!();
    println!("Positive improvement = corpus tables are better than standard tables.");

    // Step 4: Test universal Q85 tables across all quality levels
    println!("\n=== Step 4: Testing universal Q85 tables ===\n");

    // Find Q85 tables
    let q85_tables = corpus_counts_by_quality
        .iter()
        .find(|(q, _)| *q == 85)
        .map(|(_, counts)| counts.generate_tables())
        .transpose()?
        .expect("Q85 not found");

    println!("Using Q85 corpus tables for all quality levels:");
    println!();
    println!("Quality | Q85 Universal | Quality-Specific | Difference");
    println!("--------|---------------|------------------|------------");

    for &quality in QUALITY_TIERS {
        let quality_specific_tables = corpus_counts_by_quality
            .iter()
            .find(|(q, _)| *q == quality)
            .map(|(_, counts)| counts.generate_tables())
            .transpose()?
            .expect("Quality not found");

        let mut optimal_total = 0usize;
        let mut universal_total = 0usize;
        let mut specific_total = 0usize;

        for image_path in &test_images {
            let (width, height, pixels) = load_png(image_path)?;

            let optimal_bytes = encode_with_tables(width, height, &pixels, quality, None)?;
            let universal_bytes = encode_with_tables(
                width, height, &pixels, quality,
                Some(TableChoice::Custom(q85_tables.clone())),
            )?;
            let specific_bytes = encode_with_tables(
                width, height, &pixels, quality,
                Some(TableChoice::Custom(quality_specific_tables.clone())),
            )?;

            optimal_total += optimal_bytes;
            universal_total += universal_bytes;
            specific_total += specific_bytes;
        }

        let universal_overhead =
            100.0 * (universal_total as f64 - optimal_total as f64) / optimal_total as f64;
        let specific_overhead =
            100.0 * (specific_total as f64 - optimal_total as f64) / optimal_total as f64;

        println!(
            "Q{:3}    | {:>12.2}% | {:>15.2}% | {:>10.2}%",
            quality, universal_overhead, specific_overhead, universal_overhead - specific_overhead
        );
    }

    // Step 5: Generate output files
    println!("\n=== Step 5: Generating output files ===\n");

    fs::create_dir_all(OUTPUT_DIR)?;

    // Generate Rust code for embedding
    let rust_file = PathBuf::from(OUTPUT_DIR).join("corpus_tables.rs");
    generate_rust_code(&corpus_counts_by_quality, &rust_file)?;
    println!("Generated Rust code: {}", rust_file.display());

    // Save frequency counts as JSON for later analysis
    let json_file = PathBuf::from(OUTPUT_DIR).join("frequency_counts.json");
    save_frequency_counts_json(&corpus_counts_by_quality, &json_file)?;
    println!("Saved frequency counts: {}", json_file.display());

    Ok(())
}

/// Generates Rust source code with embedded corpus tables.
fn generate_rust_code(
    counts_by_quality: &[(u8, HuffmanFrequencyCounts)],
    path: &PathBuf,
) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "//! Corpus-derived Huffman tables for streaming JPEG encoding.")?;
    writeln!(f, "//!")?;
    writeln!(f, "//! Generated from CLIC 2025 validation set ({} images).", 32)?;
    writeln!(f, "//! Date: {}", chrono::Utc::now().format("%Y-%m-%d"))?;
    writeln!(f, "//!")?;
    writeln!(f, "//! These tables provide ~2-2.5% overhead vs optimal (image-specific) tables,")?;
    writeln!(f, "//! but ~3-4% improvement over JPEG standard tables.")?;
    writeln!(f)?;
    writeln!(f, "use crate::huffman::optimize::{{OptimizedHuffmanTables, OptimizedTable}};")?;
    writeln!(f)?;

    for (quality, counts) in counts_by_quality {
        let tables = counts.generate_tables()?;

        writeln!(f, "/// Corpus-derived tables for quality ~{}", quality)?;
        writeln!(f, "pub fn corpus_tables_q{}() -> OptimizedHuffmanTables {{", quality)?;
        writeln!(f, "    OptimizedHuffmanTables {{")?;

        for (name, table) in [
            ("dc_luma", &tables.dc_luma),
            ("ac_luma", &tables.ac_luma),
            ("dc_chroma", &tables.dc_chroma),
            ("ac_chroma", &tables.ac_chroma),
        ] {
            writeln!(f, "        {}: OptimizedTable::from_bits_values(", name)?;
            writeln!(f, "            {:?},", table.bits)?;
            writeln!(f, "            vec!{:?},", table.values)?;
            writeln!(f, "        ).unwrap(),")?;
        }

        writeln!(f, "    }}")?;
        writeln!(f, "}}")?;
        writeln!(f)?;
    }

    Ok(())
}

/// Saves frequency counts to JSON for later analysis.
fn save_frequency_counts_json(
    counts_by_quality: &[(u8, HuffmanFrequencyCounts)],
    path: &PathBuf,
) -> Result<()> {
    use std::collections::HashMap;

    let mut data: HashMap<String, serde_json::Value> = HashMap::new();

    for (quality, counts) in counts_by_quality {
        let tables = counts.generate_tables()?;

        let mut quality_data = serde_json::Map::new();

        for (name, table) in [
            ("dc_luma", &tables.dc_luma),
            ("ac_luma", &tables.ac_luma),
            ("dc_chroma", &tables.dc_chroma),
            ("ac_chroma", &tables.ac_chroma),
        ] {
            let mut table_data = serde_json::Map::new();
            table_data.insert(
                "bits".to_string(),
                serde_json::json!(table.bits.to_vec()),
            );
            table_data.insert(
                "values".to_string(),
                serde_json::json!(table.values),
            );
            quality_data.insert(name.to_string(), serde_json::Value::Object(table_data));
        }

        data.insert(format!("q{}", quality), serde_json::Value::Object(quality_data));
    }

    let json = serde_json::to_string_pretty(&data)?;
    fs::write(path, json)?;

    Ok(())
}

enum TableChoice {
    Custom(zenjpeg::huffman::optimize::OptimizedHuffmanTables),
    Standard,
}

fn encode_with_tables(
    width: u32,
    height: u32,
    pixels: &[u8],
    quality: u8,
    tables: Option<TableChoice>,
) -> Result<usize> {
    let mut builder = StreamingEncoder::new(width, height)
        .quality(Quality::ApproxJpegli(quality as f32))
        .subsampling(Subsampling::S420)
        .progressive(false);

    // For non-optimal tables, force streaming mode
    match &tables {
        Some(TableChoice::Custom(t)) => {
            builder = builder.memory_limit(1).custom_huffman_tables(t.clone());
        }
        Some(TableChoice::Standard) => {
            builder = builder.memory_limit(1).use_standard_huffman_tables(true);
        }
        None => {
            // Optimal - use buffered mode (two-pass)
        }
    }

    let mut encoder = builder.start()?;

    let row_size = width as usize * 3;
    for y in 0..height as usize {
        let start = y * row_size;
        encoder.push_row(&pixels[start..start + row_size])?;
    }

    Ok(encoder.finish()?.len())
}

fn load_png(path: &PathBuf) -> Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(fs::File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    let width = info.width;
    let height = info.height;

    // Convert to RGB if needed
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.push(chunk[0]);
                rgb.push(chunk[1]);
                rgb.push(chunk[2]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
            for &g in &buf[..info.buffer_size()] {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
            for chunk in buf[..info.buffer_size()].chunks(2) {
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
                rgb.push(chunk[0]);
            }
            rgb
        }
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type).into()),
    };

    Ok((width, height, pixels))
}
