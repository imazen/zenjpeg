//! Build corpus-based Huffman tables from a set of images,
//! then validate against per-image optimal and standard JPEG tables.
//!
//! ## Table Types Compared
//!
//! - **Optimal**: Per-image optimized tables (two-pass, what jpegli does by default)
//! - **Corpus**: Tables trained on a corpus of images (our new tables)
//! - **Standard**: JPEG Annex K tables (libjpeg default, jpegli with optimize=false)
//!
//! Run with:
//!   cargo run --release -p zenjpeg --example build_corpus_tables -- <image_dir> [image_dir2 ...]
//!
//! Options:
//!   --quality <n>       Single quality level (default: sweep standard tiers)
//!   --output <dir>      Output directory (default: /mnt/v/output/zenjpeg/corpus_tables)
//!   --validation <dir>  Separate validation image directory
//!
//! Output files are written to the output directory.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, HuffmanSymbolFrequencies, PixelLayout};
use zenjpeg::huffman::optimize::{FrequencyCounter, HuffmanTableSet};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Q0-Q85 step 5, Q89-Q100 step 1
const QUALITY_TIERS: &[u8] = &[
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 89, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99, 100,
];

fn default_output_dir() -> std::path::PathBuf {
    zenjpeg_bench_utils::zenjpeg_output_dir().join("corpus_tables")
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let training_images = load_images_from_dirs(&args.training_dirs)?;
    if training_images.is_empty() {
        eprintln!("No PNG images found in training directories.");
        eprintln!(
            "Usage: build_corpus_tables <image_dir> [image_dir2 ...] [--validation <dir>] [--quality <n>]"
        );
        std::process::exit(1);
    }

    let validation_images = if !args.validation_dirs.is_empty() {
        load_images_from_dirs(&args.validation_dirs)?
    } else {
        Vec::new()
    };

    println!("=== Corpus Huffman Table Builder ===\n");
    println!("Training: {} images", training_images.len());
    if !validation_images.is_empty() {
        println!("Validation: {} images", validation_images.len());
    }
    println!("Quality tiers: {:?}", args.qualities);
    println!("Output: {}", args.output_dir);
    println!();
    println!("Table types:");
    println!("  Optimal  = Per-image optimized (two-pass, jpegli default)");
    println!("  Corpus   = Trained on corpus (our tables)");
    println!("  Standard = JPEG Annex K (libjpeg/jpegli fixed tables)");
    println!();

    // Step 1: Build corpus tables from training set
    println!("=== Step 1: Building corpus tables ===\n");
    let corpus_counts = build_corpus_tables(&training_images, &args.qualities)?;

    // Step 2: Validate on training set (overfit check) and validation set
    println!("\n=== Step 2: Validation Results ===\n");

    let mut test_sets: Vec<(&str, &[PathBuf])> = Vec::new();
    test_sets.push(("Training (overfit check)", &training_images));
    if !validation_images.is_empty() {
        test_sets.push(("Validation", &validation_images));
    }

    for (name, images) in &test_sets {
        println!("--- {} ({} images) ---\n", name, images.len());
        println!(
            "{:>7} | {:>10} | {:>10} | {:>11}",
            "Quality", "Corpus %", "Standard %", "Improvement"
        );
        println!("{}", "-".repeat(50));

        for (quality, counts) in &corpus_counts {
            let tables = counts.generate_tables()?;
            let stats = validate_corpus(images, *quality, &tables)?;

            println!(
                "Q{:<5}  | {:>9.2}% | {:>9.2}% | {:>10.2}%",
                quality,
                stats.corpus_overhead,
                stats.standard_overhead,
                stats.standard_overhead - stats.corpus_overhead
            );
        }
        println!();
    }

    // Step 3: Universal Q85 tables test (if we have Q85 in our tiers)
    if let Some((_, q85_counts)) = corpus_counts.iter().find(|(q, _)| *q == 85) {
        let q85_tables = q85_counts.generate_tables()?;
        let eval_images = if !validation_images.is_empty() {
            &validation_images
        } else {
            &training_images
        };

        println!("=== Step 3: Universal Q85 Tables ===\n");
        println!(
            "{:>7} | {:>14} | {:>16} | {:>6}",
            "Quality", "Q85 Universal", "Quality-Specific", "Delta"
        );
        println!("{}", "-".repeat(55));

        for &quality in &args.qualities {
            let specific = corpus_counts
                .iter()
                .find(|(q, _)| *q == quality)
                .map(|(_, c)| c.generate_tables())
                .transpose()?
                .expect("quality tier missing");

            let uni_stats = validate_corpus(eval_images, quality, &q85_tables)?;
            let spec_stats = validate_corpus(eval_images, quality, &specific)?;

            println!(
                "Q{:<5}  | {:>13.2}% | {:>15.2}% | {:>5.2}%",
                quality,
                uni_stats.corpus_overhead,
                spec_stats.corpus_overhead,
                uni_stats.corpus_overhead - spec_stats.corpus_overhead
            );
        }
        println!();
    }

    // Step 4: Generate output files
    println!("=== Step 4: Output Files ===\n");
    fs::create_dir_all(&args.output_dir)?;

    let rust_file = PathBuf::from(&args.output_dir).join("corpus_tables.rs");
    generate_rust_code(&corpus_counts, &rust_file)?;
    println!("Rust code: {}", rust_file.display());

    let json_file = PathBuf::from(&args.output_dir).join("huffman_tables.json");
    save_json(&corpus_counts, &json_file)?;
    println!("Tables:    {}", json_file.display());

    let raw_file = PathBuf::from(&args.output_dir).join("raw_frequencies.json");
    save_raw_frequencies(&corpus_counts, &raw_file)?;
    println!("Raw freq:  {}", raw_file.display());

    Ok(())
}

// --- Args ---

struct Args {
    training_dirs: Vec<String>,
    validation_dirs: Vec<String>,
    qualities: Vec<u8>,
    output_dir: String,
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    let mut training_dirs = Vec::new();
    let mut validation_dirs = Vec::new();
    let mut qualities = Vec::new();
    let mut output_dir = default_output_dir().display().to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--quality" => {
                i += 1;
                if i < args.len() {
                    qualities.push(args[i].parse()?);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_dir = args[i].clone();
                }
            }
            "--validation" => {
                i += 1;
                if i < args.len() {
                    validation_dirs.push(args[i].clone());
                }
            }
            arg if !arg.starts_with('-') => {
                training_dirs.push(arg.to_string());
            }
            _ => eprintln!("Unknown argument: {}", args[i]),
        }
        i += 1;
    }

    if qualities.is_empty() {
        qualities = QUALITY_TIERS.to_vec();
    }

    Ok(Args {
        training_dirs,
        validation_dirs,
        qualities,
        output_dir,
    })
}

// --- Helper types ---

struct ValidationStats {
    corpus_overhead: f64,
    standard_overhead: f64,
}

// --- Core functions ---

fn build_corpus_tables(
    images: &[PathBuf],
    qualities: &[u8],
) -> Result<Vec<(u8, Box<HuffmanSymbolFrequencies>)>> {
    let mut results = Vec::new();

    for &quality in qualities {
        print!("Q{:>3}: ", quality);
        std::io::stdout().flush()?;
        let start = Instant::now();
        let mut aggregate: Option<Box<HuffmanSymbolFrequencies>> = None;
        let mut total_bytes = 0usize;

        for (i, path) in images.iter().enumerate() {
            let (w, h, pixels) = load_png_rgb(path)?;
            let (jpeg_len, counts) = encode_optimal(w, h, &pixels, quality)?;
            total_bytes += jpeg_len;

            match &mut aggregate {
                Some(agg) => agg.add(&counts),
                None => aggregate = Some(counts),
            }

            if (i + 1) % 10 == 0 {
                print!(".");
                std::io::stdout().flush()?;
            }
        }

        let aggregate = aggregate.expect("no images processed");
        println!(
            " {} KB avg, {:.1}s",
            total_bytes / images.len() / 1024,
            start.elapsed().as_secs_f64(),
        );

        results.push((quality, aggregate));
    }

    Ok(results)
}

fn validate_corpus(
    images: &[PathBuf],
    quality: u8,
    tables: &HuffmanTableSet,
) -> Result<ValidationStats> {
    let mut optimal_total = 0usize;
    let mut corpus_total = 0usize;
    let mut standard_total = 0usize;

    for path in images {
        let (w, h, pixels) = load_png_rgb(path)?;
        let (opt_len, _) = encode_optimal(w, h, &pixels, quality)?;
        optimal_total += opt_len;
        corpus_total += encode_with_tables(w, h, &pixels, quality, tables)?;
        standard_total += encode_standard(w, h, &pixels, quality)?;
    }

    Ok(ValidationStats {
        corpus_overhead: pct_overhead(corpus_total, optimal_total),
        standard_overhead: pct_overhead(standard_total, optimal_total),
    })
}

fn pct_overhead(actual: usize, baseline: usize) -> f64 {
    100.0 * (actual as f64 - baseline as f64) / baseline as f64
}

// --- Encoding functions ---

fn base_config(quality: u8) -> EncoderConfig {
    EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter).progressive(false)
}

fn encode_optimal(
    w: u32,
    h: u32,
    pixels: &[u8],
    quality: u8,
) -> Result<(usize, Box<HuffmanSymbolFrequencies>)> {
    let config = base_config(quality);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    let (jpeg, counts) = enc.finish_with_huffman_frequencies()?;
    let counts = counts.expect("optimize_huffman is on by default");
    Ok((jpeg.len(), counts))
}

fn encode_with_tables(
    w: u32,
    h: u32,
    pixels: &[u8],
    quality: u8,
    tables: &HuffmanTableSet,
) -> Result<usize> {
    let config = base_config(quality).custom_huffman_tables(tables.clone());
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    Ok(enc.finish()?.len())
}

fn encode_standard(w: u32, h: u32, pixels: &[u8], quality: u8) -> Result<usize> {
    let config = base_config(quality).optimize_huffman(false);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    Ok(enc.finish()?.len())
}

// --- I/O functions ---

fn load_images_from_dirs(dirs: &[String]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for dir in dirs {
        paths.extend(load_image_list(dir)?);
    }
    paths.sort();
    Ok(paths)
}

fn load_image_list(dir: &str) -> Result<Vec<PathBuf>> {
    Ok(fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "png" || ext == "PNG")
        })
        .collect())
}

fn load_png_rgb(path: &std::path::Path) -> Result<(u32, u32, Vec<u8>)> {
    let img = zenjpeg_bench_utils::load_png(path)
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{e}").into() })?;
    let w = img.width() as u32;
    let h = img.height() as u32;
    let bytes: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Ok((w, h, bytes))
}

// --- Output generation ---

fn generate_rust_code(
    counts: &[(u8, Box<HuffmanSymbolFrequencies>)],
    path: &PathBuf,
) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "//! Corpus-derived Huffman tables.")?;
    writeln!(
        f,
        "//! Generated: {}",
        chrono::Utc::now().format("%Y-%m-%d")
    )?;
    writeln!(f)?;
    writeln!(
        f,
        "use crate::huffman::optimize::{{HuffmanTableSet, OptimizedTable}};"
    )?;
    writeln!(f)?;

    for (quality, c) in counts {
        let t = c.generate_tables()?;
        writeln!(
            f,
            "pub fn corpus_tables_q{}() -> HuffmanTableSet {{",
            quality
        )?;
        writeln!(f, "    HuffmanTableSet {{")?;

        for (name, table) in [
            ("dc_luma", &t.dc_luma),
            ("ac_luma", &t.ac_luma),
            ("dc_chroma", &t.dc_chroma),
            ("ac_chroma", &t.ac_chroma),
        ] {
            writeln!(
                f,
                "        {}: OptimizedTable::from_bits_values({:?}, vec!{:?}).unwrap(),",
                name, table.bits, table.values
            )?;
        }

        writeln!(f, "    }}")?;
        writeln!(f, "}}")?;
        writeln!(f)?;
    }

    Ok(())
}

fn save_json(counts: &[(u8, Box<HuffmanSymbolFrequencies>)], path: &PathBuf) -> Result<()> {
    let mut data = serde_json::Map::new();

    for (quality, c) in counts {
        let t = c.generate_tables()?;
        let mut qdata = serde_json::Map::new();

        for (name, table) in [
            ("dc_luma", &t.dc_luma),
            ("ac_luma", &t.ac_luma),
            ("dc_chroma", &t.dc_chroma),
            ("ac_chroma", &t.ac_chroma),
        ] {
            qdata.insert(
                name.into(),
                serde_json::json!({
                    "bits": table.bits.to_vec(),
                    "values": table.values,
                }),
            );
        }

        data.insert(format!("q{}", quality), serde_json::Value::Object(qdata));
    }

    fs::write(path, serde_json::to_string_pretty(&data)?)?;
    Ok(())
}

fn save_raw_frequencies(
    counts: &[(u8, Box<HuffmanSymbolFrequencies>)],
    path: &PathBuf,
) -> Result<()> {
    fn counter_to_vec(counter: &FrequencyCounter) -> Vec<i64> {
        (0..=255).map(|i| counter.get_count(i)).collect()
    }

    let mut data = serde_json::Map::new();

    for (quality, c) in counts {
        let qdata = serde_json::json!({
            "dc_luma": counter_to_vec(&c.dc_luma),
            "ac_luma": counter_to_vec(&c.ac_luma),
            "dc_chroma": counter_to_vec(&c.dc_chroma),
            "ac_chroma": counter_to_vec(&c.ac_chroma),
        });
        data.insert(format!("q{}", quality), qdata);
    }

    fs::write(path, serde_json::to_string_pretty(&data)?)?;
    Ok(())
}
