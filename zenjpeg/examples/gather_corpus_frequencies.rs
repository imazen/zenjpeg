//! Gather raw Huffman symbol frequencies from codec-corpus images.
//!
//! Encodes every PNG in each training corpus at all quality × subsampling ×
//! color-space combinations, collecting the raw symbol frequencies that would
//! be used to build optimized Huffman tables.
//!
//! Run with:
//!   cargo run --release -p zenjpeg --example gather_corpus_frequencies
//!
//! Output goes to /mnt/v/output/zenjpeg/huffman-freq/

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, HuffmanSymbolFrequencies, PixelLayout, XybSubsampling,
};
use zenjpeg::huffman::optimize::FrequencyCounter;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn corpus_crate() -> std::result::Result<codec_corpus::Corpus, Box<dyn std::error::Error>> {
    Ok(codec_corpus::Corpus::new()?)
}
const OUTPUT_DIR: &str = "/mnt/v/output/zenjpeg/huffman-freq";

// Q0-Q85 step 5, Q89-Q100 step 1
const QUALITY_TIERS: &[u8] = &[
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 89, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99, 100,
];

/// A training corpus from codec-corpus.
struct Corpus {
    name: &'static str,
    /// Path relative to CODEC_CORPUS root.
    rel_path: &'static str,
}

const CORPORA: &[Corpus] = &[
    Corpus {
        name: "clic2025-training",
        rel_path: "clic2025/training",
    },
    Corpus {
        name: "CID22-training",
        rel_path: "CID22/CID22-512/training",
    },
    Corpus {
        name: "gb82",
        rel_path: "gb82",
    },
    Corpus {
        name: "gb82-sc",
        rel_path: "gb82-sc",
    },
    Corpus {
        name: "kadid10k",
        rel_path: "kadid10k",
    },
    Corpus {
        name: "kodak-legacy",
        rel_path: "kodak-legacy",
    },
    Corpus {
        name: "qoi-screenshot-web",
        rel_path: "qoi-benchmark/screenshot_web",
    },
];

/// An encoding mode combining color space and subsampling.
#[derive(Clone, Copy)]
enum Mode {
    Ycbcr444,
    Ycbcr422,
    Ycbcr420,
    XybFull,
    XybBquarter,
}

impl Mode {
    const ALL: &[Mode] = &[
        Mode::Ycbcr444,
        Mode::Ycbcr422,
        Mode::Ycbcr420,
        Mode::XybFull,
        Mode::XybBquarter,
    ];

    fn dir_name(self) -> &'static str {
        match self {
            Mode::Ycbcr444 => "ycbcr-444",
            Mode::Ycbcr422 => "ycbcr-422",
            Mode::Ycbcr420 => "ycbcr-420",
            Mode::XybFull => "xyb-full",
            Mode::XybBquarter => "xyb-bquarter",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::Ycbcr444 => "YCbCr 4:4:4",
            Mode::Ycbcr422 => "YCbCr 4:2:2",
            Mode::Ycbcr420 => "YCbCr 4:2:0",
            Mode::XybFull => "XYB Full",
            Mode::XybBquarter => "XYB BQuarter",
        }
    }

    fn make_config(self, quality: u8) -> EncoderConfig {
        match self {
            Mode::Ycbcr444 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::None).progressive(false)
            }
            Mode::Ycbcr422 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::HalfHorizontal)
                    .progressive(false)
            }
            Mode::Ycbcr420 => {
                EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter).progressive(false)
            }
            Mode::XybFull => {
                EncoderConfig::xyb(quality as f32, XybSubsampling::Full).progressive(false)
            }
            Mode::XybBquarter => {
                EncoderConfig::xyb(quality as f32, XybSubsampling::BQuarter).progressive(false)
            }
        }
    }
}

/// Per-quality frequency data plus total JPEG bytes.
struct QualityFrequencies {
    quality: u8,
    frequencies: Box<HuffmanSymbolFrequencies>,
    total_jpeg_bytes: usize,
}

/// Per-mode frequency data for a single corpus.
struct ModeResult {
    image_count: usize,
    per_quality: Vec<QualityFrequencies>,
}

fn main() -> Result<()> {
    let start = Instant::now();

    // Initialize corpus
    let cc = match corpus_crate() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Codec corpus not available: {e}");
            std::process::exit(1);
        }
    };

    // Discover corpora and count images
    let mut corpus_images: Vec<(&Corpus, Vec<PathBuf>)> = Vec::new();
    let mut total_images = 0;
    for corpus in CORPORA {
        let dir = match cc.get(corpus.rel_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "WARNING: Corpus '{}' not available: {}",
                    corpus.name,
                    e
                );
                continue;
            }
        };
        let images = load_image_list(&dir)?;
        if images.is_empty() {
            eprintln!("WARNING: No PNG images in {}", dir.display());
            continue;
        }
        total_images += images.len();
        corpus_images.push((corpus, images));
    }

    let total_encodes = total_images * QUALITY_TIERS.len() * Mode::ALL.len();
    println!("=== Corpus Huffman Frequency Gatherer ===\n");
    println!("Corpora: {}", corpus_images.len());
    println!("Total images: {total_images}");
    println!("Quality tiers: {}", QUALITY_TIERS.len());
    println!("Modes: {}", Mode::ALL.len());
    println!("Total encodes: {total_encodes}");
    println!("Output: {OUTPUT_DIR}\n");

    for (corpus, images) in &corpus_images {
        println!("  {:<25} {:>4} images", corpus.name, images.len());
    }
    println!();

    fs::create_dir_all(OUTPUT_DIR)?;

    // Per-mode aggregated frequencies (across all corpora)
    let mut aggregated: Vec<Option<Vec<QualityFrequencies>>> =
        (0..Mode::ALL.len()).map(|_| None).collect();

    let mut encodes_done = 0usize;

    for (corpus, images) in &corpus_images {
        println!("--- {} ({} images) ---\n", corpus.name, images.len());

        for (mode_idx, &mode) in Mode::ALL.iter().enumerate() {
            let mode_result = process_corpus_mode(images, mode, &mut encodes_done)?;

            // Save per-corpus frequencies
            let corpus_dir = PathBuf::from(OUTPUT_DIR)
                .join(corpus.name)
                .join(mode.dir_name());
            fs::create_dir_all(&corpus_dir)?;
            save_frequencies_json(
                &corpus_dir.join("raw_frequencies.json"),
                corpus.name,
                mode.dir_name(),
                mode_result.image_count,
                &mode_result.per_quality,
            )?;

            // Aggregate into cross-corpus totals
            aggregate_into(&mut aggregated[mode_idx], &mode_result.per_quality);
        }
        println!();
    }

    // Save aggregated frequencies
    println!("--- Saving aggregated frequencies ---\n");
    for (mode_idx, &mode) in Mode::ALL.iter().enumerate() {
        if let Some(ref agg) = aggregated[mode_idx] {
            let agg_dir = PathBuf::from(OUTPUT_DIR)
                .join("aggregated")
                .join(mode.dir_name());
            fs::create_dir_all(&agg_dir)?;
            save_frequencies_json(
                &agg_dir.join("raw_frequencies.json"),
                "aggregated",
                mode.dir_name(),
                total_images,
                agg,
            )?;
            println!("  {}", agg_dir.join("raw_frequencies.json").display());
        }
    }

    // Write manifest
    let manifest = serde_json::json!({
        "generated": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "quality_tiers": QUALITY_TIERS,
        "modes": Mode::ALL.iter().map(|m| m.dir_name()).collect::<Vec<_>>(),
        "corpora": corpus_images.iter().map(|(c, imgs)| {
            serde_json::json!({
                "name": c.name,
                "path": c.rel_path,
                "image_count": imgs.len(),
            })
        }).collect::<Vec<_>>(),
        "total_images": total_images,
        "total_encodes": total_encodes,
        "elapsed_seconds": start.elapsed().as_secs_f64(),
    });
    fs::write(
        PathBuf::from(OUTPUT_DIR).join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    println!(
        "\nDone: {} encodes in {:.1}s",
        encodes_done,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Process one corpus × one mode, returning per-quality frequency data.
fn process_corpus_mode(
    images: &[PathBuf],
    mode: Mode,
    encodes_done: &mut usize,
) -> Result<ModeResult> {
    let mode_start = Instant::now();
    print!("  {:<18}", mode.label());
    std::io::stdout().flush()?;

    let mut per_quality = Vec::with_capacity(QUALITY_TIERS.len());

    for &quality in QUALITY_TIERS {
        let mut aggregate: Option<Box<HuffmanSymbolFrequencies>> = None;
        let mut total_bytes = 0usize;

        for path in images {
            let (w, h, pixels) = load_png(path)?;
            let (jpeg_len, counts) = encode_with_frequencies(w, h, &pixels, mode, quality)?;
            total_bytes += jpeg_len;

            match &mut aggregate {
                Some(agg) => agg.add(&counts),
                None => aggregate = Some(counts),
            }
        }

        per_quality.push(QualityFrequencies {
            quality,
            frequencies: aggregate.expect("no images"),
            total_jpeg_bytes: total_bytes,
        });

        *encodes_done += images.len();
    }

    println!(
        " {:>6} encodes in {:.1}s  ({:.0}/s)",
        images.len() * QUALITY_TIERS.len(),
        mode_start.elapsed().as_secs_f64(),
        images.len() as f64 * QUALITY_TIERS.len() as f64 / mode_start.elapsed().as_secs_f64(),
    );

    Ok(ModeResult {
        image_count: images.len(),
        per_quality,
    })
}

/// Encode a single image and return (jpeg_size, frequency_counts).
fn encode_with_frequencies(
    w: u32,
    h: u32,
    pixels: &[u8],
    mode: Mode,
    quality: u8,
) -> Result<(usize, Box<HuffmanSymbolFrequencies>)> {
    let config = mode.make_config(quality);
    let mut enc = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(pixels, Unstoppable)?;
    let (jpeg, counts) = enc.finish_with_huffman_frequencies()?;
    let counts = counts.expect("optimize_huffman is on by default");
    Ok((jpeg.len(), counts))
}

/// Aggregate frequency data from one corpus-mode into the running total.
fn aggregate_into(target: &mut Option<Vec<QualityFrequencies>>, source: &[QualityFrequencies]) {
    match target {
        None => {
            // Clone the source as our starting point
            *target = Some(
                source
                    .iter()
                    .map(|qf| QualityFrequencies {
                        quality: qf.quality,
                        frequencies: qf.frequencies.clone(),
                        total_jpeg_bytes: qf.total_jpeg_bytes,
                    })
                    .collect(),
            );
        }
        Some(existing) => {
            for (e, s) in existing.iter_mut().zip(source.iter()) {
                debug_assert_eq!(e.quality, s.quality);
                e.frequencies.add(&s.frequencies);
                e.total_jpeg_bytes += s.total_jpeg_bytes;
            }
        }
    }
}

// --- I/O ---

fn load_image_list(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn load_png(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(fs::File::open(path)?);
    // Expand indexed/palette to RGB(A), low-bit-depth gray to 8-bit
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info()?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    let (w, h) = (info.width, info.height);
    // After EXPAND, indexed becomes RGB or RGBA
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        other => return Err(format!("Unsupported color type: {other:?}").into()),
    };

    Ok((w, h, pixels))
}

fn counter_to_vec(counter: &FrequencyCounter) -> Vec<i64> {
    (0..=255).map(|i| counter.get_count(i)).collect()
}

fn save_frequencies_json(
    path: &Path,
    corpus_name: &str,
    subsampling: &str,
    image_count: usize,
    data: &[QualityFrequencies],
) -> Result<()> {
    let mut map = serde_json::Map::new();
    map.insert(
        "metadata".into(),
        serde_json::json!({
            "corpus": corpus_name,
            "subsampling": subsampling,
            "image_count": image_count,
        }),
    );

    for qf in data {
        let qdata = serde_json::json!({
            "dc_luma": counter_to_vec(&qf.frequencies.dc_luma),
            "ac_luma": counter_to_vec(&qf.frequencies.ac_luma),
            "dc_chroma": counter_to_vec(&qf.frequencies.dc_chroma),
            "ac_chroma": counter_to_vec(&qf.frequencies.ac_chroma),
            "total_jpeg_bytes": qf.total_jpeg_bytes,
        });
        map.insert(format!("q{}", qf.quality), qdata);
    }

    let mut f = File::create(path)?;
    f.write_all(serde_json::to_string_pretty(&serde_json::Value::Object(map))?.as_bytes())?;
    Ok(())
}
