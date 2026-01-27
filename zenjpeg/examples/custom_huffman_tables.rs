//! Demonstrate custom Huffman tables API.
//!
//! This example shows how to:
//! 1. Extract frequency counts and Huffman tables from encodes
//! 2. Combine counts from multiple images to build "universal" tables
//! 3. Use custom tables for encoding
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example custom_huffman_tables

use zenjpeg::encode::{HuffmanFrequencyCounts, Quality, StreamingEncoder};
use zenjpeg::types::Subsampling;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    // Create some test images (normally you'd load real images)
    let images = vec![
        create_test_image(512, 512, "gradient"),
        create_test_image(512, 512, "noise"),
        create_test_image(512, 512, "edges"),
    ];

    println!("=== Step 1: Extract frequency counts from multiple images ===\n");

    let mut corpus_counts = HuffmanFrequencyCounts::new();

    for (i, (width, height, pixels)) in images.iter().enumerate() {
        let mut encoder = StreamingEncoder::new(*width, *height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .start()?;

        let row_size = *width as usize * 3;
        for y in 0..*height as usize {
            let start = y * row_size;
            encoder.push_row(&pixels[start..start + row_size])?;
        }

        // Get the encoding result with tables and counts
        let result = encoder.finish_with_tables()?;

        println!(
            "Image {}: {} bytes, AC luma entropy: {:.2} bits",
            i + 1,
            result.jpeg.len(),
            result.frequency_counts.ac_luma.entropy()
        );

        // Combine counts into corpus
        corpus_counts.add(&result.frequency_counts);
    }

    println!("\n=== Step 2: Generate tables from combined corpus ===\n");

    let corpus_tables = corpus_counts.generate_tables()?;
    println!(
        "Corpus AC luma entropy: {:.2} bits",
        corpus_counts.ac_luma.entropy()
    );
    println!(
        "Corpus DC luma symbols: {}",
        corpus_counts.dc_luma.num_symbols()
    );
    println!(
        "Corpus AC luma symbols: {}",
        corpus_counts.ac_luma.num_symbols()
    );

    println!("\n=== Step 3: Encode new images with corpus tables ===\n");

    // Encode a new image using the corpus-derived tables
    let (width, height, pixels) = create_test_image(512, 512, "mixed");

    // First, encode normally for baseline
    let baseline_jpeg = {
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .start()?;

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            encoder.push_row(&pixels[start..start + row_size])?;
        }
        encoder.finish()?
    };

    // Now encode with corpus tables
    let corpus_jpeg = {
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .custom_huffman_tables(corpus_tables.clone())
            .start()?;

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            encoder.push_row(&pixels[start..start + row_size])?;
        }
        encoder.finish()?
    };

    let overhead = 100.0 * (corpus_jpeg.len() as f64 - baseline_jpeg.len() as f64)
        / baseline_jpeg.len() as f64;

    println!("Baseline (optimized) size: {} bytes", baseline_jpeg.len());
    println!("Corpus tables size:        {} bytes", corpus_jpeg.len());
    println!("Overhead:                  {:.2}%", overhead);

    println!("\n=== Step 4: Streaming mode with custom tables ===\n");

    // In streaming mode, custom tables avoid the optimization pass
    let streaming_jpeg = {
        let mut encoder = StreamingEncoder::new(width, height)
            .quality(Quality::ApproxJpegli(85.0))
            .subsampling(Subsampling::S420)
            .progressive(false)
            .memory_limit(64 * 1024) // Force streaming mode
            .custom_huffman_tables(corpus_tables)
            .start()?;

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            encoder.push_row(&pixels[start..start + row_size])?;
        }

        let result = encoder.finish_with_tables()?;
        println!(
            "Streaming transition: {}",
            if result.jpeg.len() > 0 { "yes" } else { "no" }
        );
        result.jpeg
    };

    println!("Streaming size: {} bytes", streaming_jpeg.len());

    Ok(())
}

/// Creates a test image with different patterns.
fn create_test_image(width: u32, height: u32, pattern: &str) -> (u32, u32, Vec<u8>) {
    let mut pixels = vec![0u8; width as usize * height as usize * 3];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;

            let (r, g, b) = match pattern {
                "gradient" => {
                    let v = ((x + y) * 255 / (width as usize + height as usize)) as u8;
                    (v, v, v)
                }
                "noise" => {
                    // Simple deterministic "noise"
                    let v = ((x * 17 + y * 31) % 256) as u8;
                    (v, v, v)
                }
                "edges" => {
                    let edge = (x % 32 < 2) || (y % 32 < 2);
                    if edge {
                        (255, 255, 255)
                    } else {
                        (32, 32, 32)
                    }
                }
                "mixed" => {
                    // Mix of all patterns
                    let region = (x / 128 + (y / 128) * 4) % 4;
                    match region {
                        0 => (
                            (x * 255 / width as usize) as u8,
                            (y * 255 / height as usize) as u8,
                            128,
                        ),
                        1 => {
                            let v = ((x * 13 + y * 29) % 256) as u8;
                            (v, v, v)
                        }
                        2 => {
                            let edge = (x % 16 < 2) || (y % 16 < 2);
                            if edge {
                                (200, 200, 200)
                            } else {
                                (50, 50, 50)
                            }
                        }
                        _ => (128, 128, 128),
                    }
                }
                _ => (128, 128, 128),
            };

            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }

    (width, height, pixels)
}
