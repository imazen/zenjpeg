//! Test EOB optimization in mozjpeg mimic mode - TRUE A/B TEST.
//!
//! This test compares:
//! - zen+trellis (baseline)
//! - zen+trellis+eob (with EOB optimization enabled)
//!
//! This is a TRUE A/B test of EOB optimization, not a comparison against C mozjpeg.
//!
//! Run: cargo run --release -p zenjpeg --features mozjpeg-tables --example eob_mozjpeg_mimic

use std::io::Write;
use std::path::Path;
use std::process::Command;

use enough::Unstoppable;
use zenjpeg::encode::mozjpeg_compat::TrellisConfig;
use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, MozjpegTables, PixelLayout, QuantTablePreset,
};
use zenjpeg::hybrid::config::HybridConfig;

const CJPEG_PATH: &str = "/home/lilith/work/mozjpeg/build/cjpeg";

fn load_png(path: &str) -> Option<(u32, u32, Vec<u8>)> {
    let data = std::fs::read(path).ok()?;
    let decoder = png::Decoder::new(std::io::Cursor::new(&data));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((info.buffer_size() / 4) * 3);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return None,
    };

    Some((info.width, info.height, rgb))
}

fn write_ppm(path: &Path, rgb: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

fn encode_c_mozjpeg(ppm_path: &Path, quality: u8, trellis: bool) -> Option<Vec<u8>> {
    let out_path = format!("/tmp/cmoz_mimic_{}_{}.jpg", std::process::id(), quality);
    let mut cmd = Command::new(CJPEG_PATH);
    cmd.args(["-quality", &quality.to_string()]);
    cmd.args(["-sample", "2x2"]); // 4:2:0
    cmd.args(["-quant-table", "3"]); // Robidoux (index 3 in mozjpeg)
    cmd.arg("-optimize");
    cmd.arg("-baseline");
    cmd.arg("-quant-baseline"); // Clamp to 255
    if !trellis {
        cmd.arg("-notrellis");
        cmd.arg("-notrellis-dc");
    }
    cmd.args(["-outfile", &out_path]);
    cmd.arg(ppm_path);

    let status = cmd.status().ok()?;
    if !status.success() {
        return None;
    }

    std::fs::read(&out_path).ok()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_paths = [
        "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/5.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/13.png",
    ];

    println!("=============================================================");
    println!("       TRUE A/B TEST: EOB Optimization in mozjpeg-mimic mode");
    println!("=============================================================");
    println!();
    println!("Configuration:");
    println!("  - Robidoux quant tables (mozjpeg style)");
    println!("  - 4:2:0 chroma subsampling");
    println!("  - Baseline JPEG (sequential)");
    println!("  - Optimized Huffman tables");
    println!("  - NO jpegli AQ (HybridConfig::disabled)");
    println!("  - NO deringing");
    println!();
    println!("Comparing:");
    println!("  - zen+tr     = zenjpeg with AC+DC trellis, NO EOB optimization");
    println!("  - zen+tr+eob = zenjpeg with AC+DC trellis + EOB optimization");
    println!("  - Δ_eob      = size difference (negative = EOB helps)");
    println!();

    // Also show C mozjpeg for reference if available
    let has_cmozjpeg = Path::new(CJPEG_PATH).exists();
    if has_cmozjpeg {
        println!("Also showing C mozjpeg for reference.");
    } else {
        println!("C mozjpeg not found at {} - skipping reference.", CJPEG_PATH);
    }
    println!();

    for path in &test_paths {
        if !Path::new(path).exists() {
            eprintln!("Skipping {}: not found", path);
            continue;
        }

        let (width, height, pixels) = match load_png(path) {
            Some(data) => data,
            None => continue,
        };

        let ppm_path = Path::new("/tmp/eob_mimic_test.ppm");
        write_ppm(ppm_path, &pixels, width, height)?;

        let filename = Path::new(path).file_name().unwrap().to_str().unwrap();
        println!("=== {} ({}x{}) ===", filename, width, height);

        if has_cmozjpeg {
            println!(
                "{:>3}  {:>8} {:>8} {:>10}  {:>8}",
                "Q", "cmoz+tr", "zen+tr", "zen+tr+eob", "Δ_eob"
            );
        } else {
            println!(
                "{:>3}  {:>8} {:>10}  {:>8}",
                "Q", "zen+tr", "zen+tr+eob", "Δ_eob"
            );
        }
        println!("{}", "-".repeat(55));

        for quality in [50, 75, 90] {
            // C mozjpeg with trellis (for reference)
            let cmoz_trel = if has_cmozjpeg {
                encode_c_mozjpeg(ppm_path, quality, true)
                    .map(|v| v.len())
                    .unwrap_or(0)
            } else {
                0
            };

            // zenjpeg in mozjpeg mimic mode with trellis (NO EOB)
            let tables =
                MozjpegTables::generate_ex(quality, QuantTablePreset::Robidoux, true);
            let config_trel = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                .progressive(false)
                .tables(tables.clone())
                .allow_16bit_quant_tables(false)
                .deringing(false)
                .hybrid_config(HybridConfig::disabled())
                .trellis(TrellisConfig::default().ac_trellis(true).dc_trellis(true));

            let mut encoder =
                config_trel.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
            encoder.push(&pixels, height as usize, width as usize * 3, Unstoppable)?;
            let zen_trel = encoder.finish()?.len();

            // zenjpeg with trellis + EOB optimization (THE KEY TEST)
            let config_trel_eob = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                .progressive(false)
                .tables(tables)
                .allow_16bit_quant_tables(false)
                .deringing(false)
                .hybrid_config(HybridConfig::disabled())
                .trellis(
                    TrellisConfig::default()
                        .ac_trellis(true)
                        .dc_trellis(true)
                        .eob_optimization(true), // <-- THE DIFFERENCE
                );

            let mut encoder =
                config_trel_eob.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
            encoder.push(&pixels, height as usize, width as usize * 3, Unstoppable)?;
            let jpeg_eob = encoder.finish()?;
            let zen_trel_eob = jpeg_eob.len();

            // Save files for inspection at Q75
            if quality == 75 {
                std::fs::write(format!("/tmp/zen_tr_{}.jpg", filename), &{
                    let mut enc = config_trel.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
                    enc.push(&pixels, height as usize, width as usize * 3, Unstoppable)?;
                    enc.finish()?
                })?;
                std::fs::write(format!("/tmp/zen_tr_eob_{}.jpg", filename), &jpeg_eob)?;
            }

            // A/B comparison: zen+trellis vs zen+trellis+eob
            let delta_eob = if zen_trel > 0 {
                ((zen_trel_eob as f64 - zen_trel as f64) / zen_trel as f64) * 100.0
            } else {
                0.0
            };

            if has_cmozjpeg {
                println!(
                    "{:>3}  {:>8} {:>8} {:>10}  {:>+7.2}%",
                    quality, cmoz_trel, zen_trel, zen_trel_eob, delta_eob
                );
            } else {
                println!(
                    "{:>3}  {:>8} {:>10}  {:>+7.2}%",
                    quality, zen_trel, zen_trel_eob, delta_eob
                );
            }
        }
        println!();
    }

    println!("=============================================================");
    println!("INTERPRETATION:");
    println!("  Δ_eob < 0  => EOB optimization HELPS (smaller files)");
    println!("  Δ_eob ≈ 0  => EOB optimization has no effect");
    println!("  Δ_eob > 0  => EOB optimization HURTS (shouldn't happen)");
    println!("=============================================================");

    Ok(())
}
