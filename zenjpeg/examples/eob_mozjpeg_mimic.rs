//! Test EOB optimization in mozjpeg mimic mode.
//!
//! Run: cargo run --release -p zenjpeg --features mozjpeg-tables --example eob_mozjpeg_mimic

use std::io::Write;
use std::path::Path;
use std::process::Command;

use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, MozjpegTables, PixelLayout, QuantTablePreset,
};
use zenjpeg::encode::mozjpeg_compat::TrellisConfig;
use zenjpeg::hybrid::config::HybridConfig;
// EOB optimization functions (kept for future use)
#[allow(unused_imports)]
use zenjpeg::trellis::eob::{estimate_block_eob_info, optimize_eob_runs};
#[allow(unused_imports)]
use zenjpeg::trellis::rate::RateTable;
use enough::Unstoppable;

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
    if !Path::new(CJPEG_PATH).exists() {
        eprintln!("C cjpeg not found at {}", CJPEG_PATH);
        return Ok(());
    }

    let test_paths = [
        "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/5.png",
        "/home/lilith/work/codec-eval/codec-corpus/kodak/13.png",
    ];

    println!("Testing in TRUE mozjpeg mimic mode");
    println!("All use: Robidoux tables, 4:2:0, baseline, optimized Huffman");
    println!("zenjpeg: NO AQ, NO deringing (pure mozjpeg-style encoding)\n");

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

        println!("=== {} ({}x{}) ===", path, width, height);
        println!("{:>3}  {:>8} {:>8} {:>8} {:>8}  {:>7} {:>7}",
                 "Q", "cmoz", "cmoz+tr", "zen", "zen+tr", "Δ_base", "Δ_trel");
        println!("{}", "-".repeat(75));

        for quality in [50, 75, 90] {
            // C mozjpeg baseline (no trellis)
            let cmoz_base = encode_c_mozjpeg(ppm_path, quality, false)
                .map(|v| v.len())
                .unwrap_or(0);

            // C mozjpeg with trellis
            let cmoz_trel = encode_c_mozjpeg(ppm_path, quality, true)
                .map(|v| v.len())
                .unwrap_or(0);

            // zenjpeg in TRUE mozjpeg mimic mode:
            // - Robidoux tables
            // - NO jpegli AQ (HybridConfig::disabled)
            // - NO deringing
            // - NO trellis (first test)
            let tables = MozjpegTables::generate_ex(quality, QuantTablePreset::Robidoux, true);
            let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                .progressive(false)
                .tables(tables.clone())
                .allow_16bit_quant_tables(false)
                .deringing(false)
                .hybrid_config(HybridConfig::disabled());

            let mut encoder = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
            encoder.push(&pixels, height as usize, width as usize * 3, Unstoppable)?;
            let zen = encoder.finish()?.len();

            // zenjpeg with trellis (AC + DC), still no AQ/deringing
            let config_trel = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                .progressive(false)
                .tables(tables)
                .allow_16bit_quant_tables(false)
                .deringing(false)
                .hybrid_config(HybridConfig::disabled())
                .trellis(TrellisConfig::default().ac_trellis(true).dc_trellis(true));

            let mut encoder = config_trel.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
            encoder.push(&pixels, height as usize, width as usize * 3, Unstoppable)?;
            let zen_trel = encoder.finish()?.len();

            // Compare no-trellis: zen vs cmoz_base
            let delta_base = if cmoz_base > 0 {
                ((zen as f64 - cmoz_base as f64) / cmoz_base as f64) * 100.0
            } else { 0.0 };

            // Compare trellis: zen_trel vs cmoz_trel
            let delta_trel = if cmoz_trel > 0 {
                ((zen_trel as f64 - cmoz_trel as f64) / cmoz_trel as f64) * 100.0
            } else { 0.0 };

            println!("{:>3}  {:>8} {:>8} {:>8} {:>8}  {:>+6.2}% {:>+6.2}%",
                     quality, cmoz_base, cmoz_trel, zen, zen_trel, delta_base, delta_trel);
        }
        println!();
    }

    Ok(())
}
