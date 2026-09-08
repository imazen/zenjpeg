//! Recovered July 18 Zq probe, revised September 8 for complete candidate serving.
//!
//! Requires ZENJPEG_ZQ_BAKE, ZENJPEG_ZQ_SEED_Q, ZENJPEG_ZQ_SPATIAL and explicit
//! ZENSIM_FORMULA_REV=1. Optional fresh ZENJPEG_ZQ_TRACE_DIR records every pass.
//! Example: zq_rd_probe --image source.png --target 80 --corrections 2 --out-dir fresh
//! Sources are opaque tightly packed sRGB8; PNG color metadata is not converted.
use enough::Unstoppable;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Read},
    path::{Path, PathBuf},
    time::Instant,
};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality, zq::ZqTarget};
use zensim::{BakeScorer, RgbSlice};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_sha(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let mut reader = png::Decoder::new(BufReader::new(File::open(path)?)).read_info()?;
    if reader.info().bit_depth != png::BitDepth::Eight
        || !matches!(
            reader.info().color_type,
            png::ColorType::Rgb | png::ColorType::Grayscale
        )
    {
        return Err("source must be opaque RGB/grayscale 8-bit PNG".into());
    }
    let mut pixels = vec![0; reader.output_buffer_size().ok_or("PNG buffer overflow")?];
    let info = reader.next_frame(&mut pixels)?;
    pixels.truncate(info.buffer_size());
    if info.color_type == png::ColorType::Grayscale {
        pixels = pixels.into_iter().flat_map(|v| [v; 3]).collect();
    }
    Ok((pixels, info.width, info.height))
}
fn write_png(path: &Path, pixels: &[u8], width: u32, height: u32) -> Result<()> {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}
fn main() -> Result<()> {
    let mut input = None;
    let mut target = None;
    let mut out = None;
    let mut corrections = 2u8;
    let mut peak = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--image" => input = Some(PathBuf::from(value)),
            "--target" => target = Some(value.parse::<f32>()?),
            "--out-dir" => out = Some(PathBuf::from(value)),
            "--corrections" => corrections = value.parse()?,
            "--peak-bound" => peak = Some(value.parse::<f32>()?),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let input = input.ok_or("--image required")?.canonicalize()?;
    let target = target.ok_or("--target required")?;
    if !target.is_finite() {
        return Err("target must be finite".into());
    }
    let out = out.ok_or("--out-dir required")?;
    if out.exists() {
        return Err("output must be fresh".into());
    }
    let bake =
        PathBuf::from(std::env::var_os("ZENJPEG_ZQ_BAKE").ok_or("ZENJPEG_ZQ_BAKE required")?);
    let bake_bytes = std::fs::read(&bake)?;
    let model = zenpredict_serving::Model::from_bytes(&bake_bytes)?;
    let mut scorer = BakeScorer::new(&model)?;
    let (pixels, width, height) = load_png(&input)?;
    let source = RgbSlice::new(pixels.as_chunks::<3>().0, width as usize, height as usize);
    let target_spec = ZqTarget::new(target)
        .with_max_passes(corrections)
        .with_max_overshoot(Some(0.))
        .with_block_artifact(peak.map(zenjpeg::encode::zq::BlockArtifactBound::new));
    let config = EncoderConfig::ycbcr(Quality::ZqExplicit(target_spec), ChromaSubsampling::None);
    let started = Instant::now();
    let mut encoder = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    encoder.push_packed(&pixels, Unstoppable)?;
    let (jpeg, metrics) = encoder.finish_with_metrics()?;
    let loop_seconds = started.elapsed().as_secs_f64();
    let terminal_start = Instant::now();
    let decoded = zenjpeg::decode::Decoder::new().decode(&jpeg, Unstoppable)?;
    if decoded.width() != width || decoded.height() != height {
        return Err("decoded shape mismatch".into());
    }
    let decoded = decoded
        .into_pixels_u8()
        .ok_or("decoder returned non-u8 output")?;
    if decoded.len() != pixels.len() {
        return Err("decoded RGB shape mismatch".into());
    }
    let actual = scorer
        .compute(
            &source,
            &RgbSlice::new(decoded.as_chunks::<3>().0, width as usize, height as usize),
            Some("jpeg"),
        )?
        .score();
    if !actual.is_finite() || (actual - f64::from(metrics.achieved_score)).abs() > 1e-5 {
        return Err(format!(
            "terminal score mismatch {actual} versus {}",
            metrics.achieved_score
        )
        .into());
    }
    let terminal_seconds = terminal_start.elapsed().as_secs_f64();
    if file_sha(&bake)? != sha(&bake_bytes) {
        return Err("model bytes changed during encode".into());
    }
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("selected.jpg"), &jpeg)?;
    write_png(&out.join("selected.png"), &decoded, width, height)?;
    let report = serde_json::json!({
        "source":input,"source_sha256":file_sha(&input)?,"width":width,"height":height,
        "model_sha256":sha(&bake_bytes),"driver_sha256":file_sha(&std::env::current_exe()?)?,
        "target":target,"correction_budget":corrections,"full_encodes":metrics.passes_used,
        "achieved":metrics.achieved_score,"terminal_score":actual,"target_met":metrics.targets_met,
        "bytes":jpeg.len(),"encoded_sha256":sha(&jpeg),"decoded_sha256":sha(&decoded),
        "loop_seconds":loop_seconds,"terminal_seconds":terminal_seconds,
        "total_seconds":loop_seconds+terminal_seconds,
        "spatial_mode":std::env::var("ZENJPEG_ZQ_SPATIAL")?,"seed_q":std::env::var("ZENJPEG_ZQ_SEED_Q")?,
        "formula_revision":std::env::var("ZENSIM_FORMULA_REV")?,
        "scope":"fixed-seed native binding screen; no targeting or spatial qualification"
    });
    let text = serde_json::to_string_pretty(&report)?;
    std::fs::write(out.join("result.json"), format!("{text}\n"))?;
    println!("{text}");
    Ok(())
}
