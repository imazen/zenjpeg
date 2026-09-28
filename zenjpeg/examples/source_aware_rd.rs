//! Source-aware table experiments; see docs/recompress/SOURCE_AWARE_EXPERIMENT.md.
//! Uses existing encoder APIs only. Metrics and experimental table selection
//! are development tools, not part of the production encoder.

#[path = "source_aware/quant.rs"]
mod quant;

use enough::Unstoppable;
use imgref::ImgVec;
use rgb::RGB8;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use zenjpeg::decoder::{DecodeConfig, DecodedCoefficients};
use zenjpeg::encode::tuning::{EncodingTables, PerComponent, ScalingParams};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, MozjpegTables, OptimizationPreset, Quality, QuantTablePreset,
};
use zenjpeg::foundation::consts::JPEG_NATURAL_ORDER;
use zenjpeg::quant::{ZeroBiasParams, create_quant_table, quant_vals_to_distance};

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;
type Tables = [[u16; 64]; 3];

struct Args {
    corpus: PathBuf,
    output: PathBuf,
    limit: usize,
    sources: Vec<String>,
    destinations: Vec<String>,
    source_qualities: Vec<u8>,
    qualities: Vec<u8>,
    strengths: Vec<f64>,
    window: f64,
    sampling: ChromaSubsampling,
    sampling_name: String,
}

fn qualities(s: &str) -> AnyResult<Vec<u8>> {
    let mut values: Vec<u8> = s.split(',').map(str::parse).collect::<Result<_, _>>()?;
    if values.is_empty() || values.iter().any(|q| !(1..=100).contains(q)) {
        return Err("qualities must be in 1..=100".into());
    }
    values.sort_unstable();
    values.dedup();
    Ok(values)
}

fn modes(s: &str) -> AnyResult<Vec<String>> {
    let mut values: Vec<String> = s.split(',').map(str::to_owned).collect();
    if values
        .iter()
        .any(|s| !matches!(s.as_str(), "jpegli" | "moz" | "classic"))
    {
        return Err("modes: jpegli,moz,classic".into());
    }
    values.sort();
    values.dedup();
    Ok(values)
}

fn args() -> AnyResult<Args> {
    let mut a = Args {
        corpus: PathBuf::new(),
        output: PathBuf::new(),
        limit: 3,
        sources: modes("jpegli,moz,classic")?,
        destinations: modes("jpegli,moz,classic")?,
        source_qualities: vec![80],
        qualities: vec![45, 55, 65, 75, 85, 95],
        strengths: vec![1.0, 4.0],
        window: 0.2,
        sampling: ChromaSubsampling::Quarter,
        sampling_name: "420".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "source_aware_rd --corpus PNG_OR_DIR --output NEW_DIR\n  --limit 3 --sources jpegli,moz,classic --destinations jpegli,moz,classic\n  --source-qualities 80 --qualities 45,55,65,75,85,95\n  --strengths 1,4 --window 0.2 --sampling 420|444"
            );
            std::process::exit(0);
        }
        let value = it.next().ok_or("missing flag value")?;
        match flag.as_str() {
            "--corpus" => a.corpus = value.into(),
            "--output" => a.output = value.into(),
            "--limit" => a.limit = value.parse()?,
            "--sources" => a.sources = modes(&value)?,
            "--destinations" => a.destinations = modes(&value)?,
            "--source-qualities" => a.source_qualities = qualities(&value)?,
            "--qualities" => a.qualities = qualities(&value)?,
            "--strengths" => {
                a.strengths = value.split(',').map(str::parse).collect::<Result<_, _>>()?
            }
            "--window" => a.window = value.parse()?,
            "--sampling" => {
                a.sampling = match value.as_str() {
                    "420" => ChromaSubsampling::Quarter,
                    "444" => ChromaSubsampling::None,
                    _ => return Err("sampling must be 420 or 444".into()),
                };
                a.sampling_name = value;
            }
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    if a.corpus.as_os_str().is_empty() || a.output.as_os_str().is_empty() || a.limit == 0 {
        return Err("--corpus, --output, and a positive --limit are required".into());
    }
    if !a.window.is_finite()
        || !(0.0..=0.5).contains(&a.window)
        || a.strengths.is_empty()
        || a.strengths.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("window must be 0..=0.5; strengths must be finite and nonnegative".into());
    }
    a.strengths.sort_by(f64::total_cmp);
    a.strengths.dedup();
    Ok(a)
}

fn config(mode: &str, q: u8, sampling: ChromaSubsampling) -> EncoderConfig {
    match mode {
        "jpegli" => EncoderConfig::ycbcr(Quality::ApproxJpegli(f32::from(q)), sampling)
            .optimization(OptimizationPreset::JpegliProgressive),
        "moz" => EncoderConfig::ycbcr(Quality::ApproxMozjpeg(q), sampling)
            .optimization(OptimizationPreset::MozjpegProgressive),
        "classic" => EncoderConfig::ycbcr(Quality::ApproxJpegli(f32::from(q)), sampling)
            .tables(MozjpegTables::generate_ex(
                q,
                QuantTablePreset::JpegAnnexK,
                true,
            ))
            .aq_enabled(false)
            .deringing(false)
            .progressive(false),
        _ => unreachable!(),
    }
}

fn encode(cfg: &EncoderConfig, image: &ImgVec<RGB8>) -> AnyResult<Vec<u8>> {
    let mut enc = cfg.encode_from_rgb::<RGB8>(image.width() as u32, image.height() as u32)?;
    enc.push_packed(image.buf(), Unstoppable)?;
    Ok(enc.finish()?)
}

fn decode(bytes: &[u8]) -> AnyResult<ImgVec<RGB8>> {
    // One pinned decoder/settings for both references and all candidates.
    Ok(zenjpeg_bench_utils::decode_jpeg_with_icc(bytes)?)
}

fn coefficients(bytes: &[u8]) -> AnyResult<DecodedCoefficients> {
    Ok(DecodeConfig::new().decode_coefficients(bytes, Unstoppable)?)
}

fn tables(c: &DecodedCoefficients) -> AnyResult<Tables> {
    if c.components.len() != 3 {
        return Err("this experiment requires three YCbCr components".into());
    }
    let mut tables = [[0; 64]; 3];
    for (dst, component) in tables.iter_mut().zip(&c.components) {
        *dst = c
            .quant_tables
            .get(component.quant_table_idx as usize)
            .and_then(|t| *t)
            .ok_or("missing source DQT")?;
    }
    Ok(tables)
}

fn histograms(c: &DecodedCoefficients) -> [[quant::Histogram; 64]; 3] {
    std::array::from_fn(|ci| {
        let mut hist: [quant::Histogram; 64] = std::array::from_fn(|_| quant::Histogram::new());
        let component = &c.components[ci];
        for block in component.coeffs.chunks_exact(64) {
            for (zz, &k) in block.iter().enumerate() {
                *hist[JPEG_NATURAL_ORDER[zz] as usize].entry(k).or_default() += 1;
            }
        }
        hist
    })
}

fn exact_tables(mode: &str, q: u8, values: &Tables) -> EncodingTables {
    let mut exact = if mode == "jpegli" {
        let qt = values.map(|t| create_quant_table(t, false));
        let distance = quant_vals_to_distance(&qt[0], &qt[1], &qt[2], false);
        let biases: [_; 3] = std::array::from_fn(|c| ZeroBiasParams::for_ycbcr(distance, c));
        let mut t = EncodingTables::default_ycbcr();
        for (c, bias) in biases.iter().enumerate() {
            *t.zero_bias_mul.get_mut(c) = bias.mul;
            t.zero_bias_offset_dc[c] = bias.offset[0];
            t.zero_bias_offset_ac[c] = bias.offset[1];
        }
        t
    } else {
        *MozjpegTables::generate_ex(
            q,
            if mode == "moz" {
                QuantTablePreset::Robidoux
            } else {
                QuantTablePreset::JpegAnnexK
            },
            true,
        )
    };
    exact.quant = PerComponent::new(
        values[0].map(f32::from),
        values[1].map(f32::from),
        values[2].map(f32::from),
    );
    exact.scaling = ScalingParams::Exact;
    exact
}

fn score(reference: &ImgVec<RGB8>, image: &ImgVec<RGB8>) -> AnyResult<serde_json::Value> {
    let arrays = |v: &ImgVec<RGB8>| {
        ImgVec::new(
            v.buf().iter().map(|p| [p.r, p.g, p.b]).collect::<Vec<_>>(),
            v.width(),
            v.height(),
        )
    };
    let a = arrays(reference);
    let b = arrays(image);
    // Propagate metric errors. Never turn failures into a score of zero.
    let ssim2 = fast_ssim2::compute_ssimulacra2(a.as_ref(), b.as_ref())?;
    let butter = butteraugli::butteraugli(
        reference.as_ref(),
        image.as_ref(),
        &butteraugli::ButteraugliParams::default(),
    )?
    .score;
    if !ssim2.is_finite() || !butter.is_finite() {
        return Err("nonfinite IQA score".into());
    }
    Ok(json!({"ssim2": ssim2, "butteraugli": butter}))
}

fn main() -> AnyResult<()> {
    let args = args()?;
    let mut images: Vec<PathBuf> = if args.corpus.is_file() {
        vec![args.corpus.clone()]
    } else {
        fs::read_dir(&args.corpus)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")))
            .collect()
    };
    images.sort();
    images.truncate(args.limit);
    if images.is_empty() {
        return Err("corpus contains no PNGs".into());
    }
    if args.output.exists() {
        return Err("output directory already exists; use a new run directory".into());
    }
    fs::create_dir_all(&args.output)?;
    // Preserve the compiled experiment sources even for an uncommitted run.
    fs::write(
        args.output.join("source_aware_rd.rs"),
        include_str!("source_aware_rd.rs"),
    )?;
    fs::write(
        args.output.join("quant.rs"),
        include_str!("source_aware/quant.rs"),
    )?;
    fs::write(
        args.output.join("Cargo.lock"),
        include_str!("../../Cargo.lock"),
    )?;
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()?;
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    fs::write(
        args.output.join("run.json"),
        serde_json::to_vec_pretty(&json!({
            "argv": std::env::args().collect::<Vec<_>>(), "git_head": String::from_utf8_lossy(&revision.stdout).trim(),
            "git_status": String::from_utf8_lossy(&dirty.stdout), "images": images,
            "source_encoder": "zenjpeg mode emulation, not external C encoders",
            "decoder": "zenjpeg via decode_jpeg_with_icc", "color": "8-bit sRGB PNG inputs; no resizing",
            "metric_implementations": "Cargo.lock: fast-ssim2; butteraugli default scalar score",
            "candidate_search": "coefficient MSE / target_step^2 + strength * ln(step/target_step)^2; AC only",
            "window": args.window, "strengths": args.strengths,
        }))?,
    )?;
    let mut records = BufWriter::new(File::create(args.output.join("points.jsonl"))?);
    for (image_index, path) in images.iter().enumerate() {
        let original = zenjpeg_bench_utils::load_png(path)?;
        let image_id = format!(
            "{image_index:03}_{}",
            path.file_stem().unwrap().to_string_lossy()
        );
        let original_path = args.output.join(format!("{image_id}.png"));
        fs::copy(path, &original_path)?;
        for source_mode in &args.sources {
            for &source_q in &args.source_qualities {
                let source_bytes =
                    encode(&config(source_mode, source_q, args.sampling), &original)?;
                let source_path = args
                    .output
                    .join(format!("{image_id}__{source_mode}_q{source_q}.jpg"));
                fs::write(&source_path, &source_bytes)?;
                let source_pixels = decode(&source_bytes)?;
                let source_coeffs = coefficients(&source_bytes)?;
                let source_tables = tables(&source_coeffs)?;
                let hist = histograms(&source_coeffs);
                let prior = quant::prior();
                for mode in &args.destinations {
                    for &q in &args.qualities {
                        let cfg = config(mode, q, args.sampling);
                        let start = Instant::now();
                        let generic = encode(&cfg, &source_pixels)?;
                        let generic_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let target = tables(&coefficients(&generic)?)?;
                        let exact = exact_tables(mode, q, &target);
                        let mut variants = vec![
                            ("generic".to_owned(), None),
                            ("exact".to_owned(), Some(exact.clone())),
                        ];
                        for &strength in &args.strengths {
                            for informed in [false, true] {
                                let mut candidate = exact.clone();
                                for c in 0..3 {
                                    for f in 1..64 {
                                        let b = quant::select(
                                            source_tables[c][f],
                                            target[c][f],
                                            if informed { &hist[c][f] } else { &prior },
                                            args.window,
                                            strength,
                                        );
                                        candidate.quant.get_mut(c)[f] = f32::from(b);
                                    }
                                }
                                variants.push((
                                    format!(
                                        "{}_s{strength}",
                                        if informed { "hist" } else { "hint" }
                                    ),
                                    Some(candidate),
                                ));
                            }
                        }
                        for (variant, custom) in variants {
                            let start = Instant::now();
                            let (bytes, encode_ms) = if let Some(t) = custom {
                                let bytes =
                                    encode(&cfg.clone().tables(Box::new(t)), &source_pixels)?;
                                (bytes, start.elapsed().as_secs_f64() * 1000.0)
                            } else {
                                (generic.clone(), generic_ms)
                            };
                            let filename = format!(
                                "{image_id}__{source_mode}_q{source_q}__{mode}_q{q}__{variant}.jpg"
                            );
                            fs::write(args.output.join(&filename), &bytes)?;
                            let decoded = decode(&bytes)?;
                            let generation = score(&source_pixels, &decoded)?;
                            let cumulative = score(&original, &decoded)?;
                            let actual_tables = tables(&coefficients(&bytes)?)?;
                            if variant == "exact" && actual_tables != target {
                                return Err("exact table control changed DQT values".into());
                            }
                            if variant == "exact" && decoded.buf() != decode(&generic)?.buf() {
                                return Err("exact table control changed decoded pixels".into());
                            }
                            let record = json!({
                                "image": image_id, "source_mode": source_mode, "source_quality": source_q,
                                "destination": mode, "quality": q, "sampling": args.sampling_name, "variant": variant,
                                "width": original.width(), "height": original.height(), "source_bytes": source_bytes.len(),
                                "bytes": bytes.len(), "bpp": bytes.len() as f64 * 8.0 / (original.width() * original.height()) as f64,
                                "encode_ms": encode_ms, "generation": generation, "cumulative": cumulative, "jpeg": filename,
                                "quant_tables": actual_tables.iter().map(|t| t.to_vec()).collect::<Vec<_>>(),
                            });
                            serde_json::to_writer(&mut records, &record)?;
                            writeln!(records)?;
                            records.flush()?;
                        }
                        eprintln!("{image_id} {source_mode}/{source_q} -> {mode}/{q} complete");
                    }
                }
            }
        }
    }
    fs::write(
        args.output.join("COMPLETE"),
        b"all requested cells finished\n",
    )?;
    eprintln!("Saved {}", args.output.display());
    Ok(())
}
