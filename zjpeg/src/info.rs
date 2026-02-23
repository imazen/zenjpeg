use std::path::Path;

use anyhow::{Context, Result};
use zenjpeg::detect::content::{ContentType, classify_from_probe};
use zenjpeg::detect::{self, EncoderFamily, JpegProbe, QualityScale};

use crate::InfoArgs;
use crate::batch;

pub fn run(args: InfoArgs) -> Result<()> {
    let files = batch::expand_inputs(&args.input)?;
    if files.is_empty() {
        anyhow::bail!("no JPEG files found");
    }

    for (i, path) in files.iter().enumerate() {
        if i > 0 && !args.json {
            println!();
        }
        match show_info(path, &args) {
            Ok(()) => {}
            Err(e) => eprintln!("error: {}: {e}", path.display()),
        }
    }

    Ok(())
}

fn show_info(path: &Path, args: &InfoArgs) -> Result<()> {
    let data =
        std::fs::read(path).with_context(|| format!("failed to read '{}'", path.display()))?;
    let probe = detect::probe(&data).map_err(|e| anyhow::anyhow!("{e}"))?;

    if args.json {
        print_json(path, &probe);
    } else {
        print_human(path, &probe, args);
    }

    Ok(())
}

fn print_human(path: &Path, probe: &JpegProbe, args: &InfoArgs) {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    println!("{}", path.display());
    println!(
        "  Dimensions:   {}x{}",
        probe.dimensions.width, probe.dimensions.height
    );
    println!("  Encoder:      {}", encoder_name(&probe.encoder));
    println!("  Quality:      {}", format_quality(probe));
    println!("  Mode:         {:?}", probe.mode);
    println!("  Subsampling:  {:?}", probe.subsampling);
    println!("  Components:   {}", probe.num_components);
    println!("  Scans:        {}", probe.scan_count);
    println!("  File size:    {}", format_size(file_size));

    // Content classification
    let content = classify_from_probe(probe);
    println!("  Content:      {}", content_name(content));

    // Quality recommendation
    let rec = probe.recommended_quality();
    println!("  Recommended:  Q{}", format_quality_value(&rec));

    // Reencode settings at default tolerance
    match probe.reencode_settings(0.3) {
        Ok(settings) => {
            println!(
                "  Reencode:     Q{} {:?} (BA tol=0.3)",
                format_quality_value(&settings.quality),
                settings.subsampling
            );
        }
        Err(e) => {
            println!("  Reencode:     {e}");
        }
    }

    // DQT tables
    if args.all || args.quant {
        println!("  Quant tables: {}", probe.dqt_tables.len());
        for table in &probe.dqt_tables {
            println!(
                "    Table {} ({}bit):",
                table.index,
                if table.precision > 0 { 16 } else { 8 }
            );
            // Print 8x8 grid
            for row in 0..8 {
                print!("      ");
                for col in 0..8 {
                    print!("{:4}", table.values[row * 8 + col]);
                }
                println!();
            }
        }
    }
}

fn print_json(path: &Path, probe: &JpegProbe) {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let content = classify_from_probe(probe);
    let rec = probe.recommended_quality();

    // Manual JSON to avoid serde dependency
    println!("{{");
    println!("  \"file\": \"{}\",", path.display());
    println!("  \"width\": {},", probe.dimensions.width);
    println!("  \"height\": {},", probe.dimensions.height);
    println!("  \"encoder\": \"{}\",", encoder_name(&probe.encoder));
    println!("  \"quality_value\": {:.1},", probe.quality.value);
    println!("  \"quality_scale\": \"{:?}\",", probe.quality.scale);
    println!(
        "  \"quality_confidence\": \"{:?}\",",
        probe.quality.confidence
    );
    println!("  \"mode\": \"{:?}\",", probe.mode);
    println!("  \"subsampling\": \"{:?}\",", probe.subsampling);
    println!("  \"components\": {},", probe.num_components);
    println!("  \"scans\": {},", probe.scan_count);
    println!("  \"file_size\": {file_size},");
    println!("  \"content_type\": \"{}\",", content_name(content));
    println!(
        "  \"recommended_quality\": \"{}\"",
        format_quality_value(&rec)
    );
    println!("}}");
}

fn encoder_name(encoder: &EncoderFamily) -> &'static str {
    match encoder {
        EncoderFamily::LibjpegTurbo => "libjpeg-turbo",
        EncoderFamily::Mozjpeg => "mozjpeg",
        EncoderFamily::CjpegliYcbcr => "cjpegli (YCbCr)",
        EncoderFamily::CjpegliXyb => "cjpegli (XYB)",
        EncoderFamily::ImageMagick => "ImageMagick",
        EncoderFamily::IjgFamily => "IJG family",
        EncoderFamily::Unknown => "unknown",
        _ => "unknown",
    }
}

fn content_name(content: ContentType) -> &'static str {
    match content {
        ContentType::Photo => "photo",
        ContentType::Screenshot => "screenshot",
        ContentType::Mixed => "mixed/unknown",
    }
}

fn format_quality(probe: &JpegProbe) -> String {
    match probe.quality.scale {
        QualityScale::IjgQuality => format!("Q{:.0} (IJG)", probe.quality.value),
        QualityScale::MozjpegQuality => format!("Q{:.0} (mozjpeg)", probe.quality.value),
        QualityScale::ButteraugliDistance => format!("d={:.2} (butteraugli)", probe.quality.value),
        _ => format!("{:.1} ({:?})", probe.quality.value, probe.quality.scale),
    }
}

fn format_quality_value(q: &zenjpeg::encoder::Quality) -> String {
    match q {
        zenjpeg::encoder::Quality::ApproxJpegli(v) => format!("{v:.0}"),
        zenjpeg::encoder::Quality::ApproxMozjpeg(v) => format!("{v}"),
        zenjpeg::encoder::Quality::ApproxButteraugli(v) => format!("d={v:.2}"),
        zenjpeg::encoder::Quality::ApproxSsim2(v) => format!("s2={v:.1}"),
        _ => format!("{q:?}"),
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
