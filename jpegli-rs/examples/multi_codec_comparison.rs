//! Multi-codec comparison using CID22 dataset.
//!
//! Compares jpegli vs other codecs (mozjpeg, libjxl, avif, webp, heic, jp2)
//! using pre-computed quality metrics from the CID22 validation set.
//!
//! **DEPRECATED**: Use `quality_compare` instead for JPEG comparisons:
//!   cargo run --release --example quality_compare -- --pareto --output results.csv image.png
//!
//! Usage: cargo run --release --example multi_codec_comparison -- <cid22_dir> <output.html>

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;

#[derive(Debug, Clone)]
struct DataPoint {
    encoder: String,
    setting: String,
    bpp: f64,
    mcos: f64, // Mean Comparative Opinion Score (higher = better)
}

#[derive(Debug)]
struct CodecData {
    name: String,
    color: String,
    points: Vec<(f64, f64)>, // (bpp, mcos)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <cid22_dir> <output.html>", args[0]);
        eprintln!(
            "Example: {} /mnt/v/work/CID22/CID22 comparison.html",
            args[0]
        );
        std::process::exit(1);
    }

    let cid22_dir = &args[1];
    let output_path = &args[2];

    let csv_path = format!("{}/CID22_validation_set.csv", cid22_dir);
    let csv_content = fs::read_to_string(&csv_path).expect("Failed to read CSV");

    // Parse CSV and collect data by encoder
    let mut encoder_data: HashMap<String, Vec<DataPoint>> = HashMap::new();

    for line in csv_content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 {
            continue;
        }

        let encoder = fields[2].to_string();
        if encoder == "Reference" {
            continue;
        }

        let setting = fields[3].to_string();
        let bpp: f64 = fields[4].parse().unwrap_or(0.0);
        let mcos: f64 = fields[5].parse().unwrap_or(0.0);

        if bpp > 0.0 && mcos > 0.0 {
            encoder_data
                .entry(encoder.clone())
                .or_default()
                .push(DataPoint {
                    encoder,
                    setting,
                    bpp,
                    mcos,
                });
        }
    }

    // Aggregate by encoder and quality setting (average across images)
    let mut codec_averages: HashMap<String, HashMap<String, (f64, f64, usize)>> = HashMap::new();

    for (encoder, points) in &encoder_data {
        for point in points {
            let entry = codec_averages
                .entry(encoder.clone())
                .or_default()
                .entry(point.setting.clone())
                .or_insert((0.0, 0.0, 0));
            entry.0 += point.bpp;
            entry.1 += point.mcos;
            entry.2 += 1;
        }
    }

    // Convert to sorted codec data
    let colors = [
        ("#2196F3", "jpegli-rs"), // Blue (our encoder - placeholder)
        ("#FF5722", "JPEG"),      // Deep Orange for mozjpeg
        ("#9C27B0", "JPEG_XL"),   // Purple for jxl
        ("#4CAF50", "AVIF"),      // Green for avif
        ("#FFC107", "WEBP"),      // Amber for webp
        ("#00BCD4", "HEIC"),      // Cyan for heic
        ("#795548", "JPEG_2000"), // Brown for jp2
        ("#607D8B", "AOM"),       // Blue Gray for aom
    ];

    let color_map: HashMap<&str, &str> = colors.iter().map(|&(c, n)| (n, c)).collect();

    let mut codecs: Vec<CodecData> = Vec::new();

    for (encoder, settings) in &codec_averages {
        let color = color_map.get(encoder.as_str()).unwrap_or(&"#999999");

        let mut points: Vec<(f64, f64)> = settings
            .iter()
            .map(|(_, &(bpp_sum, mcos_sum, count))| {
                (bpp_sum / count as f64, mcos_sum / count as f64)
            })
            .collect();

        // Sort by bpp
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        codecs.push(CodecData {
            name: encoder.clone(),
            color: color.to_string(),
            points,
        });
    }

    // Sort codecs alphabetically
    codecs.sort_by(|a, b| a.name.cmp(&b.name));

    println!("Loaded {} codecs from CID22 dataset:", codecs.len());
    for codec in &codecs {
        println!("  {}: {} quality settings", codec.name, codec.points.len());
    }

    // Generate HTML
    generate_html_chart(&codecs, output_path);
    println!("\nChart saved to: {}", output_path);
}

fn generate_html_chart(codecs: &[CodecData], output_path: &str) {
    let width = 900.0;
    let height = 600.0;
    let margin = 70.0;
    let plot_width = width - 2.0 * margin;
    let plot_height = height - 2.0 * margin;

    // Find ranges
    let min_bpp = codecs
        .iter()
        .flat_map(|c| c.points.iter().map(|(b, _)| *b))
        .fold(f64::INFINITY, f64::min);
    let max_bpp = codecs
        .iter()
        .flat_map(|c| c.points.iter().map(|(b, _)| *b))
        .fold(0.0, f64::max);
    let min_mcos = codecs
        .iter()
        .flat_map(|c| c.points.iter().map(|(_, m)| *m))
        .fold(f64::INFINITY, f64::min);
    let max_mcos = codecs
        .iter()
        .flat_map(|c| c.points.iter().map(|(_, m)| *m))
        .fold(0.0, f64::max);

    // Add padding
    let bpp_range = max_bpp - min_bpp;
    let mcos_range = max_mcos - min_mcos;
    let min_bpp = (min_bpp - bpp_range * 0.05).max(0.0);
    let max_bpp = max_bpp + bpp_range * 0.05;
    let min_mcos = (min_mcos - mcos_range * 0.05).max(0.0);
    let max_mcos = max_mcos + mcos_range * 0.05;

    let scale_x = |bpp: f64| margin + (bpp - min_bpp) / (max_bpp - min_bpp) * plot_width;
    let scale_y =
        |mcos: f64| margin + plot_height - (mcos - min_mcos) / (max_mcos - min_mcos) * plot_height;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis {{ stroke: #333; stroke-width: 1; }}
    .grid {{ stroke: #ddd; stroke-width: 0.5; }}
    .label {{ font-family: sans-serif; font-size: 12px; }}
    .title {{ font-family: sans-serif; font-size: 16px; font-weight: bold; }}
    .legend {{ font-family: sans-serif; font-size: 11px; }}
  </style>
"#,
        width, height
    ));

    // Title
    svg.push_str(&format!(
        r#"  <text x="{}" y="25" class="title" text-anchor="middle">Image Codec Comparison (CID22 Dataset)</text>
"#,
        width / 2.0
    ));

    // Axes
    svg.push_str(&format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>
"#,
        margin,
        margin,
        margin,
        height - margin,
        margin,
        height - margin,
        width - margin,
        height - margin
    ));

    // X-axis label
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" class="label" text-anchor="middle">Bits per pixel (lower = smaller file)</text>
"#,
        width / 2.0, height - 15.0
    ));

    // Y-axis label
    svg.push_str(&format!(
        r#"  <text x="20" y="{}" class="label" text-anchor="middle" transform="rotate(-90, 20, {})">MCOS (higher = better quality)</text>
"#,
        height / 2.0, height / 2.0
    ));

    // Grid lines
    for i in 0..=5 {
        let x = margin + plot_width * i as f64 / 5.0;
        let y = margin + plot_height * i as f64 / 5.0;
        svg.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
            x,
            margin,
            x,
            height - margin,
            margin,
            y,
            width - margin,
            y
        ));

        // X-axis ticks
        let bpp = min_bpp + (max_bpp - min_bpp) * i as f64 / 5.0;
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{:.2}</text>
"#,
            x,
            height - margin + 15.0,
            bpp
        ));

        // Y-axis ticks
        let mcos = max_mcos - (max_mcos - min_mcos) * i as f64 / 5.0;
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="end">{:.0}</text>
"#,
            margin - 5.0,
            y + 4.0,
            mcos
        ));
    }

    // Draw each codec
    for codec in codecs {
        if codec.points.is_empty() {
            continue;
        }

        // Draw line
        let mut path = String::new();
        for (i, &(bpp, mcos)) in codec.points.iter().enumerate() {
            let x = scale_x(bpp);
            let y = scale_y(mcos);
            if i == 0 {
                path.push_str(&format!("M {} {}", x, y));
            } else {
                path.push_str(&format!(" L {} {}", x, y));
            }
        }
        svg.push_str(&format!(
            r#"  <path d="{}" stroke="{}" fill="none" stroke-width="2"/>
"#,
            path, codec.color
        ));

        // Draw points
        for &(bpp, mcos) in &codec.points {
            let x = scale_x(bpp);
            let y = scale_y(mcos);
            svg.push_str(&format!(
                r#"  <circle cx="{}" cy="{}" r="4" stroke="{}" fill="{}"/>
"#,
                x, y, codec.color, codec.color
            ));
        }
    }

    // Legend
    let legend_x = width - 130.0;
    let legend_y = 45.0;
    let legend_height = 20.0 * codecs.len() as f64 + 10.0;

    svg.push_str(&format!(
        "  <rect x=\"{}\" y=\"{}\" width=\"120\" height=\"{}\" fill=\"white\" stroke=\"#ccc\"/>\n",
        legend_x, legend_y, legend_height
    ));

    for (i, codec) in codecs.iter().enumerate() {
        let y = legend_y + 18.0 + 20.0 * i as f64;
        svg.push_str(&format!(
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>\n",
            legend_x + 10.0,
            y,
            legend_x + 30.0,
            y,
            codec.color
        ));
        svg.push_str(&format!(
            "  <circle cx=\"{}\" cy=\"{}\" r=\"3\" stroke=\"{}\" fill=\"{}\"/>\n",
            legend_x + 20.0,
            y,
            codec.color,
            codec.color
        ));
        svg.push_str(&format!(
            "  <text x=\"{}\" y=\"{}\" class=\"legend\">{}</text>\n",
            legend_x + 40.0,
            y + 4.0,
            codec.name
        ));
    }

    svg.push_str("</svg>");

    // Generate HTML
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multi-Codec Comparison (CID22)</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
        th {{ background: #f0f0f0; }}
        .note {{ color: #666; font-size: 14px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Image Codec Comparison (CID22 Dataset)</h1>
        <p>Higher and to the left is better (smaller file, better quality).</p>
        <p><strong>MCOS</strong> = Mean Comparative Opinion Score from human evaluators.</p>
        {}
        <p class="note">
            Data source: <a href="https://cloudinary.com/labs/cid22">Cloudinary Image Dataset 2022</a><br>
            This chart shows averaged metrics across 250 diverse images at various quality settings.
        </p>
    </div>
</body>
</html>"#,
        svg
    );

    fs::write(output_path, html).expect("Failed to write HTML");
}
