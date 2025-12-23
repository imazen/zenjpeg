//! Corpus-wide comparison of jpegli vs mozjpeg.
//!
//! Generates an HTML chart showing quality/size tradeoff across quality levels.
//!
//! Usage: cargo run --release --example corpus_comparison -- <corpus_dir> <output.html>

use dssim::Dssim;
use rgb::RGBA8;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let (width, height) = (info.width as usize, info.height as usize);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let orig_rgba = rgb_to_rgba(original);
    let dec_rgba = rgb_to_rgba(decoded);
    let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp = attr.create_image_rgba(&dec_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn encode_jpegli(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(jpegli::PixelFormat::Rgb)
        .quality(jpegli::quant::Quality::from_quality(quality.into()))
        .encode(rgb)
        .expect("jpegli encode")
}

fn encode_mozjpeg(rgb: &[u8], width: usize, height: usize, quality: f32) -> Vec<u8> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality);
    // Use 4:4:4 for fair comparison
    comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));

    let mut started = comp.start_compress(Vec::new()).expect("mozjpeg start");
    let row_stride = width * 3;
    for y in 0..height {
        let row = &rgb[y * row_stride..(y + 1) * row_stride];
        let _ = started.write_scanlines(row);
    }
    started.finish().expect("mozjpeg finish")
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().expect("decode")
}

#[derive(Debug)]
struct DataPoint {
    quality: u8,
    jpegli_size: f64,      // average bytes per pixel
    jpegli_dssim: f64,
    mozjpeg_size: f64,
    mozjpeg_dssim: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <corpus_dir> <output.html>", args[0]);
        eprintln!("Example: {} /mnt/v/work/corpus/CID22-512 comparison.html", args[0]);
        std::process::exit(1);
    }

    let corpus_dir = &args[1];
    let output_path = &args[2];

    // Find PNG files
    let mut files: Vec<_> = fs::read_dir(corpus_dir)
        .expect("Failed to read corpus directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext.to_ascii_lowercase() == "png")
                .unwrap_or(false)
        })
        .collect();

    files.sort_by_key(|e| e.path());

    // Limit to first N files for faster testing (use 0 for all)
    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    if max_files > 0 && files.len() > max_files {
        println!("Limiting to {} files (set MAX_FILES=0 for all)", max_files);
        files.truncate(max_files);
    }

    println!("Processing {} PNG files...", files.len());

    let quality_levels = [50, 60, 70, 75, 80, 85, 90, 95];
    let mut data_points: Vec<DataPoint> = Vec::new();

    for &quality in &quality_levels {
        print!("Quality {}: ", quality);
        std::io::stdout().flush().unwrap();

        let mut total_jpegli_size = 0usize;
        let mut total_jpegli_dssim = 0.0;
        let mut total_mozjpeg_size = 0usize;
        let mut total_mozjpeg_dssim = 0.0;
        let mut total_pixels = 0usize;
        let mut count = 0;

        for entry in &files {
            let path = entry.path();
            if let Some((rgb, width, height)) = load_png(&path) {
                let pixels = width * height;

                // Encode with both
                let jpegli_data = encode_jpegli(&rgb, width as u32, height as u32, quality);
                let mozjpeg_data = encode_mozjpeg(&rgb, width, height, quality as f32);

                // Decode and measure quality
                let jpegli_decoded = decode_jpeg(&jpegli_data);
                let mozjpeg_decoded = decode_jpeg(&mozjpeg_data);

                let jpegli_dssim = compute_dssim(&rgb, &jpegli_decoded, width, height);
                let mozjpeg_dssim = compute_dssim(&rgb, &mozjpeg_decoded, width, height);

                total_jpegli_size += jpegli_data.len();
                total_jpegli_dssim += jpegli_dssim * pixels as f64;
                total_mozjpeg_size += mozjpeg_data.len();
                total_mozjpeg_dssim += mozjpeg_dssim * pixels as f64;
                total_pixels += pixels;
                count += 1;
            }
        }

        if count > 0 {
            let jpegli_bpp = total_jpegli_size as f64 / total_pixels as f64 * 8.0;
            let mozjpeg_bpp = total_mozjpeg_size as f64 / total_pixels as f64 * 8.0;
            let jpegli_dssim = total_jpegli_dssim / total_pixels as f64;
            let mozjpeg_dssim = total_mozjpeg_dssim / total_pixels as f64;

            println!(
                "jpegli: {:.3} bpp, {:.6} DSSIM | mozjpeg: {:.3} bpp, {:.6} DSSIM",
                jpegli_bpp, jpegli_dssim, mozjpeg_bpp, mozjpeg_dssim
            );

            data_points.push(DataPoint {
                quality,
                jpegli_size: jpegli_bpp,
                jpegli_dssim,
                mozjpeg_size: mozjpeg_bpp,
                mozjpeg_dssim,
            });
        }
    }

    // Generate HTML with SVG chart
    generate_html_chart(&data_points, output_path);
    println!("\nChart saved to: {}", output_path);
}

fn generate_html_chart(data: &[DataPoint], output_path: &str) {
    let width = 800.0;
    let height = 500.0;
    let margin = 60.0;
    let plot_width = width - 2.0 * margin;
    let plot_height = height - 2.0 * margin;

    // Find ranges
    let min_bpp = data
        .iter()
        .flat_map(|d| [d.jpegli_size, d.mozjpeg_size])
        .fold(f64::INFINITY, f64::min);
    let max_bpp = data
        .iter()
        .flat_map(|d| [d.jpegli_size, d.mozjpeg_size])
        .fold(0.0, f64::max);
    let min_dssim = data
        .iter()
        .flat_map(|d| [d.jpegli_dssim, d.mozjpeg_dssim])
        .fold(f64::INFINITY, f64::min);
    let max_dssim = data
        .iter()
        .flat_map(|d| [d.jpegli_dssim, d.mozjpeg_dssim])
        .fold(0.0, f64::max);

    // Add some padding
    let bpp_range = max_bpp - min_bpp;
    let dssim_range = max_dssim - min_dssim;
    let min_bpp = min_bpp - bpp_range * 0.1;
    let max_bpp = max_bpp + bpp_range * 0.1;
    let min_dssim = (min_dssim - dssim_range * 0.1).max(0.0);
    let max_dssim = max_dssim + dssim_range * 0.1;

    let scale_x = |bpp: f64| margin + (bpp - min_bpp) / (max_bpp - min_bpp) * plot_width;
    let scale_y = |dssim: f64| margin + plot_height - (dssim - min_dssim) / (max_dssim - min_dssim) * plot_height;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis {{ stroke: #333; stroke-width: 1; }}
    .grid {{ stroke: #ddd; stroke-width: 0.5; }}
    .jpegli {{ stroke: #2196F3; fill: #2196F3; }}
    .mozjpeg {{ stroke: #4CAF50; fill: #4CAF50; }}
    .label {{ font-family: sans-serif; font-size: 12px; }}
    .title {{ font-family: sans-serif; font-size: 16px; font-weight: bold; }}
    .legend {{ font-family: sans-serif; font-size: 12px; }}
  </style>
"#,
        width, height
    ));

    // Title
    svg.push_str(&format!(
        r#"  <text x="{}" y="25" class="title" text-anchor="middle">JPEG Quality vs Size: jpegli vs mozjpeg (4:4:4)</text>
"#,
        width / 2.0
    ));

    // Axes
    svg.push_str(&format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>
"#,
        margin, margin, margin, height - margin,
        margin, height - margin, width - margin, height - margin
    ));

    // X-axis label
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" class="label" text-anchor="middle">Bits per pixel (lower = smaller file)</text>
"#,
        width / 2.0, height - 15.0
    ));

    // Y-axis label
    svg.push_str(&format!(
        r#"  <text x="15" y="{}" class="label" text-anchor="middle" transform="rotate(-90, 15, {})">DSSIM (lower = better quality)</text>
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
            x, margin, x, height - margin,
            margin, y, width - margin, y
        ));

        // X-axis ticks
        let bpp = min_bpp + (max_bpp - min_bpp) * i as f64 / 5.0;
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{:.2}</text>
"#,
            x, height - margin + 15.0, bpp
        ));

        // Y-axis ticks
        let dssim = max_dssim - (max_dssim - min_dssim) * i as f64 / 5.0;
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="end">{:.4}</text>
"#,
            margin - 5.0, y + 4.0, dssim
        ));
    }

    // Draw jpegli line and points
    let mut jpegli_path = String::new();
    for (i, d) in data.iter().enumerate() {
        let x = scale_x(d.jpegli_size);
        let y = scale_y(d.jpegli_dssim);
        if i == 0 {
            jpegli_path.push_str(&format!("M {} {}", x, y));
        } else {
            jpegli_path.push_str(&format!(" L {} {}", x, y));
        }
    }
    svg.push_str(&format!(
        r#"  <path d="{}" class="jpegli" fill="none" stroke-width="2"/>
"#,
        jpegli_path
    ));

    for d in data {
        let x = scale_x(d.jpegli_size);
        let y = scale_y(d.jpegli_dssim);
        svg.push_str(&format!(
            r#"  <circle cx="{}" cy="{}" r="5" class="jpegli"/>
  <text x="{}" y="{}" class="label" text-anchor="middle">Q{}</text>
"#,
            x, y, x, y - 8.0, d.quality
        ));
    }

    // Draw mozjpeg line and points
    let mut mozjpeg_path = String::new();
    for (i, d) in data.iter().enumerate() {
        let x = scale_x(d.mozjpeg_size);
        let y = scale_y(d.mozjpeg_dssim);
        if i == 0 {
            mozjpeg_path.push_str(&format!("M {} {}", x, y));
        } else {
            mozjpeg_path.push_str(&format!(" L {} {}", x, y));
        }
    }
    svg.push_str(&format!(
        r#"  <path d="{}" class="mozjpeg" fill="none" stroke-width="2"/>
"#,
        mozjpeg_path
    ));

    for d in data {
        let x = scale_x(d.mozjpeg_size);
        let y = scale_y(d.mozjpeg_dssim);
        svg.push_str(&format!(
            r#"  <circle cx="{}" cy="{}" r="5" class="mozjpeg"/>
"#,
            x, y
        ));
    }

    // Legend
    svg.push_str(&format!(
        "  <rect x=\"{}\" y=\"40\" width=\"120\" height=\"50\" fill=\"white\" stroke=\"#ccc\"/>\n\
  <line x1=\"{}\" y1=\"55\" x2=\"{}\" y2=\"55\" class=\"jpegli\" stroke-width=\"2\"/>\n\
  <circle cx=\"{}\" cy=\"55\" r=\"4\" class=\"jpegli\"/>\n\
  <text x=\"{}\" y=\"59\" class=\"legend\">jpegli-rs</text>\n\
  <line x1=\"{}\" y1=\"75\" x2=\"{}\" y2=\"75\" class=\"mozjpeg\" stroke-width=\"2\"/>\n\
  <circle cx=\"{}\" cy=\"75\" r=\"4\" class=\"mozjpeg\"/>\n\
  <text x=\"{}\" y=\"79\" class=\"legend\">mozjpeg (4:4:4)</text>\n",
        width - 140.0,
        width - 130.0, width - 100.0,
        width - 115.0,
        width - 90.0,
        width - 130.0, width - 100.0,
        width - 115.0,
        width - 90.0
    ));

    svg.push_str("</svg>");

    // Generate HTML
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>jpegli vs mozjpeg Comparison</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
        th {{ background: #f0f0f0; }}
        .better {{ background: #e8f5e9; font-weight: bold; }}
        .note {{ color: #666; font-size: 14px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>jpegli-rs vs mozjpeg Quality/Size Comparison</h1>
        <p>Lower-left is better (smaller file, better quality). Both encoders use 4:4:4 subsampling.</p>
        {}
        <h2>Data Table</h2>
        <table>
            <tr>
                <th>Quality</th>
                <th>jpegli bpp</th>
                <th>mozjpeg bpp</th>
                <th>jpegli DSSIM</th>
                <th>mozjpeg DSSIM</th>
                <th>Size Winner</th>
                <th>Quality Winner</th>
            </tr>
{}
        </table>
        <p class="note">
            <strong>bpp</strong> = bits per pixel (lower = smaller file)<br>
            <strong>DSSIM</strong> = structural dissimilarity (lower = better quality, 0 = identical)<br>
            <strong>Note:</strong> A curve closer to the bottom-left corner represents better quality/size efficiency.
        </p>
    </div>
</body>
</html>"#,
        svg,
        data.iter()
            .map(|d| {
                let size_winner = if d.jpegli_size < d.mozjpeg_size { "jpegli" } else { "mozjpeg" };
                let quality_winner = if d.jpegli_dssim < d.mozjpeg_dssim { "jpegli" } else { "mozjpeg" };
                let size_class = if d.jpegli_size < d.mozjpeg_size { " class=\"better\"" } else { "" };
                let qual_class = if d.jpegli_dssim < d.mozjpeg_dssim { " class=\"better\"" } else { "" };
                format!(
                    "            <tr><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.6}</td><td>{:.6}</td><td{}>{}</td><td{}>{}</td></tr>",
                    d.quality, d.jpegli_size, d.mozjpeg_size, d.jpegli_dssim, d.mozjpeg_dssim,
                    size_class, size_winner, qual_class, quality_winner
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    );

    fs::write(output_path, html).expect("Failed to write HTML");
}
