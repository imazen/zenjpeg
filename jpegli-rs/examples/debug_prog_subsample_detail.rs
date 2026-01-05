//! Debug progressive + subsampling - detailed analysis
//!
//! Check the actual JPEG structure to see what subsampling is encoded

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;

fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            data.push((x * 255 / width) as u8);
            data.push((y * 255 / height) as u8);
            data.push(128);
        }
    }
    data
}

fn parse_jpeg_sof(data: &[u8]) -> Option<(u8, u8, u8, Vec<(u8, u8, u8)>)> {
    // Find SOF0 (0xFFC0) or SOF2 (0xFFC2) marker
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF {
            let marker = data[i + 1];
            if marker == 0xC0 || marker == 0xC2 {
                // SOF marker found
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                let precision = data[i + 4];
                let height = ((data[i + 5] as u16) << 8) | (data[i + 6] as u16);
                let width = ((data[i + 7] as u16) << 8) | (data[i + 8] as u16);
                let num_components = data[i + 9];

                let mut components = Vec::new();
                for c in 0..num_components as usize {
                    let base = i + 10 + c * 3;
                    let id = data[base];
                    let sampling = data[base + 1];
                    let h_samp = (sampling >> 4) & 0x0F;
                    let v_samp = sampling & 0x0F;
                    let quant_table = data[base + 2];
                    components.push((id, h_samp, v_samp));
                }

                let sof_type = if marker == 0xC0 {
                    0 // baseline
                } else {
                    2 // progressive
                };

                return Some((sof_type, precision, num_components, components));
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    None
}

fn subsampling_from_components(components: &[(u8, u8, u8)]) -> String {
    if components.len() < 3 {
        return "grayscale".to_string();
    }
    let y = &components[0];
    let cb = &components[1];
    let cr = &components[2];

    // Y component sampling factors
    let h_y = y.1;
    let v_y = y.2;

    if h_y == 1 && v_y == 1 {
        "4:4:4".to_string()
    } else if h_y == 2 && v_y == 1 {
        "4:2:2".to_string()
    } else if h_y == 2 && v_y == 2 {
        "4:2:0".to_string()
    } else if h_y == 1 && v_y == 2 {
        "4:4:0".to_string()
    } else {
        format!("{}x{}", h_y, v_y)
    }
}

fn count_scans(data: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            count += 1;
        }
        i += 1;
    }
    count
}

fn main() {
    println!("\n=== Progressive + Subsampling Detailed Analysis ===\n");

    let width = 256;
    let height = 256;
    let quality = 85;
    let rgb = generate_gradient(width, height);

    let configs = [
        ("Baseline 444", JpegMode::Baseline, Subsampling::S444),
        ("Baseline 422", JpegMode::Baseline, Subsampling::S422),
        ("Baseline 440", JpegMode::Baseline, Subsampling::S440),
        ("Baseline 420", JpegMode::Baseline, Subsampling::S420),
        ("Progressive 444", JpegMode::Progressive, Subsampling::S444),
        ("Progressive 422", JpegMode::Progressive, Subsampling::S422),
        ("Progressive 440", JpegMode::Progressive, Subsampling::S440),
        ("Progressive 420", JpegMode::Progressive, Subsampling::S420),
    ];

    println!(
        "{:<20} {:>8} {:>8} {:>10} {:>20}",
        "Config", "Size", "Scans", "Actual", "Components"
    );
    println!("{:-<75}", "");

    for (name, mode, sub) in &configs {
        eprintln!("Testing: {}", name);
        let jpeg = match Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .mode(*mode)
            .subsampling(*sub)
            .optimize_huffman(true)
            .jpegli_quality(Quality::from_quality(quality as f32))
            .encode(&rgb)
        {
            Ok(j) => j,
            Err(e) => {
                println!("{:<20} ENCODE FAILED: {}", name, e);
                continue;
            }
        };

        let scans = count_scans(&jpeg);

        if let Some((sof_type, _precision, _num_comp, components)) = parse_jpeg_sof(&jpeg) {
            let actual_sub = subsampling_from_components(&components);
            let comp_str: String = components
                .iter()
                .map(|(id, h, v)| format!("C{}:{}x{}", id, h, v))
                .collect::<Vec<_>>()
                .join(" ");

            let sof_name = if sof_type == 0 { "SOF0" } else { "SOF2" };

            println!(
                "{:<20} {:>8} {:>8} {:>10} {:>20}",
                name,
                jpeg.len(),
                scans,
                format!("{} {}", sof_name, actual_sub),
                comp_str
            );

            // Save files for inspection
            let filename = format!("/tmp/test_{}.jpg", name.to_lowercase().replace(' ', "_"));
            let _ = fs::write(&filename, &jpeg);
        }
    }

    println!("\n\nFiles saved to /tmp/test_*.jpg for inspection");
    println!("Use: djpeg -v /tmp/test_*.jpg to check with reference decoder");
}
