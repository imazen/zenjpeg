//! Debug aq_strength values at Q100

use jpegli::adaptive_quant::compute_aq_strength_map;
use std::path::Path;

fn main() {
    let test_img =
        "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";
    if !Path::new(test_img).exists() {
        println!("Test image not found");
        return;
    }

    // Load PNG
    let png_data = std::fs::read(test_img).unwrap();
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let width = info.width as usize;
    let height = info.height as usize;

    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    // Extract Y plane (BT.601)
    let y_plane: Vec<f32> = rgb
        .chunks(3)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();

    // Compute aq_strength map at Q100 (y_quant_01 = 1 for all-1s quant table)
    let y_quant_01: u16 = 1;
    let aq_map = compute_aq_strength_map(&y_plane, width, height, y_quant_01);

    // Collect all aq_strength values
    let blocks_x = (width + 7) / 8;
    let blocks_y = (height + 7) / 8;
    let mut values: Vec<f32> = Vec::new();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            values.push(aq_map.get(bx, by));
        }
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = values.first().unwrap();
    let max = values.last().unwrap();
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    let median = values[values.len() / 2];

    println!("=== aq_strength statistics for flower_small at Q100 ===");
    println!("Image: {}x{}, {} blocks", width, height, values.len());
    println!("min: {:.6}", min);
    println!("max: {:.6}", max);
    println!("mean: {:.6}", mean);
    println!("median: {:.6}", median);

    // Distribution
    println!("\nDistribution:");
    let buckets = [
        0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.30,
    ];
    for i in 0..buckets.len() - 1 {
        let count = values
            .iter()
            .filter(|&&v| v >= buckets[i] && v < buckets[i + 1])
            .count();
        let pct = count as f32 / values.len() as f32 * 100.0;
        println!(
            "  [{:.2}, {:.2}): {} ({:.1}%)",
            buckets[i],
            buckets[i + 1],
            count,
            pct
        );
    }
}
