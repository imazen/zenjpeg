//! Debug quant_field values

use std::path::Path;

fn main() {
    let test_img = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";
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

    // Need to access internal compute stages
    // For now, let's just check what aq_strength we get and work backwards

    // aq_strength = (0.6 / quant_field - 1.0).max(0.0)
    // So: quant_field = 0.6 / (aq_strength + 1.0)

    let aq_strength = 0.486;
    let quant_field = 0.6 / (aq_strength + 1.0);

    println!("Given aq_strength = {:.4}", aq_strength);
    println!(
        "  quant_field = 0.6 / ({:.4} + 1) = {:.4}",
        aq_strength, quant_field
    );

    println!("\nExpected from C++:");
    let cpp_aq_mean = 0.08;
    let cpp_qf = 0.6 / (cpp_aq_mean + 1.0);
    println!("  aq_strength ≈ {:.4}", cpp_aq_mean);
    println!("  quant_field ≈ {:.4}", cpp_qf);

    println!("\nOur quant_field is {:.1}× too low", cpp_qf / quant_field);
    println!(
        "This causes aq_strength to be {:.1}× too high",
        aq_strength / cpp_aq_mean
    );
}
