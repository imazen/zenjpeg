fn main() {
    let noise: Vec<u8> = (0..64)
        .flat_map(|y| {
            (0..64).flat_map(move |x| {
                let r = ((x * 17 ^ y * 31) % 256) as u8;
                let g = ((x * 13 ^ y * 23) % 256) as u8;
                let b = ((x * 11 ^ y * 19) % 256) as u8;
                [r, g, b]
            })
        })
        .collect();
    
    let mut encoder = png::Encoder::new(std::fs::File::create("/tmp/noise64.png").unwrap(), 64, 64);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&noise).unwrap();
    println!("Wrote /tmp/noise64.png");
}
