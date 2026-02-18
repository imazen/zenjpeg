use enough::Unstoppable;

fn main() {
    let data = std::fs::read("/tmp/cjpegli_prog0_64.jpg").expect("read");
    eprintln!("JPEG level 0: {} bytes", data.len());

    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(&data, Unstoppable).expect("decode failed");
    let pixels = result.into_pixels_u8().expect("pixels");

    let dj_ppm = std::fs::read("/tmp/dj0_64.ppm").expect("read djpegli ppm");
    let mut offset = 0;
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj_ppm[offset] == b'#' {
        while dj_ppm[offset] != b'\n' {
            offset += 1;
        }
        offset += 1;
    }
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    let dj_pixels = &dj_ppm[offset..];

    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for i in 0..pixels.len().min(dj_pixels.len()) {
        let diff = pixels[i].abs_diff(dj_pixels[i]);
        sum_diff += diff as u64;
        if diff > max_diff {
            max_diff = diff;
        }
    }
    let mean_diff = sum_diff as f64 / pixels.len() as f64;
    eprintln!("Level 0: max_diff={}, mean_diff={:.2}", max_diff, mean_diff);
}
