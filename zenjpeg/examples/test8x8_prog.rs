use enough::Unstoppable;

fn main() {
    let data = std::fs::read("/tmp/test8x8_prog.jpg").expect("read");
    eprintln!("JPEG: {} bytes", data.len());

    // Decode with zenjpeg
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(&data, Unstoppable).expect("decode failed");
    let pixels = result.into_pixels_u8().expect("pixels");

    // Read djpegli reference
    let dj_ppm = std::fs::read("/tmp/test8x8_dj.ppm").expect("read djpegli");
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
    eprintln!(
        "max_diff={}, mean_diff={:.2}",
        max_diff,
        sum_diff as f64 / pixels.len() as f64
    );

    // Print all pixels
    for y in 0..8 {
        eprint!("zen {:2}: ", y);
        for x in 0..8 {
            let i = (y * 8 + x) * 3;
            eprint!("({:3},{:3},{:3}) ", pixels[i], pixels[i + 1], pixels[i + 2]);
        }
        eprintln!();
        eprint!("dj  {:2}: ", y);
        for x in 0..8 {
            let i = (y * 8 + x) * 3;
            eprint!(
                "({:3},{:3},{:3}) ",
                dj_pixels[i],
                dj_pixels[i + 1],
                dj_pixels[i + 2]
            );
        }
        eprintln!();
    }
}
