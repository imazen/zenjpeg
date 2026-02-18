use enough::Unstoppable;

fn main() {
    let data = std::fs::read("/tmp/cjpegli_prog_64.jpg").expect("read cjpegli progressive");
    eprintln!("JPEG: {} bytes", data.len());

    // Decode with zenjpeg
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(&data, Unstoppable).expect("decode failed");
    eprintln!("Decoded: {}x{}", result.width(), result.height());
    let pixels = result.into_pixels_u8().expect("pixel conversion");

    // Read djpegli reference
    let dj_ppm = std::fs::read("/tmp/dj_64.ppm").expect("read djpegli ppm");
    // Parse PPM (skip header lines until data)
    let mut offset = 0;
    // Skip "P6\n"
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    // Skip comments
    while dj_ppm[offset] == b'#' {
        while dj_ppm[offset] != b'\n' {
            offset += 1;
        }
        offset += 1;
    }
    // Skip dimensions line
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    // Skip maxval line
    while dj_ppm[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    let dj_pixels = &dj_ppm[offset..];

    eprintln!(
        "zenjpeg: {} bytes, djpegli: {} bytes",
        pixels.len(),
        dj_pixels.len()
    );

    // Compare pixel by pixel
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    let mut worst_idx = 0;
    for i in 0..pixels.len().min(dj_pixels.len()) {
        let diff = pixels[i].abs_diff(dj_pixels[i]);
        sum_diff += diff as u64;
        if diff > max_diff {
            max_diff = diff;
            worst_idx = i;
        }
    }
    let mean_diff = sum_diff as f64 / pixels.len() as f64;
    let worst_pixel = worst_idx / 3;
    let worst_channel = worst_idx % 3;
    let px = worst_pixel % 64;
    let py = worst_pixel / 64;
    eprintln!(
        "Max diff: {} at pixel ({},{}) channel {} (zen={}, dj={})",
        max_diff, px, py, worst_channel, pixels[worst_idx], dj_pixels[worst_idx]
    );
    eprintln!("Mean diff: {:.2}", mean_diff);

    // Print first 8 pixels of each row for visual comparison
    for row in 0..8 {
        eprint!("Row {:2} zen: ", row);
        for col in 0..8 {
            let idx = (row * 64 + col) * 3;
            eprint!(
                "({:3},{:3},{:3}) ",
                pixels[idx],
                pixels[idx + 1],
                pixels[idx + 2]
            );
        }
        eprintln!();
        eprint!("Row {:2} dj:  ", row);
        for col in 0..8 {
            let idx = (row * 64 + col) * 3;
            eprint!(
                "({:3},{:3},{:3}) ",
                dj_pixels[idx],
                dj_pixels[idx + 1],
                dj_pixels[idx + 2]
            );
        }
        eprintln!();
    }
}
