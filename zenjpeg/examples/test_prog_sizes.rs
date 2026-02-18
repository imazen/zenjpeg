use enough::Unstoppable;

fn test_file(path: &str) {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}: read error: {}", path, e);
            return;
        }
    };

    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = match decoder.decode(&data, Unstoppable) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: decode error: {}", path, e);
            return;
        }
    };
    let w = result.width() as usize;
    let h = result.height() as usize;
    let pixels = result.into_pixels_u8().expect("pixels");

    let dj_path = path.replace("_prog.jpg", "_dj.ppm");
    let dj_ppm = match std::fs::read(&dj_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}: djpegli read error: {}", dj_path, e);
            return;
        }
    };

    // Parse PPM
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
    eprintln!(
        "{}x{}: max_diff={}, mean_diff={:.2}",
        w, h, max_diff, mean_diff
    );
}

fn main() {
    for sz in [8, 16, 32, 48, 64] {
        test_file(&format!("/tmp/test{sz}_prog.jpg"));
    }
}
