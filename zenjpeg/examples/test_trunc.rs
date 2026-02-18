use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;

fn test_file(n: usize) -> Option<(u8, f64)> {
    let path = format!("/tmp/test32_trunc{n}.jpg");
    let dj_path = format!("/tmp/test32_trunc{n}_dj.ppm");

    let data = std::fs::read(&path).ok()?;
    let dj_data = std::fs::read(&dj_path).ok()?;

    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = match decoder.decode(&data, Unstoppable) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Scan {n}: decode error: {e}");
            return None;
        }
    };
    let pixels = result.into_pixels_u8().expect("pixels");

    // Parse PPM
    let mut offset = 0;
    while dj_data[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj_data[offset] == b'#' {
        while dj_data[offset] != b'\n' {
            offset += 1;
        }
        offset += 1;
    }
    while dj_data[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj_data[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    let dj_pixels = &dj_data[offset..];

    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for i in 0..pixels.len().min(dj_pixels.len()) {
        let diff = pixels[i].abs_diff(dj_pixels[i]);
        sum_diff += diff as u64;
        if diff > max_diff {
            max_diff = diff;
        }
    }
    let mean = sum_diff as f64 / pixels.len() as f64;
    Some((max_diff, mean))
}

fn main() {
    for n in 1..=13 {
        match test_file(n) {
            Some((max, mean)) => eprintln!(
                "After scan {:2}: max_diff={:3}, mean_diff={:.2}",
                n, max, mean
            ),
            None => eprintln!("After scan {:2}: FAILED", n),
        }
    }
}
