use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;

fn test_file(path: &str, dj_path: &str, label: &str) {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}: read error: {}", label, e);
            return;
        }
    };
    let dj_data = match std::fs::read(dj_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}: dj read error: {}", label, e);
            return;
        }
    };

    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = match decoder.decode(&data, Unstoppable) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: decode error: {}", label, e);
            return;
        }
    };
    let pixels = result.into_pixels_u8().expect("pixels");

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
    let mut worst_idx = 0;
    for i in 0..pixels.len().min(dj_pixels.len()) {
        let diff = pixels[i].abs_diff(dj_pixels[i]);
        sum_diff += diff as u64;
        if diff > max_diff {
            max_diff = diff;
            worst_idx = i;
        }
    }
    let mean = sum_diff as f64 / pixels.len() as f64;
    eprintln!(
        "{}: max_diff={:3}, mean_diff={:.2} (worst at byte {})",
        label, max_diff, mean, worst_idx
    );
}

fn main() {
    for n in 1..=13 {
        test_file(
            &format!("/tmp/orig64_trunc{n}.jpg"),
            &format!("/tmp/orig64_trunc{n}_dj.ppm"),
            &format!("After scan {:2}", n),
        );
    }
}
