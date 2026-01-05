//! Test progressive + subsampling with frymire.png (1118x1105)

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;
use std::path::Path;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let file = fs::File::open(path).expect("open png");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("next frame");

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("unsupported color type"),
    };

    (rgb, info.0, info.1)
}

fn decode_mozjpeg(data: &[u8]) -> Vec<u8> {
    let d = mozjpeg::Decompress::new_mem(data).unwrap();
    let mut dec = d.rgb().unwrap();
    let mut buf = vec![0u8; dec.width() * dec.height() * 3];
    #[allow(deprecated)]
    let _ = dec.read_scanlines_into::<u8>(&mut buf);
    buf
}

fn main() {
    let png_path = Path::new("/home/lilith/work/codec-corpus/imageflow/test_inputs/frymire.png");
    let (original, width, height) = load_png(png_path);

    println!("=== frymire.png {}x{} ===\n", width, height);
    println!("{:<20} {:>10} {:>10} {:>10}", "Config", "Size", "MaxDiff", "AvgDiff");
    println!("{:-<55}", "");

    let configs = [
        ("Baseline 444", JpegMode::Baseline, Subsampling::S444),
        ("Baseline 420", JpegMode::Baseline, Subsampling::S420),
        ("Progressive 444", JpegMode::Progressive, Subsampling::S444),
        ("Progressive 420", JpegMode::Progressive, Subsampling::S420),
    ];

    for (name, mode, sub) in &configs {
        let jpeg = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .mode(*mode)
            .subsampling(*sub)
            .optimize_huffman(true)
            .jpegli_quality(Quality::from_quality(85.0))
            .encode(&original)
            .unwrap();

        let decoded = decode_mozjpeg(&jpeg);

        let mut max_diff = 0u8;
        let mut sum_diff = 0u64;
        for (&o, &d) in original.iter().zip(decoded.iter()) {
            let diff = (o as i16 - d as i16).unsigned_abs() as u8;
            max_diff = max_diff.max(diff);
            sum_diff += diff as u64;
        }
        let avg_diff = sum_diff as f64 / original.len() as f64;

        let status = if avg_diff > 20.0 { " <-- BAD" } else { "" };
        println!("{:<20} {:>10} {:>10} {:>10.2}{}", name, jpeg.len(), max_diff, avg_diff, status);

        // Save for inspection
        let safe_name = name.replace(' ', "_").to_lowercase();
        fs::write(format!("/tmp/frymire_{}.jpg", safe_name), &jpeg).unwrap();
    }

    // C++ comparison
    println!("\n=== C++ Reference ===\n");
    let cjpegli = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli";

    for (sub, prog) in [("444", "0"), ("420", "0"), ("444", "2"), ("420", "2")] {
        let name = format!("{} p{}", sub, prog);
        let out = format!("/tmp/frymire_cpp_{}_{}.jpg", sub, prog);
        let _ = std::process::Command::new(cjpegli)
            .args([png_path.to_str().unwrap(), &out, "-q", "85",
                   &format!("--chroma_subsampling={}", sub),
                   &format!("--progressive_level={}", prog)])
            .output();

        if let Ok(jpeg) = fs::read(&out) {
            let decoded = decode_mozjpeg(&jpeg);
            let mut max_diff = 0u8;
            let mut sum_diff = 0u64;
            for (&o, &d) in original.iter().zip(decoded.iter()) {
                let diff = (o as i16 - d as i16).unsigned_abs() as u8;
                max_diff = max_diff.max(diff);
                sum_diff += diff as u64;
            }
            let avg_diff = sum_diff as f64 / original.len() as f64;
            println!("C++ {:<16} {:>10} {:>10} {:>10.2}", name, jpeg.len(), max_diff, avg_diff);
        }
    }
}
