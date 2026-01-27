use std::fs;
use std::path::Path;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

#[test]
#[ignore = "requires testdata and C++ cjpegli"]
fn compare_rust_cpp_420() {
    let png_path = zenjpeg::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");

    if !png_path.exists() {
        println!("Skipping: test image not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let (pixels, width, height) = load_png(&png_path).expect("load PNG");
    println!("Image: {}x{}", width, height);

    // Encode with Rust 4:2:0
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let rust_jpeg = encode_rgb(width, height, &pixels, &config).expect("encode");
    fs::write("/tmp/rust_420_flower.jpg", &rust_jpeg).unwrap();
    println!("Rust 4:2:0: {} bytes", rust_jpeg.len());

    // Encode with C++ cjpegli
    let cjpegli = match zenjpeg::test_utils::find_cjpegli() {
        Some(p) => p,
        None => {
            println!("Skipping C++ comparison: cjpegli not found. Set CJPEGLI_PATH env var.");
            return;
        }
    };

    let status = std::process::Command::new(&cjpegli)
        .args([
            png_path.to_str().unwrap(),
            "/tmp/cpp_420_flower.jpg",
            "-q",
            "85",
            "--chroma_subsampling=420",
        ])
        .output();

    if let Ok(output) = status {
        if output.status.success() {
            let cpp_jpeg = fs::read("/tmp/cpp_420_flower.jpg").unwrap();
            println!("C++ 4:2:0:  {} bytes", cpp_jpeg.len());

            let diff_pct =
                100.0 * (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64;
            println!("Size difference: {:+.1}%", diff_pct);
        } else {
            println!(
                "C++ encoding failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        println!("Command execution failed for cjpegli at {:?}", cjpegli);
    }

    // Also compare 4:4:4
    let config444 = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
    let rust_444 = encode_rgb(width, height, &pixels, &config444).expect("encode");
    println!("Rust 4:4:4: {} bytes", rust_444.len());

    let reduction = 100.0 * (1.0 - rust_jpeg.len() as f64 / rust_444.len() as f64);
    println!("4:2:0 vs 4:4:4 size reduction: {:.1}%", reduction);
}
