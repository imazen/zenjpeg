//! Compare bitstream for flower image (real-world case)

use std::fs;
use std::process::Command;

fn main() {
    // Load flower PNG
    let png_path = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";
    let decoder = png::Decoder::new(fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let bytes = &buf[..info.buffer_size()];
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => panic!("Unsupported"),
    };

    println!("Image: {}x{}", info.width, info.height);

    // Write PPM for C++
    let ppm_path = "/tmp/flower_compare.ppm";
    {
        use std::io::Write;
        let mut f = fs::File::create(ppm_path).unwrap();
        writeln!(f, "P6").unwrap();
        writeln!(f, "{} {}", info.width, info.height).unwrap();
        writeln!(f, "255").unwrap();
        f.write_all(&rgb).unwrap();
    }

    // Encode with C++
    let cpp_path = "/tmp/cpp_flower_compare.jpg";
    Command::new("/home/lilith/work/jpegli/build/tools/cjpegli")
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p", "0",
            "--fixed_code",
            ppm_path,
            cpp_path,
            "-q", "90",
        ])
        .output()
        .expect("cjpegli failed");

    // Encode with Rust
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(info.width)
        .height(info.height)
        .quality(jpegli::quant::Quality::Traditional(90.0))
        .encode(&rgb)
        .expect("Rust encoding failed");

    let cpp_jpeg = fs::read(cpp_path).unwrap();

    println!("C++ size: {} bytes", cpp_jpeg.len());
    println!("Rust size: {} bytes", rust_jpeg.len());
    println!("Difference: {} bytes ({:+.1}%)",
             rust_jpeg.len() as i64 - cpp_jpeg.len() as i64,
             100.0 * (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64);

    // Count segment sizes
    println!("\n=== C++ Segment Sizes ===");
    let cpp_scan = measure_scan_data(&cpp_jpeg);

    println!("\n=== Rust Segment Sizes ===");
    let rust_scan = measure_scan_data(&rust_jpeg);

    println!("\n=== Scan Data Comparison ===");
    println!("C++ scan data: {} bytes", cpp_scan);
    println!("Rust scan data: {} bytes", rust_scan);
    println!("Scan difference: {} bytes ({:+.1}%)",
             rust_scan as i64 - cpp_scan as i64,
             100.0 * (rust_scan as f64 - cpp_scan as f64) / cpp_scan as f64);

    // The scan data size is the key metric - it's pure entropy coded coefficients

    fs::write("/tmp/rust_flower_compare.jpg", &rust_jpeg).unwrap();
}

fn measure_scan_data(data: &[u8]) -> usize {
    let mut i = 0;
    let mut total_header = 0;
    let mut scan_start = 0;

    while i < data.len() {
        if data[i] == 0xFF && i + 1 < data.len() {
            let marker = data[i + 1];
            let name = match marker {
                0xD8 => "SOI",
                0xE0 => "APP0",
                0xE2 => "APP2",
                0xDB => "DQT",
                0xC0 => "SOF0",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xD9 => "EOI",
                0x00 => { i += 2; continue; }
                _ => "?",
            };

            if marker == 0xD8 {
                println!("  {:#06x}: {} (2 bytes)", i, name);
                total_header += 2;
                i += 2;
            } else if marker == 0xD9 {
                println!("  {:#06x}: {} (2 bytes)", i, name);
                i += 2;
            } else if marker >= 0xC0 && i + 3 < data.len() {
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                println!("  {:#06x}: {} ({} bytes)", i, name, len + 2);
                total_header += len + 2;

                if marker == 0xDA {
                    i += 2 + len;
                    scan_start = i;
                    while i < data.len() - 1 {
                        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
                            break;
                        }
                        i += 1;
                    }
                    let scan_len = i - scan_start;
                    println!("  {:#06x}: SCAN ({} bytes)", scan_start, scan_len);
                    return scan_len;
                } else {
                    i += 2 + len;
                }
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    0
}
