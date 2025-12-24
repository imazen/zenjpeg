//! Debug XYB decoding issue

use std::fs;

fn main() {
    // First create an XYB JPEG using the Rust encoder
    let png_path = "/home/lilith/work/jpegli/testdata/jxl/flower/flower_small.rgb.png";

    // Load PNG
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

    let width = info.width;
    let height = info.height;
    println!("Source image: {}x{}", width, height);

    // Encode YCbCr JPEG
    let ycbcr_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(80.0))
        .encode(&rgb)
        .unwrap();
    fs::write("/tmp/test_ycbcr.jpg", &ycbcr_jpeg).unwrap();
    println!("YCbCr JPEG: {} bytes", ycbcr_jpeg.len());

    // Encode XYB JPEG
    let xyb_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(80.0))
        .use_xyb(true)
        .encode(&rgb)
        .unwrap();
    fs::write("/tmp/test_xyb.jpg", &xyb_jpeg).unwrap();
    println!("XYB JPEG: {} bytes", xyb_jpeg.len());

    // Print markers from both
    println!("\n=== YCbCr JPEG markers ===");
    print_markers(&ycbcr_jpeg);

    println!("\n=== XYB JPEG markers ===");
    print_markers(&xyb_jpeg);

    // Try decoding YCbCr
    println!("\n=== Decoding YCbCr ===");
    match jpegli::Decoder::new().decode(&ycbcr_jpeg) {
        Ok(result) => println!(
            "Success: {}x{}, {} bytes",
            result.width,
            result.height,
            result.data.len()
        ),
        Err(e) => println!("Error: {:?}", e),
    }

    // Try decoding XYB
    println!("\n=== Decoding XYB ===");
    match jpegli::Decoder::new().decode(&xyb_jpeg) {
        Ok(result) => println!(
            "Success: {}x{}, {} bytes",
            result.width,
            result.height,
            result.data.len()
        ),
        Err(e) => println!("Error: {:?}", e),
    }

    // Try read_info for XYB
    println!("\n=== Reading XYB info only ===");
    match jpegli::Decoder::new().read_info(&xyb_jpeg) {
        Ok(info) => println!("Info: {:?}", info),
        Err(e) => println!("Error: {:?}", e),
    }
}

fn print_markers(data: &[u8]) {
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];
            let name = match marker {
                0xD8 => "SOI",
                0xD9 => "EOI",
                0xDB => "DQT",
                0xC0 => "SOF0 (Baseline)",
                0xC2 => "SOF2 (Progressive)",
                0xC4 => "DHT",
                0xDA => "SOS",
                0xE0 => "APP0 (JFIF)",
                0xE1 => "APP1",
                0xE2 => "APP2 (ICC)",
                0xDD => "DRI",
                _ => "???",
            };

            if marker == 0xD8 || marker == 0xD9 {
                println!("  {:04X}: FF {:02X} ({})", i, marker, name);
                i += 2;
            } else if marker == 0xDA {
                // SOS - print scan params
                let len = ((data[i + 2] as u16) << 8) | (data[i + 3] as u16);
                let num_comps = data[i + 4];
                let ss = data[i + 5 + num_comps as usize * 2];
                let se = data[i + 6 + num_comps as usize * 2];
                let ah_al = data[i + 7 + num_comps as usize * 2];
                println!(
                    "  {:04X}: FF {:02X} ({}) len={} comps={} ss={} se={} ah={} al={}",
                    i,
                    marker,
                    name,
                    len,
                    num_comps,
                    ss,
                    se,
                    ah_al >> 4,
                    ah_al & 0x0F
                );
                i += 2 + len as usize;
            } else {
                let len = if i + 3 < data.len() {
                    ((data[i + 2] as u16) << 8) | (data[i + 3] as u16)
                } else {
                    0
                };
                println!("  {:04X}: FF {:02X} ({}) len={}", i, marker, name, len);
                i += 2 + len as usize;
            }
        } else {
            i += 1;
        }
    }
}
