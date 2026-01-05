use std::fs;

fn main() {
    let path = "/home/lilith/work/codec-corpus/zune/test-images/jpeg/rebuilt_relax_fill_bytes_before_marker.jpg";
    let data = fs::read(path).expect("read file");
    println!("File size: {} bytes", data.len());

    // Parse markers manually
    println!("\n=== Marker structure ===");
    let mut pos = 0;
    while pos < data.len() - 1 {
        if data[pos] == 0xFF {
            let start = pos;
            while pos < data.len() && data[pos] == 0xFF {
                pos += 1;
            }
            let ff_count = pos - start;
            if pos < data.len() {
                let marker_code = data[pos];
                if marker_code != 0x00 {
                    let marker_name = match marker_code {
                        0xD8 => "SOI",
                        0xD9 => "EOI",
                        0xE0..=0xEF => "APPx",
                        0xC0 => "SOF0",
                        0xC2 => "SOF2",
                        0xC4 => "DHT",
                        0xDB => "DQT",
                        0xDD => "DRI",
                        0xDA => "SOS",
                        0xFE => "COM",
                        0xD0..=0xD7 => "RSTx",
                        _ => "????",
                    };
                    // Get marker length if applicable
                    let mut info = format!("{} FF(s)", ff_count);
                    if marker_code != 0xD8
                        && marker_code != 0xD9
                        && !(0xD0..=0xD7).contains(&marker_code)
                    {
                        if pos + 2 < data.len() {
                            let length = ((data[pos + 1] as u16) << 8) | (data[pos + 2] as u16);
                            info = format!("{}, len={}", info, length);
                        }
                    }
                    println!(
                        "0x{:04X}: FF{:02X} {} ({})",
                        start, marker_code, marker_name, info
                    );
                }
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }

    // Now try decoding
    println!("\n=== Attempting decode ===");
    let decoder = jpegli::Decoder::new();
    match decoder.decode(&data) {
        Ok(img) => {
            println!("jpegli: SUCCESS {}x{}", img.width, img.height);
        }
        Err(e) => {
            println!("jpegli: FAILED {:?}", e);
        }
    }

    println!("\n=== zune-jpeg ===");
    let mut ref_dec = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&data[..]));
    match ref_dec.decode() {
        Ok(pixels) => {
            let (w, h) = ref_dec.dimensions().unwrap();
            println!(
                "zune-jpeg: SUCCESS {}x{}, {} bytes",
                w,
                h,
                pixels.len()
            );
        }
        Err(e) => {
            println!("jpeg-decoder: FAILED {:?}", e);
        }
    }
}
