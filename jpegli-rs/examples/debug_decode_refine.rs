// Debug decoding of progressive AC refinement
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};
use std::io::Cursor;

fn gray_photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn find_scan_headers(data: &[u8]) -> Vec<(usize, u8, u8, u8, u8)> {
    // Returns (offset, ss, se, ah, al) for each SOS
    let mut scans = Vec::new();
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            let length = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
            let ns = data[i + 4];
            let ss = data[i + 5 + ns as usize * 2];
            let se = data[i + 5 + ns as usize * 2 + 1];
            let ahl = data[i + 5 + ns as usize * 2 + 2];
            let ah = ahl >> 4;
            let al = ahl & 0x0F;
            scans.push((i + 2 + length, ss, se, ah, al));
        }
        i += 1;
    }
    scans
}

fn main() {
    let data_50 = gray_photo_like(50, 50);

    println!("Encoding 50x50...");
    std::env::set_var("DEBUG_REFINE_SYMBOLS", "1");

    let jpeg_50 = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data_50)
        .expect("encode failed");

    println!("\nJPEG size: {} bytes", jpeg_50.len());

    // Find scans
    let scans = find_scan_headers(&jpeg_50);
    println!("\nScans:");
    for (i, (offset, ss, se, ah, al)) in scans.iter().enumerate() {
        let scan_type = if *ss == 0 && *se == 0 {
            if *ah == 0 {
                "DC first"
            } else {
                "DC refine"
            }
        } else {
            if *ah == 0 {
                "AC first"
            } else {
                "AC refine"
            }
        };
        println!(
            "  Scan {}: offset={:#X}, ss={}, se={}, ah={}, al={} [{}]",
            i + 1,
            offset,
            ss,
            se,
            ah,
            al,
            scan_type
        );
    }

    // Decode with external decoder
    println!("\n=== Decode with jpeg_decoder ===");
    match jpeg_decoder::Decoder::new(&jpeg_50[..]).decode() {
        Ok(_) => println!("decode OK"),
        Err(e) => println!("decode FAILED: {:?}", e),
    }

    // Try our own decoder
    println!("\n=== Decode with our decoder ===");
    match jpegli::Decoder::new().decode(&jpeg_50) {
        Ok(result) => {
            println!("decode OK, {} pixels", result.data.len());
        }
        Err(e) => println!("decode FAILED: {:?}", e),
    }
}
