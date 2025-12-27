//! Debug decode with tracing

use jpegli::{
    encode::Encoder,
    types::{JpegMode, PixelFormat, Subsampling},
    Quality,
};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(90.0))
        .subsampling(Subsampling::S444)
        .mode(JpegMode::Progressive);

    let encoded = encoder.encode(&pixels).expect("encode");
    println!("Encoded {} bytes", encoded.len());

    // Parse and trace the decode process manually
    // Find SOS markers and their positions
    println!("\n=== SOS markers ===");
    let mut sos_positions = Vec::new();
    let mut i = 0;
    while i < encoded.len() - 1 {
        if encoded[i] == 0xFF && encoded[i + 1] == 0xDA {
            if i + 5 < encoded.len() {
                let len = u16::from_be_bytes([encoded[i + 2], encoded[i + 3]]) as usize;
                let num_comp = encoded[i + 4];

                let base = i + 5 + (num_comp as usize * 2);
                let ss = encoded[base];
                let se = encoded[base + 1];
                let ah_al = encoded[base + 2];
                let ah = ah_al >> 4;
                let al = ah_al & 0x0F;

                // Data starts after SOS header
                let data_start = i + 2 + len;

                println!(
                    "SOS at {:04x}: comps={} Ss={} Se={} Ah={} Al={} data_start={:04x}",
                    i, num_comp, ss, se, ah, al, data_start
                );
                sos_positions.push((data_start, ss, se, ah, al));

                i = data_start;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    // For each scan, show the first few bytes of entropy data
    println!("\n=== Scan entropy data ===");
    for (scan_idx, (start, ss, se, ah, al)) in sos_positions.iter().enumerate() {
        println!(
            "\nScan {} (Ss={} Se={} Ah={} Al={}):",
            scan_idx, ss, se, ah, al
        );

        // Find end of scan (next marker)
        let mut end = *start;
        while end < encoded.len() - 1 {
            if encoded[end] == 0xFF && encoded[end + 1] != 0x00 && encoded[end + 1] != 0xFF {
                if encoded[end + 1] >= 0xD0 && encoded[end + 1] <= 0xD7 {
                    // Restart marker, skip it
                    end += 2;
                } else {
                    break;
                }
            } else {
                end += 1;
            }
        }

        let scan_len = end - start;
        println!(
            "  Data length: {} bytes (from {:04x} to {:04x})",
            scan_len, start, end
        );

        // Show first 32 bytes
        let show_len = scan_len.min(32);
        print!("  First {} bytes: ", show_len);
        for j in 0..show_len {
            print!("{:02x} ", encoded[start + j]);
        }
        println!();

        // Analyze if this is a refinement scan
        if *ah > 0 && *ss > 0 {
            println!("  This is an AC REFINEMENT scan");
        } else if *ah > 0 {
            println!("  This is a DC REFINEMENT scan");
        } else if *ss > 0 {
            println!("  This is an AC FIRST scan");
        } else {
            println!("  This is a DC FIRST scan");
        }
    }

    // Try decode and catch the exact error
    println!("\n=== Attempting decode ===");
    let decoder = jpegli::decode::Decoder::new().output_format(PixelFormat::Rgb);
    match decoder.decode(&encoded) {
        Ok(img) => println!("SUCCESS: {}x{}", img.width, img.height),
        Err(e) => println!("FAIL: {:?}", e),
    }
}
