// Compare the raw scan data bytes between working and failing sizes
// to find exactly where they diverge

use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn find_sos_markers(jpeg: &[u8]) -> Vec<(usize, u8, u8, u8, u8)> {
    // Find all SOS markers and return (offset, ss, se, ah, al)
    let mut markers = Vec::new();
    let mut i = 0;
    while i < jpeg.len() - 10 {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
            // SOS marker
            let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let num_components = jpeg[i + 4];
            // Skip component specs (2 bytes each)
            let spec_offset = i + 5 + (num_components as usize * 2);
            if spec_offset + 2 < jpeg.len() {
                let ss = jpeg[spec_offset];
                let se = jpeg[spec_offset + 1];
                let ah_al = jpeg[spec_offset + 2];
                let ah = ah_al >> 4;
                let al = ah_al & 0x0F;
                let scan_start = i + 2 + len;
                markers.push((scan_start, ss, se, ah, al));
            }
            i += 2 + len;
        } else {
            i += 1;
        }
    }
    markers
}

fn find_scan_end(jpeg: &[u8], start: usize) -> usize {
    // Find end of scan data (next marker or EOF)
    let mut i = start;
    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] != 0x00 && jpeg[i + 1] != 0xFF {
            // Found marker (not stuffed byte)
            return i;
        }
        i += 1;
    }
    jpeg.len()
}

fn main() {
    println!("Comparing progressive scan data between sizes\n");

    // Create both images
    let data_49 = photo_like(49, 49);
    let data_50 = photo_like(50, 50);

    let jpeg_49 = Encoder::new()
        .width(49)
        .height(49)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data_49)
        .expect("encode failed");

    let jpeg_50 = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive)
        .encode(&data_50)
        .expect("encode failed");

    // Find scans in each
    let scans_49 = find_sos_markers(&jpeg_49);
    let scans_50 = find_sos_markers(&jpeg_50);

    println!("49x49 scans:");
    for (i, (offset, ss, se, ah, al)) in scans_49.iter().enumerate() {
        let end = find_scan_end(&jpeg_49, *offset);
        println!(
            "  Scan {}: ss={} se={} ah={} al={} @ offset {} len {}",
            i,
            ss,
            se,
            ah,
            al,
            offset,
            end - offset
        );
    }

    println!("\n50x50 scans:");
    for (i, (offset, ss, se, ah, al)) in scans_50.iter().enumerate() {
        let end = find_scan_end(&jpeg_50, *offset);
        println!(
            "  Scan {}: ss={} se={} ah={} al={} @ offset {} len {}",
            i,
            ss,
            se,
            ah,
            al,
            offset,
            end - offset
        );
    }

    // Focus on the refinement scan (ah=2, al=1) - this is where the problem is
    println!("\n--- Refinement Scan Analysis (ah=2, al=1) ---");

    let refine_49 = scans_49
        .iter()
        .find(|(_, _, _, ah, al)| *ah == 2 && *al == 1);
    let refine_50 = scans_50
        .iter()
        .find(|(_, _, _, ah, al)| *ah == 2 && *al == 1);

    if let (Some(&(off_49, _, _, _, _)), Some(&(off_50, _, _, _, _))) = (refine_49, refine_50) {
        let end_49 = find_scan_end(&jpeg_49, off_49);
        let end_50 = find_scan_end(&jpeg_50, off_50);

        let scan_49 = &jpeg_49[off_49..end_49];
        let scan_50 = &jpeg_50[off_50..end_50];

        println!("Scan 49x49: {} bytes", scan_49.len());
        println!("Scan 50x50: {} bytes", scan_50.len());

        // Show first 100 bytes of each in hex
        println!("\n49x49 first bytes:");
        for (i, chunk) in scan_49
            .iter()
            .take(100)
            .collect::<Vec<_>>()
            .chunks(20)
            .enumerate()
        {
            print!("  {:4}: ", i * 20);
            for b in chunk {
                print!("{:02X} ", b);
            }
            println!();
        }

        println!("\n50x50 first bytes:");
        for (i, chunk) in scan_50
            .iter()
            .take(100)
            .collect::<Vec<_>>()
            .chunks(20)
            .enumerate()
        {
            print!("  {:4}: ", i * 20);
            for b in chunk {
                print!("{:02X} ", b);
            }
            println!();
        }

        // Find first difference
        let min_len = scan_49.len().min(scan_50.len());
        let mut first_diff = None;
        for i in 0..min_len {
            if scan_49[i] != scan_50[i] {
                first_diff = Some(i);
                break;
            }
        }

        if let Some(diff_at) = first_diff {
            println!("\nFirst difference at byte {}", diff_at);
            let context_start = diff_at.saturating_sub(10);
            let context_end = (diff_at + 20).min(min_len);

            println!("49x49 context:");
            for i in context_start..context_end {
                let marker = if i == diff_at { ">>>" } else { "   " };
                println!("{} {:4}: {:02X} {:08b}", marker, i, scan_49[i], scan_49[i]);
            }

            println!("50x50 context:");
            for i in context_start..context_end {
                let marker = if i == diff_at { ">>>" } else { "   " };
                println!("{} {:4}: {:02X} {:08b}", marker, i, scan_50[i], scan_50[i]);
            }
        } else {
            println!("\nNo difference found in first {} bytes!", min_len);
        }
    }

    // Try decoding just the first AC scan (ah=0) to see if that works
    println!("\n--- Testing individual scans ---");

    // Write 50x50 to file for external analysis
    std::fs::write("/tmp/fail_50x50.jpg", &jpeg_50).ok();
    std::fs::write("/tmp/ok_49x49.jpg", &jpeg_49).ok();

    // Use jpeg_decoder
    let res_49 = decode_zune(&jpeg_49[..]);
    let res_50 = decode_zune(&jpeg_50[..]);

    println!(
        "49x49 decode: {}",
        if res_49.is_ok() { "OK" } else { "FAIL" }
    );
    println!(
        "50x50 decode: {}",
        if res_50.is_ok() { "OK" } else { "FAIL" }
    );

    // Let's also check if baseline works for both
    let baseline_49 = Encoder::new()
        .width(49)
        .height(49)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .mode(JpegMode::Baseline)
        .encode(&data_49)
        .expect("encode failed");

    let baseline_50 = Encoder::new()
        .width(50)
        .height(50)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .mode(JpegMode::Baseline)
        .encode(&data_50)
        .expect("encode failed");

    let res_b49 = decode_zune(&baseline_49[..]);
    let res_b50 = decode_zune(&baseline_50[..]);

    println!(
        "49x49 baseline: {}",
        if res_b49.is_ok() { "OK" } else { "FAIL" }
    );
    println!(
        "50x50 baseline: {}",
        if res_b50.is_ok() { "OK" } else { "FAIL" }
    );
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
