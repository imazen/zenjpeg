// Test progressive scans separately to isolate the problem
// We'll encode with different progressive levels to see which scan causes the issue

use jpegli::{Encoder, PixelFormat, Quality};

fn photo_like(w: u32, h: u32) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8))
        .collect()
}

fn main() {
    let size = 50u32;
    let data = photo_like(size, size);

    println!("Testing 50x50 with different progressive levels:\n");

    // Progressive level 0: DC + AC scans, no successive approximation
    // (fewer scans, no refinement)
    let jpeg_level0 = Encoder::new()
        .width(size)
        .height(size)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .progressive_level(0)
        .encode(&data)
        .expect("encode failed");

    let result0 = jpeg_decoder::Decoder::new(&jpeg_level0[..]).decode();
    println!(
        "Level 0 (no refinement): {} ({} bytes)",
        if result0.is_ok() { "OK" } else { "FAIL" },
        jpeg_level0.len()
    );

    // Progressive level 1: Adds more scans but still no SA refinement
    let jpeg_level1 = Encoder::new()
        .width(size)
        .height(size)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .progressive_level(1)
        .encode(&data)
        .expect("encode failed");

    let result1 = jpeg_decoder::Decoder::new(&jpeg_level1[..]).decode();
    println!(
        "Level 1: {} ({} bytes)",
        if result1.is_ok() { "OK" } else { "FAIL" },
        jpeg_level1.len()
    );

    // Progressive level 2: Full progressive with successive approximation (refinement)
    let jpeg_level2 = Encoder::new()
        .width(size)
        .height(size)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(75.0))
        .optimize_huffman(false)
        .progressive_level(2)
        .encode(&data)
        .expect("encode failed");

    let result2 = jpeg_decoder::Decoder::new(&jpeg_level2[..]).decode();
    println!(
        "Level 2 (with refinement): {} ({} bytes)",
        if result2.is_ok() { "OK" } else { "FAIL" },
        jpeg_level2.len()
    );

    // Let's also try to identify which specific scan is problematic
    // by looking at the scan structure of level 2
    println!("\nLevel 2 scan structure:");

    fn find_sos_markers(jpeg: &[u8]) -> Vec<(usize, u8, u8, u8, u8)> {
        let mut markers = Vec::new();
        let mut i = 0;
        while i < jpeg.len() - 10 {
            if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
                let len = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
                let num_components = jpeg[i + 4];
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

    for (i, (offset, ss, se, ah, al)) in find_sos_markers(&jpeg_level2).iter().enumerate() {
        let scan_type = if *ah > 0 {
            "REFINEMENT"
        } else if *ss == 0 && *se == 0 {
            "DC"
        } else {
            "AC first"
        };
        println!(
            "  Scan {}: ss={} se={} ah={} al={} @ {} ({})",
            i, ss, se, ah, al, offset, scan_type
        );
    }

    // Let's decode each prefix of the jpeg to find where it fails
    println!("\nSearching for first failing scan...");

    let scans = find_sos_markers(&jpeg_level2);
    for i in 0..scans.len() {
        let (offset, ss, se, ah, al) = scans[i];

        // Find end of this scan
        let mut end = offset;
        while end < jpeg_level2.len() - 1 {
            if jpeg_level2[end] == 0xFF
                && jpeg_level2[end + 1] != 0x00
                && jpeg_level2[end + 1] != 0xFF
            {
                break;
            }
            end += 1;
        }

        // Skip to after EOI if this is the last scan
        let is_refinement = ah > 0;

        println!(
            "  Scan {} (ss={} se={} ah={} al={}): {} bytes {}",
            i,
            ss,
            se,
            ah,
            al,
            end - offset,
            if is_refinement { "<-- REFINEMENT" } else { "" }
        );
    }
}
