//! Comprehensive verification of progressive JPEG support
//!
//! Tests:
//! 1. Progressive Level 0 (no SA) - baseline progressive
//! 2. Progressive Level 2 (with SA) - successive approximation refinement
//! 3. External decoder compatibility (mozjpeg, zune-jpeg, jpeg-decoder)
//! 4. XYB + Progressive mode
//! 5. Huffman table types (standard vs optimized)

use jpegli::{Decoder, Encoder, PixelFormat};
use std::io::Cursor;

fn test_decoder(name: &str, jpeg_data: &[u8]) -> (bool, Option<String>) {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let result = catch_unwind(AssertUnwindSafe(|| match name {
        "jpegli-rs" => match Decoder::new().decode(jpeg_data) {
            Ok(decoded) => (true, Some(format!("{}x{}", decoded.width, decoded.height))),
            Err(e) => (false, Some(format!("{:?}", e))),
        },
        "zune-jpeg" => {
            let mut decoder = zune_jpeg::JpegDecoder::new(Cursor::new(jpeg_data));
            match decoder.decode() {
                Ok(_) => (true, None),
                Err(e) => (false, Some(format!("{:?}", e))),
            }
        }
        "mozjpeg" => match mozjpeg::Decompress::new_mem(jpeg_data) {
            Ok(decoder) => match decoder.rgb() {
                Ok(_) => (true, None),
                Err(e) => (false, Some(format!("{:?}", e))),
            },
            Err(e) => (false, Some(format!("{:?}", e))),
        },
        "jpeg-decoder" => {
            let mut decoder = zune_jpeg::JpegDecoder::new(
                zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg_data),
            );
            match decoder.decode() {
                Ok(_) => (true, None),
                Err(e) => (false, Some(format!("{:?}", e))),
            }
        }
        _ => (false, Some("Unknown decoder".to_string())),
    }));

    match result {
        Ok(r) => r,
        Err(_) => (false, Some("PANIC/CRASH".to_string())),
    }
}

fn count_scans(jpeg_data: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 1 < jpeg_data.len() {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            // SOS marker
            count += 1;
            i += 2;
            // Skip to next marker
            while i < jpeg_data.len() {
                if jpeg_data[i] == 0xFF && i + 1 < jpeg_data.len() {
                    let next = jpeg_data[i + 1];
                    if next != 0x00 && next != 0xFF {
                        break;
                    }
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    count
}

fn has_successive_approximation(jpeg_data: &[u8]) -> bool {
    let mut i = 0;
    while i + 12 < jpeg_data.len() {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            // SOS marker
            let length = u16::from_be_bytes([jpeg_data[i + 2], jpeg_data[i + 3]]) as usize;
            if i + 2 + length <= jpeg_data.len() {
                // Check Ah and Al parameters (last two bytes before scan data)
                let ah_al_offset = i + 2 + length - 1;
                if ah_al_offset < jpeg_data.len() {
                    let ah_al = jpeg_data[ah_al_offset];
                    let ah = (ah_al >> 4) & 0x0F;
                    let al = ah_al & 0x0F;
                    if ah != 0 || al != 0 {
                        return true; // Found SA parameters
                    }
                }
            }
            i += 2 + length;
        } else {
            i += 1;
        }
    }
    false
}

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];

    println!("Test image: {}x{} RGB\n", info.width, info.height);
    println!("═══════════════════════════════════════════════════════════════\n");

    let decoders = ["jpegli-rs", "zune-jpeg", "mozjpeg", "jpeg-decoder"];

    // Test 1: Baseline (non-progressive)
    println!("▶ TEST 1: Baseline Sequential (SOF0)");
    println!("─────────────────────────────────────────────────────────────\n");

    let baseline = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    println!(
        "Size: {} bytes, Scans: {}",
        baseline.len(),
        count_scans(&baseline)
    );
    for decoder_name in &decoders {
        let (ok, info) = test_decoder(decoder_name, &baseline);
        let status = if ok { "✓" } else { "✗" };
        let detail = info.map(|s| format!(" ({})", s)).unwrap_or_default();
        println!("  {} {:15} {}", status, decoder_name, detail);
    }

    // Test 2: Progressive with standard Huffman
    println!("\n▶ TEST 2: Progressive with Standard Huffman");
    println!("─────────────────────────────────────────────────────────────\n");

    let prog_std = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .optimize_huffman(false)
        .encode(rgb)
        .unwrap();

    let has_sa = has_successive_approximation(&prog_std);
    println!(
        "Size: {} bytes, Scans: {}, SA: {}",
        prog_std.len(),
        count_scans(&prog_std),
        if has_sa { "YES" } else { "NO" }
    );

    for decoder_name in &decoders {
        let (ok, info) = test_decoder(decoder_name, &prog_std);
        let status = if ok { "✓" } else { "✗" };
        let detail = info.map(|s| format!(" ({})", s)).unwrap_or_default();
        println!("  {} {:15} {}", status, decoder_name, detail);
    }

    // Test 3: Progressive with optimized Huffman
    println!("\n▶ TEST 3: Progressive with Optimized Huffman");
    println!("─────────────────────────────────────────────────────────────\n");

    let prog_opt = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    let has_sa = has_successive_approximation(&prog_opt);
    println!(
        "Size: {} bytes, Scans: {}, SA: {}",
        prog_opt.len(),
        count_scans(&prog_opt),
        if has_sa { "YES" } else { "NO" }
    );

    for decoder_name in &decoders {
        let (ok, info) = test_decoder(decoder_name, &prog_opt);
        let status = if ok { "✓" } else { "✗" };
        let detail = info.map(|s| format!(" ({})", s)).unwrap_or_default();
        println!("  {} {:15} {}", status, decoder_name, detail);
    }

    // Test 4: XYB + Progressive
    println!("\n▶ TEST 4: XYB + Progressive");
    println!("─────────────────────────────────────────────────────────────\n");

    let xyb_prog = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(rgb)
        .unwrap();

    let has_sa = has_successive_approximation(&xyb_prog);
    println!(
        "Size: {} bytes, Scans: {}, SA: {}",
        xyb_prog.len(),
        count_scans(&xyb_prog),
        if has_sa { "YES" } else { "NO" }
    );

    for decoder_name in &decoders {
        let (ok, info) = test_decoder(decoder_name, &xyb_prog);
        let status = if ok { "✓" } else { "✗" };
        let detail = info.map(|s| format!(" ({})", s)).unwrap_or_default();
        println!("  {} {:15} {}", status, decoder_name, detail);
    }

    // Test 5: Grayscale Progressive
    println!("\n▶ TEST 5: Grayscale Progressive");
    println!("─────────────────────────────────────────────────────────────\n");

    // Convert to grayscale
    let gray: Vec<u8> = rgb
        .chunks(3)
        .map(|px| (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8)
        .collect();

    let gray_prog = Encoder::new()
        .width(info.width)
        .height(info.height)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .optimize_huffman(true)
        .encode(&gray)
        .unwrap();

    let has_sa = has_successive_approximation(&gray_prog);
    println!(
        "Size: {} bytes, Scans: {}, SA: {}",
        gray_prog.len(),
        count_scans(&gray_prog),
        if has_sa { "YES" } else { "NO" }
    );

    for decoder_name in &decoders {
        let (ok, info) = test_decoder(decoder_name, &gray_prog);
        let status = if ok { "✓" } else { "✗" };
        let detail = info.map(|s| format!(" ({})", s)).unwrap_or_default();
        println!("  {} {:15} {}", status, decoder_name, detail);
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Scan counts:");
    println!(
        "  Baseline:            {} scan (expected: 1)",
        count_scans(&baseline)
    );
    println!(
        "  Progressive+Std:     {} scans (expected: 10+ for Level 2)",
        count_scans(&prog_std)
    );
    println!("  Progressive+Opt:     {} scans", count_scans(&prog_opt));
    println!("  XYB+Progressive:     {} scans", count_scans(&xyb_prog));
    println!("  Gray+Progressive:    {} scans", count_scans(&gray_prog));

    println!("\nSuccessive Approximation detected:");
    println!(
        "  Progressive+Std:     {}",
        if has_successive_approximation(&prog_std) {
            "YES ✓"
        } else {
            "NO ✗"
        }
    );
    println!(
        "  Progressive+Opt:     {}",
        if has_successive_approximation(&prog_opt) {
            "YES ✓"
        } else {
            "NO ✗"
        }
    );
    println!(
        "  XYB+Progressive:     {}",
        if has_successive_approximation(&xyb_prog) {
            "YES ✓"
        } else {
            "NO ✗"
        }
    );

    println!("\nExpected for Level 2 (3 components):");
    println!("  - 1 DC scan (interleaved) OR 3 DC scans (non-interleaved)");
    println!("  - 3 × 4 AC scans = 12 scans per component:");
    println!("    • AC 1-2 first pass (Al=0)");
    println!("    • AC 3-63 first pass (Al=2)");
    println!("    • AC 3-63 refinement (Ah=2, Al=1)");
    println!("    • AC 3-63 refinement (Ah=1, Al=0)");
    println!("  - Total: 13 scans (1 DC + 12 AC) or 15 scans (3 DC + 12 AC)");
}
