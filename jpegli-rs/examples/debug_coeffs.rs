// Debug the actual coefficient values in progressive scans
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn main() {
    // Create the same RGB gradient
    let rgb_grad: Vec<u8> = (0..64).flat_map(|i| vec![i as u8 * 4, 128, 64]).collect();

    println!("=== Debug Coefficient Values ===\n");

    // We need to intercept the coefficient values during encoding.
    // For now, let's compute what they SHOULD be.

    // Step 1: Convert RGB to YCbCr
    let mut y_pixels = vec![0i16; 64];
    let mut cb_pixels = vec![0i16; 64];
    let mut cr_pixels = vec![0i16; 64];

    for i in 0..64 {
        let r = rgb_grad[i * 3] as f32;
        let g = rgb_grad[i * 3 + 1] as f32;
        let b = rgb_grad[i * 3 + 2] as f32;

        // JFIF BT.601 formula (used by jpegli)
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

        y_pixels[i] = y.round() as i16;
        cb_pixels[i] = cb.round() as i16;
        cr_pixels[i] = cr.round() as i16;
    }

    println!("Y pixels (first 16): {:?}", &y_pixels[..16]);
    println!("Cb pixels (first 16): {:?}", &cb_pixels[..16]);
    println!("Cr pixels (first 16): {:?}", &cr_pixels[..16]);

    // Step 2: Level shift and DCT
    // (This is approximate - jpegli uses its own DCT)
    println!("\nNote: DCT and quantization are done by the encoder.");
    println!("The coefficients that matter are the QUANTIZED values.");

    // Let's just run the encoder and check what scans produce
    println!("\n=== Running Encoder ===\n");

    let encoder = Encoder::new()
        .width(8)
        .height(8)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    let jpeg_data = encoder.encode(&rgb_grad).expect("encode failed");

    // Parse and show data for each scan
    let mut i = 0;
    let mut scan_num = 0;
    while i < jpeg_data.len() - 1 {
        if jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA {
            let len = ((jpeg_data[i + 2] as usize) << 8) | (jpeg_data[i + 3] as usize);
            let num_comp = jpeg_data[i + 4];

            let mut comp_names = Vec::new();
            for c in 0..num_comp as usize {
                let comp_id = jpeg_data[i + 5 + c * 2];
                let name = match comp_id {
                    1 => "Y",
                    2 => "Cb",
                    3 => "Cr",
                    _ => "?",
                };
                comp_names.push(name);
            }

            let base = i + 5 + num_comp as usize * 2;
            let ss = jpeg_data[base];
            let se = jpeg_data[base + 1];
            let ah_al = jpeg_data[base + 2];
            let ah = ah_al >> 4;
            let al = ah_al & 0xF;

            let scan_start = i + 2 + len;
            let mut scan_end = scan_start;
            while scan_end < jpeg_data.len() - 1 {
                if jpeg_data[scan_end] == 0xFF && jpeg_data[scan_end + 1] != 0x00 {
                    break;
                }
                scan_end += 1;
            }

            let scan_type = if ss == 0 && se == 0 {
                if ah > 0 {
                    "DC refine"
                } else {
                    "DC first"
                }
            } else if ah > 0 {
                "AC refine"
            } else {
                "AC first"
            };

            let data = &jpeg_data[scan_start..scan_end];

            println!(
                "Scan {}: {:?} Ss={}-{} Ah={} Al={} [{}]",
                scan_num,
                comp_names.join(","),
                ss,
                se,
                ah,
                al,
                scan_type
            );
            println!("  Data ({} bytes): {:02X?}", data.len(), data);

            scan_num += 1;
            i = scan_end;
        } else {
            i += 1;
        }
    }

    // Compare with a simpler case - grayscale (which works)
    println!("\n=== Grayscale (should work) ===\n");

    let gray_grad: Vec<u8> = (0..64).map(|i| i as u8 * 4).collect();

    let encoder = Encoder::new()
        .width(8)
        .height(8)
        .pixel_format(PixelFormat::Gray)
        .jpegli_quality(Quality::from_quality(90.0))
        .optimize_huffman(false)
        .mode(JpegMode::Progressive);

    let gray_jpeg = encoder.encode(&gray_grad).expect("encode failed");

    let mut i = 0;
    let mut scan_num = 0;
    while i < gray_jpeg.len() - 1 {
        if gray_jpeg[i] == 0xFF && gray_jpeg[i + 1] == 0xDA {
            let len = ((gray_jpeg[i + 2] as usize) << 8) | (gray_jpeg[i + 3] as usize);
            let num_comp = gray_jpeg[i + 4];

            let base = i + 5 + num_comp as usize * 2;
            let ss = gray_jpeg[base];
            let se = gray_jpeg[base + 1];
            let ah_al = gray_jpeg[base + 2];
            let ah = ah_al >> 4;
            let al = ah_al & 0xF;

            let scan_start = i + 2 + len;
            let mut scan_end = scan_start;
            while scan_end < gray_jpeg.len() - 1 {
                if gray_jpeg[scan_end] == 0xFF && gray_jpeg[scan_end + 1] != 0x00 {
                    break;
                }
                scan_end += 1;
            }

            let scan_type = if ss == 0 && se == 0 {
                if ah > 0 {
                    "DC refine"
                } else {
                    "DC first"
                }
            } else if ah > 0 {
                "AC refine"
            } else {
                "AC first"
            };

            let data = &gray_jpeg[scan_start..scan_end];

            println!(
                "Scan {}: Y Ss={}-{} Ah={} Al={} [{}]",
                scan_num, ss, se, ah, al, scan_type
            );
            println!("  Data ({} bytes): {:02X?}", data.len(), data);

            scan_num += 1;
            i = scan_end;
        } else {
            i += 1;
        }
    }

    // Decode test
    println!("\n=== Decode Tests ===");
    match decode_zune(&jpeg_data[..]) {
        Ok(_) => println!("RGB: OK"),
        Err(e) => println!("RGB: FAILED - {:?}", e),
    }
    match decode_zune(&gray_jpeg[..]) {
        Ok(_) => println!("Gray: OK"),
        Err(e) => println!("Gray: FAILED - {:?}", e),
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
