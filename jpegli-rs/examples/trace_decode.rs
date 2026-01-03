//! Trace the decoding process to find where it fails

use jpegli::encode::Encoder;
use jpegli::quant::Quality;

fn main() {
    // Create the 64x64 gradient image that fails
    let width = 64u32;
    let height = 64u32;
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = ((x * 4) % 256) as u8;
            rgb[i + 1] = ((y * 4) % 256) as u8;
            rgb[i + 2] = 128;
        }
    }

    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true);

    let jpeg = encoder.encode(&rgb).expect("encode");

    // Parse JPEG manually to extract sampling factors
    let mut pos = 2; // Skip SOI
    let mut h_samp = [0u8; 4];
    let mut v_samp = [0u8; 4];
    let mut num_components = 3;

    while pos < jpeg.len() - 1 {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = jpeg[pos + 1];
        pos += 2;

        if marker == 0xC0 || marker == 0xC2 {
            // SOF
            num_components = jpeg[pos + 7];
            for i in 0..num_components as usize {
                let offset = pos + 8 + i * 3;
                let sampling = jpeg[offset + 1];
                h_samp[i] = sampling >> 4;
                v_samp[i] = sampling & 0x0F;
                println!("Component {}: {}x{} sampling", i, h_samp[i], v_samp[i]);
            }
            break;
        } else if marker >= 0xC0 && marker <= 0xFE && marker != 0xD8 && marker != 0xD9 {
            // Skip marker with length
            let len = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
            pos += len;
        }
    }

    // Calculate correct MCU dimensions
    let max_h = *h_samp[..num_components as usize].iter().max().unwrap();
    let max_v = *v_samp[..num_components as usize].iter().max().unwrap();

    let mcu_width = (max_h as usize) * 8;
    let mcu_height = (max_v as usize) * 8;
    let mcu_cols = (width as usize + mcu_width - 1) / mcu_width;
    let mcu_rows = (height as usize + mcu_height - 1) / mcu_height;

    println!("\nMCU: {}x{} pixels", mcu_width, mcu_height);
    println!(
        "MCU grid: {}x{} = {} MCUs",
        mcu_cols,
        mcu_rows,
        mcu_cols * mcu_rows
    );

    let mut blocks_per_mcu = 0;
    for i in 0..num_components as usize {
        let blocks = (h_samp[i] as usize) * (v_samp[i] as usize);
        blocks_per_mcu += blocks;
        println!("Component {} contributes {} blocks per MCU", i, blocks);
    }
    println!("Total blocks per MCU: {}", blocks_per_mcu);
    println!("Total blocks: {}", mcu_cols * mcu_rows * blocks_per_mcu);

    // Try to decode with jpeg-decoder to get reference output
    println!(
        "\njpeg-decoder: {}",
        match jpeg_decoder::Decoder::new(&jpeg[..]).decode() {
            Ok(_) => "OK",
            Err(e) => {
                println!("ERROR: {:?}", e);
                "FAIL"
            }
        }
    );

    // Now try native decoder and see where it fails
    use jpegli::decode::Decoder;
    print!("\nNative decoder: ");
    match Decoder::new().decode(&jpeg) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }

    // Let's try comparing what happens with a working pattern
    println!("\n=== Testing with 'solid' pattern (should work) ===");
    let solid = vec![128u8; (width * height * 3) as usize];
    let encoder_solid = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(90.0))
        .use_xyb(true);
    let jpeg_solid = encoder_solid.encode(&solid).expect("encode solid");
    println!("Solid JPEG: {} bytes", jpeg_solid.len());
    println!(
        "jpeg-decoder: {}",
        match jpeg_decoder::Decoder::new(&jpeg_solid[..]).decode() {
            Ok(_) => "OK",
            Err(_) => "FAIL",
        }
    );
    print!("Native decoder: ");
    match Decoder::new().decode(&jpeg_solid) {
        Ok(_) => println!("OK"),
        Err(e) => println!("FAIL - {:?}", e),
    }

    // Save both for comparison
    std::fs::write("/tmp/gradient_64.jpg", &jpeg).unwrap();
    std::fs::write("/tmp/solid_64.jpg", &jpeg_solid).unwrap();
    println!("\nSaved /tmp/gradient_64.jpg and /tmp/solid_64.jpg");
}
