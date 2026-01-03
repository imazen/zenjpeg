use std::fs;

fn main() {
    let path = "/home/lilith/work/codec-corpus/jpeg-conformance/valid/partial_progressive.jpg";
    let data = fs::read(path).expect("read file");
    println!("File size: {} bytes", data.len());

    // Check the ICC profile header
    let mut pos = 0;
    while pos < data.len() - 1 {
        if data[pos] == 0xFF && data[pos + 1] == 0xE2 {
            let len = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
            println!("APP2 at 0x{:04X}, len={}", pos, len);
            
            // Check ICC_PROFILE signature
            let sig_start = pos + 4;
            if sig_start + 14 <= data.len() {
                let sig = &data[sig_start..sig_start + 12];
                println!("Signature: {:?}", String::from_utf8_lossy(sig));
                
                // Chunk number and total
                let chunk_num = data[sig_start + 12];
                let chunk_total = data[sig_start + 13];
                println!("Chunk {} of {}", chunk_num, chunk_total);
                
                // ICC profile header starts at sig_start + 14
                let icc_start = sig_start + 14;
                if icc_start + 128 <= data.len() {
                    let icc_size = u32::from_be_bytes([
                        data[icc_start], data[icc_start + 1],
                        data[icc_start + 2], data[icc_start + 3]
                    ]);
                    println!("ICC profile size in header: {}", icc_size);
                    println!("Available ICC data: {}", len - 16); // len - marker overhead
                    
                    // Color space at offset 16
                    let color_space = &data[icc_start + 16..icc_start + 20];
                    println!("Color space: {:?}", String::from_utf8_lossy(color_space));
                    
                    // PCS at offset 20
                    let pcs = &data[icc_start + 20..icc_start + 24];
                    println!("PCS: {:?}", String::from_utf8_lossy(pcs));
                    
                    // Profile class at offset 12
                    let class = &data[icc_start + 12..icc_start + 16];
                    println!("Device class: {:?}", String::from_utf8_lossy(class));
                }
            }
            break;
        }
        pos += 1;
    }

    // Try decoding with ICC disabled
    println!("\n=== jpegli decode (ICC disabled) ===");
    let decoder = jpegli::Decoder::new().apply_icc(false);
    match decoder.decode(&data) {
        Ok(img) => println!("SUCCESS: {}x{}", img.width, img.height),
        Err(e) => println!("FAILED: {:?}", e),
    }
    
    // Try with ICC enabled
    println!("\n=== jpegli decode (ICC enabled) ===");
    let decoder = jpegli::Decoder::new().apply_icc(true);
    match decoder.decode(&data) {
        Ok(img) => println!("SUCCESS: {}x{}", img.width, img.height),
        Err(e) => println!("FAILED: {:?}", e),
    }
}
