use enough::Unstoppable;
use zenjpeg::decoder::Decoder;

fn main() {
    let data = std::fs::read("zenjpeg/tests/testdata/photoshop-444-scrubbed.jpg").unwrap();

    // Scanline: get YCbCr i16 for first MCU row (8 rows)
    let mut reader = Decoder::new()
        .scanline_reader(&data)
        .expect("scanline_reader");
    let sw = reader.width() as usize;
    let mut y_buf = vec![0i16; sw * 8];
    let mut cb_buf = vec![0i16; sw * 8];
    let mut cr_buf = vec![0i16; sw * 8];
    let _ = reader.read_rows_ycbcr_native_i16(&mut y_buf, sw, &mut cb_buf, &mut cr_buf, sw, 1);

    println!("Scanline YCbCr i16 (first 4 pixels):");
    for i in 0..4 {
        println!("  px{i}: Y={} Cb={} Cr={}", y_buf[i], cb_buf[i], cr_buf[i]);
    }

    // Buffered: decode coefficients and manually compute first pixel
    let decoder = Decoder::new();
    let coeffs = decoder.decode_coefficients(&data, Unstoppable).expect("coeff decode");
    
    // First block (8x8) of each component  
    let y_block = coeffs.components[0].block(0);
    let cb_block = coeffs.components[1].block(0);
    let cr_block = coeffs.components[2].block(0);
    
    println!("\nBuffered coefficients block[0] DC:");
    println!("  Y_DC={} Cb_DC={} Cr_DC={}", y_block[0], cb_block[0], cr_block[0]);
    println!("  Y  first 16: {:?}", &y_block[..16]);
    println!("  Cb first 16: {:?}", &cb_block[..16]);
    println!("  Cr first 16: {:?}", &cr_block[..16]);
    
    // Now use the coefficient path to decode to RGB
    // by simulating what to_pixels_fast_i16 does for the first block
    
    // Quant tables are all 1, so dequantized = coefficients
    // IDCT should produce the same Y/Cb/Cr strip values as scanline
    
    // Let's manually compute what the IDCT should give for pixel (0,0):
    // For a DC-only block: pixel = (DC * quant + 4 + 1024) >> 3
    // But this block has AC coefficients too, so let's use the full IDCT
    
    println!("\nDC-only estimate: Y={} Cb={} Cr={}",
        (y_block[0] as i32 + 4 + 1024) >> 3,
        (cb_block[0] as i32 + 4 + 1024) >> 3,
        (cr_block[0] as i32 + 4 + 1024) >> 3);
    
    // Compare buffered RGB decode
    let decoder = Decoder::new();
    let result = decoder.decode(&data, Unstoppable).expect("rgb decode");
    let buf_rgb = result.into_pixels_u8().unwrap();
    
    // Reference
    let mut ref_dec = jpeg_decoder::Decoder::new(&data[..]);
    let ref_rgb = ref_dec.decode().unwrap();
    
    println!("\nFirst pixel comparison:");
    println!("  ref: ({},{},{})", ref_rgb[0], ref_rgb[1], ref_rgb[2]);
    println!("  buf: ({},{},{})", buf_rgb[0], buf_rgb[1], buf_rgb[2]);
    
    // Manually compute expected RGB from scanline YCbCr
    let y = y_buf[0] as f64;
    let cb = cb_buf[0] as f64 - 128.0;
    let cr = cr_buf[0] as f64 - 128.0;
    let r = (y + 1.402 * cr).clamp(0.0, 255.0);
    let g = (y - 0.34414 * cb - 0.71414 * cr).clamp(0.0, 255.0);
    let b = (y + 1.772 * cb).clamp(0.0, 255.0);
    println!("  expected from scan YCbCr: ({:.0},{:.0},{:.0})", r, g, b);
    
    // Compare scanline RGB
    let mut reader = Decoder::new()
        .scanline_reader(&data)
        .expect("scanline_reader");
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut scan_rgb = vec![0u8; stride * h];
    let out = imgref::ImgRefMut::new(&mut scan_rgb, stride, h);
    reader.read_rows_rgb8(out).unwrap();
    println!("  scan: ({},{},{})", scan_rgb[0], scan_rgb[1], scan_rgb[2]);
}
