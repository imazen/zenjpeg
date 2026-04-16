use enough::Unstoppable;
use zenjpeg::decode::PixelFormat;
use zenjpeg::decoder::Decoder;

fn main() {
    // Encode a 64x64 solid color JPEG
    let config =
        zenjpeg::encoder::EncoderConfig::ycbcr(85, zenjpeg::encode::ChromaSubsampling::Quarter)
            .progressive(zenjpeg::encode::ProgressiveScanMode::Baseline);
    let pixels: Vec<rgb::RGB8> = (0..64 * 64)
        .map(|_| rgb::RGB8::new(0x40, 0x80, 0xFF))
        .collect();
    let jpeg = config.encode(&pixels, 64, 64).expect("encode");
    eprintln!("JPEG: {} bytes", jpeg.len());

    // decode() — the working path (returns Vec)
    let result = Decoder::new()
        .output_format(PixelFormat::Bgra)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let decode_pixels = result.into_pixels_u8().unwrap();
    eprintln!(
        "decode(): {} bytes ({} per row)",
        decode_pixels.len(),
        decode_pixels.len() / 64
    );

    // decode_into() — the direct path
    let mut dst = vec![0xCCu8; 64 * 64 * 4];
    let written = Decoder::new()
        .output_format(PixelFormat::Bgra)
        .decode_into(&jpeg, PixelFormat::Bgra, &mut dst, Unstoppable)
        .unwrap();
    eprintln!("decode_into(): wrote {written} bytes");

    // Count sentinel
    let sentinel = dst.iter().filter(|&&b| b == 0xCC).count();
    eprintln!("Sentinel 0xCC remaining: {sentinel}/{}", dst.len());

    // Compare
    let mut diffs = 0;
    for i in 0..decode_pixels.len().min(dst.len()) {
        if decode_pixels[i] != dst[i] {
            diffs += 1;
        }
    }
    eprintln!("Byte diffs vs decode(): {diffs}/{}", decode_pixels.len());

    if sentinel > 0 {
        // Find first sentinel
        for y in 0..64usize {
            for x in 0..64 * 4 {
                if dst[y * 64 * 4 + x] == 0xCC {
                    eprintln!("First sentinel at y={y} byte_x={x} (pixel {})", x / 4);
                    return;
                }
            }
        }
    }
}
