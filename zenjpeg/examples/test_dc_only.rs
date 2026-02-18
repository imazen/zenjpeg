use enough::Unstoppable;
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;

fn main() {
    let data = std::fs::read("/tmp/orig64_trunc1.jpg").expect("read");
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(&data, Unstoppable).expect("decode");
    let pixels = result.into_pixels_u8().expect("pixels");

    // Print first 8 pixels
    eprint!("zenjpeg first 8px: ");
    for i in 0..8 {
        let idx = i * 3;
        eprint!("({},{},{}) ", pixels[idx], pixels[idx + 1], pixels[idx + 2]);
    }
    eprintln!();

    // Read djpegli
    let dj = std::fs::read("/tmp/orig64_trunc1_dj.ppm").expect("dj");
    let mut offset = 0;
    while dj[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj[offset] == b'#' {
        while dj[offset] != b'\n' {
            offset += 1;
        }
        offset += 1;
    }
    while dj[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    while dj[offset] != b'\n' {
        offset += 1;
    }
    offset += 1;
    let dj_px = &dj[offset..];

    eprint!("djpegli first 8px: ");
    for i in 0..8 {
        let idx = i * 3;
        eprint!("({},{},{}) ", dj_px[idx], dj_px[idx + 1], dj_px[idx + 2]);
    }
    eprintln!();

    // For DC-only, all pixels in a block should be identical
    // Check first block (8x8)
    eprintln!("\nFirst block (8x8) zenjpeg row 0: {:?}", &pixels[0..24]);
    eprintln!("First block (8x8) djpegli row 0: {:?}", &dj_px[0..24]);

    // Check if our block is uniform (DC-only should give flat blocks)
    let (r0, g0, b0) = (pixels[0], pixels[1], pixels[2]);
    let mut block_uniform = true;
    for y in 0..8 {
        for x in 0..8 {
            let idx = (y * 64 + x) * 3;
            if pixels[idx] != r0 || pixels[idx + 1] != g0 || pixels[idx + 2] != b0 {
                block_uniform = false;
                break;
            }
        }
    }
    eprintln!("Our first block uniform: {}", block_uniform);

    let (dr0, dg0, db0) = (dj_px[0], dj_px[1], dj_px[2]);
    let mut dj_uniform = true;
    for y in 0..8 {
        for x in 0..8 {
            let idx = (y * 64 + x) * 3;
            if dj_px[idx] != dr0 || dj_px[idx + 1] != dg0 || dj_px[idx + 2] != db0 {
                dj_uniform = false;
                break;
            }
        }
    }
    eprintln!("DJ first block uniform: {}", dj_uniform);
}
