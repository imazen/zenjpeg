//! zune-jpeg decode profiling binary for callgrind comparison.
//!
//! Generates the same progressive JPEG as profile_progressive_decode and decodes
//! it with zune-jpeg, for direct instruction count comparison.
//!
//! ```sh
//! cargo build --release --example profile_zune_decode
//! valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind-zune-2048.out \
//!     target/release/examples/profile_zune_decode 2048
//! ```

use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let size: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    eprintln!("Generating {size}x{size} progressive JPEG...");

    let mut data = vec![0u8; (size * size * 3) as usize];
    for y in 0..size as usize {
        for x in 0..size as usize {
            let idx = (y * size as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;

            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let noise = (h >> 24) as u8;

            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / size as usize) as u8;
                    data[idx + 1] = ((y * 255) / size as usize) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255 - edge;
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
    let mut enc = config
        .encode_from_bytes(size, size, PixelLayout::Rgb8Srgb)
        .expect("encoder creation");
    enc.push_packed(&data, Unstoppable).expect("push");
    let jpeg = enc.finish().expect("encoding");
    drop(data);

    eprintln!(
        "JPEG size: {} KB, decoding {size}x{size} with zune-jpeg...",
        jpeg.len() / 1024
    );

    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let cursor = std::io::Cursor::new(&jpeg);
    let mut decoder = JpegDecoder::new_with_options(cursor, options);
    let pixels = decoder.decode().expect("zune decode failed");

    eprintln!("Decoded {} bytes of RGB output", pixels.len());
}
