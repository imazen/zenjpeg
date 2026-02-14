//! Single-decode profiling binary for callgrind.
//!
//! Generates a progressive JPEG from a test pattern and decodes it once.
//! Run under callgrind to get per-function instruction counts:
//!
//! ```sh
//! cargo build --release --features decoder --example profile_progressive_decode
//! valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind-prog-decode.out \
//!     target/release/examples/profile_progressive_decode 2048
//! kcachegrind /tmp/callgrind-prog-decode.out
//! ```

#[cfg(not(feature = "decoder"))]
fn main() {
    eprintln!("ERROR: Run with --features decoder");
}

#[cfg(feature = "decoder")]
fn main() {
    use enough::Unstoppable;
    use zenjpeg::decode::Decoder;
    use zenjpeg::decoder::PixelFormat;
    use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

    let size: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    eprintln!("Generating {size}x{size} progressive JPEG...");

    // Noise+patches pattern (same as benchmark) for realistic DCT coefficients
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
    drop(data); // Free source pixels before decode

    eprintln!(
        "JPEG size: {} KB, decoding {size}x{size}...",
        jpeg.len() / 1024
    );

    // Save JPEG if --save <path> is passed
    if let Some(save_path) = std::env::args().nth(2).filter(|s| s == "--save").and_then(|_| std::env::args().nth(3)) {
        std::fs::write(&save_path, &jpeg).expect("failed to write JPEG");
        eprintln!("Saved to {save_path}");
    }

    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder.decode(&jpeg, Unstoppable).expect("decode failed");
    let pixels = result.into_pixels_u8().unwrap();

    eprintln!("Decoded {} bytes of RGB output", pixels.len());
}
