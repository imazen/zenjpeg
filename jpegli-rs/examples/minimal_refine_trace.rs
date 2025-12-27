// Minimal trace of AC refinement encoding
use jpegli::{types::JpegMode, Encoder, PixelFormat, Quality};

fn test_size(size: u32, pattern: &str, optimize: bool) -> bool {
    let data: Vec<u8> = match pattern {
        "gradient" => (0..size * size)
            .map(|i| {
                let x = i % size;
                let y = i / size;
                ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8
            })
            .collect(),
        "flat" => vec![128u8; (size * size) as usize],
        "noise" => (0..size * size)
            .map(|i| {
                // Simple pseudo-random
                let v = i.wrapping_mul(1103515245).wrapping_add(12345);
                (v >> 16) as u8
            })
            .collect(),
        _ => vec![128u8; (size * size) as usize],
    };

    let jpeg = Encoder::new()
        .width(size)
        .height(size)
        .pixel_format(PixelFormat::Gray)
        .quality(Quality::from_quality(75.0))
        .optimize_huffman(optimize)
        .mode(JpegMode::Progressive)
        .encode(&data)
        .expect("encode failed");

    jpeg_decoder::Decoder::new(&jpeg[..]).decode().is_ok()
}

fn main() {
    for optimize in [false, true] {
        let mode = if optimize { "optimized" } else { "standard" };
        println!("Testing {} Huffman tables:\n", mode);

        for pattern in ["flat", "gradient", "noise"] {
            let mut fails = Vec::new();

            for size in 1..=64 {
                let ok = test_size(size, pattern, optimize);
                let blocks = (size + 7) / 8;
                let mod8 = size % 8;
                if !ok {
                    fails.push((size, blocks, mod8));
                }
            }

            if fails.is_empty() {
                println!("  {} pattern: All sizes pass!", pattern);
            } else {
                println!(
                    "  {} pattern FAILS: {:?}",
                    pattern,
                    fails.iter().map(|(s, _, _)| s).collect::<Vec<_>>()
                );
            }
        }
        println!();
    }
}
