//! Size + speed measurement for the fixed-Huffman-table (optimize_huffman(false))
//! baseline path, for A/B against the pre-table-completion build.

use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Noise+patches photo-like content (per the no-smooth-gradients rule).
fn photo_like(w: u32, h: u32) -> Vec<u8> {
    let mut state: u32 = 0x9E3779B9;
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (state >> 26) as i32 - 32; // mild noise
            let base = (((x / 37) * 53 + (y / 29) * 31) % 200) as i32 + 20; // patches
            let r = (base + n).clamp(0, 255) as u8;
            let g = (base + (n >> 1) + 10).clamp(0, 255) as u8;
            let b = (base - (n >> 1) - 10).clamp(0, 255) as u8;
            px.extend_from_slice(&[r, g, b]);
        }
    }
    px
}

fn main() {
    let (w, h) = (1024u32, 768u32);
    let px = photo_like(w, h);
    for ss in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        for q in [30.0f32, 50.0, 75.0, 90.0] {
            let mut times = Vec::new();
            let mut size = 0usize;
            for _ in 0..9 {
                let t0 = Instant::now();
                let mut enc = EncoderConfig::ycbcr(q, ss)
                    .progressive(false)
                    .optimize_huffman(false)
                    .restart_mcu_rows(0)
                    .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
                    .unwrap();
                enc.push_packed(&px, Unstoppable).unwrap();
                size = enc.finish().unwrap().len();
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{ss:?}/q{q}: {size} bytes, median {:.2} ms",
                times[times.len() / 2]
            );
        }
    }
}
