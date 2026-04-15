//! Tiny driver for callgrind/flamegraph profiling of XYB encode.
//!
//! Usage:
//!   cargo build --release -p zenjpeg --example profile_xyb --features trellis
//!   valgrind --tool=callgrind --collect-systime=yes \
//!     target/release/examples/profile_xyb xyb-full progressive 1024 200
//!
//! Args: <mode> <scan> <size> <iters>
//!   mode: ycbcr | xyb-bquarter | xyb-full
//!   scan: progressive | baseline
//!   size: image edge in pixels (square)
//!   iters: encode iterations

use enough::Unstoppable;
use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout, XybSubsampling};

fn noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40));
            rgb[i + 1] = ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80));
            rgb[i + 2] = ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120));
        }
    }
    rgb
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(|s| s.as_str()).unwrap_or("xyb-full");
    let scan = args.get(1).map(|s| s.as_str()).unwrap_or("progressive");
    let size: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1024);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(50);

    let rgb = noise_patches(size as usize, size as usize);
    let progressive = scan == "progressive";

    let mut total = 0usize;
    for _ in 0..iters {
        let cfg = match mode {
            "ycbcr" => {
                EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(progressive)
            }
            "xyb-bquarter" => {
                EncoderConfig::xyb(85.0, XybSubsampling::BQuarter).progressive(progressive)
            }
            "xyb-full" => EncoderConfig::xyb(85.0, XybSubsampling::Full).progressive(progressive),
            other => panic!("unknown mode: {other}"),
        };
        let mut e = cfg
            .encode_from_bytes(size, size, PixelLayout::Rgb8Srgb)
            .unwrap();
        e.push_packed(&rgb, Unstoppable).unwrap();
        let bytes = e.finish().unwrap();
        total += bytes.len();
    }
    eprintln!("mode={mode} scan={scan} size={size} iters={iters} total_bytes={total}");
}
