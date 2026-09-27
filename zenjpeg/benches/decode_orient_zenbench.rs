//! Auto-orient overhead A/B (#150): decode a 12 MP EXIF-oriented baseline
//! JPEG upright vs with `auto_orient(true)`, for RGB8 (3 bpp) and BGRA8
//! (4 bpp) output. The difference is the cost of the pixel-domain permute.
//!
//! Run it twice to compare the two permute implementations:
//!
//! ```bash
//! cargo bench -p zenjpeg --bench decode_orient_zenbench                      # scalar gather
//! cargo bench -p zenjpeg --bench decode_orient_zenbench --features zencodec  # zenpixels-convert
//! ```
//!
//! Sequential (`num_threads(1)`) so the permute is not hidden behind the
//! parallel decode's wall time.

use enough::Unstoppable;
use zenbench::prelude::*;
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encode::exif::{Exif, Orientation};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

const W: u32 = 4000;
const H: u32 = 3000;

fn noise_patches_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 3) as usize];
    let mut seed = 0x9E37_79B9u32;
    for (i, b) in v.iter_mut().enumerate() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let x = (i / 3) % w as usize;
        let y = (i / 3) / w as usize;
        *b = (((x / 32 + y / 32) * 41) as u8).wrapping_add((seed >> 27) as u8);
    }
    v
}

fn encode(orient: Orientation) -> Vec<u8> {
    let px = noise_patches_rgb(W, H);
    EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .request()
        .exif(Exif::build().orientation(orient))
        .encode_bytes(&px, W, H, PixelLayout::Rgb8Srgb)
        .expect("encode")
}

struct Fixtures {
    upright: Vec<u8>,
    rotate90: Vec<u8>,
    rotate180: Vec<u8>,
}

static FIXTURES: std::sync::OnceLock<Fixtures> = std::sync::OnceLock::new();

fn fixtures() -> &'static Fixtures {
    FIXTURES.get_or_init(|| Fixtures {
        upright: encode(Orientation::Normal),
        rotate90: encode(Orientation::Rotate90),
        rotate180: encode(Orientation::Rotate180),
    })
}

fn decode(jpeg: &[u8], format: PixelFormat, auto_orient: bool) -> usize {
    Decoder::new()
        .output_format(format)
        .auto_orient(auto_orient)
        .num_threads(1)
        .decode(jpeg, Unstoppable)
        .expect("decode")
        .pixels_u8()
        .map(|p| p.len())
        .unwrap_or(0)
}

fn bench_orient(suite: &mut Suite) {
    let f = fixtures();
    for (label, format) in [("rgb8", PixelFormat::Rgb), ("bgra8", PixelFormat::Bgra)] {
        suite.group(format!("orient_12mp_{label}"), move |g| {
            g.throughput(Throughput::Elements(u64::from(W) * u64::from(H)));
            g.bench("upright (auto_orient off)", move |b| {
                b.iter(move || decode(&f.upright, format, false))
            });
            g.bench("rotate90 (EXIF 6, auto_orient on)", move |b| {
                b.iter(move || decode(&f.rotate90, format, true))
            });
            g.bench("rotate180 (EXIF 3, auto_orient on)", move |b| {
                b.iter(move || decode(&f.rotate180, format, true))
            });
        });
    }
}

zenbench::main!(bench_orient);
