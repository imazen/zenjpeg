//! Minimal binary for callgrind analysis of boundary_rd.
//!
//! Usage:
//!   cargo build --release -p zenjpeg --example boundary_rd_callgrind
//!   valgrind --tool=callgrind --callgrind-out-file=/tmp/cg_off.out \
//!     ./target/release/examples/boundary_rd_callgrind off <image.png> [iters]
//!   valgrind --tool=callgrind --callgrind-out-file=/tmp/cg_on.out \
//!     ./target/release/examples/boundary_rd_callgrind on <image.png> [iters]
//!   callgrind_annotate --threshold=0.5 /tmp/cg_on.out > /tmp/cg_on.txt
//!
//! Loads the PNG, rescales to ≤ 512×512 MCU-aligned, encodes at Q75 4:2:0
//! with the specified boundary_rd mode N times (default 3). One image,
//! N encodes — keeps callgrind run short and focused.

use std::path::Path;

use zenjpeg::encode::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};
use zenjpeg::encoder::{BoundaryRd, BoundaryRdConfig};

fn load_img(path: &Path, max_side: u32) -> (Vec<u8>, usize, usize) {
    let img = zenjpeg_bench_utils::load_png(path).expect("open png");
    let (w, h) = (img.width() as u32, img.height() as u32);
    let scaled = if w.max(h) > max_side {
        let (tw, th) = if w >= h {
            (
                max_side,
                (h as u64 * max_side as u64 / w as u64).max(1) as u32,
            )
        } else {
            (
                (w as u64 * max_side as u64 / h as u64).max(1) as u32,
                max_side,
            )
        };
        let config = zenresize::ResizeConfig::builder(w, h, tw, th)
            .filter(zenresize::Filter::Triangle)
            .build();
        zenresize::resize_3ch(img.as_ref(), tw, th, &config)
    } else {
        img
    };
    let w = scaled.width() & !7;
    let h = scaled.height() & !7;
    let orig_w = scaled.width();
    let mut buf = Vec::with_capacity(w * h * 3);
    let raw = scaled.buf();
    for y in 0..h {
        let row_start = y * orig_w;
        for p in &raw[row_start..row_start + w] {
            buf.extend_from_slice(&[p.r, p.g, p.b]);
        }
    }
    (buf, w, h)
}

fn main() {
    let mode = std::env::args().nth(1).expect("mode: off|on");
    let path = std::env::args().nth(2).expect("image path");
    let iters: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let (rgb, w, h) = load_img(Path::new(&path), 512);
    eprintln!("image: {}x{} ({} bytes rgb)", w, h, rgb.len());

    for _ in 0..iters {
        let config = match mode.as_str() {
            "off" => {
                EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter).boundary_rd(BoundaryRd::Off)
            }
            "on" => EncoderConfig::ycbcr(75.0, ChromaSubsampling::Quarter)
                .boundary_rd(BoundaryRd::On(BoundaryRdConfig::default())),
            _ => panic!("mode must be off|on"),
        };
        let out = config
            .encode_bytes(&rgb, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .expect("encode");
        std::hint::black_box(&out);
    }
}
