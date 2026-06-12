//! Wall-time probe for #154's gray-source kernel change: decode a JPEG to
//! Rgb N times, report min/median. Run against pre- and post-fix builds for
//! the A/B (single-threaded, default features).
//!
//! Usage: gray_decode_bench <file.jpg> [iters]

use enough::Unstoppable;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gray_decode_bench <file.jpg> [iters]");
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let data = std::fs::read(&path).unwrap();

    let mut times = Vec::with_capacity(iters);
    let mut sink = 0u64;
    for _ in 0..iters {
        let t = Instant::now();
        let res = zenjpeg::decoder::Decoder::new()
            .output_format(zenjpeg::decoder::PixelFormat::Rgb)
            .decode(&data, Unstoppable)
            .expect("decode");
        times.push(t.elapsed());
        sink = sink.wrapping_add(res.pixels_u8().map(|p| p[0] as u64).unwrap_or(0));
    }
    times.sort();
    println!(
        "{}x{} iters={iters} min={:?} median={:?} (sink {sink})",
        times.len(),
        0,
        times[0],
        times[iters / 2]
    );
}
