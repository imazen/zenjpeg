//! Compare zenjpeg decode throughput between RGB and BGRA output formats.
//!
//! This probes whether the RGB→BGRA swizzle (an extra full-buffer pass at
//! 4 bytes/pixel) is the source of the BGRA-path slowdown in the imageflow
//! decode bench.
//!
//! Usage: `cargo run --release --example profile_bgra_vs_rgb -- /tmp/zenjpeg_profile_1024x1024.jpg 500`

use enough::Unstoppable;
use std::hint::black_box;
use std::time::Instant;
use zenjpeg::decoder::{Decoder, PixelFormat};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("path").clone();
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);

    let jpeg = std::fs::read(&path).expect("read jpeg");
    eprintln!("JPEG: {} ({} bytes), reps={}", path, jpeg.len(), reps);

    // Warmup + first run to capture dimensions.
    let warm = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .expect("decode");
    eprintln!(
        "decoded to {}x{}, {} bytes RGB",
        warm.width(),
        warm.height(),
        warm.pixels_u8().map(|x| x.len()).unwrap_or(0)
    );

    for fmt in [PixelFormat::Rgb, PixelFormat::Rgba, PixelFormat::Bgra] {
        // Warmup
        for _ in 0..5 {
            let d = Decoder::new()
                .output_format(fmt)
                .decode(black_box(&jpeg), Unstoppable)
                .unwrap();
            black_box(&d);
        }
        let t = Instant::now();
        for _ in 0..reps {
            let d = Decoder::new()
                .output_format(fmt)
                .decode(black_box(&jpeg), Unstoppable)
                .unwrap();
            black_box(&d);
        }
        let ns = t.elapsed().as_nanos() as f64;
        let per = ns / reps as f64;
        let mpx = (warm.width() * warm.height()) as f64 * 1000.0 / per;
        eprintln!("{:?}: {:.0} us/iter, {:.0} Mpx/s", fmt, per / 1000.0, mpx);
    }
}
