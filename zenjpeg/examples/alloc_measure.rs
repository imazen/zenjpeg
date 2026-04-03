use std::time::Instant;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn main() {
    let img = std::fs::read("/mnt/v/input/BRAG/karwin-luo-4k-420-q85-baseline.jpg").unwrap();
    let decoded = zenjpeg::decode::Decoder::new()
        .output_format(zenjpeg::decoder::PixelFormat::Rgb)
        .decode(&img, enough::Unstoppable).unwrap();
    let (w, h) = (decoded.width, decoded.height);
    let pixels = decoded.pixels_u8().unwrap();
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);

    // First run: cold thread-local buffers
    let start = Instant::now();
    let _ = config.encode_bytes_parallel(pixels, w, h).unwrap();
    let cold = start.elapsed();

    // Subsequent runs: warm thread-local buffers
    let iters = 30u32;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = config.encode_bytes_parallel(pixels, w, h).unwrap();
    }
    let warm = start.elapsed() / iters;

    eprintln!("Cold (first run):   {:>8.2?}", cold);
    eprintln!("Warm (avg of {}): {:>8.2?}", iters, warm);
    eprintln!("Cold overhead:      {:>8.2?} ({:.0}%)", 
        cold - warm, (cold.as_secs_f64() / warm.as_secs_f64() - 1.0) * 100.0);

    // Measure just the StreamingEncoder creation (quant table setup)
    let start = Instant::now();
    for _ in 0..iters {
        let _ = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    }
    let setup = start.elapsed() / iters;
    eprintln!("Encoder setup:      {:>8.2?} ({:.0}% of warm)", 
        setup, setup.as_secs_f64() / warm.as_secs_f64() * 100.0);
}
