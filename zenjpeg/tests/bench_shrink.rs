//! Shrink-on-load performance benchmark
//! Run with: cargo test --release -p zenjpeg --test bench_shrink -- --nocapture --ignored

use enough::Unstoppable;
use std::time::Instant;
use zenjpeg::decoder::{DctScale, Decoder, PixelFormat, ShrinkHint, ShrinkQuality};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn create_test_jpeg(width: u32, height: u32, quality: f32, subsamp: ChromaSubsampling) -> Vec<u8> {
    // Use a simple LCG for deterministic "noisy" content that compresses realistically.
    // Pure gradients compress to almost nothing; real photos have much higher entropy.
    let mut rng = 0x12345678u64;
    let mut data = vec![0u8; width as usize * height as usize * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            // Gradient base + noise
            let base_r = ((x * 255) / width as usize) as u8;
            let base_g = ((y * 255) / height as usize) as u8;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((rng >> 33) % 60) as u8;
            data[idx] = base_r.wrapping_add(noise);
            data[idx + 1] = base_g.wrapping_add(noise);
            data[idx + 2] = 128u8.wrapping_add(noise);
        }
    }
    let config = EncoderConfig::ycbcr(quality, subsamp);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn area_downsample_rgb(pixels: &[u8], width: usize, height: usize, factor: usize) -> Vec<u8> {
    let out_w = width / factor;
    let out_h = height / factor;
    let mut out = vec![0u8; out_w * out_h * 3];
    let inv = 1.0 / (factor * factor) as f32;
    for oy in 0..out_h {
        for ox in 0..out_w {
            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            for dy in 0..factor {
                for dx in 0..factor {
                    let idx = ((oy * factor + dy) * width + ox * factor + dx) * 3;
                    r += pixels[idx] as u32;
                    g += pixels[idx + 1] as u32;
                    b += pixels[idx + 2] as u32;
                }
            }
            let oidx = (oy * out_w + ox) * 3;
            out[oidx] = (r as f32 * inv + 0.5) as u8;
            out[oidx + 1] = (g as f32 * inv + 0.5) as u8;
            out[oidx + 2] = (b as f32 * inv + 0.5) as u8;
        }
    }
    out
}

fn median(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn bench_config(label: &str, jpeg: &[u8], iterations: usize, f: impl Fn(&[u8])) -> f64 {
    // Warmup
    for _ in 0..2 {
        f(jpeg);
    }
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        f(jpeg);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let med = median(&mut times);
    eprintln!("  {label:35} {med:8.2} ms");
    med
}

#[test]
#[ignore]
fn bench_shrink_100mp() {
    // 10240x10240 = 104.9 MP, target ~400x400
    // 1/8 scale: 1280x1280
    // 1/4 scale: 2560x2560
    // 1/2 scale: 5120x5120
    let w = 10240u32;
    let h = 10240u32;
    let iters = 5;

    eprintln!("Encoding {w}x{h} test image...");
    let t = Instant::now();
    let jpeg = create_test_jpeg(w, h, 95.0, ChromaSubsampling::Quarter);
    eprintln!(
        "Encoded in {:.1}s, {} bytes ({:.1} MB)\n",
        t.elapsed().as_secs_f64(),
        jpeg.len(),
        jpeg.len() as f64 / 1e6
    );

    eprintln!("=== {w}x{h} 4:2:0 Q95 ===\n");

    // Show output dimensions for each scale
    for scale in [DctScale::Half, DctScale::Quarter, DctScale::Eighth] {
        let sw = scale.scaled_dimension(w);
        let sh = scale.scaled_dimension(h);
        eprintln!("  {scale}: output {sw}x{sh}");
    }
    eprintln!();

    let full = bench_config("Full decode (10240x10240)", &jpeg, iters, |data| {
        let _ = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .max_pixels(0)
            .decode(data, Unstoppable)
            .unwrap();
    });

    let fast_half = bench_config("Shrink Fast 1/2 (5120x5120)", &jpeg, iters, |data| {
        let _ = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .shrink(ShrinkHint::ExactScale(DctScale::Half))
            .shrink_quality(ShrinkQuality::Fast)
            .decode(data, Unstoppable)
            .unwrap();
    });

    let best_half = bench_config("Shrink Best 1/2 (5120x5120)", &jpeg, iters, |data| {
        let _ = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .shrink(ShrinkHint::ExactScale(DctScale::Half))
            .shrink_quality(ShrinkQuality::Best)
            .decode(data, Unstoppable)
            .unwrap();
    });

    let fast_quarter = bench_config("Shrink Fast 1/4 (2560x2560)", &jpeg, iters, |data| {
        let _ = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .shrink(ShrinkHint::ExactScale(DctScale::Quarter))
            .shrink_quality(ShrinkQuality::Fast)
            .decode(data, Unstoppable)
            .unwrap();
    });

    let best_quarter = bench_config("Shrink Best 1/4 (2560x2560)", &jpeg, iters, |data| {
        let _ = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .shrink(ShrinkHint::ExactScale(DctScale::Quarter))
            .shrink_quality(ShrinkQuality::Best)
            .decode(data, Unstoppable)
            .unwrap();
    });

    let fast_eighth = bench_config("Shrink Fast 1/8 (1280x1280)", &jpeg, iters, |data| {
        let _ = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .shrink(ShrinkHint::ExactScale(DctScale::Eighth))
            .shrink_quality(ShrinkQuality::Fast)
            .decode(data, Unstoppable)
            .unwrap();
    });

    let best_eighth = bench_config("Shrink Best 1/8 (1280x1280)", &jpeg, iters, |data| {
        let _ = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .shrink(ShrinkHint::ExactScale(DctScale::Eighth))
            .shrink_quality(ShrinkQuality::Best)
            .decode(data, Unstoppable)
            .unwrap();
    });

    // Full decode + resize
    let full_resize_eighth = bench_config("Full + resize 1/8 (1280x1280)", &jpeg, iters, |data| {
        let r = Decoder::new()
            .max_pixels(0)
            .output_format(PixelFormat::Rgb)
            .max_pixels(0)
            .decode(data, Unstoppable)
            .unwrap();
        let px = r.pixels_u8().unwrap();
        let _ = area_downsample_rgb(px, r.width() as usize, r.height() as usize, 8);
    });

    // Memory
    eprintln!("\n  --- Output buffer sizes ---");
    eprintln!(
        "  Full:    {w}x{h} = {:.0} MP, {:.0} MB RGB",
        w as f64 * h as f64 / 1e6,
        w as f64 * h as f64 * 3.0 / 1e6
    );
    for scale in [DctScale::Half, DctScale::Quarter, DctScale::Eighth] {
        let sw = scale.scaled_dimension(w);
        let sh = scale.scaled_dimension(h);
        let name = match scale {
            DctScale::Half => "1/2",
            DctScale::Quarter => "1/4",
            DctScale::Eighth => "1/8",
            _ => "?",
        };
        eprintln!(
            "  {name:8} {sw}x{sh} = {:.1} MP, {:.1} MB RGB",
            sw as f64 * sh as f64 / 1e6,
            sw as f64 * sh as f64 * 3.0 / 1e6
        );
    }

    eprintln!("\n  --- Summary ---");
    eprintln!("  {:37} {:>8}  {:>6}", "Path", "Time", "Speedup");
    eprintln!("  {:37} {:>8}  {:>6}", "----", "----", "-------");
    let rows: Vec<(&str, f64)> = vec![
        ("Full decode (10240x10240)", full),
        ("Shrink Fast 1/2 (5120x5120)", fast_half),
        ("Shrink Best 1/2 (5120x5120)", best_half),
        ("Shrink Fast 1/4 (2560x2560)", fast_quarter),
        ("Shrink Best 1/4 (2560x2560)", best_quarter),
        ("Shrink Fast 1/8 (1280x1280)", fast_eighth),
        ("Shrink Best 1/8 (1280x1280)", best_eighth),
        ("Full + resize 1/8 (1280x1280)", full_resize_eighth),
    ];
    for (label, time) in &rows {
        eprintln!("  {:37} {:7.1} ms  {:5.2}x", label, time, full / time);
    }
}
