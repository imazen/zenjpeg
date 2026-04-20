//! Measure the TinyFileMode crossover point.
//!
//! Walks a directory of PNG images, decodes each with the `image` crate,
//! resizes to a range of target sizes, encodes each at q=75 three ways
//! (Off, Auto, Force), and prints size deltas. The goal is to find the
//! pixel-count threshold where `Force` stops saving bytes.
//!
//! Usage:
//!
//!   cargo run --release --example tiny_file_crossover -- <dir> [<dir> ...]
//!
//! If no paths are given, falls back to synthetic gradients.

use std::env;
use std::path::Path;
use std::time::Instant;

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, TinyFileMode};

fn encode(
    data: &[u8],
    width: u32,
    height: u32,
    mode: TinyFileMode,
    quality: u8,
    subsampling: ChromaSubsampling,
) -> usize {
    let cfg = EncoderConfig::ycbcr(quality, subsampling)
        .progressive(false)
        .tiny_file_mode(mode);
    let mut enc = cfg
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup failed");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push failed");
    enc.finish().expect("finish failed").len()
}

fn encode_gray(
    data: &[u8],
    width: u32,
    height: u32,
    mode: TinyFileMode,
    quality: u8,
) -> usize {
    let cfg = EncoderConfig::grayscale(quality)
        .progressive(false)
        .tiny_file_mode(mode);
    let mut enc = cfg
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("encoder setup failed");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push failed");
    enc.finish().expect("finish failed").len()
}

/// Nearest-neighbor resize to `out_w × out_h` for simplicity.
fn resize_nearest(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    channels: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; (out_w as usize) * (out_h as usize) * channels];
    for y in 0..out_h {
        let src_y = (y as u64 * src_h as u64 / out_h as u64) as u32;
        for x in 0..out_w {
            let src_x = (x as u64 * src_w as u64 / out_w as u64) as u32;
            let src_idx =
                ((src_y as usize) * (src_w as usize) + (src_x as usize)) * channels;
            let dst_idx = ((y as usize) * (out_w as usize) + (x as usize)) * channels;
            for c in 0..channels {
                out[dst_idx + c] = src[src_idx + c];
            }
        }
    }
    out
}

fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w as usize) * (h as usize) * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y as usize) * (w as usize) + (x as usize)) * 3;
            out[idx] = ((x * 255) / w.max(1)) as u8;
            out[idx + 1] = ((y * 255) / h.max(1)) as u8;
            out[idx + 2] = ((x ^ y) & 0xFF) as u8;
        }
    }
    out
}

fn gradient_gray(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w as usize) * (h as usize)];
    for y in 0..h {
        for x in 0..w {
            let idx = (y as usize) * (w as usize) + (x as usize);
            out[idx] = (((x * y) / w.max(1) as u32) & 0xFF) as u8;
        }
    }
    out
}

fn load_rgb_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

struct Measurement {
    width: u32,
    height: u32,
    quality: u8,
    subsampling: &'static str,
    off: usize,
    force: usize,
    auto: usize,
}

impl Measurement {
    fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
    fn delta(&self) -> i64 {
        self.off as i64 - self.force as i64
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let paths: Vec<std::path::PathBuf> = if args.is_empty() {
        Vec::new()
    } else {
        args.iter()
            .flat_map(|p| {
                let path = Path::new(p);
                if path.is_dir() {
                    std::fs::read_dir(path)
                        .map(|rd| {
                            rd.flatten()
                                .map(|e| e.path())
                                .filter(|p| {
                                    p.extension().and_then(|s| s.to_str()) == Some("png")
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                } else {
                    vec![path.to_path_buf()]
                }
            })
            .collect()
    };

    let sizes = [32u32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
    let qualities = [75u8, 90];
    let subsamplings = [
        ("4:2:0", ChromaSubsampling::Quarter),
        ("4:4:4", ChromaSubsampling::None),
    ];

    let sources: Vec<(String, Vec<u8>, u32, u32)> = if paths.is_empty() {
        // One synthetic 1024×1024 reference image, resized.
        vec![(
            "synthetic".into(),
            gradient_rgb(1024, 1024),
            1024,
            1024,
        )]
    } else {
        let t0 = Instant::now();
        let loaded: Vec<_> = paths
            .iter()
            .take(64) // cap to keep runtime sane
            .filter_map(|p| {
                load_rgb_png(p).map(|(data, w, h)| {
                    (
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string(),
                        data,
                        w,
                        h,
                    )
                })
            })
            .collect();
        eprintln!("loaded {} source PNGs in {:.2?}", loaded.len(), t0.elapsed());
        loaded
    };

    if sources.is_empty() {
        eprintln!("no valid sources; aborting");
        return;
    }

    let mut measurements: Vec<Measurement> = Vec::new();

    for (_name, src, src_w, src_h) in &sources {
        for &target in &sizes {
            if target > *src_w && target > *src_h {
                continue;
            }
            let resized = resize_nearest(src, *src_w, *src_h, target, target, 3);
            for &q in &qualities {
                for &(sub_name, sub) in &subsamplings {
                    let off = encode(&resized, target, target, TinyFileMode::Off, q, sub);
                    let force = encode(&resized, target, target, TinyFileMode::Force, q, sub);
                    let auto = encode(&resized, target, target, TinyFileMode::Auto, q, sub);
                    measurements.push(Measurement {
                        width: target,
                        height: target,
                        quality: q,
                        subsampling: sub_name,
                        off,
                        force,
                        auto,
                    });
                }
            }
        }
    }

    // Aggregate by (size, quality, subsampling).
    println!(
        "{:>5} {:>5} {:>5} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "size", "q", "sub", "n", "avg_off", "avg_force", "avg_delta", "avg_pct", "auto_matches"
    );
    for &target in &sizes {
        for &q in &qualities {
            for &(sub_name, _) in &subsamplings {
                let bucket: Vec<&Measurement> = measurements
                    .iter()
                    .filter(|m| {
                        m.width == target && m.quality == q && m.subsampling == sub_name
                    })
                    .collect();
                if bucket.is_empty() {
                    continue;
                }
                let n = bucket.len();
                let sum_off: usize = bucket.iter().map(|m| m.off).sum();
                let sum_force: usize = bucket.iter().map(|m| m.force).sum();
                let sum_delta: i64 = bucket.iter().map(|m| m.delta()).sum();
                let avg_pct =
                    100.0 * sum_delta as f64 / sum_off as f64;
                let auto_matches_force = bucket.iter().filter(|m| m.auto == m.force).count();
                println!(
                    "{:>5} {:>5} {:>5} {:>5} {:>9.1} {:>9.1} {:>+9.1} {:>+8.2}% {:>5}/{:<3}",
                    target,
                    q,
                    sub_name,
                    n,
                    sum_off as f64 / n as f64,
                    sum_force as f64 / n as f64,
                    sum_delta as f64 / n as f64,
                    avg_pct,
                    auto_matches_force,
                    n,
                );
            }
        }
    }

    // Grayscale sweep (synthetic only, since we just want to know where shared
    // Huffman stops helping for 1-component images).
    println!();
    println!("grayscale synthetic sweep:");
    for &target in &sizes {
        let gray = gradient_gray(target, target);
        let off = encode_gray(&gray, target, target, TinyFileMode::Off, 75);
        let force = encode_gray(&gray, target, target, TinyFileMode::Force, 75);
        let auto = encode_gray(&gray, target, target, TinyFileMode::Auto, 75);
        println!(
            "  {:>5}² off={:<6} force={:<6} Δ={:+5}  auto={}",
            target,
            off,
            force,
            off as i64 - force as i64,
            if auto == force {
                "Force"
            } else if auto == off {
                "Off"
            } else {
                "??"
            }
        );
    }
}
