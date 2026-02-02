//! Can trellis match jpegli quality at smaller file size?

use std::path::Path;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout};
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};
use enough::Unstoppable;

fn encode(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut e = config.encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb).ok()?;
    e.push_packed(&img.pixels, Unstoppable).ok()?;
    e.finish().ok()
}

fn ba(img: &ImageData, jpeg: &[u8]) -> Option<f64> {
    let o = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
    let d: RgbImage = decode_jpeg_to_rgb(jpeg).ok()?;
    Some(QualityMetrics::butteraugli(o.as_ref(), d.as_ref()))
}

fn main() {
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";
    let images: Vec<_> = if Path::new(base_dir).exists() {
        std::fs::read_dir(base_dir).unwrap().filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png")).take(15).collect()
    } else {
        let fb = "/home/lilith/work/codec-eval/codec-corpus/cid22";
        std::fs::read_dir(fb).unwrap().filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .map(|e| e.path()).take(15).collect()
    };
    let loaded: Vec<_> = images.iter().filter_map(|p| ImageData::from_path(p)).collect();
    
    println!("=== Trellis vs Jpegli: Do Curves Cross? ===\n");
    
    // Collect full curves
    let mut jpegli_curve: Vec<(f64, f64)> = Vec::new();
    let mut trellis_curve: Vec<(f64, f64)> = Vec::new();
    
    for q in (75..=95).step_by(1) {
        let q = q as f32;
        
        // Jpegli
        let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliProgressive);
        let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
        for img in &loaded {
            if let Some(j) = encode(&cfg, img) {
                if let Some(v) = ba(img, &j) { sz += j.len(); b += v; n += 1; }
            }
        }
        jpegli_curve.push((sz as f64 / n as f64, b / n as f64));
        
        // Trellis (HybridProgressive)
        let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridProgressive);
        let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
        for img in &loaded {
            if let Some(j) = encode(&cfg, img) {
                if let Some(v) = ba(img, &j) { sz += j.len(); b += v; n += 1; }
            }
        }
        trellis_curve.push((sz as f64 / n as f64, b / n as f64));
    }
    
    // Print curves side by side
    println!("{:>4} {:>12} {:>8}  {:>12} {:>8}  {:>8}",
             "Q", "Jpegli sz", "Jpegli BA", "Trellis sz", "Trellis BA", "BA diff%");
    println!("{}", "-".repeat(70));
    
    for (i, q) in (75..=95).enumerate() {
        let (jsz, jba) = jpegli_curve[i];
        let (tsz, tba) = trellis_curve[i];
        let ba_diff = (tba - jba) / jba * 100.0;
        println!("{:>4} {:>12.0} {:>8.3}  {:>12.0} {:>8.3}  {:>+8.1}%",
                 q, jsz, jba, tsz, tba, ba_diff);
    }
    
    // Find: for each jpegli quality, what trellis quality gives same BA?
    println!("\n=== Match BA: What Size Savings? ===\n");
    println!("{:>10} {:>10} {:>8}  {:>15} {:>10} {:>8}  {:>10}",
             "Jpegli Q", "Size", "BA", "Trellis match", "Size", "BA", "Size win?");
    println!("{}", "-".repeat(90));
    
    for jq in [80, 85, 90] {
        let ji = jq - 75;
        let (jsz, jba) = jpegli_curve[ji];
        
        // Find trellis Q that gives same or better BA
        let mut best_match: Option<(usize, f64, f64)> = None;
        for (ti, &(tsz, tba)) in trellis_curve.iter().enumerate() {
            if tba <= jba * 1.01 {  // Within 1% of jpegli BA
                if best_match.is_none() || tsz < best_match.unwrap().1 {
                    best_match = Some((ti + 75, tsz, tba));
                }
            }
        }
        
        if let Some((tq, tsz, tba)) = best_match {
            let size_diff = (tsz - jsz) / jsz * 100.0;
            let win = if size_diff < -1.0 { "★ YES" } else { "no" };
            println!("{:>10} {:>10.0} {:>8.3}  {:>15} {:>10.0} {:>8.3}  {:>10} ({:+.1}%)",
                     format!("Q{}", jq), jsz, jba, format!("Q{}", tq), tsz, tba, win, size_diff);
        } else {
            println!("{:>10} {:>10.0} {:>8.3}  {:>15}", format!("Q{}", jq), jsz, jba, "no match found");
        }
    }
    
    test_with_scan_opt();
    println!("\n★ = Trellis can match quality at smaller size (Pareto win)");
}

// Also test with scan optimization
fn test_with_scan_opt() {
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";
    let images: Vec<_> = if std::path::Path::new(base_dir).exists() {
        std::fs::read_dir(base_dir).unwrap().filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png")).take(15).collect()
    } else {
        let fb = "/home/lilith/work/codec-eval/codec-corpus/cid22";
        std::fs::read_dir(fb).unwrap().filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .map(|e| e.path()).take(15).collect()
    };
    let loaded: Vec<ImageData> = images.iter().filter_map(|p| ImageData::from_path(p)).collect();
    
    println!("\n=== Adding Scan Optimization ===\n");
    println!("{:>20} {:>10} {:>8}", "Config @ Q90", "Size", "BA");
    println!("{}", "-".repeat(42));
    
    let q = 90.0;
    
    // Jpegli baseline
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::JpegliProgressive);
    let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
    for img in &loaded {
        if let Some(j) = encode(&cfg, img) { if let Some(v) = ba(img, &j) { sz += j.len(); b += v; n += 1; } }
    }
    let base = (sz as f64 / n as f64, b / n as f64);
    println!("{:>20} {:>10.0} {:>8.3}", "Jpegli", base.0, base.1);
    
    // Jpegli + scan opt
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::JpegliProgressive)
        .optimize_scans(true);
    let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
    for img in &loaded {
        if let Some(j) = encode(&cfg, img) { if let Some(v) = ba(img, &j) { sz += j.len(); b += v; n += 1; } }
    }
    println!("{:>20} {:>10.0} {:>8.3} ({:+.1}% size)", "Jpegli+ScanOpt", 
             sz as f64 / n as f64, b / n as f64, (sz as f64 / n as f64 - base.0) / base.0 * 100.0);
    
    // Trellis
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridProgressive);
    let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
    for img in &loaded {
        if let Some(j) = encode(&cfg, img) { if let Some(v) = ba(img, &j) { sz += j.len(); b += v; n += 1; } }
    }
    println!("{:>20} {:>10.0} {:>8.3} ({:+.1}% size)", "Trellis", 
             sz as f64 / n as f64, b / n as f64, (sz as f64 / n as f64 - base.0) / base.0 * 100.0);
    
    // Trellis + scan opt
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridProgressive)
        .optimize_scans(true);
    let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
    for img in &loaded {
        if let Some(j) = encode(&cfg, img) { if let Some(v) = ba(img, &j) { sz += j.len(); b += v; n += 1; } }
    }
    println!("{:>20} {:>10.0} {:>8.3} ({:+.1}% size)", "Trellis+ScanOpt", 
             sz as f64 / n as f64, b / n as f64, (sz as f64 / n as f64 - base.0) / base.0 * 100.0);
}
