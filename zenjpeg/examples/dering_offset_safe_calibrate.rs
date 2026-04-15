//! Find the maximum distance offset that KEEPS SSIM2 >= dering-off case
//! across all tested images, per Q.

use enough::Unstoppable;
use imgref::ImgVec;
use rgb::RGB8;
use std::fs;
use std::path::{Path, PathBuf};

use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

const Q_LEVELS: &[u8] = &[10, 30, 50, 70, 85, 95];

fn zen_encode_with_offset(
    rgb: &[u8],
    w: u32,
    h: u32,
    q: u8,
    dering: bool,
    dist_offset: f32,
) -> Vec<u8> {
    let base = Quality::ApproxJpegli(q as f32).to_distance();
    let cfg = EncoderConfig::ycbcr(
        Quality::ApproxButteraugli(base + dist_offset),
        ChromaSubsampling::Quarter,
    )
    .progressive(false)
    .deringing(dering);
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

fn ssim2(orig: &ImgVec<RGB8>, bytes: &[u8], w: usize, h: usize) -> f64 {
    let dec = zenjpeg_bench_utils::decode_jpeg_to_rgb(bytes).unwrap();
    let o3: Vec<[u8; 3]> = orig.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let d3: Vec<[u8; 3]> = dec.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let oi = ImgVec::new(o3, w, h);
    let di = ImgVec::new(d3, w, h);
    fast_ssim2::compute_ssimulacra2(oi.as_ref(), di.as_ref()).unwrap_or(0.0)
}

fn load_rgb(path: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((w, h, img.into_raw()))
}

fn list_pngs(dir: &Path, limit: usize) -> Vec<PathBuf> {
    let Ok(rd) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("png"))
        })
        .collect();
    v.sort();
    v.truncate(limit);
    v
}

/// Find max offset such that ssim(dering_on, offset) >= ssim(dering_off) - tolerance.
fn max_safe_offset(rgb: &[u8], w: u32, h: u32, q: u8, tolerance: f64) -> (f32, f64, f64) {
    let orig_rgb: Vec<RGB8> = rgb
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig = ImgVec::new(orig_rgb, w as usize, h as usize);

    let off_bytes = zen_encode_with_offset(rgb, w, h, q, false, 0.0);
    let off_ssim = ssim2(&orig, &off_bytes, w as usize, h as usize);

    // Binary search for largest offset satisfying constraint
    // Test at candidates: 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4
    let candidates: [f32; 10] = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50];
    let mut best = 0.0;
    let mut best_ssim = off_ssim;
    for &c in &candidates {
        let bytes = zen_encode_with_offset(rgb, w, h, q, true, c);
        let ss = ssim2(&orig, &bytes, w as usize, h as usize);
        if ss >= off_ssim - tolerance {
            best = c;
            best_ssim = ss;
        }
    }
    (best, best_ssim, off_ssim)
}

fn main() {
    let home: PathBuf = std::env::var("HOME")
        .unwrap_or_else(|_| "/home/lilith".into())
        .into();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let photos_dir = home.join("work/codec-eval/codec-corpus/CID22/CID22-512/training");
    let graphics_dir = home.join("work/codec-eval/codec-corpus/gb82-sc");
    let frymire = manifest.join("tests/images/frymire.png");

    // Tolerance: we allow up to 0.05 SSIM2 points below dering-off (noise floor)
    let tolerance: f64 = 0.05;

    let mut images: Vec<(String, String, PathBuf)> = Vec::new();
    images.push(("frymire".into(), "frymire".into(), frymire));
    for p in list_pngs(&graphics_dir, 10) {
        images.push((
            "graphic".into(),
            p.file_stem().unwrap().to_str().unwrap().into(),
            p.clone(),
        ));
    }
    for p in list_pngs(&photos_dir, 10) {
        images.push((
            "photo".into(),
            p.file_stem().unwrap().to_str().unwrap().into(),
            p.clone(),
        ));
    }

    // Per-Q: for each image, find max safe offset. Then take MIN across images per Q.
    // That gives a conservative offset that won't regress any image.
    println!(
        "{:<10} {:<20} Q max_safe_offset ssim_on ssim_off",
        "category", "name"
    );
    let mut per_q_offsets: std::collections::BTreeMap<u8, Vec<(String, f32)>> = Default::default();
    for (cat, name, path) in &images {
        let Some((w, h, rgb)) = load_rgb(path) else {
            continue;
        };
        for &q in Q_LEVELS {
            let (off, ssim_on, ssim_off) = max_safe_offset(&rgb, w, h, q, tolerance);
            println!(
                "{:<10} {:<20} {:>2} {:>6.3} {:>7.3} {:>7.3}",
                cat, name, q, off, ssim_on, ssim_off
            );
            per_q_offsets
                .entry(q)
                .or_default()
                .push((format!("{cat}/{name}"), off));
        }
    }

    println!("\n=== MIN safe offset per Q (constrain to no image regressing) ===");
    for (q, lst) in &per_q_offsets {
        let min_off = lst.iter().map(|(_, o)| *o).fold(f32::INFINITY, f32::min);
        let worst = lst
            .iter()
            .filter(|(_, o)| *o == min_off)
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>();
        // Also report p25/p50 since we don't want to be overly conservative
        let mut offs: Vec<f32> = lst.iter().map(|(_, o)| *o).collect();
        offs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p25 = offs[offs.len() / 4];
        let p50 = offs[offs.len() / 2];
        println!(
            "Q={:>2}: min={:.3} p25={:.3} p50={:.3}  worst={:?}",
            q, min_off, p25, p50, worst
        );
    }
}
