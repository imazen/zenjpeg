//! Calibrate a per-Q distance offset that compensates for deringing size overhead.
//!
//! For each image in the calibration set and each Q in {10,30,50,70,85,95}, encodes
//! with deringing ON at a range of distance offsets and picks the offset that most
//! closely matches cjpegli's size at that Q. Aggregates per-Q across corpus.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

fn cjpegli() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .unwrap()
        .join("internal/jpegli-cpp/build/tools/cjpegli")
}

fn write_ppm(path: &str, w: u32, h: u32, rgb: &[u8]) {
    let mut f = fs::File::create(path).unwrap();
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    f.write_all(rgb).unwrap();
}

fn cjpegli_size(ppm: &str, q: u8) -> u64 {
    let out = format!("/tmp/dering_calib_cpp_{q}.jpg");
    let status = Command::new(cjpegli())
        .args([
            ppm,
            &out,
            "-q",
            &q.to_string(),
            "--chroma_subsampling",
            "420",
            "-p",
            "0",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
    fs::metadata(&out).unwrap().len()
}

fn zen_size_with_offset(rgb: &[u8], w: u32, h: u32, q: u8, dering: bool, dist_offset: f32) -> u64 {
    // Get base distance from Q
    let base_q = Quality::ApproxJpegli(q as f32);
    let base_dist = base_q.to_distance();
    let eff = base_dist + dist_offset;
    let cfg = EncoderConfig::ycbcr(Quality::ApproxButteraugli(eff), ChromaSubsampling::Quarter)
        .progressive(false)
        .deringing(dering);
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    let b = e.finish().unwrap();
    b.len() as u64
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

fn calibrate_one(rgb: &[u8], w: u32, h: u32, q: u8) -> f32 {
    // Binary search over dist_offset in [0, 2.0]
    let ppm = "/tmp/dering_calib.ppm";
    write_ppm(ppm, w, h, rgb);
    let target = cjpegli_size(ppm, q) as i64;
    // size decreases as offset increases — monotone
    let mut lo = 0.0f32;
    let mut hi = 2.0f32;
    // For safety, check endpoints — if even offset=0 is smaller than target, return 0.
    let s_lo = zen_size_with_offset(rgb, w, h, q, true, lo) as i64;
    if s_lo <= target {
        return 0.0;
    }
    // Bisect
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        let s = zen_size_with_offset(rgb, w, h, q, true, mid) as i64;
        if s > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 0.002 {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn main() {
    let home: PathBuf = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into()).into();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let photos_dir = home.join("work/codec-eval/codec-corpus/CID22/CID22-512/training");
    let graphics_dir = home.join("work/codec-eval/codec-corpus/gb82-sc");
    let frymire = manifest.join("tests/images/frymire.png");

    let photos = list_pngs(&photos_dir, 10);
    let graphics = list_pngs(&graphics_dir, 10);

    let q_levels = [10u8, 30, 50, 70, 85, 95];

    println!("Calibrating per-Q distance offsets to match cjpegli size...\n");

    // Collect per-Q list of offsets
    let mut per_q_graphic: std::collections::BTreeMap<u8, Vec<f32>> = Default::default();
    let mut per_q_photo: std::collections::BTreeMap<u8, Vec<f32>> = Default::default();
    let mut per_q_frymire: std::collections::BTreeMap<u8, f32> = Default::default();

    // Frymire
    if let Some((w, h, rgb)) = load_rgb(&frymire) {
        for q in q_levels {
            let off = calibrate_one(&rgb, w, h, q);
            per_q_frymire.insert(q, off);
            println!("frymire  Q={q:>2}: offset = {off:.3}");
        }
    }

    for p in &graphics {
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        let Some((w, h, rgb)) = load_rgb(p) else { continue };
        for q in q_levels {
            let off = calibrate_one(&rgb, w, h, q);
            per_q_graphic.entry(q).or_default().push(off);
            print!("graphic {name:<20} Q={q:>2}: offset = {off:.3}\n");
        }
    }

    for p in &photos {
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        let Some((w, h, rgb)) = load_rgb(p) else { continue };
        for q in q_levels {
            let off = calibrate_one(&rgb, w, h, q);
            per_q_photo.entry(q).or_default().push(off);
            print!("photo   {name:<20} Q={q:>2}: offset = {off:.3}\n");
        }
    }

    fn stats(xs: &[f32]) -> (f32, f32, f32) {
        let mut v: Vec<f32> = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let median = v[v.len() / 2];
        let p75 = v[(v.len() * 3) / 4];
        (mean, median, p75)
    }

    println!("\n=== Per-Q offset summary ===");
    println!("{:<5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Q", "frymire", "g_mean", "g_median", "g_p75", "p_mean", "p_median", "p_p75");
    for q in q_levels {
        let g = per_q_graphic.get(&q).cloned().unwrap_or_default();
        let p = per_q_photo.get(&q).cloned().unwrap_or_default();
        let f = per_q_frymire.get(&q).copied().unwrap_or(0.0);
        let (gm, gmd, gp75) = if g.is_empty() { (0.0,0.0,0.0) } else { stats(&g) };
        let (pm, pmd, pp75) = if p.is_empty() { (0.0,0.0,0.0) } else { stats(&p) };
        println!("{:<5} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            q, f, gm, gmd, gp75, pm, pmd, pp75);
    }
}
