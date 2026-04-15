//! Find the maximum distance offset that keeps zen SSIM2 >= cjpegli SSIM2 - tol
//! across all tested images, per Q.

use enough::Unstoppable;
use imgref::ImgVec;
use rgb::RGB8;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

const Q_LEVELS: &[u8] = &[10, 30, 50, 70, 85, 95];

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

fn cjpegli_bytes(ppm: &str, q: u8) -> Vec<u8> {
    let out = "/tmp/vscpp.jpg";
    Command::new(cjpegli())
        .args([
            ppm,
            out,
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
    fs::read(out).unwrap()
}

fn zen_bytes(rgb: &[u8], w: u32, h: u32, q: u8, offset: f32) -> Vec<u8> {
    let base = Quality::ApproxJpegli(q as f32).to_distance();
    let cfg = EncoderConfig::ycbcr(
        Quality::ApproxButteraugli(base + offset),
        ChromaSubsampling::Quarter,
    )
    .progressive(false)
    .deringing(true);
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

fn main() {
    let home: PathBuf = std::env::var("HOME")
        .unwrap_or_else(|_| "/home/lilith".into())
        .into();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let photos_dir = home.join("work/codec-eval/codec-corpus/CID22/CID22-512/training");
    let graphics_dir = home.join("work/codec-eval/codec-corpus/gb82-sc");
    let frymire = manifest.join("tests/images/frymire.png");

    // Tolerance vs cjpegli: zen SSIM >= cpp SSIM - 0.5
    let tol: f64 = 0.5;

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

    let candidates: [f32; 12] = [
        0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.30, 0.40,
    ];

    let mut per_q_offsets: std::collections::BTreeMap<u8, Vec<(String, f32)>> = Default::default();
    let ppm = "/tmp/vscpp.ppm";
    for (cat, name, path) in &images {
        let Some((w, h, rgb)) = load_rgb(path) else {
            continue;
        };
        let orig_rgb: Vec<RGB8> = rgb
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let orig = ImgVec::new(orig_rgb, w as usize, h as usize);
        write_ppm(ppm, w, h, &rgb);

        for &q in Q_LEVELS {
            let cpp = cjpegli_bytes(ppm, q);
            let cpp_ssim = ssim2(&orig, &cpp, w as usize, h as usize);

            // Find max offset with zen_ssim >= cpp_ssim - tol
            let mut best = 0.0f32;
            for &c in &candidates {
                let zb = zen_bytes(&rgb, w, h, q, c);
                let zs = ssim2(&orig, &zb, w as usize, h as usize);
                if zs >= cpp_ssim - tol {
                    best = c;
                } else {
                    break;
                }
            }
            println!(
                "{:<10} {:<20} Q={:>2} max_off={:.3} cpp_ssim={:.2}",
                cat, name, q, best, cpp_ssim
            );
            per_q_offsets.entry(q).or_default().push((format!("{cat}/{name}"), best));
        }
    }

    println!("\n=== MIN max_offset per Q (no regression vs cjpegli-0.5) ===");
    for (q, lst) in &per_q_offsets {
        let mut offs: Vec<f32> = lst.iter().map(|(_, o)| *o).collect();
        offs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = offs[0];
        let p25 = offs[offs.len() / 4];
        let p50 = offs[offs.len() / 2];
        let p75 = offs[offs.len() * 3 / 4];
        let worst: Vec<String> = lst
            .iter()
            .filter(|(_, o)| (*o - min).abs() < 1e-6)
            .map(|(n, _)| n.clone())
            .collect();
        println!(
            "Q={:>2}: min={:.3} p25={:.3} p50={:.3} p75={:.3}  limiting={:?}",
            q, min, p25, p50, p75, worst
        );
    }
}
