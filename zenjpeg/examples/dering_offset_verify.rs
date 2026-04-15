//! Verify that the dering distance offset matches cjpegli size while keeping SSIM2
//! at least as high as the dering-off case.

use enough::Unstoppable;
use imgref::ImgVec;
use rgb::RGB8;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

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

fn cjpegli_encode(ppm: &str, q: u8) -> Vec<u8> {
    let out = "/tmp/verify_cpp.jpg";
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

fn zen_encode(rgb: &[u8], w: u32, h: u32, q: u8, dering: bool) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .progressive(false)
        .deringing(dering);
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

fn ssim2_u8(orig: &ImgVec<RGB8>, dist_bytes: &[u8], w: usize, h: usize) -> f64 {
    // Decode JPEG first
    let dec = zenjpeg_bench_utils::decode_jpeg_to_rgb(dist_bytes).unwrap();
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

fn verify_image(category: &str, name: &str, path: &Path) {
    let Some((w, h, rgb)) = load_rgb(path) else {
        return;
    };
    let orig_rgb: Vec<RGB8> = rgb
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig = ImgVec::new(orig_rgb, w as usize, h as usize);

    let ppm = "/tmp/verify.ppm";
    write_ppm(ppm, w, h, &rgb);

    for &q in Q_LEVELS {
        let cpp = cjpegli_encode(ppm, q);
        let cpp_ssim = ssim2_u8(&orig, &cpp, w as usize, h as usize);

        let zen_on = zen_encode(&rgb, w, h, q, true);
        let zen_on_ssim = ssim2_u8(&orig, &zen_on, w as usize, h as usize);

        let zen_off = zen_encode(&rgb, w, h, q, false);
        let zen_off_ssim = ssim2_u8(&orig, &zen_off, w as usize, h as usize);

        let size_delta_pct = (zen_on.len() as f64 - cpp.len() as f64) / cpp.len() as f64 * 100.0;
        let ssim_delta_cpp = zen_on_ssim - cpp_ssim;
        let ssim_delta_off = zen_on_ssim - zen_off_ssim;
        println!(
            "{:<10} {:<30} Q={:>2} | cpp={:>7} zen={:>7} ({:+.2}%) | ssim cpp={:>5.2} zen={:>5.2} off={:>5.2} | Δvcpp={:+.3} Δvoff={:+.3}",
            category, name, q, cpp.len(), zen_on.len(), size_delta_pct,
            cpp_ssim, zen_on_ssim, zen_off_ssim, ssim_delta_cpp, ssim_delta_off
        );
    }
}

fn main() {
    let home: PathBuf = std::env::var("HOME")
        .unwrap_or_else(|_| "/home/lilith".into())
        .into();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let photos_dir = home.join("work/codec-eval/codec-corpus/CID22/CID22-512/training");
    let graphics_dir = home.join("work/codec-eval/codec-corpus/gb82-sc");
    let frymire = manifest.join("tests/images/frymire.png");

    println!("== frymire ==");
    verify_image("frymire", "frymire", &frymire);

    println!("\n== graphics ==");
    for p in list_pngs(&graphics_dir, 5) {
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        verify_image("graphic", &name, &p);
    }

    println!("\n== photos ==");
    for p in list_pngs(&photos_dir, 10) {
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        verify_image("photo", &name, &p);
    }
}
