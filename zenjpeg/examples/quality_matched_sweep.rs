//! Quality-matched size sweep: zenjpeg (Sharp YUV iter=2 + deringing) vs cjpegli
//! at matched-quality (SSIM2 or butteraugli), across Q ∈ {10,30,50,70,85,95}.
//!
//! For each (image, Q):
//!   1. cjpegli at Q  → cjpegli_size, cjpegli_ssim2, cjpegli_butter
//!   2. zen at base_distance(Q) + Δ for Δ in {-0.4,-0.25,-0.15,-0.08,0,0.05,0.12,0.2,0.3,0.5,0.8,1.2,1.8,2.5}
//!      → records (size, ssim2, butter) at each Δ
//!
//! Emits one CSV row per (image, Q, source, offset).
//!
//! Usage:
//!   cargo run --release -p zenjpeg --example quality_matched_sweep -- \
//!       --out benchmarks/quality_matched_2026-04-14.csv \
//!       --max-photos 25 --max-graphics 10

use enough::Unstoppable;
use imgref::ImgVec;
use rgb::RGB8;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use butteraugli::ButteraugliParams;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

const Q_LEVELS: &[u8] = &[10, 30, 50, 70, 85, 95];

// Distance offsets to sweep (distance = base(Q) + Δ; larger = lower quality, smaller files)
const OFFSETS: &[f32] = &[
    -0.4, -0.25, -0.15, -0.08, 0.0, 0.05, 0.12, 0.20, 0.30, 0.50, 0.80, 1.20, 1.80, 2.50,
];

fn cjpegli_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .unwrap()
        .join("internal/jpegli-cpp/build/tools/cjpegli")
}

struct Args {
    photos: PathBuf,
    graphics: PathBuf,
    frymire: PathBuf,
    max_photos: usize,
    max_graphics: usize,
    out: PathBuf,
}

fn parse_args() -> Args {
    let home: PathBuf = std::env::var("HOME")
        .unwrap_or_else(|_| "/home/lilith".into())
        .into();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut a = Args {
        photos: home.join("work/codec-eval/codec-corpus/CID22/CID22-512/training"),
        graphics: home.join("work/codec-eval/codec-corpus/gb82-sc"),
        frymire: manifest.join("tests/images/frymire.png"),
        max_photos: 25,
        max_graphics: 10,
        out: PathBuf::from("/tmp/quality_matched_sweep.csv"),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--photos" => a.photos = it.next().unwrap().into(),
            "--graphics" => a.graphics = it.next().unwrap().into(),
            "--frymire" => a.frymire = it.next().unwrap().into(),
            "--max-photos" => a.max_photos = it.next().unwrap().parse().unwrap(),
            "--max-graphics" => a.max_graphics = it.next().unwrap().parse().unwrap(),
            "--out" => a.out = it.next().unwrap().into(),
            other => panic!("unknown arg: {other}"),
        }
    }
    a
}

fn list_pngs(dir: &Path, limit: usize) -> Vec<PathBuf> {
    let Ok(rd) = fs::read_dir(dir) else {
        eprintln!("warn: could not read {}", dir.display());
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

fn load_rgb(path: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    Some((w, h, img.into_raw()))
}

fn write_ppm(path: &Path, w: u32, h: u32, rgb: &[u8]) {
    let mut f = fs::File::create(path).unwrap();
    writeln!(f, "P6\n{w} {h}\n255").unwrap();
    f.write_all(rgb).unwrap();
}

fn cjpegli_encode(ppm: &Path, out_jpg: &Path, q: u8) -> Vec<u8> {
    Command::new(cjpegli_path())
        .args([
            ppm.to_str().unwrap(),
            out_jpg.to_str().unwrap(),
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
    fs::read(out_jpg).unwrap()
}

fn zen_encode(rgb: &[u8], w: u32, h: u32, q: u8, offset: f32) -> Vec<u8> {
    // base = jpegli distance formula at Q
    let base = Quality::ApproxJpegli(q as f32).to_distance();
    let distance = (base + offset).max(0.01);
    let cfg = EncoderConfig::ycbcr(
        Quality::ApproxButteraugli(distance),
        ChromaSubsampling::Quarter,
    )
    .progressive(false)
    .deringing(true)
    .sharp_yuv(true);
    let mut e = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode_from_bytes");
    e.push_packed(rgb, Unstoppable).expect("push_packed");
    e.finish().expect("finish")
}

fn ssim2(orig: &ImgVec<RGB8>, jpg: &[u8], w: usize, h: usize) -> f64 {
    let dec = match zenjpeg_bench_utils::decode_jpeg_to_rgb(jpg) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("decode failed: {e:?}");
            return 0.0;
        }
    };
    if dec.width() != w || dec.height() != h {
        eprintln!("decoded dim mismatch: {}x{} != {}x{}", dec.width(), dec.height(), w, h);
        return 0.0;
    }
    let o3: Vec<[u8; 3]> = orig.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let d3: Vec<[u8; 3]> = dec.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let oi = ImgVec::new(o3, w, h);
    let di = ImgVec::new(d3, w, h);
    fast_ssim2::compute_ssimulacra2(oi.as_ref(), di.as_ref()).unwrap_or(0.0)
}

fn butter(orig: &[RGB8], jpg: &[u8], w: usize, h: usize) -> f64 {
    let dec = match zenjpeg_bench_utils::decode_jpeg_to_rgb(jpg) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };
    if dec.width() != w || dec.height() != h {
        return f64::NAN;
    }
    let dist_pixels: Vec<RGB8> = dec.pixels().collect();
    let oi = imgref::Img::new(orig, w, h);
    let di = imgref::Img::new(&dist_pixels[..], w, h);
    let params = ButteraugliParams::default();
    match butteraugli::butteraugli(oi, di, &params) {
        Ok(r) => r.score as f64,
        Err(_) => f64::NAN,
    }
}

fn main() {
    let args = parse_args();
    println!("photos: {}", args.photos.display());
    println!("graphics: {}", args.graphics.display());
    println!("frymire: {}", args.frymire.display());
    println!("out: {}", args.out.display());
    println!("cjpegli: {}", cjpegli_path().display());
    assert!(cjpegli_path().exists(), "cjpegli missing");

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    let mut writer = fs::File::create(&args.out).expect("create csv");
    writeln!(
        writer,
        "category,name,width,height,q,source,offset,base_distance,bytes,ssim2,butter,encode_ms"
    )
    .unwrap();

    let mut images: Vec<(String, String, PathBuf)> = Vec::new();
    if args.frymire.exists() {
        images.push((
            "frymire".into(),
            args.frymire.file_stem().unwrap().to_str().unwrap().into(),
            args.frymire.clone(),
        ));
    }
    for p in list_pngs(&args.photos, args.max_photos) {
        images.push((
            "photo".into(),
            p.file_stem().unwrap().to_str().unwrap().into(),
            p.clone(),
        ));
    }
    for p in list_pngs(&args.graphics, args.max_graphics) {
        images.push((
            "graphic".into(),
            p.file_stem().unwrap().to_str().unwrap().into(),
            p.clone(),
        ));
    }

    let ppm = PathBuf::from("/tmp/qmsweep.ppm");
    let cppjpg = PathBuf::from("/tmp/qmsweep_cpp.jpg");

    let total_start = Instant::now();
    for (ci, (cat, name, path)) in images.iter().enumerate() {
        let Some((w, h, rgb)) = load_rgb(path) else {
            eprintln!("skip {}: load failed", path.display());
            continue;
        };
        let orig_rgb: Vec<RGB8> = rgb
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let orig = ImgVec::new(orig_rgb.clone(), w as usize, h as usize);
        write_ppm(&ppm, w, h, &rgb);

        for &q in Q_LEVELS {
            let base = Quality::ApproxJpegli(q as f32).to_distance();

            // cjpegli
            let t = Instant::now();
            let cpp = cjpegli_encode(&ppm, &cppjpg, q);
            let cpp_ms = t.elapsed().as_secs_f64() * 1000.0;
            let cpp_ssim = ssim2(&orig, &cpp, w as usize, h as usize);
            let cpp_butt = butter(&orig_rgb, &cpp, w as usize, h as usize);
            writeln!(
                writer,
                "{cat},{name},{w},{h},{q},cjpegli,0.000,{base:.3},{bytes},{s:.4},{b:.4},{ms:.2}",
                bytes = cpp.len(),
                s = cpp_ssim,
                b = cpp_butt,
                ms = cpp_ms,
            )
            .unwrap();

            // zen at each offset
            for &off in OFFSETS {
                let t = Instant::now();
                let jpg = zen_encode(&rgb, w, h, q, off);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let s = ssim2(&orig, &jpg, w as usize, h as usize);
                let b = butter(&orig_rgb, &jpg, w as usize, h as usize);
                writeln!(
                    writer,
                    "{cat},{name},{w},{h},{q},zen,{off:.3},{base:.3},{bytes},{s:.4},{b:.4},{ms:.2}",
                    bytes = jpg.len(),
                )
                .unwrap();
            }
            writer.flush().unwrap();
            let elapsed = total_start.elapsed().as_secs_f64();
            println!(
                "[{:>4}/{}] {:<8} {:<24} Q={:>2} cpp={}B ssim={:.2} butter={:.3} [t={:.0}s]",
                ci + 1,
                images.len(),
                cat,
                name,
                q,
                cpp.len(),
                cpp_ssim,
                cpp_butt,
                elapsed,
            );
        }
    }

    println!("\nDone. CSV: {}", args.out.display());
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());
}
