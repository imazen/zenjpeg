//! Corpus sweep for `preprocess_deringing` feature.
//!
//! For each (image, Q), encodes with deringing ON and OFF, decodes, and
//! measures SSIMULACRA2 + size. Writes one CSV row per (image, Q, dering).
//!
//! Usage:
//!   cargo run --release -p zenjpeg --example dering_corpus_sweep -- \
//!       --out /mnt/v/output/zenjpeg/dering_sweep.csv
//!
//! Inputs:
//!   --photos DIR   (default: ~/work/codec-eval/codec-corpus/CID22/CID22-512/training)
//!   --graphics DIR (default: ~/work/codec-eval/codec-corpus/gb82-sc)
//!   --frymire PATH (default: zenjpeg/tests/images/frymire.png)
//!   --max-photos N (default: 20)
//!   --max-graphics N (default: 10)

use enough::Unstoppable;
use rgb::RGB8;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use imgref::ImgVec;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg_bench_utils::decode_jpeg_to_rgb;

const Q_LEVELS: &[u8] = &[10, 30, 50, 70, 85, 95];

fn find_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into()))
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
    let home = find_home();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut a = Args {
        photos: home.join("work/codec-eval/codec-corpus/CID22/CID22-512/training"),
        graphics: home.join("work/codec-eval/codec-corpus/gb82-sc"),
        frymire: manifest.join("tests/images/frymire.png"),
        max_photos: 20,
        max_graphics: 10,
        out: PathBuf::from("/tmp/dering_sweep.csv"),
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

fn encode_with_dering(
    rgb: &[u8],
    w: u32,
    h: u32,
    q: u8,
    dering: bool,
) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .progressive(false)
        .deringing(dering);
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode_from_bytes");
    enc.push_packed(rgb, Unstoppable).expect("push_packed");
    enc.finish().expect("finish")
}

fn ssim2(orig: &ImgVec<RGB8>, dist_bytes: &[u8], w: usize, h: usize) -> f64 {
    let dist_rgb: Vec<RGB8> = dist_bytes
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dist = ImgVec::new(dist_rgb, w, h);
    // Convert to [u8; 3] for fast-ssim2
    let o3: Vec<[u8; 3]> = orig.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let d3: Vec<[u8; 3]> = dist.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let oi = ImgVec::new(o3, orig.width(), orig.height());
    let di = ImgVec::new(d3, dist.width(), dist.height());
    fast_ssim2::compute_ssimulacra2(oi.as_ref(), di.as_ref()).unwrap_or(0.0)
}

fn process_image(
    category: &str,
    name: &str,
    path: &Path,
    writer: &mut fs::File,
) {
    let Some((w, h, rgb)) = load_rgb(path) else {
        eprintln!("skip {}: load failed", path.display());
        return;
    };
    let orig_rgb: Vec<RGB8> = rgb
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let orig = ImgVec::new(orig_rgb, w as usize, h as usize);

    for &q in Q_LEVELS {
        for dering in [true, false] {
            let t = Instant::now();
            let jpg = encode_with_dering(&rgb, w, h, q, dering);
            let encode_ms = t.elapsed().as_secs_f64() * 1000.0;

            let decoded = decode_jpeg_to_rgb(&jpg).expect("decode");
            assert_eq!(decoded.width(), w as usize);
            assert_eq!(decoded.height(), h as usize);
            // decoded is ImgVec<RGB8>; convert back to bytes for ssim2 helper
            let dec_bytes: Vec<u8> = decoded
                .pixels()
                .flat_map(|p| [p.r, p.g, p.b])
                .collect();

            let s = ssim2(&orig, &dec_bytes, w as usize, h as usize);

            writeln!(
                writer,
                "{category},{name},{w},{h},{q},{dering},{bytes},{ssim2:.6},{ms:.2}",
                bytes = jpg.len(),
                ssim2 = s,
                ms = encode_ms,
            )
            .unwrap();

            println!(
                "{:<10} {:<32} Q={:>2} dering={:>5} bytes={:>7} ssim2={:>7.3}",
                category, name, q, dering, jpg.len(), s
            );
        }
    }
    writer.flush().unwrap();
}

fn main() {
    let args = parse_args();
    println!("photos: {}", args.photos.display());
    println!("graphics: {}", args.graphics.display());
    println!("frymire: {}", args.frymire.display());
    println!("out: {}", args.out.display());

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    let mut writer = fs::File::create(&args.out).expect("create csv");
    writeln!(
        writer,
        "category,name,width,height,q,dering,bytes,ssim2,encode_ms"
    )
    .unwrap();

    // Frymire (screenshot reference)
    if args.frymire.exists() {
        process_image(
            "frymire",
            args.frymire.file_stem().unwrap().to_str().unwrap(),
            &args.frymire,
            &mut writer,
        );
    }

    // Photos (CID22)
    let photos = list_pngs(&args.photos, args.max_photos);
    println!("\n== photos ({}) ==", photos.len());
    for p in &photos {
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        process_image("photo", &name, p, &mut writer);
    }

    // Graphics / screenshots
    let graphics = list_pngs(&args.graphics, args.max_graphics);
    println!("\n== graphics ({}) ==", graphics.len());
    for p in &graphics {
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        process_image("graphic", &name, p, &mut writer);
    }

    println!("\nDone. CSV: {}", args.out.display());
}
