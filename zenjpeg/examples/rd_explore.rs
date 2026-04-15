//! RD exploration: compares 5 zenjpeg configurations on a small corpus.
//!
//! Goal: quantify the size-at-matched-SSIM2 gap between shipping defaults
//! (`EncoderConfig::ycbcr`) and opt-in RD knobs that are available but not on by
//! default (auto_optimize, DC trellis, XYB).
//!
//! For each (image, quality):
//! - cpp:          cjpegli -q Q
//! - zen_default:  EncoderConfig::ycbcr(Q) + sharp_yuv + deringing (current shipping RD)
//! - zen_auto:     + auto_optimize(true) (enables hybrid trellis λ=14.5)
//! - zen_auto_dc:  + auto_optimize with dc_enabled=true (via hybrid_config)
//! - zen_xyb:      EncoderConfig::xyb(Q) + sharp_yuv + deringing
//!
//! Emits CSV rows: image,category,Q,config,bytes,ssim2,butter
//!
//! Usage:
//!   cargo run --release -p zenjpeg --example rd_explore -- --out benchmarks/rd_explore_2026-04-14.csv

use enough::Unstoppable;
use imgref::ImgVec;
use rgb::RGB8;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use zenjpeg::encode::trellis::HybridConfig;
use zenjpeg::encode::tuning::EncodingTables;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality, XybSubsampling};

const Q_LEVELS: &[u8] = &[50, 70, 85, 95];

fn cjpegli_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .unwrap()
        .join("internal/jpegli-cpp/build/tools/cjpegli")
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

fn zen_default(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .progressive(false)
        .deringing(true)
        .sharp_yuv(true);
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

fn zen_auto(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .auto_optimize(true)
        .deringing(true)
        .sharp_yuv(true);
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

fn zen_auto_dc(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    // Mirror auto_optimize: hybrid λ=14.5 + progressive, but enable DC trellis.
    let distance = Quality::from(q).to_distance();
    let should_use_hybrid = distance < 5.0;
    let mut cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .progressive(true)
        .deringing(true)
        .sharp_yuv(true);
    if should_use_hybrid {
        cfg = cfg.hybrid_config(HybridConfig {
            enabled: true,
            base_lambda_scale1: 14.5,
            dc_enabled: true,
            ..HybridConfig::default()
        });
    }
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

fn zen_xyb(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    // XYB color space, progressive
    let cfg = EncoderConfig::xyb(q, XybSubsampling::BQuarter)
        .progressive(true)
        .deringing(true)
        .sharp_yuv(true);
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

/// XYB encode with B-channel base-quant scaled by `b_factor`.
///
/// `b_factor > 1.0` coarsens the B (blue-yellow) quant table relative to the
/// X and Y components. The Jpegli quality scaling still applies on top of this
/// base table, so larger `b_factor` reduces surviving B AC magnitudes.
fn zen_xyb_bcoarse(rgb: &[u8], w: u32, h: u32, q: u8, b_factor: f32) -> Vec<u8> {
    let mut tables = EncodingTables::default_xyb();
    tables.quant.scale_component(2, b_factor); // component 2 = B in XYB
    let cfg = EncoderConfig::xyb(q, XybSubsampling::BQuarter)
        .progressive(true)
        .deringing(true)
        .sharp_yuv(true)
        .tables(Box::new(tables));
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

/// XYB encode at 4:4:4 (no B subsampling). Baseline for the 4:4:4 sweep.
fn zen_xyb444(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    let cfg = EncoderConfig::xyb(q, XybSubsampling::Full)
        .progressive(true)
        .deringing(true)
        .sharp_yuv(true);
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

/// XYB 4:4:4 with B-channel base-quant scaled by `b_factor`.
fn zen_xyb444_bcoarse(rgb: &[u8], w: u32, h: u32, q: u8, b_factor: f32) -> Vec<u8> {
    let mut tables = EncodingTables::default_xyb();
    tables.quant.scale_component(2, b_factor);
    let cfg = EncoderConfig::xyb(q, XybSubsampling::Full)
        .progressive(true)
        .deringing(true)
        .sharp_yuv(true)
        .tables(Box::new(tables));
    let mut e = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
    e.push_packed(rgb, Unstoppable).unwrap();
    e.finish().unwrap()
}

/// Decode using zenjpeg with ICC color correction so XYB JPEGs decode correctly.
///
/// `decode_jpeg_to_rgb` (zune-jpeg) ignored the embedded XYB ICC profile, which
/// caused all `zen_xyb*` rows to score ~-60 SSIM2.
fn ssim2(orig: &ImgVec<RGB8>, jpg: &[u8], w: usize, h: usize) -> f64 {
    let dec = match zenjpeg_bench_utils::decode_jpeg_with_icc(jpg) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("decode failed: {e:?}");
            return 0.0;
        }
    };
    if dec.width() != w || dec.height() != h {
        return 0.0;
    }
    let o3: Vec<[u8; 3]> = orig.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let d3: Vec<[u8; 3]> = dec.pixels().map(|p| [p.r, p.g, p.b]).collect();
    let oi = ImgVec::new(o3, w, h);
    let di = ImgVec::new(d3, w, h);
    fast_ssim2::compute_ssimulacra2(oi.as_ref(), di.as_ref()).unwrap_or(0.0)
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
        max_photos: 6,
        max_graphics: 3,
        out: PathBuf::from("/tmp/rd_explore.csv"),
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
    let args = parse_args();
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let mut writer = fs::File::create(&args.out).expect("create csv");
    writeln!(
        writer,
        "category,name,width,height,q,config,bytes,ssim2,encode_ms"
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

    let ppm = PathBuf::from("/tmp/rd_explore.ppm");
    let cppjpg = PathBuf::from("/tmp/rd_explore_cpp.jpg");

    let total = Instant::now();
    for (ci, (cat, name, path)) in images.iter().enumerate() {
        let Some((w, h, rgb)) = load_rgb(path) else {
            continue;
        };
        let orig_rgb: Vec<RGB8> = rgb
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let orig = ImgVec::new(orig_rgb, w as usize, h as usize);
        write_ppm(&ppm, w, h, &rgb);

        for &q in Q_LEVELS {
            // cjpegli
            let t = Instant::now();
            let cpp = cjpegli_encode(&ppm, &cppjpg, q);
            let cpp_ms = t.elapsed().as_secs_f64() * 1000.0;
            let cpp_s = ssim2(&orig, &cpp, w as usize, h as usize);
            writeln!(
                writer,
                "{cat},{name},{w},{h},{q},cpp,{b},{s:.4},{m:.2}",
                b = cpp.len(),
                s = cpp_s,
                m = cpp_ms
            )
            .unwrap();

            // base zen variants
            let variants: [(&str, fn(&[u8], u32, u32, u8) -> Vec<u8>); 4] = [
                ("zen_default", zen_default),
                ("zen_auto", zen_auto),
                ("zen_auto_dc", zen_auto_dc),
                ("zen_xyb", zen_xyb),
            ];
            for (label, f) in variants {
                let t = Instant::now();
                let j = f(&rgb, w, h, q);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let s = ssim2(&orig, &j, w as usize, h as usize);
                writeln!(
                    writer,
                    "{cat},{name},{w},{h},{q},{label},{b},{s:.4},{m:.2}",
                    b = j.len(),
                    m = ms
                )
                .unwrap();
            }

            // XYB 4:2:0 B-channel coarseness sweep
            for &factor in &[1.25_f32, 1.5, 1.75, 2.0, 2.5, 3.0] {
                let label = format!("zen_xyb_b{:.2}", factor);
                let t = Instant::now();
                let j = zen_xyb_bcoarse(&rgb, w, h, q, factor);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let s = ssim2(&orig, &j, w as usize, h as usize);
                writeln!(
                    writer,
                    "{cat},{name},{w},{h},{q},{label},{b},{s:.4},{m:.2}",
                    b = j.len(),
                    m = ms
                )
                .unwrap();
            }

            // XYB 4:4:4 baseline + B sweep
            {
                let t = Instant::now();
                let j = zen_xyb444(&rgb, w, h, q);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let s = ssim2(&orig, &j, w as usize, h as usize);
                writeln!(
                    writer,
                    "{cat},{name},{w},{h},{q},zen_xyb444,{b},{s:.4},{m:.2}",
                    b = j.len(),
                    m = ms
                )
                .unwrap();
            }
            for &factor in &[1.25_f32, 1.5, 1.75, 2.0, 2.5, 3.0] {
                let label = format!("zen_xyb444_b{:.2}", factor);
                let t = Instant::now();
                let j = zen_xyb444_bcoarse(&rgb, w, h, q, factor);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let s = ssim2(&orig, &j, w as usize, h as usize);
                writeln!(
                    writer,
                    "{cat},{name},{w},{h},{q},{label},{b},{s:.4},{m:.2}",
                    b = j.len(),
                    m = ms
                )
                .unwrap();
            }
            writer.flush().unwrap();

            let el = total.elapsed().as_secs_f64();
            println!(
                "[{:>3}/{}] {:<8} {:<28} Q={:>2} cpp={}B ssim={:.2} [t={:.0}s]",
                ci + 1,
                images.len(),
                cat,
                name,
                q,
                cpp.len(),
                cpp_s,
                el,
            );
        }
    }
    println!("Done. CSV: {}", args.out.display());
}
