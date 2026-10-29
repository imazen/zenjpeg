//! Per-class quant table signal probe (gating experiment, 2026-05-04).
//!
//! Goal: Measure whether per-content-class quant tables are worth pursuing.
//!
//! Compares three table sets at q=15/50/85 across photo + screen content:
//!  A. v4 photo-tuned defaults (`sa_piecewise_v4::tables_for_quality`).
//!  B. "Screen-friendly": v4 with the bottom-right 4x4 high-freq region zeroed.
//!  C. "DC-emphasized": v4 with luma DC quant halved.
//!
//! NOTE: VARIANT B sets the quant matrix entries to 1 (the legal minimum
//! for u16 quant in JPEG), not literal zero — quant=0 is invalid. A quant
//! value of 1 means "do not quantize at all", which preserves all energy
//! at those frequencies.
//!
//! For each (image, table, q): encode -> decode -> bytes + butteraugli.
//! Output: TSV to stdout + summary stats to stderr.
//!
//! Run:
//!   cargo run --example per_class_signal_probe --release --features __test-utils -- \
//!     --photo  /home/lilith/work/zentrain-corpus/mlp-tune-fast/cid22-train \
//!     --screen /home/lilith/work/zentrain-corpus/mlp-tune-fast/gb82-screen \
//!     --photo-n 20 --photo-seed 7 \
//!     --output /tmp/per_class_probe.tsv

use butteraugli::ButteraugliParams;
use enough::Unstoppable;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use zenjpeg::encode::tables::sa_piecewise_v4;
use zenjpeg::encode::tuning::{EncodingTables, ScalingParams};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

const QS: [u8; 3] = [15, 50, 85];
const TABLES: [&str; 3] = ["A_v4", "B_zero_hf", "C_half_dc"];

fn parse_args() -> (PathBuf, PathBuf, usize, u64, PathBuf) {
    let mut photo = None;
    let mut screen = None;
    let mut photo_n: usize = 20;
    let mut photo_seed: u64 = 7;
    let mut output = PathBuf::from("/tmp/per_class_probe.tsv");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--photo" => photo = Some(PathBuf::from(args.next().unwrap())),
            "--screen" => screen = Some(PathBuf::from(args.next().unwrap())),
            "--photo-n" => photo_n = args.next().unwrap().parse().unwrap(),
            "--photo-seed" => photo_seed = args.next().unwrap().parse().unwrap(),
            "--output" => output = PathBuf::from(args.next().unwrap()),
            _ => panic!("unknown arg: {a}"),
        }
    }
    (
        photo.expect("--photo required"),
        screen.expect("--screen required"),
        photo_n,
        photo_seed,
        output,
    )
}

/// Tiny LCG sampler — deterministic, no external rand dep.
fn lcg_sample<T: Clone>(items: &[T], n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut idx: Vec<usize> = (0..items.len()).collect();
    // Fisher-Yates with LCG
    for i in (1..idx.len()).rev() {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (s >> 33) as usize % (i + 1);
        idx.swap(i, j);
    }
    idx.into_iter()
        .take(n.min(items.len()))
        .map(|i| items[i].clone())
        .collect()
}

fn list_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("png"))
        })
        .collect();
    out.sort();
    out
}

fn load_rgb8(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = match zenjpeg_bench_utils::load_png(path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("WARN load {path:?}: {e}");
            return None;
        }
    };
    let (buf, w, h) = img.into_contiguous_buf();
    let bytes: Vec<u8> = buf.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Some((bytes, w as u32, h as u32))
}

fn make_table_a(q: u8) -> EncodingTables {
    sa_piecewise_v4::tables_for_quality(q)
}

/// "Screen-friendly": zero-out (legal min = 1) all entries in the 4x4
/// bottom-right block (rows 4-7, cols 4-7) of luma + cb + cr quant
/// matrices. JPEG forbids quant=0, so we use 1 (= "do not quantize").
fn make_table_b(q: u8) -> EncodingTables {
    let mut t = make_table_a(q);
    for row in 4..8 {
        for col in 4..8 {
            let idx = row * 8 + col;
            t.quant.c0[idx] = 1.0;
            t.quant.c1[idx] = 1.0;
            t.quant.c2[idx] = 1.0;
        }
    }
    // Force ScalingParams::Exact so our edits aren't rescaled by quality.
    t.scaling = ScalingParams::Exact;
    t
}

/// "DC-emphasized": halve the luma DC quant.
fn make_table_c(q: u8) -> EncodingTables {
    let mut t = make_table_a(q);
    let dc = t.quant.c0[0];
    let new_dc = (dc * 0.5).max(1.0);
    t.quant.c0[0] = new_dc;
    t.scaling = ScalingParams::Exact;
    t
}

fn make_table(name: &str, q: u8) -> EncodingTables {
    match name {
        "A_v4" => make_table_a(q),
        "B_zero_hf" => make_table_b(q),
        "C_half_dc" => make_table_c(q),
        _ => unreachable!(),
    }
}

fn encode(rgb: &[u8], w: u32, h: u32, q: u8, tables: EncodingTables) -> Option<Vec<u8>> {
    let cfg = EncoderConfig::ycbcr(
        Quality::ApproxJpegli(f32::from(q)),
        ChromaSubsampling::Quarter,
    )
    .tables(Box::new(tables));
    let mut enc = match cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ERR encode_from_bytes: {e:?}");
            return None;
        }
    };
    if let Err(e) = enc.push_packed(rgb, Unstoppable) {
        eprintln!("ERR push_packed: {e:?}");
        return None;
    }
    match enc.finish() {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            eprintln!("ERR finish: {e:?}");
            None
        }
    }
}

fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    let mut decoder =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
    decoder.decode().ok()
}

fn butter(orig: &[u8], dec: &[u8], w: usize, h: usize) -> f64 {
    let orig_p: Vec<rgb::RGB8> = orig
        .chunks_exact(3)
        .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dec_p: Vec<rgb::RGB8> = dec
        .chunks_exact(3)
        .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
        .collect();
    if orig_p.len() != dec_p.len() {
        return f64::NAN;
    }
    let oimg = imgref::Img::new(&orig_p[..], w, h);
    let dimg = imgref::Img::new(&dec_p[..], w, h);
    let params = ButteraugliParams::default();
    butteraugli::butteraugli(oimg, dimg, &params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN)
}

fn main() {
    let (photo_dir, screen_dir, photo_n, photo_seed, out) = parse_args();
    eprintln!(
        "per-class signal probe:\n  photo: {photo_dir:?} (n={photo_n}, seed={photo_seed})\n  screen: {screen_dir:?}"
    );

    let all_photo = list_pngs(&photo_dir);
    let photos = lcg_sample(&all_photo, photo_n, photo_seed);
    let screens = list_pngs(&screen_dir);
    eprintln!(
        "  photo set: {} files (from {} total)",
        photos.len(),
        all_photo.len()
    );
    eprintln!("  screen set: {} files", screens.len());

    let mut tsv = fs::File::create(&out).expect("open output");
    writeln!(
        tsv,
        "image\tcontent_class\ttable\tq\tbytes\tbutteraugli\tw\th"
    )
    .unwrap();

    let work: Vec<(PathBuf, &'static str)> = photos
        .iter()
        .map(|p| (p.clone(), "photo"))
        .chain(screens.iter().map(|p| (p.clone(), "screen")))
        .collect();

    let total_cells = work.len() * TABLES.len() * QS.len();
    let mut done = 0usize;
    for (path, klass) in &work {
        let Some((rgb, w, h)) = load_rgb8(path) else {
            continue;
        };
        for table_name in TABLES {
            for &q in &QS {
                let t = make_table(table_name, q);
                let Some(jpeg) = encode(&rgb, w, h, q, t) else {
                    continue;
                };
                let Some(dec) = decode_jpeg(&jpeg) else {
                    eprintln!("WARN decode failed: {path:?} {table_name} q={q}");
                    continue;
                };
                let ba = butter(&rgb, &dec, w as usize, h as usize);
                writeln!(
                    tsv,
                    "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
                    path.file_name().unwrap().to_string_lossy(),
                    klass,
                    table_name,
                    q,
                    jpeg.len(),
                    ba,
                    w,
                    h
                )
                .unwrap();
                done += 1;
                if done % 30 == 0 {
                    eprintln!("  {done}/{total_cells}");
                }
            }
        }
    }
    eprintln!("done: {done}/{total_cells} cells -> {out:?}");
}
