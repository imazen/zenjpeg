//! Per-class quant table signal probe — B' variant (gating Step 0, 2026-05-04).
//!
//! Follow-up to `per_class_signal_probe.rs`. The prior probe (commit cc61c2d
//! in zenanalyze, 368f2ad6 in zenjpeg) found that perturbation B (force HF
//! quant=1, i.e. "preserve all HF energy") wins on at-q butteraugli on screens
//! by 4-10x more than on photos — but at +37% to +283% bytes. At matched
//! bytes B is dramatically WORSE on both classes.
//!
//! That report's recommended Step 0 to gate any per-class SA work was the
//! OPPOSITE perturbation: increase HF quant on the bottom-right 4x4 block.
//! Hypothesis: screens have ~zero HF AC energy, so coarsening HF quant should
//! cost ~nothing on screens but degrade photos. If true, that's the asymmetric
//! signal that justifies per-class SA.
//!
//! Variants (all on the bottom-right 4x4, rows 4-7 x cols 4-7, natural-order
//! quant matrix indices, applied to luma + Cb + Cr):
//!
//!   A. v4 baseline (`sa_piecewise_v4::tables_for_quality(q)`).
//!   B'_2x  — HF quant scaled x2 (capped at 255).
//!   B'_4x  — HF quant scaled x4 (capped at 255).
//!   B'_8x  — HF quant scaled x8 (capped at 255).
//!
//! Gating criterion (from the prior report, verbatim):
//!
//!   "If B' shows screen ΔBA ≤ +0.05 AND photo ΔBA ≥ +0.30 at fixed q with
//!    screens shrinking by 5-15%, that's the STRONG signal worth investing
//!    SA budget against per-class tables. If B' is also flat or noisy, the
//!    per-class hypothesis is dead."
//!
//! For each (image, table, q): encode -> decode -> bytes + butteraugli.
//! Output: TSV.
//!
//! Run:
//!   cargo run --example per_class_signal_probe_bprime --release \
//!     --features __test-utils -- \
//!     --photo  /home/lilith/work/zentrain-corpus/mlp-tune-fast/cid22-train \
//!     --screen /home/lilith/work/zentrain-corpus/mlp-tune-fast/gb82-screen \
//!     --photo-n 20 --photo-seed 7 \
//!     --output /tmp/per_class_probe_bprime.tsv

use butteraugli::ButteraugliParams;
use enough::Unstoppable;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use zenjpeg::encode::tables::sa_piecewise_v4;
use zenjpeg::encode::tuning::{EncodingTables, ScalingParams};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

const QS: [u8; 3] = [15, 50, 85];
const TABLES: [&str; 4] = ["A_v4", "Bp_2x", "Bp_4x", "Bp_8x"];

/// Per-class corpus: directory + class label + sample size.
struct Cluster {
    dir: PathBuf,
    klass: String,
    n: usize,
}

fn parse_args() -> (Vec<Cluster>, u64, PathBuf) {
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut photo_seed: u64 = 7;
    let mut output = PathBuf::from("/tmp/per_class_probe_bprime.tsv");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cluster" => {
                // --cluster name=path,n=N
                let spec = args.next().expect("--cluster needs spec");
                let mut name: Option<String> = None;
                let mut path: Option<PathBuf> = None;
                let mut n: usize = 20;
                for kv in spec.split(',') {
                    if let Some((k, v)) = kv.split_once('=') {
                        match k {
                            "name" => name = Some(v.to_string()),
                            "path" => path = Some(PathBuf::from(v)),
                            "n" => n = v.parse().unwrap(),
                            _ => panic!("unknown cluster key: {k}"),
                        }
                    }
                }
                clusters.push(Cluster {
                    dir: path.expect("cluster path"),
                    klass: name.expect("cluster name"),
                    n,
                });
            }
            // Backwards-compat with original two-cluster invocation.
            "--photo" => clusters.push(Cluster {
                dir: PathBuf::from(args.next().unwrap()),
                klass: "photo".into(),
                n: 20,
            }),
            "--screen" => clusters.push(Cluster {
                dir: PathBuf::from(args.next().unwrap()),
                klass: "screen".into(),
                n: usize::MAX,
            }),
            "--photo-n" => {
                let v: usize = args.next().unwrap().parse().unwrap();
                if let Some(c) = clusters.iter_mut().find(|c| c.klass == "photo") {
                    c.n = v;
                }
            }
            "--photo-seed" => photo_seed = args.next().unwrap().parse().unwrap(),
            "--output" => output = PathBuf::from(args.next().unwrap()),
            _ => panic!("unknown arg: {a}"),
        }
    }
    assert!(!clusters.is_empty(), "no --cluster / --photo / --screen");
    (clusters, photo_seed, output)
}

/// Tiny LCG sampler — deterministic, no external rand dep. Same as Step-0 probe.
fn lcg_sample<T: Clone>(items: &[T], n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut idx: Vec<usize> = (0..items.len()).collect();
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

/// HF-coarsen variant: multiply the bottom-right 4x4 quant entries
/// (rows 4-7, cols 4-7 in natural order) by `factor`, clamp to [1, 255].
/// Apply to luma + Cb + Cr. Uses ScalingParams::Exact so our edits aren't
/// rescaled by quality.
fn make_table_bprime(q: u8, factor: f32) -> EncodingTables {
    let mut t = make_table_a(q);
    for row in 4..8usize {
        for col in 4..8usize {
            let idx = row * 8 + col;
            t.quant.c0[idx] = (t.quant.c0[idx] * factor).clamp(1.0, 255.0);
            t.quant.c1[idx] = (t.quant.c1[idx] * factor).clamp(1.0, 255.0);
            t.quant.c2[idx] = (t.quant.c2[idx] * factor).clamp(1.0, 255.0);
        }
    }
    t.scaling = ScalingParams::Exact;
    t
}

fn make_table(name: &str, q: u8) -> EncodingTables {
    match name {
        "A_v4" => {
            let mut t = make_table_a(q);
            // Match B' calls in using Exact scaling so A is the actual
            // unscaled v4 baseline against which B' deltas are computed.
            t.scaling = ScalingParams::Exact;
            t
        }
        "Bp_2x" => make_table_bprime(q, 2.0),
        "Bp_4x" => make_table_bprime(q, 4.0),
        "Bp_8x" => make_table_bprime(q, 8.0),
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
    let (clusters, seed, out) = parse_args();
    eprintln!("per-class signal probe (B' variant — HF quant INCREASED):");

    let mut work: Vec<(PathBuf, String)> = Vec::new();
    for c in &clusters {
        let all = list_pngs(&c.dir);
        let sample = if c.n >= all.len() {
            all.clone()
        } else {
            lcg_sample(&all, c.n, seed)
        };
        eprintln!(
            "  cluster {}: {} files sampled (from {} in {:?}, seed={seed})",
            c.klass,
            sample.len(),
            all.len(),
            c.dir
        );
        work.extend(sample.into_iter().map(|p| (p, c.klass.clone())));
    }

    let mut tsv = fs::File::create(&out).expect("open output");
    writeln!(
        tsv,
        "image\tcontent_class\ttable\tq\tbytes\tbutteraugli\tw\th"
    )
    .unwrap();

    let total_cells = work.len() * TABLES.len() * QS.len();
    let mut done = 0usize;
    for (path, klass) in &work {
        let klass: &str = klass.as_str();
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
