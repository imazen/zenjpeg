// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Corpus9 instrument census for `target_quality::search_target`
//! (registration: benchmarks/zensim_instrument_census_2026-08-27.md).
//! Arms: A = anchor_guess (shipped default), B = the shipped-but-inert
//! `zq_seed` head (in-binary zenanalyze features; None ⇒ anchor fallback).
//! Judge = the pinned zensim `ZensimProfile::latest()` on decoded pixels —
//! the same calls `zq_calibrate` uses.
//!
//!   cargo run --release -p zenjpeg --features target-quality --example zensim_census -- \
//!     <corpus9.tsv> <targets-csv> <max_encodes> <arm:A|B> <out.tsv>

use std::io::Write;

use enough::Unstoppable;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenjpeg::decode::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};
use zenjpeg::target_quality::{TargetOptions, search_target};
use zenjpeg::zq_seed::{ZQ_FEATURES, predict_q0_from_features};
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

fn load_png(path: &str) -> (Vec<u8>, u32, u32) {
    let img = zenjpeg_bench_utils::load_png(std::path::Path::new(path))
        .unwrap_or_else(|e| panic!("load {path}: {e}"));
    let (buf, w, h) = img.into_contiguous_buf();
    let bytes: Vec<u8> = buf.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    (bytes, w as u32, h as u32)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (corpus, targets_csv, k, arm, out) = (&a[1], &a[2], &a[3], &a[4], &a[5]);
    let max_encodes: u8 = k.parse().expect("max_encodes");
    let targets: Vec<f64> = targets_csv.split(',').map(|t| t.parse().unwrap()).collect();
    let z = Zensim::new(ZensimProfile::latest());
    let mut tsv = std::fs::File::create(out).expect("out");
    writeln!(
        tsv,
        "image\tclass\ttarget\tarm\tseed_q\tencodes_used\tachieved\tabs_err\tbytes\tencode_ms"
    )
    .unwrap();
    for line in std::fs::read_to_string(corpus).expect("corpus").lines() {
        let mut f = line.split('\t');
        let (path, name, class) = (
            f.next().unwrap(),
            f.next().unwrap(),
            f.next().unwrap_or("image"),
        );
        let (rgb, w, h) = load_png(path);
        let chunks: &[[u8; 3]] = rgb.as_chunks::<3>().0;
        let src = RgbSlice::new(chunks, w as usize, h as usize);
        let pre = z.precompute_reference(&src).expect("precompute");
        // arm B: in-binary features once per image
        let q0_for = |t: f64| -> Option<f32> {
            if arm != "B" {
                return None;
            }
            let mut set = FeatureSet::just(ZQ_FEATURES[0]);
            for f in &ZQ_FEATURES[1..] {
                set = set.with(*f);
            }
            let an = analyze_features_rgb8(&rgb, w, h, &AnalysisQuery::new(set));
            let mut fv = [0.0f32; 6];
            for (i, feat) in ZQ_FEATURES.iter().enumerate() {
                fv[i] = an.get_f32(*feat).or_else(|| {
                    an.get(*feat).and_then(|v| match v {
                        zenanalyze::feature::FeatureValue::U32(x) => Some(x as f32),
                        zenanalyze::feature::FeatureValue::F32(x) => Some(x),
                        _ => None,
                    })
                })?;
            }
            predict_q0_from_features(&fv, t, u64::from(w) * u64::from(h))
        };
        for &t in &targets {
            let seed = q0_for(t);
            let opts = TargetOptions {
                max_encodes,
                q_start: seed,
                tolerance: 0.0,
                ..Default::default()
            };
            let mut encodes = 0u32;
            let mut last_bytes = 0usize;
            let t0 = std::time::Instant::now();
            let res = search_target(t, &opts, |q| -> Result<f64, String> {
                encodes += 1;
                let cfg = EncoderConfig::ycbcr(
                    Quality::ApproxJpegli(q),
                    ChromaSubsampling::Quarter,
                );
                let jpeg = cfg
                    .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                    .map_err(|e| format!("encode: {e:?}"))?;
                last_bytes = jpeg.len();
                let dec = Decoder::new()
                    .decode(&jpeg, Unstoppable)
                    .map_err(|e| format!("decode: {e:?}"))?
                    .into_pixels_u8()
                    .ok_or_else(|| "pixels: not u8".to_string())?;
                let dc: &[[u8; 3]] = dec.as_chunks::<3>().0;
                let ds = RgbSlice::new(dc, w as usize, h as usize);
                let r = z
                    .compute_with_ref_and_diffmap(&pre, &ds, DiffmapWeighting::Trained)
                    .map_err(|e| format!("zensim: {e:?}"))?;
                Ok(r.score())
            })
            .expect("search");
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            writeln!(
                tsv,
                "{name}\t{class}\t{t:.0}\t{arm}\t{}\t{encodes}\t{:.3}\t{:.3}\t{last_bytes}\t{ms:.1}",
                seed.map_or("anchor".into(), |q| format!("{q:.1}")),
                res.score,
                (res.score - t).abs(),
            )
            .unwrap();
            eprintln!("{name} t{t:.0} {arm}: encodes={encodes} achieved={:.2}", res.score);
        }
    }
}
