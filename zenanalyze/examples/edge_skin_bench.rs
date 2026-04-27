//! Microbench: comparing zenanalyze's `RowStream` row-fetch paths
//! (`Native`, `StripAlpha8`, `Convert`) against direct
//! `zenpixels-convert::RowConverter` use, on the same Tier-1
//! analyzer workload.
//!
//! Run: `cargo run --release -p zenanalyze --features experimental
//! --example edge_skin_bench`.

use std::time::Instant;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

fn make_random(n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    let mut state: u32 = 0xdeadbeef;
    for b in buf.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *b = (state >> 16) as u8;
    }
    buf
}

fn bench(label: &str, mut f: impl FnMut()) {
    for _ in 0..3 {
        f();
    }
    let n = 30;
    let t0 = Instant::now();
    for _ in 0..n {
        f();
    }
    let dt = t0.elapsed();
    let per = dt.as_secs_f64() * 1e3 / n as f64;
    println!("  {:50} mean = {:7.3} ms / iter", label, per);
}

fn main() {
    println!(
        "RowStream path microbench, 4 MP image, AVX2 dispatch, no target-cpu=native\n\
         Workload: `analyze_features` with `FeatureSet::SUPPORTED`\n"
    );
    let (w, h) = (2048u32, 2048u32);

    println!("## RGB8 input (RowStream::Native — zero-copy slice subindex)\n");
    {
        let buf = make_random((w as usize) * (h as usize) * 3);
        let stride = (w as usize) * 3;
        let mk =
            || PixelSlice::new(&buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
        bench("RGB8_SRGB → Native row fetch", || {
            let _ = zenanalyze::analyze_features(mk(), &q).unwrap();
        });
    }

    println!("\n## RGBA8 input (RowStream::StripAlpha8 — tight scalar strip)\n");
    {
        let buf = make_random((w as usize) * (h as usize) * 4);
        let stride = (w as usize) * 4;
        let mk =
            || PixelSlice::new(&buf, w, h, stride, PixelDescriptor::RGBA8_SRGB).unwrap();
        let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
        bench("RGBA8_SRGB → StripAlpha8", || {
            let _ = zenanalyze::analyze_features(mk(), &q).unwrap();
        });
    }

    println!("\n## BGRA8 input (RowStream::StripAlpha8 — strip + channel swap)\n");
    {
        let buf = make_random((w as usize) * (h as usize) * 4);
        let stride = (w as usize) * 4;
        let mk =
            || PixelSlice::new(&buf, w, h, stride, PixelDescriptor::BGRA8_SRGB).unwrap();
        let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
        bench("BGRA8_SRGB → StripAlpha8", || {
            let _ = zenanalyze::analyze_features(mk(), &q).unwrap();
        });
    }

    println!(
        "\n## RGB16 input (RowStream::Convert — zenpixels-convert RowConverter)\n"
    );
    {
        let buf = make_random((w as usize) * (h as usize) * 6);
        let stride = (w as usize) * 6;
        let mk =
            || PixelSlice::new(&buf, w, h, stride, PixelDescriptor::RGB16_SRGB).unwrap();
        let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
        bench("RGB16_SRGB → Convert (zenpixels-convert)", || {
            let _ = zenanalyze::analyze_features(mk(), &q).unwrap();
        });
    }

    println!(
        "\n## RGBA16 input (RowStream::Convert — zenpixels-convert handles strip + narrowing)\n"
    );
    {
        let buf = make_random((w as usize) * (h as usize) * 8);
        let stride = (w as usize) * 8;
        let mk =
            || PixelSlice::new(&buf, w, h, stride, PixelDescriptor::RGBA16_SRGB).unwrap();
        let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
        bench("RGBA16_SRGB → Convert", || {
            let _ = zenanalyze::analyze_features(mk(), &q).unwrap();
        });
    }

    println!(
        "\n## Row-level microbench: zenanalyze scalar strip vs garb SIMD strip\n\
         (One row, 2048 px wide, RGBA8 → RGB8.)\n"
    );
    {
        let row_bytes_in = (w as usize) * 4;
        let row_bytes_out = (w as usize) * 3;
        let src = make_random(row_bytes_in);
        let mut dst = vec![0u8; row_bytes_out];
        bench_row("zenanalyze strip_alpha_row (scalar, plain loop)", || {
            // mirror the strip_alpha_row scalar pattern
            let r = 0usize;
            let g = 1usize;
            let b = 2usize;
            for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
                d[0] = s[r];
                d[1] = s[g];
                d[2] = s[b];
            }
        });
        bench_row("garb::bytes::rgba_to_rgb (incant! SIMD dispatch)", || {
            garb::bytes::rgba_to_rgb(&src, &mut dst).unwrap();
        });
        bench_row("garb::bytes::bgra_to_rgb (incant! SIMD dispatch)", || {
            garb::bytes::bgra_to_rgb(&src, &mut dst).unwrap();
        });
    }
}

fn bench_row(label: &str, mut f: impl FnMut()) {
    for _ in 0..1000 {
        f();
    }
    let n = 100_000usize;
    let t0 = Instant::now();
    for _ in 0..n {
        f();
    }
    let dt = t0.elapsed();
    let per_us = dt.as_secs_f64() * 1e6 / n as f64;
    println!("  {:55} mean = {:7.3} µs / row", label, per_us);
}
