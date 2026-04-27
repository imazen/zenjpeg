//! Tier 1 hot-path microbench. Measures Tier-1-only cost at 1/4/16 MP
//! to identify where the mass lives.
use std::time::Instant;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

fn make_random(n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    let mut s: u32 = 0xdeadbeef;
    for b in buf.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *b = (s >> 16) as u8;
    }
    buf
}

fn bench(label: &str, iters: usize, mut f: impl FnMut()) {
    for _ in 0..3 { f(); }
    let t0 = Instant::now();
    for _ in 0..iters { f(); }
    let dt = t0.elapsed();
    let per = dt.as_secs_f64() * 1e3 / iters as f64;
    println!("  {:55} mean = {:7.3} ms / iter", label, per);
}

fn main() {
    // Tier 1 only: just request one Tier 1 feature so other tiers
    // skip themselves.
    let q_t1 = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
    let q_all = AnalysisQuery::new(FeatureSet::SUPPORTED);

    for (label, w, h) in [
        ("1 MP   1024x1024", 1024u32, 1024u32),
        ("4 MP   2048x2048", 2048u32, 2048u32),
        ("16 MP  4096x4096", 4096u32, 4096u32),
    ] {
        println!("\n=== {} ===", label);
        let buf = make_random((w as usize) * (h as usize) * 3);
        let stride = (w as usize) * 3;
        let mk = || PixelSlice::new(&buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();

        let iters = if w >= 4096 { 10 } else { 30 };

        // Tier 1 only
        bench("Tier 1 only (Variance)", iters, || {
            let _ = zenanalyze::analyze_features(mk(), &q_t1).unwrap();
        });

        // Full pipeline
        bench("FeatureSet::SUPPORTED", iters, || {
            let _ = zenanalyze::analyze_features(mk(), &q_all).unwrap();
        });
    }
}
