//! Wall-clock timing for the lossless transform/restructure path.
//!
//! Used to verify the #194/#195 geometry fixes introduced no perf regression.
//! Prints median of N iterations for transform(Rotate90) and progressive
//! restructure on a 2000x1333 4:2:0 image.

use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestartInterval, RestructureConfig,
    TransformConfig, restructure, transform,
};

fn gen_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut state: u32 = 0x2468_ACE1;
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h * 3) {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        px.push((state >> 24) as u8);
    }
    px
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let (w, h) = (2000u32, 1333u32);
    let mut enc = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&gen_rgb(w, h), Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();

    let n = 20;

    let mut t_transform = Vec::with_capacity(n);
    let mut out_len = 0usize;
    for _ in 0..n {
        let t0 = Instant::now();
        let out = transform(
            &jpeg,
            &TransformConfig {
                transform: LosslessTransform::Rotate90,
                edge_handling: EdgeHandling::TrimPartialBlocks,
            },
            Unstoppable,
        )
        .unwrap();
        t_transform.push(t0.elapsed().as_secs_f64() * 1000.0);
        out_len = out.len();
    }
    println!(
        "transform(Rotate90) 2000x1333 4:2:0: median {:.2} ms over {n} iters ({out_len} bytes)",
        median_ms(t_transform)
    );

    let mut t_prog = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        let out = restructure(
            &jpeg,
            &RestructureConfig {
                output_mode: OutputMode::Progressive,
                restart_interval: RestartInterval::None,
                transform: None,
            },
            Unstoppable,
        )
        .unwrap();
        t_prog.push(t0.elapsed().as_secs_f64() * 1000.0);
        out_len = out.len();
    }
    println!(
        "restructure(Progressive) 2000x1333 4:2:0: median {:.2} ms over {n} iters ({out_len} bytes)",
        median_ms(t_prog)
    );

    let mut t_seq = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        let out = restructure(
            &jpeg,
            &RestructureConfig {
                output_mode: OutputMode::Sequential,
                restart_interval: RestartInterval::None,
                transform: None,
            },
            Unstoppable,
        )
        .unwrap();
        t_seq.push(t0.elapsed().as_secs_f64() * 1000.0);
        out_len = out.len();
    }
    println!(
        "restructure(Sequential) 2000x1333 4:2:0: median {:.2} ms over {n} iters ({out_len} bytes)",
        median_ms(t_seq)
    );
}
