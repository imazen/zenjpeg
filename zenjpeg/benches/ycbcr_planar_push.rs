//! `YCbCrPlanarEncoder::push_subsampled`, tightly packed against strided planes.
//!
//! Run: `cargo bench -p zenjpeg --bench ycbcr_planar_push`

use zenbench::prelude::*;
use zenjpeg::encode::encoder_types::ChromaSubsampling;
use zenjpeg::encode::{EncoderConfig, YCbCrPlanes};

fn plane(w: usize, h: usize, stride: usize, offset: f32) -> Vec<f32> {
    let mut out = vec![1.0e6f32; (h - 1) * stride + w];
    for y in 0..h {
        for x in 0..w {
            out[y * stride + x] = ((x * 3 + y * 5 + (x * y) / 7) % 251) as f32 - offset;
        }
    }
    out
}

fn bench(suite: &mut Suite) {
    for (w, h) in [(64usize, 64usize), (1024, 1024)] {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        suite.compare(format!("push_subsampled_420_{w}x{h}"), |g| {
            g.throughput(Throughput::Bytes((w * h * 3 / 2 * 4) as u64));
            for (lane, pad) in [("tight", 0usize), ("strided", 13)] {
                let (ys, cs) = (w + pad, cw + pad);
                let y: &'static [f32] = Box::leak(plane(w, h, ys, 0.0).into_boxed_slice());
                let c: &'static [f32] = Box::leak(plane(cw, ch, cs, 128.0).into_boxed_slice());
                g.bench(lane, move |b| {
                    b.iter(|| {
                        let planes = YCbCrPlanes {
                            y,
                            y_stride: ys,
                            cb: c,
                            cb_stride: cs,
                            cr: c,
                            cr_stride: cs,
                        };
                        let mut enc = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                            .encode_from_ycbcr_planar(w as u32, h as u32)
                            .unwrap();
                        enc.push_subsampled(&planes, h, enough::Unstoppable)
                            .unwrap();
                        enc.finish().unwrap()
                    })
                });
            }
        });
    }
}

zenbench::main!(bench);
