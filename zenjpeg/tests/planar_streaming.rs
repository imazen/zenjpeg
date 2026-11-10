//! Pre-sampled planar pushes must retain sampling phase across every row split.
use zenjpeg::encoder::{ChromaSubsampling as Sub, EncoderConfig, Unstoppable, YCbCrPlanes};

const SUBS: [Sub; 4] = [
    Sub::None,
    Sub::HalfHorizontal,
    Sub::Quarter,
    Sub::HalfVertical,
];

struct Planes {
    w: usize,
    h: usize,
    cw: usize,
    vf: usize,
    ys: usize,
    cs: usize,
    y: Vec<f32>,
    cb: Vec<f32>,
    cr: Vec<f32>,
}
impl Planes {
    fn new(w: usize, h: usize, sub: Sub, pad: usize) -> Self {
        let cw = w.div_ceil(sub.h_factor() as usize);
        let vf = sub.v_factor() as usize;
        let (ys, cs) = (w + pad, cw + pad + usize::from(pad > 0));
        let mut p = Self {
            w,
            h,
            cw,
            vf,
            ys,
            cs,
            y: vec![9999.; ys * h],
            cb: vec![8888.; cs * h.div_ceil(vf)],
            cr: vec![-7777.; cs * h.div_ceil(vf)],
        };
        for y in 0..h {
            for x in 0..w {
                p.y[y * ys + x] = ((x * 43 + y * 71) % 256) as f32;
            }
        }
        for y in 0..h.div_ceil(vf) {
            for x in 0..cw {
                p.cb[y * cs + x] = ((x * 19 + y * 31) % 151) as f32 - 71.;
                p.cr[y * cs + x] = ((x * 37 + y * 11) % 173) as f32 - 83.;
            }
        }
        p
    }
    fn at(&self, row: usize) -> YCbCrPlanes<'_> {
        YCbCrPlanes {
            y: &self.y[row * self.ys..],
            y_stride: self.ys,
            cb: &self.cb[(row / self.vf) * self.cs..],
            cb_stride: self.cs,
            cr: &self.cr[(row / self.vf) * self.cs..],
            cr_stride: self.cs,
        }
    }
    fn encode(&self, cfg: &EncoderConfig, parts: &[usize]) -> Vec<u8> {
        let mut enc = cfg
            .encode_from_ycbcr_planar(self.w as u32, self.h as u32)
            .unwrap();
        let mut row = 0;
        let mut part = 0;
        while row < self.h {
            let n = parts[part % parts.len()].min(self.h - row);
            enc.push_subsampled(&self.at(row), n, Unstoppable)
                .unwrap_or_else(|e| panic!("{}x{} row{row} +{n}: {e}", self.w, self.h));
            row += n;
            part += 1;
            assert_eq!(enc.rows_pushed(), row as u32);
            assert_eq!(enc.rows_remaining(), (self.h - row) as u32);
        }
        enc.finish().unwrap()
    }
}

#[test]
fn presampled_single_row_pushes_match_one_push() {
    for sub in SUBS {
        let p = Planes::new(17, 33, sub, 3);
        let cfg = EncoderConfig::ycbcr(40, sub);
        assert_eq!(p.encode(&cfg, &[1]), p.encode(&cfg, &[p.h]), "{sub:?}");
    }
}

#[test]
fn presampled_odd_boundary_splits_match_one_push() {
    for sub in SUBS {
        let p = Planes::new(33, 49, sub, 0);
        let cfg = EncoderConfig::ycbcr(40, sub);
        assert_eq!(
            p.encode(&cfg, &[3, 1, 17, 2]),
            p.encode(&cfg, &[p.h]),
            "{sub:?}"
        );
    }
}
