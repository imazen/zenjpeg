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
    rs: usize,
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
            rs: cs + if pad > 0 { 2 } else { 0 },
            y: vec![9999.; ys * h],
            cb: vec![8888.; cs * h.div_ceil(vf)],
            cr: vec![-7777.; (cs + if pad > 0 { 2 } else { 0 }) * h.div_ceil(vf)],
        };
        for y in 0..h {
            for x in 0..w {
                p.y[y * ys + x] = ((x * 43 + y * 71) % 256) as f32;
            }
        }
        for y in 0..h.div_ceil(vf) {
            for x in 0..cw {
                p.cb[y * cs + x] = ((x * 19 + y * 31) % 151) as f32 - 71.;
                p.cr[y * p.rs + x] = ((x * 37 + y * 11) % 173) as f32 - 83.;
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
            cr: &self.cr[(row / self.vf) * self.rs..],
            cr_stride: self.rs,
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

#[test]
fn all_strip_residues_strides_and_partitions_match_packed_reference() {
    let mut comparisons = 0;
    for sub in SUBS {
        for w in (1usize..=33).chain([63, 64, 65, 129]) {
            for h in (1usize..=49).chain([65]) {
                let cfg = EncoderConfig::ycbcr(40, sub);
                let expected = Planes::new(w, h, sub, 0).encode(&cfg, &[h]);
                let p = Planes::new(w, h, sub, if (w + h) % 2 == 0 { 0 } else { 3 });
                for parts in [&[1][..], &[3, 1, 17, 2][..], &[7, 16, 5][..]] {
                    assert_eq!(p.encode(&cfg, parts), expected, "{sub:?} {w}x{h} {parts:?}");
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!("complete planar strip residues/strides/partitions: {comparisons} byte comparisons");
}

#[test]
fn quality_and_entropy_modes_preserve_planar_partition_identity() {
    let mut comparisons = 0;
    for sub in SUBS {
        for q in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
            for (progressive, optimized) in [(false, false), (false, true), (true, true)] {
                for (w, h) in [(1, 2), (9, 17), (33, 49)] {
                    let cfg = EncoderConfig::ycbcr(q, sub)
                        .progressive(progressive)
                        .optimize_huffman(optimized);
                    let p = Planes::new(w, h, sub, 3);
                    let expected = p.encode(&cfg, &[h]);
                    for parts in [&[1][..], &[15, 2, 1, 17][..]] {
                        assert_eq!(
                            p.encode(&cfg, parts),
                            expected,
                            "{sub:?} q{q} {progressive}/{optimized} {w}x{h}"
                        );
                        comparisons += 1;
                    }
                }
            }
        }
    }
    eprintln!("planar quality/entropy partition identity: {comparisons} comparisons");
}

fn expanded(p: &Planes, sub: Sub) -> (Vec<f32>, Vec<f32>) {
    let mut cb = vec![6666.; p.ys * p.h];
    let mut cr = cb.clone();
    for y in 0..p.h {
        for x in 0..p.w {
            cb[y * p.ys + x] = p.cb[(y / p.vf) * p.cs + x / sub.h_factor() as usize];
            cr[y * p.ys + x] = p.cr[(y / p.vf) * p.rs + x / sub.h_factor() as usize];
        }
    }
    (cb, cr)
}
fn full_at<'a>(p: &'a Planes, row: usize, cb: &'a [f32], cr: &'a [f32]) -> YCbCrPlanes<'a> {
    YCbCrPlanes {
        y: &p.y[row * p.ys..],
        y_stride: p.ys,
        cb: &cb[row * p.ys..],
        cb_stride: p.ys,
        cr: &cr[row * p.ys..],
        cr_stride: p.ys,
    }
}

#[test]
fn modes_can_alternate_at_strip_boundaries_but_not_inside_a_strip() {
    use zenjpeg::encoder::ErrorKind;
    for sub in SUBS {
        let p = Planes::new(17, 65, sub, 3);
        let (cb, cr) = expanded(&p, sub);
        let cfg = EncoderConfig::ycbcr(40, sub);
        let expected = p.encode(&cfg, &[p.h]);
        let sh = sub.v_factor() as usize * 8;
        for first_sampled in [false, true] {
            let mut enc = cfg
                .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
                .unwrap();
            let mut row = 0;
            while row < p.h {
                let n = sh.min(p.h - row);
                let sampled = first_sampled ^ ((row / sh) % 2 == 1);
                if sampled {
                    enc.push_subsampled(&p.at(row), n, Unstoppable).unwrap();
                } else {
                    enc.push(&full_at(&p, row, &cb, &cr), n, Unstoppable)
                        .unwrap();
                }
                row += n;
            }
            assert_eq!(enc.finish().unwrap(), expected, "mixed {sub:?}");
            let mut enc = cfg
                .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
                .unwrap();
            if first_sampled {
                enc.push_subsampled(&p.at(0), 1, Unstoppable).unwrap();
            } else {
                enc.push(&full_at(&p, 0, &cb, &cr), 1, Unstoppable).unwrap();
            }
            let err = if first_sampled {
                enc.push(&full_at(&p, 1, &cb, &cr), 1, Unstoppable)
            } else {
                enc.push_subsampled(&p.at(1), 1, Unstoppable)
            }
            .unwrap_err();
            assert!(matches!(err.kind(), ErrorKind::InvalidConfig(_)), "{err:?}");
            assert_eq!(enc.rows_pushed(), 1);
            if first_sampled {
                enc.push_subsampled(&p.at(1), p.h - 1, Unstoppable).unwrap();
            } else {
                enc.push(&full_at(&p, 1, &cb, &cr), p.h - 1, Unstoppable)
                    .unwrap();
            }
            assert_eq!(enc.finish().unwrap(), expected);
        }
    }
}

#[test]
fn shared_chroma_rows_must_agree_and_rejection_preserves_buffered_input() {
    use zenjpeg::encoder::ErrorKind;
    for sub in [Sub::Quarter, Sub::HalfVertical] {
        for channel in [0, 1] {
            let p = Planes::new(17, 33, sub, 3);
            let cfg = EncoderConfig::ycbcr(40, sub);
            let mut enc = cfg
                .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
                .unwrap();
            enc.push_subsampled(&p.at(0), 1, Unstoppable).unwrap();
            let mut bad_cb = p.cb.clone();
            let mut bad_cr = p.cr.clone();
            if channel == 0 {
                bad_cb[0] += 1.;
            } else {
                bad_cr[0] += 1.;
            }
            let bad = YCbCrPlanes {
                cb: &bad_cb,
                cr: &bad_cr,
                ..p.at(1)
            };
            assert!(matches!(
                enc.push_subsampled(&bad, 2, Unstoppable)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidConfig(_)
            ));
            assert_eq!(enc.rows_pushed(), 1);
            // An odd-start two-row push needs TWO chroma rows, not ceil(2/2).
            let bad = YCbCrPlanes {
                cb: &p.cb[..p.cw],
                ..p.at(1)
            };
            assert!(matches!(
                enc.push_subsampled(&bad, 2, Unstoppable)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidBufferSize { .. }
            ));
            assert_eq!(enc.rows_pushed(), 1);
            enc.push_subsampled(&p.at(1), p.h - 1, Unstoppable).unwrap();
            assert_eq!(enc.finish().unwrap(), p.encode(&cfg, &[p.h]));
        }
    }
}

#[test]
fn invalid_layouts_and_row_counts_do_not_consume_input() {
    use zenjpeg::encoder::ErrorKind;
    for sub in SUBS {
        let p = Planes::new(17, 33, sub, 3);
        let cfg = EncoderConfig::ycbcr(40, sub);
        let (full_cb, full_cr) = expanded(&p, sub);
        for sampled in [false, true] {
            for channel in 0..3 {
                for failure in 0..3 {
                    let mut enc = cfg
                        .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
                        .unwrap();
                    let mut bad = if sampled {
                        p.at(0)
                    } else {
                        full_at(&p, 0, &full_cb, &full_cr)
                    };
                    let rows: usize = 3;
                    let crows = if sampled { rows.div_ceil(p.vf) } else { rows };
                    let cw = if sampled { p.cw } else { p.w };
                    let (data, stride, width, height) = match channel {
                        0 => (&mut bad.y, &mut bad.y_stride, p.w, rows),
                        1 => (&mut bad.cb, &mut bad.cb_stride, cw, crows),
                        _ => (&mut bad.cr, &mut bad.cr_stride, cw, crows),
                    };
                    match failure {
                        0 => *stride = width - 1,
                        1 => *data = &data[..(height - 1) * *stride + width - 1],
                        _ => *stride = usize::MAX,
                    }
                    let err = if sampled {
                        enc.push_subsampled(&bad, rows, Unstoppable)
                    } else {
                        enc.push(&bad, rows, Unstoppable)
                    }
                    .unwrap_err();
                    assert!(
                        match failure {
                            0 => matches!(err.kind(), ErrorKind::StrideTooSmall { .. }),
                            1 => matches!(err.kind(), ErrorKind::InvalidBufferSize { .. }),
                            _ => matches!(err.kind(), ErrorKind::InvalidConfig(_)),
                        },
                        "{sub:?} channel{channel} case{failure}: {err:?}"
                    );
                    assert_eq!(enc.rows_pushed(), 0);
                    let too_many = if sampled {
                        enc.push_subsampled(&p.at(0), usize::MAX, Unstoppable)
                    } else {
                        enc.push(&full_at(&p, 0, &full_cb, &full_cr), usize::MAX, Unstoppable)
                    }
                    .unwrap_err();
                    assert!(matches!(too_many.kind(), ErrorKind::TooManyRows { .. }));
                    if sampled {
                        enc.push_subsampled(&p.at(0), p.h, Unstoppable).unwrap();
                    } else {
                        enc.push(&full_at(&p, 0, &full_cb, &full_cr), p.h, Unstoppable)
                            .unwrap();
                    }
                    assert_eq!(enc.finish().unwrap(), p.encode(&cfg, &[p.h]));
                }
            }
        }
    }
}

#[test]
fn zero_rows_finish_variants_and_tight_last_rows() {
    use zenjpeg::encoder::ErrorKind;
    for sub in SUBS {
        let mut p = Planes::new(17, 33, sub, 3);
        let cfg = EncoderConfig::ycbcr(40, sub);
        p.y.truncate((p.h - 1) * p.ys + p.w);
        p.cb.truncate((p.h.div_ceil(p.vf) - 1) * p.cs + p.cw);
        p.cr.truncate((p.h.div_ceil(p.vf) - 1) * p.rs + p.cw);
        let expected = p.encode(&cfg, &[1]);
        let empty = YCbCrPlanes {
            y: &[],
            y_stride: 0,
            cb: &[],
            cb_stride: 0,
            cr: &[],
            cr_stride: 0,
        };
        for finish_mode in 0..3 {
            let mut enc = cfg
                .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
                .unwrap();
            enc.push_subsampled(&empty, 0, Unstoppable).unwrap();
            enc.push_subsampled(&p.at(0), 1, Unstoppable).unwrap();
            enc.push(&empty, 0, Unstoppable).unwrap();
            enc.push_subsampled(&empty, 0, Unstoppable).unwrap();
            enc.push_subsampled(&p.at(1), p.h - 1, Unstoppable).unwrap();
            enc.push_subsampled(&empty, 0, Unstoppable).unwrap();
            let actual = match finish_mode {
                0 => enc.finish().unwrap(),
                1 => {
                    let mut out = vec![0x5a; 7];
                    enc.finish_into(&mut out).unwrap();
                    out
                }
                _ => enc.finish_to(Vec::new()).unwrap(),
            };
            assert_eq!(actual, expected);
        }
        let mut enc = cfg
            .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
            .unwrap();
        enc.push_subsampled(&p.at(0), 1, Unstoppable).unwrap();
        assert!(matches!(
            enc.finish().unwrap_err().kind(),
            ErrorKind::IncompleteImage { pushed: 1, .. }
        ));
    }
}

#[derive(Clone)]
struct Deadline {
    polls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    fail_at: usize,
}
impl enough::Stop for Deadline {
    fn check(&self) -> Result<(), enough::StopReason> {
        if self
            .polls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            >= self.fail_at
        {
            Err(enough::StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}
#[test]
fn every_stop_poll_keeps_reason_and_resumable_buffer_state() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zenjpeg::encoder::ErrorKind;
    let mut checks = 0;
    for sub in SUBS {
        for pad in [0, 3] {
            let p = Planes::new(17, 65, sub, pad);
            let cfg = EncoderConfig::ycbcr(40, sub);
            let (cb, cr) = expanded(&p, sub);
            let expected = p.encode(&cfg, &[p.h]);
            for sampled in [false, true] {
                let polls = Arc::new(AtomicUsize::new(0));
                let mut enc = cfg
                    .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
                    .unwrap();
                let first = if sampled {
                    p.at(0)
                } else {
                    full_at(&p, 0, &cb, &cr)
                };
                if sampled {
                    enc.push_subsampled(&first, 1, Unstoppable).unwrap();
                } else {
                    enc.push(&first, 1, Unstoppable).unwrap();
                }
                let rest = if sampled {
                    p.at(1)
                } else {
                    full_at(&p, 1, &cb, &cr)
                };
                let stop = Deadline {
                    polls: polls.clone(),
                    fail_at: usize::MAX,
                };
                if sampled {
                    enc.push_subsampled(&rest, p.h - 1, stop).unwrap();
                } else {
                    enc.push(&rest, p.h - 1, stop).unwrap();
                }
                let total = polls.load(Ordering::Relaxed);
                assert!(total > 0);
                assert_eq!(enc.finish().unwrap(), expected);
                for fail_at in 0..total {
                    polls.store(0, Ordering::Relaxed);
                    let mut enc = cfg
                        .encode_from_ycbcr_planar(p.w as u32, p.h as u32)
                        .unwrap();
                    if sampled {
                        enc.push_subsampled(&first, 1, Unstoppable).unwrap();
                    } else {
                        enc.push(&first, 1, Unstoppable).unwrap();
                    }
                    let stop = Deadline {
                        polls: polls.clone(),
                        fail_at,
                    };
                    let err = if sampled {
                        enc.push_subsampled(&rest, p.h - 1, stop)
                    } else {
                        enc.push(&rest, p.h - 1, stop)
                    }
                    .unwrap_err();
                    assert!(
                        matches!(
                            err.kind(),
                            ErrorKind::Cancelled(enough::StopReason::TimedOut)
                        ),
                        "{err:?}"
                    );
                    let row = enc.rows_pushed() as usize;
                    let rest = if sampled {
                        p.at(row)
                    } else {
                        full_at(&p, row, &cb, &cr)
                    };
                    if sampled {
                        enc.push_subsampled(&rest, p.h - row, Unstoppable).unwrap();
                    } else {
                        enc.push(&rest, p.h - row, Unstoppable).unwrap();
                    }
                    assert_eq!(
                        enc.finish().unwrap(),
                        expected,
                        "{sub:?} sampled{sampled} poll{fail_at}"
                    );
                    checks += 1;
                }
            }
        }
    }
    eprintln!("planar stop/resume: {checks} injected deadline checkpoints");
}
