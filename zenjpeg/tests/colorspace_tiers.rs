//! SIMD-tier checks for the colour paths (`tests/colorspace_paths.rs` holds
//! the path checks).
//!
//! `archmage::testing::for_each_token_permutation` disables SIMD tokens for
//! the whole process while it runs, so any test running alongside it in the
//! same binary would silently decode on a different tier. Tests that use it
//! live here, away from tests that compare tiers implicitly (every
//! `colorspace_paths` test compares paths that dispatch independently).

#[path = "support/jpeg_writer.rs"]
#[allow(dead_code)] // shared with colorspace_paths and colorspace_oracles
mod jpeg_writer;

use imgref::ImgRefMut;
use jpeg_writer::{Comp, Segment, Spec, jfif};
use zenjpeg::decode::{ChromaUpsampling, Decoder, OutputTarget};

const UPS: [ChromaUpsampling; 2] = [
    ChromaUpsampling::Triangle,
    ChromaUpsampling::NearestNeighbor,
];

fn decoder(ups: ChromaUpsampling, threads: usize) -> Decoder {
    Decoder::new().chroma_upsampling(ups).num_threads(threads)
}

#[derive(Clone, Copy)]
enum Read {
    Rgb8,
    Bgr8,
    Rgbx8,
    Bgrx8,
}

impl Read {
    fn bpp(self) -> usize {
        match self {
            Read::Rgb8 | Read::Bgr8 => 3,
            Read::Rgbx8 | Read::Bgrx8 => 4,
        }
    }
}

/// Packed 3- or 4-byte pixels as RGB, swapping B and R for BGR orders.
fn to_rgb(px: &[u8], bpp: usize, bgr: bool) -> Vec<u8> {
    px.chunks_exact(bpp)
        .flat_map(|p| {
            if bgr {
                [p[2], p[1], p[0]]
            } else {
                [p[0], p[1], p[2]]
            }
        })
        .collect()
}

/// C1-18: chroma far outside [0, 255] (crafted coefficients) reaches the
/// row kernels through the unclamped IDCT of f32 output targets. The AVX2 and
/// AVX-512 kernels centred chroma in i16 (wrapping for Cb near -32768) and the
/// default constants overflowed i32 past |Cb - 128| = 18,492, so 8-bit reads
/// depended on the SIMD tier and disagreed with the f32 read. The unclamped
/// IDCT now bounds samples to `+-SAMPLE_LIMIT` (2047). DC-only frames with
/// Cb = -32697 and Cr = 24503 (quantizer 200, after reviewer 2's fixtures),
/// 4:4:4 at 17x8 (a scalar remainder after one 16-pixel block) and 4:2:0 at
/// 34x16 (the upsamplers): on every tier, both IDCT methods and both
/// upsamplers, `rgb8`/`bgr8`/`rgbx8`/`bgrx8` agree with each other and with
/// the first tier, and every channel agrees with `read_rows_rgba_f32`
/// (saturated channels exactly, others within 1 after the sRGB transfer).
#[test]
fn extreme_chroma_reads_agree_on_every_tier() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    use zenjpeg::decode::IdctMethod;
    let fixtures: Vec<(String, Vec<u8>)> = [
        ("cb -32697", [0i16, -1313, 0]),
        ("cr +24503", [0i16, 0, 975]),
        ("both", [0i16, -1313, 975]),
    ]
    .iter()
    .flat_map(|&(name, dcs)| {
        [((17u32, 8u32), (1u8, 1u8)), ((34, 16), (2, 2))].map(move |((w, h), (sh, sv))| {
            let block = move |c: usize, _bx: usize, _by: usize| {
                let mut b = [0i16; 64];
                b[0] = dcs[c];
                b
            };
            let spec = Spec {
                width: w,
                height: h,
                comps: vec![
                    Comp {
                        id: 1,
                        h: sh,
                        v: sv,
                    },
                    Comp { id: 2, h: 1, v: 1 },
                    Comp { id: 3, h: 1, v: 1 },
                ],
                block: &block,
                restart_interval: 0,
                quant: 200,
                segments: vec![Segment::PreFrame(jfif())],
            };
            (format!("{name} {w}x{h}"), jpeg_writer::write(&spec))
        })
    })
    .collect();
    let reads =
        |jpeg: &[u8], method: IdctMethod, ups: ChromaUpsampling| -> (Vec<Vec<u8>>, Vec<f32>) {
            let dec = decoder(ups, 1)
                .idct_method(method)
                .output_target(OutputTarget::SrgbF32);
            let mut out = Vec::new();
            for rd in [Read::Rgb8, Read::Bgr8, Read::Rgbx8, Read::Bgrx8] {
                let mut r = dec.scanline_reader(jpeg).expect("reader");
                let (w, h) = (r.width() as usize, r.height() as usize);
                let stride = w * rd.bpp();
                let mut buf = vec![0u8; stride * h];
                let mut row = 0;
                while row < h {
                    let img = ImgRefMut::new(&mut buf[row * stride..], stride, h - row);
                    row += match rd {
                        Read::Rgb8 => r.read_rows_rgb8(img),
                        Read::Bgr8 => r.read_rows_bgr8(img),
                        Read::Rgbx8 => r.read_rows_rgbx8(img),
                        Read::Bgrx8 => r.read_rows_bgrx8(img),
                    }
                    .expect("read");
                }
                out.push(to_rgb(
                    &buf,
                    rd.bpp(),
                    matches!(rd, Read::Bgr8 | Read::Bgrx8),
                ));
            }
            let mut r = dec.scanline_reader(jpeg).expect("reader");
            let (w, h) = (r.width() as usize, r.height() as usize);
            let mut f = vec![0f32; w * h * 4];
            let mut row = 0;
            while row < h {
                row += r
                    .read_rows_rgba_f32(ImgRefMut::new(&mut f[row * w * 4..], w * 4, h - row))
                    .expect("rgba_f32");
            }
            (out, f)
        };
    let first = std::sync::Mutex::new(None::<Vec<Vec<u8>>>);
    let failures = std::sync::Mutex::new(Vec::<String>::new());
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut all = Vec::new();
        for (name, jpeg) in &fixtures {
            for method in [IdctMethod::Libjpeg, IdctMethod::Jpegli] {
                for ups in UPS {
                    let label = format!("{name} {method:?} {ups:?} at {perm}");
                    let (u8s, f) = reads(jpeg, method, ups);
                    for (i, other) in u8s.iter().enumerate().skip(1) {
                        if *other != u8s[0] {
                            failures
                                .lock()
                                .unwrap()
                                .push(format!("{label}: read {i} differs from rgb8"));
                        }
                    }
                    for (px, rgb) in u8s[0].as_chunks::<3>().0.iter().enumerate() {
                        for c in 0..3 {
                            let lin = f[px * 4 + c];
                            let enc = linear_srgb::default::linear_to_srgb(lin) * 255.0;
                            let ok = match rgb[c] {
                                0 => lin == 0.0,
                                255 => lin == 1.0,
                                v => (enc - v as f32).abs() <= 1.0,
                            };
                            if !ok {
                                failures.lock().unwrap().push(format!(
                                    "{label}: pixel {px} channel {c}: u8 {} vs rgba_f32 {lin} ({enc:.2})",
                                    rgb[c]
                                ));
                            }
                        }
                    }
                    all.push(u8s[0].clone());
                }
            }
        }
        let mut first = first.lock().unwrap();
        match first.as_ref() {
            None => *first = Some(all),
            Some(want) => {
                if *want != all {
                    failures
                        .lock()
                        .unwrap()
                        .push(format!("u8 reads at {perm} differ from the first tier"));
                }
            }
        }
    });
    eprintln!("extreme chroma: {report}");
    let failures = failures.into_inner().unwrap();
    let mut seen = std::collections::BTreeSet::new();
    let failures: Vec<String> = failures
        .into_iter()
        .filter(|m| seen.insert(m.split(':').next().unwrap_or("").to_string()))
        .collect();
    assert!(
        failures.is_empty(),
        "{} failure(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
