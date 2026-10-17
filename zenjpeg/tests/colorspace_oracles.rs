//! Independent-decoder cross-checks for the colour-model decision and for
//! RGB-as-RGB pixels. Opt-in: `just colorspace-oracles` (feature
//! `__oracle-tests`). When the feature is on, the oracles are required — a
//! missing tool fails the test.
//!
//! | oracle | tool | colour rule | upsampling compared |
//! |---|---|---|---|
//! | libjpeg-turbo | Pillow (`python3`) | jpegli's rule, decided at the first SOS | triangle (fancy), exact |
//! | IJG libjpeg 9 | `djpeg -nosmooth` | component IDs first, then JFIF, then APP14 | nearest (replication), exact |
//! | jpegli | `djpegli` (path in `ZENJPEG_DJPEGLI`, required when set) | `ProcessSOF`, decided at SOF | triangle, within 1 of zenjpeg's `dequant_bias(true)` decode |
//!
//! Each decoder's rule is modelled below, so a fixture where a decoder
//! interprets the colour model differently from zenjpeg is asserted to be
//! interpreted that decoder's documented way — a divergence is a checked fact,
//! not a skipped case. zenjpeg follows jpegli (`ColorModel::decide`).
//!
//! Interpretation is classified by comparing the oracle's pixels with
//! zenjpeg's decode of two relabelled twins of the same planes: one signalled
//! unambiguously RGB, one unambiguously YCbCr. zenjpeg's own decode of each
//! fixture must equal, exactly and on both upsamplers, its decode of the twin
//! jpegli's rule picks, so the twins it is judged by are the ones it chose.
//!
//! Every fixture has odd width and height. jpegli's triangle upsampler treats
//! the block-padded chroma plane as the image (`raw_height_ =
//! height_in_blocks * 8`, `decode.cc:517`; the `row_bot` choice at
//! `render.cc:704`), so on an even-sized frame whose chroma does not end on a
//! block boundary the last output row and column interpolate against whatever
//! the encoder wrote into the block padding; libjpeg-turbo and zenjpeg
//! replicate the last real chroma row and column instead. The two agree only
//! where the padding repeats the last real chroma, and zenjpeg's encoder (fused
//! 4:2:0 path) leaves the previous strip's chroma in the padding rows of a
//! partial bottom MCU row (CLAUDE.md Known Bugs 6). On the progressive fixture
//! that gives 22 at 1602x1300 against 1 at 1603x1301; the C1 review traced
//! most of the 22 to the stale padding. With odd sizes the last output row and
//! column take the upper/left phase, which reads only real chroma, so these
//! fixtures check the colour model and the decoders, not the encoder's
//! padding. The encoder fix comes with an even-size djpegli fixture.

#![cfg(feature = "__oracle-tests")]

#[path = "support/jpeg_writer.rs"]
mod jpeg_writer;

use enough::Unstoppable;
use jpeg_writer::{Comp, Segment, Spec, app14, dc_block, jfif, resignal};
use std::path::{Path, PathBuf};
use std::process::Command;
use zenjpeg::decode::{ChromaUpsampling, Decoder, IdctMethod};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Interp {
    Rgb,
    YCbCr,
}

/// The colour-relevant markers of a fixture, split by position.
#[derive(Clone, Debug, Default)]
struct Signals {
    ids: [u8; 3],
    jfif_pre_sof: bool,
    adobe_pre_sof: Option<u8>,
    jfif_pre_sos: bool,
    adobe_pre_sos: Option<u8>,
}

/// jpegli `ProcessSOF` (and zenjpeg): decided at SOF.
fn rule_jpegli(s: &Signals) -> Interp {
    rule_jpegli_with(s.jfif_pre_sof, s.adobe_pre_sof, s.ids)
}

/// libjpeg-turbo `default_decompress_parms`: jpegli's rule, at the first SOS.
fn rule_turbo(s: &Signals) -> Interp {
    rule_jpegli_with(s.jfif_pre_sos, s.adobe_pre_sos, s.ids)
}

fn rule_jpegli_with(jfif: bool, adobe: Option<u8>, ids: [u8; 3]) -> Interp {
    match (jfif, adobe) {
        (true, _) => Interp::YCbCr,
        (false, Some(0)) => Interp::Rgb,
        (false, Some(_)) => Interp::YCbCr,
        (false, None) if ids == *b"RGB" => Interp::Rgb,
        _ => Interp::YCbCr,
    }
}

/// IJG libjpeg 9 `default_decompress_parms` (jdapimin.c): component IDs
/// 1,2,3 or 'R','G','B' decide first; then JFIF; then the APP14 transform
/// (0 RGB, anything else YCbCr); then YCbCr. Decided at the first SOS.
fn rule_ijg9(s: &Signals) -> Interp {
    if s.ids == [1, 2, 3] {
        Interp::YCbCr
    } else if s.ids == *b"RGB" {
        Interp::Rgb
    } else {
        rule_jpegli_with(s.jfif_pre_sos, s.adobe_pre_sos, [0; 3])
    }
}

fn tmp_dir() -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("colorspace_oracles");
    std::fs::create_dir_all(&d).expect("tmp dir");
    d
}

fn read_pnm_rgb(path: &Path) -> Vec<u8> {
    let data = std::fs::read(path).expect("pnm");
    // P6 header: magic, width, height, maxval, each separated by whitespace.
    let mut fields = 0;
    let mut i = 0;
    while fields < 4 {
        while data[i].is_ascii_whitespace() {
            i += 1;
        }
        if data[i] == b'#' {
            while data[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        while !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields += 1;
    }
    assert_eq!(
        &data[..2],
        b"P6",
        "{} is not a binary RGB PPM",
        path.display()
    );
    data[i + 1..].to_vec()
}

fn require(cmd: &str) {
    let found = Command::new("sh")
        .args(["-c", &format!("command -v {cmd}")])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    assert!(
        found,
        "oracle tool `{cmd}` is required by __oracle-tests and was not found"
    );
}

fn pillow(jpeg: &Path) -> Vec<u8> {
    // Not `-I`: Pillow may live in the user site-packages. The working
    // directory is the fixture directory, which holds only our own files.
    let out = Command::new("python3")
        .current_dir(tmp_dir())
        .args([
            "-c",
            "import sys; from PIL import Image; \
             sys.stdout.buffer.write(Image.open(sys.argv[1]).convert('RGB').tobytes())",
        ])
        .arg(jpeg)
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "Pillow failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// The IJG rule modelled by `rule_ijg9` is libjpeg 9's; libjpeg-turbo also
/// installs a `djpeg`, with the other rule. Fail unless the banner says IJG 9.
fn require_ijg9_djpeg(sample: &Path) {
    let out = Command::new("djpeg")
        .args(["-verbose", "-outfile", "/dev/null"])
        .arg(sample)
        .output()
        .expect("djpeg");
    let banner = String::from_utf8_lossy(&out.stderr);
    assert!(
        banner.contains("Independent JPEG Group's DJPEG, version 9"),
        "`djpeg` on PATH is not IJG libjpeg 9 (banner: {:?})",
        banner.lines().next().unwrap_or("")
    );
}

fn djpeg_nosmooth(jpeg: &Path) -> Vec<u8> {
    let ppm = jpeg.with_extension("djpeg.ppm");
    let status = Command::new("djpeg")
        .args(["-nosmooth", "-dct", "int", "-outfile"])
        .arg(&ppm)
        .arg(jpeg)
        .status()
        .expect("djpeg");
    // Exit 2 is IJG's EXIT_WARNING: decoded, with a warning (e.g. an unknown
    // Adobe transform, JWRN_ADOBE_XFORM).
    assert!(
        matches!(status.code(), Some(0 | 2)),
        "djpeg failed on {}: {status}",
        jpeg.display()
    );
    read_pnm_rgb(&ppm)
}

fn djpegli(bin: &Path, jpeg: &Path) -> Vec<u8> {
    let ppm = jpeg.with_extension("djpegli.ppm");
    let status = Command::new(bin)
        .arg(jpeg)
        .arg(&ppm)
        .arg("--quiet")
        .status()
        .expect("djpegli");
    assert!(status.success(), "djpegli failed on {}", jpeg.display());
    read_pnm_rgb(&ppm)
}

fn zen(jpeg: &[u8], ups: ChromaUpsampling) -> Vec<u8> {
    Decoder::new()
        .chroma_upsampling(ups)
        .idct_method(IdctMethod::Libjpeg)
        .decode(jpeg, Unstoppable)
        .expect("zenjpeg decode")
        .into_pixels_u8()
        .expect("u8")
}

/// zenjpeg's port of jpegli's decoder reconstruction: Laplacian
/// dequantization biases and the f32 IDCT (`dequant_bias(true)`), quantized.
fn zen_jpegli(jpeg: &[u8]) -> Vec<u8> {
    Decoder::new()
        .dequant_bias(true)
        .decode(jpeg, Unstoppable)
        .expect("zenjpeg dequant-bias decode")
        .into_pixels_f32()
        .expect("f32")
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "size mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

fn hash(a: usize, b: usize, c: usize) -> u32 {
    let mut x = (a as u32).wrapping_mul(0x9E37_79B1)
        ^ (b as u32).wrapping_mul(0x85EB_CA77)
        ^ (c as u32).wrapping_mul(0xC2B2_AE3D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^ (x >> 13)
}

/// DC + low-frequency AC per block, DC kept away from clipping.
fn content(c: usize, bx: usize, by: usize) -> [i16; 64] {
    let mut b = dc_block(56 + (hash(c, bx, by) % 144) as u8);
    for (k, coef) in b.iter_mut().enumerate().take(8).skip(1) {
        *coef = (hash(c * 64 + k, bx, by) % 7) as i16 - 3;
    }
    b
}

/// `content` luma over DC-only chroma that changes from block to block: the
/// regime real photographs put chroma in (repo CLAUDE.md "flat-chroma
/// coverage"), and the only one that reaches the DC-only IDCT and upsampler
/// branches on the chroma planes.
fn content_flat_chroma(c: usize, bx: usize, by: usize) -> [i16; 64] {
    if c == 0 {
        content(c, bx, by)
    } else {
        dc_block(56 + (hash(c, bx, by) % 144) as u8)
    }
}

/// A fixture: the frame, its two unambiguous twins and its signals.
struct Fixture {
    name: String,
    jpeg: Vec<u8>,
    rgb_twin: Vec<u8>,
    ycc_twin: Vec<u8>,
    signals: Signals,
    /// Only zenjpeg's per-component f32 path serves this layout (asymmetric
    /// chroma, 4:1:1): its f32 upsampling rounds independently of
    /// libjpeg-turbo's integer kernels.
    per_component: bool,
}

fn build(name: &str, samp: [(u8, u8); 3], ids: [u8; 3], segments: Vec<Segment>) -> Fixture {
    build_sized(name, samp, ids, segments, (157, 93), false)
}

fn build_sized(
    name: &str,
    samp: [(u8, u8); 3],
    ids: [u8; 3],
    segments: Vec<Segment>,
    size: (u32, u32),
    mcu_row_dri: bool,
) -> Fixture {
    build_with(name, samp, ids, segments, size, mcu_row_dri, content)
}

/// `mcu_row_dri` sets the restart interval to one MCU row: with at least
/// 1024 MCUs and `parallel`, that reaches the fused kernels. `block` supplies
/// each block's coefficients (component, block x, block y).
fn build_with(
    name: &str,
    samp: [(u8, u8); 3],
    ids: [u8; 3],
    segments: Vec<Segment>,
    (width, height): (u32, u32),
    mcu_row_dri: bool,
    block: fn(usize, usize, usize) -> [i16; 64],
) -> Fixture {
    assert!(
        width % 2 == 1 && height % 2 == 1,
        "{name}: odd sizes keep jpegli's triangle off the block padding"
    );
    let write = |ids: [u8; 3], segments: Vec<Segment>| {
        let comps = (0..3)
            .map(|i| Comp {
                id: ids[i],
                h: samp[i].0,
                v: samp[i].1,
            })
            .collect();
        let mut spec = Spec {
            width,
            height,
            comps,
            block: &block,
            restart_interval: 0,
            quant: 8,
            segments,
        };
        if mcu_row_dri {
            spec.restart_interval = spec.mcu_cols() as u16;
        }
        jpeg_writer::write(&spec)
    };
    let mut signals = Signals {
        ids,
        ..Signals::default()
    };
    for seg in &segments {
        let (bytes, pre_sof) = match seg {
            Segment::PreFrame(b) => (b, true),
            Segment::PostFrame(b) => (b, false),
        };
        match bytes[1] {
            0xE0 => {
                signals.jfif_pre_sos = true;
                signals.jfif_pre_sof |= pre_sof;
            }
            0xEE => {
                signals.adobe_pre_sos = Some(bytes[15]);
                if pre_sof {
                    signals.adobe_pre_sof = Some(bytes[15]);
                }
            }
            _ => {}
        }
    }
    let max = |k: fn(&(u8, u8)) -> u8| samp.iter().map(k).max().unwrap();
    let standard = samp[1] == (1, 1)
        && samp[2] == (1, 1)
        && samp[0].0 <= 2
        && samp[0].1 <= 2
        && samp[0] == (max(|s| s.0), max(|s| s.1));
    Fixture {
        name: name.to_string(),
        jpeg: write(ids, segments),
        rgb_twin: write(*b"RGB", vec![Segment::PreFrame(app14(0))]),
        ycc_twin: write([1, 2, 3], vec![Segment::PreFrame(jfif())]),
        signals,
        per_component: !standard,
    }
}

fn fixtures() -> Vec<Fixture> {
    use Segment::{PostFrame, PreFrame};
    let s420 = [(2, 2), (1, 1), (1, 1)];
    let mut v = vec![
        build("ids RGB, no markers", s420, *b"RGB", vec![]),
        build(
            "ids RGB + APP14 t0",
            s420,
            *b"RGB",
            vec![PreFrame(app14(0))],
        ),
        build(
            "ids 123 + APP14 t0",
            s420,
            [1, 2, 3],
            vec![PreFrame(app14(0))],
        ),
        build(
            "ids 7,8,9 + APP14 t0",
            s420,
            [7, 8, 9],
            vec![PreFrame(app14(0))],
        ),
        build(
            "ids RGB + APP14 t1",
            s420,
            *b"RGB",
            vec![PreFrame(app14(1))],
        ),
        build(
            "ids 7,8,9 + APP14 t3",
            s420,
            [7, 8, 9],
            vec![PreFrame(app14(3))],
        ),
        build(
            "ids 7,8,9 + JFIF + APP14 t0",
            s420,
            [7, 8, 9],
            vec![PreFrame(jfif()), PreFrame(app14(0))],
        ),
        build("ids RGB + JFIF", s420, *b"RGB", vec![PreFrame(jfif())]),
        build(
            "ids 7,8,9 + APP14 t0 after SOF",
            s420,
            [7, 8, 9],
            vec![PostFrame(app14(0))],
        ),
        build(
            "ids RGB + APP14 t1 after SOF",
            s420,
            *b"RGB",
            vec![PostFrame(app14(1))],
        ),
        build("ids 123, no markers", s420, [1, 2, 3], vec![]),
    ];
    // The fused parallel kernels (with `parallel`): >= 1024 MCUs, MCU-row DRI.
    for (name, samp) in [
        ("rgb 4:4:4 DRI", [(1, 1); 3]),
        ("rgb 4:2:2 DRI", [(2, 1), (1, 1), (1, 1)]),
        ("rgb 4:2:0 DRI", s420),
    ] {
        let sig = vec![Segment::PreFrame(app14(0))];
        v.push(build_sized(name, samp, *b"RGB", sig, (521, 519), true));
    }
    // DC-only chroma (block to block) on the streaming and fused paths: the
    // DC-only IDCT and the triangle / box upsamplers next to MCU boundaries,
    // exact against libjpeg-turbo.
    let ycc = || vec![Segment::PreFrame(jfif())];
    for (name, samp, size, dri) in [
        ("ycc 4:2:0 flat chroma", s420, (157, 93), false),
        ("ycc 4:2:0 flat chroma DRI", s420, (521, 519), true),
        (
            "ycc 4:2:2 flat chroma DRI",
            [(2, 1), (1, 1), (1, 1)],
            (521, 519),
            true,
        ),
    ] {
        v.push(build_with(
            name,
            samp,
            [1, 2, 3],
            ycc(),
            size,
            dri,
            content_flat_chroma,
        ));
    }
    v.push(build_with(
        "rgb 4:2:0 flat G/B DRI",
        s420,
        *b"RGB",
        vec![Segment::PreFrame(app14(0))],
        (521, 519),
        true,
        content_flat_chroma,
    ));
    // The fast integer coefficient path, threaded at >= 2,000,000 px: an
    // encoder-built progressive 4:2:0 file relabelled RGB (odd sizes, see the
    // module doc).
    v.push(progressive_rgb_fixture(1603, 1301));
    // RGB pixels on every layout the reference decoders accept.
    for (name, samp) in [
        ("rgb 4:4:4", [(1, 1); 3]),
        ("rgb 4:2:2", [(2, 1), (1, 1), (1, 1)]),
        ("rgb 4:4:0", [(1, 2), (1, 1), (1, 1)]),
        ("rgb 4:1:1", [(4, 1), (1, 1), (1, 1)]),
        ("rgb 2x2/2x2/1x1", [(2, 2), (2, 2), (1, 1)]),
    ] {
        v.push(build(
            name,
            samp,
            *b"RGB",
            vec![Segment::PreFrame(app14(0))],
        ));
    }
    v
}

fn progressive_rgb_fixture(w: u32, h: u32) -> Fixture {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    assert!(w % 2 == 1 && h % 2 == 1, "odd sizes (module doc)");
    // Mid-range blocky content: no reconstructed sample near 0 or 255, where
    // jpegli (float, clamps at the end) and libjpeg-turbo / zenjpeg (clamp
    // after the IDCT) legitimately differ.
    let src: Vec<u8> = (0..(w * h * 3) as usize)
        .map(|i| 72 + (hash(i % 3, i / 3 % w as usize / 9, i / 3 / w as usize / 7) % 112) as u8)
        .collect();
    let ycc = EncoderConfig::ycbcr(88, ChromaSubsampling::Quarter)
        .progressive(true)
        .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode");
    let rgb = resignal(&ycc, *b"RGB", &[app14(0)], &[]);
    Fixture {
        name: format!("rgb progressive 4:2:0 {w}x{h}"),
        jpeg: rgb.clone(),
        rgb_twin: rgb,
        ycc_twin: resignal(&ycc, [1, 2, 3], &[jfif()], &[]),
        signals: Signals {
            ids: *b"RGB",
            adobe_pre_sof: Some(0),
            adobe_pre_sos: Some(0),
            ..Signals::default()
        },
        per_component: false,
    }
}

struct Oracle<'a> {
    name: &'static str,
    run: Box<dyn Fn(&Path) -> Vec<u8> + 'a>,
    rule: fn(&Signals) -> Interp,
    /// zenjpeg's decode of a twin with the reconstruction this oracle uses.
    zen: fn(&[u8]) -> Vec<u8>,
    /// Max channel difference allowed against zenjpeg's decode of the
    /// interpretation the oracle chose: (RGB, YCbCr).
    tolerance: (u8, u8),
    /// Allowed on layouts only the per-component f32 path serves.
    per_component_tolerance: u8,
}

#[test]
fn oracles_agree_on_colour_model_and_rgb_pixels() {
    require("python3");
    require("djpeg");
    let djpegli_bin = std::env::var_os("ZENJPEG_DJPEGLI").map(PathBuf::from);
    if let Some(bin) = &djpegli_bin {
        assert!(
            bin.is_file(),
            "ZENJPEG_DJPEGLI={} does not exist",
            bin.display()
        );
    }

    let mut oracles = vec![
        Oracle {
            name: "libjpeg-turbo (Pillow)",
            run: Box::new(pillow),
            rule: rule_turbo,
            zen: |j| zen(j, ChromaUpsampling::Triangle),
            // zenjpeg's Libjpeg IDCT, triangle upsampling and 16-bit colour
            // are libjpeg-turbo's.
            tolerance: (0, 0),
            // f32 triangle upsampling vs turbo's integer biases (measured 1).
            per_component_tolerance: 1,
        },
        Oracle {
            name: "IJG libjpeg 9 (djpeg -nosmooth)",
            run: Box::new(djpeg_nosmooth),
            rule: rule_ijg9,
            zen: |j| zen(j, ChromaUpsampling::NearestNeighbor),
            // Same islow IDCT and replication; IJG 9's YCbCr→RGB rounds
            // differently from libjpeg-turbo's in places (measured 1).
            tolerance: (0, 1),
            per_component_tolerance: 0,
        },
    ];
    if let Some(bin) = djpegli_bin.clone() {
        oracles.push(Oracle {
            name: "jpegli (djpegli)",
            run: Box::new(move |p| djpegli(&bin, p)),
            rule: rule_jpegli,
            zen: zen_jpegli,
            // zenjpeg's `dequant_bias(true)` decode ports jpegli's
            // reconstruction (Laplacian biases, f32 IDCT); jpegli keeps the
            // whole pipeline in f32 and rounds once, at the end. Measured 1 on
            // every fixture (2026-10-10, djpegli from internal/jpegli-cpp).
            tolerance: (1, 1),
            per_component_tolerance: 1,
        });
    }

    let dir = tmp_dir();
    let mut failures = Vec::new();
    for (i, fx) in fixtures().iter().enumerate() {
        let path = dir.join(format!("fixture_{i:02}.jpg"));
        std::fs::write(&path, &fx.jpeg).expect("write fixture");
        if i == 0 {
            require_ijg9_djpeg(&path);
        }
        // zenjpeg decodes the fixture exactly as the twin jpegli's rule picks.
        let chosen = match rule_jpegli(&fx.signals) {
            Interp::Rgb => &fx.rgb_twin,
            Interp::YCbCr => &fx.ycc_twin,
        };
        for ups in [
            ChromaUpsampling::Triangle,
            ChromaUpsampling::NearestNeighbor,
        ] {
            let d = max_diff(&zen(&fx.jpeg, ups), &zen(chosen, ups));
            if d != 0 {
                failures.push(format!(
                    "{}: zenjpeg's {ups:?} decode differs from its decode of the {:?} twin by {d}",
                    fx.name,
                    rule_jpegli(&fx.signals)
                ));
            }
        }
        // zenjpeg's own interpretation is jpegli's rule.
        let zen_interp = match Decoder::new()
            .read_info(&fx.jpeg)
            .expect("info")
            .color_space
        {
            zenjpeg::types::ColorSpace::Rgb => Interp::Rgb,
            _ => Interp::YCbCr,
        };
        if zen_interp != rule_jpegli(&fx.signals) {
            failures.push(format!(
                "{}: zenjpeg reads {zen_interp:?}, jpegli rule says {:?}",
                fx.name,
                rule_jpegli(&fx.signals)
            ));
        }
        for o in &oracles {
            let got = (o.run)(&path);
            let as_rgb = (o.zen)(&fx.rgb_twin);
            let as_ycc = (o.zen)(&fx.ycc_twin);
            let (d_rgb, d_ycc) = (max_diff(&got, &as_rgb), max_diff(&got, &as_ycc));
            let (interp, d) = if d_rgb <= d_ycc {
                (Interp::Rgb, d_rgb)
            } else {
                (Interp::YCbCr, d_ycc)
            };
            let want = (o.rule)(&fx.signals);
            let mut tolerance = match interp {
                Interp::Rgb => o.tolerance.0,
                Interp::YCbCr => o.tolerance.1,
            };
            if fx.per_component {
                tolerance = tolerance.max(o.per_component_tolerance);
            }
            eprintln!(
                "{:<34} {:<32} interp {interp:?} (rule {want:?}, zenjpeg {zen_interp:?}) worst diff {d} (allowed {tolerance})",
                fx.name, o.name
            );
            if interp != want {
                failures.push(format!(
                    "{}: {} decoded as {interp:?}, its documented rule says {want:?}",
                    fx.name, o.name
                ));
            }
            if d > tolerance {
                failures.push(format!(
                    "{}: {} differs from zenjpeg's {interp:?} decode by {d} (allowed {tolerance})",
                    fx.name, o.name
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} oracle failure(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
