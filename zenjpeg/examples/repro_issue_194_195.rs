//! Replication harness for issues #194 and #195.
//!
//! #194: lossless::transform() Transpose/Transverse produce wrong pixels on
//!       subsampled chroma (4:2:2 / 4:2:0), pixel-identical on 4:4:4.
//! #195: EdgeHandling::TrimPartialBlocks emits a corrupt JPEG whenever a trim
//!       actually happens (non-MCU-aligned input).
//!
//! Reference: mozjpeg jpegtran/djpeg at /opt/homebrew/opt/mozjpeg/bin (or
//! MOZJPEG_BIN env). Usage:
//! ```bash
//! cargo run --release --example repro_issue_194_195 -- <outdir>
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestartInterval, RestructureConfig,
    TransformConfig, restructure, transform,
};

fn moz_bin(tool: &str) -> PathBuf {
    let base =
        std::env::var("MOZJPEG_BIN").unwrap_or_else(|_| "/opt/homebrew/opt/mozjpeg/bin".into());
    Path::new(&base).join(tool)
}

fn gen_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            rgb[i] = (x % 256) as u8;
            rgb[i + 1] = (y % 256) as u8;
            rgb[i + 2] = ((x ^ y) % 256) as u8;
        }
    }
    rgb
}

fn encode(w: u32, h: u32, rgb: &[u8], ss: ChromaSubsampling) -> Vec<u8> {
    let mut enc = EncoderConfig::ycbcr(90.0, ss)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder config");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn jpegtran_args(t: LosslessTransform) -> &'static [&'static str] {
    match t {
        LosslessTransform::None => &[],
        LosslessTransform::FlipHorizontal => &["-flip", "horizontal"],
        LosslessTransform::FlipVertical => &["-flip", "vertical"],
        LosslessTransform::Transpose => &["-transpose"],
        LosslessTransform::Transverse => &["-transverse"],
        LosslessTransform::Rotate90 => &["-rotate", "90"],
        LosslessTransform::Rotate180 => &["-rotate", "180"],
        LosslessTransform::Rotate270 => &["-rotate", "270"],
    }
}

/// Decode a JPEG file with mozjpeg djpeg to PPM. Returns (w, h, samples, stderr).
fn djpeg(path: &Path) -> Option<(u32, u32, Vec<u8>, String)> {
    let out = Command::new(moz_bin("djpeg"))
        .arg("-pnm")
        .arg(path)
        .output()
        .expect("run djpeg");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        eprintln!("    djpeg FAILED on {}: {}", path.display(), stderr.trim());
        return None;
    }
    let (w, h, pix) = parse_ppm(&out.stdout)?;
    Some((w, h, pix, stderr))
}

fn parse_ppm(data: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    // P6\n<w> <h>\n<maxval>\n<binary>
    let mut fields = Vec::new();
    let mut pos = 0usize;
    while fields.len() < 4 && pos < data.len() {
        while pos < data.len() && data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < data.len() && data[pos] == b'#' {
            while pos < data.len() && data[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < data.len() && !data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(std::str::from_utf8(&data[start..pos]).ok()?.to_string());
    }
    pos += 1; // single whitespace after maxval
    if fields.len() != 4 || fields[0] != "P6" {
        return None;
    }
    let w: u32 = fields[1].parse().ok()?;
    let h: u32 = fields[2].parse().ok()?;
    Some((w, h, data[pos..].to_vec()))
}

struct DiffStats {
    total: usize,
    differing: usize,
    max_abs: u32,
}

fn diff(a: &[u8], b: &[u8]) -> DiffStats {
    assert_eq!(a.len(), b.len(), "sample count mismatch");
    let mut differing = 0usize;
    let mut max_abs = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        let d = (i32::from(x) - i32::from(y)).unsigned_abs();
        if d != 0 {
            differing += 1;
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    DiffStats {
        total: a.len(),
        differing,
        max_abs,
    }
}

fn run_issue_194(outdir: &Path) {
    println!("=== Issue #194: Transpose/Transverse on subsampled chroma (640x480, aligned) ===");
    let (w, h) = (640u32, 480u32);
    let rgb = gen_rgb(w, h);
    let subsamplings = [
        ("4:4:4", ChromaSubsampling::None),
        ("4:2:2", ChromaSubsampling::HalfHorizontal),
        ("4:2:0", ChromaSubsampling::Quarter),
    ];
    let transforms = [
        LosslessTransform::FlipHorizontal,
        LosslessTransform::FlipVertical,
        LosslessTransform::Rotate90,
        LosslessTransform::Rotate180,
        LosslessTransform::Rotate270,
        LosslessTransform::Transpose,
        LosslessTransform::Transverse,
    ];
    for (ss_name, ss) in subsamplings {
        let src = encode(w, h, &rgb, ss);
        let src_path = outdir.join(format!("i194-in-{}.jpg", ss_name.replace(':', "")));
        std::fs::write(&src_path, &src).unwrap();
        for t in transforms {
            let cfg = TransformConfig {
                transform: t,
                edge_handling: EdgeHandling::RejectPartialBlocks,
            };
            let zen_out = match transform(&src, &cfg, Unstoppable) {
                Ok(v) => v,
                Err(e) => {
                    println!("{ss_name} {t:?}: zenjpeg transform ERR: {e}");
                    continue;
                }
            };
            let zen_path =
                outdir.join(format!("i194-{}-{:?}-zen.jpg", ss_name.replace(':', ""), t));
            std::fs::write(&zen_path, &zen_out).unwrap();

            let ref_path =
                outdir.join(format!("i194-{}-{:?}-moz.jpg", ss_name.replace(':', ""), t));
            let jt = Command::new(moz_bin("jpegtran"))
                .args(["-copy", "none"])
                .args(jpegtran_args(t))
                .arg("-outfile")
                .arg(&ref_path)
                .arg(&src_path)
                .output()
                .expect("run jpegtran");
            assert!(
                jt.status.success(),
                "jpegtran failed: {}",
                String::from_utf8_lossy(&jt.stderr)
            );

            let zen_dec = djpeg(&zen_path);
            let moz_dec = djpeg(&ref_path).expect("reference must decode");
            match zen_dec {
                None => {
                    println!("{ss_name} {t:?}: zenjpeg output REJECTED by mozjpeg djpeg");
                }
                Some((zw, zh, zp, zwarn)) => {
                    let (mw, mh, mp, _) = moz_dec;
                    if (zw, zh) != (mw, mh) {
                        println!(
                            "{ss_name} {t:?}: DIMENSION MISMATCH zen {zw}x{zh} vs moz {mw}x{mh}"
                        );
                        continue;
                    }
                    let s = diff(&zp, &mp);
                    let pct = 100.0 * s.differing as f64 / s.total as f64;
                    let warn = if zwarn.trim().is_empty() {
                        String::new()
                    } else {
                        format!("  [djpeg warn: {}]", zwarn.trim().replace('\n', " | "))
                    };
                    if s.differing == 0 {
                        println!("{ss_name} {t:?}: ok (identical, {} samples){warn}", s.total);
                    } else {
                        println!(
                            "{ss_name} {t:?}: DIFFERS on {}/{} samples ({pct:.4}%), max |d|={}{warn}",
                            s.differing, s.total, s.max_abs
                        );
                    }
                }
            }
        }
    }
}

fn encode_mode(w: u32, h: u32, rgb: &[u8], ss: ChromaSubsampling, progressive: bool) -> Vec<u8> {
    let mut enc = EncoderConfig::ycbcr(90.0, ss)
        .progressive(progressive)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder config");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn run_issue_195(outdir: &Path) {
    println!();
    println!("=== Issue #195: TrimPartialBlocks corrupt output (2000x1333, 4:2:0) ===");
    let (w, h) = (2000u32, 1333u32);
    let rgb = gen_rgb(w, h);

    let transforms = [
        LosslessTransform::None,
        LosslessTransform::FlipHorizontal,
        LosslessTransform::Rotate270,
        LosslessTransform::FlipVertical,
        LosslessTransform::Rotate180,
        LosslessTransform::Rotate90,
    ];
    for (in_name, in_prog) in [("baseline-in", false), ("progressive-in", true)] {
        let src = encode_mode(w, h, &rgb, ChromaSubsampling::Quarter, in_prog);
        let src_path = outdir.join(format!("i195-{in_name}-2000x1333.jpg"));
        std::fs::write(&src_path, &src).unwrap();
        for (out_name, out_mode) in [
            ("seq-out", OutputMode::Sequential),
            ("prog-out", OutputMode::Progressive),
        ] {
            run_195_case(outdir, &src, &transforms, in_name, out_name, out_mode);
        }
    }
}

fn run_195_case(
    outdir: &Path,
    src: &[u8],
    transforms: &[LosslessTransform],
    in_name: &str,
    out_name: &str,
    out_mode: OutputMode,
) {
    for &t in transforms {
        // Control: does RejectPartialBlocks say a trim is needed?
        let reject_cfg = TransformConfig {
            transform: t,
            edge_handling: EdgeHandling::RejectPartialBlocks,
        };
        let needs_trim = transform(src, &reject_cfg, Unstoppable).is_err();

        let cfg = RestructureConfig {
            output_mode: out_mode,
            restart_interval: RestartInterval::None,
            transform: Some(TransformConfig {
                transform: t,
                edge_handling: EdgeHandling::TrimPartialBlocks,
            }),
        };
        let out = match restructure(src, &cfg, Unstoppable) {
            Ok(v) => v,
            Err(e) => {
                println!("{in_name}/{out_name} {t:?}: restructure ERR: {e}");
                continue;
            }
        };
        let out_path = outdir.join(format!("i195-{in_name}-{out_name}-{t:?}.jpg"));
        std::fs::write(&out_path, &out).unwrap();

        // mozjpeg verdict
        let moz = match djpeg(&out_path) {
            None => "REJECT".to_string(),
            Some((dw, dh, _, warn)) => {
                if warn.trim().is_empty() {
                    format!("OK {dw}x{dh}")
                } else {
                    format!(
                        "OK-with-warnings {dw}x{dh} [{}]",
                        warn.trim().replace('\n', " | ")
                    )
                }
            }
        };
        // zenjpeg's own decoder verdict
        let zen = match Decoder::new().decode(&out, Unstoppable) {
            Ok(r) => format!("OK {}x{}", r.width, r.height),
            Err(e) => format!("ERR: {e}"),
        };
        println!(
            "{in_name}/{out_name} {t:?}: trim_needed={needs_trim}  mozjpeg-djpeg: {moz}  zenjpeg-decode: {zen}"
        );
    }
}

fn run_aligned_controls(outdir: &Path) {
    println!();
    println!("=== Controls: fully MCU-aligned inputs (no partial blocks anywhere) ===");
    for (w, h) in [(640u32, 480u32), (2000u32, 1328u32)] {
        let rgb = gen_rgb(w, h);
        let src = encode_mode(w, h, &rgb, ChromaSubsampling::Quarter, false);
        for (out_name, out_mode) in [
            ("seq-out", OutputMode::Sequential),
            ("prog-out", OutputMode::Progressive),
        ] {
            for t in [
                LosslessTransform::None,
                LosslessTransform::Rotate90,
                LosslessTransform::Rotate270,
            ] {
                let cfg = RestructureConfig {
                    output_mode: out_mode,
                    restart_interval: RestartInterval::None,
                    transform: Some(TransformConfig {
                        transform: t,
                        edge_handling: EdgeHandling::RejectPartialBlocks,
                    }),
                };
                let out = match restructure(&src, &cfg, Unstoppable) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("{w}x{h} {out_name} {t:?}: restructure ERR: {e}");
                        continue;
                    }
                };
                let out_path = outdir.join(format!("ctrl-{w}x{h}-{out_name}-{t:?}.jpg"));
                std::fs::write(&out_path, &out).unwrap();
                let moz = match djpeg(&out_path) {
                    None => "REJECT".to_string(),
                    Some((dw, dh, _, warn)) => {
                        if warn.trim().is_empty() {
                            format!("OK {dw}x{dh}")
                        } else {
                            format!("OK-with-warnings [{}]", warn.trim().replace('\n', " | "))
                        }
                    }
                };
                println!("{w}x{h} {out_name} {t:?}: mozjpeg-djpeg: {moz}");
            }
        }
    }
}

fn main() {
    let outdir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("repro-194-195-out"));
    std::fs::create_dir_all(&outdir).unwrap();
    run_issue_194(&outdir);
    run_issue_195(&outdir);
    run_aligned_controls(&outdir);
}
