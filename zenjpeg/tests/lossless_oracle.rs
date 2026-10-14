//! Third-party oracles for the lossless pipeline: libjpeg-turbo's `jpegtran`
//! and `djpeg`, and `exiftool`.
//!
//! These tests run only when the caller sets `ZENJPEG_ORACLE_TOOLS=1` (the
//! CI `oracle-tools` job and `just oracle` do); with the variable set, a
//! missing tool is a hard failure, never a silent pass. Without it every test
//! here returns immediately — the skip is the caller's decision, visible in
//! the workflow and the justfile, not something decided inside the test.
//!
//! Fixtures are synthetic (zenjpeg's own encoder); temporary files go under
//! `CARGO_TARGET_TMPDIR`.

use std::path::{Path, PathBuf};
use std::process::Command;

use enough::Unstoppable;
use zenjpeg::container::xmp::generate_primary_xmp;
use zenjpeg::encode::EncoderSegments;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::foundation::consts::JPEG_NATURAL_ORDER;
use zenjpeg::lossless::{EdgeHandling, LosslessTransform, TransformConfig, transform};

const ENV: &str = "ZENJPEG_ORACLE_TOOLS";

fn oracles_requested() -> bool {
    std::env::var_os(ENV).is_some_and(|v| v == "1")
}

/// Resolve a tool, failing loudly when the caller asked for oracles but the
/// tool is not installed.
fn tool(name: &str) -> String {
    let ok = Command::new(name)
        .arg(if name == "exiftool" {
            "-ver"
        } else {
            "-version"
        })
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty() || !o.stdout.is_empty())
        .unwrap_or(false);
    assert!(
        ok,
        "{ENV}=1 but `{name}` is not runnable (install libjpeg-turbo-progs / exiftool)"
    );
    name.to_string()
}

fn tmp_path(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("lossless_oracle");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn run(cmd: &str, args: &[&str]) -> Vec<u8> {
    let out = Command::new(cmd).args(args).output().expect("spawn");
    assert!(
        out.status.success(),
        "{cmd} {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// `djpeg -pnm` decode as raw bytes (header + pixels), the oracle's view of
/// the pixels.
fn djpeg_pnm(djpeg: &str, path: &Path) -> Vec<u8> {
    run(djpeg, &["-pnm", path.to_str().unwrap()])
}

// ── fixture helpers (same as the in-crate tests) ─────────────────────────────

fn for_each_segment(jpeg: &[u8], mut f: impl FnMut(u8, usize, &[u8])) {
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        let marker = jpeg[i + 1];
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        f(marker, i, &jpeg[i + 4..i + 2 + len]);
        if marker == 0xDA {
            break;
        }
        i += 2 + len;
    }
}

/// Rewrite every 8-bit DQT so `Q[r][c] = 1 + r + 2c` (asymmetric off-diagonal).
fn patch_quant_tables_asymmetric(jpeg: &mut [u8]) {
    let mut edits = Vec::new();
    for_each_segment(jpeg, |marker, off, payload| {
        if marker != 0xDB {
            return;
        }
        let mut p = 0;
        while p < payload.len() {
            assert_eq!(payload[p] >> 4, 0, "8-bit DQT expected");
            for k in 0..64 {
                let n = JPEG_NATURAL_ORDER[k] as usize;
                edits.push((off + 4 + p + 1 + k, (1 + n / 8 + 2 * (n % 8)) as u8));
            }
            p += 65;
        }
    });
    for (pos, v) in edits {
        jpeg[pos] = v;
    }
}

fn textured_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut px = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            px[i] = (128 + ((x * 3 + y) % 9) as i32 - 4) as u8;
            px[i + 1] = (128 + ((x + y * 5) % 7) as i32 - 3) as u8;
            px[i + 2] = (128 + ((x * y) % 5) as i32 - 2) as u8;
        }
    }
    px
}

fn encode(w: u32, h: u32, px: &[u8], sub: ChromaSubsampling) -> Vec<u8> {
    let mut enc = EncoderConfig::ycbcr(92, sub)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(px, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// jpegtran's spelling of each D4 element (jpegtran rotates clockwise, as
/// `LosslessTransform::Rotate90` does).
fn jpegtran_args(t: LosslessTransform) -> Vec<&'static str> {
    match t {
        LosslessTransform::None => vec![],
        LosslessTransform::FlipHorizontal => vec!["-flip", "horizontal"],
        LosslessTransform::FlipVertical => vec!["-flip", "vertical"],
        LosslessTransform::Rotate90 => vec!["-rotate", "90"],
        LosslessTransform::Rotate180 => vec!["-rotate", "180"],
        LosslessTransform::Rotate270 => vec!["-rotate", "270"],
        LosslessTransform::Transpose => vec!["-transpose"],
        LosslessTransform::Transverse => vec!["-transverse"],
    }
}

/// Every transform of an asymmetric-table JPEG decodes (with djpeg) to exactly
/// what jpegtran's transform of the same file decodes to. Both operate on the
/// same coefficients; with the quantization tables transposed correctly the
/// two outputs dequantize identically, so the oracle decode is byte-identical.
#[test]
fn transforms_match_jpegtran_under_djpeg() {
    if !oracles_requested() {
        return;
    }
    let (jpegtran, djpeg) = (tool("jpegtran"), tool("djpeg"));
    for (sub, tag) in [
        (ChromaSubsampling::None, "444"),
        (ChromaSubsampling::Quarter, "420"),
    ] {
        // MCU-aligned for both 8×8 and 16×16 so jpegtran needs no trimming.
        let (w, h) = (48u32, 32u32);
        let mut jpeg = encode(w, h, &textured_rgb(w, h), sub);
        patch_quant_tables_asymmetric(&mut jpeg);
        let src = tmp_path(&format!("src-{tag}.jpg"));
        std::fs::write(&src, &jpeg).unwrap();

        for t in LosslessTransform::ALL {
            let ours = transform(
                &jpeg,
                &TransformConfig {
                    transform: t,
                    edge_handling: EdgeHandling::RejectPartialBlocks,
                },
                Unstoppable,
            )
            .unwrap();
            let ours_path = tmp_path(&format!("ours-{tag}-{t:?}.jpg"));
            std::fs::write(&ours_path, &ours).unwrap();

            let theirs_path = tmp_path(&format!("jpegtran-{tag}-{t:?}.jpg"));
            let mut args = jpegtran_args(t);
            args.extend([
                "-perfect",
                "-outfile",
                theirs_path.to_str().unwrap(),
                src.to_str().unwrap(),
            ]);
            run(&jpegtran, &args);

            let a = djpeg_pnm(&djpeg, &ours_path);
            let b = djpeg_pnm(&djpeg, &theirs_path);
            assert_eq!(
                a.len(),
                b.len(),
                "{tag} {t:?}: djpeg output size differs from jpegtran's"
            );
            let diffs = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert_eq!(
                diffs, 0,
                "{tag} {t:?}: {diffs} bytes differ between djpeg(ours) and djpeg(jpegtran)"
            );
        }
    }
}

/// After a transform of a Multi-Picture JPEG, exiftool must locate the
/// carried secondary through the rebuilt MPF index (`-b -MPImage2` walks the
/// MP entry offsets) and get exactly the bytes we appended, and `-validate`
/// must not complain about the file.
#[test]
fn mpf_secondary_is_reachable_through_exiftool() {
    if !oracles_requested() {
        return;
    }
    let exiftool = tool("exiftool");

    let (sw, sh) = (32u32, 24u32);
    let gray: Vec<u8> = (0..sw * sh).map(|i| (40 + i % 160) as u8).collect();
    let mut enc = EncoderConfig::grayscale(80)
        .encode_from_bytes(sw, sh, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&gray, Unstoppable).unwrap();
    let secondary = enc.finish().unwrap();

    let (pw, ph) = (64u32, 48u32);
    let segments = EncoderSegments::new()
        .set_xmp(&generate_primary_xmp(secondary.len()))
        .add_gainmap(secondary.clone());
    let mut enc = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter)
        .with_segments(segments)
        .encode_from_bytes(pw, ph, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&textured_rgb(pw, ph), Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();

    for t in [
        LosslessTransform::None,
        LosslessTransform::Rotate90,
        LosslessTransform::FlipHorizontal,
        LosslessTransform::Transverse,
    ] {
        let out = transform(
            &jpeg,
            &TransformConfig {
                transform: t,
                edge_handling: EdgeHandling::RejectPartialBlocks,
            },
            Unstoppable,
        )
        .unwrap();
        let path = tmp_path(&format!("mpf-{t:?}.jpg"));
        std::fs::write(&path, &out).unwrap();

        // The bytes we appended are the tail of the file.
        let carried = {
            let dec = zenjpeg::decode::DecodeConfig::new()
                .preserve(zenjpeg::decode::PreserveConfig::all());
            let (_, extras) = dec
                .decode_coefficients_with_extras(&out, Unstoppable)
                .unwrap();
            extras
                .unwrap()
                .gainmap()
                .expect("carried secondary")
                .to_vec()
        };
        let via_exiftool = run(&exiftool, &["-b", "-MPImage2", path.to_str().unwrap()]);
        assert_eq!(
            via_exiftool, carried,
            "{t:?}: exiftool's MPImage2 (via the MPF index) is not the carried secondary"
        );
        // exiftool sees two MP images and a clean file.
        let n = String::from_utf8(run(
            &exiftool,
            &["-s3", "-NumberOfImages", path.to_str().unwrap()],
        ))
        .unwrap();
        assert_eq!(n.trim(), "2", "{t:?}: NumberOfImages");
        let warnings = String::from_utf8(run(
            &exiftool,
            &["-validate", "-warning", "-a", "-s3", path.to_str().unwrap()],
        ))
        .unwrap();
        assert!(
            !warnings.to_ascii_lowercase().contains("mpf")
                && !warnings.to_ascii_lowercase().contains("offset"),
            "{t:?}: exiftool -validate warnings mention MPF/offsets:\n{warnings}"
        );
    }
}
