//! diffmap-RD probe cell runner (2026-07-18, diffmap-RD worktree).
//!
//! Encodes a P6 PPM source at one or more target zensim scores under a chosen
//! driver, decodes each result, and emits one TSV row per cell (stdout) plus
//! the decoded P6 PPM (for the external independent judge panel). One PROCESS
//! per (driver, image): the `ZENJPEG_ZQ_PROFILE` / `ZENJPEG_ZQ_MODEL_MAP` env
//! selection and the cached model-sensitivity gradient are process-global.
//!
//! Design: zensim `docs/RD_TARGET_EVAL_DESIGN_2026-07-18.md`.
//!
//! ```sh
//! ZENJPEG_ZQ_PROFILE=b cargo run --release -p zenjpeg --features target-zq \
//!   --example zq_rd_probe -- --image x.ppm --targets 25,40,55,70,80,90 \
//!   --driver aq --label zensimB_aq --out-dir /mnt/v/output/.../decoded
//! ```
//!
//! Drivers:
//! - `global` — ZqExplicit, NO per-block artifact bound (global-q search only)
//! - `aq`     — ZqExplicit + BlockArtifactBound (diffmap-driven per-block AQ)
//! - `picker` — ZqPicker one-shot (no measurement; the efficiency floor)
//!
//! TSV columns:
//! image  driver  target  bytes  achieved_score  passes_used  max_block_artifact  encode_ms

use std::io::Write as _;
use std::time::Instant;

use enough::Unstoppable;
use zenjpeg::encode::zq::{BlockArtifactBound, ZqTarget};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

fn read_ppm(path: &str) -> (Vec<u8>, u32, u32) {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // Minimal strict P6 parser: "P6" ws w ws h ws 255 single-ws raw.
    let mut pos = 0usize;
    let mut tok = || {
        while pos < data.len() && data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < data.len() && data[pos] == b'#' {
            while pos < data.len() && data[pos] != b'\n' {
                pos += 1;
            }
            while pos < data.len() && data[pos].is_ascii_whitespace() {
                pos += 1;
            }
        }
        let start = pos;
        while pos < data.len() && !data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        std::str::from_utf8(&data[start..pos]).unwrap().to_string()
    };
    assert_eq!(tok(), "P6", "{path}: not a P6 ppm");
    let w: u32 = tok().parse().expect("width");
    let h: u32 = tok().parse().expect("height");
    let maxv: u32 = tok().parse().expect("maxval");
    assert_eq!(maxv, 255, "{path}: 8-bit PPM required");
    pos += 1; // single whitespace after maxval
    let need = (w as usize) * (h as usize) * 3;
    let px = data[pos..pos + need].to_vec();
    (px, w, h)
}

fn write_ppm(path: &str, px: &[u8], w: u32, h: u32) {
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    f.write_all(px).unwrap();
}

fn main() {
    let mut image = String::new();
    let mut targets: Vec<f32> = vec![55.0];
    let mut driver = "aq".to_string();
    let mut label = String::new();
    let mut out_dir = String::from(".");
    let mut block_ceiling: f32 = 0.02;
    let mut max_passes: u8 = 3;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => image = args.next().unwrap(),
            "--targets" => {
                targets = args
                    .next()
                    .unwrap()
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--driver" => driver = args.next().unwrap(),
            "--label" => label = args.next().unwrap(),
            "--out-dir" => out_dir = args.next().unwrap(),
            "--block-ceiling" => block_ceiling = args.next().unwrap().parse().unwrap(),
            "--max-passes" => max_passes = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    if label.is_empty() {
        label = driver.clone();
    }
    let stem = std::path::Path::new(&image)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let (px, w, h) = read_ppm(&image);
    std::fs::create_dir_all(&out_dir).ok();

    for &t in &targets {
        let quality = match driver.as_str() {
            "global" => Quality::ZqExplicit(ZqTarget::new(t).with_max_passes(max_passes)),
            "aq" => Quality::ZqExplicit(
                ZqTarget::new(t)
                    .with_max_passes(max_passes)
                    .with_block_artifact(Some(BlockArtifactBound::new(block_ceiling))),
            ),
            "picker" => Quality::ZqPicker(t),
            other => panic!("unknown driver {other}"),
        };
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
        let t0 = Instant::now();
        let mut enc = config
            .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
            .expect("encoder");
        enc.push_packed(&px, Unstoppable).expect("push");
        let (jpeg, m) = enc.finish_with_metrics().expect("finish_with_metrics");
        let ms = t0.elapsed().as_secs_f64() * 1e3;

        // Decode for the external judge panel.
        let dec = zenjpeg::decode::Decoder::new()
            .decode(&jpeg, Unstoppable)
            .expect("decode");
        let dpx = dec.into_pixels_u8().expect("u8 pixels");
        let out = format!("{out_dir}/{label}__{stem}__t{t:.0}.ppm");
        write_ppm(&out, &dpx, w, h);

        println!(
            "{stem}\t{label}\t{t:.0}\t{}\t{:.3}\t{}\t{:.5}\t{ms:.1}",
            jpeg.len(),
            m.achieved_score,
            m.passes_used,
            m.achieved_max_block_artifact,
        );
    }
}
