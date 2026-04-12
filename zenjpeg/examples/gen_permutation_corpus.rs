//! Permutation-based JPEG corpus generator.
//!
//! Runs cjpeg (libjpeg-turbo) and cjpegli (jpegli) over a matrix of synthetic
//! source images and parameter combinations, dedupes outputs by content hash,
//! writes files into a sharded directory tree, and records a manifest TSV.
//!
//! Output dir:  $ZENJPEG_PERM_OUT (default: ~/work/zen/zenjpeg-perm-corpus)
//! Cap bytes:   $ZENJPEG_PERM_CAP_MB (default: 6144 MiB)
//! Quick mode:  $ZENJPEG_PERM_QUICK=1 — shrinks axes for a fast smoke run
//!
//! Run: `cargo run --release --example gen_permutation_corpus`
//!
//! TODO: move this generator (and `gb reference decoder output generation) to
//! a new `zen/allthejpegs` repo that uses Docker to reproducibly install and
//! run a much wider set of encoders (mozjpeg, libjpeg-turbo, libjpeg6b, guetzli,
//! jpegli, sjpeg, trimage, nanojpeg, lepton, etc.) against a stable source
//! image set, producing a versioned reference corpus that any zen crate can
//! consume. Reference decoder output (mozjpeg, libjpeg-turbo) would also live
//! in-repo as pre-decoded pixel hashes so downstream crates don't need FFI.
//! See issue (to be filed): "create zen/allthejpegs reproducible corpus repo".

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

const DEFAULT_OUT: &str = "/home/lilith/work/zen/zenjpeg-perm-corpus";
const CJPEG: &str = "/usr/bin/cjpeg";
const CJPEGLI: &str = "/usr/local/bin/cjpegli";

// ── Source synthesis ───────────────────────────────────────────────────────

#[derive(Clone)]
struct Source {
    name: String,
    channels: u32, // 1 or 3
    ppm_path: PathBuf,
}

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

fn gen_noise(w: u32, h: u32, c: u32, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..(w * h * c)).map(|_| (lcg(&mut s) & 0xff) as u8).collect()
}

fn gen_noise_patches(w: u32, h: u32, c: u32, seed: u64) -> Vec<u8> {
    let mut out = gen_noise(w, h, c, seed);
    let mut s = seed ^ 0xdeadbeef;
    let n_patches = 6 + (lcg(&mut s) % 6);
    for _ in 0..n_patches {
        let px = lcg(&mut s) % w;
        let py = lcg(&mut s) % h;
        let pw = 1 + (lcg(&mut s) % (w / 3).max(1));
        let ph = 1 + (lcg(&mut s) % (h / 3).max(1));
        let col = [
            (lcg(&mut s) & 0xff) as u8,
            (lcg(&mut s) & 0xff) as u8,
            (lcg(&mut s) & 0xff) as u8,
        ];
        for y in py..(py + ph).min(h) {
            for x in px..(px + pw).min(w) {
                let idx = ((y * w + x) * c) as usize;
                for k in 0..c as usize {
                    out[idx + k] = col[k.min(2)];
                }
            }
        }
    }
    out
}

fn gen_checkerboard(w: u32, h: u32, c: u32, block: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * c) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = if ((x / block) ^ (y / block)) & 1 == 0 {
                240
            } else {
                16
            };
            let idx = ((y * w + x) * c) as usize;
            for k in 0..c as usize {
                out[idx + k] = v;
            }
        }
    }
    out
}

fn gen_edges(w: u32, h: u32, c: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * c) as usize];
    for y in 0..h {
        for x in 0..w {
            let a = ((y * 11) & 0xff) as u8;
            let b = ((x * 13) & 0xff) as u8;
            let v = a.wrapping_add(b);
            let idx = ((y * w + x) * c) as usize;
            if c == 3 {
                out[idx] = v;
                out[idx + 1] = a;
                out[idx + 2] = b;
            } else {
                out[idx] = v;
            }
        }
    }
    out
}

fn gen_bands(w: u32, h: u32, c: u32, seed: u64) -> Vec<u8> {
    // Horizontal + vertical bands at irregular positions.
    let mut out = vec![128u8; (w * h * c) as usize];
    let mut s = seed;
    for _ in 0..12 {
        let yy = lcg(&mut s) % h;
        let hh = 1 + lcg(&mut s) % (h / 4 + 1);
        let shade = (lcg(&mut s) & 0xff) as u8;
        for y in yy..(yy + hh).min(h) {
            for x in 0..w {
                let idx = ((y * w + x) * c) as usize;
                for k in 0..c as usize {
                    out[idx + k] = shade;
                }
            }
        }
    }
    for _ in 0..12 {
        let xx = lcg(&mut s) % w;
        let ww = 1 + lcg(&mut s) % (w / 4 + 1);
        let shade = (lcg(&mut s) & 0xff) as u8;
        for y in 0..h {
            for x in xx..(xx + ww).min(w) {
                let idx = ((y * w + x) * c) as usize;
                for k in 0..c as usize {
                    out[idx + k] = shade;
                }
            }
        }
    }
    out
}

fn write_ppm(path: &Path, pixels: &[u8], w: u32, h: u32, channels: u32) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let magic = if channels == 1 { "P5" } else { "P6" };
    write!(f, "{magic}\n{w} {h}\n255\n")?;
    f.write_all(pixels)?;
    f.flush()?;
    Ok(())
}

fn build_sources(tmp: &Path) -> Vec<Source> {
    let mut srcs = Vec::new();
    let mut push = |name: &str, w: u32, h: u32, c: u32, pixels: Vec<u8>| {
        let ppm = tmp.join(format!("src-{name}.pnm"));
        write_ppm(&ppm, &pixels, w, h, c).expect("write ppm");
        srcs.push(Source {
            name: name.to_string(),
            channels: c,
            ppm_path: ppm,
        });
    };

    // Dimensions chosen to exercise MCU boundaries and chroma-odd cases:
    // - 16, 32, 64, 128: clean 8×/16× aligned
    // - 17, 33, 65: 1-pixel overhang (MCU edge with 1-pixel partial)
    // - 23×29, 31×17, 47×63: fully odd, asymmetric
    // - 7×7, 9×9: below single-MCU (minimum valid JPEG)
    let dims: &[(u32, u32)] = &[
        (7, 7),
        (9, 11),
        (16, 16),
        (17, 19),
        (23, 29),
        (31, 17),
        (32, 32),
        (33, 31),
        (47, 63),
        (64, 64),
        (65, 63),
        (96, 72),
        (128, 128),
    ];

    // RGB sources
    for &(w, h) in dims {
        push(
            &format!("noise_{w}x{h}_rgb"),
            w,
            h,
            3,
            gen_noise(w, h, 3, 0x1234_5678 ^ (w as u64) << 16 ^ h as u64),
        );
    }
    for &(w, h) in &[(16, 16), (32, 32), (33, 31), (65, 63), (96, 72), (128, 128)] {
        push(
            &format!("patches_{w}x{h}_rgb"),
            w,
            h,
            3,
            gen_noise_patches(w, h, 3, 0xabcd_ef01 ^ (w as u64) << 16 ^ h as u64),
        );
    }
    for &(w, h, block) in &[(32u32, 32, 2), (33, 31, 3), (64, 64, 4), (128, 128, 8)] {
        push(
            &format!("checker_{w}x{h}_b{block}_rgb"),
            w,
            h,
            3,
            gen_checkerboard(w, h, 3, block),
        );
    }
    for &(w, h) in &[(32u32, 32), (33, 31), (64, 64), (128, 128)] {
        push(&format!("edges_{w}x{h}_rgb"), w, h, 3, gen_edges(w, h, 3));
    }
    for &(w, h) in &[(47u32, 63), (65, 63), (128, 128)] {
        push(
            &format!("bands_{w}x{h}_rgb"),
            w,
            h,
            3,
            gen_bands(w, h, 3, 0x9e37_79b9 ^ (w as u64) << 16 ^ h as u64),
        );
    }

    // Grayscale sources
    for &(w, h) in &[(16u32, 16), (17, 19), (32, 32), (33, 31), (65, 63), (128, 128)] {
        push(
            &format!("noise_{w}x{h}_gray"),
            w,
            h,
            1,
            gen_noise(w, h, 1, 0x5555_aaaa ^ (w as u64) << 16 ^ h as u64),
        );
    }
    for &(w, h) in &[(32u32, 32), (65, 63), (128, 128)] {
        push(
            &format!("patches_{w}x{h}_gray"),
            w,
            h,
            1,
            gen_noise_patches(w, h, 1, 0x1337_c0de ^ (w as u64) << 16 ^ h as u64),
        );
    }

    srcs
}

// ── Task matrix ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Task {
    tool: &'static str,
    source_idx: usize,
    // Arguments excluding the input file (and output for cjpegli).
    args: Vec<String>,
    // Serialized param description for manifest.
    desc: String,
    // True when invalid-for-zenjpeg but interesting to record (expected-fail).
    expect_zenjpeg_fail: bool,
}

fn build_tasks(sources: &[Source], quick: bool) -> Vec<Task> {
    let mut out = Vec::new();

    // ── cjpeg (libjpeg-turbo) ──────────────────────────────────────────────
    let qualities: &[u32] = if quick {
        &[25, 75, 95]
    } else {
        &[1, 5, 15, 35, 55, 75, 85, 92, 97, 100]
    };
    let rgb_subsamp: &[(&str, &str)] = if quick {
        &[("444", "1x1,1x1,1x1"), ("420", "2x2,1x1,1x1")]
    } else {
        &[
            ("444", "1x1,1x1,1x1"),
            ("422", "2x1,1x1,1x1"),
            ("420", "2x2,1x1,1x1"),
            ("440", "1x2,1x1,1x1"),
            ("411", "4x1,1x1,1x1"),
            ("mixed1", "2x2,2x1,1x2"),
        ]
    };
    let restart_choices: &[u32] = if quick { &[0] } else { &[0, 1, 8] };

    for (idx, src) in sources.iter().enumerate() {
        if src.channels == 1 {
            // Gray axis: quality × progressive × optimize × restart × dct
            let dct_methods: &[&str] = if quick { &["int"] } else { &["int", "fast", "float"] };
            for &q in qualities {
                for prog in &[false, true] {
                    for opt in &[false, true] {
                        for &r in restart_choices {
                            for &dct in dct_methods {
                                let mut args = vec![
                                    "-quality".into(),
                                    q.to_string(),
                                    "-grayscale".into(),
                                    "-dct".into(),
                                    dct.into(),
                                ];
                                if *prog {
                                    args.push("-progressive".into());
                                }
                                if *opt {
                                    args.push("-optimize".into());
                                }
                                if r > 0 {
                                    args.push("-restart".into());
                                    args.push(r.to_string());
                                }
                                let desc = format!(
                                    "q={q} prog={prog} opt={opt} rst={r} dct={dct} gray"
                                );
                                out.push(Task {
                                    tool: "cjpeg",
                                    source_idx: idx,
                                    args,
                                    desc,
                                    expect_zenjpeg_fail: false,
                                });
                            }
                        }
                    }
                }
            }
            // Arithmetic (expected fail in zenjpeg)
            if !quick {
                for &q in &[25u32, 75, 92] {
                    let args = vec![
                        "-quality".into(),
                        q.to_string(),
                        "-grayscale".into(),
                        "-arithmetic".into(),
                    ];
                    out.push(Task {
                        tool: "cjpeg",
                        source_idx: idx,
                        args,
                        desc: format!("q={q} arith gray"),
                        expect_zenjpeg_fail: true,
                    });
                }
            }
        } else {
            // RGB axis: quality × progressive × optimize × restart × subsamp
            for &q in qualities {
                for prog in &[false, true] {
                    for opt in &[false, true] {
                        for &r in restart_choices {
                            for &(sub_name, sub_spec) in rgb_subsamp {
                                let mut args = vec![
                                    "-quality".into(),
                                    q.to_string(),
                                    "-sample".into(),
                                    sub_spec.into(),
                                ];
                                if *prog {
                                    args.push("-progressive".into());
                                }
                                if *opt {
                                    args.push("-optimize".into());
                                }
                                if r > 0 {
                                    args.push("-restart".into());
                                    args.push(r.to_string());
                                }
                                let desc = format!(
                                    "q={q} prog={prog} opt={opt} rst={r} sub={sub_name}"
                                );
                                out.push(Task {
                                    tool: "cjpeg",
                                    source_idx: idx,
                                    args,
                                    desc,
                                    expect_zenjpeg_fail: false,
                                });
                            }
                        }
                    }
                }
            }
            // DCT method variants at selected qualities
            if !quick {
                for &q in &[25u32, 75, 92] {
                    for &dct in &["fast", "float"] {
                        for &(sub_name, sub_spec) in &rgb_subsamp[..2] {
                            out.push(Task {
                                tool: "cjpeg",
                                source_idx: idx,
                                args: vec![
                                    "-quality".into(),
                                    q.to_string(),
                                    "-sample".into(),
                                    sub_spec.into(),
                                    "-dct".into(),
                                    dct.into(),
                                ],
                                desc: format!("q={q} dct={dct} sub={sub_name}"),
                                expect_zenjpeg_fail: false,
                            });
                        }
                    }
                }
                // Smoothing (pre-DCT spatial smooth)
                for &q in &[55u32, 85] {
                    out.push(Task {
                        tool: "cjpeg",
                        source_idx: idx,
                        args: vec![
                            "-quality".into(),
                            q.to_string(),
                            "-smooth".into(),
                            "50".into(),
                        ],
                        desc: format!("q={q} smooth=50"),
                        expect_zenjpeg_fail: false,
                    });
                }
                // Baseline mode (forces 8-bit quant tables)
                for &q in &[55u32, 85] {
                    out.push(Task {
                        tool: "cjpeg",
                        source_idx: idx,
                        args: vec!["-quality".into(), q.to_string(), "-baseline".into()],
                        desc: format!("q={q} baseline"),
                        expect_zenjpeg_fail: false,
                    });
                }
                // -rgb (no YCbCr conversion, store as RGB-YCbCr identity)
                for &q in &[55u32, 85] {
                    out.push(Task {
                        tool: "cjpeg",
                        source_idx: idx,
                        args: vec!["-quality".into(), q.to_string(), "-rgb".into()],
                        desc: format!("q={q} rgb-mode"),
                        expect_zenjpeg_fail: false,
                    });
                }
                // Arithmetic coding
                for &q in &[25u32, 75, 92] {
                    out.push(Task {
                        tool: "cjpeg",
                        source_idx: idx,
                        args: vec![
                            "-quality".into(),
                            q.to_string(),
                            "-arithmetic".into(),
                        ],
                        desc: format!("q={q} arith"),
                        expect_zenjpeg_fail: true,
                    });
                }
            }
        }
    }

    // ── cjpegli (jpegli) ───────────────────────────────────────────────────
    //
    // Axes:
    //   --distance        quality knob (lower = higher quality)
    //   --chroma_subsampling  444/440/422/420
    //   -p 0/1/2          sequential / progressive-light / progressive-full
    //   --xyb             XYB colorspace (RGB only, forces SOF1 + 16-bit DQT)
    //   --std_quant       Annex K quant tables instead of jpegli's
    //   --noadaptive_quantization   disable AQ
    //   --fixed_code      no Huffman optimization (only with -p 0)
    let qli_distance: &[f32] = if quick {
        &[1.0, 3.0]
    } else {
        &[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    };
    let qli_subsamp: &[&str] = if quick {
        &["444", "420"]
    } else {
        &["444", "440", "422", "420"]
    };
    let qli_prog: &[u32] = if quick { &[2] } else { &[0, 1, 2] };

    for (idx, src) in sources.iter().enumerate() {
        for &d in qli_distance {
            for &sub in qli_subsamp {
                // cjpegli ignores subsampling for grayscale; skip redundant combos
                if src.channels == 1 && sub != "444" {
                    continue;
                }
                for &p in qli_prog {
                    // Core combo (YCbCr)
                    let args = vec![
                        "-d".into(),
                        format!("{d}"),
                        format!("--chroma_subsampling={sub}"),
                        "-p".into(),
                        p.to_string(),
                    ];
                    out.push(Task {
                        tool: "cjpegli",
                        source_idx: idx,
                        args,
                        desc: format!("d={d} sub={sub} p={p}"),
                        expect_zenjpeg_fail: false,
                    });

                    // XYB variant (RGB sources only — XYB is inherently 3-channel)
                    if src.channels == 3 {
                        out.push(Task {
                            tool: "cjpegli",
                            source_idx: idx,
                            args: vec![
                                "-d".into(),
                                format!("{d}"),
                                format!("--chroma_subsampling={sub}"),
                                "-p".into(),
                                p.to_string(),
                                "--xyb".into(),
                            ],
                            desc: format!("d={d} sub={sub} p={p} xyb"),
                            expect_zenjpeg_fail: false,
                        });
                    }
                }
            }
        }

        // ── Extras: std_quant, noadaptive, fixed_code ──────────────────────
        // Applied at a reduced slice to keep task count manageable.
        if !quick {
            let extra_distances: &[f32] = &[0.5, 1.0, 3.0];
            // std_quant (Annex K tables) at sub=444, p=2
            for &d in extra_distances {
                out.push(Task {
                    tool: "cjpegli",
                    source_idx: idx,
                    args: vec![
                        "-d".into(),
                        format!("{d}"),
                        "--chroma_subsampling=444".into(),
                        "-p".into(),
                        "2".into(),
                        "--std_quant".into(),
                    ],
                    desc: format!("d={d} sub=444 p=2 std_quant"),
                    expect_zenjpeg_fail: false,
                });
                // XYB + std_quant (RGB only)
                if src.channels == 3 {
                    out.push(Task {
                        tool: "cjpegli",
                        source_idx: idx,
                        args: vec![
                            "-d".into(),
                            format!("{d}"),
                            "--chroma_subsampling=444".into(),
                            "-p".into(),
                            "2".into(),
                            "--xyb".into(),
                            "--std_quant".into(),
                        ],
                        desc: format!("d={d} sub=444 p=2 xyb std_quant"),
                        expect_zenjpeg_fail: false,
                    });
                }
            }
            // noadaptive_quantization across sub + p
            for &d in extra_distances {
                for &sub in &["444", "420"] {
                    if src.channels == 1 && sub != "444" {
                        continue;
                    }
                    for &p in &[0u32, 2] {
                        out.push(Task {
                            tool: "cjpegli",
                            source_idx: idx,
                            args: vec![
                                "-d".into(),
                                format!("{d}"),
                                format!("--chroma_subsampling={sub}"),
                                "-p".into(),
                                p.to_string(),
                                "--noadaptive_quantization".into(),
                            ],
                            desc: format!("d={d} sub={sub} p={p} noaq"),
                            expect_zenjpeg_fail: false,
                        });
                    }
                }
            }
            // fixed_code requires -p 0
            for &d in extra_distances {
                for &sub in &["444", "420"] {
                    if src.channels == 1 && sub != "444" {
                        continue;
                    }
                    out.push(Task {
                        tool: "cjpegli",
                        source_idx: idx,
                        args: vec![
                            "-d".into(),
                            format!("{d}"),
                            format!("--chroma_subsampling={sub}"),
                            "-p".into(),
                            "0".into(),
                            "--fixed_code".into(),
                        ],
                        desc: format!("d={d} sub={sub} p=0 fixed_code"),
                        expect_zenjpeg_fail: false,
                    });
                }
            }
        }
    }

    out
}

// ── Task execution ─────────────────────────────────────────────────────────

struct TaskOutput {
    hash: String,
    bytes: Vec<u8>,
    tool: &'static str,
    source_name: String,
    desc: String,
    expect_zenjpeg_fail: bool,
}

fn short_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    let mut s = String::with_capacity(32);
    for b in &d[..16] {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn run_cjpeg(ppm: &Path, args: &[String]) -> Option<Vec<u8>> {
    let out = Command::new(CJPEG)
        .args(args)
        .arg(ppm)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.len() < 4 {
        return None;
    }
    if out.stdout[0] != 0xFF || out.stdout[1] != 0xD8 {
        return None;
    }
    Some(out.stdout)
}

fn run_cjpegli(ppm: &Path, args: &[String], worker_id: usize) -> Option<Vec<u8>> {
    let tmp = std::env::temp_dir().join(format!(
        "perm-gen-{}-w{}.jpg",
        std::process::id(),
        worker_id
    ));
    let mut cmd = Command::new(CJPEGLI);
    cmd.arg(ppm)
        .arg(&tmp)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let status = cmd.status().ok()?;
    if !status.success() {
        let _ = fs::remove_file(&tmp);
        return None;
    }
    let bytes = fs::read(&tmp).ok();
    let _ = fs::remove_file(&tmp);
    let b = bytes?;
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    Some(b)
}

fn run_task(task: &Task, source: &Source, worker_id: usize) -> Option<TaskOutput> {
    let bytes = match task.tool {
        "cjpeg" => run_cjpeg(&source.ppm_path, &task.args)?,
        "cjpegli" => run_cjpegli(&source.ppm_path, &task.args, worker_id)?,
        _ => return None,
    };
    let hash = short_hash(&bytes);
    Some(TaskOutput {
        hash,
        bytes,
        tool: task.tool,
        source_name: source.name.clone(),
        desc: task.desc.clone(),
        expect_zenjpeg_fail: task.expect_zenjpeg_fail,
    })
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let out_dir = PathBuf::from(env::var("ZENJPEG_PERM_OUT").unwrap_or_else(|_| DEFAULT_OUT.into()));
    let cap_mb: u64 = env::var("ZENJPEG_PERM_CAP_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6144);
    let cap_bytes = cap_mb * 1024 * 1024;
    let quick = env::var("ZENJPEG_PERM_QUICK").is_ok();

    fs::create_dir_all(&out_dir).expect("create out dir");

    println!("=== zenjpeg permutation corpus generator ===");
    println!("output:     {}", out_dir.display());
    println!("cap:        {cap_mb} MiB");
    println!("mode:       {}", if quick { "quick" } else { "full" });
    println!("cjpeg:      {CJPEG}");
    println!("cjpegli:    {CJPEGLI}");

    let tmp = std::env::temp_dir().join(format!("perm-gen-src-{}", std::process::id()));
    fs::create_dir_all(&tmp).expect("create tmp");

    let t_srcs = Instant::now();
    let sources = build_sources(&tmp);
    println!(
        "sources:    {} (generated in {:.2}s)",
        sources.len(),
        t_srcs.elapsed().as_secs_f32()
    );

    let tasks = build_tasks(&sources, quick);
    println!("tasks:      {}", tasks.len());

    let (tx, rx) = mpsc::sync_channel::<TaskOutput>(256);

    let writer_out_dir = out_dir.clone();
    let total_bytes = AtomicU64::new(0);
    let writer = thread::spawn(move || {
        let manifest_path = writer_out_dir.join("manifest.tsv");
        let mut mf = BufWriter::new(File::create(&manifest_path).expect("manifest"));
        writeln!(
            mf,
            "hash\trel_path\ttool\tsource\tparams\tbytes\texpect_zenjpeg_fail"
        )
        .ok();
        let mut seen: HashSet<String> = HashSet::new();
        let mut n_written: u64 = 0;
        let mut n_dup: u64 = 0;
        let mut bytes_total: u64 = 0;
        let mut capped = false;
        while let Ok(r) = rx.recv() {
            if capped {
                continue;
            }
            if !seen.insert(r.hash.clone()) {
                n_dup += 1;
                continue;
            }
            let sub = &r.hash[..2];
            let sub_dir = writer_out_dir.join(sub);
            fs::create_dir_all(&sub_dir).ok();
            let rel_path = format!("{sub}/{}.jpg", r.hash);
            let full_path = writer_out_dir.join(&rel_path);
            if fs::write(&full_path, &r.bytes).is_err() {
                continue;
            }
            bytes_total += r.bytes.len() as u64;
            n_written += 1;
            writeln!(
                mf,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.hash,
                rel_path,
                r.tool,
                r.source_name,
                r.desc,
                r.bytes.len(),
                r.expect_zenjpeg_fail as u8
            )
            .ok();
            if n_written % 500 == 0 {
                println!(
                    "  wrote {n_written} unique, {n_dup} dup, {:.1} MiB so far",
                    bytes_total as f64 / 1024.0 / 1024.0
                );
                mf.flush().ok();
            }
            if bytes_total > cap_bytes {
                println!("!! cap hit at {bytes_total} bytes — stopping dedup writes");
                capped = true;
            }
        }
        mf.flush().ok();
        (n_written, n_dup, bytes_total)
    });

    let n_tasks = tasks.len();
    let t_run = Instant::now();
    let done = AtomicU64::new(0);
    tasks.par_iter().for_each_with(tx.clone(), |tx, task| {
        let src = &sources[task.source_idx];
        let worker = rayon::current_thread_index().unwrap_or(0);
        if let Some(out) = run_task(task, src, worker) {
            let b = out.bytes.len() as u64;
            let _ = tx.send(out);
            total_bytes.fetch_add(b, Ordering::Relaxed);
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 1000 == 0 {
            println!(
                "  task {d}/{n_tasks}  (~{:.0} tasks/s)",
                d as f64 / t_run.elapsed().as_secs_f64()
            );
        }
    });
    drop(tx);

    let (n_written, n_dup, bytes_total) = writer.join().expect("writer");
    let elapsed = t_run.elapsed();

    println!();
    println!("=== done ===");
    println!("elapsed:    {:.1}s", elapsed.as_secs_f64());
    println!("tasks:      {n_tasks}");
    println!("written:    {n_written} unique");
    println!("duplicates: {n_dup}");
    println!(
        "total size: {:.1} MiB",
        bytes_total as f64 / 1024.0 / 1024.0
    );
    println!(
        "avg size:   {:.1} KiB",
        if n_written > 0 {
            bytes_total as f64 / n_written as f64 / 1024.0
        } else {
            0.0
        }
    );

    // Cleanup tmp sources
    let _ = fs::remove_dir_all(&tmp);
}
