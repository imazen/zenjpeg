//! perf_wall — in-memory wall-clock matrix for performance A/B work.
//!
//! Prints `op\tmode\tsize\tcontent\tvariant\titer\tns\tbytes` rows plus a
//! `median` row per cell. Self-checks: repeated encodes are byte-identical;
//! every decode equals the ST reference pixels. Writes the produced JPEGs to
//! <outdir> so callgrind decode runs can exclude fixture encoding.
//!
//! Usage: perf_wall <outdir> [--iters N] [--allow-missing-corpus]
//!
//! The CLIC photo cell requires the codec-corpus cache; when it is missing the
//! run exits nonzero unless `--allow-missing-corpus` is passed (reduced scope
//! is a caller decision, never silent). The header line reports both hardware
//! parallelism and the effective rayon worker count.
//!
//! Matrix: sizes 64², 256², 1024², 2048², 4096²; content noise-patches
//! (photo-like DCT mix), flat-chroma (DC-only blocks), CLIC photo; baseline
//! and progressive JPEGs; ST and MT (--features parallel) encode and decode.
//! MT decode rows use num_threads(0) = auto.

use enough::Unstoppable;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn gen_noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let (bx, by) = ((x / 8) as u32, (y / 8) as u32);
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let mut hh = (x as u32)
                .wrapping_mul(374761393)
                .wrapping_add((y as u32).wrapping_mul(668265263));
            hh = (hh ^ (hh >> 13)).wrapping_mul(1274126177);
            let noise = (hh >> 24) as u8;
            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / w.max(1)) as u8;
                    data[idx + 1] = ((y * 255) / h.max(1)) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) {
                        200u8
                    } else {
                        55u8
                    };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255 - edge;
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }
    data
}

fn gen_flat_chroma(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = match (y / 16) % 4 {
                0 => (200i32, 90, 70),
                1 => (70, 180, 90),
                2 => (80, 90, 210),
                _ => (190, 190, 70),
            };
            let luma = ((x * 7 + y * 3) % 90) as i32 / 3;
            rgb[i] = (r + luma).clamp(0, 255) as u8;
            rgb[i + 1] = (g + luma).clamp(0, 255) as u8;
            rgb[i + 2] = (b + luma).clamp(0, 255) as u8;
        }
    }
    rgb
}

fn load_photo() -> Result<(Vec<u8>, u32, u32), String> {
    let corpus = codec_corpus::Corpus::new().map_err(|e| format!("corpus init failed: {e}"))?;
    let dir = corpus
        .get("clic2025/final-test")
        .map_err(|e| format!("clic2025/final-test unavailable: {e}"))?;
    let mut pngs: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir failed: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    pngs.sort();
    for p in pngs {
        let file = std::fs::File::open(&p).map_err(|e| format!("open {p:?}: {e}"))?;
        let dec = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = dec
            .read_info()
            .map_err(|e| format!("png header {p:?}: {e}"))?;
        let mut buf = vec![0u8; reader.output_buffer_size().ok_or("png size")?];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| format!("png decode {p:?}: {e}"))?;
        if info.width < 2048 || info.height < 2048 {
            continue;
        }
        let n = (info.width * info.height) as usize;
        let rgb: Vec<u8> = match info.color_type {
            png::ColorType::Rgb => buf[..n * 3].to_vec(),
            png::ColorType::Rgba => buf[..n * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect(),
            _ => continue,
        };
        return Ok((rgb, info.width, info.height));
    }
    Err("no >=2048px RGB/RGBA PNG found in clic2025/final-test".into())
}

fn crop_rgb8(src: &[u8], sw: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        out[y * w * 3..(y + 1) * w * 3].copy_from_slice(&src[y * sw * 3..y * sw * 3 + w * 3]);
    }
    out
}

fn encode_once(config: &EncoderConfig, pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(pixels, Unstoppable).unwrap();
    enc.finish().unwrap()
}

struct Cell {
    op: &'static str,
    mode: &'static str,
    size: String,
    content: String,
    variant: &'static str,
    times: Vec<u128>,
    bytes: usize,
}

fn median(times: &[u128]) -> u128 {
    let mut t = times.to_vec();
    t.sort();
    t[t.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let outdir = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .map(Path::new)
        .expect("usage: perf_wall <outdir> [--iters N] [--allow-missing-corpus]");
    std::fs::create_dir_all(outdir).unwrap();
    let allow_missing_corpus = args.iter().any(|a| a == "--allow-missing-corpus");
    let iters: usize = args
        .iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    eprintln!(
        "# perf_wall parallel={} hw_parallelism={} rayon_pool_workers={} iters={}",
        cfg!(feature = "parallel"),
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        rayon::current_num_threads(),
        iters
    );

    // ---- input matrix -----------------------------------------------------
    let photo = match load_photo() {
        Ok(p) => Some(p),
        Err(e) if allow_missing_corpus => {
            eprintln!("# photo source unavailable ({e}); photo rows explicitly skipped");
            None
        }
        Err(e) => {
            eprintln!("perf_wall requires the codec corpus photo source: {e}");
            eprintln!("pass --allow-missing-corpus to run the reduced matrix");
            std::process::exit(2);
        }
    };
    struct Input {
        size: (u32, u32),
        content: String,
        rgb: Vec<u8>,
    }
    let mut inputs: Vec<Input> = Vec::new();
    for &s in &[64usize, 256, 1024, 2048, 4096] {
        inputs.push(Input {
            size: (s as u32, s as u32),
            content: "noise".into(),
            rgb: gen_noise_patches(s, s),
        });
    }
    inputs.push(Input {
        size: (1024, 1024),
        content: "flatchroma".into(),
        rgb: gen_flat_chroma(1024, 1024),
    });
    if let Some((rgb, sw, _sh)) = &photo {
        inputs.push(Input {
            size: (2048, 2048),
            content: "photo".into(),
            rgb: crop_rgb8(rgb, *sw as usize, 2048, 2048),
        });
        inputs.push(Input {
            size: (1024, 1024),
            content: "photo".into(),
            rgb: crop_rgb8(rgb, *sw as usize, 1024, 1024),
        });
    }

    let modes: Vec<(&str, bool)> = if cfg!(feature = "parallel") {
        vec![("st", false), ("mt", true)]
    } else {
        vec![("st", false)]
    };

    println!("op\tmode\tsize\tcontent\tvariant\titer\tns\tbytes");
    let mut cells: Vec<Cell> = Vec::new();

    for inp in &inputs {
        let (w, h) = inp.size;
        // Encode both baseline and progressive fixtures (ST; the JPEG itself
        // is the decode input — decoder doesn't care which mode produced it).
        let cfg_base = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .restart_mcu_rows(4);
        let cfg_prog = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(true)
            .restart_mcu_rows(4);
        let jpeg_base = encode_once(&cfg_base, &inp.rgb, w, h);
        let jpeg_prog = encode_once(&cfg_prog, &inp.rgb, w, h);
        std::fs::write(
            outdir.join(format!("fixture_{}x{}_{}_base.jpg", w, h, inp.content)),
            &jpeg_base,
        )
        .unwrap();
        std::fs::write(
            outdir.join(format!("fixture_{}x{}_{}_prog.jpg", w, h, inp.content)),
            &jpeg_prog,
        )
        .unwrap();

        // ST reference pixels for decode self-checks (baseline + progressive).
        let st_ref = Decoder::new()
            .num_threads(1)
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg_base, Unstoppable)
            .unwrap();
        let st_ref_px = st_ref.pixels_u8().unwrap().to_vec();
        let st_ref_prog_px = Decoder::new()
            .num_threads(1)
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg_prog, Unstoppable)
            .unwrap()
            .pixels_u8()
            .unwrap()
            .to_vec();

        for (mode, mt) in &modes {
            let enc_cfg = cfg_base.clone();
            #[cfg(feature = "parallel")]
            let enc_cfg = if *mt {
                enc_cfg.parallel(zenjpeg::encoder::ParallelEncoding::Auto)
            } else {
                enc_cfg
            };
            #[cfg(not(feature = "parallel"))]
            let _ = mt;

            // warmup
            let warm = encode_once(&enc_cfg, &inp.rgb, w, h);
            let warm_dec = Decoder::new()
                .num_threads(if *mt { 0 } else { 1 })
                .output_format(PixelFormat::Rgb)
                .decode(&jpeg_base, Unstoppable)
                .unwrap();
            assert_eq!(
                st_ref_px,
                warm_dec.pixels_u8().unwrap(),
                "decode warmup {mode} {w}x{h}"
            );
            eprintln!(
                "# {w}x{h} {} {mode} enc_equals_st={} jpeg={}B",
                inp.content,
                warm == jpeg_base,
                jpeg_base.len()
            );

            let mut enc = Cell {
                op: "encode",
                mode,
                size: format!("{w}x{h}"),
                content: inp.content.clone(),
                variant: "base",
                times: vec![],
                bytes: warm.len(),
            };
            let mut dec = Cell {
                op: "decode",
                mode,
                size: format!("{w}x{h}"),
                content: inp.content.clone(),
                variant: "base",
                times: vec![],
                bytes: st_ref_px.len(),
            };
            let mut dec_prog = Cell {
                op: "decode",
                mode,
                size: format!("{w}x{h}"),
                content: inp.content.clone(),
                variant: "prog",
                times: vec![],
                bytes: st_ref_px.len(),
            };
            let decoder = Decoder::new()
                .num_threads(if *mt { 0 } else { 1 })
                .output_format(PixelFormat::Rgb);
            for it in 1..=iters {
                let t = Instant::now();
                let e = black_box(encode_once(&enc_cfg, black_box(&inp.rgb), w, h));
                let ns = t.elapsed().as_nanos();
                assert_eq!(e, jpeg_base, "encode {mode} vs ST {w}x{h} it{it}");
                enc.times.push(ns);
                enc.bytes = e.len();
                println!(
                    "encode\t{mode}\t{w}x{h}\t{}\tbase\t{it}\t{ns}\t{}",
                    inp.content,
                    e.len()
                );

                let t = Instant::now();
                let d = black_box(decoder.decode(black_box(&jpeg_base), Unstoppable).unwrap());
                let ns = t.elapsed().as_nanos();
                assert_eq!(
                    st_ref_px,
                    d.pixels_u8().unwrap(),
                    "decode {mode} {w}x{h} it{it}"
                );
                dec.times.push(ns);
                println!(
                    "decode\t{mode}\t{w}x{h}\t{}\tbase\t{it}\t{ns}\t{}",
                    inp.content,
                    d.pixels_u8().unwrap().len()
                );

                // Progressive decode at >=1024 only (keeps total runtime sane).
                if w >= 1024 {
                    let t = Instant::now();
                    let d = black_box(decoder.decode(black_box(&jpeg_prog), Unstoppable).unwrap());
                    let ns = t.elapsed().as_nanos();
                    assert_eq!(
                        st_ref_prog_px,
                        d.pixels_u8().unwrap(),
                        "decode prog {mode} {w}x{h} it{it}"
                    );
                    dec_prog.times.push(ns);
                    println!(
                        "decode\t{mode}\t{w}x{h}\t{}\tprog\t{it}\t{ns}\t{}",
                        inp.content,
                        d.pixels_u8().unwrap().len()
                    );
                }
            }
            cells.push(enc);
            cells.push(dec);
            if w >= 1024 {
                cells.push(dec_prog);
            }
        }
    }

    println!("\n# medians");
    for c in &cells {
        println!(
            "{}\t{}\t{}\t{}\t{}\tmedian\t{}\t{}",
            c.op,
            c.mode,
            c.size,
            c.content,
            c.variant,
            median(&c.times),
            c.bytes
        );
    }
}
