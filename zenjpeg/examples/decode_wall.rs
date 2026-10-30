//! decode_wall — wall-clock decode A/B: zenjpeg vs libjpeg-turbo (mozjpeg-sys, NASM).
//!
//! Interleaves the decoders each rep so ambient load hits both lanes equally;
//! reports per-decoder median ms and MPix/s, plus the zen/moz time ratio
//! (>1.0 = zenjpeg slower).
//!
//! Usage:
//!   decode_wall <file.jpg> [--reps N]
//!   decode_wall <file.png> [--reps N] [--q 85] [--sub 420|444] [--prog]
//!   ... [--format rgb|bgr]   zenjpeg output format (mozjpeg-nasm stays RGB)
//!
//! PNG input is encoded with mozjpeg-rs (BaselineBalanced / ProgressiveSmallest)
//! first, so the fixture matches decode_zenbench.rs.

use enough::Unstoppable;
use std::hint::black_box;
use std::time::Instant;
use zenjpeg::decode::{ChromaUpsampling, Decoder};
use zenjpeg::decoder::PixelFormat;

fn decode_libjpeg_turbo(jpeg: &[u8]) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);
        jpeg_mem_src(&mut cinfo, jpeg.as_ptr(), jpeg.len() as _);
        jpeg_read_header(&mut cinfo, true as boolean);
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);
        let (w, h) = (cinfo.output_width, cinfo.output_height);
        let stride = w as usize * cinfo.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while cinfo.output_scanline < h {
            let off = cinfo.output_scanline as usize * stride;
            let mut p = out[off..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut p, 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
        out
    }
}

fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for c in src.as_chunks::<4>().0 {
                rgb.extend_from_slice(&c[..3]);
            }
            rgb
        }
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn median(v: &mut [u128]) -> u128 {
    v.sort_unstable();
    if v.is_empty() {
        return 0;
    }
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: decode_wall <file.(jpg|png)> [--reps N] [--q Q] [--sub 420|444] [--prog] [--tile WxH] [--save out.jpg] [--only moz|zen|zen-box] [--format rgb|bgr]"
        );
        std::process::exit(1);
    }
    let path = std::path::Path::new(&args[1]);
    let mut reps = 30usize;
    let mut quality = 85u8;
    let mut sub = mozjpeg_rs::Subsampling::S420;
    let mut restart_interval = 0u16;
    let mut progressive = false;
    let mut tile: Option<(u32, u32)> = None;
    let mut save: Option<String> = None;
    let mut only: Option<String> = None;
    let mut format = PixelFormat::Rgb;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--reps" => {
                reps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--q" => {
                quality = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--sub" => {
                sub = match args[i + 1].as_str() {
                    "420" => mozjpeg_rs::Subsampling::S420,
                    "422" => mozjpeg_rs::Subsampling::S422,
                    "444" => mozjpeg_rs::Subsampling::S444,
                    "440" => mozjpeg_rs::Subsampling::S440,
                    "gray" | "400" => mozjpeg_rs::Subsampling::Gray,
                    other => {
                        eprintln!("unknown subsampling {other}");
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--prog" => {
                progressive = true;
                i += 1;
            }
            "--rst" => {
                restart_interval = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--tile" => {
                let (w, h) = args[i + 1].split_once('x').expect("--tile WxH");
                tile = Some((w.parse().unwrap(), h.parse().unwrap()));
                i += 2;
            }
            "--save" => {
                save = Some(args[i + 1].clone());
                i += 2;
            }
            "--only" => {
                only = Some(args[i + 1].clone());
                i += 2;
            }
            "--format" => {
                format = match args[i + 1].as_str() {
                    "rgb" => PixelFormat::Rgb,
                    "bgr" => PixelFormat::Bgr,
                    other => {
                        eprintln!("unknown format {other}");
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }

    let jpeg: Vec<u8> = if path.extension().is_some_and(|e| e == "png") {
        let (mut pix, w, h) = load_png_rgb(path).expect("png decode");
        let (tw, th) = tile.unwrap_or((w, h));
        if (tw, th) != (w, h) {
            let mut out = vec![0u8; (tw * th * 3) as usize];
            for y in 0..th {
                for x in 0..tw {
                    let si = ((y % h) * w + (x % w)) as usize * 3;
                    let di = (y * tw + x) as usize * 3;
                    out[di..di + 3].copy_from_slice(&pix[si..si + 3]);
                }
            }
            pix = out;
        }
        let preset = if progressive {
            mozjpeg_rs::Preset::ProgressiveSmallest
        } else {
            mozjpeg_rs::Preset::BaselineBalanced
        };
        mozjpeg_rs::Encoder::new(preset)
            .quality(quality)
            .subsampling(sub)
            .restart_interval(restart_interval)
            .encode_rgb(&pix, tw, th)
            .expect("mozjpeg encode")
    } else {
        std::fs::read(path).expect("read jpeg")
    };

    if let Some(p) = &save {
        std::fs::write(p, &jpeg).expect("save fixture");
    }

    // Dimensions + decode sanity check (outputs must agree).
    let reader = Decoder::new().scanline_reader(&jpeg).expect("scanline");
    let (w, h) = (reader.width(), reader.height());
    drop(reader);
    let moz = decode_libjpeg_turbo(&jpeg);
    let mut zen = Decoder::new()
        .output_format(format)
        .decode(&jpeg, Unstoppable)
        .expect("zen decode")
        .into_pixels_u8()
        .expect("pixels");
    if format == PixelFormat::Bgr {
        let unswapped = moz
            .iter()
            .zip(zen.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        eprintln!("bgr output read as rgb: max|zen-moz|={unswapped}");
        for px in zen.as_chunks_mut::<3>().0 {
            px.swap(0, 2);
        }
    }
    let max_diff = moz
        .iter()
        .zip(zen.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    let n = (w as usize * h as usize) as f64;
    eprintln!(
        "{}: {}x{} {} bytes, max|zen-moz|={}",
        args[1],
        w,
        h,
        jpeg.len(),
        max_diff
    );

    let lanes: Vec<(&str, Box<dyn Fn(&[u8])>)> = vec![
        (
            "mozjpeg-nasm",
            Box::new(|d| {
                black_box(decode_libjpeg_turbo(d));
            }),
        ),
        (
            "zen-default",
            Box::new(|d| {
                black_box(
                    Decoder::new()
                        .output_format(format)
                        .decode(d, Unstoppable)
                        .expect("zen decode"),
                );
            }),
        ),
        (
            "zune-jpeg",
            Box::new(|d| {
                use zune_core::options::DecoderOptions;
                let options = DecoderOptions::default()
                    .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB);
                let mut decoder =
                    zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(d), options);
                black_box(decoder.decode().expect("zune decode"));
            }),
        ),
        (
            "zen-box",
            Box::new(|d| {
                black_box(
                    Decoder::new()
                        .output_format(format)
                        .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
                        .decode(d, Unstoppable)
                        .expect("zen decode"),
                );
            }),
        ),
    ];
    let lanes: Vec<_> = match &only {
        Some(name) => lanes
            .into_iter()
            .filter(|(n, _)| *n == name || (name == "moz" && *n == "mozjpeg-nasm"))
            .collect(),
        None => lanes,
    };
    if lanes.is_empty() {
        eprintln!("--only matched no lanes");
        std::process::exit(1);
    }

    let mut times: Vec<Vec<u128>> = lanes.iter().map(|_| Vec::with_capacity(reps)).collect();
    // Warmup
    for (_, f) in &lanes {
        for _ in 0..3 {
            f(&jpeg);
        }
    }
    for _ in 0..reps {
        for (li, (_, f)) in lanes.iter().enumerate() {
            let t = Instant::now();
            f(black_box(&jpeg));
            times[li].push(t.elapsed().as_nanos());
        }
    }

    let mut moz_med = 0u128;
    for (li, (name, _)) in lanes.iter().enumerate() {
        let med = median(&mut times[li]);
        if li == 0 {
            moz_med = med;
        }
        let mpix_s = n / (med as f64 / 1e9) / 1e6;
        eprintln!(
            "{name:>14}: {:8.3} ms  {:7.1} MPix/s  ratio-vs-moz {:.3}",
            med as f64 / 1e6,
            mpix_s,
            med as f64 / moz_med.max(1) as f64
        );
    }
}
