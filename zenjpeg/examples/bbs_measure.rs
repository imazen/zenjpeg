//! BBS (Block Boundary Score) measurement CLI.
//!
//! Measures the cross-seam gradient distortion introduced by JPEG encoding.
//! See `zenjpeg::metrics::bbs` for the algorithm and `issue #91` for the
//! motivation.
//!
//! Two modes:
//!
//! 1. **Encode + measure:** give it a lossless PNG and one or more quality
//!    levels. The tool encodes with the default zenjpeg YCbCr config, decodes
//!    with zune-jpeg, and reports BBS per quality.
//!
//!    ```bash
//!    cargo run --release --example bbs_measure -- \
//!        --original input.png --quality 50 --quality 75 --quality 85 --quality 95
//!    ```
//!
//! 2. **Compare pre-encoded JPEG vs reference:** give it a lossless original
//!    and a reconstructed (JPEG-decoded or PNG) image.
//!
//!    ```bash
//!    cargo run --release --example bbs_measure -- \
//!        --original input.png --reconstructed out.jpg
//!    ```
//!
//! Add `--csv out.csv` to append a machine-readable row per measurement.
//! Omit `--csv` for human-readable tabular output.

use enough::Unstoppable;
use imgref::ImgRef;
use rgb::RGB;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::metrics::{BbsResult, bbs_rgb8};

struct Args {
    original: PathBuf,
    reconstructed: Option<PathBuf>,
    qualities: Vec<u8>,
    csv: Option<PathBuf>,
    subsampling: ChromaSubsampling,
    progressive: bool,
    label: Option<String>,
}

fn print_usage_and_exit() -> ! {
    eprintln!("Usage: bbs_measure --original <file.png> [--reconstructed <file.jpg>] \\");
    eprintln!("                     [--quality Q]... [--csv out.csv] [--444|--422|--420]");
    eprintln!("                     [--progressive] [--label NAME]");
    eprintln!();
    eprintln!("With --reconstructed: measure BBS of that file vs original.");
    eprintln!("Without: encode the original with zenjpeg at each --quality and measure.");
    eprintln!("Default qualities: 50 75 85 95");
    eprintln!("Default subsampling: 4:2:0 (chroma quarter)");
    std::process::exit(2);
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut original: Option<PathBuf> = None;
    let mut reconstructed: Option<PathBuf> = None;
    let mut qualities: Vec<u8> = Vec::new();
    let mut csv: Option<PathBuf> = None;
    let mut subsampling = ChromaSubsampling::Quarter;
    let mut progressive = false;
    let mut label: Option<String> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--original" => {
                i += 1;
                original = Some(PathBuf::from(&argv[i]));
            }
            "--reconstructed" => {
                i += 1;
                reconstructed = Some(PathBuf::from(&argv[i]));
            }
            "--quality" => {
                i += 1;
                qualities.push(argv[i].parse().expect("bad --quality"));
            }
            "--csv" => {
                i += 1;
                csv = Some(PathBuf::from(&argv[i]));
            }
            "--label" => {
                i += 1;
                label = Some(argv[i].clone());
            }
            "--444" => subsampling = ChromaSubsampling::None,
            "--422" => subsampling = ChromaSubsampling::HalfHorizontal,
            "--420" => subsampling = ChromaSubsampling::Quarter,
            "--progressive" => progressive = true,
            "--help" | "-h" => print_usage_and_exit(),
            other => {
                eprintln!("unknown flag: {}", other);
                print_usage_and_exit();
            }
        }
        i += 1;
    }

    let Some(original) = original else {
        eprintln!("error: --original is required");
        print_usage_and_exit();
    };
    if qualities.is_empty() && reconstructed.is_none() {
        qualities = vec![50, 75, 85, 95];
    }

    Args {
        original,
        reconstructed,
        qualities,
        csv,
        subsampling,
        progressive,
        label,
    }
}

/// Load any image path that `image` crate can read (PNG, JPEG, BMP, etc.),
/// into a flat RGB8 buffer + (width, height).
fn load_rgb(path: &Path) -> (Vec<RGB<u8>>, usize, usize) {
    let img =
        image::open(path).unwrap_or_else(|e| panic!("failed to open {}: {}", path.display(), e));
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut buf = Vec::with_capacity(w * h);
    for row in rgb.rows() {
        for p in row {
            buf.push(RGB {
                r: p.0[0],
                g: p.0[1],
                b: p.0[2],
            });
        }
    }
    (buf, w, h)
}

/// Decode a JPEG byte stream to flat RGB u8 using zune-jpeg (no ICC applied —
/// matches the "looking at pixels on screen" view).
fn decode_jpeg_rgb(data: &[u8]) -> (Vec<RGB<u8>>, usize, usize) {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder.decode().expect("decode failed");
    let info = decoder.info().expect("jpeg info missing");
    let (w, h) = (info.width as usize, info.height as usize);
    let rgb: Vec<RGB<u8>> = pixels
        .chunks_exact(3)
        .map(|c| RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    assert_eq!(rgb.len(), w * h);
    (rgb, w, h)
}

/// Encode the given flat RGB u8 buffer to JPEG using the default zenjpeg
/// YCbCr config at `quality` with `subsampling`.
fn encode_jpeg(
    rgb: &[RGB<u8>],
    w: usize,
    h: usize,
    quality: u8,
    subsampling: ChromaSubsampling,
    progressive: bool,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, subsampling).progressive(progressive);
    let mut encoder = config
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder build failed");
    // encode_from_bytes takes interleaved bytes; flatten RGB<u8> to &[u8].
    let bytes: &[u8] = bytemuck::cast_slice(rgb);
    encoder
        .push_packed(bytes, Unstoppable)
        .expect("push failed");
    encoder.finish().expect("encode finish failed")
}

fn print_row_header(human: bool) {
    if human {
        println!(
            "{:<28} {:>4} {:>9} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7}",
            "image",
            "Q",
            "bytes",
            "bpp",
            "bbs_total",
            "bbs_Y",
            "bbs_Cb",
            "bbs_Cr",
            "interior",
            "ratio",
            "enc_ms"
        );
    }
}

fn print_row(
    human: bool,
    name: &str,
    quality: u8,
    bytes: usize,
    bpp: f64,
    bbs: &BbsResult,
    enc_ms: f64,
) {
    if human {
        let ratio = bbs
            .interior_ratio()
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<28} {:>4} {:>9} {:>6.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10} {:>6.1}",
            truncate(name, 28),
            quality,
            bytes,
            bpp,
            bbs.total,
            bbs.per_channel[0],
            bbs.per_channel[1],
            bbs.per_channel[2],
            bbs.interior_total,
            ratio,
            enc_ms,
        );
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[s.len() - max..]
    }
}

fn csv_header(file: &mut File) -> std::io::Result<()> {
    writeln!(
        file,
        "image,quality,bytes,bpp,bbs_total,bbs_y,bbs_cb,bbs_cr,bbs_horizontal,bbs_vertical,interior_total,interior_ratio,seam_pixels,interior_pixels,enc_ms"
    )
}

fn csv_row(
    file: &mut File,
    name: &str,
    quality: u8,
    bytes: usize,
    bpp: f64,
    bbs: &BbsResult,
    enc_ms: f64,
) -> std::io::Result<()> {
    let ratio = bbs
        .interior_ratio()
        .map(|v| format!("{:.6}", v))
        .unwrap_or_else(|| "".into());
    writeln!(
        file,
        "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.3}",
        name,
        quality,
        bytes,
        bpp,
        bbs.total,
        bbs.per_channel[0],
        bbs.per_channel[1],
        bbs.per_channel[2],
        bbs.horizontal_total,
        bbs.vertical_total,
        bbs.interior_total,
        ratio,
        bbs.seam_pixels,
        bbs.interior_pixels,
        enc_ms,
    )
}

fn main() {
    let args = parse_args();

    let (orig_rgb, w, h) = load_rgb(&args.original);
    let orig_img: ImgRef<'_, RGB<u8>> = ImgRef::new(&orig_rgb, w, h);
    let pixels = (w * h) as f64;

    let image_name = args.label.clone().unwrap_or_else(|| {
        args.original
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| args.original.display().to_string())
    });

    eprintln!(
        "loaded {} ({}x{}, {:.2} MP)",
        args.original.display(),
        w,
        h,
        pixels / 1_000_000.0
    );

    // Open CSV if requested, write header if new.
    let mut csv_file: Option<File> = None;
    if let Some(path) = &args.csv {
        let new_file = !path.exists();
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("cannot open --csv for writing");
        if new_file {
            csv_header(&mut f).expect("csv header write");
        }
        csv_file = Some(f);
    }

    let human = csv_file.is_none();
    print_row_header(human);

    if let Some(rec_path) = &args.reconstructed {
        // Mode 2: compare pre-encoded file.
        let (rec_rgb, rw, rh) = if rec_path
            .extension()
            .and_then(|s| s.to_str())
            .map(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
            .unwrap_or(false)
        {
            let data = std::fs::read(rec_path).expect("read reconstructed");
            decode_jpeg_rgb(&data)
        } else {
            load_rgb(rec_path)
        };
        assert_eq!(rw, w, "reconstructed width mismatch");
        assert_eq!(rh, h, "reconstructed height mismatch");
        let rec_img = ImgRef::new(&rec_rgb, rw, rh);
        let bbs = bbs_rgb8(rec_img, orig_img);
        let bytes = std::fs::metadata(rec_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let bpp = (bytes as f64) * 8.0 / pixels;
        print_row(human, &image_name, 0, bytes, bpp, &bbs, 0.0);
        if let Some(f) = &mut csv_file {
            csv_row(f, &image_name, 0, bytes, bpp, &bbs, 0.0).expect("csv write");
        }
    } else {
        // Mode 1: encode at each quality, measure.
        for q in &args.qualities {
            let start = Instant::now();
            let jpeg = encode_jpeg(&orig_rgb, w, h, *q, args.subsampling, args.progressive);
            let enc_ms = start.elapsed().as_secs_f64() * 1000.0;
            let (rec_rgb, rw, rh) = decode_jpeg_rgb(&jpeg);
            assert_eq!(rw, w);
            assert_eq!(rh, h);
            let rec_img = ImgRef::new(&rec_rgb, rw, rh);
            let bbs = bbs_rgb8(rec_img, orig_img);
            let bpp = (jpeg.len() as f64) * 8.0 / pixels;
            print_row(human, &image_name, *q, jpeg.len(), bpp, &bbs, enc_ms);
            if let Some(f) = &mut csv_file {
                csv_row(f, &image_name, *q, jpeg.len(), bpp, &bbs, enc_ms).expect("csv write");
            }
        }
    }

    if let Some(path) = &args.csv {
        eprintln!("wrote {}", path.display());
    }
}
