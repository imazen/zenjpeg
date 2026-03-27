//! Decode worst-case corpus files with all 4 decoders, save as PPM for
//! visual comparison via ImageMagick.
//!
//! Decoders: zenjpeg, mozjpeg (libjpeg-turbo), zune-jpeg, jpeg-decoder
//!
//! Run: cargo test --release -p zenjpeg --test dump_decoder_diffs -- --nocapture --ignored

use enough::Unstoppable;

fn out_dir() -> std::path::PathBuf {
    zenjpeg_bench_utils::zenjpeg_output_dir().join("decoder_diff")
}

fn decode_zenjpeg(data: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let d = zenjpeg::decoder::Decoder::new();
    let r = d.decode(data, Unstoppable).ok()?;
    let px = r.pixels_u8()?.to_vec();
    Some((r.width, r.height, px))
}

fn decode_mozjpeg(data: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut ci: jpeg_decompress_struct = mem::zeroed();
        ci.common.err = &mut err;
        jpeg_create_decompress(&mut ci);
        jpeg_mem_src(&mut ci, data.as_ptr(), data.len() as _);
        if jpeg_read_header(&mut ci, 1) != 1 {
            jpeg_destroy_decompress(&mut ci);
            return None;
        }
        ci.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut ci);
        let w = ci.output_width;
        let h = ci.output_height;
        let stride = w as usize * ci.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while ci.output_scanline < h {
            let off = ci.output_scanline as usize * stride;
            let mut p = out[off..].as_mut_ptr();
            jpeg_read_scanlines(&mut ci, &mut p, 1);
        }
        jpeg_finish_decompress(&mut ci);
        jpeg_destroy_decompress(&mut ci);
        Some((w, h, out))
    }
}

fn decode_zune(data: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::options::DecoderOptions;
    let mut d = JpegDecoder::new_with_options(ZCursor::new(data), DecoderOptions::new_fast());
    d.decode_headers().ok()?;
    let info = d.info()?;
    let w = info.width as u32;
    let h = info.height as u32;
    let px = d.decode().ok()?;
    Some((w, h, px))
}

fn decode_jpeg_decoder(data: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    use jpeg_decoder::Decoder;
    let mut d = Decoder::new(data);
    d.read_info().ok()?;
    let info = d.info()?;
    let w = info.width as u32;
    let h = info.height as u32;
    let px = d.decode().ok()?;
    // jpeg-decoder outputs in the image's native colorspace by default
    // For JPEG YCbCr images it converts to RGB, for CMYK it stays CMYK, etc.
    // If the pixel format is not RGB, skip
    if info.pixel_format != jpeg_decoder::PixelFormat::RGB24 {
        return None;
    }
    Some((w, h, px))
}

fn write_ppm(path: &str, w: u32, h: u32, rgb: &[u8]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    f.write_all(rgb).unwrap();
}

fn diff_stats(a: &[u8], b: &[u8]) -> (u8, f64) {
    let max = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    let mean: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as f64)
        .sum::<f64>()
        / a.len() as f64;
    (max, mean)
}

fn print_diff(label: &str, a: &Option<(u32, u32, Vec<u8>)>, b: &Option<(u32, u32, Vec<u8>)>) {
    if let (Some((_, _, ap)), Some((_, _, bp))) = (a, b) {
        if ap.len() == bp.len() {
            let (max, mean) = diff_stats(ap, bp);
            println!("  {label:<12} max={max:<4} mean={mean:.3}");
        } else {
            println!("  {label:<12} SIZE MISMATCH ({} vs {})", ap.len(), bp.len());
        }
    }
}

#[test]
#[ignore = "requires corpus + output dir"]
fn dump_diffs() {
    let corpus = zenjpeg_bench_utils::corpus_builder_dir();
    let out = out_dir();
    let files = [
        (
            "prophoto",
            corpus.join("wide-gamut/prophoto-rgb/reddit_36d104c8b6b9e5dd.jpg"),
        ),
        (
            "adobe1",
            corpus.join("wide-gamut/adobe-rgb/flickr_841c1e16a9a5484a.jpg"),
        ),
        ("src_da76", corpus.join("source_jpegs/da76cd8775e67305.jpg")),
        (
            "adobe2",
            corpus.join("wide-gamut/adobe-rgb/flickr_5e9c282e096363d7.jpg"),
        ),
        ("src_2fe0", corpus.join("source_jpegs/2fe0acf8200b556b.jpg")),
    ];

    let _ = std::fs::create_dir_all(&out);

    for (name, path) in &files {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {name}: {e}");
                continue;
            }
        };
        println!("\n=== {name} ({} bytes) ===", data.len());

        let zen = decode_zenjpeg(&data);
        let moz = decode_mozjpeg(&data);
        let zune = decode_zune(&data);
        let jpd = decode_jpeg_decoder(&data);

        // Print decode results
        for (label, result) in [("zen", &zen), ("moz", &moz), ("zune", &zune), ("jpd", &jpd)] {
            if let Some((w, h, px)) = result {
                println!("  {label:<5} {w}x{h}, {} bytes", px.len());
                write_ppm(&format!("{}/{name}_{label}.ppm", out.display()), *w, *h, px);
            } else {
                println!("  {label:<5} FAILED");
            }
        }

        // All pairwise diffs
        println!();
        print_diff("zen-moz:", &zen, &moz);
        print_diff("zen-zune:", &zen, &zune);
        print_diff("zen-jpd:", &zen, &jpd);
        print_diff("moz-zune:", &moz, &zune);
        print_diff("moz-jpd:", &moz, &jpd);
        print_diff("zune-jpd:", &zune, &jpd);
    }
    println!("\nPPM files written to {}/", out.display());
}
