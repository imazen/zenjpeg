//! Compare deblocking quality between decode() (f32) and scanline_reader() (i16) paths.
//! Both use Boundary4Tap. Measures zensim vs original for each path.

use enough::Unstoppable;
use std::path::Path;
use zenjpeg::decode::DeblockMode;
use zenjpeg::decoder::Decoder;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
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
            for c in src.chunks_exact(4) {
                rgb.extend_from_slice(&c[..3]);
            }
            rgb
        }
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn as_rgb(d: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(d)
}

#[test]
#[ignore = "requires CID22 corpus"]
fn deblock_path_quality_comparison() {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(_) => {
            println!("no corpus");
            return;
        }
    };
    let dir = match corpus.get("CID22/CID22-512/training") {
        Ok(d) => d,
        Err(_) => {
            println!("no CID22");
            return;
        }
    };

    let paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .take(25)
        .collect();

    let z = Zensim::new(ZensimProfile::codec_target());
    let qualities = [10, 20, 50, 85];

    println!("  Comparing decode() vs scanline_reader(), both with Boundary4Tap");
    println!("  decode():          f32 planes, Jpegli IDCT");
    println!("  scanline_reader(): i16 planes, streaming IDCT\n");

    println!(
        "  {:>3}  {:>8}  {:>8}  {:>8}  {:>6}  {:>4}",
        "Q", "no_debl", "f32_dbl", "i16_dbl", "f32-i16", "max"
    );
    println!("  {}", "-".repeat(48));

    for &q in &qualities {
        let mut sum_plain = 0.0f64;
        let mut sum_f32 = 0.0f64;
        let mut sum_i16 = 0.0f64;
        let mut max_diff_pixels = 0u8;
        let mut count = 0usize;

        for path in &paths {
            let (orig, w, h) = match load_png_rgb(path) {
                Some(v) => v,
                None => continue,
            };

            // Encode with mozjpeg baseline (scanline-compatible, has DRI)
            let jpeg = mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::BaselineBalanced)
                .quality(q)
                .subsampling(mozjpeg_rs::Subsampling::S420)
                .encode_rgb(&orig, w, h)
                .unwrap();

            // Plain decode (no deblock)
            let plain = Decoder::new()
                .decode(&jpeg, Unstoppable)
                .unwrap()
                .into_pixels_u8()
                .unwrap();

            // decode() path with Boundary4Tap (f32 planes)
            let f32_debl = Decoder::new()
                .deblock(DeblockMode::Boundary4Tap)
                .decode(&jpeg, Unstoppable)
                .unwrap()
                .into_pixels_u8()
                .unwrap();

            // scanline_reader() path with Boundary4Tap (i16 planes)
            let mut sr = Decoder::new()
                .deblock(DeblockMode::Boundary4Tap)
                .scanline_reader(&jpeg)
                .unwrap();
            let mut i16_debl = vec![0u8; (w * h * 3) as usize];
            sr.read_rows_rgb8(imgref::ImgRefMut::new(
                &mut i16_debl,
                w as usize * 3,
                h as usize,
            ))
            .unwrap();

            let ws = w as usize;
            let hs = h as usize;
            let s_plain = z
                .compute(
                    &RgbSlice::new(as_rgb(&orig), ws, hs),
                    &RgbSlice::new(as_rgb(&plain), ws, hs),
                )
                .map(|r| r.score())
                .unwrap_or(-1.0);
            let s_f32 = z
                .compute(
                    &RgbSlice::new(as_rgb(&orig), ws, hs),
                    &RgbSlice::new(as_rgb(&f32_debl), ws, hs),
                )
                .map(|r| r.score())
                .unwrap_or(-1.0);
            let s_i16 = z
                .compute(
                    &RgbSlice::new(as_rgb(&orig), ws, hs),
                    &RgbSlice::new(as_rgb(&i16_debl), ws, hs),
                )
                .map(|r| r.score())
                .unwrap_or(-1.0);

            // Max pixel diff between f32 and i16 deblocked output
            let md = f32_debl
                .iter()
                .zip(i16_debl.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);

            sum_plain += s_plain;
            sum_f32 += s_f32;
            sum_i16 += s_i16;
            if md > max_diff_pixels {
                max_diff_pixels = md;
            }
            count += 1;
        }

        let n = count as f64;
        let mp = sum_plain / n;
        let mf = sum_f32 / n;
        let mi = sum_i16 / n;
        println!(
            "  Q{q:<2}  {mp:>8.2}  {mf:>8.2}  {mi:>8.2}  {:>+5.2}  {max_diff_pixels:>4}",
            mf - mi
        );
    }
}
