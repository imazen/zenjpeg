//! Compare chroma methods through the full JPEG pipeline:
//! encode RGB → JPEG Q85 4:2:0 → decode → compare to original.
//!
//! Run: `cargo test --release -p zenjpeg --test sharp_yuv_jpeg_roundtrip -- --nocapture`

mod tests {
    use enough::Unstoppable;
    use zenjpeg::decode::Decoder;
    use zenjpeg::encode::EncoderConfig;
    use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};

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
                let mut out = Vec::with_capacity((info.width * info.height * 3) as usize);
                for c in src.chunks_exact(4) {
                    out.extend_from_slice(&c[..3]);
                }
                out
            }
            _ => return None,
        };
        Some((rgb, info.width, info.height))
    }

    fn mean_abs_err(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len().min(b.len());
        let mut s = 0u64;
        for i in 0..n {
            s += a[i].abs_diff(b[i]) as u64;
        }
        s as f64 / n as f64
    }

    #[allow(dead_code)]
    fn jpeg_roundtrip(rgb: &[u8], w: u32, h: u32, config: &EncoderConfig) -> Vec<u8> {
        let jpeg = config
            .encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb)
            .expect("encode failed");
        let decoder = Decoder::new();
        let result = decoder.decode(&jpeg, Unstoppable).expect("decode failed");
        result.pixels_u8().unwrap().to_vec()
    }

    #[test]
    fn sharp_yuv_jpeg_quality() {
        let corpus_dir =
            std::path::Path::new(&std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into()))
                .join("work/codec-eval/codec-corpus/CID22/CID22-512/training");

        let mut paths: Vec<_> = std::fs::read_dir(&corpus_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("png"))
            })
            .map(|e| e.path())
            .collect();
        paths.sort();
        paths.truncate(10);

        if paths.is_empty() {
            eprintln!("No CID22 corpus found, skipping");
            return;
        }

        eprintln!(
            "=== JPEG Roundtrip Quality (Q85 4:2:0, {} images) ===",
            paths.len()
        );
        eprintln!(
            "{:>15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "image", "box_err", "box_kb", "sharp_er", "sharp_kb", "444_err", "444_kb"
        );

        let mut sum_box = 0.0f64;
        let mut sum_sharp = 0.0f64;
        let mut sum_444 = 0.0f64;
        let mut sum_box_kb = 0.0f64;
        let mut sum_sharp_kb = 0.0f64;
        let mut sum_444_kb = 0.0f64;
        let mut count = 0u32;

        for p in &paths {
            let (rgb, w, h) = match load_png_rgb(p) {
                Some(v) => v,
                None => continue,
            };

            // Box-average 4:2:0
            let cfg_box = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
            let jpeg_box = cfg_box
                .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                .unwrap();
            let box_kb = jpeg_box.len() as f64 / 1024.0;
            let dec = Decoder::new();
            let rt_box = dec
                .decode(&jpeg_box, Unstoppable)
                .unwrap()
                .pixels_u8()
                .unwrap()
                .to_vec();
            let box_err = mean_abs_err(&rgb, &rt_box);

            // Sharp YUV 4:2:0
            let cfg_sharp = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).sharp_yuv(true);
            let jpeg_sharp = cfg_sharp
                .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                .unwrap();
            let sharp_kb = jpeg_sharp.len() as f64 / 1024.0;
            let rt_sharp = dec
                .decode(&jpeg_sharp, Unstoppable)
                .unwrap()
                .pixels_u8()
                .unwrap()
                .to_vec();
            let sharp_err = mean_abs_err(&rgb, &rt_sharp);

            // 4:4:4 (no subsampling, maximum chroma quality)
            let cfg_444 = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
            let jpeg_444 = cfg_444
                .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
                .unwrap();
            let four44_kb = jpeg_444.len() as f64 / 1024.0;
            let rt_444 = dec
                .decode(&jpeg_444, Unstoppable)
                .unwrap()
                .pixels_u8()
                .unwrap()
                .to_vec();
            let four44_err = mean_abs_err(&rgb, &rt_444);

            let name = p.file_stem().unwrap().to_string_lossy();
            eprintln!(
                "{name:>15} {box_err:8.4} {box_kb:7.1}K {sharp_err:8.4} {sharp_kb:7.1}K {four44_err:8.4} {four44_kb:7.1}K"
            );

            sum_box += box_err;
            sum_sharp += sharp_err;
            sum_444 += four44_err;
            sum_box_kb += box_kb;
            sum_sharp_kb += sharp_kb;
            sum_444_kb += four44_kb;
            count += 1;
        }

        let n = count as f64;
        let avg_box = sum_box / n;
        let avg_sharp = sum_sharp / n;
        let avg_444 = sum_444 / n;
        let avg_box_kb = sum_box_kb / n;
        let avg_sharp_kb = sum_sharp_kb / n;
        let avg_444_kb = sum_444_kb / n;

        eprintln!(
            "{:>15} {avg_box:8.4} {avg_box_kb:7.1}K {avg_sharp:8.4} {avg_sharp_kb:7.1}K {avg_444:8.4} {avg_444_kb:7.1}K",
            "MEAN"
        );
        eprintln!();
        eprintln!(
            "sharp vs box: error {:.2}%, size {:.2}%",
            (avg_sharp - avg_box) / avg_box * 100.0,
            (avg_sharp_kb - avg_box_kb) / avg_box_kb * 100.0
        );
        eprintln!(
            "444 vs box:   error {:.2}%, size {:.2}%",
            (avg_444 - avg_box) / avg_box * 100.0,
            (avg_444_kb - avg_box_kb) / avg_box_kb * 100.0
        );
    }
}
