//! Direct comparison: jpegli (with/without AQ) vs mozjpeg for sharpened images
//!
//! Tests what matters when trellis is involved

use std::env;
use std::fs;
use std::path::PathBuf;

use dssim::Dssim;
use jpegli::adaptive_quant::AQStrengthMap;

fn main() {
    let corpus_dir = PathBuf::from("/home/lilith/work/codec-eval/corpus/sharpened-800px");

    let mut files: Vec<PathBuf> = fs::read_dir(&corpus_dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "png") &&
            p.file_name().unwrap().to_string_lossy().starts_with("clic_")
        })
        .collect();
    files.sort();

    let max_files: usize = env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    files.truncate(max_files);

    println!("Comparing encoders on {} sharpened images\n", files.len());
    println!("Testing what happens with/without AQ when trellis is used\n");

    let quality = 75u8; // Match mozjpeg quality scale
    let attr = Dssim::new();

    #[derive(Default)]
    struct Totals { size: usize, dssim: f64, butter: f64, count: usize }
    let mut jpegli_totals = Totals::default();
    let mut jpegli_noaq_totals = Totals::default();
    let mut mozjpeg_totals = Totals::default();

    println!("{:35} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Image", "jpegli", "jpegli_noaq", "mozjpeg", "j_dssim", "noaq_dssim", "moz_dssim");
    println!("{}", "-".repeat(105));

    for file in &files {
        let filename = file.file_name().unwrap().to_string_lossy();

        let Ok(f) = fs::File::open(file) else { continue };
        let decoder = png::Decoder::new(f);
        let Ok(mut reader) = decoder.read_info() else { continue };
        let mut buf = vec![0; reader.output_buffer_size()];
        let Ok(info) = reader.next_frame(&mut buf) else { continue };

        if info.color_type != png::ColorType::Rgb { continue }

        let pixels = &buf[..info.buffer_size()];
        let width = info.width as usize;
        let height = info.height as usize;

        let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
            .chunks(3)
            .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
            .collect();
        let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();

        // 1. jpegli with default AQ
        let jpegli_result = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .quality(jpegli::quant::Quality::from_quality(quality.into()))
            .encode(pixels)
            .unwrap();

        // 2. jpegli with AQ disabled (uniform quant like mozjpeg)
        let jpegli_noaq_result = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .quality(jpegli::quant::Quality::from_quality(quality.into()))
            .use_adaptive_quantization(false)
            .encode(pixels)
            .unwrap();

        // 3. mozjpeg (has its own trellis, no AQ concept)
        let mozjpeg_result = encode_mozjpeg(pixels, width, height, quality);

        let j_dssim = compute_dssim(&attr, &orig_img, &jpegli_result, width, height);
        let noaq_dssim = compute_dssim(&attr, &orig_img, &jpegli_noaq_result, width, height);
        let moz_dssim = compute_dssim(&attr, &orig_img, &mozjpeg_result, width, height);

        println!("{:35} {:>10} {:>10} {:>10} {:>10.5} {:>10.5} {:>10.5}",
            &filename[..filename.len().min(35)],
            jpegli_result.len(),
            jpegli_noaq_result.len(),
            mozjpeg_result.len(),
            j_dssim, noaq_dssim, moz_dssim);

        jpegli_totals.size += jpegli_result.len();
        jpegli_totals.dssim += j_dssim;
        jpegli_totals.count += 1;

        jpegli_noaq_totals.size += jpegli_noaq_result.len();
        jpegli_noaq_totals.dssim += noaq_dssim;
        jpegli_noaq_totals.count += 1;

        mozjpeg_totals.size += mozjpeg_result.len();
        mozjpeg_totals.dssim += moz_dssim;
        mozjpeg_totals.count += 1;
    }

    println!("{}", "-".repeat(105));
    let n = jpegli_totals.count as f64;
    println!("{:35} {:>10} {:>10} {:>10} {:>10.5} {:>10.5} {:>10.5}",
        "TOTAL/AVG",
        jpegli_totals.size,
        jpegli_noaq_totals.size,
        mozjpeg_totals.size,
        jpegli_totals.dssim / n,
        jpegli_noaq_totals.dssim / n,
        mozjpeg_totals.dssim / n);

    println!("\n=== Analysis ===\n");

    let j_avg_dssim = jpegli_totals.dssim / n;
    let noaq_avg_dssim = jpegli_noaq_totals.dssim / n;
    let moz_avg_dssim = mozjpeg_totals.dssim / n;

    println!("Size comparison (vs jpegli):");
    println!("  jpegli (AQ):    {} bytes (baseline)", jpegli_totals.size);
    println!("  jpegli (no AQ): {} bytes ({:+.1}%)", jpegli_noaq_totals.size,
        (jpegli_noaq_totals.size as f64 / jpegli_totals.size as f64 - 1.0) * 100.0);
    println!("  mozjpeg:        {} bytes ({:+.1}%)", mozjpeg_totals.size,
        (mozjpeg_totals.size as f64 / jpegli_totals.size as f64 - 1.0) * 100.0);

    println!("\nQuality comparison (DSSIM, lower is better):");
    println!("  jpegli (AQ):    {:.5} (baseline)", j_avg_dssim);
    println!("  jpegli (no AQ): {:.5} ({:+.1}%)", noaq_avg_dssim,
        (noaq_avg_dssim / j_avg_dssim - 1.0) * 100.0);
    println!("  mozjpeg:        {:.5} ({:+.1}%)", moz_avg_dssim,
        (moz_avg_dssim / j_avg_dssim - 1.0) * 100.0);

    println!("\n=== What's Different ===\n");
    println!("1. Quantization tables: jpegli uses psychovisually-tuned tables");
    println!("2. AQ: jpegli varies quant per-block based on content complexity");
    println!("3. Trellis: mozjpeg has trellis, jpegli-rs hybrid mode adds it");
    println!("4. Huffman: both optimize, different algorithms");
    println!("\nWhen AQ is disabled, jpegli behaves more like mozjpeg (uniform quant).");
    println!("The remaining difference is quantization tables and Huffman coding.");
}

fn compute_dssim(
    attr: &Dssim,
    orig: &dssim::DssimImage<f32>,
    jpeg_data: &[u8],
    width: usize,
    height: usize,
) -> f64 {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    let decoded = decoder.decode().expect("decode");

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = attr.create_image_rgba(&decoded_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(orig, decoded_img);
    dssim.into()
}

fn encode_mozjpeg(pixels: &[u8], width: usize, height: usize, quality: u8) -> Vec<u8> {
    std::panic::catch_unwind(|| {
        use mozjpeg::{ColorSpace, Compress};

        let mut comp = Compress::new(ColorSpace::JCS_RGB);
        comp.set_size(width, height);
        comp.set_quality(quality as f32);
        comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1)); // 4:4:4

        let mut started = comp.start_compress(Vec::new()).expect("start");

        let row_stride = width * 3;
        for y in 0..height {
            let row_start = y * row_stride;
            let row = &pixels[row_start..row_start + row_stride];
            let _ = started.write_scanlines(row);
        }

        started.finish().expect("finish")
    })
    .unwrap_or_default()
}
