//! Generate comparison JPEGs and diff montages for deringing × sharp YUV × cjpegli.
//!
//! Produces 8 files in /mnt/v/output/dering-sharp-comparison/:
//!   4 "vs original" diff montages (zen variants)
//!   4 "vs cjpegli" diff montages (zen variants)
//!
//! Run: cargo run --release --example dering_sharp_montage --features decoder

use enough::Unstoppable;
use std::path::Path;
use std::process::Command;
use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

const Q: f32 = 10.0;
const OUT_DIR: &str = "/mnt/v/output/dering-sharp-comparison";
const GUI_PNG: &str =
    "/home/lilith/work/codec-eval/codec-corpus/gb82-sc/gui.png";
const CJPEGLI: &str =
    "/home/lilith/work/zen/zenjpeg-yuv-internal/internal/jpegli-cpp/build/tools/cjpegli";

fn load_png(path: &str) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).expect("open png");
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let w = info.width;
    let h = info.height;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut out = Vec::with_capacity((w * h * 3) as usize);
            for c in src.chunks_exact(4) {
                out.extend_from_slice(&c[..3]);
            }
            out
        }
        _ => panic!("unsupported color type"),
    };
    (rgb, w, h)
}

fn encode_zen(rgb: &[u8], w: u32, h: u32, dering: bool, sharp: bool) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(Q, ChromaSubsampling::Quarter)
        .progressive(true)
        .restart_mcu_rows(0) // no restart in progressive — matches the fix
        .deringing(dering)
        .sharp_yuv(sharp);
    config
        .encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode")
}

fn encode_cjpegli(png_path: &str) -> Vec<u8> {
    let out_path = format!("{OUT_DIR}/_tmp_cjpegli.jpg");
    let status = Command::new(CJPEGLI)
        .args([
            png_path,
            &out_path,
            "-q", &Q.to_string(),
            "-p", "1",
            "--chroma_subsampling", "420",
        ])
        .status()
        .expect("cjpegli");
    assert!(status.success(), "cjpegli failed");
    std::fs::read(&out_path).expect("read cjpegli output")
}

fn decode_jpeg(jpeg: &[u8]) -> Vec<u8> {
    let dec = Decoder::new().output_format(PixelFormat::Rgb);
    let result = dec.decode(jpeg, Unstoppable).expect("decode");
    result.pixels_u8().unwrap().to_vec()
}

fn save_raw_ppm(rgb: &[u8], w: u32, h: u32, path: &str) {
    let header = format!("P6\n{w} {h}\n255\n");
    let mut data = header.into_bytes();
    data.extend_from_slice(rgb);
    std::fs::write(path, data).expect("write ppm");
}

fn compute_ssim2(original: &[u8], decoded: &[u8], w: u32, h: u32) -> f64 {
    use fast_ssim2::compute_ssimulacra2;
    // Convert packed RGB u8 to &[[u8; 3]] for ImgRef
    let orig_px: &[[u8; 3]] =
        bytemuck::cast_slice(&original[..(w as usize * h as usize * 3)]);
    let dec_px: &[[u8; 3]] =
        bytemuck::cast_slice(&decoded[..(w as usize * h as usize * 3)]);
    let orig_img = imgref::ImgRef::new(orig_px, w as usize, h as usize);
    let dec_img = imgref::ImgRef::new(dec_px, w as usize, h as usize);
    compute_ssimulacra2(orig_img, dec_img).unwrap_or(-1.0)
}

fn make_diff_montage(
    original_ppm: &str,
    decoded_ppm: &str,
    out_path: &str,
    label: &str,
    ssim2: f64,
    size: usize,
) {
    // Create amplified difference image (|orig - decoded| * 4)
    let diff_path = format!("{OUT_DIR}/_tmp_diff.ppm");
    let status = Command::new("convert")
        .args([
            original_ppm,
            decoded_ppm,
            "-compose", "difference",
            "-composite",
            "-evaluate", "multiply", "4",
            &diff_path,
        ])
        .status()
        .expect("convert diff");
    assert!(status.success());

    // Montage: original | decoded | diff×4
    let title = format!("{label}  SSIM2={ssim2:.2}  {:.1}KB", size as f64 / 1024.0);
    let status = Command::new("montage")
        .args([
            original_ppm,
            decoded_ppm,
            &diff_path,
            "-tile", "3x1",
            "-geometry", "+2+2",
            "-title", &title,
            out_path,
        ])
        .status()
        .expect("montage");
    assert!(status.success());
}

fn main() {
    std::fs::create_dir_all(OUT_DIR).ok();

    let (rgb, w, h) = load_png(GUI_PNG);
    eprintln!("Loaded gui.png: {w}x{h}");

    // Save original as PPM for montage
    let orig_ppm = format!("{OUT_DIR}/_orig.ppm");
    save_raw_ppm(&rgb, w, h, &orig_ppm);

    // Encode cjpegli reference
    let cpp_jpeg = encode_cjpegli(GUI_PNG);
    let cpp_decoded = decode_jpeg(&cpp_jpeg);
    let cpp_ppm = format!("{OUT_DIR}/_cpp.ppm");
    save_raw_ppm(&cpp_decoded, w, h, &cpp_ppm);
    let cpp_ssim = compute_ssim2(&rgb, &cpp_decoded, w, h);
    eprintln!(
        "cjpegli Q{Q}: {:.1}KB  SSIM2={cpp_ssim:.2}",
        cpp_jpeg.len() as f64 / 1024.0
    );
    std::fs::write(format!("{OUT_DIR}/cjpegli_q{Q}.jpg"), &cpp_jpeg).ok();

    // 4 zen variants
    let variants = [
        (false, false, "nodering_nosharp"),
        (true, false, "dering_nosharp"),
        (false, true, "nodering_sharp"),
        (true, true, "dering_sharp"),
    ];

    for (dering, sharp, name) in &variants {
        let jpeg = encode_zen(&rgb, w, h, *dering, *sharp);
        let decoded = decode_jpeg(&jpeg);
        let ssim = compute_ssim2(&rgb, &decoded, w, h);
        let size = jpeg.len();

        // Save JPEG
        std::fs::write(format!("{OUT_DIR}/zen_{name}_q{Q}.jpg"), &jpeg).ok();

        // Save decoded PPM
        let dec_ppm = format!("{OUT_DIR}/_zen_{name}.ppm");
        save_raw_ppm(&decoded, w, h, &dec_ppm);

        let label_long = format!(
            "zen {}{}",
            if *dering { "+dering" } else { "-dering" },
            if *sharp { " +sharp" } else { " -sharp" },
        );

        eprintln!(
            "{label_long}: {:.1}KB  SSIM2={ssim:.2}",
            size as f64 / 1024.0
        );

        // Montage vs original
        make_diff_montage(
            &orig_ppm,
            &dec_ppm,
            &format!("{OUT_DIR}/vs_orig_{name}.png"),
            &label_long,
            ssim,
            size,
        );

        // Montage vs cjpegli
        let cpp_ssim_vs = compute_ssim2(&cpp_decoded, &decoded, w, h);
        make_diff_montage(
            &cpp_ppm,
            &dec_ppm,
            &format!("{OUT_DIR}/vs_cpp_{name}.png"),
            &format!("{label_long} vs cjpegli"),
            cpp_ssim_vs,
            size,
        );
    }

    // Summary
    eprintln!("\nOutput in {OUT_DIR}/");
    eprintln!("  4× vs_orig_*.png  — each zen variant vs original (diff×4)");
    eprintln!("  4× vs_cpp_*.png   — each zen variant vs cjpegli (diff×4)");
    eprintln!("  5× *.jpg          — encoded JPEGs");
}
