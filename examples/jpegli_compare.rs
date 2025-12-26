//! Compare zenjpeg vs jpegli quality

use dssim::Dssim;
use rgb::RGB8;

fn main() {
    // Create synthetic test image
    let width = 512usize;
    let height = 512usize;
    let mut rgb_data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let r = ((fx * 200.0 + fy * 50.0) % 256.0) as u8;
            let g = ((fy * 180.0 + (x * 3) as f32 % 100.0) as u32 % 256) as u8;
            let b = (((x + y) % 64) as f32 / 64.0 * 128.0 + 64.0) as u8;
            rgb_data.push(r);
            rgb_data.push(g);
            rgb_data.push(b);
        }
    }

    println!("=== zenjpeg vs jpegli Comparison ({}x{}) ===\n", width, height);

    for q in [60, 75, 85, 95] {
        // zenjpeg with mozjpeg strategy (trellis)
        let zen_moz = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(q))
            .strategy(zenjpeg::EncodingStrategy::Mozjpeg)
            .encode_rgb(&rgb_data, width, height)
            .unwrap();

        // zenjpeg with jpegli strategy (perceptual quant tables + zero-bias)
        let zen_jpegli = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(q))
            .strategy(zenjpeg::EncodingStrategy::Jpegli)
            .encode_rgb(&rgb_data, width, height)
            .unwrap();

        // jpegli (Rust port) - uses Traditional quality mode
        let jpegli_result = jpegli::Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(jpegli::PixelFormat::Rgb)
            .quality(jpegli::Quality::Traditional(q as f32))
            .encode(&rgb_data);
        
        let jpegli_bytes = match jpegli_result {
            Ok(b) => b,
            Err(e) => {
                println!("Q{}: jpegli failed: {:?}", q, e);
                continue;
            }
        };

        // mozjpeg-oxide for reference
        let moz = mozjpeg_oxide::Encoder::new()
            .quality(q)
            .subsampling(mozjpeg_oxide::Subsampling::S444)
            .encode_rgb(&rgb_data, width as u32, height as u32)
            .unwrap();

        // Decode all
        let zen_moz_dec = jpeg_decoder::Decoder::new(&zen_moz[..]).decode().unwrap();
        let zen_jpegli_dec = jpeg_decoder::Decoder::new(&zen_jpegli[..]).decode().unwrap();
        let jpegli_dec = jpeg_decoder::Decoder::new(&jpegli_bytes[..]).decode().unwrap();
        let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

        // Calculate DSSIM
        let dssim_zen_moz = calculate_dssim(&rgb_data, &zen_moz_dec, width, height);
        let dssim_zen_jpegli = calculate_dssim(&rgb_data, &zen_jpegli_dec, width, height);
        let dssim_jpegli = calculate_dssim(&rgb_data, &jpegli_dec, width, height);
        let dssim_moz = calculate_dssim(&rgb_data, &moz_dec, width, height);

        println!("Q{}:", q);
        println!("  Size:");
        println!("    zen/moz:    {:6}  ({:+.1}% vs mozjpeg)", zen_moz.len(),
            (zen_moz.len() as f64 / moz.len() as f64 - 1.0) * 100.0);
        println!("    zen/jpegli: {:6}  ({:+.1}% vs ref jpegli)", zen_jpegli.len(),
            (zen_jpegli.len() as f64 / jpegli_bytes.len() as f64 - 1.0) * 100.0);
        println!("    ref jpegli: {:6}", jpegli_bytes.len());
        println!("    mozjpeg:    {:6}", moz.len());
        println!("  DSSIM:");
        println!("    zen/moz:    {:.6}", dssim_zen_moz);
        println!("    zen/jpegli: {:.6}", dssim_zen_jpegli);
        println!("    ref jpegli: {:.6}", dssim_jpegli);
        println!("    mozjpeg:    {:.6}", dssim_moz);
        println!();
    }
}

fn calculate_dssim(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    
    let orig_rgb: Vec<RGB8> = orig.chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dec_rgb: Vec<RGB8> = decoded.chunks(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    
    let orig_img = attr.create_image_rgb(&orig_rgb, width, height).unwrap();
    let dec_img = attr.create_image_rgb(&dec_rgb, width, height).unwrap();
    
    let (dssim_val, _) = attr.compare(&orig_img, dec_img);
    dssim_val.into()
}

fn calculate_mae(orig: &[u8], decoded: &[u8]) -> f64 {
    let mut total = 0i64;
    for i in 0..orig.len() {
        total += (orig[i] as i64 - decoded[i] as i64).abs();
    }
    total as f64 / orig.len() as f64
}
