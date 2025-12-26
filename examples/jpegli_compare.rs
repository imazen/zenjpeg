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
        // zenjpeg
        let zen = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(q))
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
        let zen_dec = jpeg_decoder::Decoder::new(&zen[..]).decode().unwrap();
        let jpegli_dec = jpeg_decoder::Decoder::new(&jpegli_bytes[..]).decode().unwrap();
        let moz_dec = jpeg_decoder::Decoder::new(&moz[..]).decode().unwrap();

        // Calculate DSSIM
        let dssim_zen = calculate_dssim(&rgb_data, &zen_dec, width, height);
        let dssim_jpegli = calculate_dssim(&rgb_data, &jpegli_dec, width, height);
        let dssim_moz = calculate_dssim(&rgb_data, &moz_dec, width, height);

        // Calculate MAE
        let mae_zen = calculate_mae(&rgb_data, &zen_dec);
        let mae_jpegli = calculate_mae(&rgb_data, &jpegli_dec);
        let mae_moz = calculate_mae(&rgb_data, &moz_dec);

        println!("Q{}:", q);
        println!("  Size:  zen={:6} jpegli={:6} moz={:6}", zen.len(), jpegli_bytes.len(), moz.len());
        println!("  DSSIM: zen={:.6} jpegli={:.6} moz={:.6}", dssim_zen, dssim_jpegli, dssim_moz);
        println!("  MAE:   zen={:.3} jpegli={:.3} moz={:.3}", mae_zen, mae_jpegli, mae_moz);
        
        // Pareto analysis: bits per pixel per quality unit
        let bpp_zen = (zen.len() * 8) as f64 / (width * height) as f64;
        let bpp_jpegli = (jpegli_bytes.len() * 8) as f64 / (width * height) as f64;
        let bpp_moz = (moz.len() * 8) as f64 / (width * height) as f64;
        
        println!("  BPP:   zen={:.3} jpegli={:.3} moz={:.3}", bpp_zen, bpp_jpegli, bpp_moz);
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
