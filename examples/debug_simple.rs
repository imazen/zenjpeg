use std::io::Write;
use zenjpeg::{Encoder, Quality, ScanScript};

fn main() {
    let width = 16usize;
    let height = 16usize;
    let mut rgb = vec![128u8; width * height * 3];

    // Create gradient pattern
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            rgb[i * 3] = ((x * 16) % 256) as u8;
            rgb[i * 3 + 1] = ((y * 16) % 256) as u8;
            rgb[i * 3 + 2] = 128;
        }
    }

    // Test with standard Huffman tables (no optimization)
    let simple_std = Encoder::new()
        .quality(Quality::Standard(85))
        .progressive(true)
        .scan_script(ScanScript::Simple)
        .optimize_huffman(false) // Use standard tables
        .encode_rgb(&rgb, width, height)
        .expect("Simple with std tables failed");

    // Write to file for external inspection
    let mut f = std::fs::File::create("/tmp/simple_std.jpg").unwrap();
    f.write_all(&simple_std).unwrap();
    println!("Wrote {} bytes to /tmp/simple_std.jpg", simple_std.len());

    // Try decoding with jpeg_decoder
    match jpeg_decoder::Decoder::new(std::io::Cursor::new(&simple_std)).decode() {
        Ok(dec) => {
            let psnr = calculate_psnr(&rgb, &dec);
            println!("Decoded OK, PSNR: {:.2} dB", psnr);
        }
        Err(e) => {
            println!("Decode error: {:?}", e);
        }
    }

    // Try djpeg
    let status = std::process::Command::new("djpeg")
        .args(["-ppm", "/tmp/simple_std.jpg"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => println!("djpeg: SUCCESS"),
        Ok(s) => println!("djpeg: FAILED (exit {})", s.code().unwrap_or(-1)),
        Err(e) => println!("djpeg: error {}", e),
    }
}

fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    let len = original.len().min(decoded.len());
    let mse: f64 = original[..len]
        .iter()
        .zip(decoded[..len].iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / len as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}
