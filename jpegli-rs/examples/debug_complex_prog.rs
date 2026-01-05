//! Debug progressive scan sizes for complex image
use jpegli::{Encoder, PixelFormat};

fn main() {
    // Complex pattern (same as cpp_rust_matrix)
    let width = 512;
    let height = 512;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;
            data[idx] = ((fx * 255.0) + (fx * fy * 50.0).sin() * 30.0).clamp(0.0, 255.0) as u8;
            data[idx + 1] = ((fy * 255.0) + (fx * fy * 100.0).cos() * 40.0).clamp(0.0, 255.0) as u8;
            data[idx + 2] = (128.0 + ((fx + fy) * 50.0).sin() * 50.0).clamp(0.0, 255.0) as u8;
        }
    }

    println!("=== Encoding 512x512 complex pattern with YCbCr Progressive ===\n");

    let jpeg = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(false)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    println!("Total: {} bytes\n", jpeg.len());
    analyze_scans(&jpeg);
}

fn analyze_scans(data: &[u8]) {
    let mut i = 0;
    let mut scan_num = 0;
    let mut total_dc = 0;
    let mut total_ac_first = 0;
    let mut total_ac_refine = 0;

    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xDA {
            let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            scan_num += 1;

            let num_components = data[i + 4];
            let mut offset = i + 5;
            offset += 2 * num_components as usize;

            let ss = data[offset];
            let se = data[offset + 1];
            let ah_al = data[offset + 2];
            let ah = ah_al >> 4;
            let al = ah_al & 0x0F;

            // Find scan data end
            let scan_start = i + 2 + length;
            let mut scan_end = scan_start;
            while scan_end + 1 < data.len() {
                if data[scan_end] == 0xFF
                    && data[scan_end + 1] != 0x00
                    && data[scan_end + 1] != 0xFF
                {
                    break;
                }
                scan_end += 1;
            }
            let scan_size = scan_end - scan_start;

            let scan_type = if ss == 0 {
                "DC"
            } else if ah == 0 {
                "AC First"
            } else {
                "AC Refine"
            };

            println!(
                "Scan #{:2}: {:10} Ss={:2}-{:2}, Ah={}, Al={}, {} comps → {:6} bytes",
                scan_num, scan_type, ss, se, ah, al, num_components, scan_size
            );

            if ss == 0 {
                total_dc += scan_size;
            } else if ah == 0 {
                total_ac_first += scan_size;
            } else {
                total_ac_refine += scan_size;
            }

            i = scan_end;
        } else {
            i += 1;
        }
    }

    println!("\n=== TOTALS ===");
    println!("DC scans:        {:6} bytes", total_dc);
    println!("AC First scans:  {:6} bytes", total_ac_first);
    println!("AC Refine scans: {:6} bytes", total_ac_refine);
    println!(
        "Sum:             {:6} bytes",
        total_dc + total_ac_first + total_ac_refine
    );
}
