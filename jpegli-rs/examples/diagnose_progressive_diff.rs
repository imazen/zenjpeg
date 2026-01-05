//! Diagnose where progressive decoder diverges from zune-jpeg
//!
//! Strategy: Decode block-by-block and compare intermediate coefficient values

use jpegli::Decoder;
use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        // Load PNG
        let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let rgb = &buf[..info.buffer_size()];
        let width = info.width as u32;
        let height = info.height as u32;

        // Write PPM
        let ppm_path = "/tmp/diagnose.ppm";
        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", width, height).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        // Encode with C++ cjpegli (progressive)
        let cpp_prog_path = "/tmp/cpp_progressive_diagnose.jpg";
        Command::new(&cjpegli)
            .args([ppm_path, cpp_prog_path, "-q", "90"])
            .output()
            .unwrap();

        let cpp_jpeg = fs::read(cpp_prog_path).unwrap();
        println!("C++ progressive JPEG: {} bytes\n", cpp_jpeg.len());

        // Decode with jpegli-rs
        println!("=== Decoding with jpegli-rs ===");
        let jpegli_result = Decoder::new().decode(&cpp_jpeg);

        match &jpegli_result {
            Ok(img) => {
                println!(
                    "✓ Decoded: {}x{}, {} bytes",
                    img.width,
                    img.height,
                    img.data.len()
                );
            }
            Err(e) => {
                println!("✗ Failed: {:?}", e);
                return;
            }
        }

        // Decode with zune-jpeg
        println!("\n=== Decoding with zune-jpeg ===");
        let mut zune_decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&cpp_jpeg));
        let zune_result = zune_decoder.decode();

        match &zune_result {
            Ok(pixels) => {
                let info = zune_decoder.info().unwrap();
                println!(
                    "✓ Decoded: {}x{}, {} bytes",
                    info.width,
                    info.height,
                    pixels.len()
                );
            }
            Err(e) => {
                println!("✗ Failed: {:?}", e);
                return;
            }
        }

        // Compare outputs
        if let (Ok(jpegli_img), Ok(zune_pixels)) = (jpegli_result, zune_result) {
            println!("\n=== Pixel Comparison ===");

            // Sample a few blocks to see the differences
            let sample_blocks = [
                (0, 0),     // Top-left
                (32, 32),   // Interior
                (256, 256), // Middle
                (480, 500), // Near bottom-right
            ];

            for &(x, y) in &sample_blocks {
                if x < width && y < height {
                    let idx = ((y * width + x) * 3) as usize;
                    if idx + 2 < jpegli_img.data.len() && idx + 2 < zune_pixels.len() {
                        let jpegli_rgb = [
                            jpegli_img.data[idx],
                            jpegli_img.data[idx + 1],
                            jpegli_img.data[idx + 2],
                        ];
                        let zune_rgb =
                            [zune_pixels[idx], zune_pixels[idx + 1], zune_pixels[idx + 2]];

                        let diff = [
                            (jpegli_rgb[0] as i16 - zune_rgb[0] as i16).abs(),
                            (jpegli_rgb[1] as i16 - zune_rgb[1] as i16).abs(),
                            (jpegli_rgb[2] as i16 - zune_rgb[2] as i16).abs(),
                        ];

                        println!(
                            "Pixel ({:3}, {:3}): jpegli={:3?} zune={:3?} diff={:2?}",
                            x, y, jpegli_rgb, zune_rgb, diff
                        );
                    }
                }
            }

            // Overall statistics
            println!("\n=== Overall Statistics ===");
            let mut max_diff = [0i16; 3];
            let mut diff_count = [0usize; 3];
            let mut diff_histogram = vec![vec![0usize; 256]; 3]; // Per channel

            for i in (0..jpegli_img.data.len().min(zune_pixels.len())).step_by(3) {
                for c in 0..3 {
                    if i + c < jpegli_img.data.len() && i + c < zune_pixels.len() {
                        let diff =
                            (jpegli_img.data[i + c] as i16 - zune_pixels[i + c] as i16).abs();
                        max_diff[c] = max_diff[c].max(diff);
                        if diff > 0 {
                            diff_count[c] += 1;
                        }
                        if diff < 256 {
                            diff_histogram[c][diff as usize] += 1;
                        }
                    }
                }
            }

            let total_pixels = jpegli_img.data.len() / 3;
            println!("Channel | Max Diff | Pixels Different | Percentage");
            println!("--------|----------|------------------|------------");
            for c in 0..3 {
                let channel_name = ["R", "G", "B"][c];
                println!(
                    "{:7} | {:8} | {:16} | {:9.2}%",
                    channel_name,
                    max_diff[c],
                    diff_count[c],
                    (diff_count[c] as f64 / total_pixels as f64) * 100.0
                );
            }

            // Show diff distribution
            println!("\n=== Difference Distribution (combined) ===");
            let mut combined_hist = vec![0usize; 256];
            for c in 0..3 {
                for (val, count) in diff_histogram[c].iter().enumerate() {
                    combined_hist[val] += count;
                }
            }

            for diff_val in 0..20 {
                if combined_hist[diff_val] > 0 {
                    let pct = (combined_hist[diff_val] as f64 / (total_pixels * 3) as f64) * 100.0;
                    println!(
                        "  Diff {:2}: {:7} pixels ({:5.2}%)",
                        diff_val, combined_hist[diff_val], pct
                    );
                }
            }
        }
    } else {
        println!("Error: cjpegli not found");
    }
}
