//! Validate JPEG files with third-party decoders.
//!
//! Usage: cargo run --example validate_jpeg --release -- <jpeg_file_or_dir>

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <jpeg_file_or_dir>", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);

    if path.is_dir() {
        let mut files: Vec<_> = fs::read_dir(path)
            .expect("Failed to read directory")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| {
                        ext.to_ascii_lowercase() == "jpg" || ext.to_ascii_lowercase() == "jpeg"
                    })
                    .unwrap_or(false)
            })
            .collect();
        files.sort_by_key(|e| e.path());

        for entry in files.iter().take(10) {
            validate_file(&entry.path());
        }
    } else {
        validate_file(path);
    }
}

fn validate_file(path: &Path) {
    println!("=== {} ===", path.display());

    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            println!("  Read error: {}", e);
            return;
        }
    };

    println!("  File size: {} bytes", data.len());

    // Test with zune-jpeg
    print!("  zune-jpeg: ");
    match zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&data)).decode() {
        Ok(pixels) => println!("OK ({} bytes decoded)", pixels.len()),
        Err(e) => println!("ERROR: {:?}", e),
    }

    // Test with our decoder
    print!("  jpegli-rs: ");
    match jpegli::Decoder::new().decode(&data) {
        Ok(img) => println!(
            "OK ({}x{}, {} bytes)",
            img.width,
            img.height,
            img.data.len()
        ),
        Err(e) => println!("ERROR: {}", e),
    }

    // Check markers
    print!("  Markers: ");
    let mut i = 0;
    let mut markers = Vec::new();
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];
            markers.push(format!("{:02X}", marker));
            if marker == 0xD9 {
                break; // EOI
            }
        }
        i += 1;
    }
    println!("{}", markers.join(" "));

    println!();
}
