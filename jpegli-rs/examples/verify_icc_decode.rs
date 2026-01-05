//! Verify ICC conversion

use std::fs;

fn main() {
    let jpeg_data = fs::read("/tmp/samesize_xyb_90.jpg").expect("read jpeg");

    #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
    {
        match jpegli::icc::decode_jpeg_with_icc(&jpeg_data) {
            Ok((pixels, width, height)) => {
                println!(
                    "Rust CMS decoded: {}x{}, {} bytes",
                    width,
                    height,
                    pixels.len()
                );

                println!("\nFirst 10 pixels (Rust CMS):");
                for i in 0..10 {
                    println!(
                        "  {}: ({}, {}, {})",
                        i,
                        pixels[i * 3],
                        pixels[i * 3 + 1],
                        pixels[i * 3 + 2]
                    );
                }

                // Also decode without ICC for comparison
                let no_icc: Vec<u8> = zune_jpeg::JpegDecoder::new(
                    zune_jpeg::zune_core::bytestream::ZCursor::new(&jpeg_data[..]),
                )
                .decode()
                .expect("decode");
                println!("\nFirst 10 pixels (No ICC):");
                for i in 0..10 {
                    println!(
                        "  {}: ({}, {}, {})",
                        i,
                        no_icc[i * 3],
                        no_icc[i * 3 + 1],
                        no_icc[i * 3 + 2]
                    );
                }
            }
            Err(e) => println!("Rust CMS error: {:?}", e),
        }
    }

    #[cfg(not(any(feature = "cms-lcms2", feature = "cms-moxcms")))]
    {
        println!("No CMS feature enabled!");
    }
}
