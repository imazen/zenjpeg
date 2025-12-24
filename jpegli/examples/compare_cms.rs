//! Compare lcms2 vs moxcms ICC transform outputs
//!
//! Usage: cargo run --release --example compare_cms --features cms-lcms2
//!        cargo run --release --example compare_cms --features cms-moxcms

use jpegli::icc;
use std::fs;

fn main() {
    // Load an XYB JPEG
    let path = "/tmp/cpp_xyb_blue.jpg";
    if !std::path::Path::new(path).exists() {
        eprintln!("Test file not found. Run the pareto_front tests first.");
        return;
    }

    let jpeg_data = fs::read(path).expect("read jpeg");

    // Extract ICC profile
    let profile = icc::extract_icc_profile(&jpeg_data);
    println!("ICC profile: {:?}", profile.as_ref().map(|p| p.len()));

    if let Some(ref profile) = profile {
        println!("Is XYB profile: {}", icc::is_xyb_profile(profile));
    }

    // Try to decode with ICC
    #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
    {
        match icc::decode_jpeg_with_icc(&jpeg_data) {
            Ok((pixels, w, h)) => {
                println!("Decoded: {}x{}, {} bytes", w, h, pixels.len());
                if pixels.len() >= 3 {
                    println!("First pixel: ({}, {}, {})", pixels[0], pixels[1], pixels[2]);
                    println!("Expected: ~(0, 0, 128) for blue");
                }
            }
            Err(e) => {
                eprintln!("Decode error: {}", e);
            }
        }

        #[cfg(feature = "cms-lcms2")]
        println!("Using: lcms2");
        #[cfg(all(feature = "cms-moxcms", not(feature = "cms-lcms2")))]
        println!("Using: moxcms");
    }

    #[cfg(not(any(feature = "cms-lcms2", feature = "cms-moxcms")))]
    {
        eprintln!("No CMS feature enabled. Use --features cms-lcms2 or --features cms-moxcms");
    }
}
