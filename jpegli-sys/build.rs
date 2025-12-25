//! Build script for jpegli-sys.
//!
//! This builds the C++ jpegli library and links it for FFI testing.
//!
//! Features:
//! - `butteraugli`: Links with jxl_extras-internal for butteraugli FFI

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let butteraugli_enabled = env::var("CARGO_FEATURE_BUTTERAUGLI").is_ok();

    // Check if the jpegli C++ source is available
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let jpegli_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Check if build directory exists with pre-built library
    // When butteraugli is enabled, prefer build/ since build_test/ doesn't have extras
    let prebuilt_lib = if butteraugli_enabled {
        // Need jxl_extras-internal which is only in build/
        if jpegli_root.join("build/lib/libjxl_extras-internal.a").exists() {
            Some(jpegli_root.join("build"))
        } else {
            None
        }
    } else if jpegli_root.join("build_test/lib/libjpegli-static.a").exists() {
        Some(jpegli_root.join("build_test"))
    } else if jpegli_root.join("build/lib/libjpegli-static.a").exists() {
        Some(jpegli_root.join("build"))
    } else {
        None
    };

    if let Some(build_dir) = prebuilt_lib {
        // Use pre-built library
        let lib_dir = build_dir.join("lib");
        let hwy_dir = build_dir.join("third_party/highway");
        let lcms2_dir = build_dir.join("third_party");

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=jpegli-static");

        // Link jxl_extras-internal for butteraugli support
        if butteraugli_enabled {
            println!("cargo:rustc-link-lib=static=jxl_extras-internal");

            // lcms2 is needed for color management
            if lcms2_dir.join("liblcms2.a").exists() {
                println!("cargo:rustc-link-search=native={}", lcms2_dir.display());
                println!("cargo:rustc-link-lib=static=lcms2");
            }
        }

        if hwy_dir.join("libhwy.a").exists() {
            println!("cargo:rustc-link-search=native={}", hwy_dir.display());
            println!("cargo:rustc-link-lib=static=hwy");
        }

        // Link C++ standard library
        println!("cargo:rustc-link-lib=stdc++");

        // Link math library (needed for log2, pow, etc.)
        println!("cargo:rustc-link-lib=m");
    } else {
        // Build using cmake
        let mut config = cmake::Config::new(jpegli_root);
        config
            .define("BUILD_TESTING", "OFF")
            .define("JPEGXL_ENABLE_TOOLS", "OFF")
            .define("JPEGXL_ENABLE_JPEGLI_LIBJPEG", "ON")
            .define("JPEGXL_ENABLE_SJPEG", "OFF");

        // Build both jpegli-static and jxl_extras-internal if butteraugli is enabled
        if butteraugli_enabled {
            config.build_target("jxl_extras-internal");
        }
        config.build_target("jpegli-static");

        let dst = config.build();

        println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
        println!("cargo:rustc-link-lib=static=jpegli-static");

        if butteraugli_enabled {
            println!("cargo:rustc-link-lib=static=jxl_extras-internal");
            println!(
                "cargo:rustc-link-search=native={}/build/third_party",
                dst.display()
            );
            println!("cargo:rustc-link-lib=static=lcms2");
        }

        // Link highway (SIMD library)
        println!(
            "cargo:rustc-link-search=native={}/build/third_party/highway",
            dst.display()
        );
        println!("cargo:rustc-link-lib=static=hwy");

        // Link C++ standard library
        println!("cargo:rustc-link-lib=stdc++");

        // Link math library
        println!("cargo:rustc-link-lib=m");
    }
}
