//! Build script for jpegli-sys.
//!
//! This builds the C++ jpegli library and links it for FFI testing.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Check if the jpegli C++ source is available
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let jpegli_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Check if build directory exists with pre-built library
    let prebuilt_lib = jpegli_root.join("build_test/lib/libjpegli-static.a");

    if prebuilt_lib.exists() {
        // Use pre-built library
        let lib_dir = prebuilt_lib.parent().unwrap();
        let hwy_lib = jpegli_root.join("build_test/third_party/highway/libhwy.a");

        println!(
            "cargo:rustc-link-search=native={}",
            lib_dir.display()
        );
        println!("cargo:rustc-link-lib=static=jpegli-static");

        if hwy_lib.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                hwy_lib.parent().unwrap().display()
            );
            println!("cargo:rustc-link-lib=static=hwy");
        }

        // Link C++ standard library
        println!("cargo:rustc-link-lib=stdc++");
    } else {
        // Build using cmake
        let dst = cmake::Config::new(jpegli_root)
            .define("BUILD_TESTING", "OFF")
            .define("JPEGXL_ENABLE_TOOLS", "OFF")
            .define("JPEGXL_ENABLE_JPEGLI_LIBJPEG", "ON")
            .define("JPEGXL_ENABLE_SJPEG", "OFF")
            .build_target("jpegli-static")
            .build();

        println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
        println!("cargo:rustc-link-lib=static=jpegli-static");

        // Link highway (SIMD library)
        println!(
            "cargo:rustc-link-search=native={}/build/third_party/highway",
            dst.display()
        );
        println!("cargo:rustc-link-lib=static=hwy");

        // Link C++ standard library
        println!("cargo:rustc-link-lib=stdc++");
    }
}
