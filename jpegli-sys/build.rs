//! Build script for jpegli-sys.
//!
//! This will eventually build the C++ jpegli library and link it.
//! For now, it's a placeholder that doesn't link anything.

fn main() {
    // TODO: Build and link jpegli C++ library
    // For now, we're just providing the Rust interface without the actual C++ library.

    // When ready, uncomment and configure:
    // let dst = cmake::Config::new("../../")
    //     .define("BUILD_TESTING", "OFF")
    //     .define("JPEGXL_ENABLE_JPEGLI", "ON")
    //     .build();
    //
    // println!("cargo:rustc-link-search=native={}/lib", dst.display());
    // println!("cargo:rustc-link-lib=static=jpegli-static");

    println!("cargo:rerun-if-changed=build.rs");
}
