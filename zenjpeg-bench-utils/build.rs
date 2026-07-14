//! Emit `missing_jpegli_cpp` when the jpegli-cpp submodule is absent, so
//! the `cjpegli-ffi` helpers degrade to a runtime error instead of a
//! compile failure against the (empty) jpegli-internals-sys crate.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(missing_jpegli_cpp)");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let jpegli_root = manifest_dir
        .parent()
        .unwrap()
        .join("internal")
        .join("jpegli-cpp");
    // Re-probe when the submodule appears or vanishes; without this cargo
    // replays a cached verdict from a tree state that no longer exists.
    println!(
        "cargo:rerun-if-changed={}",
        jpegli_root.join("CMakeLists.txt").display()
    );
    if !jpegli_root.join("CMakeLists.txt").exists() {
        println!("cargo:rustc-cfg=missing_jpegli_cpp");
    }
}
