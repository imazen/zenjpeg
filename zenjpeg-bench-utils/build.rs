//! Emit `missing_jpegli_cpp` when the C++ jpegli build is unavailable, so the
//! `cjpegli-ffi` helpers degrade to a runtime error instead of a compile
//! failure against the (empty) jpegli-internals-sys crate.
//!
//! The authoritative signal is the sys crate's own build outcome: it sets
//! `links = "jpegli_internals_ffi"` and emits `cargo:available=0/1`, which
//! reaches us as `DEP_JPEGLI_INTERNALS_FFI_AVAILABLE`. That crate degrades on
//! a *missing submodule OR a failed/flaky C++ compile OR `ZENJPEG_SKIP_CPP`*,
//! and we must degrade in exact lockstep — otherwise our FFI code would try to
//! call symbols the empty sys crate never defined. The sys crate is our
//! dependency only under `cjpegli-ffi`, which is exactly when our FFI code
//! compiles, so the metadata is present whenever it matters.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(missing_jpegli_cpp)");
    println!("cargo:rerun-if-env-changed=ZENJPEG_SKIP_CPP");

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

    let cpp_available = match std::env::var("DEP_JPEGLI_INTERNALS_FFI_AVAILABLE") {
        // The sys crate is a dependency (cjpegli-ffi on) and reported its
        // verdict — follow it exactly.
        Ok(v) => v == "1",
        // No metadata: the sys crate is not linked (cjpegli-ffi off, so our
        // FFI code is feature-gated off and the cfg is irrelevant). Fall back
        // to the submodule-presence probe, still honoring the skip override.
        Err(_) => {
            std::env::var_os("ZENJPEG_SKIP_CPP").is_none()
                && jpegli_root.join("CMakeLists.txt").exists()
        }
    };
    if !cpp_available {
        println!("cargo:rustc-cfg=missing_jpegli_cpp");
    }
}
