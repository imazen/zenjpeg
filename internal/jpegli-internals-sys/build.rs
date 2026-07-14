//! Build script for jpegli-internals-sys.
//!
//! This builds the C++ jpegli library from the jpegli-cpp submodule and links it for FFI testing.
//!
//! ## Supported Platforms
//! - Linux (GCC, Clang)
//! - macOS (Clang)
//! - Windows (MSVC, MinGW)
//!
//! ## Features
//! - `butteraugli`: Links with jxl_extras-internal for butteraugli FFI

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let jpegli_root = manifest_dir.parent().unwrap().join("jpegli-cpp");

    // Let rustc know about our conditional cfg (silences unexpected-cfg lint)
    println!("cargo:rustc-check-cfg=cfg(missing_jpegli_cpp)");

    // Re-probe when the submodule appears or vanishes; without this cargo
    // replays a cached verdict from a tree state that no longer exists.
    println!(
        "cargo:rerun-if-changed={}",
        jpegli_root.join("CMakeLists.txt").display()
    );
    println!("cargo:rerun-if-env-changed=ZENJPEG_SKIP_CPP");

    // Reasons to skip the C++ build and compile this crate EMPTY. Every
    // zenjpeg consumer of these bindings is behind `--features __ffi-tests`
    // (or the `cjpegli-ffi` bench-utils feature, which degrades in lockstep
    // via the `available` metadata below), so the Rust test suite still
    // builds and runs — only the C++-parity tests surface a runtime error
    // via the `missing_jpegli_cpp` cfg instead of a compile failure.
    //
    //   1. `ZENJPEG_SKIP_CPP` set — explicit opt-out for a flaky or absent
    //      toolchain (`ZENJPEG_SKIP_CPP=1 cargo test` always builds).
    //   2. the jpegli-cpp submodule is absent (bare clone).
    if env::var_os("ZENJPEG_SKIP_CPP").is_some() {
        degrade("ZENJPEG_SKIP_CPP is set");
        return;
    }
    if !jpegli_root.join("CMakeLists.txt").exists() {
        degrade(&format!(
            "jpegli-cpp submodule not found at {jpegli_root:?} \
             (init with: git submodule update --init --recursive)"
        ));
        return;
    }

    let butteraugli_enabled = env::var("CARGO_FEATURE_BUTTERAUGLI").is_ok();

    // Attempt the full C++ build. cmake failures panic (caught here); the cc
    // wrapper compiles use `try_compile` and return `Err`. ANY failure — a
    // broken or missing toolchain, a transient compiler crash, a missing
    // system header — degrades to the empty crate instead of failing the
    // whole workspace build, so `cargo test` keeps working when C++
    // compilation is unreliable.
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build_cpp(
            &manifest_dir,
            &jpegli_root,
            &out_dir,
            &target,
            &host,
            butteraugli_enabled,
        )
    }));
    match attempt {
        // C++ available: dependents (zenjpeg-bench-utils) read this metadata
        // to enable their own FFI code in lockstep — see that crate's
        // build.rs, which keys off `DEP_JPEGLI_INTERNALS_FFI_AVAILABLE`.
        Ok(Ok(())) => println!("cargo:available=1"),
        Ok(Err(e)) => degrade(&format!("C++ jpegli compile failed: {e}")),
        Err(_) => degrade("C++ jpegli build panicked (cmake/toolchain failure)"),
    }
}

/// Compile this crate EMPTY (the `missing_jpegli_cpp` cfg) and tell direct
/// dependents that C++ is unavailable (`available=0`), so the Rust test
/// suite still builds. `reason` explains why in the cargo warning.
fn degrade(reason: &str) {
    println!(
        "cargo:warning=jpegli-internals-sys: {reason} — building EMPTY crate so the \
         Rust test suite still builds. C++-parity tests (--features __ffi-tests) are \
         disabled until a working C++ toolchain + jpegli-cpp submodule are present."
    );
    println!("cargo:rustc-cfg=missing_jpegli_cpp");
    println!("cargo:available=0");
}

/// The full C++ build + link. Returns `Err` on a cc-wrapper compile failure;
/// cmake failures panic inside `build_with_cmake` and are caught by the
/// caller's `catch_unwind`. Either way the caller degrades gracefully.
fn build_cpp(
    manifest_dir: &Path,
    jpegli_root: &Path,
    out_dir: &Path,
    target: &str,
    host: &str,
    butteraugli_enabled: bool,
) -> Result<(), String> {
    // Try to find a pre-built library first (for faster iteration during development)
    let prebuilt = find_prebuilt_library(jpegli_root, butteraugli_enabled);

    let build_dir = if let Some(prebuilt_dir) = prebuilt {
        println!("cargo:warning=Using pre-built jpegli from {prebuilt_dir:?}");
        prebuilt_dir
    } else {
        // Build using cmake (panics on failure — caught by the caller)
        build_with_cmake(jpegli_root, out_dir, target, host, butteraugli_enabled)
    };

    // Build and link the C wrapper for butteraugli if enabled
    if butteraugli_enabled {
        build_butteraugli_wrapper(manifest_dir, jpegli_root, &build_dir, target)?;
    }

    // Build the jpegli test FFI wrapper (for fast math, AQ functions)
    build_jpegli_test_ffi(jpegli_root, &build_dir, target)?;

    // Link the libraries
    link_libraries(&build_dir, target, butteraugli_enabled);
    Ok(())
}

/// Build the butteraugli C wrapper using cc crate. Returns `Err` (rather than
/// `process::exit`) on compile failure so the caller can degrade gracefully.
fn build_butteraugli_wrapper(
    manifest_dir: &Path,
    jpegli_root: &Path,
    build_dir: &Path,
    target: &str,
) -> Result<(), String> {
    println!("cargo:rerun-if-changed=cpp/butteraugli_c.cc");
    println!("cargo:rerun-if-changed=cpp/butteraugli_c.h");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .file(manifest_dir.join("cpp/butteraugli_c.cc"))
        // Include paths for jpegli-cpp headers
        .include(jpegli_root)
        .include(jpegli_root.join("lib"))
        // Include path for generated config headers
        .include(build_dir.join("lib/include"))
        // Local header
        .include(manifest_dir.join("cpp"))
        // Optimization
        .opt_level(2);

    // Platform-specific C++ standard
    if target.contains("msvc") {
        build.flag("/std:c++17");
    } else {
        build.flag("-std=c++17");
    }

    // Suppress some warnings in C++ code
    if !target.contains("msvc") {
        build.flag("-Wno-unused-parameter");
    }

    // try_compile returns Err instead of process::exit(1) on failure.
    build
        .try_compile("butteraugli_c")
        .map_err(|e| format!("butteraugli_c wrapper: {e}"))
}

/// Build the jpegli test FFI wrapper (fast math, AQ functions). Returns `Err`
/// (rather than `panic`/`process::exit`) on a missing source or compile
/// failure so the caller can degrade to an empty crate.
fn build_jpegli_test_ffi(jpegli_root: &Path, build_dir: &Path, target: &str) -> Result<(), String> {
    let ffi_source = jpegli_root.join("lib/extras/jpegli_test_ffi.cc");

    // The FFI source is required for parity testing. If missing, the
    // jpegli-cpp submodule is on the wrong branch — degrade rather than block
    // the whole test build (the warning tells the user how to fix it).
    if !ffi_source.exists() {
        return Err(format!(
            "jpegli_test_ffi.cc not found at {ffi_source:?} — the jpegli-cpp submodule \
             is on the wrong branch (fix: cd internal/jpegli-cpp && git checkout instrumented, \
             or git submodule update --init --remote internal/jpegli-cpp)"
        ));
    }

    // Also check for the header
    let ffi_header = jpegli_root.join("lib/extras/jpegli_test_ffi.h");
    if !ffi_header.exists() {
        return Err(format!(
            "jpegli_test_ffi.h not found at {ffi_header:?} (submodule may be corrupted)"
        ));
    }

    println!("cargo:rerun-if-changed={}", ffi_source.display());
    println!("cargo:rerun-if-changed={}", ffi_header.display());

    // Success message - confirms instrumented C++ is available
    println!("cargo:warning=✓ Building with instrumented C++ FFI (jpegli_test_ffi)");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .file(&ffi_source)
        // Include paths for jpegli-cpp headers
        .include(jpegli_root)
        .include(jpegli_root.join("lib"))
        // Include path for generated config headers
        .include(build_dir.join("lib/include"))
        // Include third_party for highway
        .include(jpegli_root.join("third_party"))
        // Optimization
        .opt_level(2);

    // Platform-specific C++ standard
    if target.contains("msvc") {
        build.flag("/std:c++17");
    } else {
        build.flag("-std=c++17");
    }

    // Suppress some warnings in C++ code
    if !target.contains("msvc") {
        build.flag("-Wno-unused-parameter");
        build.flag("-Wno-sign-compare");
        // Promote infinite-recursion to a hard error so the next typo
        // (like the aligned_alloc_xplat self-call we shipped for two weeks)
        // breaks the build instead of silently producing an FFI that
        // returns zero buffers at runtime. Clang-specific; GCC ignores it
        // because the warning name doesn't exist there.
        build.flag_if_supported("-Werror=infinite-recursion");
    }

    // try_compile returns Err instead of process::exit(1) on failure.
    build
        .try_compile("jpegli_test_ffi")
        .map_err(|e| format!("jpegli_test_ffi wrapper: {e}"))
}

/// Find a pre-built library in common locations
fn find_prebuilt_library(jpegli_root: &Path, butteraugli_enabled: bool) -> Option<PathBuf> {
    let candidates = [
        "build",
        "build_release",
        "build_test",
        "cmake-build-release",
    ];

    let (prefix, ext) = if cfg!(windows) {
        ("", "lib")
    } else {
        ("lib", "a")
    };

    for candidate in &candidates {
        let build_dir = jpegli_root.join(candidate);

        // Check lib/ and lib/Release/ (VS generator puts libs in Release/)
        let found_jpegli = ["lib", "lib/Release"].iter().any(|sub| {
            build_dir
                .join(sub)
                .join(format!("{prefix}jpegli-static.{ext}"))
                .exists()
        });

        if !found_jpegli {
            continue;
        }

        if butteraugli_enabled {
            let found_extras = ["lib", "lib/Release"].iter().any(|sub| {
                build_dir
                    .join(sub)
                    .join(format!("{prefix}jxl_extras-internal.{ext}"))
                    .exists()
            });
            if !found_extras {
                continue;
            }
        }

        return Some(build_dir);
    }

    None
}

/// Build jpegli using cmake
///
/// Cross-platform build logic modeled after jpegxl-src from libjxl-rs.
fn build_with_cmake(
    jpegli_root: &Path,
    out_dir: &Path,
    target: &str,
    _host: &str,
    butteraugli_enabled: bool,
) -> PathBuf {
    let mut config = cmake::Config::new(jpegli_root);

    // Basic configuration — disable everything we don't need
    config
        .define("BUILD_TESTING", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("JPEGXL_STATIC", "ON")
        .define("JPEGXL_ENABLE_DOXYGEN", "OFF")
        .define("JPEGXL_ENABLE_MANPAGES", "OFF")
        .define("JPEGXL_ENABLE_BENCHMARK", "OFF")
        .define("JPEGXL_ENABLE_EXAMPLES", "OFF")
        .define("JPEGXL_ENABLE_JNI", "OFF")
        .define("JPEGXL_ENABLE_SJPEG", "OFF")
        .define("JPEGXL_ENABLE_OPENEXR", "OFF")
        .define("JPEGXL_ENABLE_SKCMS", "OFF")
        .define("JPEGXL_ENABLE_TCMALLOC", "OFF")
        .define("JPEGXL_ENABLE_FUZZERS", "OFF")
        .define("JPEGXL_ENABLE_VIEWERS", "OFF")
        .define("JPEGXL_BUNDLE_LIBPNG", "OFF")
        .define("JPEGXL_ENABLE_JPEGLI_LIBJPEG", "ON");

    // For butteraugli, we need jxl_extras which requires JPEGXL_ENABLE_TOOLS
    // (jxl_extras.cmake is only included when JPEGXL_ENABLE_TOOLS or BUILD_TESTING is ON)
    if butteraugli_enabled {
        config.define("JPEGXL_ENABLE_TOOLS", "ON");
    } else {
        config.define("JPEGXL_ENABLE_TOOLS", "OFF");
    }

    // Release build for performance
    config.profile("Release");

    // Parallel build support (from jpegxl-src pattern)
    if let Ok(p) = std::thread::available_parallelism() {
        config.env("CMAKE_BUILD_PARALLEL_LEVEL", format!("{p}"));
    }

    // Sanitizer support (from jpegxl-src pattern)
    if cfg!(asan) {
        config
            .env("SANITIZER", "asan")
            .cflag("-g -DADDRESS_SANITIZER -fsanitize=address")
            .cxxflag("-g -DADDRESS_SANITIZER -fsanitize=address")
            .define("JPEGXL_ENABLE_TCMALLOC", "OFF");
    } else if cfg!(tsan) {
        config
            .env("SANITIZER", "tsan")
            .cflag("-g -DTHREAD_SANITIZER -fsanitize=thread")
            .cxxflag("-g -DTHREAD_SANITIZER -fsanitize=thread")
            .define("JPEGXL_ENABLE_TCMALLOC", "OFF");
    }

    // Platform-specific settings (from jpegxl-src pattern)
    if target.contains("msvc") {
        // Windows MSVC: Use ClangCL for Highway SIMD compatibility.
        // Use MultiThreadedDLL (/MD) to match the cc crate's default — Rust on
        // MSVC defaults to dynamic CRT, so cc-compiled wrappers (jpegli_test_ffi,
        // butteraugli_c) use /MD. Mixing /MT cmake with /MD cc causes LNK2038.
        // jpegxl-src uses /MT+/Zl because it has no cc-compiled wrappers.
        config
            .generator_toolset("ClangCL")
            .define(
                "CMAKE_VS_GLOBALS",
                "UseMultiToolTask=true;EnforceProcessCountAcrossBuilds=true",
            )
            .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");

        if cfg!(asan) {
            config.define(
                "CMAKE_EXE_LINKER_FLAGS",
                "clang_rt.asan_dynamic-x86_64.lib clang_rt.asan_dynamic_runtime_thunk-x86_64.lib",
            );
        }
    }

    // Build the appropriate target(s)
    // Note: jxl_extras-internal depends on jpegli-static, so building extras
    // will also build jpegli-static
    if butteraugli_enabled {
        config.build_target("jxl_extras-internal");
    } else {
        config.build_target("jpegli-static");
    }

    let prefix = config.build();

    // Detect lib directory — some platforms use lib64 (from jpegxl-src pattern)
    let lib_dir = prefix.join("lib");
    if !lib_dir.exists() {
        let lib64_dir = prefix.join("lib64");
        if lib64_dir.exists() {
            // Symlink lib64 → lib so downstream code can use consistent paths
            #[cfg(unix)]
            {
                let _ = std::os::unix::fs::symlink(&lib64_dir, &lib_dir);
            }
        }
    }

    // The cmake crate puts built files in out_dir/build
    out_dir.join("build")
}

/// Find the directory containing a static library, checking common paths.
///
/// On Windows with VS generators, cmake puts libraries in `lib/Release/` or
/// `lib/Debug/` instead of directly in `lib/`.
fn find_lib_dir(build_dir: &Path, lib_name: &str, target: &str) -> PathBuf {
    let (prefix, ext) = if target.contains("msvc") {
        ("", "lib")
    } else {
        ("lib", "a")
    };
    let filename = format!("{prefix}{lib_name}.{ext}");

    // Search order: lib/, lib/Release/, lib64/, lib64/Release/
    let candidates = [
        build_dir.join("lib"),
        build_dir.join("lib").join("Release"),
        build_dir.join("lib64"),
        build_dir.join("lib64").join("Release"),
    ];

    for candidate in &candidates {
        if candidate.join(&filename).exists() {
            return candidate.clone();
        }
    }

    // Fall back to lib/ — cmake may create it during linking
    build_dir.join("lib")
}

/// Link the built libraries
fn link_libraries(build_dir: &Path, target: &str, butteraugli_enabled: bool) {
    let lib_dir = find_lib_dir(build_dir, "jpegli-static", target);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Highway: check both third_party/highway/ and third_party/highway/Release/
    let hwy_base = build_dir.join("third_party").join("highway");
    for sub in ["", "Release"] {
        let dir = if sub.is_empty() {
            hwy_base.clone()
        } else {
            hwy_base.join(sub)
        };
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    // Brotli
    let brotli_base = build_dir.join("third_party").join("brotli");
    for sub in ["", "Release"] {
        let dir = if sub.is_empty() {
            brotli_base.clone()
        } else {
            brotli_base.join(sub)
        };
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    // Link jpegli-static
    println!("cargo:rustc-link-lib=static=jpegli-static");

    // Link highway (SIMD library)
    println!("cargo:rustc-link-lib=static=hwy");

    // Link butteraugli/extras if enabled
    if butteraugli_enabled {
        println!("cargo:rustc-link-lib=static=jxl_extras-internal");

        // jxl_extras may need additional dependencies — search third_party tree
        let third_party = build_dir.join("third_party");
        for sub in ["", "Release"] {
            let dir = if sub.is_empty() {
                third_party.clone()
            } else {
                third_party.join(sub)
            };
            if dir.exists() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }

        // lcms2 may be built as a static lib
        let lcms2_name = if target.contains("msvc") {
            "lcms2.lib"
        } else {
            "liblcms2.a"
        };
        for sub in ["", "Release"] {
            let dir = if sub.is_empty() {
                third_party.clone()
            } else {
                third_party.join(sub)
            };
            if dir.join(lcms2_name).exists() {
                println!("cargo:rustc-link-search=native={}", dir.display());
                println!("cargo:rustc-link-lib=static=lcms2");
                break;
            }
        }
    }

    // Platform-specific C++ runtime and system libraries
    link_cpp_runtime(target);
    link_system_libraries(target);
}

/// Link the appropriate C++ runtime for the target platform (from jpegxl-src pattern)
fn link_cpp_runtime(target: &str) {
    if target.contains("msvc") {
        // MSVC: C++ runtime is linked automatically
    } else if target.contains("apple") || target.contains("darwin") || target.contains("freebsd") {
        // macOS / FreeBSD: Use libc++
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") {
        // MinGW on Windows
        println!("cargo:rustc-link-lib=stdc++");
    } else {
        // Linux and other Unix-like systems
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// Link platform-specific system libraries
fn link_system_libraries(target: &str) {
    if target.contains("windows") {
        // Windows system libraries (if needed)
        // Most are linked automatically by MSVC
    } else if target.contains("apple") || target.contains("darwin") {
        // macOS system libraries
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    } else {
        // Linux/Unix system libraries
        println!("cargo:rustc-link-lib=m"); // Math library
        println!("cargo:rustc-link-lib=pthread"); // Threads
    }
}
