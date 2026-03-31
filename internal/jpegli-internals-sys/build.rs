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

    // Check if jpegli-cpp submodule exists
    if !jpegli_root.join("CMakeLists.txt").exists() {
        panic!(
            "jpegli-cpp submodule not found at {:?}. Run: git submodule update --init --recursive",
            jpegli_root
        );
    }

    let butteraugli_enabled = env::var("CARGO_FEATURE_BUTTERAUGLI").is_ok();

    // Try to find a pre-built library first (for faster iteration during development)
    let prebuilt = find_prebuilt_library(&jpegli_root, butteraugli_enabled);

    let build_dir = if let Some(prebuilt_dir) = prebuilt {
        println!(
            "cargo:warning=Using pre-built jpegli from {:?}",
            prebuilt_dir
        );
        prebuilt_dir
    } else {
        // Build using cmake
        build_with_cmake(&jpegli_root, &out_dir, &target, &host, butteraugli_enabled)
    };

    // Build and link the C wrapper for butteraugli if enabled
    if butteraugli_enabled {
        build_butteraugli_wrapper(&manifest_dir, &jpegli_root, &build_dir, &target);
    }

    // Build the jpegli test FFI wrapper (for fast math, AQ functions)
    build_jpegli_test_ffi(&jpegli_root, &build_dir, &target);

    // Link the libraries
    link_libraries(&build_dir, &target, butteraugli_enabled);
}

/// Build the butteraugli C wrapper using cc crate
fn build_butteraugli_wrapper(
    manifest_dir: &Path,
    jpegli_root: &Path,
    build_dir: &Path,
    target: &str,
) {
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

    build.compile("butteraugli_c");
}

/// Build the jpegli test FFI wrapper (fast math, AQ functions)
fn build_jpegli_test_ffi(jpegli_root: &Path, build_dir: &Path, target: &str) {
    let ffi_source = jpegli_root.join("lib/extras/jpegli_test_ffi.cc");

    // The FFI source MUST exist - it's required for parity testing.
    // If missing, the jpegli-cpp submodule is likely on the wrong branch.
    if !ffi_source.exists() {
        panic!(
            "\n\n\
            ========================================================================\n\
            ERROR: jpegli_test_ffi.cc not found at {:?}\n\
            \n\
            The jpegli-cpp submodule appears to be on the wrong branch.\n\
            \n\
            Fix with:\n\
            \n\
            cd internal/jpegli-cpp && git checkout instrumented\n\
            \n\
            Or reinitialize the submodule:\n\
            \n\
            git submodule update --init --remote internal/jpegli-cpp\n\
            ========================================================================\n\n",
            ffi_source
        );
    }

    // Also check for the header
    let ffi_header = jpegli_root.join("lib/extras/jpegli_test_ffi.h");
    if !ffi_header.exists() {
        panic!(
            "ERROR: jpegli_test_ffi.h not found at {:?}. Submodule may be corrupted.",
            ffi_header
        );
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
    }

    build.compile("jpegli_test_ffi");
}

/// Find a pre-built library in common locations
fn find_prebuilt_library(jpegli_root: &Path, butteraugli_enabled: bool) -> Option<PathBuf> {
    // Check standard build directories
    let candidates = [
        "build",
        "build_release",
        "build_test",
        "cmake-build-release",
    ];

    for candidate in &candidates {
        let build_dir = jpegli_root.join(candidate);
        let lib_dir = build_dir.join("lib");

        // Check if the required library exists
        let jpegli_lib = if cfg!(windows) {
            lib_dir.join("jpegli-static.lib")
        } else {
            lib_dir.join("libjpegli-static.a")
        };

        if !jpegli_lib.exists() {
            continue;
        }

        // If butteraugli is enabled, we also need jxl_extras-internal
        if butteraugli_enabled {
            let extras_lib = if cfg!(windows) {
                lib_dir.join("jxl_extras-internal.lib")
            } else {
                lib_dir.join("libjxl_extras-internal.a")
            };
            if !extras_lib.exists() {
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
        // Windows MSVC: Use ClangCL for Highway SIMD compatibility
        let mut exeflags = "MSVCRTD.lib".to_string();
        if cfg!(asan) {
            exeflags.push_str(
                " clang_rt.asan_dynamic-x86_64.lib clang_rt.asan_dynamic_runtime_thunk-x86_64.lib",
            );
        }

        config
            .generator_toolset("ClangCL")
            .define(
                "CMAKE_VS_GLOBALS",
                "UseMultiToolTask=true;EnforceProcessCountAcrossBuilds=true",
            )
            .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded")
            .define("CMAKE_EXE_LINKER_FLAGS", exeflags)
            .cflag("/Zl");
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

/// Link the built libraries
fn link_libraries(build_dir: &Path, target: &str, butteraugli_enabled: bool) {
    // Try lib/ first, fall back to lib64/ (from jpegxl-src pattern)
    let lib_dir = {
        let lib = build_dir.join("lib");
        if lib.exists() {
            lib
        } else {
            let lib64 = build_dir.join("lib64");
            if lib64.exists() {
                lib64
            } else {
                // Fall back to lib/ — cmake may create it during linking
                lib
            }
        }
    };

    let hwy_dir = build_dir.join("third_party").join("highway");
    let brotli_dir = build_dir.join("third_party").join("brotli");

    // Add library search paths
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    if hwy_dir.exists() {
        println!("cargo:rustc-link-search=native={}", hwy_dir.display());
    }

    if brotli_dir.exists() {
        println!("cargo:rustc-link-search=native={}", brotli_dir.display());
    }

    // Link jpegli-static
    println!("cargo:rustc-link-lib=static=jpegli-static");

    // Link highway (SIMD library)
    println!("cargo:rustc-link-lib=static=hwy");

    // Link butteraugli/extras if enabled
    if butteraugli_enabled {
        println!("cargo:rustc-link-lib=static=jxl_extras-internal");

        // jxl_extras may need additional dependencies
        let third_party = build_dir.join("third_party");
        if third_party.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                third_party.display()
            );
        }

        // lcms2 may be built as a static lib
        let lcms2_lib = if target.contains("msvc") {
            third_party.join("lcms2.lib")
        } else {
            third_party.join("liblcms2.a")
        };
        if lcms2_lib.exists() {
            println!("cargo:rustc-link-lib=static=lcms2");
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
