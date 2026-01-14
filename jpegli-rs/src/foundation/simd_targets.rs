//! SIMD target definitions for multiversion dispatch.
//!
//! This module provides macros for consistent SIMD target specification across the codebase.
//! Use `simd_multiversion!` instead of `#[multiversion(targets(...))]` directly.
//!
//! # Usage
//!
//! ```ignore
//! // In any module:
//! crate::simd_multiversion! {
//!     pub fn my_simd_function(data: &[f32]) -> f32 {
//!         // Implementation using wide crate
//!         use wide::f32x8;
//!         let v = f32x8::from([data[0], data[1], data[2], data[3],
//!                              data[4], data[5], data[6], data[7]]);
//!         v.reduce_add()
//!     }
//! }
//! ```
//!
//! # Available Macros
//!
//! - `simd_multiversion!` - Primary targets (AVX2, NEON, WASM SIMD128)
//! - `simd_multiversion_extended!` - Adds AVX-512, dotprod, relaxed-simd
//! - `simd_multiversion_full!` - All supported extensions (slower compile)
//!
//! # Target Coverage
//!
//! | Macro | x86-64 | ARM64 | WASM |
//! |-------|--------|-------|------|
//! | `simd_multiversion` | AVX2+FMA, SSE4.1 | NEON | simd128 |
//! | `simd_multiversion_extended` | +AVX-512 | +dotprod | +relaxed-simd |
//! | `simd_multiversion_full` | +AVX-512CD | +fp16 | (same) |

/// Primary SIMD targets for most functions.
///
/// Covers ~99% of modern hardware with good performance.
#[macro_export]
macro_rules! simd_multiversion {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $($rest:tt)*
    ) => {
        #[multiversion::multiversion(targets(
            // x86-64 targets (in priority order)
            "x86_64+avx2+fma",              // AVX2+FMA (Haswell 2013+)
            "x86_64+sse4.1",                // SSE4.1 (Penryn 2008+)
            "x86+avx2+fma",                 // 32-bit AVX2
            "x86+sse4.1",                   // 32-bit SSE4.1
            // ARM64 targets
            "aarch64+neon",                 // All ARM64 (baseline)
            "arm+neon",                     // 32-bit ARM NEON
            // WebAssembly targets
            "wasm32+simd128",               // WASM SIMD (2021+)
        ))]
        $(#[$meta])*
        $vis fn $name $($rest)*
    };
}

/// Extended SIMD targets including AVX-512 and SVE2.
///
/// Use for hot paths where AVX-512/SVE2 provides significant benefit.
#[macro_export]
macro_rules! simd_multiversion_extended {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $($rest:tt)*
    ) => {
        #[multiversion::multiversion(targets(
            // x86-64 targets (in priority order)
            // AVX-512 subset (Skylake-X 2017+, Zen 4 2022+)
            "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
            "x86_64+avx2+fma",              // AVX2+FMA (Haswell 2013+)
            "x86_64+sse4.1",                // SSE4.1
            "x86+avx2+fma",
            "x86+sse4.1",
            // ARM64 targets (in priority order)
            "aarch64+neon+dotprod",         // ARMv8.2+ (A75 2017+)
            "aarch64+neon",                 // All ARM64 (baseline)
            "arm+neon",
            // WebAssembly targets (in priority order)
            "wasm32+simd128+relaxed-simd",  // Relaxed SIMD (2023+)
            "wasm32+simd128",               // WASM SIMD (2021+)
        ))]
        $(#[$meta])*
        $vis fn $name $($rest)*
    };
}

/// Full SIMD targets including all known extensions.
///
/// Use sparingly - increases compile time significantly.
#[macro_export]
macro_rules! simd_multiversion_full {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $($rest:tt)*
    ) => {
        #[multiversion::multiversion(targets(
            // x86-64 targets (in priority order)
            // Full AVX-512 (Icelake 2019+)
            "x86_64+avx512f+avx512bw+avx512dq+avx512vl+avx512cd",
            // AVX-512 basic (Skylake-X 2017+)
            "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
            "x86_64+avx2+fma",              // AVX2+FMA
            "x86_64+sse4.1",                // SSE4.1
            "x86+avx2+fma",
            "x86+sse4.1",
            // ARM64 targets (in priority order)
            "aarch64+neon+dotprod+fp16",    // ARMv8.2+ with extensions
            "aarch64+neon+dotprod",         // ARMv8.2+
            "aarch64+neon",                 // All ARM64
            "arm+neon",
            // WebAssembly targets (in priority order)
            "wasm32+simd128+relaxed-simd",  // Relaxed SIMD (2023+)
            "wasm32+simd128",               // WASM SIMD (2021+)
        ))]
        $(#[$meta])*
        $vis fn $name $($rest)*
    };
}

// Re-export for convenience
pub use simd_multiversion;
pub use simd_multiversion_extended;
pub use simd_multiversion_full;

#[cfg(test)]
mod tests {
    use wide::f32x8;

    crate::simd_multiversion! {
        fn test_simd_add(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
            let va = f32x8::from(*a);
            let vb = f32x8::from(*b);
            (va + vb).to_array()
        }
    }

    #[test]
    fn test_macro_generates_valid_function() {
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];
        let result = test_simd_add(&a, &b);
        assert_eq!(result, [3.0f32; 8]);
    }
}
