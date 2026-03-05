//! Archmage-based SIMD implementations using capability tokens.
//!
//! These functions provide SIMD-optimized implementations using archmage's token
//! system for compile-time safety. The `mage_` prefix distinguishes them from
//! the `wide`-based autovectorized versions.
//!
//! # Token Model
//!
//! Each function requires a capability token that proves the CPU supports the
//! required instruction sets. Use `Desktop64::summon()` at encoder init to get
//! a token, then pass it to all SIMD functions. This gives zero per-call overhead.
//!
//! # Example
//!
//! ```rust,ignore
//! use archmage::{Desktop64, SimdToken};
//!
//! // Get token once at init
//! if let Some(token) = Desktop64::summon() {
//!     for block in blocks {
//!         mage_forward_dct_8x8(token, &input, &mut output);
//!     }
//! }
//! ```

#![cfg(target_arch = "x86_64")]

use archmage::{X64V3Token, X64V4Token, arcane, rite};
use core::arch::x86_64::*;
use safe_unaligned_simd::x86_64 as safe_simd;

// Re-export Desktop64 for callers
pub use archmage::Desktop64;

// ============================================================================
// DCT Constants
// ============================================================================

// WC4 coefficients for 4-point DCT
const WC4_0: f32 = 0.541196100146197;
const WC4_1: f32 = 1.3065629648763764;

// WC8 coefficients for 8-point DCT
const WC8_0: f32 = 0.5097955791041592;
const WC8_1: f32 = 0.6013448869350453;
const WC8_2: f32 = 0.8999762231364156;
const WC8_3: f32 = 2.5629154477415055;

const SQRT2: f32 = 1.41421356237;

// ============================================================================
// 8x8 Transpose (In-Place on Registers)
// ============================================================================

/// In-place 8x8 transpose on 8 __m256 registers using AVX.
///
/// After transpose, `r[i]` contains column i from all 8 original rows.
/// Uses the 3-phase unpack/shuffle/permute pattern.
#[rite]
fn mage_transpose_8x8_inplace_inner(_token: X64V3Token, r: &mut [__m256; 8]) {
    // Phase 1: Interleave pairs (unpack)
    let q0 = _mm256_unpacklo_ps(r[0], r[2]);
    let q1 = _mm256_unpacklo_ps(r[1], r[3]);
    let q2 = _mm256_unpackhi_ps(r[0], r[2]);
    let q3 = _mm256_unpackhi_ps(r[1], r[3]);
    let q4 = _mm256_unpacklo_ps(r[4], r[6]);
    let q5 = _mm256_unpacklo_ps(r[5], r[7]);
    let q6 = _mm256_unpackhi_ps(r[4], r[6]);
    let q7 = _mm256_unpackhi_ps(r[5], r[7]);

    // Phase 2: Another round of unpack
    let s0 = _mm256_unpacklo_ps(q0, q1);
    let s1 = _mm256_unpackhi_ps(q0, q1);
    let s2 = _mm256_unpacklo_ps(q2, q3);
    let s3 = _mm256_unpackhi_ps(q2, q3);
    let s4 = _mm256_unpacklo_ps(q4, q5);
    let s5 = _mm256_unpackhi_ps(q4, q5);
    let s6 = _mm256_unpacklo_ps(q6, q7);
    let s7 = _mm256_unpackhi_ps(q6, q7);

    // Phase 3: Exchange 128-bit halves
    r[0] = _mm256_permute2f128_ps::<0x20>(s0, s4);
    r[1] = _mm256_permute2f128_ps::<0x20>(s1, s5);
    r[2] = _mm256_permute2f128_ps::<0x20>(s2, s6);
    r[3] = _mm256_permute2f128_ps::<0x20>(s3, s7);
    r[4] = _mm256_permute2f128_ps::<0x31>(s0, s4);
    r[5] = _mm256_permute2f128_ps::<0x31>(s1, s5);
    r[6] = _mm256_permute2f128_ps::<0x31>(s2, s6);
    r[7] = _mm256_permute2f128_ps::<0x31>(s3, s7);
}

/// Public wrapper for in-place transpose. Token proves AVX is available.
#[arcane]
#[inline]
pub fn mage_transpose_8x8_inplace(token: X64V3Token, r: &mut [__m256; 8]) {
    mage_transpose_8x8_inplace_inner(token, r);
}

// ============================================================================
// DCT Butterfly Operations
// ============================================================================

/// DCT base case for N=2: out0 = in0 + in1, out1 = in0 - in1
///
/// This is pure AVX (add/sub), no FMA needed.
#[rite]
fn mage_dct1d_2_inner(_token: X64V3Token, m0: &mut __m256, m1: &mut __m256) {
    let in0 = *m0;
    let in1 = *m1;
    *m0 = _mm256_add_ps(in0, in1);
    *m1 = _mm256_sub_ps(in0, in1);
}

/// DCT for N=4 using FMA for the weighted operations.
#[rite]
fn mage_dct1d_4_inner(token: X64V3Token, m: &mut [__m256; 4]) {
    let wc4_0 = _mm256_set1_ps(WC4_0);
    let wc4_1 = _mm256_set1_ps(WC4_1);
    let sqrt2 = _mm256_set1_ps(SQRT2);

    // AddReverse<2>: tmp[0:2] = m[0:2] + reverse(m[2:4])
    let t0 = _mm256_add_ps(m[0], m[3]);
    let t1 = _mm256_add_ps(m[1], m[2]);

    // SubReverse<2>
    let t2 = _mm256_sub_ps(m[0], m[3]);
    let t3 = _mm256_sub_ps(m[1], m[2]);

    // DCT1D<2> on first half
    let r0 = _mm256_add_ps(t0, t1);
    let r1 = _mm256_sub_ps(t0, t1);

    // Multiply by WC4
    let t2_scaled = _mm256_mul_ps(t2, wc4_0);
    let t3_scaled = _mm256_mul_ps(t3, wc4_1);

    // DCT1D<2> on second half
    let r2 = _mm256_add_ps(t2_scaled, t3_scaled);
    let r3 = _mm256_sub_ps(t2_scaled, t3_scaled);

    // B<2>: r2 = r2 * sqrt2 + r3 (use FMA)
    let _ = token; // FMA token proves FMA available
    let r2_final = _mm256_fmadd_ps(r2, sqrt2, r3);

    // InverseEvenOdd<4>: interleave
    m[0] = r0;
    m[1] = r2_final;
    m[2] = r1;
    m[3] = r3;
}

/// DCT for N=8 using FMA. Processes 8 independent 8-point DCTs in parallel.
#[rite]
fn mage_dct1d_8_inner(token: X64V3Token, m: &mut [__m256; 8]) {
    let wc8_0 = _mm256_set1_ps(WC8_0);
    let wc8_1 = _mm256_set1_ps(WC8_1);
    let wc8_2 = _mm256_set1_ps(WC8_2);
    let wc8_3 = _mm256_set1_ps(WC8_3);
    let sqrt2 = _mm256_set1_ps(SQRT2);

    // AddReverse<4>: tmp[0:4] = m[0:4] + reverse(m[4:8])
    let t0 = _mm256_add_ps(m[0], m[7]);
    let t1 = _mm256_add_ps(m[1], m[6]);
    let t2 = _mm256_add_ps(m[2], m[5]);
    let t3 = _mm256_add_ps(m[3], m[4]);

    // SubReverse<4>
    let t4 = _mm256_sub_ps(m[0], m[7]);
    let t5 = _mm256_sub_ps(m[1], m[6]);
    let t6 = _mm256_sub_ps(m[2], m[5]);
    let t7 = _mm256_sub_ps(m[3], m[4]);

    // DCT1D<4> on first half
    let mut first = [t0, t1, t2, t3];
    mage_dct1d_4_inner(token, &mut first);

    // Multiply by WC8
    let t4_scaled = _mm256_mul_ps(t4, wc8_0);
    let t5_scaled = _mm256_mul_ps(t5, wc8_1);
    let t6_scaled = _mm256_mul_ps(t6, wc8_2);
    let t7_scaled = _mm256_mul_ps(t7, wc8_3);

    // DCT1D<4> on second half
    let mut second = [t4_scaled, t5_scaled, t6_scaled, t7_scaled];
    mage_dct1d_4_inner(token, &mut second);

    // B<4>: cumulative sum with FMA
    // second[0] = second[0] * sqrt2 + second[1]
    second[0] = _mm256_fmadd_ps(second[0], sqrt2, second[1]);
    // second[1] += second[2]
    second[1] = _mm256_add_ps(second[1], second[2]);
    // second[2] += second[3]
    second[2] = _mm256_add_ps(second[2], second[3]);
    // second[3] stays the same

    // InverseEvenOdd<8>: interleave
    m[0] = first[0];
    m[1] = second[0];
    m[2] = first[1];
    m[3] = second[1];
    m[4] = first[2];
    m[5] = second[2];
    m[6] = first[3];
    m[7] = second[3];
}

// ============================================================================
// Full 8x8 Forward DCT
// ============================================================================

/// Full 8x8 forward DCT using AVX2+FMA intrinsics via archmage tokens.
///
/// This is the archmage-based equivalent of `forward_dct_8x8_avx2`.
/// The token can be cached outside hot loops for zero overhead.
///
/// # Algorithm
///
/// 1. Load 8 rows into registers
/// 2. Transpose: reg[i] = column i (position i of all rows)
/// 3. Row DCT: 8 parallel 8-point DCTs
/// 4. Transpose: rearrange for column processing
/// 5. Column DCT: 8 parallel 8-point DCTs
/// 6. Scale by 1/8 and store
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Desktop64, SimdToken};
///
/// if let Some(token) = Desktop64::summon() {
///     let input: [f32; 64] = /* ... */;
///     let mut output = [0.0f32; 64];
///     mage_forward_dct_8x8(token, &input, &mut output);
/// }
/// ```
#[arcane]
#[inline]
pub fn mage_forward_dct_8x8(token: X64V3Token, input: &[f32; 64], output: &mut [f32; 64]) {
    let scale = _mm256_set1_ps(1.0 / 8.0);

    // Load 8 rows using safe SIMD load operations
    // Split input into 8 contiguous chunks of 8 f32s each
    let mut reg = [
        safe_simd::_mm256_loadu_ps(input[0..8].try_into().unwrap()),
        safe_simd::_mm256_loadu_ps(input[8..16].try_into().unwrap()),
        safe_simd::_mm256_loadu_ps(input[16..24].try_into().unwrap()),
        safe_simd::_mm256_loadu_ps(input[24..32].try_into().unwrap()),
        safe_simd::_mm256_loadu_ps(input[32..40].try_into().unwrap()),
        safe_simd::_mm256_loadu_ps(input[40..48].try_into().unwrap()),
        safe_simd::_mm256_loadu_ps(input[48..56].try_into().unwrap()),
        safe_simd::_mm256_loadu_ps(input[56..64].try_into().unwrap()),
    ];

    // Transpose: reg[i] = column i = [row0[i], row1[i], ..., row7[i]]
    mage_transpose_8x8_inplace_inner(token, &mut reg);

    // Row DCT: all 8 rows processed in parallel
    mage_dct1d_8_inner(token, &mut reg);

    // Scale by 1/8 after row DCT (first pass)
    for r in &mut reg {
        *r = _mm256_mul_ps(*r, scale);
    }

    // Transpose: reg[i][j] = coef[i, j] (row-major coefficient matrix)
    mage_transpose_8x8_inplace_inner(token, &mut reg);

    // Column DCT: all 8 columns processed in parallel
    mage_dct1d_8_inner(token, &mut reg);

    // Scale by 1/8 after col DCT (second pass) - total scaling: 1/64
    // Store using safe SIMD store operations
    safe_simd::_mm256_storeu_ps(
        (&mut output[0..8]).try_into().unwrap(),
        _mm256_mul_ps(reg[0], scale),
    );
    safe_simd::_mm256_storeu_ps(
        (&mut output[8..16]).try_into().unwrap(),
        _mm256_mul_ps(reg[1], scale),
    );
    safe_simd::_mm256_storeu_ps(
        (&mut output[16..24]).try_into().unwrap(),
        _mm256_mul_ps(reg[2], scale),
    );
    safe_simd::_mm256_storeu_ps(
        (&mut output[24..32]).try_into().unwrap(),
        _mm256_mul_ps(reg[3], scale),
    );
    safe_simd::_mm256_storeu_ps(
        (&mut output[32..40]).try_into().unwrap(),
        _mm256_mul_ps(reg[4], scale),
    );
    safe_simd::_mm256_storeu_ps(
        (&mut output[40..48]).try_into().unwrap(),
        _mm256_mul_ps(reg[5], scale),
    );
    safe_simd::_mm256_storeu_ps(
        (&mut output[48..56]).try_into().unwrap(),
        _mm256_mul_ps(reg[6], scale),
    );
    safe_simd::_mm256_storeu_ps(
        (&mut output[56..64]).try_into().unwrap(),
        _mm256_mul_ps(reg[7], scale),
    );
}

/// Wide-native 2D forward DCT: takes Block8x8f, returns Block8x8f.
///
/// This is the archmage equivalent of `dct::simd::forward_dct_8x8_wide`.
/// Zero conversion overhead when data is already in wide format.
#[arcane]
#[inline]
pub fn mage_forward_dct_8x8_wide(
    token: X64V3Token,
    input: &crate::foundation::simd_types::Block8x8f,
) -> crate::foundation::simd_types::Block8x8f {
    use crate::foundation::simd_types::Block8x8f;

    let scale = _mm256_set1_ps(1.0 / 8.0);

    // Cast Block8x8f to [[f32; 8]; 8] via bytemuck, then load into __m256 registers
    let rows: &[[f32; 8]; 8] = bytemuck::cast_ref(input);
    let mut reg: [__m256; 8] = [
        safe_simd::_mm256_loadu_ps(&rows[0]),
        safe_simd::_mm256_loadu_ps(&rows[1]),
        safe_simd::_mm256_loadu_ps(&rows[2]),
        safe_simd::_mm256_loadu_ps(&rows[3]),
        safe_simd::_mm256_loadu_ps(&rows[4]),
        safe_simd::_mm256_loadu_ps(&rows[5]),
        safe_simd::_mm256_loadu_ps(&rows[6]),
        safe_simd::_mm256_loadu_ps(&rows[7]),
    ];

    // Transpose: reg[i] = column i = [row0[i], row1[i], ..., row7[i]]
    mage_transpose_8x8_inplace_inner(token, &mut reg);

    // Row DCT: all 8 rows processed in parallel
    mage_dct1d_8_inner(token, &mut reg);

    // Scale by 1/8 after row DCT (first pass)
    for r in &mut reg {
        *r = _mm256_mul_ps(*r, scale);
    }

    // Transpose: reg[i][j] = coef[i, j] (row-major coefficient matrix)
    mage_transpose_8x8_inplace_inner(token, &mut reg);

    // Column DCT: all 8 columns processed in parallel
    mage_dct1d_8_inner(token, &mut reg);

    // Scale by 1/8 after col DCT (second pass) - total scaling: 1/64
    // Store back via bytemuck
    let mut output = Block8x8f::default();
    let out_rows: &mut [[f32; 8]; 8] = bytemuck::cast_mut(&mut output);
    safe_simd::_mm256_storeu_ps(&mut out_rows[0], _mm256_mul_ps(reg[0], scale));
    safe_simd::_mm256_storeu_ps(&mut out_rows[1], _mm256_mul_ps(reg[1], scale));
    safe_simd::_mm256_storeu_ps(&mut out_rows[2], _mm256_mul_ps(reg[2], scale));
    safe_simd::_mm256_storeu_ps(&mut out_rows[3], _mm256_mul_ps(reg[3], scale));
    safe_simd::_mm256_storeu_ps(&mut out_rows[4], _mm256_mul_ps(reg[4], scale));
    safe_simd::_mm256_storeu_ps(&mut out_rows[5], _mm256_mul_ps(reg[5], scale));
    safe_simd::_mm256_storeu_ps(&mut out_rows[6], _mm256_mul_ps(reg[6], scale));
    safe_simd::_mm256_storeu_ps(&mut out_rows[7], _mm256_mul_ps(reg[7], scale));
    output
}

// ============================================================================
// AVX-512 Dual-Block Forward DCT (processes 2 blocks simultaneously)
// ============================================================================

/// In-place 8x8 transpose on 8 __m512 registers, operating on BOTH 256-bit halves.
///
/// Each ZMM register holds two rows: [blockA_row[i], blockB_row[i]]
/// After transpose: [blockA_col[i], blockB_col[i]]
///
/// Uses extract/insert to process each 256-bit half with the AVX2 transpose pattern,
/// then recombines. This is correct because it keeps block A and B data separate.
#[rite]
fn mage_transpose_8x8_dual_inner(token: X64V4Token, r: &mut [__m512; 8]) {
    // Extract low (block A) and high (block B) halves
    let mut a: [__m256; 8] = [
        _mm512_castps512_ps256(r[0]),
        _mm512_castps512_ps256(r[1]),
        _mm512_castps512_ps256(r[2]),
        _mm512_castps512_ps256(r[3]),
        _mm512_castps512_ps256(r[4]),
        _mm512_castps512_ps256(r[5]),
        _mm512_castps512_ps256(r[6]),
        _mm512_castps512_ps256(r[7]),
    ];
    let mut b: [__m256; 8] = [
        _mm512_extractf32x8_ps::<1>(r[0]),
        _mm512_extractf32x8_ps::<1>(r[1]),
        _mm512_extractf32x8_ps::<1>(r[2]),
        _mm512_extractf32x8_ps::<1>(r[3]),
        _mm512_extractf32x8_ps::<1>(r[4]),
        _mm512_extractf32x8_ps::<1>(r[5]),
        _mm512_extractf32x8_ps::<1>(r[6]),
        _mm512_extractf32x8_ps::<1>(r[7]),
    ];

    // Transpose block A using AVX2 pattern
    // Phase 1: Interleave pairs
    let q0 = _mm256_unpacklo_ps(a[0], a[2]);
    let q1 = _mm256_unpacklo_ps(a[1], a[3]);
    let q2 = _mm256_unpackhi_ps(a[0], a[2]);
    let q3 = _mm256_unpackhi_ps(a[1], a[3]);
    let q4 = _mm256_unpacklo_ps(a[4], a[6]);
    let q5 = _mm256_unpacklo_ps(a[5], a[7]);
    let q6 = _mm256_unpackhi_ps(a[4], a[6]);
    let q7 = _mm256_unpackhi_ps(a[5], a[7]);

    // Phase 2
    let s0 = _mm256_unpacklo_ps(q0, q1);
    let s1 = _mm256_unpackhi_ps(q0, q1);
    let s2 = _mm256_unpacklo_ps(q2, q3);
    let s3 = _mm256_unpackhi_ps(q2, q3);
    let s4 = _mm256_unpacklo_ps(q4, q5);
    let s5 = _mm256_unpackhi_ps(q4, q5);
    let s6 = _mm256_unpacklo_ps(q6, q7);
    let s7 = _mm256_unpackhi_ps(q6, q7);

    // Phase 3: Exchange 128-bit halves
    a[0] = _mm256_permute2f128_ps::<0x20>(s0, s4);
    a[1] = _mm256_permute2f128_ps::<0x20>(s1, s5);
    a[2] = _mm256_permute2f128_ps::<0x20>(s2, s6);
    a[3] = _mm256_permute2f128_ps::<0x20>(s3, s7);
    a[4] = _mm256_permute2f128_ps::<0x31>(s0, s4);
    a[5] = _mm256_permute2f128_ps::<0x31>(s1, s5);
    a[6] = _mm256_permute2f128_ps::<0x31>(s2, s6);
    a[7] = _mm256_permute2f128_ps::<0x31>(s3, s7);

    // Transpose block B using same pattern
    let q0 = _mm256_unpacklo_ps(b[0], b[2]);
    let q1 = _mm256_unpacklo_ps(b[1], b[3]);
    let q2 = _mm256_unpackhi_ps(b[0], b[2]);
    let q3 = _mm256_unpackhi_ps(b[1], b[3]);
    let q4 = _mm256_unpacklo_ps(b[4], b[6]);
    let q5 = _mm256_unpacklo_ps(b[5], b[7]);
    let q6 = _mm256_unpackhi_ps(b[4], b[6]);
    let q7 = _mm256_unpackhi_ps(b[5], b[7]);

    let s0 = _mm256_unpacklo_ps(q0, q1);
    let s1 = _mm256_unpackhi_ps(q0, q1);
    let s2 = _mm256_unpacklo_ps(q2, q3);
    let s3 = _mm256_unpackhi_ps(q2, q3);
    let s4 = _mm256_unpacklo_ps(q4, q5);
    let s5 = _mm256_unpackhi_ps(q4, q5);
    let s6 = _mm256_unpacklo_ps(q6, q7);
    let s7 = _mm256_unpackhi_ps(q6, q7);

    b[0] = _mm256_permute2f128_ps::<0x20>(s0, s4);
    b[1] = _mm256_permute2f128_ps::<0x20>(s1, s5);
    b[2] = _mm256_permute2f128_ps::<0x20>(s2, s6);
    b[3] = _mm256_permute2f128_ps::<0x20>(s3, s7);
    b[4] = _mm256_permute2f128_ps::<0x31>(s0, s4);
    b[5] = _mm256_permute2f128_ps::<0x31>(s1, s5);
    b[6] = _mm256_permute2f128_ps::<0x31>(s2, s6);
    b[7] = _mm256_permute2f128_ps::<0x31>(s3, s7);

    // Recombine into ZMM registers
    let _ = token;
    r[0] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[0]), b[0]);
    r[1] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[1]), b[1]);
    r[2] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[2]), b[2]);
    r[3] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[3]), b[3]);
    r[4] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[4]), b[4]);
    r[5] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[5]), b[5]);
    r[6] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[6]), b[6]);
    r[7] = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a[7]), b[7]);
}

/// AVX-512 DCT base case for N=2: out0 = in0 + in1, out1 = in0 - in1
#[rite]
fn mage_dct1d_2_avx512_inner(_token: X64V4Token, m0: &mut __m512, m1: &mut __m512) {
    let in0 = *m0;
    let in1 = *m1;
    *m0 = _mm512_add_ps(in0, in1);
    *m1 = _mm512_sub_ps(in0, in1);
}

/// AVX-512 DCT for N=4 using FMA
#[rite]
fn mage_dct1d_4_avx512_inner(token: X64V4Token, m: &mut [__m512; 4]) {
    let wc4_0 = _mm512_set1_ps(WC4_0);
    let wc4_1 = _mm512_set1_ps(WC4_1);
    let sqrt2 = _mm512_set1_ps(SQRT2);

    // AddReverse<2>
    let t0 = _mm512_add_ps(m[0], m[3]);
    let t1 = _mm512_add_ps(m[1], m[2]);

    // SubReverse<2>
    let t2 = _mm512_sub_ps(m[0], m[3]);
    let t3 = _mm512_sub_ps(m[1], m[2]);

    // DCT1D<2> on first half
    let r0 = _mm512_add_ps(t0, t1);
    let r1 = _mm512_sub_ps(t0, t1);

    // Multiply by WC4
    let t2_scaled = _mm512_mul_ps(t2, wc4_0);
    let t3_scaled = _mm512_mul_ps(t3, wc4_1);

    // DCT1D<2> on second half
    let r2 = _mm512_add_ps(t2_scaled, t3_scaled);
    let r3 = _mm512_sub_ps(t2_scaled, t3_scaled);

    // B<2>: r2 = r2 * sqrt2 + r3 (FMA)
    let _ = token;
    let r2_final = _mm512_fmadd_ps(r2, sqrt2, r3);

    // InverseEvenOdd<4>
    m[0] = r0;
    m[1] = r2_final;
    m[2] = r1;
    m[3] = r3;
}

/// AVX-512 DCT for N=8 using FMA. Processes 16 independent 8-point DCTs in parallel
/// (8 from block A, 8 from block B).
#[rite]
fn mage_dct1d_8_avx512_inner(token: X64V4Token, m: &mut [__m512; 8]) {
    let wc8_0 = _mm512_set1_ps(WC8_0);
    let wc8_1 = _mm512_set1_ps(WC8_1);
    let wc8_2 = _mm512_set1_ps(WC8_2);
    let wc8_3 = _mm512_set1_ps(WC8_3);
    let sqrt2 = _mm512_set1_ps(SQRT2);

    // AddReverse<4>
    let t0 = _mm512_add_ps(m[0], m[7]);
    let t1 = _mm512_add_ps(m[1], m[6]);
    let t2 = _mm512_add_ps(m[2], m[5]);
    let t3 = _mm512_add_ps(m[3], m[4]);

    // SubReverse<4>
    let t4 = _mm512_sub_ps(m[0], m[7]);
    let t5 = _mm512_sub_ps(m[1], m[6]);
    let t6 = _mm512_sub_ps(m[2], m[5]);
    let t7 = _mm512_sub_ps(m[3], m[4]);

    // DCT1D<4> on first half
    let mut first = [t0, t1, t2, t3];
    mage_dct1d_4_avx512_inner(token, &mut first);

    // Multiply by WC8
    let t4_scaled = _mm512_mul_ps(t4, wc8_0);
    let t5_scaled = _mm512_mul_ps(t5, wc8_1);
    let t6_scaled = _mm512_mul_ps(t6, wc8_2);
    let t7_scaled = _mm512_mul_ps(t7, wc8_3);

    // DCT1D<4> on second half
    let mut second = [t4_scaled, t5_scaled, t6_scaled, t7_scaled];
    mage_dct1d_4_avx512_inner(token, &mut second);

    // B<4>: cumulative sum with FMA
    second[0] = _mm512_fmadd_ps(second[0], sqrt2, second[1]);
    second[1] = _mm512_add_ps(second[1], second[2]);
    second[2] = _mm512_add_ps(second[2], second[3]);

    // InverseEvenOdd<8>
    m[0] = first[0];
    m[1] = second[0];
    m[2] = first[1];
    m[3] = second[1];
    m[4] = first[2];
    m[5] = second[2];
    m[6] = first[3];
    m[7] = second[3];
}

/// AVX-512 dual-block forward DCT: processes TWO 8x8 blocks simultaneously.
///
/// **WARNING: Experimental - actually 2.3x SLOWER than AVX2 single-block!**
///
/// The transpose overhead (extract/AVX2/insert pattern) negates any benefit from
/// 512-bit arithmetic. 8x8 blocks fit AVX2 perfectly; AVX-512's 16-wide registers
/// just add packing/unpacking overhead. See CLAUDE.md "Failed Explorations" for details.
///
/// Kept for reference and potential future optimization with different data layouts.
///
/// # Arguments
///
/// * `token` - AVX-512F + FMA capability token
/// * `input_a` - First 8x8 block (64 f32s, row-major)
/// * `input_b` - Second 8x8 block (64 f32s, row-major)
/// * `output_a` - Output for first block
/// * `output_b` - Output for second block
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Avx512fToken, SimdToken};
///
/// if let Some(token) = X64V4Token::summon() {
///     let block_a = [0.0f32; 64];
///     let block_b = [0.0f32; 64];
///     let mut out_a = [0.0f32; 64];
///     let mut out_b = [0.0f32; 64];
///     mage_forward_dct_8x8_dual(token, &block_a, &block_b, &mut out_a, &mut out_b);
/// }
/// ```
#[arcane]
#[inline]
pub fn mage_forward_dct_8x8_dual(
    token: X64V4Token,
    input_a: &[f32; 64],
    input_b: &[f32; 64],
    output_a: &mut [f32; 64],
    output_b: &mut [f32; 64],
) {
    let scale = _mm512_set1_ps(1.0 / 8.0);

    // Load 8 rows from each block, interleaved into ZMM registers
    // ZMM[i] = [blockA_row[i], blockB_row[i]]
    let mut reg: [__m512; 8] = [
        // Combine two 256-bit loads into one 512-bit register
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[0..8].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[0..8].try_into().unwrap()),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[8..16].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[8..16].try_into().unwrap()),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[16..24].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[16..24].try_into().unwrap()),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[24..32].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[24..32].try_into().unwrap()),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[32..40].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[32..40].try_into().unwrap()),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[40..48].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[40..48].try_into().unwrap()),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[48..56].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[48..56].try_into().unwrap()),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                input_a[56..64].try_into().unwrap(),
            )),
            safe_simd::_mm256_loadu_ps(input_b[56..64].try_into().unwrap()),
        ),
    ];

    // Transpose both blocks (operates on each 256-bit half independently)
    mage_transpose_8x8_dual_inner(token, &mut reg);

    // Row DCT: all 16 rows (8 per block) processed in parallel
    mage_dct1d_8_avx512_inner(token, &mut reg);

    // Scale by 1/8 after row DCT (first pass)
    for r in &mut reg {
        *r = _mm512_mul_ps(*r, scale);
    }

    // Transpose both blocks again
    mage_transpose_8x8_dual_inner(token, &mut reg);

    // Column DCT: all 16 columns processed in parallel
    mage_dct1d_8_avx512_inner(token, &mut reg);

    // Scale by 1/8 after col DCT (second pass) - total scaling: 1/64
    // Store back to separate output arrays
    for i in 0..8 {
        let scaled = _mm512_mul_ps(reg[i], scale);
        // Extract low 256 bits (block A) and high 256 bits (block B)
        let lo = _mm512_castps512_ps256(scaled);
        let hi = _mm512_extractf32x8_ps::<1>(scaled);

        safe_simd::_mm256_storeu_ps((&mut output_a[i * 8..(i + 1) * 8]).try_into().unwrap(), lo);
        safe_simd::_mm256_storeu_ps((&mut output_b[i * 8..(i + 1) * 8]).try_into().unwrap(), hi);
    }
}

/// Wide-native dual-block forward DCT: takes two Block8x8f, returns two Block8x8f.
///
/// **WARNING: Experimental - actually 2.3x SLOWER than AVX2 single-block!**
/// See `mage_forward_dct_8x8_dual` docs for explanation.
#[arcane]
#[inline]
pub fn mage_forward_dct_8x8_wide_dual(
    token: X64V4Token,
    input_a: &crate::foundation::simd_types::Block8x8f,
    input_b: &crate::foundation::simd_types::Block8x8f,
) -> (
    crate::foundation::simd_types::Block8x8f,
    crate::foundation::simd_types::Block8x8f,
) {
    use crate::foundation::simd_types::Block8x8f;

    let scale = _mm512_set1_ps(1.0 / 8.0);

    // Cast Block8x8f to [[f32; 8]; 8]
    let rows_a: &[[f32; 8]; 8] = bytemuck::cast_ref(input_a);
    let rows_b: &[[f32; 8]; 8] = bytemuck::cast_ref(input_b);

    // Load interleaved
    let mut reg: [__m512; 8] = [
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[0])),
            safe_simd::_mm256_loadu_ps(&rows_b[0]),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[1])),
            safe_simd::_mm256_loadu_ps(&rows_b[1]),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[2])),
            safe_simd::_mm256_loadu_ps(&rows_b[2]),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[3])),
            safe_simd::_mm256_loadu_ps(&rows_b[3]),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[4])),
            safe_simd::_mm256_loadu_ps(&rows_b[4]),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[5])),
            safe_simd::_mm256_loadu_ps(&rows_b[5]),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[6])),
            safe_simd::_mm256_loadu_ps(&rows_b[6]),
        ),
        _mm512_insertf32x8::<1>(
            _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(&rows_a[7])),
            safe_simd::_mm256_loadu_ps(&rows_b[7]),
        ),
    ];

    // Transpose, row DCT, scale, transpose, col DCT
    mage_transpose_8x8_dual_inner(token, &mut reg);
    mage_dct1d_8_avx512_inner(token, &mut reg);

    // Scale by 1/8 after row DCT (first pass)
    for r in &mut reg {
        *r = _mm512_mul_ps(*r, scale);
    }

    mage_transpose_8x8_dual_inner(token, &mut reg);
    mage_dct1d_8_avx512_inner(token, &mut reg);

    // Scale by 1/8 after col DCT (second pass) - total scaling: 1/64
    // Store back
    let mut output_a = Block8x8f::default();
    let mut output_b = Block8x8f::default();
    let out_rows_a: &mut [[f32; 8]; 8] = bytemuck::cast_mut(&mut output_a);
    let out_rows_b: &mut [[f32; 8]; 8] = bytemuck::cast_mut(&mut output_b);

    for i in 0..8 {
        let scaled = _mm512_mul_ps(reg[i], scale);
        safe_simd::_mm256_storeu_ps(&mut out_rows_a[i], _mm512_castps512_ps256(scaled));
        safe_simd::_mm256_storeu_ps(&mut out_rows_b[i], _mm512_extractf32x8_ps::<1>(scaled));
    }

    (output_a, output_b)
}

// ============================================================================
// RGB to YCbCr Color Conversion
// ============================================================================

use crate::foundation::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
};

/// Convert 8 RGB pixels to YCbCr using AVX2+FMA.
///
/// Takes pre-separated R, G, B values as f32 arrays and produces Y, Cb, Cr output.
/// This is the core color matrix multiplication using FMA for precision.
///
/// Y  = 0.299*R + 0.587*G + 0.114*B
/// Cb = 128 - 0.169*R - 0.331*G + 0.500*B
/// Cr = 128 + 0.500*R - 0.419*G - 0.081*B
#[arcane]
#[inline]
pub fn mage_rgb_to_ycbcr_8px(
    _token: X64V3Token,
    r: &[f32; 8],
    g: &[f32; 8],
    b: &[f32; 8],
    y_out: &mut [f32; 8],
    cb_out: &mut [f32; 8],
    cr_out: &mut [f32; 8],
) {
    // Load input vectors
    let r_vec = safe_simd::_mm256_loadu_ps(r);
    let g_vec = safe_simd::_mm256_loadu_ps(g);
    let b_vec = safe_simd::_mm256_loadu_ps(b);

    // Coefficients
    let r_to_y = _mm256_set1_ps(YCBCR_R_TO_Y);
    let g_to_y = _mm256_set1_ps(YCBCR_G_TO_Y);
    let b_to_y = _mm256_set1_ps(YCBCR_B_TO_Y);
    let r_to_cb = _mm256_set1_ps(YCBCR_R_TO_CB);
    let g_to_cb = _mm256_set1_ps(YCBCR_G_TO_CB);
    let b_to_cb = _mm256_set1_ps(YCBCR_B_TO_CB);
    let r_to_cr = _mm256_set1_ps(YCBCR_R_TO_CR);
    let g_to_cr = _mm256_set1_ps(YCBCR_G_TO_CR);
    let b_to_cr = _mm256_set1_ps(YCBCR_B_TO_CR);
    let offset_128 = _mm256_set1_ps(128.0);

    // Y = r * r_to_y + g * g_to_y + b * b_to_y
    // Using FMA: result = a * b + c
    let y = _mm256_fmadd_ps(
        b_vec,
        b_to_y,
        _mm256_fmadd_ps(g_vec, g_to_y, _mm256_mul_ps(r_vec, r_to_y)),
    );

    // Cb = 128 + r * r_to_cb + g * g_to_cb + b * b_to_cb
    let cb = _mm256_fmadd_ps(
        b_vec,
        b_to_cb,
        _mm256_fmadd_ps(g_vec, g_to_cb, _mm256_fmadd_ps(r_vec, r_to_cb, offset_128)),
    );

    // Cr = 128 + r * r_to_cr + g * g_to_cr + b * b_to_cr
    let cr = _mm256_fmadd_ps(
        b_vec,
        b_to_cr,
        _mm256_fmadd_ps(g_vec, g_to_cr, _mm256_fmadd_ps(r_vec, r_to_cr, offset_128)),
    );

    // Store results
    safe_simd::_mm256_storeu_ps(y_out, y);
    safe_simd::_mm256_storeu_ps(cb_out, cb);
    safe_simd::_mm256_storeu_ps(cr_out, cr);
}

/// Box filter downsample 2x2: average 4 adjacent pixels.
///
/// Takes evens and odds from two rows and computes (sum * 0.25).
/// For chroma downsampling in 4:2:0 encoding.
#[arcane]
#[inline]
pub fn mage_box_filter_2x2(
    _token: X64V3Token,
    row0_evens: __m256,
    row0_odds: __m256,
    row1_evens: __m256,
    row1_odds: __m256,
) -> __m256 {
    let scale = _mm256_set1_ps(0.25);
    let sum = _mm256_add_ps(
        _mm256_add_ps(row0_evens, row0_odds),
        _mm256_add_ps(row1_evens, row1_odds),
    );
    _mm256_mul_ps(sum, scale)
}

// ============================================================================
// Even/Odd Deinterleave (for chroma downsampling)
// ============================================================================

/// AVX2-optimized deinterleave using Highway's ConcatEven/ConcatOdd pattern.
///
/// Given 16 consecutive f32s: [e0,o0,e1,o1,e2,o2,e3,o3, e4,o4,e5,o5,e6,o6,e7,o7]
/// Returns evens = [e0,e1,e2,e3,e4,e5,e6,e7], odds = [o0,o1,o2,o3,o4,o5,o6,o7]
///
/// This is ~4x faster than element-by-element construction.
#[arcane]
#[inline]
pub fn mage_gather_even_odd_x8(_token: X64V3Token, data: &[f32; 16]) -> (__m256, __m256) {
    // Load 16 consecutive floats as two YMM registers
    let lo = safe_simd::_mm256_loadu_ps(data[0..8].try_into().unwrap());
    let hi = safe_simd::_mm256_loadu_ps(data[8..16].try_into().unwrap());

    // Highway's ConcatEven pattern for f32:
    // _mm256_shuffle_ps with 0x88 selects elements [0,2] from each source per lane
    let v2020 = _mm256_shuffle_ps(lo, hi, 0x88);
    // _mm256_permute4x64_epi64 with 0xD8 reorders 64-bit chunks: [0,2,1,3]
    let evens = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(v2020), 0xD8));

    // Highway's ConcatOdd pattern for f32:
    // _mm256_shuffle_ps with 0xDD selects elements [1,3] from each source per lane
    let v3131 = _mm256_shuffle_ps(lo, hi, 0xDD);
    let odds = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(v3131), 0xD8));

    (evens, odds)
}

// ============================================================================
// Adaptive Quantization SIMD
// ============================================================================

// AQ Constants (from quant/aq/simd.rs)
const K_INPUT_SCALING: f32 = 1.0 / 255.0;
const K_EPSILON_RATIO: f32 = 1e-2;
const K_NUM_OFFSET_RATIO: f32 = K_EPSILON_RATIO / K_INPUT_SCALING / K_INPUT_SCALING;
const K_SG_MUL: f32 = 226.0480446705883;
const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_INV_LOG2E: f32 = 0.6931471805599453;
const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_INV_LOG2E;
const K_NUM_MUL_RATIO: f32 = K_SG_RET_MUL * 3.0 * K_SG_MUL;
const K_SG_VOFFSET: f32 = 7.14672470003;
const K_VOFFSET_RATIO: f32 = (K_SG_VOFFSET * K_INV_LOG2E + K_EPSILON_RATIO) / K_INPUT_SCALING;
const K_DEN_MUL_RATIO: f32 = K_INV_LOG2E * K_SG_MUL * K_INPUT_SCALING * K_INPUT_SCALING;
const K_MASKING_LOG_OFFSET: f32 = 28.0;
const K_MASKING_MUL: f32 = 211.50759899638012;
const K_BIAS_AQ: f32 = 0.16 / K_INPUT_SCALING; // 40.8
const LIMIT_AQ: f32 = 0.2;
const MATCH_GAMMA_OFFSET: f32 = 0.019;
const GAMMA_OFFSET_AQ: f32 = MATCH_GAMMA_OFFSET / K_INPUT_SCALING; // ~4.845

/// SIMD ratio_of_derivatives (non-inverted) using AVX2+FMA.
///
/// Computes: den / num where:
///   v = max(val, 0)
///   v2 = v * v
///   num = v2 * K_NUM_MUL_RATIO + K_NUM_OFFSET_RATIO
///   den = (v * K_DEN_MUL_RATIO) * v2 + K_VOFFSET_RATIO
///
/// Returns 8 results in a __m256.
#[arcane]
#[inline]
pub fn mage_ratio_of_derivatives_x8(token: X64V3Token, vals: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let num_mul = _mm256_set1_ps(K_NUM_MUL_RATIO);
    let num_offset = _mm256_set1_ps(K_NUM_OFFSET_RATIO);
    let den_mul = _mm256_set1_ps(K_DEN_MUL_RATIO);
    let voffset = _mm256_set1_ps(K_VOFFSET_RATIO);

    // v = max(vals, 0)
    let v = _mm256_max_ps(vals, zero);
    // v2 = v * v
    let v2 = _mm256_mul_ps(v, v);
    // num = v2 * num_mul + num_offset (FMA)
    let num = _mm256_fmadd_ps(v2, num_mul, num_offset);
    // den = (v * den_mul) * v2 + voffset (FMA)
    let v_den = _mm256_mul_ps(v, den_mul);
    let den = _mm256_fmadd_ps(v_den, v2, voffset);
    // Result: den / num
    let _ = token;
    _mm256_div_ps(den, num)
}

/// SIMD ratio_of_derivatives (inverted) using AVX2+FMA.
///
/// Same as above but returns num / den.
#[arcane]
#[inline]
pub fn mage_ratio_of_derivatives_inv_x8(token: X64V3Token, vals: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let num_mul = _mm256_set1_ps(K_NUM_MUL_RATIO);
    let num_offset = _mm256_set1_ps(K_NUM_OFFSET_RATIO);
    let den_mul = _mm256_set1_ps(K_DEN_MUL_RATIO);
    let voffset = _mm256_set1_ps(K_VOFFSET_RATIO);

    let v = _mm256_max_ps(vals, zero);
    let v2 = _mm256_mul_ps(v, v);
    let num = _mm256_fmadd_ps(v2, num_mul, num_offset);
    let v_den = _mm256_mul_ps(v, den_mul);
    let den = _mm256_fmadd_ps(v_den, v2, voffset);
    let _ = token;
    _mm256_div_ps(num, den)
}

/// SIMD masking_sqrt using AVX2+FMA.
///
/// Computes: 0.25 * sqrt(v * sqrt(K_MASKING_MUL * 1e8) + K_MASKING_LOG_OFFSET)
#[arcane]
#[inline]
pub fn mage_masking_sqrt_x8(token: X64V3Token, v: __m256) -> __m256 {
    let k_mul_sqrt = _mm256_set1_ps((K_MASKING_MUL * 1e8_f32).sqrt());
    let k_offset = _mm256_set1_ps(K_MASKING_LOG_OFFSET);
    let quarter = _mm256_set1_ps(0.25);

    // inner = v * k_mul_sqrt + k_offset (FMA)
    let inner = _mm256_fmadd_ps(v, k_mul_sqrt, k_offset);
    // result = 0.25 * sqrt(inner)
    let _ = token;
    _mm256_mul_ps(quarter, _mm256_sqrt_ps(inner))
}

/// SIMD pre_erosion_pixel computation using AVX2+FMA.
///
/// For each of 8 pixels: compute neighbor average, ratio_of_derivatives, diff, masking_sqrt.
///
/// Returns 8 masked diff values.
#[arcane]
#[inline]
pub fn mage_pre_erosion_pixel_x8(
    token: X64V3Token,
    pixels: __m256,
    left: __m256,
    right: __m256,
    top: __m256,
    bottom: __m256,
) -> __m256 {
    let quarter = _mm256_set1_ps(0.25);
    let gamma_offset = _mm256_set1_ps(GAMMA_OFFSET_AQ);
    let limit = _mm256_set1_ps(LIMIT_AQ);

    // base = 0.25 * (left + right + top + bottom)
    let sum_lr = _mm256_add_ps(left, right);
    let sum_tb = _mm256_add_ps(top, bottom);
    let sum_all = _mm256_add_ps(sum_lr, sum_tb);
    let base = _mm256_mul_ps(quarter, sum_all);

    // ratio = ratio_of_derivatives(pixels + gamma_offset, false)
    let pixels_offset = _mm256_add_ps(pixels, gamma_offset);
    let ratio = mage_ratio_of_derivatives_x8(token, pixels_offset);

    // diff = ratio * (pixels - base)
    let pixels_minus_base = _mm256_sub_ps(pixels, base);
    let diff = _mm256_mul_ps(ratio, pixels_minus_base);

    // diff_sq = min(diff * diff, LIMIT)
    let diff_sq = _mm256_min_ps(_mm256_mul_ps(diff, diff), limit);

    // Return masking_sqrt(diff_sq)
    mage_masking_sqrt_x8(token, diff_sq)
}

/// Horizontal sum of 8 f32s in a __m256 register.
///
/// Uses efficient reduction: hadd + extract + add.
#[rite]
fn mage_hsum_ps(_token: X64V3Token, v: __m256) -> f32 {
    // Sum pairs horizontally
    let sum1 = _mm256_hadd_ps(v, v); // [a+b, c+d, a+b, c+d, e+f, g+h, e+f, g+h]
    let sum2 = _mm256_hadd_ps(sum1, sum1); // [a+b+c+d, ..., e+f+g+h, ...]
    // Extract low and high 128-bit lanes
    let low = _mm256_castps256_ps128(sum2);
    let high = _mm256_extractf128_ps(sum2, 1);
    // Add them together
    let result = _mm_add_ss(low, high);
    _mm_cvtss_f32(result)
}

/// Compute HF modulation sum: |p - right| + |p - below| for 8x8 block.
///
/// Uses AVX2 intrinsics for better performance than wide crate.
///
/// # Arguments
/// * `token` - SIMD capability token
/// * `block` - Pointer to top-left of block data
/// * `stride` - Row stride in floats
/// * `block_y` - Y position of block
/// * `img_height` - Total image height
///
/// # Returns
/// Sum of horizontal and vertical differences
#[arcane]
#[inline]
pub fn mage_hf_modulation_sum_8x8(
    token: X64V3Token,
    block: &[f32],
    stride: usize,
    block_y: usize,
    img_height: usize,
) -> f32 {
    // Mask to zero out the 8th element for horizontal differences
    let mask_first_7 = _mm256_set_ps(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let sign_mask = _mm256_set1_ps(-0.0); // For abs via andnot

    let mut h_sum = _mm256_setzero_ps();
    let mut v_sum = _mm256_setzero_ps();

    for dy in 0..8 {
        let y = block_y + dy;
        if y >= img_height {
            continue;
        }

        let row_start = dy * stride;

        // Horizontal differences: |p - p_right| for positions 0..6
        if row_start + 9 <= block.len() {
            let p = safe_simd::_mm256_loadu_ps(block[row_start..row_start + 8].try_into().unwrap());
            let p_right =
                safe_simd::_mm256_loadu_ps(block[row_start + 1..row_start + 9].try_into().unwrap());
            // abs(p - p_right) using andnot with sign mask
            let diff = _mm256_sub_ps(p, p_right);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            // Mask out 8th element
            h_sum = _mm256_add_ps(h_sum, _mm256_mul_ps(abs_diff, mask_first_7));
        }

        // Vertical differences: |p - p_below| for first 7 rows
        if dy < 7 && y + 1 < img_height {
            let next_row_start = (dy + 1) * stride;
            if row_start + 8 <= block.len() && next_row_start + 8 <= block.len() {
                let p =
                    safe_simd::_mm256_loadu_ps(block[row_start..row_start + 8].try_into().unwrap());
                let p_below = safe_simd::_mm256_loadu_ps(
                    block[next_row_start..next_row_start + 8]
                        .try_into()
                        .unwrap(),
                );
                let diff = _mm256_sub_ps(p, p_below);
                let abs_diff = _mm256_andnot_ps(sign_mask, diff);
                v_sum = _mm256_add_ps(v_sum, abs_diff);
            }
        }
    }

    // Single horizontal reduction at the end
    mage_hsum_ps(token, h_sum) + mage_hsum_ps(token, v_sum)
}

/// Compute gamma modulation sum (ratio_of_derivatives inverted) for 8x8 block.
///
/// Uses AVX2+FMA intrinsics.
#[arcane]
#[inline]
pub fn mage_gamma_modulation_sum_8x8(
    token: X64V3Token,
    block: &[f32],
    stride: usize,
    block_y: usize,
    img_height: usize,
) -> f32 {
    let bias = _mm256_set1_ps(K_BIAS_AQ);
    let mut sum = _mm256_setzero_ps();

    for dy in 0..8 {
        let y = block_y + dy;
        if y >= img_height {
            continue;
        }

        let row_start = dy * stride;
        if row_start + 8 <= block.len() {
            let row =
                safe_simd::_mm256_loadu_ps(block[row_start..row_start + 8].try_into().unwrap());
            let row_biased = _mm256_add_ps(row, bias);
            let ratio = mage_ratio_of_derivatives_inv_x8(token, row_biased);
            sum = _mm256_add_ps(sum, ratio);
        }
    }

    mage_hsum_ps(token, sum)
}

/// Fast exp2 approximation using AVX2+FMA.
///
/// Uses polynomial approximation + bit manipulation for 2^x.
/// Accurate to ~1e-4 relative error for inputs in [-126, 127].
#[arcane]
#[inline]
pub fn mage_fast_exp2_x8(token: X64V3Token, x: __m256) -> __m256 {
    // Clamp to prevent overflow/underflow
    let min_val = _mm256_set1_ps(-126.0);
    let max_val = _mm256_set1_ps(127.0);
    let x_clamped = _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);

    // Split into integer and fractional parts
    let xi = _mm256_floor_ps(x_clamped);
    let xf = _mm256_sub_ps(x_clamped, xi);

    // Minimax polynomial approximation for 2^xf where xf in [0, 1)
    // p = 1 + xf * (c1 + xf * (c2 + xf * (c3 + xf * c4)))
    let c4 = _mm256_set1_ps(0.009618129107628477);
    let c3 = _mm256_set1_ps(0.055504108664821579);
    let c2 = _mm256_set1_ps(0.24022650695910071);
    let c1 = _mm256_set1_ps(0.6931471805599453);
    let one = _mm256_set1_ps(1.0);

    let p = _mm256_fmadd_ps(
        xf,
        _mm256_fmadd_ps(xf, _mm256_fmadd_ps(xf, _mm256_fmadd_ps(xf, c4, c3), c2), c1),
        one,
    );

    // Compute 2^xi using IEEE 754 bit manipulation
    // bits = (xi + 127) << 23
    let xi_i32 = _mm256_cvtps_epi32(xi);
    let bias = _mm256_set1_epi32(127);
    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(xi_i32, bias), 23);
    let exp_f = _mm256_castsi256_ps(exp_bits);

    // Result: 2^xi * p(xf)
    let _ = token;
    _mm256_mul_ps(exp_f, p)
}

/// Fast log2 approximation using AVX2+FMA.
///
/// Uses bit manipulation + polynomial for log2(x).
/// Accurate to ~0.01 absolute error for positive inputs.
#[arcane]
#[inline]
pub fn mage_fast_log2_x8(token: X64V3Token, x: __m256) -> __m256 {
    // Extract exponent using IEEE 754 bit manipulation
    let bits = _mm256_castps_si256(x);
    let e_bits = _mm256_srli_epi32(bits, 23);
    let bias = _mm256_set1_epi32(127);
    let e = _mm256_cvtepi32_ps(_mm256_sub_epi32(e_bits, bias));

    // Extract mantissa as float in [1, 2)
    let mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
    let one_bits = _mm256_set1_epi32(0x3F800000);
    let f_bits = _mm256_or_si256(_mm256_and_si256(bits, mantissa_mask), one_bits);
    let f = _mm256_castsi256_ps(f_bits);

    // Compute log2(f) for f in [1, 2) using polynomial
    let one = _mm256_set1_ps(1.0);
    let t = _mm256_sub_ps(f, one); // f - 1

    // Horner's method: log2(f) ≈ t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    let c5 = _mm256_set1_ps(0.2885390082);
    let c4 = _mm256_set1_ps(-0.3606737602);
    let c3 = _mm256_set1_ps(0.4808983470);
    let c2 = _mm256_set1_ps(-0.7213475204);
    let c1 = _mm256_set1_ps(1.442695041);

    let log2_f = _mm256_mul_ps(
        t,
        _mm256_fmadd_ps(
            t,
            _mm256_fmadd_ps(t, _mm256_fmadd_ps(t, _mm256_fmadd_ps(t, c5, c4), c3), c2),
            c1,
        ),
    );

    // Result: e + log2(f)
    let _ = token;
    _mm256_add_ps(e, log2_f)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    /// Load 8 f32s into an __m256 register (archmage provides target feature).
    #[arcane]
    fn load_f32x8(_token: Desktop64, data: &[f32; 8]) -> __m256 {
        safe_simd::_mm256_loadu_ps(data)
    }

    /// Store an __m256 register into 8 f32s (archmage provides target feature).
    #[arcane]
    fn store_f32x8(_token: Desktop64, dst: &mut [f32; 8], val: __m256) {
        safe_simd::_mm256_storeu_ps(dst, val);
    }

    #[test]
    fn test_mage_forward_dct_8x8_identity() {
        if let Some(token) = Desktop64::summon() {
            // Identity-like input (all zeros except DC)
            let mut input = [0.0f32; 64];
            input[0] = 64.0; // DC coefficient

            let mut output = [0.0f32; 64];
            mage_forward_dct_8x8(token, &input, &mut output);

            // DC should be non-zero, others should be small
            assert!(output[0].abs() > 0.1, "DC should be non-zero");
        }
    }

    #[test]
    fn test_mage_forward_dct_8x8_flat_block() {
        if let Some(token) = Desktop64::summon() {
            // Flat block (constant value)
            let input = [128.0f32; 64];
            let mut output = [0.0f32; 64];

            mage_forward_dct_8x8(token, &input, &mut output);

            // For a flat block, only DC should be non-zero
            // DC = value (with 1/64 total scaling from 2D DCT)
            assert!(
                output[0].abs() > 100.0 && output[0].abs() < 150.0,
                "DC should be ~128 for flat block, got {}",
                output[0]
            );

            // AC coefficients should be near zero
            for i in 1..64 {
                assert!(
                    output[i].abs() < 0.001,
                    "AC[{}] = {} should be ~0 for flat block",
                    i,
                    output[i]
                );
            }
        }
    }

    #[test]
    fn test_mage_transpose_8x8_inplace() {
        if let Some(token) = Desktop64::summon() {
            let original: [f32; 64] = core::array::from_fn(|i| i as f32);

            let mut reg = [
                load_f32x8(token, original[0..8].try_into().unwrap()),
                load_f32x8(token, original[8..16].try_into().unwrap()),
                load_f32x8(token, original[16..24].try_into().unwrap()),
                load_f32x8(token, original[24..32].try_into().unwrap()),
                load_f32x8(token, original[32..40].try_into().unwrap()),
                load_f32x8(token, original[40..48].try_into().unwrap()),
                load_f32x8(token, original[48..56].try_into().unwrap()),
                load_f32x8(token, original[56..64].try_into().unwrap()),
            ];

            mage_transpose_8x8_inplace(token, &mut reg);

            let mut result = [0.0f32; 64];
            store_f32x8(token, (&mut result[0..8]).try_into().unwrap(), reg[0]);
            store_f32x8(token, (&mut result[8..16]).try_into().unwrap(), reg[1]);
            store_f32x8(token, (&mut result[16..24]).try_into().unwrap(), reg[2]);
            store_f32x8(token, (&mut result[24..32]).try_into().unwrap(), reg[3]);
            store_f32x8(token, (&mut result[32..40]).try_into().unwrap(), reg[4]);
            store_f32x8(token, (&mut result[40..48]).try_into().unwrap(), reg[5]);
            store_f32x8(token, (&mut result[48..56]).try_into().unwrap(), reg[6]);
            store_f32x8(token, (&mut result[56..64]).try_into().unwrap(), reg[7]);

            for row in 0..8 {
                for col in 0..8 {
                    let orig_val = original[row * 8 + col];
                    let trans_val = result[col * 8 + row];
                    assert_eq!(
                        orig_val, trans_val,
                        "Mismatch at ({}, {}): expected {}, got {}",
                        row, col, orig_val, trans_val
                    );
                }
            }
        }
    }

    #[test]
    fn test_mage_gather_even_odd_x8() {
        if let Some(token) = Desktop64::summon() {
            // Test data: [0, 1, 2, 3, ..., 15] interleaved as [e0,o0,e1,o1,...]
            let data: [f32; 16] = core::array::from_fn(|i| i as f32);

            let (evens, odds) = mage_gather_even_odd_x8(token, &data);

            let mut evens_arr = [0.0f32; 8];
            let mut odds_arr = [0.0f32; 8];
            store_f32x8(token, &mut evens_arr, evens);
            store_f32x8(token, &mut odds_arr, odds);

            // Expected: evens = [0, 2, 4, 6, 8, 10, 12, 14]
            //           odds  = [1, 3, 5, 7, 9, 11, 13, 15]
            let expected_evens = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
            let expected_odds = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];

            for i in 0..8 {
                assert_eq!(
                    evens_arr[i], expected_evens[i],
                    "evens[{}]: got {}, expected {}",
                    i, evens_arr[i], expected_evens[i]
                );
                assert_eq!(
                    odds_arr[i], expected_odds[i],
                    "odds[{}]: got {}, expected {}",
                    i, odds_arr[i], expected_odds[i]
                );
            }
        }
    }

    #[test]
    fn test_mage_rgb_to_ycbcr_8px() {
        if let Some(token) = Desktop64::summon() {
            // Test with pure white (255, 255, 255) -> Y=255, Cb=128, Cr=128
            let r = [255.0f32; 8];
            let g = [255.0f32; 8];
            let b = [255.0f32; 8];
            let mut y = [0.0f32; 8];
            let mut cb = [0.0f32; 8];
            let mut cr = [0.0f32; 8];

            mage_rgb_to_ycbcr_8px(token, &r, &g, &b, &mut y, &mut cb, &mut cr);

            for i in 0..8 {
                // Y should be ~255 (0.299*255 + 0.587*255 + 0.114*255 = 255)
                assert!(
                    (y[i] - 255.0).abs() < 0.1,
                    "Y[{}] = {} should be ~255",
                    i,
                    y[i]
                );
                // Cb should be ~128 (neutral)
                assert!(
                    (cb[i] - 128.0).abs() < 0.1,
                    "Cb[{}] = {} should be ~128",
                    i,
                    cb[i]
                );
                // Cr should be ~128 (neutral)
                assert!(
                    (cr[i] - 128.0).abs() < 0.1,
                    "Cr[{}] = {} should be ~128",
                    i,
                    cr[i]
                );
            }

            // Test with pure black (0, 0, 0) -> Y=0, Cb=128, Cr=128
            let r = [0.0f32; 8];
            let g = [0.0f32; 8];
            let b = [0.0f32; 8];

            mage_rgb_to_ycbcr_8px(token, &r, &g, &b, &mut y, &mut cb, &mut cr);

            for i in 0..8 {
                assert!(
                    y[i].abs() < 0.1,
                    "Y[{}] = {} should be ~0 for black",
                    i,
                    y[i]
                );
                assert!(
                    (cb[i] - 128.0).abs() < 0.1,
                    "Cb[{}] = {} should be ~128 for black",
                    i,
                    cb[i]
                );
                assert!(
                    (cr[i] - 128.0).abs() < 0.1,
                    "Cr[{}] = {} should be ~128 for black",
                    i,
                    cr[i]
                );
            }

            // Test with pure red (255, 0, 0) -> Y=76.2, Cb=84.5, Cr=255
            let r = [255.0f32; 8];
            let g = [0.0f32; 8];
            let b = [0.0f32; 8];

            mage_rgb_to_ycbcr_8px(token, &r, &g, &b, &mut y, &mut cb, &mut cr);

            for i in 0..8 {
                // Y = 0.299 * 255 = 76.245
                assert!(
                    (y[i] - 76.245).abs() < 0.1,
                    "Y[{}] = {} should be ~76.245 for red",
                    i,
                    y[i]
                );
            }
        }
    }

    #[test]
    fn test_mage_box_filter_2x2() {
        if let Some(token) = Desktop64::summon() {
            // Each "pixel" should be averaged from 4 neighbors
            let row0_evens_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let row0_odds_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let row1_evens_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let row1_odds_arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            let row0_evens = load_f32x8(token, &row0_evens_arr);
            let row0_odds = load_f32x8(token, &row0_odds_arr);
            let row1_evens = load_f32x8(token, &row1_evens_arr);
            let row1_odds = load_f32x8(token, &row1_odds_arr);

            let result = mage_box_filter_2x2(token, row0_evens, row0_odds, row1_evens, row1_odds);

            let mut result_arr = [0.0f32; 8];
            store_f32x8(token, &mut result_arr, result);

            // Each output should be (4 * input) * 0.25 = input
            for i in 0..8 {
                let expected = (i + 1) as f32;
                assert!(
                    (result_arr[i] - expected).abs() < 0.001,
                    "result[{}] = {} should be {}",
                    i,
                    result_arr[i],
                    expected
                );
            }
        }
    }

    // ============================================================================
    // AQ Function Tests
    // ============================================================================

    /// Scalar reference for ratio_of_derivatives (for testing)
    fn ratio_of_derivatives_scalar(val: f32, invert: bool) -> f32 {
        let v = val.max(0.0);
        let v2 = v * v;
        let num = v2.mul_add(K_NUM_MUL_RATIO, K_NUM_OFFSET_RATIO);
        let den = (v * K_DEN_MUL_RATIO).mul_add(v2, K_VOFFSET_RATIO);
        if invert { num / den } else { den / num }
    }

    /// Scalar reference for masking_sqrt
    fn masking_sqrt_scalar(v: f32) -> f32 {
        0.25 * v
            .mul_add((K_MASKING_MUL * 1e8_f32).sqrt(), K_MASKING_LOG_OFFSET)
            .sqrt()
    }

    #[test]
    fn test_mage_ratio_of_derivatives_x8() {
        if let Some(token) = Desktop64::summon() {
            let inputs = [128.0f32, 64.0, 192.0, 255.0, 0.0, 32.0, 100.0, 200.0];
            let input_vec = load_f32x8(token, &inputs);

            let result = mage_ratio_of_derivatives_x8(token, input_vec);

            let mut result_arr = [0.0f32; 8];
            store_f32x8(token, &mut result_arr, result);

            for i in 0..8 {
                let expected = ratio_of_derivatives_scalar(inputs[i], false);
                let rel_err = (result_arr[i] - expected).abs() / expected.abs().max(1e-10);
                assert!(
                    rel_err < 1e-5,
                    "ratio_of_derivatives mismatch at {}: got {}, expected {}, rel_err {}",
                    i,
                    result_arr[i],
                    expected,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_mage_ratio_of_derivatives_inv_x8() {
        if let Some(token) = Desktop64::summon() {
            let inputs = [128.0f32, 64.0, 192.0, 255.0, 0.0, 32.0, 100.0, 200.0];
            let input_vec = load_f32x8(token, &inputs);

            let result = mage_ratio_of_derivatives_inv_x8(token, input_vec);

            let mut result_arr = [0.0f32; 8];
            store_f32x8(token, &mut result_arr, result);

            for i in 0..8 {
                let expected = ratio_of_derivatives_scalar(inputs[i], true);
                let rel_err = (result_arr[i] - expected).abs() / expected.abs().max(1e-10);
                assert!(
                    rel_err < 1e-5,
                    "ratio_of_derivatives_inv mismatch at {}: got {}, expected {}, rel_err {}",
                    i,
                    result_arr[i],
                    expected,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_mage_masking_sqrt_x8() {
        if let Some(token) = Desktop64::summon() {
            let inputs = [0.0f32, 0.01, 0.05, 0.1, 0.15, 0.2, 0.05, 0.08];
            let input_vec = load_f32x8(token, &inputs);

            let result = mage_masking_sqrt_x8(token, input_vec);

            let mut result_arr = [0.0f32; 8];
            store_f32x8(token, &mut result_arr, result);

            for i in 0..8 {
                let expected = masking_sqrt_scalar(inputs[i]);
                let rel_err = (result_arr[i] - expected).abs() / expected.abs().max(1e-10);
                assert!(
                    rel_err < 1e-5,
                    "masking_sqrt mismatch at {}: got {}, expected {}, rel_err {}",
                    i,
                    result_arr[i],
                    expected,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_mage_fast_exp2_x8() {
        if let Some(token) = Desktop64::summon() {
            let inputs = [-5.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0];
            let input_vec = load_f32x8(token, &inputs);

            let result = mage_fast_exp2_x8(token, input_vec);

            let mut result_arr = [0.0f32; 8];
            store_f32x8(token, &mut result_arr, result);

            for i in 0..8 {
                let expected = inputs[i].exp2();
                let rel_err = (result_arr[i] - expected).abs() / expected.abs().max(1e-10);
                assert!(
                    rel_err < 5e-4,
                    "fast_exp2 mismatch at {} (input={}): got {}, expected {}, rel_err {}",
                    i,
                    inputs[i],
                    result_arr[i],
                    expected,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_mage_fast_log2_x8() {
        if let Some(token) = Desktop64::summon() {
            let inputs = [0.01f32, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0];
            let input_vec = load_f32x8(token, &inputs);

            let result = mage_fast_log2_x8(token, input_vec);

            let mut result_arr = [0.0f32; 8];
            store_f32x8(token, &mut result_arr, result);

            for i in 0..8 {
                let expected = inputs[i].log2();
                let abs_err = (result_arr[i] - expected).abs();
                assert!(
                    abs_err < 0.1,
                    "fast_log2 mismatch at {} (input={}): got {}, expected {}, abs_err {}",
                    i,
                    inputs[i],
                    result_arr[i],
                    expected,
                    abs_err
                );
            }
        }
    }

    /// Safe wrapper for calling #[rite] mage_hsum_ps from test code.
    #[arcane]
    fn call_hsum_ps(_token: Desktop64, v: __m256) -> f32 {
        mage_hsum_ps(_token, v)
    }

    #[test]
    fn test_mage_hsum_ps() {
        if let Some(token) = Desktop64::summon() {
            let inputs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let input_vec = load_f32x8(token, &inputs);

            let result = call_hsum_ps(token, input_vec);
            let expected: f32 = inputs.iter().sum();

            assert!(
                (result - expected).abs() < 1e-5,
                "hsum mismatch: got {}, expected {}",
                result,
                expected
            );
        }
    }

    // ========================================================================
    // AVX-512 DCT Tests
    // ========================================================================

    #[test]
    fn test_mage_forward_dct_8x8_dual_flat_blocks() {
        use archmage::X64V4Token;

        if let Some(token) = X64V4Token::summon() {
            // Two flat blocks with different constant values
            let input_a = [128.0f32; 64];
            let input_b = [64.0f32; 64];
            let mut output_a = [0.0f32; 64];
            let mut output_b = [0.0f32; 64];

            mage_forward_dct_8x8_dual(token, &input_a, &input_b, &mut output_a, &mut output_b);

            // For flat blocks, only DC should be non-zero
            // DC = sum / 64 = value * 64 / 64 = value
            // DC_a = 128, DC_b = 64
            assert!(
                output_a[0].abs() > 100.0 && output_a[0].abs() < 150.0,
                "DC_a should be ~128 for flat block, got {}",
                output_a[0]
            );
            assert!(
                output_b[0].abs() > 50.0 && output_b[0].abs() < 80.0,
                "DC_b should be ~64 for flat block, got {}",
                output_b[0]
            );

            // AC coefficients should be near zero
            for i in 1..64 {
                assert!(
                    output_a[i].abs() < 0.001,
                    "AC_a[{}] = {} should be ~0 for flat block",
                    i,
                    output_a[i]
                );
                assert!(
                    output_b[i].abs() < 0.001,
                    "AC_b[{}] = {} should be ~0 for flat block",
                    i,
                    output_b[i]
                );
            }
        } else {
            println!("AVX-512 not available, skipping test");
        }
    }

    #[test]
    fn test_mage_forward_dct_8x8_dual_matches_single() {
        use archmage::X64V4Token;

        if let Some(token) = X64V4Token::summon() {
            // Create test patterns
            let input_a: [f32; 64] = core::array::from_fn(|i| (i % 256) as f32);
            let input_b: [f32; 64] = core::array::from_fn(|i| ((i * 3 + 17) % 256) as f32);

            // Process with dual-block AVX-512
            let mut output_a_dual = [0.0f32; 64];
            let mut output_b_dual = [0.0f32; 64];
            mage_forward_dct_8x8_dual(
                token,
                &input_a,
                &input_b,
                &mut output_a_dual,
                &mut output_b_dual,
            );

            // Process individually with AVX2
            let mut output_a_single = [0.0f32; 64];
            let mut output_b_single = [0.0f32; 64];
            mage_forward_dct_8x8(token.v3(), &input_a, &mut output_a_single);
            mage_forward_dct_8x8(token.v3(), &input_b, &mut output_b_single);

            // Compare results - should be identical (within floating point tolerance)
            for i in 0..64 {
                let diff_a = (output_a_dual[i] - output_a_single[i]).abs();
                let diff_b = (output_b_dual[i] - output_b_single[i]).abs();
                let max_a = output_a_single[i].abs().max(1e-10);
                let max_b = output_b_single[i].abs().max(1e-10);

                assert!(
                    diff_a / max_a < 1e-5 || diff_a < 1e-6,
                    "Block A mismatch at {}: dual={}, single={}, diff={}",
                    i,
                    output_a_dual[i],
                    output_a_single[i],
                    diff_a
                );
                assert!(
                    diff_b / max_b < 1e-5 || diff_b < 1e-6,
                    "Block B mismatch at {}: dual={}, single={}, diff={}",
                    i,
                    output_b_dual[i],
                    output_b_single[i],
                    diff_b
                );
            }
        } else {
            println!("AVX-512 not available, skipping test");
        }
    }

    /// Load two 8x8 f32 blocks into interleaved ZMM registers.
    /// Safe via archmage token: all AVX-512 intrinsics are gated by capability proof.
    #[arcane]
    fn load_dual_blocks_avx512(
        _token: X64V4Token,
        block_a: &[f32; 64],
        block_b: &[f32; 64],
    ) -> [__m512; 8] {
        core::array::from_fn(|i| {
            let off = i * 8;
            _mm512_insertf32x8::<1>(
                _mm512_castps256_ps512(safe_simd::_mm256_loadu_ps(
                    block_a[off..off + 8].try_into().unwrap(),
                )),
                safe_simd::_mm256_loadu_ps(block_b[off..off + 8].try_into().unwrap()),
            )
        })
    }

    /// Store interleaved ZMM registers back to two 8x8 f32 blocks.
    /// Safe via archmage token: all AVX-512 intrinsics are gated by capability proof.
    #[arcane]
    fn store_dual_blocks_avx512(
        _token: X64V4Token,
        reg: &[__m512; 8],
        result_a: &mut [f32; 64],
        result_b: &mut [f32; 64],
    ) {
        for i in 0..8 {
            safe_simd::_mm256_storeu_ps(
                (&mut result_a[i * 8..(i + 1) * 8]).try_into().unwrap(),
                _mm512_castps512_ps256(reg[i]),
            );
            safe_simd::_mm256_storeu_ps(
                (&mut result_b[i * 8..(i + 1) * 8]).try_into().unwrap(),
                _mm512_extractf32x8_ps::<1>(reg[i]),
            );
        }
    }

    /// Safe wrapper for calling #[rite] mage_transpose_8x8_dual_inner from test code.
    #[arcane]
    fn call_transpose_dual(_token: archmage::X64V4Token, r: &mut [__m512; 8]) {
        mage_transpose_8x8_dual_inner(_token, r);
    }

    #[test]
    fn test_mage_transpose_8x8_dual() {
        use archmage::X64V4Token;

        if let Some(token) = X64V4Token::summon() {
            // Create test data: two 8x8 blocks
            let original_a: [f32; 64] = core::array::from_fn(|i| i as f32);
            let original_b: [f32; 64] = core::array::from_fn(|i| (i + 100) as f32);

            // Load into ZMM registers (interleaved) — safe via #[arcane]
            let mut reg = load_dual_blocks_avx512(token, &original_a, &original_b);

            // Transpose
            call_transpose_dual(token, &mut reg);

            // Store back — safe via #[arcane]
            let mut result_a = [0.0f32; 64];
            let mut result_b = [0.0f32; 64];
            store_dual_blocks_avx512(token, &reg, &mut result_a, &mut result_b);

            // Verify transpose for block A: result[col * 8 + row] == original[row * 8 + col]
            for row in 0..8 {
                for col in 0..8 {
                    let orig_val_a = original_a[row * 8 + col];
                    let trans_val_a = result_a[col * 8 + row];
                    assert_eq!(
                        orig_val_a, trans_val_a,
                        "Block A mismatch at ({}, {}): expected {}, got {}",
                        row, col, orig_val_a, trans_val_a
                    );

                    let orig_val_b = original_b[row * 8 + col];
                    let trans_val_b = result_b[col * 8 + row];
                    assert_eq!(
                        orig_val_b, trans_val_b,
                        "Block B mismatch at ({}, {}): expected {}, got {}",
                        row, col, orig_val_b, trans_val_b
                    );
                }
            }
        } else {
            println!("AVX-512 not available, skipping test");
        }
    }
}
