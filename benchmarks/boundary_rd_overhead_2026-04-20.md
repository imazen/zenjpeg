# boundary_rd SIMD kernel overhead — 2026-04-20

Measured encode-time overhead of
`EncoderConfig::boundary_rd(BoundaryRd::On(BoundaryRdConfig::default()))`
vs `BoundaryRd::Off` before and after the SIMD optimization landed on
top of PR #102.

## Measurement setup

- CPU: AMD Ryzen 9 7950X (Zen 4, 16 core / 32 thread), WSL2 release build
- Compiler: stock `cargo build --release` (no `-C target-cpu=native` —
  runtime SIMD dispatch via archmage/magetypes)
- Corpus: `cid22:5` (first 5 images from CID22-512 validation)
- Qualities: 65, 75, 85, 95 — 20 encodes per (scalar|SIMD) × (default|on)
- Encoder config: YCbCr 4:2:0, Q65/75/85/95 — paired against
  `boundary_rd_on_default` (which is `BoundaryRd::On(BoundaryRdConfig::default())`)
- 3 repeats per configuration, median aggregated

## Command

```
cargo run --release -p zenjpeg --features "trellis decoder __test-utils" \
  --example rd_compare -- \
    --baseline default --candidate boundary_rd_on_default \
    --corpus cid22:5 --qualities 65,75,85,95 --metrics ssim2
```

## Results

| Build | `default` median (ms) | `on_default` median (ms) | boundary-RD work (ms) | Relative overhead |
|---|---:|---:|---:|---:|
| Scalar kernels (`ssd_col`, `ssd_seam_jump`, `ac_dct_energy`, scalar ×8 pack) | 2.283 | 3.575 | 1.292 | +56.6% |
| SIMD kernels (f32x8 FMA D_b, f32x8 FMA AC energy, f32x8 ×8 pack) | 2.305 | 3.384 | 1.079 | +46.8% |

- **Kernel speedup: 1.20x** (SIMD boundary-RD work 1.079ms vs scalar 1.292ms)
- **Kernel saving:  −16.6%** of the boundary-RD-only execution time

The ×8 scaler (`mage_scale_block_x8`), D_b fused computation
(`mage_boundary_distortion`), and AC energy sum-of-squares
(`mage_ac_dct_energy`) are now f32x8 FMA chains dispatched through
magetypes multi-tier (x86-v3 / NEON / wasm128 / scalar fallback). The
IDCT itself was already archmage-backed via the decoder's
`inverse_dct_8x8` on both builds.

## Honest framing

- The target in the task brief was **<12% total overhead**. We did not
  reach that: the SIMD changes improve the boundary-RD kernel by
  16.6% but total overhead remains +46.8% at Q85 for 512px CID22
  photos.
- The remaining overhead is dominated by the two IDCTs per block
  (`idct_reference_block` + `idct_quantized_block`), the repeated
  `quantize_with_zero_bias_zigzag` calls on refinement retries, and
  the scalar gather-multiply in `idct_quantized_block` that walks
  `JPEG_ZIGZAG_ORDER` to reconstruct a natural-order block. The IDCT
  is already archmage-backed; the gather is hard to SIMD cleanly
  without rewriting the inverse-zigzag as a shuffle table.
- Further overhead reduction would require algorithmic work: caching
  the left neighbor's reference-IDCT left-edge so the current block
  doesn't IDCT it again (currently recomputed every block for bx > 0,
  see `quantize_y_with_boundary_rd` in `strip/mod.rs`). That is
  out of scope for this SIMD-focused PR.
- The byte-identity guarantee is preserved: `BoundaryRd::Off` still
  produces identical output to a config that never touched the API,
  verified by `off_is_byte_identical_to_untouched_config` in
  `tests/boundary_rd_hash_lock.rs`.

## Verification

- **SIMD-vs-scalar parity tests** added to
  `zenjpeg/src/encode/boundary_rd.rs` (see `simd_boundary_distortion_matches_scalar`,
  `simd_ac_dct_energy_matches_scalar`, `simd_ac_dct_energy_dc_only_is_zero`).
  These lock the SIMD kernels against a scalar reference with 1e-5
  relative tolerance (f32 FMA vs scalar `+=` reorders add-accumulate
  and produces slightly different rounding).
- **Hash-lock test** (`off_is_byte_identical_to_untouched_config`)
  still passes — enabling the feature flag and leaving `BoundaryRd::Off`
  does not perturb any bits.
- **Full test suite** (`cargo test --release -p zenjpeg --features
  "trellis decoder"`) passes everywhere except the pre-existing
  `test_dispatch_parity` failure which also fails on `origin/main`
  and is unrelated (a SIMD dispatch-permutation sanity check in
  `tests/encoder_regression.rs`).
