# Locked Encoder Values

This folder contains locked reference values for encoder output verification.

## Files

- `values_archmage.csv` - Expected hash/size values for archmage-simd variant
- `values_wide.csv` - Expected hash/size values for wide crate variant
- `history/` - Previous versions with justification for each change

## How It Works

1. **Tests load the appropriate CSV** via `include_str!` at compile time based on `#[cfg]`
2. **Each file is protected** by its own SHA-256 hash constant in the test code
3. **Regeneration requires explicit action** - set `REGENERATE_LOCKED_VALUES=1`
4. **Regeneration always fails** - forces you to update the hash constant in code
5. **History is preserved** - old versions are moved to `history/` with justification

## SIMD Variants

The encoder produces slightly different output depending on SIMD implementation:

- **archmage** (`#[cfg(target_arch = "x86_64")]`): Uses token-based intrinsics (mandatory on x86_64)
- **wide** (`#[cfg(not(target_arch = "x86_64"))]`): Uses `wide` crate for portable SIMD

Most entries are identical, but Q90 configurations may differ due to floating-point rounding in DCT.

## Updating Values

When encoder output intentionally changes:

```bash
# 1. Regenerate archmage values (fails, but writes new CSV)
REGENERATE_LOCKED_VALUES=1 cargo test --release -p zenjpeg --test locked_values -- regenerate --ignored --nocapture

# 2. Regenerate wide values (fails, but writes new CSV)
REGENERATE_LOCKED_VALUES=1 cargo test --release -p zenjpeg --test locked_values --no-default-features --features "std,yuv" -- regenerate --ignored --nocapture

# 3. Archive old files to history/ with justification
cp tests/locked_values/values_archmage.csv "tests/locked_values/history/$(date +%Y-%m-%d)_archmage_description.csv"
cp tests/locked_values/values_wide.csv "tests/locked_values/history/$(date +%Y-%m-%d)_wide_description.csv"

# 4. Update the VALUES_FILE_HASH constants in locked_values.rs with the new hashes

# 5. Run tests again to verify
cargo test --release -p zenjpeg --test locked_values
cargo test --release -p zenjpeg --test locked_values --no-default-features --features "std,yuv"
```

## File Format

```csv
# Locked encoder output values for frymire.png (1118x1105)
# Generated: 2026-02-01
# Reason: Initial generation with allow_16bit_quant_tables=false default
# SIMD variant: archmage
#
# Fields: mode,subsampling,huffman,quality,simd,hash,size
baseline,444,opt,50,archmage,76b246b90d689af3b427ca575074819df7d1c4cd80ceed8e70f465d369803aec,330100
...
```

## Why This Design

- **Explicit updates**: Can't accidentally change values
- **Audit trail**: History folder shows what changed and why
- **Compile-time inclusion**: No runtime file I/O in tests
- **Git-friendly**: CSV diffs are readable
- **cfg-parameterized**: Each SIMD variant has its own file and hash constant
- **Independent updates**: Changing one variant doesn't affect the other's hash
