# Security Issue Tracker

This document tracks security vulnerabilities in C++ jpegli/libjxl and their status in jpegli-rs.

## CVE Vulnerabilities

### CVE-2024-11403: Huffman Table Out-of-Bounds Write

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2024-11403](https://www.miggo.io/vulnerability-database/cve/CVE-2024-11403) |
| **Severity** | High |
| **C++ Fix** | `f510b589` (cherry-pick of `9cc451b91b74ba470fd72bd48c121e9f33d24c99`) |
| **Rust Status** | ✅ **Not Vulnerable** - Different implementation |
| **Description** | JPEG decoder Huffman table builder doesn't properly check bounds with incomplete codes |

**Details**: The C++ `kJpegHuffmanLutSize` was 758 entries, but malformed JPEGs with incomplete Huffman codes could cause out-of-bounds writes. Fixed by increasing to 1024 entries.

**Rust Analysis**: `huffman.rs` uses `Vec` with dynamic sizing and bounds-checked access. The fast_lookup table is fixed at `1 << FAST_BITS` (512 entries) with explicit bounds check at line 251:
```rust
if idx < table.fast_lookup.len() {
    table.fast_lookup[idx] = ...;
}
```

---

### CVE-2021-22564: Memory Corruption in JPEG XL Handler

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2021-22564](https://vuldb.com/?id.185862) |
| **Severity** | Critical |
| **C++ Fix** | libjxl 0.6.0 |
| **Rust Status** | ✅ **Not Applicable** - JXL decoder not implemented |

---

## C++ jpegli Security Fixes

### Buffer Overflow in Chroma Refinement (2024-10)

| Field | Value |
|-------|-------|
| **C++ Commit** | `c0dfce4c` |
| **Rust Status** | ⚠️ **Review Needed** |
| **Component** | Progressive encoder, entropy coding |

**Root Cause**: `num_refinement_bits` computed incorrectly when multiple color components have different refinement sequences. Single 1D array `int num_refinement_scans[DCTSIZE2]` was inadequate.

**C++ Fix**:
```cpp
// Before: single array for all components
int num_refinement_scans[DCTSIZE2];

// After: per-component tracking
// (actual fix adds proper bounds checking)
```

**Rust Analysis**: `encode.rs` progressive encoding uses different structure. Need to verify `encode_progressive_scan()` handles multi-component refinement correctly.

---

### Assertion Failure in Refinement Bits (2024-09)

| Field | Value |
|-------|-------|
| **C++ Commit** | `403631c7` |
| **Rust Status** | ✅ **Not Vulnerable** - No assertions in hot path |
| **Component** | Entropy coding |

**Root Cause**: Assertion `m->next_refinement_bit == refinement_bits + num_refinement_bits` was too strict for chroma refinement.

**C++ Fix**: Changed `==` to `<=` since num_refinement_bits is an upper bound.

---

### Crash with _GLIBCXX_ASSERTIONS (2022-06)

| Field | Value |
|-------|-------|
| **C++ Commit** | `5653d8f2` |
| **Rust Status** | ✅ **Not Applicable** - Different runtime |
| **Component** | Encoder finish |

**Root Cause**: Accessing `huffman_codes[dht_index]` when `num_dht == 0`.

---

### Integer Overflow on i386 (2024-07)

| Field | Value |
|-------|-------|
| **C++ Commit** | `0e1976eb` |
| **Rust Status** | ❌ **Vulnerable** - Same issue possible |
| **Component** | Image allocation |

**Root Cause**: Unsigned integer multiplication overflow on 32-bit systems during image allocation.

**Rust Status**: We have the same issue. See `SECURITY.md` P0-3 for fix using `checked_mul`.

---

### Allocation Failure Not Handled (2023-10)

| Field | Value |
|-------|-------|
| **C++ Commit** | `8740dc2f` |
| **Rust Status** | ❌ **Vulnerable** - Same issue |
| **Component** | Plane allocation |

**Root Cause**: `Plane<T>` allocation failures not propagated, causing crashes.

**Rust Status**: We use `vec![]` which panics on OOM. See `SECURITY.md` P0-1 for fix using `try_reserve`.

---

### Large Allocations Not Tracked (2024-06)

| Field | Value |
|-------|-------|
| **C++ Commit** | `eeb331ce` |
| **Rust Status** | ❌ **Missing Feature** |
| **Component** | Memory manager |

**Root Cause**: Some large allocations weren't tracked by memory manager, allowing memory limit bypass.

**Rust Status**: We don't have allocation tracking yet. See `SECURITY.md` P3-1.

---

### MSAN Error with Mixed Subsampling (2024-10)

| Field | Value |
|-------|-------|
| **C++ Commit** | `39a47b34` |
| **Rust Status** | ⚠️ **Review Needed** |
| **Component** | Decoder, subsampling |

**Root Cause**: Uninitialized memory read with images having both 4x and 2x subsampled channels.

**Rust Analysis**: Need to verify `decode.rs` subsampling handling initializes all buffers.

---

## jpegli-rs Security Fixes

### Decoder Hardening (2024-12)

| Field | Value |
|-------|-------|
| **Rust Commit** | `7e9ec31a` |
| **Components** | huffman.rs, entropy.rs, decode.rs |

**Fixes Applied**:
- Bounds check in Huffman fast_lookup table construction
- Safe `get_dc_table`/`get_ac_table` helper methods
- Fixed `decode_value` overflow for category > 15
- Use `wrapping_add` for DC prediction
- Clamp Huffman table indices to valid range

**Test Coverage**: 1836 fuzz corpus files tested, 0 panics.

---

## Vulnerability Status Summary

| Issue | C++ Commit | Rust Status |
|-------|------------|-------------|
| CVE-2024-11403 Huffman OOB | `f510b589` | ✅ Not vulnerable |
| Chroma refinement overflow | `c0dfce4c` | ⚠️ Review needed |
| Refinement bits assertion | `403631c7` | ✅ Not vulnerable |
| i386 integer overflow | `0e1976eb` | ❌ Vulnerable |
| Allocation failure crash | `8740dc2f` | ❌ Vulnerable |
| Memory tracking bypass | `eeb331ce` | ❌ Missing feature |
| MSAN subsampling error | `39a47b34` | ⚠️ Review needed |
| Decoder malformed input | N/A | ✅ Fixed (`7e9ec31a`) |

---

## References

- [libjxl Security Policy](https://github.com/libjpeg-turbo/libjpeg-turbo/security/policy)
- [CVE-2024-11403 Details](https://www.wiz.io/vulnerability-database/cve/cve-2024-11403)
- [Gentoo GLSA 202210-36](https://security.gentoo.org/glsa/202210-36)
- [Debian DSA 5958-1](https://lists.debian.org/debian-security-announce/2025/msg00122.html)
- [Brunsli Huffman Fix Video](https://www.youtube.com/watch?v=_ACCK0AUQ8Q&t=696s)

---

## Changelog

| Date | Action |
|------|--------|
| 2024-12-25 | Initial security analysis and document creation |
| 2024-12-25 | Added CVE-2024-11403 analysis |
| 2024-12-25 | Mapped C++ fixes to Rust status |
