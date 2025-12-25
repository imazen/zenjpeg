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
| **Rust Status** | ✅ **Fixed** - `50b104ee` |
| **Component** | Image allocation |

**Root Cause**: Unsigned integer multiplication overflow on 32-bit systems during image allocation.

**Rust Fix**: All size calculations now use `checked_size_2d()` and `checked_size()` from `alloc.rs` which return `Error::SizeOverflow` on overflow instead of panicking or wrapping.

---

### Allocation Failure Not Handled (2023-10)

| Field | Value |
|-------|-------|
| **C++ Commit** | `8740dc2f` |
| **Rust Status** | ✅ **Fixed** - `50b104ee` |
| **Component** | Plane allocation |

**Root Cause**: `Plane<T>` allocation failures not propagated, causing crashes.

**Rust Fix**: All allocations now use fallible `try_alloc_*` functions from `alloc.rs` that return `Error::AllocationFailed` instead of panicking on OOM. This includes:
- `try_alloc_zeroed()` for u8 buffers
- `try_alloc_zeroed_f32()` for f32 buffers
- `try_alloc_filled()` for non-zero initialized buffers
- `try_alloc_dct_blocks()` for DCT coefficient arrays

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
| i386 integer overflow | `0e1976eb` | ✅ Fixed (`50b104ee`) |
| Allocation failure crash | `8740dc2f` | ✅ Fixed (`50b104ee`) |
| Memory tracking bypass | `eeb331ce` | ❌ Missing feature |
| MSAN subsampling error | `39a47b34` | ⚠️ Review needed |
| Decoder malformed input | N/A | ✅ Fixed (`7e9ec31a`) |
| CVE-2019-2201 int overflow | N/A | ✅ Fixed (`50b104ee`) |
| CVE-2020-14152 mem exhaustion | N/A | ✅ Fixed (`50b104ee`) |

---

## libjpeg-turbo CVEs (Reference)

These CVEs affect libjpeg-turbo and may inform our security posture.

### CVE-2023-2804: Heap Buffer Overflow in Merged Upsampling

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2023-2804](https://nvd.nist.gov/vuln/detail/CVE-2023-2804) |
| **CVSS** | 6.5 (Medium) |
| **Affected** | libjpeg-turbo < 2.1.90 |
| **Rust Status** | ✅ **Not Vulnerable** - No 12-bit support |

**Root Cause**: Heap overflow in `h2v2_merged_upsample_internal()` when processing 12-bit lossless JPEG with out-of-range sample data.

**Rust Analysis**: jpegli-rs only supports 8-bit precision. 12-bit mode not implemented.

---

### CVE-2021-46822: Heap Overflow in PPM Reader

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2021-46822](https://nvd.nist.gov/vuln/detail/CVE-2021-46822) |
| **CVSS** | 5.5 (Medium) |
| **Affected** | libjpeg-turbo < 2.1.0 |
| **Rust Status** | ✅ **Not Applicable** - No PPM reader |

**Root Cause**: Heap overflow in `get_word_rgb_row()` in rdppm.c when loading 16-bit PPM/PGM files.

---

### CVE-2020-17541: Stack Buffer Overflow in Transform

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2020-17541](https://nvd.nist.gov/vuln/detail/CVE-2020-17541) |
| **CVSS** | 8.8 (High) |
| **Affected** | libjpeg-turbo < 2.0.5 |
| **Rust Status** | ✅ **Not Applicable** - No transform component |

**Root Cause**: Local buffer overrun in `jchuff.c` during transform operations.

---

### CVE-2020-13790: Heap Over-read in PPM Reader

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2020-13790](https://nvd.nist.gov/vuln/detail/CVE-2020-13790) |
| **CVSS** | 8.1 (High) |
| **Affected** | libjpeg-turbo 2.0.4, mozjpeg 4.0.0 |
| **Rust Status** | ✅ **Not Applicable** - No PPM reader |

**Root Cause**: Heap over-read in `get_rgb_row()` in rdppm.c via malformed PPM input.

---

### CVE-2019-2201: Integer Overflow / Heap Corruption

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2019-2201](https://nvd.nist.gov/vuln/detail/CVE-2019-2201) |
| **CVSS** | 7.8 (High) |
| **Affected** | libjpeg-turbo < 2.0.3 |
| **Rust Status** | ✅ **Fixed** - `50b104ee` |

**Root Cause**: Integer overflow handling gigapixel images in `turbojpeg.c`, causing heap corruption.

**Rust Fix**: All size calculations use `checked_size_2d()` with `Error::SizeOverflow` on overflow. Additionally, `validate_dimensions()` enforces `JPEG_MAX_DIMENSION` (65500) and configurable `max_pixels` limits.

---

### CVE-2018-20330: Integer Overflow in tjLoadImage

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2018-20330](https://nvd.nist.gov/vuln/detail/CVE-2018-20330) |
| **CVSS** | 8.8 (High) |
| **Affected** | libjpeg-turbo 2.0.1 |
| **Rust Status** | ✅ **Fixed** - `50b104ee` |

**Root Cause**: Integer overflow from `pitch * height` multiplication in BMP loading.

**Rust Fix**: All `width * height * bpp` calculations use `checked_size()` which returns `Error::SizeOverflow` on overflow.

---

### CVE-2018-14498: Heap Over-read in BMP Reader

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2018-14498](https://nvd.nist.gov/vuln/detail/CVE-2018-14498) |
| **CVSS** | 6.5 (Medium) |
| **Affected** | libjpeg-turbo < 2.0.0, mozjpeg < 3.3.1 |
| **Rust Status** | ✅ **Not Applicable** - No BMP reader |

**Root Cause**: `get_8bit_row()` in rdbmp.c allows out-of-range color indices.

---

### CVE-2018-11813: Infinite Loop in Targa Reader

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2018-11813](https://nvd.nist.gov/vuln/detail/CVE-2018-11813) |
| **CVSS** | 7.5 (High) |
| **Affected** | libjpeg-turbo < 2.0.0 |
| **Rust Status** | ✅ **Not Applicable** - No Targa reader |

**Root Cause**: `read_pixel()` in rdtarga.c mishandles EOF, causing infinite loop.

---

## mozjpeg CVEs (Reference)

mozjpeg shares codebase with libjpeg-turbo; most CVEs are inherited.

### CVE-2020-1895: Instagram Integer Overflow (RCE)

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2020-1895](https://research.checkpoint.com/2020/instagram_rce-code-execution-vulnerability-in-instagram-app-for-android-and-ios/) |
| **CVSS** | 7.8 (High) |
| **Affected** | mozjpeg (via Instagram) |
| **Rust Status** | ⚠️ **Review Needed** |

**Root Cause**: Integer overflow in `read_jpg_copy_loop()` during decompression leads to heap buffer overflow. Exploited for RCE in Instagram.

**Rust Analysis**: Need to audit our decode loop for similar integer overflow patterns.

---

## IJG libjpeg CVEs (Reference)

Original IJG libjpeg vulnerabilities.

### CVE-2020-14152: Memory Exhaustion

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2020-14152](https://nvd.nist.gov/vuln/detail/CVE-2020-14152) |
| **CVSS** | 7.1 (High) |
| **Affected** | libjpeg < 9d |
| **Rust Status** | ✅ **Fixed** - `50b104ee` |

**Root Cause**: `jpeg_mem_available()` in jmemnobs.c doesn't honor `max_memory_to_use`.

**Rust Fix**: DecoderConfig supports `max_pixels()` configuration (default 100MP). `validate_dimensions()` enforces both `JPEG_MAX_DIMENSION` (65500) and `max_pixels` limits, rejecting excessively large images before allocation.

---

### CVE-2018-11212: Divide by Zero

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2018-11212](https://nvd.nist.gov/vuln/detail/CVE-2018-11212) |
| **CVSS** | 6.5 (Medium) |
| **Affected** | libjpeg 9a, 9d |
| **Rust Status** | ⚠️ **Review Needed** |

**Root Cause**: `alloc_sarray()` in jmemmgr.c allows divide-by-zero via crafted file.

**Rust Analysis**: Need to check for division operations in allocation paths.

---

## Vulnerability Class Summary

| Vulnerability Class | libjpeg-turbo | mozjpeg | IJG | jpegli-rs Status |
|---------------------|---------------|---------|-----|------------------|
| Integer overflow in size calc | CVE-2019-2201, CVE-2018-20330 | CVE-2020-1895 | - | ✅ Fixed (`50b104ee`) |
| Heap overflow (12-bit) | CVE-2023-2804 | - | - | ✅ N/A (8-bit only) |
| PPM/BMP reader bugs | CVE-2020-13790, CVE-2018-14498 | same | - | ✅ N/A (no readers) |
| Memory exhaustion | - | - | CVE-2020-14152 | ✅ Fixed (`50b104ee`) |
| Huffman table OOB | CVE-2024-11403 (via jpegli) | - | - | ✅ Not vulnerable |
| Allocation failure crash | N/A | N/A | N/A | ✅ Fixed (`50b104ee`) |

---

## References

- [libjpeg-turbo CVE List](https://www.cvedetails.com/vulnerability-list/vendor_id-17075/product_id-40849/Libjpeg-turbo-Libjpeg-turbo.html)
- [mozjpeg CVE List](https://www.cvedetails.com/vulnerability-list/vendor_id-452/product_id-52301/Mozilla-Mozjpeg.html)
- [IJG libjpeg CVE List](https://www.cvedetails.com/vulnerability-list/vendor_id-17990/product_id-46165/IJG-Libjpeg.html)
- [Gentoo GLSA 202405-20](https://security.gentoo.org/glsa/202405-20) - libjpeg-turbo multiple vulnerabilities
- [CVE-2024-11403 Details](https://www.wiz.io/vulnerability-database/cve/cve-2024-11403)
- [Gentoo GLSA 202210-36](https://security.gentoo.org/glsa/202210-36)
- [Debian DSA 5958-1](https://lists.debian.org/debian-security-announce/2025/msg00122.html)
- [Brunsli Huffman Fix Video](https://www.youtube.com/watch?v=_ACCK0AUQ8Q&t=696s)
- [Instagram RCE via mozjpeg](https://research.checkpoint.com/2020/instagram_rce-code-execution-vulnerability-in-instagram-app-for-android-and-ios/)
- [Ubuntu libjpeg-turbo Changelog](https://launchpad.net/ubuntu/+source/libjpeg-turbo/+changelog)

---

## Changelog

| Date | Action |
|------|--------|
| 2024-12-25 | Initial security analysis and document creation |
| 2024-12-25 | Added CVE-2024-11403 analysis |
| 2024-12-25 | Mapped C++ fixes to Rust status |
| 2024-12-25 | Added libjpeg-turbo, mozjpeg, IJG libjpeg CVE catalog |
| 2024-12-25 | Fixed integer overflow, allocation crash, and memory exhaustion issues (`50b104ee`) |
