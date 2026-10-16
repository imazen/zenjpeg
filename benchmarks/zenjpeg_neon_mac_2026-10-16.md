# macOS ARM64 NEON isolation, 2026-10-16

Source: `ce0220ad` (`main` before the benchmark recipe fix). Host: `mac`
(Apple M4 Pro, 12 logical CPUs). Command:

```
run-heavy --mem 12G --jobs 4 -- just arm-tiers-macos
```

The recipe uses `cargo bench -p zenjpeg --features _dev --bench tier_isolation`
with `--cpu-policy=unrestricted --format=llm`, default target CPU, and the
workspace's local path patches. The committed `Cargo.lock` hash was
`e7be6d0cf8f26e8e30725f65bae4a3f2c15cb16c6f6a0be96cb3932199f9adaa`;
Cargo's local path patch resolution changed only the zencodec source records
and added one unused patch record (resolved lock hash
`805eca0044a0fd1a971e09f364ea0a895b456f0ab0434b55b2b76d9d0d1cba32`).
The fixture is the benchmark's 1024×1024 noise-and-patches RGB image, encoded
as Q85 4:2:0. Its NEON and scalar encoders produced identical JPEG bytes, and
decoding the same JPEG produced identical pixels.

| Pipeline | NEON median | Forced scalar median | Scalar vs NEON (zenbench) |
|---|---:|---:|---:|
| Encode, Q85 4:2:0, 1 MP | 21.01 ms | 27.40 ms | +29.46% |
| Decode, Q85 4:2:0, 1 MP | 11.41 ms | 15.00 ms | +29.64% |

NEON also beat scalar in each measured small kernel: 8×8 forward DCT
(33.1 vs 113.1 ns), linear-to-sRGB ×8 (10.8 vs 36.5 ns), u16 linear-to-sRGB
×8 (9.1 vs 35.1 ns), RGB16-to-YCbCr ×8 (18.0 vs 117.9 ns), and
RGBf32-to-YCbCr ×8 (17.3 vs 109.2 ns). Those nanosecond groups had substantial
variation (up to 58% CV), so the whole-image measurements carry more weight.

This is exploratory evidence: macOS did not provide an enforceable performance
core boundary, and the unrestricted setting cannot authorize a zenbench
baseline promotion. No JPEG kernel change follows from these measurements.
`run-heavy` reported `rc=0`, peak RSS 0.20 GiB, minimum available 5199 MiB,
and peak load 5.08 over 90 seconds. The full output is
`~/tmp/zenjpeg-arm-tiers.log` on the measured Mac.
