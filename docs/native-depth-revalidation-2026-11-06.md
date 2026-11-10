# Deep CMYK output revalidation, 2026-11-06

At `a8ebcd0b`, the existing regression passes on both native seed files:
`16-p12-c4-sof30-mode0-s1.jpg` and `33-p16-c4-sof31-mode0-s1.jpg`.
Default adapter decode returns `CmykF32Inverted`; every float equals the native
integer decoder code divided by 4095 or 65535 respectively, exactly as F32.
An explicit `CMYK8` preference still returns CMYK8. No test expectations or
production output behavior changed.

Run `just deep-cmyk-check` from the workspace root. `cargo fmt -p zenjpeg --check`
also passed. This is an adapter-versus-native decoder comparison, not a fresh
independent libjpeg-turbo decode. It does not qualify all CMYK files, gray ICC
lowering, or a new shared dependency cohort.

The resource wrapper recorded 74 seconds including compilation, peak RSS
2.20 GiB, minimum available 14042 MiB and peak load 5.18. These describe the
test/build process, not codec performance or per-image memory consumption.
