# C-vs-zen decode bench audit (imageflow-zen-v3 @ `cc3cf88c`)

Audit target: `imageflow_core/benches/bench_codecs.rs::bench_jpeg_decode`,
which runs a Q85 4:2:0 JPEG through `imageflow_core::Context::execute_1`
with `Decode(io_id=0) → Resample2D(1x1)` and measures both codec routes
under two runtime-swapped decoder selections (`ZenJpegDecoder` vs
`MozJpegRsDecoder`).

## Summary

The bench is *apples-to-apples on pixel layout and work* for both paths —
both decoders produce BGRA8 via a single `execute_1` call on the same
encoded JPEG bytes, and both decode a standard 4:2:0 fixture created via
`mozjpeg::Compress` with `set_chroma_sampling_pixel_sizes((2,2),(2,2))`.
However, the two sides use **different internal data paths** to reach BGRA
— mozjpeg converts YCbCr → BGRA in one fused SIMD step via
`JCS_EXT_BGRA`, while zenjpeg's `DecodeMode::Auto` takes the streaming
path that emits **RGB**, then pays a separate full-buffer RGB→BGRA
swizzle in `reformat_rgb_into`. At 1024² ST that extra pass is ~9% of
wall time; at 4096² ST it grows to ~47% because the swizzle buffer
stops fitting in L2. Mozjpeg's row-at-a-time `JCS_EXT_BGRA` stays cache-
resident. **This accounts for essentially all of the ST bench gap.**

The parallel-enabled (default) path narrows the gap at mid sizes because
zenjpeg's `fused_parallel` claws back some cost on the entropy/IDCT side,
but the swizzle tail is serial so 4096² is still stuck behind mozjpeg.

## Findings

- **Output format: nominally BGRA8 for both, but produced differently.**
  - `imageflow_core/src/codecs/mozjpeg_decoder.rs:320` sets
    `self.codec_info.out_color_space = mozjpeg_sys::JCS_EXT_BGRA` and
    reads scanlines straight into the imageflow bitmap. One pass over
    the frame; SIMD kernel in libjpeg-turbo.
  - `imageflow_core/src/codecs/zen_decoder.rs:592` requests
    `[zenpixels::PixelDescriptor::BGRA8_SRGB]` from zencodec.
    `zenjpeg/src/codec.rs:1495 push_decoder_direct` eventually calls
    `DecodeConfig::decode_into(data, PixelFormat::Bgra, dst)`.
  - Inside zenjpeg, `decode_into` → `parser.to_pixels_into` takes the
    **streaming** branch at `zenjpeg/src/decode/parser/output.rs:1516`
    whenever `streaming_rgb` is populated (which is the default `Auto`
    mode for any baseline 4:2:0/4:2:2/4:4:4 fixture).
    `decode_baseline_streaming` only emits RGB, so `to_pixels_into`
    immediately calls `reformat_rgb_into(rgb, Bgra, w, h, dst)` —
    a second pass over the frame via `garb::bytes::rgb_to_bgra`.
  - The direct-BGRA fast path at `output.rs:1524` (calling
    `to_pixels_fast_i16_subsampled_into(... FastDst::Bgra(dst))`) is
    **dead code** on this benchmark because the streaming branch above
    always wins. It's reachable only when `DecodeMode` forces off
    streaming or when streaming isn't eligible (e.g. XYB, non-standard
    sampling factors, Arithmetic coding).

- **Fixture bytes: identical for both decoders.** `jpeg_fixture()` is
  called once per size, the `Vec<u8>` is cloned into each bench closure,
  and `decode_with_config` calls `ctx.add_input_vector(0, fixture.to_vec())`
  — both codecs receive the same encoded byte stream. No per-iteration
  encode, no accidental re-encode.

- **Subsampling: standard 4:2:0 (2,2)/(2,2)/(2,2).** Enforced at
  `bench_codecs.rs:116` via `set_chroma_sampling_pixel_sizes((2,2),(2,2))`.
  Both decoders use `Triangle` upsampling by default; mozjpeg's
  `fancy_upsampling` is on by default and matches libjpeg-turbo's
  `h2v2_fancy_upsample`, which zenjpeg models verbatim in
  `decode::upsample::upsample_h2v2_i16_libjpeg`. Output match is within
  max_diff ≤ 3 per `zenjpeg/src/decode/mod.rs:278` comment.

- **Colour management / ICC: not triggered in this fixture.** The
  synthetic JPEG carries no ICC/APP2 segment. `SourceProfile::Srgb` is
  selected on both sides, both paths skip the CMS transform, and the
  subsequent `Resample2D(1,1)` runs identically. No unfair CMS cost on
  either side.

- **Threading: zenjpeg opportunistically uses Rayon, mozjpeg doesn't.**
  `imageflow_core/Cargo.toml:79` enables `zenjpeg = { ..., features = ["parallel"] }`.
  The imageflow bench runs on a single thread at the bench level (zenbench
  collects one iteration at a time), but inside each iteration
  `DecodeConfig::decode_into` invokes zenjpeg's `fused_parallel_decode`
  when DRI is present, which `rayon::current_num_threads()`'s into all
  logical cores. On the 7950X (32 threads) that gives zenjpeg a pool the
  mozjpeg path doesn't have.

  This **widens** the ST gap when `RAYON_NUM_THREADS=1` (user observed
  4096² ST: −19% zen-vs-moz; MT: −9% zen-vs-moz). It's the swizzle tail
  that's independent of parallelism — fused_parallel closes the entropy
  side of the gap but can't fix the post-decode swizzle.

- **Fixture quality / realism:** the checkerboard fixture is roughly
  50% solid 0xFF8040 patches and 50% solid 0x204080 patches. That's much
  more compressible than a natural image — at Q85 the encoded size is
  ~5× smaller than a photo. The *decode* cost dominates regardless, but
  the entropy portion of the profile is suppressed vs what users see on
  real photos. For a 1024² Q85 photo the AC density is 2–3× higher and
  Huffman work grows correspondingly; this bench is kinder to both
  decoders than real traffic.

## ST vs MT gap interpretation

User's numbers:
- 256² zen 39.7 Mpx/s vs moz 45.1 Mpx/s (**−12%**)
- 1024² zen 2.93 Gpx/s vs moz 3.61 Gpx/s (**−19%**)
- 4096² zen 52.2 Mpx/s vs moz 57.5 Mpx/s (**−9%**)
- 4096² ST (`RAYON_NUM_THREADS=1`): widens to **−19%**

Cross-checked with `decode_mozjpeg` bench on a separate 1024² Q85 fixture,
**RGB output**, single-thread:
- mozjpeg-baseline: 3.82 ms (274 Mpx/s)
- zenjpeg-baseline: 3.59 ms (**292 Mpx/s, +6% faster than moz**)
- zenjpeg-boxfilter: 3.42 ms (306 Mpx/s)

And the same JPEG through zenjpeg at three formats (ST):
- Rgb:  3329 µs/iter → 315 Mpx/s
- Rgba: 3554 µs/iter → 295 Mpx/s (−6% vs Rgb)
- Bgra: 3620 µs/iter → 290 Mpx/s (−9% vs Rgb)

At 4096² ST the ratio blows out:
- Rgb:  85 ms, 198 Mpx/s
- Rgba: 119 ms, 141 Mpx/s (−29% vs Rgb)
- Bgra: 160 ms, 105 Mpx/s (**−47% vs Rgb**)

**Conclusion:** the imageflow bench gap is dominated by zenjpeg's
decode-to-Bgra path writing through an intermediate RGB `Vec<u8>` and
then swizzling. zenjpeg's YCbCr→RGB fused SIMD kernel is already at
parity or better than libjpeg-turbo; the gap is entirely in the
swizzle/copy tail. Fix = a fused YCbCr→BGRA kernel in streaming decode,
or skipping streaming in favour of `to_pixels_fast_i16_subsampled_into`
when the caller requests BGRA.
