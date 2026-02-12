# zenjpeg — CCTX/KLT branch (abandoned)

Per-image KLT color decorrelation for JPEG, inspired by AV2's CCTX. Computes an optimal 3x3 color transform per image via eigendecomposition of the RGB covariance matrix, embeds the inverse via an ICC matrix profile for decoder compatibility.

**Result: not viable within JPEG.** Even with full pipeline optimization (trellis quantization, AQ, optimized Huffman, eigenvalue-adapted quant tables), KLT costs +25-46% more bits at matched perceptual quality (BD-rate, 5 representative CID22 images, butteraugli + ssimulacra2). This is consistent with published research — Malvar & Sullivan showed the theoretical ceiling for KLT-derived color transforms is only ~0.5-1.0 dB over YCbCr, and HEVC's ACT (with a codec designed for it) only achieves -2.2% BD-rate. JPEG's fixed 8x8 DCT, perceptually-tuned YCbCr quant tables, and chroma subsampling leave no room for KLT to help.

See commit history for the full investigation.

---

## zenjpeg

Pure Rust JPEG encoder/decoder with perceptual optimizations. Port of Google's jpegli from the JPEG XL project.

## License

Sustainable, large-scale open source work requires a funding model, and I have been
doing this full-time for 15 years. If you are using this for closed-source development
AND make over $1 million per year, you'll need to buy a commercial license at
https://www.imazen.io/pricing

Commercial licenses are similar to the Apache 2 license but company-specific, and on
a sliding scale. You can also use this under the AGPL v3.
