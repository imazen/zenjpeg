//! Umbrella binary bundling 117 integration tests into one compile unit.
//!
//! Each bundled file keeps its own `#![cfg(...)]` inner attrs; when a
//! feature is off, the module body becomes empty. Files carry their own
//! gating so they also compile standalone if moved back out.
//!
//! Kept standalone (in `tests/` root, not here):
//!   - encoder_regression.rs         — archmage token permutation/lock
//!   - decode_path_dispatch_parity.rs — archmage token permutation
//!   - corpus_cpp_comparison.rs      — env::set_var(MAX_IMAGES)
//!
//! rustc treats this file as a crate root, so `mod foo;` without `#[path]`
//! would search next to it (tests/foo.rs) — we explicitly point at
//! `bundled/foo.rs` via `#[path]`.

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::all)]

// Shared test helpers (was duplicated via `#[path]` in each bundled file,
// which collided on its `#[macro_export] macro_rules! skip_if_missing`).
// One definition at the umbrella level puts the macro in the crate root of
// this test binary; submodules reference items as `crate::test_utils::*`.
#[path = "../src/test_utils.rs"]
mod test_utils;

#[path = "bundled/ac_refinement_cpp_comparison.rs"]
mod ac_refinement_cpp_comparison;
#[path = "bundled/ac_refinement_image_types.rs"]
mod ac_refinement_image_types;
#[path = "bundled/ac_refinement_parity.rs"]
mod ac_refinement_parity;
#[path = "bundled/arithmetic_coef_verify.rs"]
mod arithmetic_coef_verify;
#[path = "bundled/arithmetic_decode.rs"]
mod arithmetic_decode;
#[path = "bundled/arithmetic_first_block.rs"]
mod arithmetic_first_block;
#[path = "bundled/arithmetic_first_decode.rs"]
mod arithmetic_first_decode;
#[path = "bundled/arithmetic_minimal.rs"]
mod arithmetic_minimal;
#[path = "bundled/arithmetic_tolerance.rs"]
mod arithmetic_tolerance;
#[path = "bundled/chroma_benchmark.rs"]
mod chroma_benchmark;
#[path = "bundled/chroma_upsample_regression.rs"]
mod chroma_upsample_regression;
#[path = "bundled/cmyk_transform.rs"]
mod cmyk_transform;
#[path = "bundled/codec_corpus_conformance.rs"]
mod codec_corpus_conformance;
#[path = "bundled/codec_coverage.rs"]
mod codec_coverage;
#[path = "bundled/compare_420_cpp.rs"]
mod compare_420_cpp;
#[path = "bundled/compare_sizes.rs"]
mod compare_sizes;
#[path = "bundled/comprehensive_cpp_comparison.rs"]
mod comprehensive_cpp_comparison;
#[path = "bundled/conformance_corpus.rs"]
mod conformance_corpus;
#[path = "bundled/corpus_decode_all.rs"]
mod corpus_decode_all;
#[path = "bundled/corpus_decoder_comparison.rs"]
mod corpus_decoder_comparison;
#[path = "bundled/corpus_zensim_comparison.rs"]
mod corpus_zensim_comparison;
#[path = "bundled/coverage_extension.rs"]
mod coverage_extension;
#[path = "bundled/coverage_upsample_and_api.rs"]
mod coverage_upsample_and_api;
#[path = "bundled/cpp_comparison.rs"]
mod cpp_comparison;
#[path = "bundled/cpp_filesize_comparison.rs"]
mod cpp_filesize_comparison;
#[path = "bundled/cpp_parity_locked.rs"]
mod cpp_parity_locked;
#[path = "bundled/cpp_quality_comparison.rs"]
mod cpp_quality_comparison;
#[path = "bundled/cpp_reference_parity.rs"]
mod cpp_reference_parity;
#[path = "bundled/crash_sweep.rs"]
mod crash_sweep;
#[path = "bundled/crop_corpus.rs"]
mod crop_corpus;
#[path = "bundled/deblock_api.rs"]
mod deblock_api;
#[path = "bundled/deblock_path_parity.rs"]
mod deblock_path_parity;
#[path = "bundled/debug_gray_linear.rs"]
mod debug_gray_linear;
#[path = "bundled/decode_accuracy_corpus.rs"]
mod decode_accuracy_corpus;
#[path = "bundled/decode_api.rs"]
mod decode_api;
#[path = "bundled/decode_api_guide.rs"]
mod decode_api_guide;
#[path = "bundled/decode_callback.rs"]
mod decode_callback;
#[path = "bundled/decode_external.rs"]
mod decode_external;
#[path = "bundled/decode_perf_locked.rs"]
mod decode_perf_locked;
#[path = "bundled/decode_xyb_failures.rs"]
mod decode_xyb_failures;
#[path = "bundled/decoder_consistency.rs"]
mod decoder_consistency;
#[path = "bundled/decoder_defaults_eval.rs"]
mod decoder_defaults_eval;
#[path = "bundled/decoder_error_handling.rs"]
mod decoder_error_handling;
#[path = "bundled/decoder_extras.rs"]
mod decoder_extras;
#[path = "bundled/decoder_leniency_comparison.rs"]
mod decoder_leniency_comparison;
#[path = "bundled/decoder_parity.rs"]
mod decoder_parity;
#[path = "bundled/dequant_bias_comparison.rs"]
mod dequant_bias_comparison;
#[path = "bundled/deringing_quality.rs"]
mod deringing_quality;
#[path = "bundled/determinism.rs"]
mod determinism;
#[path = "bundled/diagnose_decoder_diff.rs"]
mod diagnose_decoder_diff;
#[path = "bundled/dump_decoder_diffs.rs"]
mod dump_decoder_diffs;
#[path = "bundled/edge_tile_ssim2_comparison.rs"]
mod edge_tile_ssim2_comparison;
#[path = "bundled/effort_benchmark.rs"]
mod effort_benchmark;
#[path = "bundled/encode_api.rs"]
mod encode_api;
#[path = "bundled/encode_request_guide.rs"]
mod encode_request_guide;
#[path = "bundled/encoder_matrix.rs"]
mod encoder_matrix;
#[path = "bundled/error_handling.rs"]
mod error_handling;
#[path = "bundled/fast_math_cpp_comparison.rs"]
mod fast_math_cpp_comparison;
#[path = "bundled/fused_parallel_decode.rs"]
mod fused_parallel_decode;
#[path = "bundled/grayscale_decode_test.rs"]
mod grayscale_decode_test;
#[path = "bundled/high_bit_depth_roundtrip.rs"]
mod high_bit_depth_roundtrip;
#[path = "bundled/icc_corpus_extraction.rs"]
mod icc_corpus_extraction;
#[path = "bundled/icc_extraction.rs"]
mod icc_extraction;
#[path = "bundled/idct_comparison.rs"]
mod idct_comparison;
#[path = "bundled/imageflow_corpus_zensim.rs"]
mod imageflow_corpus_zensim;
#[path = "bundled/issue27_progressive_dc_pt.rs"]
mod issue27_progressive_dc_pt;
#[path = "bundled/issue7_repro.rs"]
mod issue7_repro;
#[path = "bundled/layout_pipeline.rs"]
mod layout_pipeline;
#[path = "bundled/linear_pixel_formats.rs"]
mod linear_pixel_formats;
#[path = "bundled/locked_values.rs"]
mod locked_values;
#[path = "bundled/mcu_border_roundtrip.rs"]
mod mcu_border_roundtrip;
#[path = "bundled/metadata_integration.rs"]
mod metadata_integration;
#[path = "bundled/metrics_comparison.rs"]
mod metrics_comparison;
#[path = "bundled/multi_decoder_compatibility.rs"]
mod multi_decoder_compatibility;
#[path = "bundled/new_api_test.rs"]
mod new_api_test;
#[path = "bundled/orientation_descriptor.rs"]
mod orientation_descriptor;
#[path = "bundled/parametrized_quality.rs"]
mod parametrized_quality;
#[path = "bundled/parity_enforcement.rs"]
mod parity_enforcement;
#[path = "bundled/parity_reference_locked.rs"]
mod parity_reference_locked;
#[path = "bundled/permutation_corpus_decode.rs"]
mod permutation_corpus_decode;
#[path = "bundled/permutation_regression.rs"]
mod permutation_regression;
#[path = "bundled/photo_corpus_zensim.rs"]
mod photo_corpus_zensim;
#[path = "bundled/photoshop_444_regression.rs"]
mod photoshop_444_regression;
#[path = "bundled/precision_matrix.rs"]
mod precision_matrix;
#[path = "bundled/progressive_encoding.rs"]
mod progressive_encoding;
#[path = "bundled/progressive_requires_optimization.rs"]
mod progressive_requires_optimization;
#[path = "bundled/progressive_subsampling_bug.rs"]
mod progressive_subsampling_bug;
#[path = "bundled/progressive_xyb_decode.rs"]
mod progressive_xyb_decode;
#[path = "bundled/q100_comparison.rs"]
mod q100_comparison;
#[path = "bundled/quality_matrix.rs"]
mod quality_matrix;
#[path = "bundled/quality_regression.rs"]
mod quality_regression;
#[path = "bundled/quant_16bit_comparison.rs"]
mod quant_16bit_comparison;
#[path = "bundled/quant_config_effects.rs"]
mod quant_config_effects;
#[path = "bundled/roundtrip_corpus.rs"]
mod roundtrip_corpus;
#[path = "bundled/roundtrip_quality.rs"]
mod roundtrip_quality;
#[path = "bundled/rst_resync.rs"]
mod rst_resync;
#[path = "bundled/s440_progressive_bug.rs"]
mod s440_progressive_bug;
#[path = "bundled/scan_optimize_integration.rs"]
mod scan_optimize_integration;
#[path = "bundled/strip_edge_cpp_comparison.rs"]
mod strip_edge_cpp_comparison;
#[path = "bundled/subsampling_tests.rs"]
mod subsampling_tests;
#[path = "bundled/test_prog_xyb_quality.rs"]
mod test_prog_xyb_quality;
#[path = "bundled/trellis_config_effects.rs"]
mod trellis_config_effects;
#[path = "bundled/trellis_mozjpeg_comparison.rs"]
mod trellis_mozjpeg_comparison;
#[path = "bundled/ultrahdr_gainmap_decode.rs"]
mod ultrahdr_gainmap_decode;
#[path = "bundled/ultrahdr_roundtrip.rs"]
mod ultrahdr_roundtrip;
#[path = "bundled/visual_diff_regression.rs"]
mod visual_diff_regression;
#[path = "bundled/visual_encoding.rs"]
mod visual_encoding;
#[path = "bundled/wasm_decode.rs"]
mod wasm_decode;
#[path = "bundled/wasm_simd.rs"]
mod wasm_simd;
#[path = "bundled/xyb_cpp_comparison.rs"]
mod xyb_cpp_comparison;
#[path = "bundled/xyb_encoding_basic.rs"]
mod xyb_encoding_basic;
#[path = "bundled/xyb_linear_cpp_parity.rs"]
mod xyb_linear_cpp_parity;
#[path = "bundled/xyb_roundtrip.rs"]
mod xyb_roundtrip;
#[path = "bundled/ycbcr_locked.rs"]
mod ycbcr_locked;
#[path = "bundled/yuv_crate_comparison.rs"]
mod yuv_crate_comparison;
#[path = "bundled/zero_quant_clamp.rs"]
mod zero_quant_clamp;
#[path = "bundled/zune_crash_repro.rs"]
mod zune_crash_repro;
