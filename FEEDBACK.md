# User Feedback Log

## 2026-02-02
- User request: Implement plan for mozjpeg-rs bug fixes + OptimizationPreset enum
  - Bug 1: DC trellis row-by-row propagation (mozjpeg-rs 8c7f411)
  - Bug 2: delta_dc_weight support in TrellisConfig (mozjpeg-rs ec6db5a)
  - Bug 3: Scan optimizer freq split only at Al=0 (mozjpeg-rs 01fddb9)
  - Bug 4: Freq-split-beats-SA comparison (mozjpeg-rs 01fddb9)
  - OptimizationPreset enum with 8 variants across 3 lineages (jpegli/mozjpeg/hybrid)
- User request: Fix 3 bugs in optimization() + unify state to prevent invalid combos
  - Fixed: deringing decoupled from AQ (was incorrectly tied to uses_aq())
  - Fixed: mozjpeg presets now use Thorough trellis (was incorrectly using Adaptive)
  - Fixed: TrellisSpeedMode::Adaptive comments no longer claim C mozjpeg parity
  - Added: QuantTableConfig enum (Jpegli/JpegliSharedChroma/MozjpegRobidoux/Custom)
  - Added: ScanMode enum (Baseline/Progressive/ProgressiveMozjpeg/ProgressiveSearch)
  - EncoderConfig now uses type-safe enums instead of loose field combos
- User request: Implement ExpertConfig for external optimization (simulated annealing)
  - Flat struct exposing all ~30 tunable encoder parameters with no overlapping fields
  - from_preset() for all 8 OptimizationPreset variants, to_encoder_config() for encoding
  - TrellisConfig fields changed pub(crate) → pub to allow direct construction
  - Review pass: found/fixed dead code in blend_zero_bias() and to_encoder_config()
  - Documented 4 dead params in hybrid mode (pre-existing HybridConfig limitation)
- User request: Dig into ignored params — test file size with permutations, find sensible ranges
  - Wrote test_parameter_sensitivity: 256x256 noise+patches, all fields permuted
  - Found hybrid mode completely broken (create_hybrid_ctx never called, all coupling values = no trellis)
  - Found 4 standalone trellis dead params: eob_opt (disabled, later deleted), lambda_weight_tbl (flat weights), num_loops (never read), speed_mode (same output)
  - Found quality has zero effect on mozjpeg presets (Exact tables + zero zero-bias)
  - Documented working params: quant tables (±65%), lambda_scale1/2 (±46%), zero_bias_mul (±31%), scan_mode (up to -2%)
  - Updated all ExpertConfig field docs with measured impact data and ranges

## 2026-01-31
- User request: Fix pre-erosion lookahead timing (C++ has 4-row overlap at iMCU boundaries)
- Investigation found root cause was v_samp=1 for XYB AQ instead of v_samp=2
- User noted concern about memory usage from large buffer dumps, suggested expanding rotating buffer instead
- User request: Implement LayoutParams immutable substruct refactor (planned in CLAUDE.md TODO)
  - Goal: eliminate derived-state sync bugs by computing all geometry once
  - Result: completed, removed 12 fields from StripProcessor, removed set_xyb_mode/set_strip_stride
- User request: Add QuantTableSource enum + wire granular flags in OptimizationPreset
  - Goal: mozjpeg presets switch to Robidoux tables, all presets set allow_16bit_quant_tables=false
  - Result: completed, new QuantTableSource enum (Jpegli/MozjpegDefault), mozjpeg_table_data.rs always compiled, pipeline wired through
- User request: Investigate why zenjpeg Permissive mode rejects files that libjpeg-turbo accepts
  - Goal: categorize error types for 8 specific invalid JPEG files from conformance corpus
  - Result: identified 4 error categories, all files are heavily corrupted fuzz targets

## 2026-02-15
- User request: Continue decode speed optimization, implement Strictness::Permissive
  - Added 4th strictness level with 6 recovery mechanisms: RST resync, zero quant clamp,
    malformed segment skip, DNL skip, Huffman table index clamp, header error recovery
  - Results: 8 more files accepted vs Lenient, non-conformant parity with libjpeg-turbo (14/20)
  - 17 remaining gap files are 613-byte fuzz-mutated (diminishing returns)
