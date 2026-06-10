# Changelog

All notable changes to zenjpeg are documented here. Earlier history
(pre-2026-06) lives in `git log` and `docs/TUNING_HISTORY.md`.

## [Unreleased]

### Added

- `detect`: Windows GDI+/WIC encoder detection
  (`EncoderFamily::WindowsImaging`, `QualityScale::WindowsQuality`).
  Windows emits byte-exact IJG tables at index `quality - 1` (except
  multiples of 25); detection keys on the JFIF 96×96 DPI density
  stamp vs libjpeg-turbo's 1×1 aspect ratio. Verified against a real
  q=1..=100 Windows-encoded sweep (100/100 family + quality recovery;
  fixtures in `zenjpeg/tests/testdata/windows_encoder/`, analysis in
  `docs/quality_estimation_research.md`).
- `jpeg_inspect`: `--detect` flag prints encoder family + estimated
  quality.
