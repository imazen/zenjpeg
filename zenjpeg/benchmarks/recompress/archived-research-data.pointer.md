# Archived recompress research data (2026-05-29)

The `recompress` module was developed as the standalone `zenjpeg-recompress`
crate before being merged into zenjpeg (2026-05-29). The full development
history and the bulky raw research sweeps are archived in block storage
(kept out of git per the >30 KB rule):

`/mnt/v/zen/zenjpeg-recompress-archive-2026-05-29/`

- `zenjpeg-recompress-repo-src.tar.gz` — the full standalone crate including
  its git/jj history (source only; `target/` excluded). 2.0 MB.
- `benchmarks/` — all 30 development sweep TSVs (lever 1–5 forced sweeps,
  15-image calibration sweeps, AQ ablations, tri-metric cross-check, etc.)
  with `SHA256SUMS.txt`. The large ones (>30 KB) are NOT in git.

The **canonical n=50 calibration data** lives separately at
`/mnt/v/zen/zenjpeg-recompress/calibration-n50-2026-05-29/` — see
`calibration-n50-2026-05-29.pointer.md` in this directory. The small
development TSVs kept in this directory are provenance for the lever work
documented in `../../docs/recompress/RECOMPRESSION_COMPENDIUM.md`.
