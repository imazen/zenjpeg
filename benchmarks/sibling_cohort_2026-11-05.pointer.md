# Shared sibling cohort: blocked test build (2026-11-05)

Base: `762fa7128c2124b42cfa6e23b30024a688f5b7a3`.
Proposed root/fuzz cohort: core `1816b4d082234c04c3805411b437b8396040f522`,
pixels `fa836abf72afe9826e866dc25712b7c800323962`,
PNG `02a7f0df5f72dd79b0afa58d66ced7cdd8395e31`,
UltraHDR `85cc309c07526341e6fa8b7e4d6130aaf64c8fea`.
The UltraHDR old/new source diff contains manifests, lock and changelog changes only.

Baseline formatting and strict library clippy pass:
`rc=0 12s | peak-RSS 0.54GiB | min-avail 8327MiB | peak-load 1.92`.
The new root/fuzz graphs resolve, but `just sibling-cohort-check` fails while
building the default tests, before any new-cohort test result:
`rc=101 189s | peak-RSS 0.73GiB | min-avail 7643MiB | peak-load 3.45`.
Two E0308 errors are at zensim-regress `pixel_ops/mod.rs:228,577`:
its helpers accept zenpixels 0.2.16 buffers, while current PNG returns 0.3.1.
Both zensim and zensim-regress are deliberately pinned to `85577588` in JPEG;
zensimx main also retains the exact 0.2.16 workspace hold.

This change is locally preserved and not pushed. No expectations, skips,
metric profiles or native codec algorithms changed. The optional C++ FFI
parity suite is outside the chosen feature gates and its submodule is absent.

Full logs/hashes, checked on dev:
`/mnt/v/output/zensim-siblings-20261105/zenjpeg/`.

The approved preparatory graph subsequently landed at zensimx
`02243a4ba84ed71611f2cf445553eb4d41c2fbac`, with unchanged frozen4678-pair
and split gates. JPEG's two paired dev-tool pins now name that commit.
Default library/integration targets pass; the subsequent production feature
build found nine E0308 errors from a second pixel source:
`rc=101 437s | peak-RSS 1.35GiB | min-avail 6544MiB | peak-load 3.91`.
A forced pixel update alone confirms published zenresize0.3.1's ^0.2.10 hold
(`rc=101 0s | peak-RSS 0.04GiB | min-avail 8551MiB | peak-load 1.95`).
The existing git resize revisione3975fb9 admits0.3; a root patch plus explicit
lock refresh removes0.2.16. Cargo's production tree now has one0.3.1/fa836abf
source shared across JPEG, UltraHDR, resize, PNG and core. Full requalification
is in progress. No bridge conversion, signature or expectation was changed.
