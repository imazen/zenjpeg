# Stable dependency cohort, 2026-11-07

This separate candidate preserves every implementation and test file from
reviewed JPEG `c3849daea32df2575f7d5b4dfc69a1d573f57e9a`. No original review
ref or main was overwritten. Dependency manifests, the root lock and this
receipt change; pixel assertions and numerical thresholds do not.

The resolved graph selects registry archmage/archmage-macros/magetypes
0.9.31 and moxcms 0.9.2 for the current CMS. It retains one core `1fb997bb`
and pixel `fde59b96` family and frozen linear-srgb `2969ee84`. Legacy
archmage 0.4.0 and moxcms 0.8.1 remain isolated in older dependency families;
this does not claim that incompatible major families were unified.

## Unchanged gates

Full library/test/bench strict Clippy passes with the original feature list.
The pinned instrumented C++ oracle is enabled (source `631732a5`).

| Release gate | Passed | Failed | Existing ignored |
|---|---:|---:|---:|
| default | 2469 | 0 | 112 |
| user | 3037 | 0 | 119 |
| ffi | 2866 | 0 | 170 |
| expert | 2869 | 0 | 119 |
| recompress | 35 | 0 | 0 |

These groups overlap and must not be summed as unique tests. Recompress
uses its pre-existing explicitly selected module/target recipe.
The complete independent colorspace oracle target passes 2/2 controls with
zero ignored/filtered tests, retaining its exact/one-code thresholds across
Pillow/libjpeg-turbo, IJG libjpeg 9 and jpegli.

The initial oracle attempt fails because PATH selected libjpeg-turbo's djpeg
instead of the required IJG libjpeg 9. The retained failure is a tool-selection
receipt, not a pixel mismatch. Selecting the previously built pinned IJG 9
binary makes the unchanged target pass. No source or assertion changed.

The five-group runtime wrapper plus failed first oracle attempt: 914 seconds,
peak RSS 2.04 GiB, minimum available 13773 MiB, peak load 4.13. Corrected
oracle rerun: 2 seconds, peak RSS 0.08 GiB. These describe validation jobs,
not JPEG throughput or per-image memory. Full strict Clippy: 29 seconds,
peak RSS 0.89 GiB. No speed comparison is claimed.

## Dependency source map

Every capped dependency is a manifest-only descendant of its frozen source:

| Component | Original | Stable-compatible source |
|---|---|---|
| zenblend | `c61d051bb0a1` | `d477b29c4ffb` |
| zentone | `a8e3394cc0a2` | `2598054c140b` |
| zenflate | `17233d1edd61` | `a2c7d540fc8f` |
| zenresize | `4ad64b3c0a29` | `5c5fe0d53b82` |
| garb | `605276d717ea` | `8f9aa10d09f8` |
| fast-ssim2 | `16237fe54781` | `0fac45a5f22a` |
| butteraugli | `d52a668d1903` | `948840350f40` |
| zenanalyze | `87f69b14993a` | `0e2c2d8f6cd8` |
| zensimx | `73ae3948f9f0` | `824a4f45a36e` |
| PNG facade | `6b740345` | `d32e82e529e2` |
| HDR current | `30249993` | `5a3b158c35cd` |
| HDR legacy oracle | `641ab72e` | `7516b501aedb` |

The full HDR workspace review is a distinct graph rooted at `6df45a2e`, with
JPEG `162a1f35`; do not replace it with this graph. Its genuine no_std/WASM
float-method blocker remains visible and is not qualified by these std gates.
Platform, standalone fuzz, diagnostics and API-snapshot gates have not been
rerun for this dependency candidate. No merge or publication is requested.
