# Temporary analysis integration removal

Proposal recorded 2026-10-26 against zenjpeg `310c53fa`. Implementation
awaits explicit approval because it removes public API and Cargo features.
The aim is a smaller release dependency graph while preserving ordinary
encoding, decoding, JPEG adaptive quantization, trellis, lossless operations
and basic recompression.

## Concrete scope

- Remove `EncoderConfig::adaptive` and `adaptive_with`, `AdaptiveOptions`
  and `RestartMarkers`, and the content-analysis implementation behind them.
- Remove the precomputed-analysis hook and its stored feature data.
- Remove the learned picker, its direct zenpredict dependencies, and the
  zenanalyze/zenanalyze-api dependencies and root patches used by these paths.
- Remove `target-zq`, `__picker-research`, `__zensim-research` and the associated
  Zq targeting API if the approved choice is a complete temporary removal.
- A complete zenpredict cut also requires removing zensim-backed targeting
  and recompress-IQA from the published library's dependency graph. Preserve
  research/calibration tools separately where they remain useful; inspect
  workspace members, examples and fuzz manifests as well as the library.

Zenpicker is not currently a direct zenjpeg dependency. Disabling default
features alone is insufficient: target-zq is already off by default, while
zenanalyze is unconditional. Optional dependency declarations also need valid
publication metadata.

The alternative is to remove content analysis/learned pickers while retaining
metric targeting. That retains zenpredict through zensim and does not meet a
complete dependency removal goal.

## Validation for an implementation

Review the actual public API/feature diff and affected consumers. Preserve
existing codec regression expectations. Check the published package's normal
dependency graph, minimal/default builds, the remaining feature matrix,
examples and fuzz manifests. Run all-target and doc tests on the release
candidate. Confirm byte identity for unchanged encoder configurations; do not
substitute fallback stubs for removed targeting contracts.

Reintroducing the integration later requires a separate API review, current
feature-definition qualification and model/codec measurements. Existing Git
history and the preserved research artifacts provide the recovery source.

## V4 graph use relevant to this decision

The refreshed graph branch remains `84800dd1`. The ecosystem census found no
explicit graph-only predictor pin in dependent repositories. An additional
unknown historical predictor pin was checked against both main and the graph
branch; it is a separate older divergence, not a graph consumer. Fresh checks
of zensim and JXL main still show the v3 `f0903489` pin.

The graph branch's own demo, reference/parity evaluators and fuzz targets do
exercise it. Its gate record also documents a zensim run using an explicit
patch configuration. Those are experimental qualification uses, not evidence
of a mainline v4 deployment. The inspected codec/metric model artifacts are
v3. Main's v3 SmoothHinge activation byte 3 also requires reconciliation with
the graph branch's Exp byte 3 and its older v3 parser before that branch lands.

Sources: [graph design](https://github.com/imazen/zenanalyze/blob/84800dd1feb2/zenpredict/docs/ZNPR_V4_GRAPH.md),
[graph gate record](https://github.com/imazen/zenanalyze/blob/84800dd1feb2/benchmarks/znpr_v4_graph_gates_2026-10-09.md),
[ecosystem audit](https://github.com/imazen/zenanalyze/blob/208ee7ec/docs/ECOSYSTEM_USAGE_2026-10-25.md).
