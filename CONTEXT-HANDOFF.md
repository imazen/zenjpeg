# Context Handoff: Hybrid Trellis Investigation Complete

## Final Conclusion

**Hybrid coupling (`aq_lambda_scale`) is redundant.**

It traces the same rate-distortion curve as simply changing the quality number. There's no advantage.

## Evidence

At matched file size (~58k bytes):
| Config | BA |
|--------|-----|
| Jpegli Q85 | 1.922 |
| Hybrid+5 Q84 | 1.938 |
| Hybrid-4 Q87 | 2.031 |

- Hybrid+5 at same size = same quality as Jpegli
- Hybrid-4 at same size = **worse** quality than Jpegli

## Bug Fixed

`hybrid_config()` wasn't being applied because `HybridProgressive` preset set `trellis` first.
Fixed by clearing `trellis` when `hybrid_config(enabled=true)` is called. Commit `df94d12`.

## Recommendation

Remove or deprecate `aq_lambda_scale` coupling. Users should just adjust quality instead.

The only value in HybridProgressive vs JpegliProgressive is:
- HybridProgressive has trellis enabled (mozjpeg-style rate-distortion optimization)
- JpegliProgressive has no trellis (jpegli-compatible output)

The trellis itself provides the size/quality tradeoff, not the coupling parameter.

## Files

| File | Purpose |
|------|---------|
| `examples/coupling_vs_quality.rs` | Proves coupling = quality change |
| `examples/pareto_vs_jpegli.rs` | Comparison vs JpegliProgressive |
| `examples/pareto_*.rs` | Earlier exploration (misleading, compared trellis vs trellis) |

## Commits

- `042c52e` investigate: coupling vs quality - traces same curve
- `df94d12` fix: hybrid_config now clears trellis
- `b94692a` docs: update handoff
- `065abb9` investigate: positive coupling Pareto analysis
