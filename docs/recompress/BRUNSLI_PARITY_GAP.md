# Brunsli Parity & JPEG XL JBRD Lossless-Recompression Gap Analysis

**Date:** 2026-06-05
**Provenance:** Synthesised from four parallel source audits (brunsli `c/` at `github.com/google/brunsli`; `jxl-encoder/src/jpeg/`; `zenjxl-decoder/src/jpeg/`; `zenjpeg/src/decode/`) plus the Google "Journey to JPEG XL" blog (2026-06). All `file:line` anchors are from a read at this date — verify before acting.

---

## TL;DR

**Brunsli** (2015, Google) losslessly *repacks an existing JPEG into a smaller footprint and reconstructs it byte-for-byte.* It was absorbed into JPEG XL as the **JBRD** ("JPEG Bitstream Reconstruction Data") mode. Our equivalent is **not one crate** — it is a three-stage pipeline:

```
   zenjpeg                 jxl-encoder              libjxl djxl  (external C++)
   decoder        ──►      JPEG→JXL        ──►      OR
   (coeffs + JBRD          transcode               zenjxl-decoder (pure Rust)
    metadata source)       (src/jpeg/)             JXL→JPEG reconstruct
```

A JPEG round-trips byte-exactly **only if every stage on the chosen path supports its features** — the pipeline is bounded by its *weakest* stage.

**Where we stand:**

- **Encode side (jxl-encoder → libjxl `djxl`) is already at libjxl/cjxl parity** for the dominant web case: baseline + progressive, YCbCr/grayscale, 8-bit, the four standard subsampling modes, restart markers, multi-table Huffman/quant, full metadata + padding-bit reconstruction. The committed mixed bench reports **200/200 byte-identical** via `djxl --reconstruct_jpeg` (jxl-encoder CLAUDE.md, "EX-J31", commit `fb57425a`).
- **Three things keep us short of "match brunsli across a large corpus":**
  1. **Verification is currently dark.** The `/mnt/v/output/jxl-encoder/jpeg-reencoding/` corpus is **empty (0 files)** on this machine and **`djxl`/`cjxl` are not installed**, so the broad byte-exact tests can't run and the hermetic one skips its assertion. The correctness gate depends entirely on an external binary that isn't here.
  2. **The pure-Rust reconstructor (`zenjxl-decoder`) is baseline-only** and *silently* produces wrong bytes for progressive/SOF1/CMYK rather than erroring. So a pure-Rust (no-libjxl) round-trip covers far less than the encode side.
  3. **A few real defects** inside the "supported" set — most importantly **silent mis-encoding of chroma sampling factors > 2** in jxl-encoder (should refuse, doesn't).
- **Two brunsli capabilities are bounded by the JXL-JBRD format itself** (not just our code): **4-component CMYK/YCCK** and **arbitrary sampling factors (4:1:1 etc.)**. `cjxl` refuses both too. Matching brunsli here is a format-boundary decision, not a bug-fix.

The single highest-leverage deliverable is **a hermetic, always-on corpus conformance harness that round-trips through the pure-Rust reconstructor** — that is what "ensure we match brunsli across a large corpus" actually requires, and it is the forcing function for every gap below.

---

## 1. Brunsli's capability surface (the reference spec)

Brunsli parses a JPEG into a `JPEGData` model, re-entropy-codes the quantised DCT coefficients with a neighbour-predicting **context model over a clustered rANS + binary-arithmetic stream**, Brotli-compresses the metadata, and hoards every non-pixel quirk so reconstruction is byte-exact.

**Accepts (full recompression):** 8-bit, Huffman-coded **baseline (SOF0) / extended-sequential (SOF1) / progressive (SOF2)**; 1–4 components (grayscale / YCbCr / **CMYK / YCCK**); any chroma subsampling and **H/V sampling factors 1–15** (each must divide the max); 8- or 16-bit quant tables; restart markers / DRI; arbitrary multi-scan progressive scripts (Ss/Se/Ah/Al).

**Byte-exact quirk set it preserves** (`JPEGData`, `c/common/jpeg_data.h:213-252`): end-of-scan **padding bits** + `has_zero_padding_bit`; `0xFF`/`0x00` stuffing; **inter-marker fill bytes**; **trailing `tail_data`** after EOI; exact **marker order**; **APPn/COM bytes verbatim**; DHT/DQT segment grouping (`is_last`); **extra-zero-runs**; **double-EOB reset points**; over-wide quant precision.

**Refuses** (errors out unless caller invokes explicit raw **bypass**): arithmetic (SOF9-11), lossless (SOF3), hierarchical/differential (SOF5-7), DNL, **12-bit**, > 4 components, > 65535 px, > 2²¹ blocks/component, structurally invalid scans.

**Brunsli's corpus-wide robustness secret:** when full recompression can't run, it **bypass-stores the original JPEG bytes verbatim** inside a brunsli container (`kFallbackVersion = 1`, `c/enc/brunsli_encode.cc:1551`). So *every* JPEG round-trips; only the compressible subset gets smaller. Our pipeline currently **errors** (or worse, corrupts) on the same inputs — we have no fallback.

---

## 2. Where each stage stands

| Stage | Crate / path | Verdict |
|---|---|---|
| **Source** | `zenjpeg` `src/decode/` + `JbrdMetadata` (`decode/image.rs:544`) | **Most complete stage.** Decodes baseline, **progressive**, **arithmetic** (SOF9/10), **CMYK 4-comp**, restart, 16-bit quant, all four subsamplings. Refuses 12-bit (`parser/mod.rs:675`), lossless/hierarchical, DNL, > 4 comp. JBRD metadata (`scans`, `reset_points`, `extra_zero_runs`, `padding_bits`, `has_zero_padding_bit`) is emitted **only for progressive**; baseline gets coeffs + padding bits but **empty `scans`**. Does **not** capture inter-marker/tail/marker-order in JBRD (jxl-encoder re-scans markers itself). |
| **Encode** | `jxl-encoder` `src/jpeg/{parse,jbrd,encode,data}.rs` | **Solid on the web case, libjxl-parity.** Baseline + progressive, gray/YCbCr/RGB, 8-bit, four standard subsamplings, restart, padding bits. **Refuses** arithmetic/lossless/12-bit (clean error) and **CMYK** (`encode.rs:140-160`, matches cjxl). **Defects:** silent mis-encode of sampling > 2 (`compute_jpeg_upsampling`, `encode.rs:1541`); `last_needed_pass` hard-coded 0 (`jbrd.rs:142`); `num_intermarkers` always 0 (inter-marker fill lost). |
| **Reconstruct (Rust)** | `zenjxl-decoder` `src/jpeg/{jbrd,writer,data}.rs` | **Baseline-only.** Verified: 8-bit YCbCr baseline at 4:4:4 / 4:2:0. **Silently wrong** for progressive (writes a *baseline* entropy stream, ignores Ss/Se/Ah/Al — `writer.rs:303-444`), SOF1 (drops the SOF marker), arithmetic, DNL. CMYK structurally impossible (`[Vec<i16>;3]`). **Reconstruction errors are swallowed** by `if let Ok(...)` (`frame/.../sections.rs:392`) → returns wrong JPEG, not `None`. |
| **Reconstruct (C++)** | external libjxl `djxl --reconstruct_jpeg` | The oracle the encode-side tests actually use. Handles everything jxl-encoder emits (incl. progressive). **Not present on this machine.** |

---

## 3. Combined cross-stage capability matrix

Round-trip status is the **AND** across the path. "RT (djxl)" = JPEG→JXL→JPEG via libjxl; "RT (Rust)" = via zenjxl-decoder.

| JPEG feature | brunsli | zenjpeg decode | jxl-encode | djxl recon | zenjxl recon | **RT (djxl)** | **RT (Rust)** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SOF0 baseline 8-bit | ✓ | ✓ | ✓ | ✓ | ✓ | **✓ tested** | **✓ tested** |
| SOF1 extended-seq 8-bit | ✓ | ✓ | ✓ | ✓ | ✗ drops SOF | ✓* | **✗ silent** |
| **SOF2 progressive** | ✓ | ✓ +JBRD | ◐ (`lnp`=0) | ✓ | ✗ baseline stream | **✓** (200-file bench) | **✗ silent** |
| Arithmetic SOF9-11 | ✗ refuse | ✓ decode, no JBRD | ✗ refuse | ✗ | ✗ | ✗ (parity) | ✗ (parity) |
| Lossless/hierarchical | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (parity) | ✗ (parity) |
| 12-bit precision | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (parity) | ✗ (parity) |
| 1-comp grayscale | ✓ | ✓ | ✓ | ✓ | ◐ untested | ✓ | ◐ untested |
| 3-comp YCbCr | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **✓** |
| 3-comp RGB (kNone) | ✓ | ✓ | ✓ untested | ✓ | ◐ untested | ◐ | ◐ |
| **4-comp CMYK/YCCK** | **✓** | ✓ decode | ✗ refuse | ✗ | ✗ impossible | **✗ GAP** | **✗ GAP** |
| 4:4:4 | ✓ | ✓ | ✓ (CfL) | ✓ | ✓ | **✓** | **✓** |
| 4:2:0 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **✓** |
| 4:2:2 | ✓ | ✓ | ✓ | ✓ | ◐ untested | ✓ | ◐ |
| 4:4:0 | ✓ | ✓ | ✓ | ✓ | ◐ untested | ✓ | ◐ |
| **Arbitrary H/V > 2** (4:1:1…) | ✓ (1-15) | ◐ coeff-ok | **✗ SILENT CORRUPT** | — | ✗ impossible | **✗ DEFECT** | **✗** |
| Restart / DRI | ✓ | ✓ | ✓ | ✓ | ✓ (caveat) | ✓ | ◐ lightly |
| Multi / non-interleaved scans | ✓ | ✓ | ◐ | ✓ | ◐ untested | ◐ | ◐ |
| 8/16-bit quant tables | ✓ | ✓ | ✓ | ✓ | ✓ (16-bit untested) | ✓ | ◐ |
| Multi / non-optimal Huffman | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **✓** |
| DNL marker | ✗ refuse | ✗ | ✗ | ✗ | ✗ | ✗ (parity) | ✗ (parity) |
| Padding bits / zero-pad | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **✓** |
| Inter-marker fill bytes | ✓ | ✗ | ✗ (`num_intermarkers`=0) | ✓ | ✓ replay | **✗ A4** | ✗ |
| Trailing tail after EOI | ✓ | ✗ (jxl re-scans) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Marker order | ✓ | ✗ (jxl re-scans) | ✓ | ✓ | ✓ | ✓ | ✓ |
| APP/COM verbatim | ✓ | ◐ (extras path) | ✓ | ✓ | ✓ untested | ✓ | ◐ |
| extra-zero-runs / reset-points | ✓ | ✓ progressive | ✓ | ✓ | ◐ baseline | ✓ | ◐ |
| `last_needed_pass` (multi-scan) | ✓ | ✗ not computed | ✗ =0 | ✓ derives | ✗ ignored | ◐ latent | ✗ |

✓ = works · ◐ = partial/untested · ✗ = absent/broken · ✓* = works but no in-repo fixture · "parity" = we refuse exactly as brunsli does (acceptable).

---

## 4. Gaps, prioritised — what to add

### P0 — Correctness: turn silent corruption into honest refusal (cheap, do first)

Per house rule "the user's pixels are sacred": a path that silently emits wrong bytes is a shipping bug, worse than a clean error.

1. **jxl-encoder: reject chroma sampling factors > 2.** `compute_jpeg_upsampling` (`jxl-encoder/src/jpeg/encode.rs:1541-1558`) reduces `(h,v)` to a 2-bit mode via `(h>0, v>0)` booleans — `h=4` collapses to the same mode as `h=2`. No guard rejects `h/v > 2`, so a 4:1:1 JPEG is silently mis-encoded. **Add a guard in `parse.rs`/`encode.rs` that errors `Unsupported("sampling factor {h}x{v} not representable in JXL")`.** ~10 lines. The JXL `YCbCrChromaSubsampling` field only carries the 4 standard modes, so refusing is correct.
2. **zenjxl-decoder: stop swallowing reconstruction errors.** `frame.jpeg_reconstruct()` is consumed by `if let Ok(...)` (`.../codestream_parser/sections.rs:392`, "non-fatal"). A reconstruction that breaks should surface as a hard error / `None`, never as wrong bytes silently returned by `take_jpeg_reconstruction()`. **Propagate the error.**
3. **zenjxl-decoder writer: reject unsupported markers instead of dropping them.** The marker-replay `_ =>` no-op arm (`writer.rs:102-104`) silently swallows SOF1/SOF2/SOF9-11/DNL/DAC, corrupting both content and marker order. **Error on any marker the writer can't faithfully emit.** Pair with a `is_progressive` guard at the top of `write_scan_data` until progressive is implemented (P1).

### P1 — Verification: the "large corpus" requirement

4. **Build a hermetic corpus conformance harness.** This is the heart of "match brunsli across a large corpus" and currently the biggest hole (corpus dir empty, djxl absent). Brunsli's own value was proven by round-tripping a vast JPEG corpus; we need the same, self-contained:
   - **Corpus:** vendor/generate a diverse, *committed* (or `codec-corpus`-managed, per house rule against ad-hoc file-existence skips) set spanning every matrix row: baseline + **progressive** (incl. successive-approximation), gray/YCbCr/RGB/**CMYK**, all four subsamplings + an exotic-sampling sample, restart-interval, 16-bit DQT, metadata-heavy (ICC/EXIF/XMP), inter-marker fill, tail data, multi-scan. Pull real-world JPEGs from the existing corpora plus synthesised edge cases. Mind the house rule: low-quality/aggressive JPEGs (q5–q40) as densely as high-q.
   - **Oracle:** round-trip JPEG→JXL→JPEG through **`zenjxl-decoder` (pure Rust)**, not only `djxl`. Assert byte-exact. Keep an *optional* `djxl` arm (gated through the justfile/CI, not a runtime `println!` skip) as a cross-check when present.
   - **Reporting:** bucket every failure by detected feature (SOF type, components, subsampling, progressive, restart, metadata) so a run says *"CMYK: 0/40, progressive: 0/120 (Rust) / 120/120 (djxl)"* — the live version of the matrix above. Persist results to `benchmarks/brunsli_parity_<date>.tsv` with commit hash (house rule).
   - **Wire it into CI** so the gate stops being dark. The hermetic 4:4:4 case in `tests/jpeg_transcode_roundtrip.rs` is the seed to generalise.
5. **Pure-Rust progressive reconstruction in `zenjxl-decoder`.** The dominant real gap for a no-libjxl round-trip — a large share of the web JPEG corpus is progressive. `write_scan_data` (`writer.rs:303-444`) must implement spectral-selection band restriction + successive-approximation bit-plane coding (DC first/refine, AC first/refine, EOBRUN), mirroring zenjpeg's existing **progressive decoder** (`zenjpeg/src/decode/parser/progressive.rs`) in reverse. Consume `last_needed_pass` and the per-scan `reset_points`/`extra_zero_runs` already in JBRD.
6. **Fix `last_needed_pass` end-to-end.** Compute it (zenjpeg, or derive in jxl-encoder from the scan list), write the real value (`jxl-encoder/src/jpeg/jbrd.rs:142`, currently `0`), and consume it (`zenjxl-decoder` parses it at `jbrd.rs:200` then ignores it). Latent multi-scan-progressive correctness bug today; a hard blocker for #5.

### P2 — Feature parity with brunsli

7. **CMYK / YCCK 4-component.** ~1.8% of the codec-corpus (jxl-encoder CLAUDE.md backlog #6). Requires: jxl-encoder to stop refusing (`encode.rs:140-160`) **and** zenjxl-decoder to carry a 4th channel (`jpeg_coeffs: [Vec<i16>;3]` → 4). **Check the JXL spec first** — this may need a JXL `ExtraChannel` representation; if JXL-JBRD genuinely can't carry 4 DCT channels, this drops to §5 (fallback). zenjpeg already *decodes* CMYK, so the source is ready.
8. **Inter-marker fill bytes (A4).** jxl-encoder's marker scanner never inserts the synthetic `0xFF` `marker_order` entries brunsli uses, so `num_intermarkers` is always 0 and fill bytes between markers are lost (`jbrd.rs:187`). Near-zero empirical impact but a true byte-exactness gap. Mirror brunsli's `marker_order` + `inter_marker_data` model in the scanner.
9. **Baseline `reset_points` / `extra_zero_runs`.** zenjpeg emits these only for progressive (`progressive.rs:412` is the sole push site); baseline JPEGs *with restart intervals* also have entropy-segment boundaries. Verify baseline-with-restart round-trips and, if not, emit the baseline JBRD scan signals from zenjpeg.
10. **Lock the untested-but-plausible rows** with committed fixtures + hard asserts: grayscale, RGB(kNone), 4:2:2, 4:4:0, 16-bit DQT, restart/DRI as a dedicated case, non-interleaved scans, APP/COM verbatim re-injection (esp. ICC/EXIF/XMP stitched from container boxes). The harness (#4) does this empirically; fixtures pin it in unit tests.

### P3 — Corpus-wide robustness (brunsli's real lesson)

11. **Add a brunsli-style fallback contract.** Brunsli guarantees *every* JPEG round-trips by bypass-storing the raw bytes when recompression can't run. We should offer the same at the API boundary: when jxl-encoder refuses (arithmetic, 12-bit, CMYK if unsupported, exotic sampling, anything that fails the round-trip self-check), the caller gets a clear signal to **store the original JPEG unchanged** (or wrap it). This converts "errors on N% of the corpus" into "100% round-trip, compresses M%" — which is what "match brunsli across a large corpus" ultimately means. Consider a **transcode self-verification** (encode → reconstruct → compare in-process) before committing to the transcoded output, falling back to passthrough on any mismatch. This is the cheapest way to *guarantee* corpus-wide byte-exactness even while features 5–10 are still landing.

---

## 5. Honest ceiling: JXL-JBRD format boundaries

Two brunsli capabilities are narrower in the **JXL-JBRD container itself**, not merely in our code:

- **Arbitrary sampling factors > 2** (4:1:1, and 3/5…15): JXL's `jpeg_upsampling` is a 2-bit field carrying only the four standard modes. `cjxl` refuses these. **There is no in-format fix** — the only honest options are clean refusal (P0-1) or fallback (P3-11). Do **not** pretend to support them.
- **4-component CMYK/YCCK:** `cjxl` also refuses. Whether JXL-JBRD can carry a 4th DCT channel needs a spec check (P2-7); if not, this too is a fallback case.

So **true brunsli parity = libjxl/cjxl parity + a fallback for the format-bounded residue.** For the *compressible* web corpus (baseline + progressive, YCbCr/gray, 8-bit, standard subsampling — the overwhelming majority), full parity is achievable and the encode side already has it; the work is making the **pure-Rust reconstructor** and the **verification harness** catch up.

---

## 6. Recommended sequencing

1. **P0 (1–3):** stop the silent corruption — a day's work, immediately removes the "wrong pixels" risk. (Honest refusal > false success.)
2. **P1-4:** stand up the hermetic corpus harness with the pure-Rust oracle + CI wiring. Everything else is measured against it. This also re-lights the currently-dark gate.
3. **P3-11:** add transcode self-verification + passthrough fallback → instant corpus-wide 100% round-trip guarantee while features land.
4. **P1-5/6:** pure-Rust progressive reconstruction + `last_needed_pass` — the biggest single capability win for a no-libjxl pipeline.
5. **P2 (7–10):** CMYK (spec-permitting), inter-marker, baseline reset signals, fixture lock-down.

---

## Appendix — key file:line anchors

**brunsli (`/home/lilith/work/_brunsli-read/c/`):** data model `common/jpeg_data.h:213-252`; parser `enc/jpeg_data_reader.cc:1021` (marker switch `:1057-1124`, `UNSUPPORTED_MARKER :1118`); writer `dec/jpeg_data_writer.cc:967`; bypass `enc/brunsli_encode.cc:1551`; section tags `common/constants.h:59-67`; context model `common/context.h`.

**zenjpeg:** `JbrdMetadata` `src/decode/image.rs:544`, `JbrdScanInfo` `:585`; coeff+JBRD entry `decode_coefficients_with_jbrd_metadata` `src/decode/mod.rs:2541`; SOF dispatch `src/decode/parser/markers.rs:30-64`; 12-bit reject `src/decode/parser/mod.rs:675`; progressive decoder `src/decode/parser/progressive.rs`; JBRD scan push `progressive.rs:412`.

**jxl-encoder:** transcode entry `src/api.rs:3284` (`encode_jpeg_transcode`); parser `src/jpeg/parse.rs:32`; sampling-mode defect `src/jpeg/encode.rs:1541`; CMYK refuse `encode.rs:140-160`; `last_needed_pass`=0 `src/jpeg/jbrd.rs:142`; `num_intermarkers` `jbrd.rs:187`; hermetic roundtrip test `tests/jpeg_transcode_roundtrip.rs`; non-hermetic matrix `tests/jpeg_reencoding.rs` (fixtures in empty `/mnt/v/output/jxl-encoder/jpeg-reencoding/`).

**zenjxl-decoder:** reconstruct glue `src/frame/mod.rs:382` (`jpeg_reconstruct`); swallowed-error site `.../codestream_parser/sections.rs:392`; writer marker loop `src/jpeg/writer.rs:48-106`; baseline-only scan writer `writer.rs:303-444`; `last_needed_pass` parsed-then-ignored `src/jpeg/jbrd.rs:200`; tests `src/tests/jpeg_reconstruction.rs` (6 baseline fixtures, committed).
