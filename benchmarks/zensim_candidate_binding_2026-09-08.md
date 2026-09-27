# JPEG complete candidate binding and AQ repairs — September 8, 2026

The recovered JPEG Zq experiment now serves complete candidate scalar scores
and current spatial maps through Rust `BakeScorer`. The first real-byte control
also found and repaired two old AQ defects: a unit scale clamped legitimate
strengths to 0.20, and the last image strip never reached the controller.
These repairs make neutral byte-identical and cover all partial edge blocks.
A fractional-q clamp panic near q=100 is also fixed.

The private loop retains its existing global correction and allocation policy.
Research feature `__zensim-research` and restored `zq_rd_probe` select an exact
bake, explicit seed and scalar/neutral/active mode. Each encode call owns its
scorer, cached source and scratch; no global profile/gradient cache is imported
from the July 18 branch. Active maps use absolute integrated score contribution
per 8x8 block. Unsupported spatial terms, corruption gates, unsupported pixel
layouts and legacy peak bounds are explicitly refused. Named paths keep their
existing public interfaces, including linear-f32, color and subsampling support.

## Frozen engagement screen

One canonical training family, imazen-26 origin 2010, original 205×256 sRGB8
variant, exact D by-ID artifact, formula revision 1, 444, jpegli q=80. The target
is the initial scalar score minus five, with zero over-target tolerance and
**two correction passes, potentially three complete encodes**. This deliberately
forces the existing claw-back branch. It is not train-calibrated realistic
1/2/3-shot targeting. No validation or terminal family is used.

| Final output | D score | Bytes | SSIMULACRA2 | Butteraugli pnorm3 |
|---|---:|---:|---:|---:|
| Scalar / neutral | 77.438789 | 16,864 | 74.922542 | 1.133673 |
| Active / active repeat | 77.314751 | 16,795 | 74.804280 | 1.139087 |

Active saves 69 bytes (0.409%) and lowers quality under all three judges.
This proves engagement and a repeatable tradeoff, **not better rate-distortion**.
The requested score is 72.438789, so the existing fixed-seed claw-back does not
reach a tight target band. `targets_met` is the existing floor-only flag; it
must not be reported as a ±tolerance hit.

All scalar and neutral passes equal the initial JPEG and decoded pixels exactly.
Active repeats every JPEG, decoded pixel, map and scale field exactly. The two
active corrections consume non-neutral factors on 208 then 217 blocks, actually
changing 186 then 195 AQ strengths. All 832 blocks, including the partial right
edge and final strip, reach the callback. Current maps change after changed
pixels. Every returned JPEG is decoded and scored again through `BakeScorer`
with agreement within 1e-5. All 13 per-pass decoded RGB images have finite,
source-bound SSIMULACRA2 and Butteraugli results from the pinned CPU judge.

## Preserved failure and regression checks

Before repair, neutral corrections changed 756/806 visited strengths despite
all factors being one, and grew intermediate JPEGs from 16,864 to 18,737 bytes.
The final 26 blocks were skipped. The canonical AQ conversion is
`max(0, 0.6 / quant_field - 1)`, which has no 0.20 upper bound. The old clamp
was therefore changing neutral behavior and suppressing meaningful differences.
The initial output remained selected, masking the defect in final-only reports.
Raw faulty traces and their original binary remain beside the repaired results.

The newer scorer exposed an existing global-q panic when `current_q + 1 > 100`;
clamping that minimum to 100 fixes it. One old strict-bound test assumed its
fixed 0.0002 total limit was impossible, but observed 0.0001261128 is inside it.
The replacement test measures the identical permissive search, then requires
strict finalization to reject a limit below its actual positive peak. No
product limit or model gate is weakened.

Validation passed: all 18 original Zq tests before changes; all 18 after changes
both with and without the research feature; 22 AQ tests including full-range
neutral identity and final-flush callback coverage; exact example Clippy and
release build; default Clippy; regenerated API snapshot check; scoped formatting
and diff checks. The snapshot changes only the excluded-feature header, with no
new public Rust signatures. Ten CLI rejection controls cover absent/nonfinite/
out-of-range seeds, bad mode, profile alias, alpha, legacy peak bound, wrong
formula, unsupported B spatial terms and nonfinite target. No CI was awaited.

## Cost, reproduction and remaining work

The final screen costs 13 full encodes and ordinary decode/score comparisons,
nine of which include map evaluation, plus five independent terminal decodes
and scores. Four non-neutral maps are consumed across active and its repeat;
final unused maps still count as evaluations. Independent judging adds 26
comparisons. Prototype/repaired repetitions and rejection controls are additional
costs. Trace file I/O is included in loop timing; no speedup or per-arm memory
claim is made from this one-image screen.

[Result JSON](zensim_candidate_binding_2026-09-08.json) pins the models, source,
binaries, commands and all intermediate results. Source declaration and repair
chronology: [registration](../docs/zensim-candidate-binding-2026-09-08.md).
Artifacts: `/mnt/v/output/zensim/jpeg-candidate-binding-2026-09-08/`, with final
results in `final/`; `run_screen.py` and `check_screen.py` retain orchestration.
Use fresh output/trace paths. Candidate/scorer dependencies are pinned together;
the pre-existing picker path dependency additionally uses the recorded
zenanalyze sibling revision. No local patch supplies candidate scoring.

Build: `cargo build --locked --release -p zenjpeg --features __zensim-research
--example zq_rd_probe`. Set `ZENSIM_FORMULA_REV=1`, `RAYON_NUM_THREADS=8`,
`ZENJPEG_ZQ_BAKE=<exact-file>`, `ZENJPEG_ZQ_SEED_Q=80`,
`ZENJPEG_ZQ_SPATIAL=scalar|neutral|active`, optionally
`ZENJPEG_ZQ_TRACE_DIR=<fresh-directory>`. Run `zq_rd_probe --image <source.png>
--target <score> --corrections 2 --out-dir <fresh-directory>`.

Shippable-model qualification, realistic train-calibrated JPEG targeting,
matched-quality spatial gains and HDR/alpha candidate targeting remain
incomplete. The historical B bucket anchors cannot seed a new candidate. Extend
the shared Rust native targeting owner for the broader experiment, and reuse
this corrected AQ callback and complete measurement binding.
