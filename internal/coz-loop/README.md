# JPEG profiling worker

Development-only Coz loops and a checked external-process decode worker. Build
with `just coz-build`; run lint checks with `just coz-check`.

For `just coz-worker pair CASE.txt OUTPUT.rgb`, the case file contains a JPEG
specification on the first line and a positive iteration count on the second:

```text
file:/absolute/path/input.jpg
1000
```

The worker warms up for five decodes plus 500 ms, then prints three
`elapsed_ns=` totals for fixed batches. Each decode includes allocation and
output destruction. Validation runs outside the timed batches and creates
`OUTPUT.rgb` with every decoded RGB byte and `OUTPUT.input.jpg` with the exact
input. Pair plans must check both files and record dimensions and input hashes.
Output paths must be new. The older `time` command retains its original timing
semantics; its results are separate from `pair` results.

Synthetic specifications have the form
`gen:noise:420:base:85:256x256`. These are synthetic coverage fixtures.
The historical `flat` generator varies red per pixel, so it does not imply
DC-only chroma. Constant RGB (`solid`) also does not guarantee AC-free partial
chroma blocks. The worker checks every coefficient before timing `solid` and
fails if any AC coefficient is present.

For explicit DC-only inputs, set `ZENJPEG_DC_FIXTURE_DIR` to a new directory
and run `just coz-fixtures`. The existing coefficient emitters generate 504
fixtures: 8x8, 17x33 and 64x64; q0 through q100 in steps of five; 444, 422,
420 and 440; baseline and progressive. The test zeros AC coefficients and
checks decoded ACs, actual-grid DCs, quantization tables, sampling and geometry.
It always runs these assertions, even without an output directory. Saved
fixtures carry the build commit in `MANIFEST.tsv`; keep their hashes beside
benchmark evidence. They exercise entropy decoding and are not quality data.

Use whole-image source inputs alongside these structural fixtures. Coz
attribution and exploratory timing do not establish an accepted speedup;
confirmation needs frozen plans, exact output checks and admitted resource
observations.
