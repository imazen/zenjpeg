#!/usr/bin/env python3
"""Fit per-encoder source-quality anchors: encoder IJG-Q → median
cumulative zensim-A vs the original.

Reads a zen-metrics zensim TSV (ref_path, dist_path, zensim_gpu) where
dist filenames are `<refstem>__<encoder>__q<Q>.jpg`, and prints the
anchor arrays for pasting into `target.rs::ijg_q_to_zensim_a` plus a
provisional confidence-residual note.

Usage: fit_source_anchors.py <src_zensim.tsv>
"""

import csv
import re
import statistics
import sys
from collections import defaultdict


def main(path):
    rows = defaultdict(list)
    for r in csv.DictReader(open(path), delimiter="\t"):
        m = re.search(r"__(\w+)__q(\d+)\.jpg", r["dist_path"])
        if not m:
            continue
        rows[(m.group(1), int(m.group(2)))].append(float(r["zensim_gpu"]))

    for enc in sorted({e for (e, _) in rows}):
        qs = sorted({q for (e, q) in rows if e == enc})
        anchors = [(q, round(statistics.median(rows[(enc, q)]), 1)) for q in qs]
        print(f"// {enc}: IJG-Q → median cumulative zensim-A (n={len(rows[(enc, qs[0])])}/q)")
        body = ", ".join(f"({q}.0, {z})" for q, z in anchors)
        print(f"const {enc.upper()}: &[(f32, f32)] = &[{body}];")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
