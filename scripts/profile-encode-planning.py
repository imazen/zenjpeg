#!/usr/bin/env python3
"""Heaptrack zenjpeg effective-job planning against native startup.

Adapted from zencodec perf/jpeg-policy-validation/profile-preparation.py
(f73a7e0e). No pixels are read and no wall-clock claims are made.

Run through run-heavy. The preload library writes a regular raw file here, then
the standard interpreter processes it. This avoids heaptrack's hard-coded /tmp
FIFO and preserves every raw capture for reanalysis.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def peak_heap(trace):
    """Replay allocation events, not timestamp-sampled Massif export points.

    Format: https://github.com/KDE/heaptrack/blob/master/src/analyze/accumulatedtracedata.cpp
    Only interpreted format 3 is accepted. An 'a' defines a size/trace pair;
    '+' and '-' carry its zero-based index, which may be active repeatedly.
    """
    sizes, active = [], []
    live = peak = allocations = 0
    version = None
    for line in trace.splitlines():
        fields = line.split()
        if not fields:
            continue
        tag = fields[0]
        if tag == "v":
            version = int(fields[2], 16)
            if version != 3:
                raise ValueError(f"Unsupported heaptrack format {version}")
        elif tag == "a":
            if version != 3:
                raise ValueError("Missing format declaration")
            sizes.append(int(fields[1], 16))
            active.append(0)
        elif tag in ["+", "-"]:
            index = int(fields[1], 16)
            if index >= len(sizes):
                raise ValueError("Unknown allocation index")
            if tag == "+":
                active[index] += 1
                allocations += 1
                live += sizes[index]
                peak = max(peak, live)
            else:
                if active[index] == 0:
                    raise ValueError("Free without an active allocation")
                active[index] -= 1
                live -= sizes[index]
        elif tag == "A":
            raise ValueError("Attached-process traces are outside this probe")
    if version != 3 or allocations == 0:
        raise ValueError("Incomplete allocation capture")
    return {"peak_heap_bytes": peak, "allocation_calls": allocations,
            "live_bytes_at_exit": live}


def verify_peak(measured, analysis):
    """Cross-check our exact sum against heaptrack_print's rounded SI value."""
    match = re.search(r"^peak heap memory consumption: ([\d.]+)([BKMG])$", analysis, re.M)
    if not match:
        raise ValueError("Missing heaptrack_print peak")
    number, unit = match.groups()
    factor = {"B": 1, "K": 1000, "M": 1000000, "G": 1000000000}[unit]
    decimals = len(number.split(".")[1]) if "." in number else 0
    rounding = 0.5 * factor / 10**decimals
    if abs(measured - float(number) * factor) > rounding + 0.001:
        raise ValueError("Allocation events disagree with heaptrack_print peak")


def run(args, log, **kwargs):
    with log.open("wb") as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=120, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--profile", required=True, help="cargo profile the probe was built with")
    parser.add_argument("--heaptrack-lib", type=Path,
                        default=Path("/usr/lib/heaptrack/libheaptrack_preload.so"))
    parser.add_argument("--interpreter", type=Path,
                        default=Path("/usr/lib/heaptrack/libexec/heaptrack_interpret"))
    args = parser.parse_args()
    executable = args.executable.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"source_revision": args.source_revision,
              "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "profile": f"{args.profile}; heaptrack; no pixel reads or output consumption",
              "dimensions": [2048, 2048],
              "scope": "direct zenjpeg jobs retained together; whole-process heap peak, "
                       "not RSS, not a native-only total, not a bound for other sizes",
              "started": datetime.now(timezone.utc).isoformat(), "cells": []}
    (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    for configuration in ["fixed", "progressive"]:
        for colour in ["srgb", "p3-icc", "p3-descriptor"]:
            for stage in ["prepare", "plan", "begin"]:
                for count in [1, 4]:
                    name = f"{configuration}-{colour}-{stage}-{count}"
                    print(name, flush=True)
                    raw = output / f"{name}.raw"
                    env = dict(os.environ, LD_PRELOAD=str(args.heaptrack_lib.resolve()),
                               DUMP_HEAPTRACK_OUTPUT=str(raw))
                    run([str(executable), stage, str(count), configuration, colour],
                        output / f"{name}.run.log", env=env)
                    trace = output / f"{name}.heaptrack"
                    with raw.open("rb") as source, trace.open("wb") as target, \
                            (output / f"{name}.interpret.log").open("wb") as errors:
                        subprocess.run([str(args.interpreter)], stdin=source, stdout=target,
                                       stderr=errors, check=True, timeout=120)
                    massif = output / f"{name}.massif"
                    run(["heaptrack_print", "-f", str(trace), "-M", str(massif),
                         "--print-allocators", "0", "--print-temporary", "0"],
                        output / f"{name}.analysis.txt")
                    measured = peak_heap(trace.read_text())
                    verify_peak(measured["peak_heap_bytes"],
                                (output / f"{name}.analysis.txt").read_text())
                    report["cells"].append({"configuration": configuration, "colour": colour,
                                            "stage": stage, "jobs": count, **measured,
                                            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()})
                    report["updated"] = datetime.now(timezone.utc).isoformat()
                    (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    report["completed"] = datetime.now(timezone.utc).isoformat()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
