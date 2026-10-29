#!/usr/bin/env python3
"""Reject deferred analysis crates in zenjpeg's normal/build dependency graph."""

import json
import subprocess


metadata = json.loads(subprocess.check_output([
    "cargo", "metadata", "--locked", "--all-features", "--format-version", "1",
], text=True))
packages = {p["id"]: p for p in metadata["packages"]}
nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
root = next(p["id"] for p in metadata["packages"]
            if p["name"] == "zenjpeg" and p["id"] in metadata["workspace_members"])
pending = [root]
visited = set()
forbidden = {"zenanalyze", "zenanalyze-api", "zenpredict", "zenpicker", "zensim"}
while pending:
    current = pending.pop()
    if current in visited:
        continue
    visited.add(current)
    name = packages[current]["name"]
    if name in forbidden:
        raise SystemExit(f"Deferred release dependency is reachable: {current}")
    for dep in nodes[current]["deps"]:
        if any(kind["kind"] in (None, "build") for kind in dep["dep_kinds"]):
            pending.append(dep["pkg"])
print("zenjpeg normal/build dependency graph excludes deferred analysis crates (all features)")
