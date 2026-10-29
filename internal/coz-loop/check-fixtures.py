#!/usr/bin/env python3
"""Read every explicit fixture with ImageMagick; preserve exact provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.jpg")):
    result = subprocess.run(
        ["magick", "-regard-warnings", str(path), "-format", "%w %h", "info:"],
        capture_output=True, text=True, check=True,
    )
    shape = path.stem.split("-")[1]
    assert result.stdout.strip() == " ".join(shape.split("x")), path.name
    rows.append({"file": path.name,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "dimensions": result.stdout, "stderr": result.stderr})
assert len(rows) == 504, len(rows)
with (root / "independent-decode.json").open("x") as output:
    json.dump({"oracle": subprocess.check_output(["magick", "-version"], text=True),
               "manifest_sha256": hashlib.sha256((root / "MANIFEST.tsv").read_bytes()).hexdigest(),
               "count": len(rows), "files": rows}, output, indent=2)
print("Independent ImageMagick readback: 504/504 fixtures with expected dimensions.")
