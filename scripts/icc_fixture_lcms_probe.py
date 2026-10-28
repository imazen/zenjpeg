#!/usr/bin/env python3
"""What lcms makes of the ICC fixtures in tests/encode_carrier_policy.rs.

Builds `not_identified()` (an unidentified RGB profile and the bytes named
`lcms_2_17_rejects`) and Display P3 itself from zenpixels-convert's
DisplayP3Compat-v4.icc, and runs each through lcms
`transicc -i<bytes> -o*sRGB` on Display P3 (200, 40, 40). This records one
reader at one version; it is not an ICC validator, and the encoder does not
use it.

    git -C <zenpixels checkout> show \
        f8596aae:zenpixels-convert/src/profiles/DisplayP3Compat-v4.icc > p3.icc
    python3 -I scripts/icc_fixture_lcms_probe.py p3.icc <scratch dir>

Needs `transicc` (liblcms2-utils). Output of the 2026-10-28 run:
benchmarks/icc_fixture_lcms_probe_2026-10-28.txt.
"""
import hashlib
import pathlib
import struct
import subprocess
import sys

P3_SHA256 = "231752984cd4a5278e1b8d2390fe496767d4511fc81f54e1a5c69ae9ab4c42b5"


def put(profile, at, data):
    out = bytearray(profile)
    out[at:at + len(data)] = data
    return bytes(out)


def fixtures(p3):
    return [
        ("an unidentified profile", p3[:-1] + bytes([p3[-1] ^ 1])),
        ("the established unit test's bytes", b"fake icc profile data"),
        ("a truncated header", p3[:100]),
        ("a header without its tag count", p3[:131]),
        ("no acsp signature", put(p3, 36, b"xxxx")),
        ("a tag table past the end", put(p3, 128, struct.pack(">I", 30))),
        ("the largest tag count", put(p3, 128, struct.pack(">I", 0xFFFFFFFF))),
        ("Display P3 itself", p3),
    ]


def lcms(path):
    run = subprocess.run(
        ["transicc", f"-i{path}", "-o*sRGB", "-n", "-t1"],
        input="200 40 40\n",
        capture_output=True,
        text=True,
        timeout=10,
    )
    lines = [
        line.strip()
        for line in (run.stdout + run.stderr).splitlines()
        if line.strip() and not line.startswith(("LittleCMS", "Copyright"))
    ]
    return " | ".join(lines)


def main():
    p3 = pathlib.Path(sys.argv[1]).read_bytes()
    if hashlib.sha256(p3).hexdigest() != P3_SHA256:
        sys.exit("not zenpixels-convert's DisplayP3Compat-v4.icc (f8596aae)")
    scratch = pathlib.Path(sys.argv[2])
    scratch.mkdir(parents=True, exist_ok=True)
    for index, (what, data) in enumerate(fixtures(p3)):
        path = scratch / f"fixture{index:02}.icc"
        path.write_bytes(data)
        print(f"{what:34} {len(data):4} B  lcms: {lcms(path)}")


if __name__ == "__main__":
    main()
