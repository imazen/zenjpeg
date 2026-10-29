#!/usr/bin/env python3
"""Generate small native JPEG fixtures and independently decoded u16 samples.

Stock libjpeg-turbo 3.2.0 supplies 1–4 component fixtures. The separately
bounds-repaired reference supplies 5–10; see docs/12bit-lossless-decoding.md.
The exact encoder arguments and both SHA-256 digests are recorded in manifest.json.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

out = Path('zenjpeg/fuzz/seeds/native')
out.mkdir(parents=True, exist_ok=True)
stock = Path(os.environ['ZENJPEG_TURBO_BIN']) / 'native-components-oracle'
components = Path(os.environ['ZENJPEG_COMPONENT_ORACLE_DIR']) / 'build/native-components-oracle'
for oracle in (stock, components):
    assert subprocess.check_output([oracle, "version"]).strip() == b"3002000"
cases = []
for n in (1, 2, 3, 4, 5, 10):
    for precision, mode in ((8, 0), (12, 0), (12, 1), (12, 2), (12, 3)):
        cases.append((precision, n, 0, 1, 0, 1, mode, 9, 9))
for precision, n, predictor, point in ((2, 1, 1, 0), (8, 2, 2, 0), (12, 3, 3, 2),
                                    (16, 4, 4, 0), (16, 5, 5, 3), (12, 10, 6, 0),
                                    (16, 10, 7, 15)):
    cases.append((precision, n, 1, predictor, point, 1, 0, 9, 9))
for sampling in (2, 3, 4, 5, 6):
    cases.append((12, 10, 0, 1, 0, sampling, 0, 9, 9))
cases.extend(((12, 1, 0, 1, 0, 1, 0, 1, 1), (16, 4, 1, 1, 0, 1, 0, 1, 1)))
cases.append((12, 4, 0, 1, 0, 2, 0, 9, 9))  # YCCK
cases.extend(((12, 3, 0, 1, 0, 3, 0, 9, 9), (8, 1, 1, 1, 0, 1, 0, 9, 9)))
manifest = []
for i, case in enumerate(cases):
    precision, n, lossless, predictor, point, sampling, mode, w, h = case
    oracle = stock if n <= 4 else components
    args = ['encode', *map(str, case)]
    jpeg = subprocess.check_output([oracle, *args])
    raw = subprocess.check_output([oracle, 'decode', 'raw'], input=jpeg)
    assert len(raw) == w * h * n * 2
    name = f'{i:02d}-p{precision}-c{n}-sof3{lossless}-mode{mode}-s{sampling}'
    destination = out / (name + '.jpg')
    if destination.exists():
        assert destination.read_bytes() == jpeg, f'fixture changed: {destination}'
    else:
        destination.write_bytes(jpeg)
    manifest.append(dict(name=name, reference='stock-3.2.0' if n <= 4 else 'component-bounds-repaired-3.2.0',
                         args=args, jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),
                         samples_sha256=hashlib.sha256(raw).hexdigest()))
    if n in (1, 3, 4):
        output_format = 'gray' if n == 1 and lossless else 'cmyk' if n == 4 else 'rgb'
        converted = subprocess.check_output([oracle, 'decode', output_format], input=jpeg, timeout=30)
        assert len(converted) == w * h * ({'gray': 1, 'rgb': 3, 'cmyk': 4}[output_format]) * 2
        manifest[-1]['converted_format'] = output_format
        manifest[-1]['converted_samples_sha256'] = hashlib.sha256(converted).hexdigest()
Path('zenjpeg/tests/support/native_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'{len(cases)} fixtures, {sum(p.stat().st_size for p in out.glob("*.jpg"))} JPEG bytes')
