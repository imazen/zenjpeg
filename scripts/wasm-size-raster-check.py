#!/usr/bin/env python3
"""Compare size-experiment WASM rasters and deliberate format refusals."""
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
out = Path(sys.argv[1]).resolve()
manifest = json.loads((root/'zenjpeg/tests/support/native_manifest.json').read_text())
fixtures = [root/f'zenjpeg/fuzz/seeds/native/{r["name"]}.jpg' for r in manifest if r.get('converted_format') == 'rgb']
fixtures += sorted((root/'zenjpeg/tests/testdata/permutation_regression').glob('xyb*.jpg'))

def frame(data):
    pos = 2
    while pos < len(data):
        assert data[pos] == 255
        while data[pos] == 255: pos += 1
        marker = data[pos]; pos += 1
        size = int.from_bytes(data[pos:pos+2], 'big')
        if marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc9, 0xca):
            return marker, data[pos+2]
        pos += size
    raise ValueError('no frame')

rows=[]
for fixture in fixtures:
    data=fixture.read_bytes(); marker,precision=frame(data)
    expected=None
    for variant in ['full','no-arithmetic','no-extended','no-both']:
        r=subprocess.run(['node','--no-warnings',str(root/'scripts/wasm-size-run.mjs'),str(out/f'{variant}-1.wasm')],input=data,capture_output=True,timeout=15)
        refused=(variant in ('no-arithmetic','no-both') and marker in (0xc9,0xca)) or (variant in ('no-extended','no-both') and (precision!=8 or marker==0xc3))
        if refused:
            assert r.returncode==2,(fixture,variant,r.returncode,r.stderr)
            assert not r.stdout
        else:
            assert r.returncode==0,(fixture,variant,r.returncode,r.stderr)
            assert len(r.stdout)>8
            if expected is None: expected=r.stdout
            assert r.stdout==expected,(fixture,variant,'raster mismatch')
        rows.append(dict(fixture=fixture.name,variant=variant,refused=refused,raster_sha256=None if refused else hashlib.sha256(r.stdout).hexdigest()))
(out/'raster-checks.json').write_text(json.dumps(rows,indent=2)+'\n')
print(f'{len(fixtures)} fixtures; {sum(not r["refused"] for r in rows)} exact raster checks; {sum(r["refused"] for r in rows)} explicit refusals')
