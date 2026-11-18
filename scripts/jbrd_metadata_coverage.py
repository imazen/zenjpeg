#!/usr/bin/env python3
"""Bind frozen consumer coverage to the exact baseline metadata producer."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
root = Path(__file__).resolve().parent.parent
export = a.manifest.resolve().parent.parent / 'zenjpeg'
sources = ['zenjpeg/src/entropy/decoder.rs', 'zenjpeg/src/decode/parser/scan.rs', 'zenjpeg/src/decode/parser/mod.rs']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
identity = {'command': [], 'sources': [], 'rustc': subprocess.check_output(['rustc', '-vV'], text=True)}
for relative in sources:
    src = root / relative
    assert src.read_bytes() == (export / relative).read_bytes(), relative
    dst = a.output / 'source' / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    identity['sources'].append({'path': relative, 'sha256': sha(src)})
consumer = a.manifest.resolve().parent
for relative in ['Cargo.toml', 'Cargo.lock', 'jxl-encoder/Cargo.toml',
                 'jxl-encoder/tests/zenjpeg_baseline_metadata_contract.rs']:
    src = consumer / relative
    dst = a.output / 'consumer' / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
for src in (consumer / 'jxl-encoder/tests/testdata').glob('jpeg_scan_v012_*.jpg.hex'):
    dst = a.output / 'consumer/jxl-encoder/tests/testdata' / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)
assert (consumer / 'jxl-encoder/tests/zenjpeg_baseline_metadata_contract.rs').read_bytes() == (root / 'zenjpeg/tests/jbrd_baseline_metadata.rs').read_bytes()
command = ['cargo', 'test', '--locked', '--manifest-path', str(a.manifest), '-p', 'jxl-encoder', '--features', '__expert,parallel,jpeg-reencoding', '--test', 'zenjpeg_baseline_metadata_contract', '--no-run', '--message-format=json']
identity['command'] = command
with (a.output / 'build.jsonl').open('x') as out:
    subprocess.run(command, stdout=out, check=True)
rows = [json.loads(line) for line in (a.output / 'build.jsonl').read_text().splitlines()]
objects = [Path(r['executable']) for r in rows if r.get('reason') == 'compiler-artifact' and r.get('executable') and r['target']['name'] == 'zenjpeg_baseline_metadata_contract']
assert len(objects) == 1
binary = objects[0]
shutil.copy2(binary, a.output / 'test-binary')
identity['binary_sha256'] = sha(binary)
profile = a.output.resolve() / 'test.profraw'
run = [str(binary), '--nocapture', '--test-threads=1']
with (a.output / 'runtime.log').open('x') as out:
    result = subprocess.run(run, stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, LLVM_PROFILE_FILE=str(profile)))
assert result.returncode == 0 and re.search(r'test result: ok\. [1-9][0-9]* passed; 0 failed', (a.output / 'runtime.log').read_text())
assert profile.is_file()
identity['run'] = run
identity['profile_sha256'] = sha(profile)
for row in identity['sources']: assert sha(root / row['path']) == row['sha256']
(a.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], text=True).strip())
llvm = sysroot / 'lib/rustlib/x86_64-unknown-linux-gnu/bin'
subprocess.run([str(llvm / 'llvm-profdata'), 'merge', '-sparse', str(profile), '-o', str(a.output / 'profile.profdata')], check=True)
with (a.output / 'coverage.json').open('x') as out, (a.output / 'coverage.stderr').open('x') as err:
    subprocess.run([str(llvm / 'llvm-cov'), 'export', str(a.output / 'test-binary'), '--instr-profile=' + str(a.output / 'profile.profdata')], stdout=out, stderr=err, check=True)
assert not (a.output / 'coverage.stderr').read_text(), (a.output / 'coverage.stderr').read_text()
report = json.loads((a.output / 'coverage.json').read_text())
windows = []
for relative in sources[:2]:
    lines = (root / relative).read_text().splitlines()
    if relative.endswith('decoder.rs'):
        first = next(i for i,l in enumerate(lines) if 'pub(crate) fn decode_block_into_maybe_tracked(' in l)
        end = next(i for i,l in enumerate(lines) if i > first and 'fn decode_block_impl<' in l)
        windows.append(('tracking-dispatch', relative, first+1, end))
        for i,line in enumerate(lines):
            if 'if TRACK_ZRL' in line:
                end = next(j for j in range(i+1,len(lines)) if lines[j].strip() == '}')
                windows.append((f'tracking-at-{i+1}',relative,i+1,end+1))
    else:
        for i,line in enumerate(lines):
            if 'let mut jbrd_scan =' in line or 'decode_block_into_maybe_tracked(' in line or 'if let Some((scan, _)) = jbrd_scan' in line:
                end = next(j for j in range(i+1,len(lines)) if lines[j].strip().endswith(');') or lines[j].strip() == '}')
                windows.append((f'scan-at-{i+1}',relative,i+1,end+1))
summary = []
for label,relative,first,last in windows:
    regions = {}
    for unit in report['data']:
        for fn in unit['functions']:
            for r in fn['regions']:
                if r[7] or not fn['filenames'][r[5]].endswith('/'+relative) or r[0]<first or r[2]>last: continue
                key=tuple(r[:4]);regions[key]=max(regions.get(key,0),r[4])
    assert regions,(label,first,last)
    row={'label':label,'source':relative,'first':first,'last':last,'covered':sum(v>0 for v in regions.values()),'total':len(regions),'regions':[{'location':k,'count':v} for k,v in sorted(regions.items())]}
    summary.append(row)
    print(f"{label}: {row['covered']}/{row['total']}",flush=True)
(a.output / 'regions.json').write_text(json.dumps(summary,indent=2)+'\n')
