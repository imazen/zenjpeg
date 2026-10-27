#!/usr/bin/env python3
"""Report LLVM true/false branch outcomes on added Rust production lines.

BASE is the common base of the 12-bit and component support stack. No production error-path or module exclusions; cfg(test) assertions are excluded. Use --require-complete to enforce every measured outcome.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('base')
p.add_argument('report', type=Path)
p.add_argument('--require-complete', action='store_true')
a = p.parse_args()
root = Path.cwd()
diff = subprocess.check_output(['git', 'diff', '--no-ext-diff', '--unified=0',
                               '--ignore-all-space', a.base, '--', 'zenjpeg/src'], text=True)
added = {}
file = None
line_no = 0
for line in diff.splitlines():
    if line.startswith('+++ b/'):
        file = str(root / line[6:])
    elif line.startswith('@@'):
        line_no = int(re.search(r'\+(\d+)', line)[1])
    elif line.startswith('+') and not line.startswith('+++'):
        added.setdefault(file, set()).add(line_no)
        line_no += 1
    elif line.startswith(' '):
        line_no += 1
# Unit-test assertions live in src/ too. They are not production branches;
# derive their module names from cfg(test), then demangle function identities.
# Production functions called by those tests remain fully included.
functions = [f for data in json.loads(a.report.read_text())['data'] for f in data['functions']]
names = subprocess.check_output(['rustfilt'], input='\n'.join(f['name'] for f in functions), text=True).splitlines()
test_modules = set()
for name in added:
    source = Path(name).read_text()
    test_modules.update(re.findall(r'#\[cfg\(test\)\]\s*(?:pub\s+)?mod\s+(\w+)', source))
branches = {}
for function, identity in zip(functions, names, strict=True):
    for b in function['branches']:
        name = str(Path(function['filenames'][b[6]]).resolve())
        if b[0] not in added.get(name, set()):
            continue
        if any('::' + module + '::' in identity for module in test_modules):
            continue
        key = (name, *b[:4])
        hits = branches.setdefault(key, [0, 0])
        hits[0] += b[4]
        hits[1] += b[5]
covered = total = 0
for key, hits in sorted(branches.items()):
    total += 2
    covered += sum(h > 0 for h in hits)
    if not all(hits):
        name, line, column, _, _ = key
        print(f'{Path(name).relative_to(root)}:{line}:{column}: true={hits[0]} false={hits[1]}')
print(f'Changed branch outcomes: {covered}/{total}')
if not total or (a.require_complete and covered != total):
    raise SystemExit('Complete branch coverage gate failed.')
