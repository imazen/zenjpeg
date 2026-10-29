#!/usr/bin/env python3
"""Require 100% LCOV line coverage of changed executable Rust production lines.

No exclusions for error paths, closing lines, or allocation propagation. This is
line coverage, not branch coverage. BASE must be the intended PR base commit.
"""
import argparse
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('base')
parser.add_argument('lcov', type=Path)
args = parser.parse_args()
root = Path.cwd()
hits = {}
file = None
for line in args.lcov.read_text().splitlines():
    if line.startswith('SF:'):
        file = str(Path(line[3:]).resolve())
        hits.setdefault(file, {})
    elif line.startswith('DA:'):
        number, count, *_ = line[3:].split(',')
        number = int(number)
        hits[file][number] = hits[file].get(number, 0) + int(count)

diff = subprocess.check_output(['git', 'diff', '--no-ext-diff', '--unified=0',
                                '--ignore-all-space', args.base, '--', 'zenjpeg/src'], text=True)
changed = {}
number = 0
for line in diff.splitlines():
    if line.startswith('+++ b/'):
        file = str(root / line[6:])
    elif line.startswith('@@'):
        number = int(re.search(r'\+(\d+)', line)[1])
    elif line.startswith('+') and not line.startswith('+++'):
        if number in hits.get(file, {}):
            changed.setdefault(file, set()).add(number)
        number += 1
    elif line.startswith(' '):
        number += 1

covered = total = 0
for file, numbers in sorted(changed.items()):
    missing = sorted(n for n in numbers if hits[file][n] == 0)
    covered += len(numbers) - len(missing)
    total += len(numbers)
    print(f'{Path(file).relative_to(root)}: {len(numbers)-len(missing)}/{len(numbers)}')
    if missing:
        print('  uncovered:', ', '.join(map(str, missing)))
print(f'Changed executable lines: {covered}/{total}')
if total == 0:
    raise SystemExit('No changed executable lines matched the report; refusing a vacuous pass.')
if covered != total:
    raise SystemExit('100% changed-line coverage gate failed.')
