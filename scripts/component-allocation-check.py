#!/usr/bin/env python3
"""Run one Rust test with a one-shot allocation failure, optionally instrumented."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--coverage', action='store_true')
args = parser.parse_args()
scratch = Path(os.environ['TMPDIR'])
interposer = scratch / 'zenjpeg-fail-alloc.so'
subprocess.run(['cc', '-shared', '-fPIC', '-std=c11', '-Wall', '-Wextra', '-Werror',
               'zenjpeg/tests/support/fail_alloc.c', '-o', str(interposer)], check=True)
env = os.environ.copy()
if args.coverage:
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--no-deps', '--format-version=1'], text=True))
    env['CARGO_TARGET_DIR'] = env.get('CARGO_LLVM_COV_TARGET_DIR', str(Path(metadata['target_directory']) / 'llvm-cov-target'))
    settings = subprocess.check_output(['cargo', 'llvm-cov', 'show-env', '--sh'], env=env, text=True)
    # cargo emits shell-quoted assignments; shlex preserves spaces in flags/paths.
    import shlex
    for line in settings.splitlines():
        words = shlex.split(line)
        if words and words[0] == 'export':
            key, value = words[1].split('=', 1)
            env[key] = value.rstrip(';')
output = subprocess.check_output(['cargo', 'test', '--locked', '-p', 'zenjpeg',
    '--features', 'inventory', '--test', 'multicomponent', '--no-run',
    '--message-format=json'], env=env, text=True)
executables = [item['executable'] for line in output.splitlines()
               if line.startswith('{') and (item := json.loads(line)).get('executable')]
assert len(executables) == 1
# Only the final test subprocess loads the injector, never Cargo/rustc.
env['LD_PRELOAD'] = str(interposer)
env['ZENJPEG_FAIL_ALLOC_SIZE'] = '1820'  # (4 + 10) * size_of::<Option<[u16; 64]>>()
subprocess.run([executables[0], '--ignored', '--exact',
               'component_quantization_allocation_failure', '--test-threads=1'], env=env, check=True)
