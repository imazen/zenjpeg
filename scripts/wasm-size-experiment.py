#!/usr/bin/env python3
"""Measure temporary decoder exclusions in a disposable jj workspace.

Not production feature gates. Keeps XYB and ordinary 8-bit paths enabled.
Run under run-heavy. Outputs binaries and measurements outside the repository.
"""
import argparse, gzip, hashlib, json, os, re, subprocess, time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('workspace', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--simd', action='store_true')
a = p.parse_args()
w, out = a.workspace.resolve(), a.output.resolve()
if w == Path(__file__).resolve().parent.parent or not (w / '.jj/repo').is_file():
    raise SystemExit('Use a disposable secondary jj workspace, not the main checkout')
if (out / 'measurements.json').exists():
    raise SystemExit('Choose a new output directory; existing measurements are protected')
out.mkdir(parents=True, exist_ok=True)
source_id = subprocess.check_output(['jj', 'log', '-r', '@-', '--no-graph', '-T', 'commit_id'], cwd=w, text=True).strip()
files = list((w / 'zenjpeg/src/decode').rglob('*.rs'))
original = {f: f.read_text() for f in files}
manifest = w / 'Cargo.toml'
manifest_original = manifest.read_text()
probe = w / 'internal/wasm-size-probe'
(probe / 'src').mkdir(parents=True, exist_ok=True)
(probe / 'Cargo.toml').write_text('''[package]
name = "wasm-size-probe"
version = "0.0.0"
edition = "2024"
publish = false
[dependencies]
zenjpeg = { path = "../../zenjpeg", default-features = false }
enough = { workspace = true }
''')
(probe / 'src/main.rs').write_text('''#![forbid(unsafe_code)]
use std::io::{Read, Write};
fn main() {
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();
    match zenjpeg::decoder::Decoder::new().decode(&input, enough::Unstoppable) {
        Ok(image) => {
            let mut out = std::io::stdout().lock();
            out.write_all(&image.width().to_le_bytes()).unwrap();
            out.write_all(&image.height().to_le_bytes()).unwrap();
            out.write_all(image.pixels_u8().unwrap()).unwrap();
        }
        Err(e) => { eprintln!("{e}"); std::process::exit(2); }
    }
}
''')
manifest.write_text(manifest_original.replace('members = [', 'members = [\n    "internal/wasm-size-probe",', 1))
env = os.environ.copy()
env.update(CARGO_TARGET_DIR=str(out / 'target'), CARGO_PROFILE_RELEASE_DEBUG='0',
           CARGO_PROFILE_RELEASE_STRIP='symbols', CARGO_PROFILE_RELEASE_OPT_LEVEL='s',
           CARGO_PROFILE_RELEASE_LTO='thin', CARGO_PROFILE_RELEASE_CODEGEN_UNITS='1',
           CARGO_INCREMENTAL='0')

if a.simd:
    env['RUSTFLAGS'] = '-C target-feature=+simd128'

def stub(source, name):
    start = source.index('fn ' + name + '(')
    brace = source.index('{', start)
    depth = 0
    for m in re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[{}]', source[brace:], re.S):
        if m[0] == '{': depth += 1
        elif m[0] == '}':
            depth -= 1
            if depth == 0:
                end = brace + m.end()
                return source[:brace] + '{ Err(Error::unsupported_feature("size experiment exclusion")) }' + source[end:]
    raise RuntimeError(name)

rows = []
try:
    for variant in ['full', 'no-arithmetic', 'no-extended', 'no-both']:
        for f, s in original.items():
            if variant in ('no-arithmetic', 'no-both') and f.name == 'arithmetic_scan.rs':
                for name in ['decode_arithmetic_scan', 'decode_arithmetic_progressive_scan']:
                    s = stub(s, name)
            if variant in ('no-extended', 'no-both'):
                if f.name == 'predictive.rs':
                    for name in ['predictive_reader', 'next_row']: s = stub(s, name)
                if f.name == 'markers.rs':
                    s = s.replace('self.precision = self.read_u8()?;', 'self.precision = self.read_u8()?;\nif self.precision != 8 { return Err(Error::unsupported_feature("size experiment precision")); }', 1)
                s = re.sub(r'(?<![\w.])((?:self\.)?precision == 12)', r'false && (\1)', s)
            f.write_text(s)
        for repeat in (1, 2):
            (w / 'zenjpeg/src/lib.rs').touch()
            log = out / f'{variant}-{repeat}.log'
            with log.open('w') as stream:
                subprocess.run(['timeout', '--kill-after=15s', '180s', 'cargo', 'build', '--release', '-p', 'wasm-size-probe', '--target', 'wasm32-wasip1'], cwd=w, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            data = (out / 'target/wasm32-wasip1/release/wasm-size-probe.wasm').read_bytes()
            artifact = out / f'{variant}-{repeat}.wasm'
            artifact.write_bytes(data)
            row = dict(variant=variant, repeat=repeat, bytes=len(data), gzip_bytes=len(gzip.compress(data, mtime=0)), sha256=hashlib.sha256(data).hexdigest())
            rows.append(row)
            (out / 'measurements.json').write_text(json.dumps(dict(source=source_id, target='wasm32-wasip1', profile='opt=s, thin LTO, CGU=1, stripped', simd128=a.simd, results=rows), indent=2)+'\n')
            print(json.dumps(row), flush=True)
finally:
    for f, s in original.items(): f.write_text(s)
    manifest.write_text(manifest_original)
