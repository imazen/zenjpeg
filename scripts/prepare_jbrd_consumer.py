#!/usr/bin/env python3
"""Create an isolated scan-metadata consumer with this source's exact patch table."""
import argparse
from pathlib import Path
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
a=p.parse_args()
a.output.mkdir(exist_ok=False)
root=Path(__file__).resolve().parent.parent
manifest=(root/'Cargo.toml').read_text()
patch=manifest[manifest.index('[patch.crates-io]'):]
assert not any('path =' in line for line in patch.splitlines()), 'relative patches need explicit rebasing'
(a.output/'Cargo.toml').write_text('''[package]
name = "jbrd-metadata-consumer"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
zenjpeg = { path = "'''+str(root/'zenjpeg')+'''", default-features = false }
enough = { version = "0.4", features = ["std"] }
[features]
default = ["zenjpeg/default"]
parallel = ["zenjpeg/parallel"]
cms = ["zenjpeg/moxcms"]
'''+patch)
(a.output/'tests/testdata').mkdir(parents=True)
shutil.copy2(root/'zenjpeg/tests/jbrd_baseline_metadata.rs',a.output/'tests/jbrd_baseline_metadata.rs')
for file in (root/'zenjpeg/tests/testdata').glob('jpeg_scan_v012_*.jpg.hex'):
 shutil.copy2(file,a.output/'tests/testdata'/file.name)
print(a.output/'Cargo.toml')
