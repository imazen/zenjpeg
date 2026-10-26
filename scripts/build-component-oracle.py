#!/usr/bin/env python3
"""Build a test-only turbo oracle with frame-ID lookup repaired (no pixel changes).

Upstream 3.2.0 can encode >4 components, but get_sos() searches only the first
four. Its duplicate check already compares pointers in the current scan.
Keep the original oracle intact for all <=4-component conformance tests.
"""
import os
from pathlib import Path
import shutil
import subprocess

source = Path(os.environ['ZENJPEG_TURBO_SOURCE'])
dest = Path(os.environ['ZENJPEG_COMPONENT_ORACLE_DIR'])
if not (dest / 'source').exists():
    shutil.copytree(source, dest / 'source', ignore=shutil.ignore_patterns('build', '.git'))
marker = dest / 'source/src/jdmarker.c'
text = marker.read_text()
old = 'ci < cinfo->num_components && ci < MAX_COMPS_IN_SCAN;'
new = 'ci < cinfo->num_components;'
old_check = 'cc == compptr->component_id && !cinfo->cur_comp_info[ci]'
if old in text:
    assert text.count(old) == 1 and text.count(old_check) == 1
    text = text.replace(old, new).replace(old_check, 'cc == compptr->component_id')
    marker.write_text(text)
else:
    assert old_check not in text and new in text
# The lossless multi-scan controller indexes these buffers by frame component.
for name in ('jcdiffct.c', 'jddiffct.c'):
    path = dest / 'source/src' / name
    content = path.read_text()
    old_buffer = '_JSAMPARRAY buffer[MAX_COMPS_IN_SCAN];'
    new_buffer = '_JSAMPARRAY buffer[MAX_COMPONENTS];'
    assert content.count(old_buffer) + content.count(new_buffer) == 1
    path.write_text(content.replace(old_buffer, new_buffer))
build = dest / 'build'
subprocess.run(['cmake', '-S', str(dest / 'source'), '-B', str(build),
                '-DENABLE_SHARED=OFF', '-DWITH_SIMD=OFF', '-DCMAKE_BUILD_TYPE=Release'], check=True)
subprocess.run(['cmake', '--build', str(build), '--target', 'jpeg-static', '--parallel', '2'], check=True)
subprocess.run(['cc', '-Wall', '-Wextra', '-Werror', '-I'+str(build),
               '-I'+str(dest / 'source/src'), 'zenjpeg/tests/support/native_components.c',
               str(build / 'libjpeg.a'), '-lm', '-o', str(build / 'native-components-oracle')], check=True)
