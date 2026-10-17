#!/usr/bin/env python3
"""Path-reach check for tests/colorspace_paths.rs.

A passing test does not show it reached the path it names: gates send frames
elsewhere silently. This corrupts one planes-to-pixels or derived-read site at
a time, in a COPY of the source tree, re-runs colorspace_paths and records
which tests fail. Every site must fail at least one test.
(`zenjpeg/docs/DECODER_PATHS.md`, "Proving a test reaches a path".)

    just colorspace-mutation-reach            # copy the tree, run every site
    just colorspace-mutation-reach list       # list the sites

or by hand:

    rsync -a --exclude target --exclude .jj --exclude .git ./ ../zenjpeg-mutant/
    python3 scripts/colorspace_mutation_reach.py ../zenjpeg-mutant OUT_DIR list
    python3 scripts/colorspace_mutation_reach.py ../zenjpeg-mutant OUT_DIR [START [END]]

Never point it at a working checkout: it edits files (and restores them).
`MUTATION_CMD_PREFIX` (split like a shell word list, not run through a shell)
is prepended to the cargo command, e.g. a resource limiter; `CARGO_TARGET_DIR`
is honoured. Results: OUT_DIR/NN.log and OUT_DIR/NN.diff (the one-line
change) per site, OUT_DIR/results_START_END.json, and OUT_DIR/src, a copy of
the unmutated zenjpeg/src the site line numbers refer to.
"""
import json
import os
import re
import shutil
import shlex
import subprocess
import sys

HELPERS = '''
// MUTATION-HARNESS (temporary; removed by colorspace_mutation_reach.py)
impl RowKernel {
    pub(crate) fn row_mutated(self, c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
        self.row(c0, c1, c2, out);
        if let Some(b) = out.first_mut() { *b ^= 0x40; }
    }
    pub(crate) fn row_h2_mutated(self, c0: &[i16], c1: &[i16], c2: &[i16], out: &mut [u8]) {
        self.row_h2(c0, c1, c2, out);
        if let Some(b) = out.first_mut() { *b ^= 0x40; }
    }
}
pub(crate) fn rgb8_row_mutated(src: &[u8], layout: Layout8, out: &mut [u8]) {
    rgb8_row(src, layout, out);
    if let Some(b) = out.first_mut() { *b ^= 0x40; }
}
pub(crate) fn gray8_row_mutated(src: &[u8], layout: Layout8, out: &mut [u8]) {
    gray8_row(src, layout, out);
    if let Some(b) = out.first_mut() { *b ^= 0x40; }
}
pub(crate) fn relayout_u8_mutated(src: &[u8], source: Layout8, dst: &mut [u8], target: Layout8) {
    relayout_u8(src, source, dst, target);
    if let Some(b) = dst.first_mut() { *b ^= 0x40; }
}
pub(crate) fn rgb_to_ycbcr_normalized_mutated(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (y, cb, cr) = rgb_to_ycbcr_normalized(r, g, b);
    (y + 0.25, cb, cr)
}
'''

# (file under zenjpeg/src, pattern, replacement, label). Every occurrence of
# the pattern on a non-comment, non-definition line before `mod tests` is one
# site.
PATTERNS = [
    ('decode/parser/scan.rs', 'kernel.row(', 'kernel.row_mutated(', 'RowKernel::row'),
    ('decode/parser/baseline_streaming.rs', 'kernel.row(', 'kernel.row_mutated(', 'RowKernel::row'),
    ('decode/parser/baseline_streaming.rs', 'kernel.row_h2(', 'kernel.row_h2_mutated(', 'RowKernel::row_h2'),
    ('decode/fused_parallel.rs', 'kernel.row(', 'kernel.row_mutated(', 'RowKernel::row'),
    ('decode/fused_parallel.rs', 'kernel.row_h2(', 'kernel.row_h2_mutated(', 'RowKernel::row_h2'),
    ('decode/parser/fast_integer.rs', 'kernel.row(', 'kernel.row_mutated(', 'RowKernel::row'),
    ('decode/parser/fast_integer.rs', 'kernel.row_h2(', 'kernel.row_h2_mutated(', 'RowKernel::row_h2'),
    ('decode/scanline.rs', 'kernel.row(', 'kernel.row_mutated(', 'RowKernel::row'),
    ('decode/scanline.rs', ' rgb8_row(', ' crate::color::planes::rgb8_row_mutated(', 'rgb8_row'),
    ('decode/scanline.rs', ' gray8_row(', ' crate::color::planes::gray8_row_mutated(', 'gray8_row'),
    ('decode/parser/output.rs', 'relayout_u8(', 'crate::color::planes::relayout_u8_mutated(', 'relayout_u8'),
    ('decode/parser/output.rs', 'rgb_u8_swap_rb_inplace(&mut buf[..have]);',
     'rgb_u8_swap_rb_inplace(&mut buf[..have]); buf[0] ^= 0x40;', 'in-place RGB->BGR'),
    ('decode/parser/output.rs', 'let q = |v: f32| (v + 128.0).round()',
     'let q = |v: f32| (v + 129.0).round()', 'f32 path u8 RGB arm'),
    ('decode/parser/output.rs', '(r + 128.0) / 255.0,', '(r + 129.0) / 255.0,', 'f32 path f32 RGB arm'),
    ('decode/scanline.rs', 'rgb_luma_f32(', '0.25 + rgb_luma_f32(', 'derived luma f32'),
    ('decode/scanline.rs', 'rgb_to_ycbcr_normalized(',
     'crate::color::planes::rgb_to_ycbcr_normalized_mutated(', 'derived ycbcr f32'),
    ('decode/scanline.rs', 'srgb_to_linear(', '0.25 + srgb_to_linear(', 'derived linear'),
    ('decode/scanline.rs', 'linear_srgb::default::srgb_to_linear(',
     '0.25 + linear_srgb::default::srgb_to_linear(', 'f32 linearize'),
    # Plane reads that return the gray / Y / Cb / Cr samples themselves.
    ('decode/scanline.rs', 'a as f32 / 255.0,', '(a as f32 + 64.0) / 255.0,', 'ycbcr_f32 YCbCr arm'),
    ('decode/scanline.rs', 'put(px, (v as f32 / 255.0, 0.0, 0.0));',
     'put(px, ((v as f32 + 64.0) / 255.0, 0.0, 0.0));', 'ycbcr_f32 gray arm'),
    ('decode/scanline.rs', '*o = v as f32 / 255.0;', '*o = (v as f32 + 64.0) / 255.0;', 'gray_f32 gray/Y arm'),
    # The per-component path (f32 planes): u8 and f32 output by model.
    ('decode/parser/output.rs', 'crate::color::xyb::xyb_planes_to_srgb_u8_simd(c0, c1, c2, &mut rgb),',
     '{ crate::color::xyb::xyb_planes_to_srgb_u8_simd(c0, c1, c2, &mut rgb); rgb[0] ^= 0x40; }',
     'per-component u8 XYB arm'),
    ('decode/parser/output.rs', 'ycbcr_planes_f32_to_rgb_u8(c0, c1, c2, &mut rgb),',
     '{ ycbcr_planes_f32_to_rgb_u8(c0, c1, c2, &mut rgb); rgb[0] ^= 0x40; }',
     'per-component u8 YCbCr arm'),
    ('decode/parser/output.rs', 'gray_f32_to_gray_u8(&planes_f32[0], &mut output);',
     'gray_f32_to_gray_u8(&planes_f32[0], &mut output); output[0] ^= 0x40;', 'per-component u8 gray arm'),
    ('decode/parser/output.rs', 'gray_f32_to_rgb_u8(&planes_f32[0], &mut rgb);',
     'gray_f32_to_rgb_u8(&planes_f32[0], &mut rgb); rgb[0] ^= 0x40;', 'per-component u8 gray->RGB arm'),
    ('decode/parser/output.rs', '&planes_f32[3],', '&planes_f32[2],', 'per-component CMYK/YCCK arm (K <- plane 2)'),
    ('decode/parser/output.rs', 'crate::color::xyb::xyb_planes_to_srgb_f32_simd(c0, c1, c2, &mut rgb)',
     'crate::color::xyb::xyb_planes_to_srgb_f32_simd(c0, c1, c2, &mut rgb); rgb[0] += 0.25;',
     'per-component f32 XYB arm'),
    ('decode/parser/output.rs', 'ycbcr_planes_f32_to_rgb_f32(c0, c1, c2, &mut rgb),',
     '{ ycbcr_planes_f32_to_rgb_f32(c0, c1, c2, &mut rgb); rgb[0] += 0.25; }',
     'per-component f32 YCbCr arm'),
    ('decode/parser/output.rs', 'gray_f32_to_gray_f32(&planes_f32[0], &mut output);',
     'gray_f32_to_gray_f32(&planes_f32[0], &mut output); output[0] += 0.25;', 'per-component f32 gray arm'),
    ('decode/parser/output.rs', 'gray_f32_to_rgb_f32(&planes_f32[0], &mut rgb);',
     'gray_f32_to_rgb_f32(&planes_f32[0], &mut rgb); rgb[0] += 0.25;', 'per-component f32 gray->RGB arm'),
    # `decode_to_ycbcr_f32` of an RGB frame.
    ('decode/parser/output.rs', '(*y, *cb, *cr) = (yy - 128.0, cbb, crr);',
     '(*y, *cb, *cr) = (yy - 127.75, cbb, crr);', 'decode_to_ycbcr_f32 RGB conversion'),
]


def sites(src):
    out = []
    for rel, pat, rep, label in PATTERNS:
        lines = open(os.path.join(src, rel)).read().split('\n')
        end = next((k for k, l in enumerate(lines) if l.startswith('mod tests')), len(lines))
        for k, l in enumerate(lines[:end]):
            s = l.strip()
            if pat not in l or s.startswith('//') or s.startswith('use ') or 'fn ' in l:
                continue
            if not pat.startswith(('linear_srgb', ' ')) and ('::' + pat) in l:
                continue  # a path-qualified call: its own pattern covers it
            if l.count(pat) != 1:
                sys.exit(f'{rel}:{k + 1}: {pat!r} occurs {l.count(pat)} times')
            out.append(dict(label=f'{label} {rel}:{k + 1}', rel=rel, line=k, pat=pat, rep=rep,
                            text=s[:110]))
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    tree, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    if os.path.isdir(os.path.join(tree, '.jj')) or os.path.isdir(os.path.join(tree, '.git')):
        sys.exit(f'{tree} has .jj/.git: run on a copy of the tree, not a checkout')
    src = os.path.join(tree, 'zenjpeg/src')
    planes = os.path.join(src, 'color/planes.rs')
    all_sites = sites(src)
    if sys.argv[3:4] == ['list']:
        for i, s in enumerate(all_sites):
            print(f'{i:02} {s["label"]}: {s["text"]}')
        return
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, 'src')):
        shutil.copytree(src, os.path.join(out_dir, 'src'))
    cmd = shlex.split(os.environ.get('MUTATION_CMD_PREFIX', '')) + [
        'cargo', 'test', '--release', '-p', 'zenjpeg', '--features', 'parallel',
        '--test', 'colorspace_paths', '--', '--test-threads', '4']
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else len(all_sites)
    planes_orig = open(planes).read()
    results = []
    try:
        open(planes, 'w').write(planes_orig + HELPERS)
        for i, s in enumerate(all_sites[start:end], start):
            path = os.path.join(src, s['rel'])
            orig = open(path).read()
            lines = orig.split('\n')
            before = lines[s['line']]
            lines[s['line']] = before.replace(s['pat'], s['rep'], 1)
            with open(os.path.join(out_dir, f'{i:02}.diff'), 'w') as f:
                f.write(f"--- zenjpeg/src/{s['rel']}:{s['line'] + 1}\n-{before}\n+{lines[s['line']]}\n")
            try:
                open(path, 'w').write('\n'.join(lines))
                log = os.path.join(out_dir, f'{i:02}.log')
                with open(log, 'w') as f:
                    rc = subprocess.run(cmd, cwd=tree, stdout=f,
                                        stderr=subprocess.STDOUT).returncode
                text = open(log).read()
                r = dict(i=i, site=s['label'], text=s['text'], rc=rc,
                         compile_error='could not compile' in text,
                         failed=sorted(set(re.findall(r'^test (\S+) \.\.\. FAILED', text, re.M))))
                results.append(r)
                print(json.dumps(r), flush=True)
            finally:
                open(path, 'w').write(orig)
    finally:
        open(planes, 'w').write(planes_orig)
    json.dump(results, open(os.path.join(out_dir, f'results_{start}_{end}.json'), 'w'), indent=1)
    missed = [r['site'] for r in results if r['rc'] == 0]
    broken = [r['site'] for r in results if r['compile_error']]
    print(f'{len(results)} sites, {len(missed)} not caught, {len(broken)} did not compile')
    for m in missed + broken:
        print('  ', m)
    sys.exit(1 if missed or broken else 0)


main()
