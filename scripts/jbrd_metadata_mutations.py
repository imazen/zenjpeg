#!/usr/bin/env python3
"""Replay independent producer faults in the owned frozen consumer export."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
a.output.mkdir(exist_ok=False)
root=a.manifest.resolve().parent.parent/'zenjpeg'
paths={'entropy':root/'zenjpeg/src/entropy/decoder.rs','scan':root/'zenjpeg/src/decode/parser/scan.rs'}
fixed={k:v.read_bytes()for k,v in paths.items()}
faults=[
 ('no-fast-reset','entropy','trailing_zrl = 0;', 'trailing_zrl += 0;',0),
 ('no-table-reset','entropy','trailing_zrl = 0;', 'trailing_zrl += 0;',1),
 ('no-slow-reset','entropy','trailing_zrl = 0;', 'trailing_zrl += 0;',2),
 ('no-table-run','entropy','trailing_zrl += 1;', 'trailing_zrl += 0;',0),
 ('no-slow-run','entropy','trailing_zrl += 1;', 'trailing_zrl += 0;',1),
 ('no-index','entropy','*index += 1;', '*index += 0;',0),
 ('wrong-run-index','entropy','scan.extra_zero_runs.push((index, trailing_zrl));','scan.extra_zero_runs.push((index + 1, trailing_zrl));',0),
 ('drop-scan','scan','scans.push(scan);','let _ = scan;',0),
]
results=[]
try:
 for label,key,before,after,occurrence in faults:
  out=a.output/label;out.mkdir()
  text=fixed[key].decode();positions=[];at=0
  while (at:=text.find(before,at))>=0:positions.append(at);at+=len(before)
  assert len(positions)>occurrence,(label,positions)
  pos=positions[occurrence];fault=text[:pos]+text[pos:].replace(before,after,1)
  paths[key].write_text(fault)
  (out/'fixed.rs').write_bytes(fixed[key]);(out/'fault.rs').write_text(fault)
  command=['cargo','test','--locked','--manifest-path',str(a.manifest),'-p','jxl-encoder','--features','__expert,parallel,jpeg-reencoding','--test','zenjpeg_baseline_metadata_contract','--no-run','--message-format=json']
  with (out/'build.jsonl').open('x')as log:subprocess.run(command,stdout=log,check=True)
  rows=[json.loads(l)for l in(out/'build.jsonl').read_text().splitlines()]
  binaries=[Path(r['executable'])for r in rows if r.get('reason')=='compiler-artifact'and r.get('executable')and r['target']['name']=='zenjpeg_baseline_metadata_contract']
  assert len(binaries)==1
  shutil.copy2(binaries[0],out/'test-binary')
  with(out/'runtime.log').open('x')as log:r=subprocess.run([str(binaries[0]),'--nocapture','--test-threads=1'],stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==101 and 'test result: FAILED' in(out/'runtime.log').read_text(),label
  paths[key].write_bytes(fixed[key]);assert paths[key].read_bytes()==fixed[key]
  row={'fault':label,'runtime_exit_code':r.returncode,'source_sha256':hashlib.sha256(fault.encode()).hexdigest(),'binary_sha256':hashlib.sha256((out/'test-binary').read_bytes()).hexdigest(),'restored_sha256':hashlib.sha256(fixed[key]).hexdigest()}
  (out/'identity.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
  (a.output/'results.partial.json').write_text(json.dumps(results,indent=2)+'\n')
  print(label+': failed as required; source restored',flush=True)
finally:
 for key,path in paths.items():path.write_bytes(fixed[key]);assert path.read_bytes()==fixed[key]
(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
