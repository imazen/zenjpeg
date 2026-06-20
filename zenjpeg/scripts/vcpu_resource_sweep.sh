#!/usr/bin/env bash
# vCPU resource sweep for zenjpeg — peak heap / peak RSS / marginal WS / wall
# across (size x quality x THREAD-COUNT). zenjpeg's strip DCT parallelises over
# the GLOBAL rayon pool (no per-call thread knob), gated above ~512^2
# (PARALLEL_THRESHOLD); thread count is set via RAYON_NUM_THREADS per process.
# AQ/entropy are serial, so the parallel fraction (and thus speedup) is small —
# this sweep measures exactly how small. estimate_encode is thread-independent.
#
# Two runs per cell: clean (wall + VmHWM peak/delta + est_*) and heaptrack
# (PEAK_HEAP) at a thread subset. ONE PROCESS PER CELL, run-heavy, SERIAL.
#
# Usage: scripts/vcpu_resource_sweep.sh <driver_bin> <raw_img_dir> <out.tsv>
#   raw_img_dir holds <label>.rgb (packed RGB8) + the square size IS <label>.
set -uo pipefail
DRIVER="${1:?driver}"; IMGDIR="${2:?raw img dir}"; OUT="${3:?out tsv}"
HT_DIR="${HT_DIR:-/tmp/zenjpeg_vcpu_heaptrack}"; mkdir -p "$HT_DIR"
TMPOUT="${TMPOUT:-/tmp/zenjpeg_vcpu_out.jpg}"
export GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072

IMAGES=( "256:photo" "1024:photo" "2048:photo" )
QUALITIES=( 50 85 )               # baseline 4:2:0, trellis+brd off (isolate DCT-parallel)
THREADS=( 1 2 4 8 16 28 )
HT_THREADS="${HT_THREADS:-1 8 28}"
TRELLIS=0; BRD=0

parse_ht() { heaptrack_print "$1" 2>/dev/null | python3 -c '
import sys,re
ph=pr=0
def kb(v,u): f={"B":1/1024,"K":1,"M":1024,"G":1024*1024}.get(u[0].upper(),0); return f*float(v)
for ln in sys.stdin:
    m=re.search(r"peak heap memory consumption:\s*([\d.]+)\s*([KMGB])",ln)
    if m: ph=kb(m.group(1),m.group(2))
    m=re.search(r"peak RSS[^:]*:\s*([\d.]+)\s*([KMGB])",ln)
    if m: pr=kb(m.group(1),m.group(2))
print(f"{int(ph)} {int(pr)}")'; }
getf() { sed -n "s/.*\b$2=\([^ ]*\).*/\1/p" <<<"$1"; }

echo -e "codec\tcontent_class\tsrc\twidth\theight\tpixels\tpath\teffort\tthreads\test_min_kb\test_typ_kb\test_max_kb\test_time_ms\tmeas_peak_heap_kb\tmeas_peak_rss_kb\tmeas_vmhwm_kb\tmeas_delta_kb\tmeas_wall_ms\tmeas_user_ms\tmeas_sys_ms\tbytes\tok" > "$OUT"

total=$(( ${#IMAGES[@]} * ${#QUALITIES[@]} * ${#THREADS[@]} )); i=0
for spec in "${IMAGES[@]}"; do
  label="${spec%%:*}"; cls="${spec##*:}"; raw="$IMGDIR/${label}.rgb"
  [[ -f "$raw" ]] || { echo "MISSING $raw" >&2; continue; }
  for q in "${QUALITIES[@]}"; do
    for t in "${THREADS[@]}"; do
      i=$((i+1))
      printf '%s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "claude-resource-harness" \
        "jpeg vcpu sweep $i/$total ${label} q${q} t${t}" > .workongoing 2>/dev/null || true
      echo "[$i/$total] ${label}^2 q${q} t${t}" >&2
      line=$(RAYON_NUM_THREADS=$t "$DRIVER" "$raw" "$label" "$label" encode "$q" "$TRELLIS" "$BRD" "$TMPOUT" 2>/dev/null)
      [[ -z "$line" ]] && { echo "  FAIL clean" >&2; continue; }
      delta=$(getf "$line" delta_kb); vmhwm=$(getf "$line" peak_kb)
      wall=$(getf "$line" wall_ms);   user=$(getf "$line" user_ms)
      sys=$(getf "$line" sys_ms);     bytes=$(getf "$line" bytes)
      emin=$(getf "$line" est_min_kb); etyp=$(getf "$line" est_typ_kb)
      emax=$(getf "$line" est_max_kb); etime=$(getf "$line" est_time_ms)
      ph=""; pr=""
      if [[ " $HT_THREADS " == *" $t "* ]]; then
        htf="$HT_DIR/${label}_q${q}_t${t}"; rm -f "${htf}.zst"
        RAYON_NUM_THREADS=$t heaptrack -o "$htf" "$DRIVER" "$raw" "$label" "$label" encode "$q" "$TRELLIS" "$BRD" "$TMPOUT" >/dev/null 2>&1
        read -r ph pr < <(parse_ht "${htf}.zst")
      fi
      px=$((label*label))
      echo -e "zenjpeg\t${cls}\t${label}.rgb\t${label}\t${label}\t${px}\tlossy\t${q}\t${t}\t${emin}\t${etyp}\t${emax}\t${etime}\t${ph}\t${pr}\t${vmhwm}\t${delta}\t${wall}\t${user}\t${sys}\t${bytes}\t1" >> "$OUT"
    done
  done
done
echo "wrote $OUT ($(( $(wc -l < "$OUT") - 1 )) rows)" >&2
