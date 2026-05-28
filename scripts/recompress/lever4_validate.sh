#!/usr/bin/env bash
# Lever-4 validation: regenerate the n=50 corpus (persisted OUTSIDE /tmp so
# it survives reboots) and compare smart-router under-target at
# --max-iterations 1 (no closed loop) vs 3 (closed loop) on the two
# under-target-prone encoders (turbo, mozjpeg).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$REPO/../zjr-val-scratch"          # repo-sibling, persistent disk
ORIG_SRC=~/work/codec-eval/codec-corpus/CID22/CID22-512/training
ZM=~/work/zen/zenmetrics/target/release/zen-metrics
export ZENMETRICS_VRAM_CAP_BYTES="${ZENMETRICS_VRAM_CAP_BYTES:-4000000000}"
mkdir -p "$SCRATCH"

# pinned encoders (re-extract from cached docker images if absent)
ext() { [ -d "$2" ] && return 0; mkdir -p "$2"; docker run --rm "$1" tar -C "$3" -cf - . | tar -C "$2" -xf -; }
ext ati-mozjpeg "$SCRATCH/mozjpeg-4.1.5"  /opt/mozjpeg-4.1.5
ext ati-turbo   "$SCRATCH/turbo-3.1.0"    /opt/libjpeg-turbo-3.1.0
MOZ="$SCRATCH/mozjpeg-4.1.5/bin/cjpeg"; MOZ_LD="$SCRATCH/mozjpeg-4.1.5/lib"
TUR="$SCRATCH/turbo-3.1.0/bin/cjpeg";   TUR_LD="$SCRATCH/turbo-3.1.0/lib64:$SCRATCH/turbo-3.1.0/lib"

# 50-image corpus (first 50 sorted training PNGs — same set as the n=50 calibration)
ORIG="$SCRATCH/orig50"; mkdir -p "$ORIG"
if [ "$(ls "$ORIG"/*.png 2>/dev/null | wc -l)" -lt 50 ]; then
  i=0; for f in $(ls "$ORIG_SRC"/*.png | sort); do cp "$f" "$ORIG/"; i=$((i+1)); [ $i -ge 50 ] && break; done
fi
echo "corpus: $(ls "$ORIG"/*.png | wc -l) images"

# sources (turbo + mozjpeg, q20-95 step 5)
SRC="$SCRATCH/sources"; mkdir -p "$SRC"/{turbo,mozjpeg,_ppm}
for png in "$ORIG"/*.png; do s=$(basename "$png" .png); [ -f "$SRC/_ppm/$s.ppm" ] || convert "$png" "$SRC/_ppm/$s.ppm"; done
for ppm in "$SRC"/_ppm/*.ppm; do
  s=$(basename "$ppm" .ppm)
  for q in $(seq 20 5 95); do
    [ -f "$SRC/turbo/${s}__turbo__q${q}.jpg" ]     || LD_LIBRARY_PATH=$TUR_LD $TUR -quality "$q" -outfile "$SRC/turbo/${s}__turbo__q${q}.jpg" "$ppm" 2>/dev/null
    [ -f "$SRC/mozjpeg/${s}__mozjpeg__q${q}.jpg" ] || LD_LIBRARY_PATH=$MOZ_LD $MOZ -quality "$q" -outfile "$SRC/mozjpeg/${s}__mozjpeg__q${q}.jpg" "$ppm" 2>/dev/null
  done
done
echo "sources: turbo=$(ls "$SRC"/turbo/*.jpg|wc -l) moz=$(ls "$SRC"/mozjpeg/*.jpg|wc -l)"

ZJR="$REPO/target/release/zjr-calibrate"
for e in turbo mozjpeg; do
  for iters in 1 3; do
    "$ZJR" recompress-sweep --sources "$SRC/$e" --originals "$ORIG" \
      --output "$SCRATCH/val_iter${iters}_${e}.tsv" --targets 30,40,50,60,70,80 \
      --max-iterations "$iters" 2>&1 | tail -1
  done
done
echo "LEVER4_VALIDATE_DONE"
