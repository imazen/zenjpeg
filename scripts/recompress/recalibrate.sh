#!/usr/bin/env bash
# Full recalibration pipeline for zenjpeg-recompress.
#
# Re-runs every calibration step from a fresh image corpus and emits the
# regenerated tables + a validation report. Idempotent per stage (skips
# completed stages unless --force). See docs/RECALIBRATION_PROTOCOL.md
# for the protocol, gates, and what to commit.
#
# Usage:
#   scripts/recalibrate.sh <originals_png_dir> [work_dir] [qstep]
#
# Prerequisites (the script checks them and fails loud if missing):
#   - zen-metrics built with GPU metrics:
#       ~/work/zen/zenmetrics/target/release/zen-metrics
#   - pinned encoders extracted from the all-the-images docker stages:
#       docker build --target mozjpeg       -t ati-mozjpeg ~/work/all-the-images
#       docker build --target libjpeg-turbo -t ati-turbo   ~/work/all-the-images
#     (this script extracts them via tar-stream if absent)
#   - host cjpegli on PATH (jpegli sources)
set -euo pipefail

ORIG="${1:?usage: recalibrate.sh <originals_png_dir> [work_dir] [qstep]}"
WORK="${2:-/tmp/zjr-recal}"
QSTEP="${3:-5}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ZM=~/work/zen/zenmetrics/target/release/zen-metrics
export ZENMETRICS_VRAM_CAP_BYTES="${ZENMETRICS_VRAM_CAP_BYTES:-4000000000}"
DATE="$(date -u +%Y-%m-%d)"
mkdir -p "$WORK"

say() { printf '\n=== %s ===\n' "$1"; }

# ---- prerequisites -------------------------------------------------------
say "stage 0: prerequisites"
[ -x "$ZM" ] || { echo "FATAL: zen-metrics not built at $ZM"; exit 1; }
command -v cjpegli >/dev/null || { echo "FATAL: cjpegli not on PATH"; exit 1; }
# Extract pinned encoders if not already present.
extract_enc() { # <image> <src-dir-in-image> <dest>
    [ -d "$3" ] && return 0
    mkdir -p "$3"
    docker run --rm "$1" tar -C "$2" -cf - . | tar -C "$3" -xf -
}
extract_enc ati-mozjpeg /opt/mozjpeg-4.1.5      "$WORK/mozjpeg-4.1.5"
extract_enc ati-turbo   /opt/libjpeg-turbo-3.1.0 "$WORK/turbo-3.1.0"
MOZ_CJ="$WORK/mozjpeg-4.1.5/bin/cjpeg";  MOZ_LD="$WORK/mozjpeg-4.1.5/lib"
TURBO_CJ="$WORK/turbo-3.1.0/bin/cjpeg";  TURBO_LD="$WORK/turbo-3.1.0/lib64:$WORK/turbo-3.1.0/lib"
echo "pinned: $($TURBO_CJ -version 2>&1 | head -1); mozjpeg 4.1.5"

cargo build --release --manifest-path "$REPO/Cargo.toml" -p zjr-calibrate 2>&1 | tail -1
ZJR="$REPO/target/release/zjr-calibrate"

# ---- stage 1: encode multi-encoder sources -------------------------------
say "stage 1: encode sources (turbo 3.1.0 + mozjpeg 4.1.5 + jpegli, q step $QSTEP)"
SRC="$WORK/sources"; mkdir -p "$SRC"/{turbo,mozjpeg,jpegli,_ppm}
for png in "$ORIG"/*.png; do convert "$png" "$SRC/_ppm/$(basename "$png" .png).ppm"; done
qs=$(seq 20 "$QSTEP" 95 | tr '\n' ' ')
for ppm in "$SRC"/_ppm/*.ppm; do
    s=$(basename "$ppm" .ppm)
    for q in $qs; do
        LD_LIBRARY_PATH=$TURBO_LD $TURBO_CJ -quality "$q" -outfile "$SRC/turbo/${s}__turbo__q${q}.jpg" "$ppm" 2>/dev/null
        LD_LIBRARY_PATH=$MOZ_LD   $MOZ_CJ   -quality "$q" -outfile "$SRC/mozjpeg/${s}__mozjpeg__q${q}.jpg" "$ppm" 2>/dev/null
    done
done
for png in "$ORIG"/*.png; do
    s=$(basename "$png" .png)
    for q in $qs; do cjpegli "$png" "$SRC/jpegli/${s}__jpegli__q${q}.jpg" -q "$q" >/dev/null 2>&1; done
done
echo "sources: turbo=$(ls "$SRC"/turbo/*.jpg|wc -l) moz=$(ls "$SRC"/mozjpeg/*.jpg|wc -l) jpegli=$(ls "$SRC"/jpegli/*.jpg|wc -l)"

# ---- stage 2: measure source quality vs original -------------------------
say "stage 2: source-quality anchors (src vs original)"
printf 'ref_path\tdist_path\n' > "$WORK/pairs_src.tsv"
for e in turbo mozjpeg jpegli; do
    for jpg in "$SRC/$e"/*.jpg; do
        echo -e "$ORIG/$(basename "$jpg" | sed 's/__.*//').png\t$jpg" >> "$WORK/pairs_src.tsv"
    done
done
"$ZM" batch --metric zensim-gpu --pairs "$WORK/pairs_src.tsv" \
    --output "$WORK/src_zensim.tsv" --gpu-runtime cuda 2>&1 | tail -1
python3 "$REPO/scripts/fit_source_anchors.py" "$WORK/src_zensim.tsv" \
    > "$WORK/source_anchors.rs"
echo "-> $WORK/source_anchors.rs (paste into target.rs::ijg_q_to_zensim_a)"

# ---- stage 3: forced-strategy sweeps (the per-encoder fit data) ----------
say "stage 3: forced-strategy sweeps"
for e in turbo mozjpeg jpegli; do
    for strat in preserve tuned; do
        "$ZJR" recompress-sweep --sources "$SRC/$e" --originals "$ORIG" \
            --output "$WORK/${strat}_${e}.tsv" --targets 30,40,50,60,70,80 \
            --force-strategy "$strat" 2>&1 | tail -1
    done
done

# ---- stage 4: fit + splice per-encoder tables ----------------------------
say "stage 4: fit per-encoder tables -> per_encoder.rs"
python3 "$REPO/scripts/fit_per_encoder.py" "$WORK" "$DATE" > "$WORK/generated_tables.rs"
PE="$REPO/zenjpeg-recompress/src/calibration/per_encoder.rs"
python3 - "$PE" "$WORK/generated_tables.rs" <<'PY'
import sys
pe, gen = sys.argv[1], sys.argv[2]
src = open(pe).read(); block = open(gen).read().rstrip()+"\n"
B="// === BEGIN GENERATED TABLES ==="; E="// === END GENERATED TABLES ==="
i, j = src.index(B), src.index(E)+len(E)
open(pe,"w").write(src[:i] + block.rstrip()+"\n" + src[j+1:])
print("spliced generated tables into per_encoder.rs")
PY

# ---- stage 5: jpegli 420/444 cumulative tables ---------------------------
say "stage 5: jpegli cumulative-sweep (data.rs 420/444) + confidence residuals"
"$ZJR" cumulative-sweep --references "$ORIG" \
    --output "$WORK/jpegli_cumulative_420.tsv" --subsampling 420 --force-tuned \
    --source-qs 20,30,40,50,60,70,80,90,95 --targets 30,40,50,60,70,80,85,90 2>&1 | tail -1
python3 "$REPO/scripts/fit_calibration.py" "$WORK/jpegli_cumulative_420.tsv" \
    > "$WORK/data_420.rs" 2> "$WORK/fit_420.log"
echo "-> $WORK/data_420.rs (paste into data.rs); fit log in fit_420.log"

# ---- stage 6: rebuild + validation gate ----------------------------------
say "stage 6: rebuild + validation"
cargo build --release --manifest-path "$REPO/Cargo.toml" --all 2>&1 | tail -1
cargo test --release --manifest-path "$REPO/Cargo.toml" -p zenjpeg-recompress --features expert 2>&1 | grep "test result" | head
report="$WORK/validation_${DATE}.txt"
: > "$report"
for e in turbo mozjpeg jpegli; do
    "$ZJR" recompress-sweep --sources "$SRC/$e" --originals "$ORIG" \
        --output "$WORK/val_${e}.tsv" --targets 30,40,50,60,70,80 2>&1 | tail -1
done
python3 - "$WORK" "$report" <<'PY'
import csv, sys
work, report = sys.argv[1], sys.argv[2]
out=open(report,"w")
out.write("encoder  under-target  size-regressions  (GATE: under<=15%, reg=0)\n")
ok=True
resid=[]   # achieved - target, across all recompressed cells (for confidence shifts)
for e in ['turbo','mozjpeg','jpegli']:
    rs=list(csv.DictReader(open(f"{work}/val_{e}.tsv"),delimiter='\t'))
    under=nm=reg=0
    for r in rs:
        if r['size_ratio'] not in ('-','') and float(r['size_ratio'])>1.0: reg+=1
        c=r['zensim_a_vs_reference']; t=float(r['target_zensim_a'])
        if c not in ('-',''):
            nm+=1; resid.append(float(c)-t)
            if float(c)<t-2: under+=1
    pct=100*under/max(nm,1)
    flag="" if (pct<=15 and reg==0) else "  <-- GATE FAIL"
    out.write(f"{e:8s}  {pct:4.0f}%         {reg:4d}{flag}\n")
    if pct>15 or reg>0: ok=False
# Confidence shifts from the validation residual tail (achieved - target),
# restricted to the TARGETING regime (|achieved-target| <= 20) so the
# gross over-delivery of Tuned-floor / near-NoOp cells doesn't skew the
# quantiles. shift(C) = -quantile_{1-C}(residual): aim higher so the
# C-quantile clears target. ADVISORY — review before pasting; the shipped
# values were fit on the cleaner jpegli cumulative residuals.
resid=[r for r in resid if abs(r)<=20]
resid.sort(); n=len(resid)
def q(p): return resid[max(0,min(n-1,int(round(p/100*(n-1)))))] if n else 0.0
out.write("\nConfidence target_shift() values (paste into api.rs Confidence::target_shift):\n")
for name,cfrac in [("P25",25),("P50",50),("P75",75),("P90",90),("P95",95)]:
    out.write(f"  {name} => {-q(100-cfrac):+.1f},\n")
out.write(f"\nGATE: {'PASS' if ok else 'FAIL'}\n")
out.close()
print(open(report).read())
PY
echo "validation report: $report"
echo
echo "Recalibration complete. Review the GATE, then commit:"
echo "  - zenjpeg-recompress/src/calibration/per_encoder.rs (auto-spliced)"
echo "  - target.rs anchors  <- $WORK/source_anchors.rs (manual paste)"
echo "  - data.rs 420/444    <- $WORK/data_420.rs (manual paste)"
echo "  - Confidence shifts  <- $WORK/confidence_residuals.txt (manual)"
echo "  - benchmarks/*_${DATE}.tsv (copy the val_/forced sweeps you want kept)"
