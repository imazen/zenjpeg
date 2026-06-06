#!/usr/bin/env bash
# Generate a JPEG conformance corpus exercising every feature axis relevant to
# lossless JPEG<->JXL (JBRD) transcoding, the brunsli-equivalent round-trip.
#
# Output: $OUT/*.jpg  +  $OUT/manifest.tsv
# manifest columns: file <TAB> tags(comma-sep) <TAB> expect(roundtrip|reject)
#   roundtrip = JPEG->JXL->JPEG must be BYTE-EXACT
#   reject    = encoder must cleanly Err (never silently emit wrong bytes);
#               these are JXL-JBRD format boundaries (cjxl refuses them too)
#
# Deterministic: fixed source images, fixed encoder flags. Re-runnable.
# Tools: cjpeg/jpegtran (libjpeg-turbo), convert (ImageMagick). exiftool optional.
set -uo pipefail

OUT="${1:-/mnt/v/output/jbrd-conformance/2026-06-06}"
mkdir -p "$OUT"
MAN="$OUT/manifest.tsv"
: > "$MAN"

SRC="$OUT/_src"; mkdir -p "$SRC"
# Two small textured RGB sources (content with real DCT energy; sizes chosen so
# subsampling edge MCUs are exercised: 96x64 is MCU-aligned for 2x2; 67x53 is not).
convert -size 96x64 plasma:fractal -modulate 100,80 "$SRC/a.ppm"
convert -size 67x53 plasma:fractal -colorspace sRGB -modulate 100,60 "$SRC/b.ppm"
# A near-flat source (stresses EOB-run / all-zero AC blocks in progressive).
convert -size 80x48 gradient:'#203040'-'#a0b0c0' "$SRC/c.ppm"

emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$MAN"; }

# ---- Baseline sequential, all standard subsamplings, on both sources ----
for s in a b c; do
  cjpeg -quality 85 -sample 1x1       -outfile "$OUT/base_${s}_444.jpg"  "$SRC/$s.ppm" && emit "base_${s}_444.jpg"  "baseline,ycbcr,8bit,s444"  roundtrip
  cjpeg -quality 85 -sample 2x2       -outfile "$OUT/base_${s}_420.jpg"  "$SRC/$s.ppm" && emit "base_${s}_420.jpg"  "baseline,ycbcr,8bit,s420"  roundtrip
  cjpeg -quality 85 -sample 2x1       -outfile "$OUT/base_${s}_422.jpg"  "$SRC/$s.ppm" && emit "base_${s}_422.jpg"  "baseline,ycbcr,8bit,s422"  roundtrip
  cjpeg -quality 85 -sample 1x2       -outfile "$OUT/base_${s}_440.jpg"  "$SRC/$s.ppm" && emit "base_${s}_440.jpg"  "baseline,ycbcr,8bit,s440"  roundtrip
done

# ---- Quality extremes (low-q stresses 16-bit DQT + saturated coeffs) ----
cjpeg -quality 5   -sample 2x2 -outfile "$OUT/base_a_q5_420.jpg"  "$SRC/a.ppm" && emit "base_a_q5_420.jpg"  "baseline,ycbcr,8bit,s420,lowq" roundtrip
cjpeg -quality 100 -sample 1x1 -outfile "$OUT/base_a_q100_444.jpg" "$SRC/a.ppm" && emit "base_a_q100_444.jpg" "baseline,ycbcr,8bit,s444,highq" roundtrip

# ---- Grayscale ----
cjpeg -quality 85 -grayscale -outfile "$OUT/gray_a.jpg" "$SRC/a.ppm" && emit "gray_a.jpg" "baseline,gray,8bit,1comp" roundtrip
cjpeg -quality 85 -grayscale -progressive -outfile "$OUT/gray_a_prog.jpg" "$SRC/a.ppm" && emit "gray_a_prog.jpg" "progressive,gray,8bit,1comp" roundtrip

# ---- RGB JPEG (no chroma subsampling, component IDs R,G,B / Adobe transform 0) ----
cjpeg -quality 90 -rgb -outfile "$OUT/rgb_a.jpg" "$SRC/a.ppm" && emit "rgb_a.jpg" "baseline,rgb,8bit,3comp,kNone" roundtrip

# ---- Progressive (multi-scan, spectral-selection + successive-approximation) ----
for s in a b c; do
  cjpeg -quality 85 -sample 2x2 -progressive -outfile "$OUT/prog_${s}_420.jpg" "$SRC/$s.ppm" && emit "prog_${s}_420.jpg" "progressive,ycbcr,8bit,s420,multiscan" roundtrip
  cjpeg -quality 85 -sample 1x1 -progressive -outfile "$OUT/prog_${s}_444.jpg" "$SRC/$s.ppm" && emit "prog_${s}_444.jpg" "progressive,ycbcr,8bit,s444,multiscan" roundtrip
done

# ---- Optimized Huffman (non-default tables, exercises DHT reconstruction) ----
cjpeg -quality 85 -sample 2x2 -optimize -outfile "$OUT/base_a_opt_420.jpg" "$SRC/a.ppm" && emit "base_a_opt_420.jpg" "baseline,ycbcr,8bit,s420,optimized-huff" roundtrip

# ---- Restart markers / DRI (rows and blocks) ----
cjpeg -quality 85 -sample 2x2 -restart 1   -outfile "$OUT/base_a_rstrow_420.jpg" "$SRC/a.ppm" && emit "base_a_rstrow_420.jpg" "baseline,ycbcr,8bit,s420,restart" roundtrip
cjpeg -quality 85 -sample 2x2 -restart 4B  -outfile "$OUT/base_a_rstblk_420.jpg" "$SRC/a.ppm" && emit "base_a_rstblk_420.jpg" "baseline,ycbcr,8bit,s420,restart" roundtrip
cjpeg -quality 85 -sample 1x1 -progressive -restart 2B -outfile "$OUT/prog_a_rst_444.jpg" "$SRC/a.ppm" && emit "prog_a_rst_444.jpg" "progressive,ycbcr,8bit,s444,restart,multiscan" roundtrip

# ---- Trailing data after EOI (must be preserved verbatim) ----
cp "$OUT/base_a_444.jpg" "$OUT/base_a_444_tail.jpg"
printf 'TRAILINGBYTES\x00\xff after EOI' >> "$OUT/base_a_444_tail.jpg"
emit "base_a_444_tail.jpg" "baseline,ycbcr,8bit,s444,tail-data" roundtrip

# ---- COM + APP metadata (comment marker; ICC profile if available) ----
convert "$SRC/a.ppm" -quality 85 -sampling-factor 2x2 -set comment "zenJBRDconformance" "$OUT/meta_a_com.jpg" && emit "meta_a_com.jpg" "baseline,ycbcr,8bit,s420,com-marker,metadata" roundtrip
# sRGB ICC embed (ImageMagick ships sRGB.icc); EXIF via exiftool if present.
if convert "$SRC/a.ppm" -quality 85 -sampling-factor 1x1 -profile /usr/share/color/icc/sRGB.icc "$OUT/meta_a_icc.jpg" 2>/dev/null && [ -s "$OUT/meta_a_icc.jpg" ]; then
  emit "meta_a_icc.jpg" "baseline,ycbcr,8bit,s444,icc,metadata" roundtrip
fi
if command -v exiftool >/dev/null 2>&1; then
  cp "$OUT/base_a_444.jpg" "$OUT/meta_a_exif.jpg"
  exiftool -overwrite_original -Artist="zen" -Copyright="zen" "$OUT/meta_a_exif.jpg" >/dev/null 2>&1 && emit "meta_a_exif.jpg" "baseline,ycbcr,8bit,s444,exif,metadata" roundtrip
fi

# ===================== FORMAT BOUNDARIES (expect: reject) =====================
# These exceed what the JXL-JBRD container can represent byte-exactly; cjxl
# refuses them too. The encoder MUST cleanly Err, never silently mis-encode.

# ---- Custom subsampling factors > 2 (4:1:1 and friends) ----
cjpeg -quality 85 -sample 4x1,1x1,1x1 -outfile "$OUT/cust_a_411.jpg"   "$SRC/a.ppm" && emit "cust_a_411.jpg"   "baseline,ycbcr,8bit,sampling-gt2,s411" reject
cjpeg -quality 85 -sample 4x2,1x1,1x1 -outfile "$OUT/cust_a_410.jpg"   "$SRC/a.ppm" && emit "cust_a_410.jpg"   "baseline,ycbcr,8bit,sampling-gt2,s410" reject
cjpeg -quality 85 -sample 3x1,1x1,1x1 -outfile "$OUT/cust_a_3x1.jpg"   "$SRC/a.ppm" && emit "cust_a_3x1.jpg"   "baseline,ycbcr,8bit,sampling-gt2,nonpow2" reject
cjpeg -quality 85 -sample 2x2,2x1,1x1 -outfile "$OUT/cust_a_asym.jpg"  "$SRC/a.ppm" && emit "cust_a_asym.jpg"  "baseline,ycbcr,8bit,sampling-asym" reject

# ---- Arithmetic coding (SOF9/SOF10) ----
cjpeg -quality 85 -sample 2x2 -arithmetic -outfile "$OUT/arith_a_420.jpg"            "$SRC/a.ppm" && emit "arith_a_420.jpg"      "arithmetic,ycbcr,8bit,s420" reject
cjpeg -quality 85 -sample 1x1 -arithmetic -progressive -outfile "$OUT/arith_a_prog.jpg" "$SRC/a.ppm" && emit "arith_a_prog.jpg"   "arithmetic,progressive,ycbcr,8bit,s444" reject

# ---- 4-component CMYK / YCCK ----
if convert "$SRC/a.ppm" -colorspace CMYK -quality 85 -sampling-factor 1x1 "$OUT/cmyk_a.jpg" 2>/dev/null && [ -s "$OUT/cmyk_a.jpg" ]; then
  emit "cmyk_a.jpg" "cmyk,4comp,8bit,s444" reject
fi

printf 'GENERATED %d fixtures into %s\n' "$(grep -c . "$MAN")" "$OUT"
column -t -s$'\t' "$MAN" 2>/dev/null || cat "$MAN"
