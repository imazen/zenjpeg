#!/usr/bin/env bash
# Generate a multi-encoder source-JPEG corpus for validating
# zenjpeg-recompress calibration on REAL encoder output (not zenjpeg's
# internal synthesis).
#
# Encoders:
#   - libjpeg-turbo : host /usr/bin/cjpeg            (PPM in)
#   - mozjpeg 4.1.5 : docker `ati-mozjpeg` image     (PPM in)
#                     built via: cd ~/work/all-the-images &&
#                     docker build --target mozjpeg -t ati-mozjpeg .
#   - jpegli        : host /usr/local/bin/cjpegli    (PNG in)
#
# Naming: <refstem>__<encoder>__q<Q>.jpg  so zjr-calibrate
# recompress-sweep can match each source to its <refstem>.png original.
#
# Usage: gen_multiencoder_sources.sh <originals_png_dir> <out_dir> [qstep]
set -euo pipefail

ORIG="${1:?originals png dir}"
OUT="${2:?output dir}"
QSTEP="${3:-5}"          # granular q step; 5 = 20,25,...,95
QMIN=20
QMAX=95

mkdir -p "$OUT"/{turbo,mozjpeg,jpegli,_ppm}

echo "Converting PNG -> PPM..."
for png in "$ORIG"/*.png; do
    stem=$(basename "$png" .png)
    convert "$png" "$OUT/_ppm/$stem.ppm"
done

qs=$(seq "$QMIN" "$QSTEP" "$QMAX" | tr '\n' ' ')

echo "Encoding libjpeg-turbo (host cjpeg)..."
for ppm in "$OUT"/_ppm/*.ppm; do
    stem=$(basename "$ppm" .ppm)
    for q in $qs; do
        /usr/bin/cjpeg -quality "$q" -outfile "$OUT/turbo/${stem}__turbo__q${q}.jpg" "$ppm"
    done
done

echo "Encoding jpegli (host cjpegli)..."
for png in "$ORIG"/*.png; do
    stem=$(basename "$png" .png)
    for q in $qs; do
        cjpegli "$png" "$OUT/jpegli/${stem}__jpegli__q${q}.jpg" -q "$q" >/dev/null 2>&1
    done
done

echo "Encoding mozjpeg 4.1.5 (docker ati-mozjpeg)..."
# One container, loop inside. Mount PPMs ro + output dir.
docker run --rm \
    -v "$OUT/_ppm:/ppm:ro" \
    -v "$OUT/mozjpeg:/moz" \
    ati-mozjpeg \
    sh -c "
        CJ=/opt/mozjpeg-4.1.5/bin/cjpeg
        for ppm in /ppm/*.ppm; do
            stem=\$(basename \"\$ppm\" .ppm)
            for q in $qs; do
                \$CJ -quality \$q -outfile /moz/\${stem}__mozjpeg__q\${q}.jpg \"\$ppm\"
            done
        done
    "

echo "Done. Counts:"
for e in turbo mozjpeg jpegli; do
    printf '  %-8s %d jpegs\n' "$e" "$(ls "$OUT/$e"/*.jpg 2>/dev/null | wc -l)"
done
