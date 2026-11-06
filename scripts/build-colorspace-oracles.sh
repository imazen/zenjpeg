#!/usr/bin/env bash
# Test-only IJG9 oracle required by colorspace_oracles; never linked into zenjpeg.
set -euo pipefail
root=${1:?oracle build directory required}
mkdir -p "$root"
root=$(realpath "$root")
archive="$root/jpegsrc.v9f.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 3 https://www.ijg.org/files/jpegsrc.v9f.tar.gz -o "$archive"
fi
echo "04705c110cb2469caa79fb71fba3d7bf834914706e9641a4589485c1f832565b  $archive" | sha256sum -c -
if [[ ! -d "$root/jpeg-9f" ]]; then
  tar -xzf "$archive" -C "$root"
fi
cd "$root/jpeg-9f"
./configure --disable-shared --enable-static
make -j2 djpeg
mkdir -p "$root/bin"
cp djpeg "$root/bin/djpeg"
"$root/bin/djpeg" -verbose -outfile "$root/probe.ppm" "$root/jpeg-9f/testorig.jpg"
# Pillow provides the independent libjpeg-turbo reader; require it explicitly.
/usr/bin/python3 -c 'from PIL import Image, features; assert features.check_feature("libjpeg_turbo"); print("Pillow JPEG", Image.core.jpeglib_version, "libjpeg-turbo", features.version_feature("libjpeg_turbo"))'
