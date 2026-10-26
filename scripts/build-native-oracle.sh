#!/usr/bin/env bash
# Test-only, SHA-256-pinned libjpeg-turbo. No oracle is linked into zenjpeg.
set -euo pipefail
root=${1:?oracle build directory required}
mkdir -p "$root"
if [[ ! -f "$root/turbo.tar.gz" ]]; then
  curl -fL --retry 3 https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/libjpeg-turbo-3.2.0.tar.gz -o "$root/turbo.tar.gz"
fi
echo "6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e  $root/turbo.tar.gz" | sha256sum -c -
if [[ ! -d "$root/libjpeg-turbo-3.2.0" ]]; then
  tar -xzf "$root/turbo.tar.gz" -C "$root"
fi
cmake -S "$root/libjpeg-turbo-3.2.0" -B "$root/build" -DENABLE_SHARED=OFF -DWITH_SIMD=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" --target cjpeg-static djpeg-static jpegtran-static --parallel 2
cc -Wall -Wextra -Werror -I"$root/build" -I"$root/libjpeg-turbo-3.2.0/src" zenjpeg/tests/support/native_components.c "$root/build/libjpeg.a" -lm -o "$root/build/native-components-oracle"
