# C API readiness

Verified against the implementation on 2026-10-27. No libjpeg or TurboJPEG C ABI
is exported by this repository. The native codec remains `forbid(unsafe_code)`.

## Suspension

The opt-in controlled decoder in PR 240 yields for source requests,
interpretation decisions and row delivery. It reads the complete encoded source
into admitted storage before inspecting it and constructing the native decoder.
Final interpretation adoption precedes native pixel work, including progressive
construction. This is policy suspension, not resumable marker/entropy decoding
on incomplete input.

The ordinary scanline reader retains progress between row reads, but receives
an encoded slice whose contents are already available. Neither path implements
the libjpeg source-manager contract of suspending an in-progress JPEG operation
when input runs out and resuming that same operation after refill.

## Compatibility work remaining

A separate FFI crate can keep raw-pointer and C ABI handling outside the safe
codec. That is a design option, not an implemented layer. A drop-in libjpeg ABI
requires the selected ABI version's public structure layout, symbols, lifecycle,
source/destination and memory manager callbacks, error/warning behavior and
supported extension semantics. A TurboJPEG-style opaque-handle API is a different
compatibility surface and needs its own contract tests.

Resumable decoding needs explicit saved marker, bitstream, predictor, restart,
MCU and progressive-scan state, including partial reads and output backpressure.
C callback errors must be contained at the boundary; a caller's longjmp must not
cross Rust frames holding live values. Cancellation via Pulse is not suspension.

The existing row interfaces, native component/precision decoding, coefficient
transforms and oracle rasters supply useful building blocks. They do not prove
C API equivalence, all libjpeg output-mode parity, or 12-bit/lossless encoding.
Ordinary 8-bit XYB decoding exists, but controlled decoding and the new
native-precision/general-component path reject XYB. The profile-tagged byte
encoder qualification also requires YCbCr. These are distinct admission limits.

A compatibility claim needs an explicit symbol/behavior matrix and tests using
unmodified C callers against both libraries, including refill at every byte
boundary and exact qualified raster comparisons.

Reference contracts: [libjpeg API and I/O suspension](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/doc/libjpeg.txt)
and [public C declarations](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jpeglib.h).
