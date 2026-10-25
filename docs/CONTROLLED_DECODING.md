# Controlled decoding

The default-off `controlled-decode` feature implements zencodec's mandatory
interpretation suspension. The codec gathers facts and yields; application
policy runs in `zencodec_io::PolicyDecoder`, before native pixel work starts.
This uses the reviewed zencodec API from [PR 168](https://github.com/imazen/zencodec/pull/168).

```rust,ignore
use zencodec::{CodecSet, operation::{DecodeRequest, PolicySet}, storage::StorageBudget};
use zencodec_io::{CollectSink, PolicyDecoder, drive_from_slice};
use zenjpeg::JpegDecoderConfig;

let codecs = CodecSet::new().with_decoder(JpegDecoderConfig::new());
let session = DecodeRequest::new().start_controlled(
    &codecs, StorageBudget::new(64 * 1024 * 1024, 4),
)?;
let mut session = PolicyDecoder::new(session, PolicySet::new());
let mut pixels = CollectSink::new();
drive_from_slice(&mut session, input, &mut pixels, None, 4096)?;
```

The same implementation is available from
`DecodeJob::dyn_controlled_decoder(preferred, storage)` and
`DynDecodeJob::into_controlled_decoder(preferred, storage)`. It does not fall back
to ordinary incremental decoding when a controlled route is unavailable.

## Qualified scope

The current route reads the complete encoded source into admitted storage,
reports a provisional header, inventories recognized primary-image metadata
through EOI, then requires final interpretation adoption. Only after adoption
may it construct the native scanline decoder, including progressive coefficient
work, and deliver rows. It is not incremental entropy decoding.

Qualification tests compare every byte with ordinary native decode for baseline
and progressive RGB8, RGBA8, BGRA8, RGBX8, BGRX8 and grayscale output, header ICC,
slice windows and range reads. Late EXIF orientation is resolved before adoption.
Late, incomplete or forbidden ICC extraction cannot silently become assumed
sRGB. The route exposes typed unavailability reasons and no adopt offer.

Float output currently refuses before pixel effects: strip reconstruction differs
from the one-shot float IDCT ([issue 239](https://github.com/imazen/zenjpeg/issues/239)).
CMYK/XYB, explicit crop/transform, implicit CMS, nondefault native output targets
and gain-map rendering also need separate qualification. Source inventory is
limited to the primary image's recognized carriers; it is not a privacy or nested
content completeness certificate.

## Storage and failure scope

The shared budget covers retained encoded input, the coroutine input arena and
range index, and coroutine row staging. Native coefficients, native row buffers,
transform workspaces, metadata caches and the destination sink are outside it.
Four admitted objects suffice for this route, but no byte allowance here claims
to bound total operation RAM. Progressive/native fallback configurations may
retain coefficients or pixels internally. This route is not evidence for
network backpressure eligibility.

Input, dimension and output-size limits are checked before affected pixel work.
Policy refusal, native failure, cancellation and callback unwind release admitted
storage even when a caller retains the failed session's decision report.

`try_set_policy` accepts qualified extraction/progressive permissions and rejects
explicit strictness or partial-image-success policies before mutation. Legacy
`with_policy` remains available with its existing behavior.

Run `just controlled-check` for native route/erasure/registry parity, legacy
float packing, admission, failure cleanup and the existing shared contracts.
