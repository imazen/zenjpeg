# Checked JPEG metadata and native progress

The zencodec JPEG encode job accepts checked metadata with
`try_set_metadata_policy` or `try_with_metadata_policy`. Success retains the
policy-filtered record as an obligation. An explicit later checked replacement
can replace or clear it; a later best-effort metadata change that conflicts with
it is refused before encoder construction. A conflicting emission policy or
color declaration is refused before affected output is exposed.

JPEG currently refuses a checked request when the retained record contains
CICP, content-light level, mastering display or diffuse white: an ICC profile
is not a substitute for those typed carriers. Checked requests also refuse
preconfigured native segments, malformed or APP1-prefixed EXIF, an EXIF blob
that cannot carry its required nonidentity orientation, empty carrier blobs,
ICC over 255 JPEG chunks, and XMP outside the qualified nonempty UTF-8
single-marker path. These are conservative limits; the best-effort APIs have
different behavior. JPEG does not bake orientation merely to retain metadata.

A required ICC is emitted byte-for-byte. When the caller has not selected a
color emission policy, checked ICC retention uses Verbatim, including for a
redundant sRGB profile. An explicitly selected policy that would drop it is
refused. Checks apply to the retained job and later declared pixels, rather
than to static capability flags.

`DecodeConfig::decode_with_pulse` reports completed scan-work batches and
output rows in separate unknown-total phases when the supplied pulse supports
planning. Baseline row batches and progressive completed scans are distinct
units of completed work; scan batches are not final output rows. A pulse that
cannot plan receives output-row reports only, with a diagnostic issue. Native
precision and orientation use an unpartitioned pulse: their internal scan work
is not reported separately. The caller owns the supplied pulse's completion.

## Qualification and limits

On an aarch64 Linux host, source `92ce25dd` with zencodec `5493eaaa` and
zenpixels `f23ac753` passed formatting, strict Clippy for the library and four
named test targets (`inventory,parallel`), and those test targets: checked
metadata 13/13, cancellation 2/2, native hardening 19/19, and Pulse 9/9. The
hardening run included replay of 47 in-tree native seeds; it did not replay the
full external fuzz corpus. Its test wrapper reported peak RSS 0.83 GiB.

Source `7579c288` differs from `92ce25dd` only in the checked metadata test.
It passed that test 13/13, formatting and the same strict Clippy target set;
its focused test wrapper reported peak RSS 0.80 GiB. The added controls inspect
standard XMP APP1 bytes independently with img-parts, and compare decoded
samples with jpeg-decoder against an identical-config legacy Verbatim encode
for the exercised Adobe RGB and sRGB one-shot and row paths. The unchanged
progress, cancellation and hardening tests were not rerun on `7579c288`.

With zencodec pinned to `61fb96f1`, source `67d5c27b` passed the checked
metadata test 13/13 after a package clean. Its later all-target strict Clippy
failed on an example lint; subsequent attempts exposed additional legacy-policy
warnings. Rebased source `faa3d261` passed strict Clippy for all zenjpeg targets
with `inventory,parallel` on the same aarch64 host. Its native library, test
assertions and lock retain the 67d content; the intervening changes are scoped
legacy deprecation allowances and two example lint repairs. The final Clippy
wrapper reported peak RSS 0.83 GiB. This final run did not rerun the tests, and
the earlier seed and progress results remain scoped to their named snapshots.

This evidence does not cover every JPEG mode or platform. Progressive scan
progress is currently per completed scan, not within one long scan. Native
precision and the separate scanline reader need more interruption and reporting
coverage, and parallel workers do not yet have individual diagnostic spans.
Reporting overhead has not been measured. The tested lock has a separate old
zenpixels instance for zenresize; its flexible dependency edges require a
portable release-constraint solution before the same graph can be assumed from
an unlocked downstream resolution.
