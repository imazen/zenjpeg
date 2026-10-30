//! Forbidding the ICC profile (`embed_icc: false`) never drops a colour the
//! encoder can identify or a colour context profile: JPEG's only colour carrier
//! is the ICC profile, so those encodes refuse, as policy refusals. One-shot
//! and row encodes; a refused row encode has written nothing.
//!
//! The deprecated best-effort setter keeps its earlier behaviour for job
//! metadata on pixels without a colour context: a profile zenpixels cannot
//! identify is dropped as the policy asks, unless the job's code points or the
//! pixels' own primaries would go with it. A checked policy refuses instead.
#![cfg(feature = "zencodec")]

use std::sync::Arc;

use zencodec::encode::{EncodeDeclaration, EncodeJob, EncodePolicy, Encoder, EncoderConfig};
use zencodec::{
    At, CodecError, ColorEmitPolicy, ErrorCategory, Metadata, MetadataPolicy, PolicyKind,
    UnsupportedOperation,
};
use zenjpeg::{JpegEncodeJob, JpegEncoderConfig};
use zenpixels::{Cicp, ColorContext, ColorModel, ColorPrimaries, PixelDescriptor, PixelSlice};
use zenpixels_convert::icc_profiles::DISPLAY_P3_V4;

const W: u32 = 16;
const H: u32 = 8;

#[allow(deprecated)] // Exercises the legacy metadata setter.
fn job(meta: &Metadata) -> JpegEncodeJob {
    JpegEncoderConfig::new()
        .job()
        .with_metadata_policy(meta.clone(), MetadataPolicy::PreserveExact)
}

#[allow(deprecated)] // The legacy setter is a public way to forbid ICC output.
fn legacy(job: JpegEncodeJob, policy: EncodePolicy) -> JpegEncodeJob {
    job.with_policy(policy)
}

fn encode(
    rows: bool,
    meta: &Metadata,
    policy: Option<EncodePolicy>,
) -> Result<Vec<u8>, At<CodecError>> {
    encode_as(rows, meta, policy, PixelDescriptor::RGB8_SRGB, None)
}

/// `policy` goes through the legacy setter.
fn encode_as(
    rows: bool,
    meta: &Metadata,
    policy: Option<EncodePolicy>,
    descriptor: PixelDescriptor,
    context: Option<Arc<ColorContext>>,
) -> Result<Vec<u8>, At<CodecError>> {
    let mut job = job(meta);
    if let Some(policy) = policy {
        job = legacy(job, policy);
    }
    run(job, rows, descriptor, context)
}

/// Encode the test pixels in `descriptor`, carrying `context`. Row encodes
/// declare both before the header (`begin` declares no context).
fn run(
    job: JpegEncodeJob,
    rows: bool,
    descriptor: PixelDescriptor,
    context: Option<Arc<ColorContext>>,
) -> Result<Vec<u8>, At<CodecError>> {
    let data: Vec<u8> = (0..W * H * 3).map(|i| (i * 13 % 251) as u8).collect();
    let mut pixels = PixelSlice::new(&data, W, H, W as usize * 3, descriptor).unwrap();
    if let Some(context) = &context {
        pixels = pixels.with_color_context(Arc::clone(context));
    }
    let mut encoder = job.encoder()?;
    if !rows {
        return Ok(encoder.encode(pixels)?.into_vec());
    }
    let mut declaration = EncodeDeclaration::new(W, H, descriptor);
    if let Some(context) = &context {
        declaration = declaration.with_color(context);
    }
    if let Err(error) = encoder.begin_declared(&declaration) {
        assert!(
            encoder.pending_output().is_empty(),
            "a refused row encode wrote bytes"
        );
        return Err(error);
    }
    encoder.push_rows(pixels)?;
    Ok(encoder.finish()?.into_vec())
}

/// The ICC profile in the APP2 segments, as a test-local parser reads it.
fn icc(jpeg: &[u8]) -> Option<Vec<u8>> {
    let mut out = None::<Vec<u8>>;
    let mut pos = 2;
    while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF && jpeg[pos + 1] != 0xDA {
        let len = usize::from(u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]));
        let data = &jpeg[pos + 4..pos + 2 + len];
        if jpeg[pos + 1] == 0xE2 && data.starts_with(b"ICC_PROFILE\0") {
            out.get_or_insert_with(Vec::new)
                .extend_from_slice(&data[14..]);
        }
        pos += 2 + len;
    }
    out
}

/// An RGB profile zenpixels cannot identify (identification hashes the bytes
/// after the header): Display P3 with its last byte changed.
fn unidentified() -> Vec<u8> {
    let mut profile = DISPLAY_P3_V4.to_vec();
    let last = profile.len() - 1;
    profile[last] ^= 1;
    assert!(
        zenpixels::ColorProfileSource::Icc(&profile)
            .resolve()
            .is_none()
    );
    profile
}

fn with(profile: &[u8], at: usize, bytes: &[u8]) -> Vec<u8> {
    let mut out = profile.to_vec();
    out[at..at + bytes.len()].copy_from_slice(bytes);
    out
}

/// Bytes lcms 2.17 refused to open as a profile (`transicc -i`, 2026-10-28:
/// `benchmarks/icc_fixture_lcms_probe_2026-10-28.txt`), none of which
/// zenpixels identifies, first the established unit test's.
fn lcms_2_17_rejects() -> Vec<(&'static str, Vec<u8>)> {
    let p3 = DISPLAY_P3_V4;
    let truncated = p3[..100].to_vec();
    // The header's colour-space field alone does not make a profile.
    assert_eq!(
        zenpixels::icc::profile_color_space(&truncated),
        Some(ColorModel::Rgb)
    );
    vec![
        (
            "the established unit test's bytes",
            b"fake icc profile data".to_vec(),
        ),
        ("a truncated header", truncated),
        ("a header without its tag count", p3[..131].to_vec()),
        ("no acsp signature", with(p3, 36, b"xxxx")),
        (
            "a tag table past the end",
            with(p3, 128, &30u32.to_be_bytes()),
        ),
        (
            "the largest tag count",
            with(p3, 128, &u32::MAX.to_be_bytes()),
        ),
    ]
}

/// Profiles zenpixels cannot identify: the unidentified one, then the bytes
/// lcms 2.17 rejected.
fn not_identified() -> Vec<(&'static str, Vec<u8>)> {
    let mut all = vec![("an unidentified profile", unidentified())];
    all.extend(lcms_2_17_rejects());
    all
}

fn forbid() -> Option<EncodePolicy> {
    Some(EncodePolicy::none().with_embed_icc(false))
}

fn assert_policy_refusal(result: Result<Vec<u8>, At<CodecError>>, what: &str) {
    match result {
        Ok(file) => panic!(
            "{what}: wrote a file, ICC {:?}",
            icc(&file).map(|p| p.len())
        ),
        Err(error) => assert_eq!(
            error.error().category(),
            ErrorCategory::Policy(PolicyKind::Encode),
            "{what}: {error}"
        ),
    }
}

#[test]
fn forbidding_the_profile_of_wide_gamut_pixels_is_a_policy_refusal() {
    let meta = Metadata::none().with_icc(DISPLAY_P3_V4);
    for rows in [false, true] {
        let error = encode(rows, &meta, forbid()).unwrap_err();
        assert_eq!(
            error.error().category(),
            ErrorCategory::Policy(PolicyKind::Encode),
            "rows {rows}: {error}"
        );
    }
}

#[test]
fn the_legacy_setter_drops_a_job_profile_it_cannot_identify() {
    for (what, bytes) in not_identified() {
        let meta = Metadata::none().with_icc(bytes);
        for policy in [forbid(), Some(EncodePolicy::strip_all())] {
            for rows in [false, true] {
                // Best effort, as before: no identified colour goes with it.
                let file = encode(rows, &meta, policy)
                    .unwrap_or_else(|error| panic!("{what}, rows {rows}: {error}"));
                let absent = encode(rows, &Metadata::none(), None).unwrap();
                assert_eq!(file, absent, "{what}, rows {rows}");
            }
        }
    }
    let meta = Metadata::none().with_icc(unidentified());
    for rows in [false, true] {
        let file = encode(rows, &meta, None).unwrap();
        assert_eq!(icc(&file), Some(unidentified()), "rows {rows}");
    }
}

#[test]
fn the_legacy_setter_keeps_a_colour_it_can_identify() {
    let display_p3 = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    for (what, bytes) in not_identified() {
        for rows in [false, true] {
            // Display P3 pixels keep needing a carrier once the profile goes.
            let meta = Metadata::none().with_icc(bytes.clone());
            assert_policy_refusal(
                encode_as(rows, &meta, forbid(), display_p3, None),
                &format!("{what} over Display P3 pixels, rows {rows}"),
            );
            // So do the Display P3 code points beside it.
            let meta = meta.with_cicp(Cicp::DISPLAY_P3);
            assert_policy_refusal(
                encode(rows, &meta, forbid()),
                &format!("{what} with Display P3 code points, rows {rows}"),
            );
        }
    }
}

#[test]
fn forbidding_a_colour_context_profile_refuses() {
    let rgb8 = PixelDescriptor::RGB8_SRGB;
    for (what, bytes) in not_identified() {
        let context = Some(Arc::new(ColorContext::from_icc(bytes.clone())));
        for rows in [false, true] {
            // The context is the pixels' own, authoritative description.
            let alone = encode_as(rows, &Metadata::none(), forbid(), rgb8, context.clone());
            assert!(alone.is_err(), "{what}, rows {rows}: dropped the context");
            // Repeated as job metadata, it is still the context.
            let meta = Metadata::none().with_icc(bytes.clone());
            let both = encode_as(rows, &meta, forbid(), rgb8, context.clone());
            assert!(
                both.is_err(),
                "{what}, rows {rows}: dropped the context repeated as metadata"
            );
        }
    }
    // A complete RGB profile reaches the carrier decision: a policy refusal,
    // which planning reports too.
    let context = Arc::new(ColorContext::from_icc(unidentified()));
    for rows in [false, true] {
        assert_policy_refusal(
            encode_as(
                rows,
                &Metadata::none(),
                forbid(),
                rgb8,
                Some(Arc::clone(&context)),
            ),
            &format!("unidentified context profile, rows {rows}"),
        );
    }
    let job = legacy(job(&Metadata::none()), forbid().unwrap());
    let declaration = EncodeDeclaration::new(W, H, rgb8).with_color(&context);
    assert!(job.plan_encode(&declaration).is_err());
}

#[test]
fn setter_order_decides_whether_the_legacy_behaviour_applies() {
    let meta = Metadata::none().with_icc(unidentified());
    let rgb8 = PixelDescriptor::RGB8_SRGB;
    let strip = EncodePolicy::strip_all();
    let verbatim = EncodePolicy::none().with_color(ColorEmitPolicy::Verbatim);
    let refused = EncodePolicy::none().with_strict(true);
    for rows in [false, true] {
        let absent = encode(rows, &Metadata::none(), None).unwrap();
        // A checked `none()` specifies nothing: the legacy policy stays.
        let mut checked = legacy(job(&meta), strip);
        checked.try_set_policy(EncodePolicy::none()).unwrap();
        let file = run(checked, rows, rgb8, None).unwrap();
        assert_eq!(file, absent, "legacy, then checked none(), rows {rows}");
        // A refused checked policy leaves the job unchanged.
        let mut checked = legacy(job(&meta), strip);
        assert_eq!(
            checked.try_set_policy(refused),
            Err(UnsupportedOperation::EncodePolicy)
        );
        let file = run(checked, rows, rgb8, None).unwrap();
        assert_eq!(
            file, absent,
            "legacy, then a refused checked policy, rows {rows}"
        );
        // An admitted checked policy replaces it: Verbatim writes the profile.
        let mut checked = legacy(job(&meta), strip);
        checked.try_set_policy(verbatim).unwrap();
        let file = run(checked, rows, rgb8, None).unwrap();
        assert_eq!(
            icc(&file),
            Some(unidentified()),
            "legacy, then checked Verbatim, rows {rows}"
        );
        // The legacy setter after a checked policy replaces it in turn.
        let mut checked = job(&meta);
        checked.try_set_policy(verbatim).unwrap();
        let file = run(legacy(checked, strip), rows, rgb8, None).unwrap();
        assert_eq!(file, absent, "checked Verbatim, then legacy, rows {rows}");
    }
}

#[test]
fn nothing_to_forbid_changes_nothing() {
    let meta = Metadata::none().with_exif(b"II*\0\x08\0\0\0\0\0".to_vec());
    for rows in [false, true] {
        let forbidden = encode(rows, &meta, forbid()).unwrap();
        assert_eq!(forbidden, encode(rows, &meta, None).unwrap(), "rows {rows}");
        assert!(icc(&forbidden).is_none(), "rows {rows}");
    }
}
