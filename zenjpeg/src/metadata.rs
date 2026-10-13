//! Bounded JPEG metadata rewriting with gain-map reference repair.
// Container planner migrated from zencodecs at aa21cba0.
use alloc::{borrow::Cow, vec::Vec};
use core::ops::Range;
use zencodec::metadata_rewrite::{Action, Error, Request};
use zenmeta::rewrite::{Context as ScrubRequest, PlanBuilder as Plan};

/// Plan a metadata rewrite without re-encoding image samples.
///
/// Requires `metadata-rewrite`. Indexed images are supported only for a primary
/// JPEG plus one gain map; ambiguous ISO/XMP parameters are rejected.
pub fn rewrite<'a>(request: &Request<'a>) -> Result<zencodec::metadata_rewrite::Plan<'a>, Error> {
    let context = ScrubRequest::new(request)?;
    if !request.input().starts_with(b"\xff\xd8") {
        return Err(Error::Malformed("JPEG signature"));
    }
    plan(&context)?.finish()
}

use crate::container::{MarkerKind, marker};

const XMP: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

fn component<'a>(req: &ScrubRequest<'a>, gain_map: bool) -> Result<Plan<'a>, Error> {
    let mut plan = Plan::new();
    let mut end = 0;
    let mut scan = false;
    let mut exif_seen = false;
    let mut xmp_seen = false;
    let mut icc_count = None;
    let mut icc_parts: [Option<&[u8]>; 256] = [None; 256];
    let mut icc_ranges = Vec::new();
    for span in marker::iter(req.input()) {
        req.check()?;
        if span.offset < end || req.input()[end..span.offset].iter().any(|b| *b != 0xff) {
            return Err(Error::Malformed("JPEG marker gap"));
        }
        if end != span.offset {
            plan.remove(req, end..span.offset, "JPEG fill")?;
        }
        end = span
            .offset
            .checked_add(span.length)
            .ok_or(Error::Malformed("JPEG segment length"))?;
        let range = span.offset..end;
        match span.kind {
            MarkerKind::Soi if span.offset == 0 => plan.keep(req, range, "JPEG SOI")?,
            MarkerKind::Sos => {
                scan = true;
                plan.keep(req, range, "JPEG scan")?;
            }
            MarkerKind::Eoi => {
                if !scan {
                    return Err(Error::Malformed("JPEG has no scan"));
                }
                if let Some(count) = icc_count
                    && (1..=count).any(|i| icc_parts[i as usize].is_none())
                {
                    return Err(Error::Malformed("incomplete ICC chunks"));
                }
                normalize_icc(req, &mut plan, &icc_parts, &icc_ranges)?;
                plan.keep(req, range, "JPEG EOI")?;
                if end < req.input().len() {
                    // An unindexed secondary JPEG may be a legacy gain map. Do
                    // not silently remove an image whose interpretation is unknown.
                    if req.input()[end..].windows(2).any(|w| w == b"\xff\xd8") {
                        return Err(Error::Unsupported("unindexed secondary JPEG"));
                    }
                    plan.remove(req, end..req.input().len(), "JPEG trailer")?;
                }
                return Ok(plan);
            }
            MarkerKind::App(1) if span.payload.starts_with(b"Exif\0\0") => {
                if exif_seen {
                    return Err(Error::Malformed("duplicate JPEG EXIF"));
                }
                exif_seen = true;
                let bytes = req.exif(span.payload)?;
                let bytes = segment(0xe1, &bytes)?;
                plan.record(req, range, "EXIF", Action::Rewrite, Some(Cow::Owned(bytes)))?;
            }
            MarkerKind::App(1) if span.payload.starts_with(XMP) => {
                if xmp_seen {
                    return Err(Error::Malformed("duplicate JPEG XMP"));
                }
                xmp_seen = true;
                if let Some(xml) = req.xmp(&span.payload[XMP.len()..])? {
                    let payload = [XMP, &xml].concat();
                    plan.record(
                        req,
                        range,
                        "XMP",
                        Action::Rewrite,
                        Some(Cow::Owned(segment(0xe1, &payload)?)),
                    )?;
                } else {
                    plan.remove(req, range, "XMP")?;
                }
            }
            MarkerKind::App(1)
                if span
                    .payload
                    .starts_with(b"http://ns.adobe.com/xmp/extension/\0") =>
            {
                if req.edit_xmp() {
                    return Err(Error::Unsupported(
                        "extended XMP editing requires packet reassembly",
                    ));
                }
                plan.remove(req, range, "extended XMP")?;
            }
            MarkerKind::App(2) if span.payload.starts_with(b"ICC_PROFILE\0") => {
                let seq = *span.payload.get(12).ok_or(Error::Malformed("ICC chunk"))?;
                let count = *span.payload.get(13).ok_or(Error::Malformed("ICC chunk"))?;
                if seq == 0
                    || count == 0
                    || seq > count
                    || icc_parts[seq as usize].is_some()
                    || icc_count.is_some_and(|c| c != count)
                {
                    return Err(Error::Malformed("duplicate/inconsistent ICC chunks"));
                }
                icc_count = Some(count);
                icc_parts[seq as usize] = Some(&span.payload[14..]);
                icc_ranges.push(range.clone());
                if !req.normalize_icc() {
                    req.icc(&span.payload[14..])?;
                }
                plan.record(
                    req,
                    range.clone(),
                    "ICC",
                    Action::OpaqueIcc,
                    Some(Cow::Borrowed(&req.input()[range])),
                )?;
            }
            MarkerKind::App(2)
                if span.payload.starts_with(b"MPF\0")
                    || span.payload.starts_with(zencodec::ISO_21496_1_URN) =>
            {
                if !gain_map {
                    return Err(Error::Unsupported(
                        "multi-image/gain-map JPEG requires validated reassembly",
                    ));
                }
                plan.remove(req, range, "old gain-map signaling")?;
            }
            MarkerKind::App(0) if span.payload.starts_with(b"JFIF\0") => {
                if span.payload.len() < 14 {
                    return Err(Error::Malformed("JFIF header"));
                }
                let thumbnails = span.payload[12] as usize * span.payload[13] as usize * 3;
                if span.payload.len() != 14 + thumbnails {
                    return Err(Error::Malformed("JFIF thumbnail"));
                }
                let mut body = span.payload[..14].to_vec();
                body[12..14].fill(0);
                plan.record(
                    req,
                    range,
                    "JFIF (thumbnail removed)",
                    Action::Rewrite,
                    Some(Cow::Owned(segment(0xe0, &body)?)),
                )?;
            }
            MarkerKind::App(14)
                if span.payload.starts_with(b"Adobe") && span.payload.len() == 12 =>
            {
                plan.keep(req, range, "Adobe color transform")?;
            }
            MarkerKind::App(_) | MarkerKind::Com => {
                plan.remove(req, range, "private JPEG marker")?
            }
            MarkerKind::Sof(_)
            | MarkerKind::Dqt
            | MarkerKind::Dht
            | MarkerKind::Dri
            | MarkerKind::Dnl => {
                plan.keep(req, range, "JPEG coding")?;
            }
            _ => return Err(Error::Unsupported("JPEG marker kind")),
        }
    }
    Err(Error::Malformed("missing JPEG EOI"))
}

fn segment(marker: u8, payload: &[u8]) -> Result<Vec<u8>, Error> {
    let len = payload
        .len()
        .checked_add(2)
        .and_then(|n| u16::try_from(n).ok())
        .ok_or(Error::Limit(zencodec::LimitKind::OutputSize))?;
    let mut out = Vec::new();
    out.try_reserve_exact(payload.len() + 4)
        .map_err(|_| Error::Allocation)?;
    out.extend_from_slice(&[0xff, marker]);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

fn plan<'a>(req: &ScrubRequest<'a>) -> Result<Plan<'a>, Error> {
    let mut mpf = false;
    for (i, span) in marker::iter(req.input()).enumerate() {
        req.check()?;
        if i >= req.max_carriers() {
            return Err(Error::Limit(zencodec::LimitKind::MetadataCarriers));
        }
        if span.kind == MarkerKind::App(2) && span.payload.starts_with(b"MPF\0") {
            if mpf {
                return Err(Error::Malformed("duplicate MPF index"));
            }
            mpf = true;
        }
    }
    if !mpf {
        return component(req, false);
    }
    gain_map(req)
}

fn gain_map<'a>(req: &ScrubRequest<'a>) -> Result<Plan<'a>, Error> {
    use crate::container::{MpImageType, mpf, xmp};
    use zencodec::gainmap::{Iso21496Format, parse_iso21496_fmt, serialize_iso21496_fmt};
    let entries = mpf::parse_mpf(req.input()).map_err(|_| Error::Malformed("MPF index"))?;
    if entries.len() != 2
        || entries[0].offset != 0
        || entries[1].image_type != MpImageType::Undefined
    {
        return Err(Error::Unsupported(
            "MPF layout other than primary plus one gain map",
        ));
    }
    let base_end = entries[0].size;
    let gain_start = entries[1].offset;
    let gain_end = gain_start
        .checked_add(entries[1].size)
        .filter(|&n| n <= req.input().len())
        .ok_or(Error::Malformed("MPF gain-map bounds"))?;
    if base_end > gain_start || base_end < 2 || gain_start >= gain_end {
        return Err(Error::Malformed("MPF image overlap"));
    }
    let base = req
        .input()
        .get(..base_end)
        .ok_or(Error::Malformed("MPF primary bounds"))?;
    let gain = &req.input()[gain_start..gain_end];
    for image in [base, gain] {
        if marker::primary_bounds(image) != Some(0..image.len()) {
            return Err(Error::Malformed("MPF size disagrees with JPEG bounds"));
        }
    }
    let base_signals = signals(req, base)?;
    let gain_signals = signals(req, gain)?;
    if base_signals
        .iso
        .is_some_and(|iso| iso != zencodec::ISO_21496_1_PRIMARY_APP2_BODY)
    {
        return Err(Error::Malformed("primary ISO gain-map version marker"));
    }
    let params = if let Some(iso) = gain_signals.iso {
        // Present ISO is authoritative. A broken ISO declaration never falls
        // back to XMP or default headroom.
        parse_iso21496_fmt(iso, Iso21496Format::JpegApp2BodyWithUrn)
            .map_err(|_| Error::Malformed("ISO gain-map parameters"))?
    } else {
        let xml = gain_signals
            .xmp
            .or(base_signals.xmp)
            .ok_or(Error::Unsupported(
                "MPF image has no supported gain-map interpretation",
            ))?;
        xmp::parse_xmp(xml)
            .map_err(|_| Error::Unsupported("gain-map XMP interpretation"))?
            .0
    };
    params
        .validate()
        .map_err(|_| Error::Malformed("gain-map values"))?;
    if params.backward_direction || !params.use_base_color_space {
        return Err(Error::Unsupported(
            "legacy JPEG XMP cannot represent this gain-map direction/color contract",
        ));
    }
    // Preserve interpretation for ISO-aware and legacy XMP readers. Refuse
    // disagreeing declarations rather than silently choosing a different curve.
    for xml in [base_signals.xmp, gain_signals.xmp].into_iter().flatten() {
        let packet =
            zenmeta::xmp::Packet::parse(xml).map_err(|_| Error::Malformed("gain-map XMP XML"))?;
        if packet
            .property(xmp::HDRGM_NAMESPACE, "GainMapMax")
            .map_err(|_| Error::Malformed("gain-map XMP property"))?
            .is_some()
        {
            let other = xmp::parse_xmp(xml)
                .map_err(|_| Error::Malformed("gain-map XMP parameters"))?
                .0;
            if !same_params(&params, &other) {
                return Err(Error::Unsupported("ISO/XMP gain-map parameters disagree"));
            }
        }
    }
    if let Some(xml) = base_signals.xmp {
        let packet =
            zenmeta::xmp::Packet::parse(xml).map_err(|_| Error::Malformed("primary XMP XML"))?;
        let directory = packet
            .resource_sequence(
                xmp::CONTAINER_NAMESPACE,
                "Directory",
                xmp::CONTAINER_NAMESPACE,
                "Item",
            )
            .map_err(|_| Error::Malformed("gain-map directory"))?;
        for item in directory {
            if item.iter().any(|p| {
                p.namespace() == xmp::ITEM_NAMESPACE
                    && p.name() == "Semantic"
                    && p.value() == "GainMap"
            }) {
                let length = item
                    .iter()
                    .find(|p| p.namespace() == xmp::ITEM_NAMESPACE && p.name() == "Length")
                    .ok_or(Error::Malformed("gain-map directory length"))?
                    .value()
                    .parse::<usize>()
                    .map_err(|_| Error::Malformed("gain-map directory length"))?;
                if length != gain.len() {
                    return Err(Error::Malformed("gain-map directory/MPF length conflict"));
                }
            }
        }
    }
    let base_req = req.clone().with_input(base).with_xmp_editing(false);
    let gain_req = req.clone().with_input(gain).with_xmp_editing(false);
    let mut base_plan = component(&base_req, true)?;
    let mut gain_plan = component(&gain_req, true)?;
    let iso = if let Some(raw) = gain_signals.iso {
        raw.to_vec()
    } else {
        serialize_iso21496_fmt(&params, Iso21496Format::JpegApp2BodyWithUrn)
    };
    inject(
        &mut gain_plan,
        &gain_req,
        segment(0xe2, &iso)?,
        "regenerated ISO gain-map envelope",
    )?;
    inject(
        &mut gain_plan,
        &gain_req,
        edited_gain_xmp(
            req,
            gain_signals.xmp,
            &xmp::generate_gainmap_xmp(&params),
            Some(&params),
        )?,
        "regenerated gain-map XMP",
    )?;
    inject(
        &mut base_plan,
        &base_req,
        segment(0xe2, zencodec::ISO_21496_1_PRIMARY_APP2_BODY)?,
        "regenerated primary ISO signal",
    )?;
    inject(
        &mut base_plan,
        &base_req,
        edited_gain_xmp(
            req,
            base_signals.xmp,
            &xmp::generate_primary_xmp(gain_plan.output_len()),
            None,
        )?,
        "regenerated gain-map directory",
    )?;
    let dummy = mpf::create_mpf_header_typed(
        0,
        &[(MpImageType::Undefined, gain_plan.output_len())],
        Some(2),
    );
    let final_base = base_plan
        .output_len()
        .checked_add(dummy.len())
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(Error::Limit(zencodec::LimitKind::OutputSize))?;
    if gain_plan.output_len() > u32::MAX as usize {
        return Err(Error::Limit(zencodec::LimitKind::OutputSize));
    }
    inject(
        &mut base_plan,
        &base_req,
        mpf::create_mpf_header_typed(
            final_base,
            &[(MpImageType::Undefined, gain_plan.output_len())],
            Some(2),
        ),
        "regenerated MPF offsets",
    )?;
    let mut output = Plan::new();
    output.append(req, base_plan, 0)?;
    if base_end < gain_start {
        output.remove(req, base_end..gain_start, "inter-image padding")?;
    }
    output.append(req, gain_plan, gain_start)?;
    if gain_end < req.input().len() {
        output.remove(req, gain_end..req.input().len(), "JPEG trailer")?;
    }
    Ok(output)
}

struct Signals<'a> {
    iso: Option<&'a [u8]>,
    xmp: Option<&'a str>,
}
fn signals<'a>(req: &ScrubRequest<'_>, image: &'a [u8]) -> Result<Signals<'a>, Error> {
    let mut s = Signals {
        iso: None,
        xmp: None,
    };
    for (i, span) in marker::iter(image).enumerate() {
        req.check()?;
        if i >= req.max_carriers() {
            return Err(Error::Limit(zencodec::LimitKind::MetadataCarriers));
        }
        if span.kind == MarkerKind::App(2)
            && span.payload.starts_with(zencodec::ISO_21496_1_URN)
            && s.iso.replace(span.payload).is_some()
        {
            return Err(Error::Malformed("duplicate ISO gain-map metadata"));
        }
        if span.kind == MarkerKind::App(1) && span.payload.starts_with(XMP) {
            req.packet(span.payload.len())?;
            let xml = core::str::from_utf8(&span.payload[XMP.len()..])
                .map_err(|_| Error::Malformed("XMP UTF-8"))?;
            if s.xmp.replace(xml).is_some() {
                return Err(Error::Malformed("duplicate gain-map XMP"));
            }
        }
    }
    Ok(s)
}
fn inject<'a>(
    plan: &mut Plan<'a>,
    req: &ScrubRequest<'a>,
    bytes: Vec<u8>,
    name: &'static str,
) -> Result<(), Error> {
    plan.inject_after_first(req, bytes, name)
}
fn same_params(a: &zencodec::GainMapParams, b: &zencodec::GainMapParams) -> bool {
    let near = |x: f64, y: f64| (x - y).abs() <= 1e-6;
    a.backward_direction == b.backward_direction
        && a.use_base_color_space == b.use_base_color_space
        && near(a.base_hdr_headroom, b.base_hdr_headroom)
        && near(a.alternate_hdr_headroom, b.alternate_hdr_headroom)
        && a.channels.iter().zip(&b.channels).all(|(x, y)| {
            near(x.min, y.min)
                && near(x.max, y.max)
                && near(x.gamma, y.gamma)
                && near(x.base_offset, y.base_offset)
                && near(x.alternate_offset, y.alternate_offset)
        })
}

fn normalize_icc<'a>(
    req: &ScrubRequest<'a>,
    plan: &mut Plan<'a>,
    parts: &[Option<&[u8]>; 256],
    ranges: &[Range<usize>],
) -> Result<(), Error> {
    let total = parts.iter().flatten().try_fold(0usize, |a, p| {
        a.checked_add(p.len())
            .ok_or(Error::Limit(zencodec::LimitKind::MetadataSize))
    })?;
    req.packet(total)?;
    if !req.normalize_icc() || ranges.is_empty() {
        return Ok(());
    }
    let mut joined = Vec::new();
    joined
        .try_reserve_exact(total)
        .map_err(|_| Error::Allocation)?;
    for p in parts.iter().flatten() {
        joined.extend_from_slice(p);
    }
    if let Some(normalized) = req.icc(&joined)? {
        let count = normalized.len().div_ceil(65519);
        if count > 255 {
            return Err(Error::Limit(zencodec::LimitKind::OutputSize));
        }
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(normalized.len() + count * 18)
            .map_err(|_| Error::Allocation)?;
        for (i, chunk) in normalized.chunks(65519).enumerate() {
            let body = [
                b"ICC_PROFILE\0".as_slice(),
                &[(i + 1) as u8, count as u8],
                chunk,
            ]
            .concat();
            encoded.extend_from_slice(&segment(0xe2, &body)?);
        }
        for range in ranges {
            plan.remove_kept_range(req, range)?;
        }
        inject(plan, req, encoded, "normalized known ICC profile")?;
    }
    Ok(())
}

fn edited_gain_xmp(
    req: &ScrubRequest<'_>,
    source: Option<&str>,
    required: &str,
    params: Option<&zencodec::GainMapParams>,
) -> Result<Vec<u8>, Error> {
    use crate::container::xmp;
    if !req.edit_xmp() || source.is_none() {
        return Ok(xmp::create_xmp_app1_marker(required));
    }
    let Some(descriptive) = req.xmp(source.unwrap().as_bytes())? else {
        return Ok(xmp::create_xmp_app1_marker(required));
    };
    let merged = req.services()?.merge_xmp(
        &descriptive,
        required.as_bytes(),
        req.max_metadata(),
        req.stop(),
    )?;
    req.packet(merged.len())?;
    let merged = core::str::from_utf8(&merged).map_err(|_| Error::Service("merged XMP UTF-8"))?;
    let packet =
        zenmeta::xmp::Packet::parse(merged).map_err(|_| Error::Service("merged XMP XML"))?;
    if let Some(params) = params {
        let actual = xmp::parse_xmp(merged)
            .map_err(|_| Error::Service("merged gain-map parameters"))?
            .0;
        if !same_params(params, &actual) {
            return Err(Error::Service(
                "XMP editor changed required gain-map parameters",
            ));
        }
    } else {
        let expected =
            zenmeta::xmp::Packet::parse(required).map_err(|_| Error::Service("generated XMP"))?;
        let read = |p: &zenmeta::xmp::Packet<'_>| {
            p.resource_sequence(
                xmp::CONTAINER_NAMESPACE,
                "Directory",
                xmp::CONTAINER_NAMESPACE,
                "Item",
            )
            .map_err(|_| Error::Service("merged gain-map directory"))
        };
        if read(&packet)? != read(&expected)? {
            return Err(Error::Service(
                "XMP editor changed required gain-map directory",
            ));
        }
    }
    if packet
        .property(xmp::HDRGM_NAMESPACE, "Version")
        .map_err(|_| Error::Service("merged gain-map version"))?
        != Some(alloc::vec!["1.0".into()])
    {
        return Err(Error::Service(
            "XMP editor changed required gain-map version",
        ));
    }
    segment(0xe1, &[XMP, merged.as_bytes()].concat())
}
