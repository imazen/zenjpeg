//! Qualified controlled JPEG route. Retains admitted encoded input, then emits
//! native scanline batches. Native coefficient/transform workspace and metadata
//! caches are not part of the shared storage allowance.
use super::*;
use alloc::boxed::Box;
use core::fmt;
use zencodec::decode::{
    BatchMeta, BoxedError, ControlError, DecodeJob, DynControlledDecoder, InterpretationFacts,
    InterpretationState, InterpretationUnavailable, MetadataUpdate, StreamingDecode,
    erase_controlled,
};
use zencodec::source::AccessPattern;
use zencodec::storage::{StorageBudget, StorageError};
use zencodec_coroutine::{CoroutineDecoder, CoroutineError};

#[derive(Debug)]
enum ControlledError {
    Native(At<CodecError>),
    Coroutine(CoroutineError),
}
impl fmt::Display for ControlledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(e) => e.fmt(f),
            Self::Coroutine(e) => e.fmt(f),
        }
    }
}
impl core::error::Error for ControlledError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(match self {
            Self::Native(e) => e,
            Self::Coroutine(e) => e,
        })
    }
}
impl From<CoroutineError> for ControlledError {
    fn from(e: CoroutineError) -> Self {
        Self::Coroutine(e)
    }
}
impl From<StorageError> for ControlledError {
    fn from(e: StorageError) -> Self {
        Self::Coroutine(e.into())
    }
}
impl From<At<CodecError>> for ControlledError {
    fn from(e: At<CodecError>) -> Self {
        Self::Native(e)
    }
}
impl From<Error> for ControlledError {
    fn from(e: Error) -> Self {
        Self::Native(e.into())
    }
}

pub(super) fn start(
    job: JpegDecodeJob,
    preferred: &[PixelDescriptor],
    storage: StorageBudget,
) -> Result<Box<DynControlledDecoder>, BoxedError> {
    // Existing implicit CMS, crop and gain-map routes require their own
    // qualification. Do not silently select a different implementation.
    if job.crop_hint.is_some()
        || job.gain_map_render != zencodec::GainMapRender::BaseOnly
        || job.config.inner.correct_color.is_some()
        || job.config.inner.crop_region.is_some()
        || job.config.inner.decode_transform.is_some()
        || job.config.inner.force_f32_idct
        || job.config.inner.output_target != crate::decode::OutputTarget::Srgb8
        || job.config.inner.gain_map != crate::decode::GainMapHandling::Discard
        || (will_auto_orient(job.orientation) && !job.config.inner.auto_orient)
        || !matches!(
            job.orientation,
            zencodec::OrientationHint::Preserve | zencodec::OrientationHint::Correct
        )
    {
        return Err(Box::new(
            At::wrap(ControlError::NotControlled)
                .at_named("JPEG configuration needs another controlled route"),
        ));
    }
    if let Some(stop) = &job.stop {
        enough::Stop::check(stop).map_err(Error::from)?;
    }
    let mut formats = Vec::new();
    formats.try_reserve_exact(preferred.len())?;
    formats.extend_from_slice(preferred);
    let source_storage = storage.clone();
    let decoder = CoroutineDecoder::controlled_with_storage(
        4096,
        AccessPattern::RandomAccess,
        storage,
        move |mut ctx| async move {
            let check_stop = || {
                if let Some(stop) = &job.stop {
                    enough::Stop::check(stop).map_err(Error::from)?;
                }
                Ok::<(), Error>(())
            };
            check_stop()?;
            let total = ctx.len().await?;
            job.limits.check_input_size(total).map_err(|_| {
                Error::resource_limit_exceeded(
                    zencodec::LimitKind::InputSize,
                    total,
                    job.limits.max_input_bytes.unwrap_or(0),
                )
            })?;
            let len = usize::try_from(total).map_err(|_| StorageError::SizeOverflow)?;
            let mut input = source_storage.bytes(len)?;
            for (i, chunk) in input.as_mut_slice().chunks_mut(4096).enumerate() {
                check_stop()?;
                ctx.read_exact_at((i * 4096) as u64, chunk).await?;
            }
            let data = input.as_slice();
            let header = read_job_info(&job.limit_adjusted_inner(), data, job.policy.as_ref())?;
            job.check_progressive_policy(header.mode)?;
            job.limits
                .check_dimensions(header.dimensions.width, header.dimensions.height)
                .map_err(Error::from)?;
            ctx.set_info(job.probe(data)?);
            ctx.announce(MetadataUpdate::HeaderReady).await?;
            ctx.interpretation(InterpretationFacts::new(
                header.dimensions.width,
                header.dimensions.height,
                None,
                InterpretationState::Provisional,
            ))
            .await?;
            let (full_header, extras) = job.limit_adjusted_inner().read_full_details_preserving(
                data,
                &metadata_preserve(job.policy.as_ref()),
                job.stop
                    .as_ref()
                    .map_or(&enough::Unstoppable as &dyn enough::Stop, |s| s),
                true,
            )?;
            let mut full = to_image_info(&full_header);
            super::super::info::populate_complete_container_scope(data, &mut full);
            let mut output = job.output_info(data)?;
            output.native_format = decode_descriptor(&formats, &header, None);
            let output_bytes = u64::from(output.width)
                .checked_mul(u64::from(output.height))
                .and_then(|n| n.checked_mul(output.native_format.bytes_per_pixel() as u64))
                .ok_or(StorageError::SizeOverflow)?;
            job.limits
                .check_output_size(output_bytes)
                .map_err(Error::from)?;
            let mut after_scan = false;
            let mut has_icc = false;
            let mut late_icc = false;
            for span in crate::container::MarkerIter::for_decoder_metadata(data) {
                ctx.check_stop()?;
                check_stop()?;
                if span.kind == crate::container::MarkerKind::Sos {
                    after_scan = true;
                }
                if span.kind == crate::container::MarkerKind::App(2)
                    && span.payload.starts_with(b"ICC_PROFILE\0")
                {
                    has_icc = true;
                    late_icc |= after_scan;
                }
            }
            // Recognition of an ICC carrier is not parsing its contents. An ICC
            // extraction prohibition cannot turn an unknown profile into sRGB.
            let unavailable = if late_icc {
                Some(InterpretationUnavailable::LateColorMetadata)
            } else if has_icc && job.policy.is_some_and(|p| p.allow_icc == Some(false)) {
                Some(InterpretationUnavailable::ForbiddenColorMetadata)
            } else if has_icc && header.icc_profile.is_none() {
                Some(InterpretationUnavailable::IncompleteColorMetadata)
            } else if header.precision != 8
                || !matches!(header.num_components, 1 | 3)
                || header.is_xyb
            {
                Some(InterpretationUnavailable::UnsupportedColor)
            } else if output.native_format.channel_type() != zenpixels::ChannelType::U8 {
                // Integer strip reconstruction and whole-image float IDCT differ.
                Some(InterpretationUnavailable::UnqualifiedOutput)
            } else {
                None
            };
            full.width = output.width;
            full.height = output.height;
            if will_auto_orient(job.orientation) {
                full.orientation = zencodec::Orientation::Identity;
            }
            #[cfg(feature = "metadata")]
            let source_encoding = crate::detect::probe(data).ok();
            #[cfg(feature = "metadata")]
            if let Some(probe) = &source_encoding {
                full = full.with_source_encoding_details(probe.clone());
            }
            let mut details = zencodec::decode::DecodeDetails::new(full.clone());
            details.extensions.insert(extras);
            #[cfg(feature = "metadata")]
            {
                details.source_encoding = source_encoding.map(|probe| Box::new(probe) as _);
            }
            ctx.set_info(full);
            ctx.set_output_descriptor(output.native_format);
            let mut facts = InterpretationFacts::new(
                output.width,
                output.height,
                Some(output.native_format),
                InterpretationState::Final,
            );
            if let Some(reason) = unavailable {
                facts = facts.with_unavailable(reason);
            }
            ctx.interpretation(facts).await?;
            // Progressive construction can perform coefficient work. It must stay
            // beyond the awaited final adoption, not merely before row publication.
            check_stop()?;
            let mut native = job.streaming_decoder(Cow::Borrowed(data), &formats)?;
            while let Some((y, rows)) = native.next_batch()? {
                ctx.write_rows(BatchMeta::final_rows(y), &rows).await?;
            }
            ctx.set_details(details);
            Ok::<_, ControlledError>(output)
        },
    )?;
    Ok(erase_controlled(decoder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec::decode::DecoderConfig;

    #[test]
    fn unqualified_configuration_guards_are_independent() {
        let changes: &[fn(&mut JpegDecodeJob)] = &[
            |j| j.gain_map_render = zencodec::GainMapRender::Components,
            |j| j.config.inner.crop_region = Some(crate::decode::CropRegion::pixels(0, 0, 1, 1)),
            |j| {
                j.config.inner.decode_transform = Some(crate::lossless::LosslessTransform::Rotate90)
            },
            |j| j.config.inner.force_f32_idct = true,
            |j| j.config.inner.output_target = crate::decode::OutputTarget::Native16,
            |j| j.config.inner.gain_map = crate::decode::GainMapHandling::PreserveRaw,
            |j| {
                j.orientation = zencodec::OrientationHint::Correct;
                j.config.inner.auto_orient = false;
            },
            |j| {
                j.orientation =
                    zencodec::OrientationHint::ExactTransform(zencodec::Orientation::Identity)
            },
        ];
        for change in changes {
            let mut job = JpegDecoderConfig::new().job();
            change(&mut job);
            let storage = StorageBudget::new(1 << 20, 4);
            let error = start(job, &[], storage.clone())
                .err()
                .expect("unqualified configuration refused");
            assert_eq!(
                zencodec::find_cause::<ControlError>(&*error),
                Some(&ControlError::NotControlled)
            );
            assert_eq!(storage.reserved_bytes(), 0);
        }
    }

    #[test]
    fn error_context_and_sources_survive_wrapping() {
        use core::error::Error as _;
        let native: ControlledError = Error::invalid_jpeg_data("controlled native failure").into();
        assert!(native.to_string().contains("controlled native failure"));
        assert!(native.source().is_some());
        let storage: ControlledError = StorageError::SizeOverflow.into();
        assert_eq!(
            storage.to_string(),
            CoroutineError::from(StorageError::SizeOverflow).to_string()
        );
        assert!(storage.source().is_some());
    }
}
