//! Allocation probe for effective-job planning; profile it with heaptrack
//! (`scripts/profile-encode-planning.py`). It has no timers, allocates no
//! source raster, reads no pixels and consumes no encoded output.
//!
//! `encode_planning_probe prepare|plan|begin 1|4 fixed|progressive srgb|p3-icc|p3-descriptor`
//! retains every job (and, for `begin`, every started encoder) at once, so the
//! process peak includes all of them.
use std::hint::black_box;

use zencodec::encode::{EncodeDeclaration, EncodeJob as _, Encoder as _, EncoderConfig as _};
use zencodec::output_patch::OutputAccess;
use zenjpeg::JpegEncoderConfig;
use zenpixels::{ColorContext, ColorPrimaries, PixelDescriptor};

const WIDTH: u32 = 2048;
const HEIGHT: u32 = 2048;

// zencodec's boxed planning error is `Send + Sync`; `?` cannot widen it into a
// plain `Box<dyn Error>`, so main returns the same envelope.
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, stage, count, config, colour] = args.as_slice() else {
        return Err("usage: encode_planning_probe prepare|plan|begin 1|4 \
                    fixed|progressive srgb|p3-icc|p3-descriptor"
            .into());
    };
    let count: usize = match count.as_str() {
        "1" => 1,
        "4" => 4,
        _ => return Err("only counts 1 and 4 are measured".into()),
    };
    let config = match config.as_str() {
        "fixed" => JpegEncoderConfig::new()
            .with_progressive(false)
            .with_huffman(false),
        "progressive" => JpegEncoderConfig::new()
            .with_progressive(true)
            .with_huffman(true),
        _ => return Err("expected fixed or progressive".into()),
    };
    let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let (descriptor, context) = match colour.as_str() {
        "srgb" => (PixelDescriptor::RGB8_SRGB, None),
        "p3-icc" => (
            p3,
            Some(ColorContext::from_icc(
                zenpixels_convert::icc_profiles::DISPLAY_P3_V4,
            )),
        ),
        "p3-descriptor" => (p3, None),
        _ => return Err("expected srgb, p3-icc or p3-descriptor".into()),
    };
    let declaration =
        EncodeDeclaration::new(WIDTH, HEIGHT, descriptor).with_access(OutputAccess::Forward);
    let declaration = match &context {
        Some(context) => declaration.with_color(context),
        None => declaration,
    };
    let jobs: Vec<_> = (0..count).map(|_| config.clone().job()).collect();
    match stage.as_str() {
        "prepare" => {
            black_box(&jobs);
        }
        "plan" => {
            for job in &jobs {
                black_box(job.plan_encode(&declaration)?);
            }
            black_box(&jobs);
        }
        "begin" => {
            let mut encoders = Vec::with_capacity(count);
            for job in jobs {
                let mut encoder = job.encoder()?;
                black_box(encoder.begin_declared(&declaration)?);
                encoders.push(encoder);
            }
            black_box(&encoders);
        }
        _ => return Err("expected prepare, plan or begin".into()),
    }
    Ok(())
}
