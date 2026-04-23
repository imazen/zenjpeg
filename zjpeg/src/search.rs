//! Target-band quality search for `--search-ssim2` / `--search-distance`.
//!
//! Given a fully-preprocessed pixel buffer (post-decode, post-resize,
//! post-crop/pad) and an encode closure that takes a quality parameter,
//! run a small binary search over `--quality-range` to find the lowest
//! quality that lands the output metric inside the requested band.
//!
//! One-shot encode remains the default; this module activates only when a
//! `--search-*` flag is set.

use anyhow::Result;
use std::str::FromStr;

use zenjpeg::decoder::{DecodeConfig, OutputTarget, PreserveConfig, Strictness};

/// Quality target band as `MIN..MAX` (inclusive on both ends).
///
/// For **SSIM2** (higher = better), the output metric should be `≥ min`
/// and values `> max` waste bytes. For **butteraugli distance** (lower =
/// better), the metric should be `≤ max` and values `< min` waste bytes.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub min: f32,
    pub max: f32,
}

impl FromStr for Band {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let (lo, hi) = s
            .split_once("..")
            .ok_or_else(|| anyhow::anyhow!("expected MIN..MAX (e.g. 50..65), got '{s}'"))?;
        let min: f32 = lo
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid band MIN '{lo}' (expected a number)"))?;
        let max: f32 = hi
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid band MAX '{hi}' (expected a number)"))?;
        if !(min.is_finite() && max.is_finite()) {
            anyhow::bail!("band values must be finite (got {min}..{max})");
        }
        if min > max {
            anyhow::bail!("band MIN ({min}) > MAX ({max})");
        }
        Ok(Band { min, max })
    }
}

/// Perceptual metric to drive the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// SSIMULACRA2 — higher is better (typical range 0..100, 95+ = excellent).
    Ssim2,
    /// Butteraugli distance — lower is better (0 = identical, 1.5 ≈ visible).
    Distance,
}

/// Outcome of a search.
pub struct SearchResult {
    /// The chosen encoded JPEG bytes.
    pub jpeg: Vec<u8>,
    /// The quality parameter at which `jpeg` was produced.
    pub quality: f32,
    /// The measured metric value for `jpeg`.
    pub metric: f32,
    /// Number of encode+measure iterations actually run (≤ budget).
    pub attempts_used: u32,
    /// Whether the final output landed inside the requested band.
    pub in_band: bool,
}

/// Search for the lowest quality that lands the metric inside `band`.
///
/// `encode_at_q` is the caller's encode closure — it produces a JPEG byte
/// stream for the given quality. The search binary-searches over
/// `quality_range` (inclusive), encoding and measuring at each step.
/// On out-of-band results, the search direction is:
/// - SSIM2 above band (too good)     → lower q (smaller file, lower SSIM2)
/// - SSIM2 below band (too bad)      → raise q
/// - Distance above band (too bad)   → raise q (smaller distance)
/// - Distance below band (too good)  → lower q (larger distance, smaller file)
///
/// If all attempts are exhausted without landing in-band, returns the best
/// candidate seen — the one closest to the band (ties broken by smaller
/// file). The caller can inspect `in_band` to decide whether to emit a
/// warning.
pub fn search_for_band<F>(
    source_rgb: &[u8],
    width: u32,
    height: u32,
    band: Band,
    metric: Metric,
    quality_range: (f32, f32),
    initial_quality: Option<f32>,
    attempts: u32,
    mut encode_at_q: F,
) -> Result<SearchResult>
where
    F: FnMut(f32) -> Result<Vec<u8>>,
{
    if attempts == 0 {
        anyhow::bail!("--attempts must be ≥ 1");
    }

    let (mut lo, mut hi) = quality_range;
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }

    let mut best: Option<SearchResult> = None;
    let mut attempts_used = 0u32;
    let mut next_q = initial_quality.map(|q| q.clamp(lo, hi));

    for _ in 0..attempts {
        // Use the seed on iteration 1 when provided; otherwise bisect current bounds.
        let q = next_q
            .take()
            .unwrap_or_else(|| ((lo + hi) / 2.0).clamp(0.0, 100.0));

        let jpeg = encode_at_q(q)?;
        let measured = measure(&jpeg, source_rgb, width, height, metric)?;
        attempts_used += 1;

        let in_band = measured >= band.min && measured <= band.max;
        let candidate = SearchResult {
            jpeg,
            quality: q,
            metric: measured,
            attempts_used,
            in_band,
        };

        if in_band {
            return Ok(candidate);
        }

        // Track best-so-far (closest to band; prefer smaller file on ties).
        best = match best {
            None => Some(candidate),
            Some(prev) => {
                let prev_dist = band_distance(prev.metric, band);
                let new_dist = band_distance(candidate.metric, band);
                if new_dist < prev_dist
                    || (new_dist == prev_dist && candidate.jpeg.len() < prev.jpeg.len())
                {
                    Some(candidate)
                } else {
                    Some(prev)
                }
            }
        };

        // Adjust bounds for next iteration.
        let measured = best.as_ref().map(|b| b.metric).unwrap();
        let above_band = measured > band.max;
        let lower_q_needed = match metric {
            Metric::Ssim2 => above_band,     // too good → lower q
            Metric::Distance => !above_band, // too good (distance too low) → lower q
        };
        if lower_q_needed {
            hi = (q - 1.0).max(lo);
        } else {
            lo = (q + 1.0).min(hi);
        }
        if hi <= lo {
            break;
        }
    }

    Ok(best.expect("at least one encode attempt must have produced a candidate"))
}

fn band_distance(measured: f32, band: Band) -> f32 {
    if measured < band.min {
        band.min - measured
    } else if measured > band.max {
        measured - band.max
    } else {
        0.0
    }
}

/// Decode `jpeg_bytes` back to RGB8 and compare against `source_rgb` using
/// the chosen metric. Returns the measured value.
fn measure(
    jpeg_bytes: &[u8],
    source_rgb: &[u8],
    width: u32,
    height: u32,
    metric: Metric,
) -> Result<f32> {
    let decoded_rgb = decode_to_rgb8(jpeg_bytes, width, height)?;

    match metric {
        Metric::Ssim2 => compute_ssim2(source_rgb, &decoded_rgb, width, height),
        Metric::Distance => compute_butteraugli(source_rgb, &decoded_rgb, width, height),
    }
}

/// Decode a JPEG byte stream back to tightly-packed RGB8 at the expected
/// dimensions. Uses a permissive decoder so we don't reject our own encoder
/// output over edge-case structural quirks.
fn decode_to_rgb8(jpeg_bytes: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let mut cfg = DecodeConfig::new().preserve(PreserveConfig::none());
    cfg.strictness = Strictness::Permissive;
    cfg.output_target = OutputTarget::Srgb8;

    let result = cfg
        .decode(jpeg_bytes, enough::Unstoppable)
        .map_err(|e| anyhow::anyhow!("decode of search candidate failed: {e}"))?;

    let got_w = result.width();
    let got_h = result.height();
    if got_w != w || got_h != h {
        anyhow::bail!(
            "search: decoded candidate has unexpected dimensions ({got_w}x{got_h}, expected {w}x{h})"
        );
    }

    let pixels = result
        .pixels_u8()
        .ok_or_else(|| anyhow::anyhow!("search: decoded candidate was not RGB8"))?;
    Ok(pixels.to_vec())
}

/// Repack a packed RGB8 byte buffer as `ImgVec<[u8; 3]>` for the metric crates.
fn pack_rgb8(pixels: &[u8], w: u32, h: u32) -> imgref::ImgVec<[u8; 3]> {
    let triples: Vec<[u8; 3]> = pixels.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    imgref::ImgVec::new(triples, w as usize, h as usize)
}

/// Compute SSIMULACRA2 between two RGB8 planar buffers.
fn compute_ssim2(a: &[u8], b: &[u8], w: u32, h: u32) -> Result<f32> {
    let a_owned = pack_rgb8(a, w, h);
    let b_owned = pack_rgb8(b, w, h);
    // `compute_ssimulacra2` requires `ImgRef<[u8; 3]>`, not `ImgVec`.
    let score = fast_ssim2::compute_ssimulacra2(a_owned.as_ref(), b_owned.as_ref())
        .map_err(|e| anyhow::anyhow!("ssim2 compute failed: {e:?}"))?;
    Ok(score as f32)
}

/// Compute butteraugli distance between two RGB8 planar buffers.
fn compute_butteraugli(a: &[u8], b: &[u8], w: u32, h: u32) -> Result<f32> {
    use imgref::ImgRef;
    use rgb::RGB8;

    let a_pix: Vec<RGB8> = a
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let b_pix: Vec<RGB8> = b
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    let a_img = ImgRef::new(&a_pix, w as usize, h as usize);
    let b_img = ImgRef::new(&b_pix, w as usize, h as usize);

    let result = butteraugli::butteraugli(a_img, b_img, &butteraugli::ButteraugliParams::default())
        .map_err(|e| anyhow::anyhow!("butteraugli compute failed: {e:?}"))?;
    Ok(result.score as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_parse_valid() {
        let b: Band = "50..65".parse().unwrap();
        assert_eq!(b.min, 50.0);
        assert_eq!(b.max, 65.0);

        let b: Band = "0.8..1.5".parse().unwrap();
        assert_eq!(b.min, 0.8);
        assert_eq!(b.max, 1.5);
    }

    #[test]
    fn band_parse_rejects_inverted() {
        let err: Result<Band, _> = "65..50".parse();
        assert!(err.is_err());
    }

    #[test]
    fn band_parse_rejects_missing_dots() {
        assert!("50:65".parse::<Band>().is_err());
        assert!("50".parse::<Band>().is_err());
    }

    #[test]
    fn band_distance_inside_is_zero() {
        let b = Band {
            min: 50.0,
            max: 65.0,
        };
        assert_eq!(band_distance(55.0, b), 0.0);
        assert_eq!(band_distance(50.0, b), 0.0);
        assert_eq!(band_distance(65.0, b), 0.0);
    }

    #[test]
    fn band_distance_below_reports_gap() {
        let b = Band {
            min: 50.0,
            max: 65.0,
        };
        assert!((band_distance(45.0, b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn band_distance_above_reports_gap() {
        let b = Band {
            min: 50.0,
            max: 65.0,
        };
        assert!((band_distance(70.0, b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn search_respects_attempts_cap() {
        // Verify the search loop terminates at the attempt budget even when
        // no quality lands the metric in band. Uses a 64×64 gray buffer —
        // large enough to satisfy fast_ssim2's minimum dimensions.
        const W: u32 = 64;
        const H: u32 = 64;
        let fake_pixels = vec![128u8; (W * H * 3) as usize];
        let result = search_for_band(
            &fake_pixels,
            W,
            H,
            // Unreachable band: gray-vs-re-encoded-gray always ≈ 100 SSIM2.
            Band { min: 0.0, max: 1.0 },
            Metric::Ssim2,
            (20.0, 90.0),
            None,
            3,
            |_q| {
                use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};
                let cfg =
                    EncoderConfig::ycbcr(Quality::ApproxJpegli(50.0), ChromaSubsampling::None);
                cfg.encode_bytes(&fake_pixels, W, H, PixelLayout::Rgb8Srgb)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))
            },
        );
        let r = result.unwrap();
        assert!(r.attempts_used >= 1);
        assert!(r.attempts_used <= 3);
    }
}
