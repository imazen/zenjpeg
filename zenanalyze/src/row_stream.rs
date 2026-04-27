//! On-demand RGB8 row source for the analyzer.
//!
//! The analyzer never materializes a full RGB8 buffer for non-RGB8
//! inputs. Each tier asks for one (or a few) rows at a time; a
//! [`RowStream`] either borrows them straight out of the source slice
//! (zero-copy when the descriptor is already RGB8 / RGB8_SRGB) or
//! converts row-by-row through `zenpixels-convert::RowConverter`.
//!
//! Total scratch never exceeds one row of RGB8 plus the converter's
//! internal scratch — no per-image allocation that scales with height.
//!
//! ## Why row-pull instead of one big buffer
//!
//! Tier 1 sparse-stripe samples ~h/8 rows in the worst case (a
//! 1-row stripe step on a small image), but typically reads only
//! ~500k pixels' worth of rows. Tier 2 walks (a0, a1, a2) in a
//! 3-row sliding window. Tier 3 entropy reads every 4th pixel of
//! every row, and the high-freq DCT samples 256 8×8 blocks. None
//! of these need the full image resident in RGB8.

use core::ops::Range;

use zenpixels::{PixelDescriptor, PixelSlice};
use zenpixels_convert::RowConverter;

/// Pull RGB8 rows from any [`PixelSlice`].
///
/// Holds either:
/// - a borrowed `PixelSlice` already in RGB8 layout (every fetched
///   row is a sub-slice of the original buffer, zero-copy), OR
/// - a borrowed `PixelSlice` plus a `RowConverter` that emits one
///   row at a time into an internal `width * 3` byte scratch buffer.
pub struct RowStream<'a> {
    inner: Inner<'a>,
    width: u32,
    height: u32,
    /// Source primaries — captured at construction so per-primaries
    /// luma weights / chroma matrices can be looked up by tier code
    /// without holding a back-reference to the slice.
    primaries: zenpixels::ColorPrimaries,
    /// Row-of-RGB8 scratch reused across `fetch_into` and `borrow_row`.
    scratch: Vec<u8>,
}

enum Inner<'a> {
    Native(PixelSlice<'a>),
    /// 32-bpp RGBA-family layout (RGBA8 / BGRA8 / Rgbx8 / Bgrx8).
    /// Skips `RowConverter` entirely — populates the row scratch
    /// with a tight strip-alpha-and-maybe-swap pass that's an order
    /// of magnitude cheaper than the converter's full
    /// transfer + matrix + narrow path. The converter alloc + setup
    /// cost is also gone. Saves ~1 ms / 4 K row in profiles.
    StripAlpha8 {
        slice: PixelSlice<'a>,
        /// Byte indices within each 4-byte source pixel that map to
        /// `[R, G, B]` in the output. RGBA8 / Rgbx8 ⇒ `[0, 1, 2]`,
        /// BGRA8 / Bgrx8 ⇒ `[2, 1, 0]`. Resolved at construction so
        /// the inner loop is branch-free.
        rgb_idx: [u8; 3],
    },
    Convert {
        slice: PixelSlice<'a>,
        converter: RowConverter,
    },
}

impl<'a> RowStream<'a> {
    /// Build a row stream over `slice`. Picks the zero-copy path when
    /// the descriptor is layout-compatible with `RGB8`; otherwise
    /// constructs a `RowConverter` to `RGB8_SRGB`.
    ///
    /// # Errors
    ///
    /// Returns the underlying `RowConverter` construction error if
    /// the source descriptor isn't convertible (e.g. CMYK without a
    /// CMS plugin).
    pub fn new(slice: PixelSlice<'a>) -> Result<Self, String> {
        let width = slice.width();
        let height = slice.rows();
        let desc = slice.descriptor();

        // Native path: any 24bpp layout that's byte-equivalent to RGB8
        // (RGB8 / RGB8_SRGB / RGB8 with non-sRGB transfer/primaries).
        // We treat the bytes as display-space RGB regardless of the
        // transfer function — the analyzer math doesn't care about
        // gamma; the trees were trained on the same display-space
        // semantics.
        let rgb8_compat = desc.layout_compatible(PixelDescriptor::RGB8);

        // Strip-alpha fast path: 32-bpp RGBA-class layouts. The RGB
        // tiers want display-space RGB samples; for opaque or
        // straight-alpha sources, that's just the colour bytes with
        // alpha dropped (channel-order re-mapped for BGRA). The full
        // RowConverter machinery is overkill for that case — building
        // the converter alone allocates and bakes a multi-stage plan,
        // and `convert_row` does work that's wasted when the
        // destination is the same channel type, same transfer, just
        // 4 → 3 bytes per pixel.
        let strip_idx = strip_alpha_indices(desc.format);

        let inner = if rgb8_compat {
            Inner::Native(slice)
        } else if let Some(rgb_idx) = strip_idx {
            Inner::StripAlpha8 { slice, rgb_idx }
        } else {
            // Convert to plain RGB8_SRGB on the way in. The transfer
            // function difference vs the native input has at most
            // a few-LSB effect on luma/chroma stats and matches what
            // the coefficient reference path does (it loads via
            // `image::open(...).to_rgb8()` which is also a sRGB
            // assumption).
            let converter = RowConverter::new(desc, PixelDescriptor::RGB8_SRGB)
                .map_err(|e| format!("RowConverter::new failed: {:?}", e))?;
            Inner::Convert { slice, converter }
        };

        let scratch_len = (width as usize).saturating_mul(3);
        Ok(Self {
            inner,
            width,
            height,
            primaries: desc.primaries,
            scratch: vec![0u8; scratch_len],
        })
    }

    /// Source primaries — the analyzer tiers look this up to pick the
    /// right luma weights for a u8 RGB byte stream that's still in
    /// the source's primaries (Native zero-copy path) without
    /// converting. See [`crate::luma::LumaWeights::for_primaries`].
    #[inline]
    pub fn primaries(&self) -> zenpixels::ColorPrimaries {
        self.primaries
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Fetch row `y` into `dst[..width*3]`. Caller-provided `dst`
    /// avoids the borrow-once-then-locked problem when the analyzer
    /// needs multiple rows simultaneously (Tier 1 stripe, Tier 2
    /// sliding window, Tier 3 high-freq DCT block).
    ///
    /// # Panics
    ///
    /// Panics if `y >= height` or `dst.len() < width * 3`.
    pub fn fetch_into(&mut self, y: u32, dst: &mut [u8]) {
        assert!(
            y < self.height,
            "row {y} out of bounds (height={})",
            self.height
        );
        let len = self.width as usize * 3;
        assert!(dst.len() >= len, "dst too small for one RGB8 row");
        match &mut self.inner {
            Inner::Native(slice) => {
                let row = slice.row(y);
                dst[..len].copy_from_slice(&row[..len]);
            }
            Inner::StripAlpha8 { slice, rgb_idx } => {
                strip_alpha_row(slice.row(y), self.width as usize, *rgb_idx, &mut dst[..len]);
            }
            Inner::Convert { slice, converter } => {
                let src = slice.row(y);
                converter.convert_row(src, &mut dst[..len], self.width);
            }
        }
    }

    /// Borrow row `y` as RGB8. Native path is zero-copy; converting
    /// path writes into the internal scratch and returns a borrow of
    /// it (next call invalidates the borrow).
    pub fn borrow_row(&mut self, y: u32) -> &[u8] {
        assert!(
            y < self.height,
            "row {y} out of bounds (height={})",
            self.height
        );
        let len = self.width as usize * 3;
        match &mut self.inner {
            Inner::Native(slice) => &slice.row(y)[..len],
            Inner::StripAlpha8 { slice, rgb_idx } => {
                strip_alpha_row(slice.row(y), self.width as usize, *rgb_idx, &mut self.scratch[..len]);
                &self.scratch[..len]
            }
            Inner::Convert { slice, converter } => {
                let src = slice.row(y);
                converter.convert_row(src, &mut self.scratch[..len], self.width);
                &self.scratch[..len]
            }
        }
    }

    /// Bulk-fill `dst` with rows `range` packed back-to-back at
    /// `width * 3` stride. Used by tiers that hold a multi-row
    /// window (Tier 1 stripe scratch, Tier 2 sliding window, Tier 3
    /// high-freq DCT 8-row block).
    ///
    /// # Panics
    ///
    /// Panics if any row is out of bounds, or if `dst` can't hold
    /// `range.len() * width * 3` bytes.
    pub fn fetch_range(&mut self, range: Range<u32>, dst: &mut [u8]) {
        let row_bytes = self.width as usize * 3;
        let need = range.len() * row_bytes;
        assert!(dst.len() >= need, "dst too small for {} rows", range.len());
        for (i, y) in range.enumerate() {
            self.fetch_into(y, &mut dst[i * row_bytes..(i + 1) * row_bytes]);
        }
    }
}

/// Decide whether a 32-bpp RGBA-class format is the strip-alpha
/// fast path, and what `[R, G, B]` byte indices within each
/// 4-byte source pixel map to. Returns `None` for layouts that
/// genuinely need `RowConverter` (different channel type, gray,
/// CMYK, Oklab, …) or for non-32-bpp inputs.
fn strip_alpha_indices(format: zenpixels::PixelFormat) -> Option<[u8; 3]> {
    use zenpixels::PixelFormat;
    match format {
        PixelFormat::Rgba8 | PixelFormat::Rgbx8 => Some([0, 1, 2]),
        PixelFormat::Bgra8 | PixelFormat::Bgrx8 => Some([2, 1, 0]),
        // RGBA16 / RGBAF32 / Gray* / CMYK / etc. all need the full
        // converter — channel type doesn't match RGB8.
        _ => None,
    }
}

/// Strip alpha from a 32-bpp RGBA-class row into a tightly packed
/// 24-bpp RGB8 row, dispatching to garb's SIMD primitives:
/// `rgba_to_rgb` for the in-order case (`rgb_idx == [0, 1, 2]`,
/// RGBA / Rgbx sources) and `bgra_to_rgb` for the swapped case
/// (`rgb_idx == [2, 1, 0]`, BGRA / Bgrx). Both use
/// `archmage::incant!` to dispatch to AVX2 / NEON / WASM128 / scalar.
///
/// Measured ~7× speedup over the previous in-tree scalar strip on a
/// 2048-px row (0.13 µs vs 0.94 µs). On full-feature analyzer
/// runs (`FeatureSet::SUPPORTED`, 4 MP RGBA8) the strip cost
/// dropped from ~5.7 ms total to under 1 ms.
///
/// No allocation, no transfer-function math, no f32.
#[inline]
fn strip_alpha_row(src: &[u8], width: usize, rgb_idx: [u8; 3], dst: &mut [u8]) {
    let need_src = width * 4;
    let need_dst = width * 3;
    debug_assert!(src.len() >= need_src);
    debug_assert!(dst.len() >= need_dst);
    let src_row = &src[..need_src];
    let dst_row = &mut dst[..need_dst];
    match rgb_idx {
        [0, 1, 2] => {
            // RGBA / Rgbx → drop byte 3, keep 0..3 in order.
            // garb returns `Result<(), SizeError>`; we already
            // sized the slices to match, so unwrap is infallible.
            garb::bytes::rgba_to_rgb(src_row, dst_row).unwrap();
        }
        [2, 1, 0] => {
            // BGRA / Bgrx → drop byte 3, swap 0↔2.
            garb::bytes::bgra_to_rgb(src_row, dst_row).unwrap();
        }
        _ => {
            // Defensive fallback for any new layout that
            // `strip_alpha_indices` might map to in the future.
            // Today only the two cases above reach here.
            let r = rgb_idx[0] as usize;
            let g = rgb_idx[1] as usize;
            let b = rgb_idx[2] as usize;
            for (s, d) in src_row.chunks_exact(4).zip(dst_row.chunks_exact_mut(3)) {
                d[0] = s[r];
                d[1] = s[g];
                d[2] = s[b];
            }
        }
    }
}

#[cfg(test)]
mod strip_tests {
    use super::*;
    use zenpixels::{PixelDescriptor, PixelSlice};

    #[test]
    fn strip_alpha_path_fires_for_rgba8() {
        // RGBA8 source ⇒ should pick the StripAlpha8 path (not Convert).
        // Verify by checking that no allocation-heavy converter setup
        // happens — observable through the strip helper producing
        // bit-exact output and the rows being identical to a manual
        // strip.
        let w: u32 = 8;
        let h: u32 = 4;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            let v = (i & 0xFF) as u8;
            rgba.extend_from_slice(&[v, v.wrapping_add(1), v.wrapping_add(2), 0xFF]);
        }
        let s = PixelSlice::new(&rgba, w, h, (w * 4) as usize, PixelDescriptor::RGBA8_SRGB)
            .unwrap();
        let mut stream = RowStream::new(s).unwrap();
        let row = stream.borrow_row(0);
        assert_eq!(row.len(), (w * 3) as usize);
        for px in 0..(w as usize) {
            assert_eq!(row[px * 3], (px & 0xFF) as u8);
            assert_eq!(row[px * 3 + 1], (px as u8).wrapping_add(1));
            assert_eq!(row[px * 3 + 2], (px as u8).wrapping_add(2));
        }
    }

    #[test]
    fn strip_alpha_path_swaps_channels_for_bgra8() {
        // BGRA8 ⇒ index map [2, 1, 0]: source bytes [B, G, R, A]
        // become RGB8 [R, G, B] in the output.
        let w: u32 = 4;
        let h: u32 = 1;
        let bgra: Vec<u8> = vec![10, 20, 30, 0xFF, 11, 21, 31, 0xFF, 12, 22, 32, 0xFF, 13, 23, 33, 0xFF];
        let s = PixelSlice::new(&bgra, w, h, (w * 4) as usize, PixelDescriptor::BGRA8_SRGB)
            .unwrap();
        let mut stream = RowStream::new(s).unwrap();
        let row = stream.borrow_row(0);
        assert_eq!(row, &[30, 20, 10, 31, 21, 11, 32, 22, 12, 33, 23, 13]);
    }
}
