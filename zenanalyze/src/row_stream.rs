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
    /// Row-of-RGB8 scratch reused across `fetch_into` and `borrow_row`.
    scratch: Vec<u8>,
}

enum Inner<'a> {
    Native(PixelSlice<'a>),
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

        let inner = if rgb8_compat {
            Inner::Native(slice)
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
            scratch: vec![0u8; scratch_len],
        })
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
