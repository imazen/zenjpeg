//! Errors from the fallible `YuvContext::try_*` entry points.

use core::fmt;

/// Why a [`crate::YuvContext`] `try_*` conversion refused its input.
///
/// The infallible methods panic on the same inputs; the `try_*` methods
/// check them first, so nothing is written or allocated on refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum YuvError {
    /// `width × height` exceeds the context's pixel limit
    /// ([`crate::DEFAULT_MAX_PIXELS`] unless set with
    /// [`crate::YuvContext::with_max_pixels`]).
    TooManyPixels {
        /// `width × height`.
        pixels: u64,
        /// The limit in force.
        max: u64,
    },
    /// A buffer is shorter than the dimensions need.
    BufferTooSmall {
        /// Which argument: `"rgb"`, `"y"`, `"cb"` or `"cr"`.
        buffer: &'static str,
        /// Elements the dimensions need.
        needed: usize,
        /// Elements supplied.
        len: usize,
    },
    /// A strided plane's stride is below its width.
    StrideTooSmall {
        /// Which argument: `"rgb"`, `"y"`, `"cb"` or `"cr"`.
        buffer: &'static str,
        /// The plane's width.
        width: usize,
        /// The stride supplied.
        stride: usize,
    },
    /// A strided plane's dimensions do not match the image's.
    WrongDimensions {
        /// Which argument: `"y"`, `"cb"` or `"cr"`.
        buffer: &'static str,
        /// The dimensions the image needs, `(width, height)`.
        needed: (usize, usize),
        /// The dimensions supplied.
        got: (usize, usize),
    },
    /// A buffer size overflows `usize`.
    Overflow,
    /// A scratch allocation failed.
    OutOfMemory,
}

impl fmt::Display for YuvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyPixels { pixels, max } => {
                write!(f, "{pixels} pixels exceeds the limit of {max}")
            }
            Self::BufferTooSmall {
                buffer,
                needed,
                len,
            } => write!(f, "{buffer} buffer holds {len} elements, needs {needed}"),
            Self::StrideTooSmall {
                buffer,
                width,
                stride,
            } => write!(f, "{buffer} stride {stride} is below its width {width}"),
            Self::WrongDimensions {
                buffer,
                needed: (nw, nh),
                got: (gw, gh),
            } => write!(f, "{buffer} plane is {gw}x{gh}, needs {nw}x{nh}"),
            Self::Overflow => f.write_str("buffer size overflows usize"),
            Self::OutOfMemory => f.write_str("scratch allocation failed"),
        }
    }
}

impl core::error::Error for YuvError {}
