//! Legacy encoder implementation.
//!
//! This module contains shared encoding functionality used by the strip encoder.
//! For new code, prefer using [`StreamingEncoder`](super::streaming::StreamingEncoder).

pub(crate) mod blocks;
pub(crate) mod output;
pub(crate) mod progressive;
