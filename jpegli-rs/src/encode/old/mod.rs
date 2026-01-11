//! Legacy encoder implementation.
//!
//! This module contains the original full-plane encoder implementation.
//! For new code, prefer using [`StreamingEncoder`](super::streaming::StreamingEncoder).

pub(crate) mod baseline;
pub(crate) mod blocks;
pub(crate) mod color;
pub(crate) mod output;
pub(crate) mod progressive;

#[cfg(test)]
mod tests;
