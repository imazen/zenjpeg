//! Uninhabited reader keeps shared row plumbing independent of the capability.
use super::JpegParser;
use crate::error::{Error, Result};
use enough::Stop;

pub(in crate::decode) struct PredictiveReader {
    pub row: Vec<u16>,
    never: core::convert::Infallible,
}
impl JpegParser<'_> {
    pub(in crate::decode) fn predictive_reader(
        &mut self,
        _stop: &dyn Stop,
    ) -> Result<PredictiveReader> {
        Err(Error::unsupported_feature(
            "lossless JPEG requires extended-jpeg",
        ))
    }
}
impl PredictiveReader {
    pub(in crate::decode) fn current_row(&self) -> usize {
        match self.never {}
    }
    pub(in crate::decode) fn next_row(&mut self, _data: &[u8], _stop: &dyn Stop) -> Result<()> {
        match self.never {}
    }
}
