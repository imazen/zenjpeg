//! Metadata shared by native DCT and predictive row decoding.
use super::JpegParser;
use crate::container::marker::{self, MarkerKind};
use crate::error::Result;
use enough::Stop;

impl JpegParser<'_> {
    pub(in crate::decode) fn collect_native_metadata(&mut self, stop: &dyn Stop) -> Result<()> {
        for span in marker::iter(self.data) {
            if span.offset < self.position {
                continue;
            }
            stop.check()?;
            self.position = span.offset + 2;
            match span.kind {
                MarkerKind::App(n) => self.process_app_or_com(0xe0 + n)?,
                MarkerKind::Com => self.process_app_or_com(0xfe)?,
                MarkerKind::Eoi => break,
                _ => (),
            }
        }
        self.extract_mpf_secondary_images()?;
        Ok(())
    }
}
