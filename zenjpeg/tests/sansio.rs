#![cfg(feature = "zencodec")]
//! Incremental (sans-I/O) decode through `dyn_incremental_decoder`: the
//! testkit's incremental contract on baseline encodes.

use zencodec::decode::DecoderConfig;
use zencodec::encode::{EncodeJob, EncoderConfig};
use zencodec_testkit as tk;
use zenjpeg::{JpegDecoderConfig as D, JpegEncoderConfig as E};

#[test]
fn incremental_contract_holds() {
    assert!(D::capabilities().incremental());
    for img in [
        tk::TestImage::rgb8_gradient(37, 21),
        tk::TestImage::rgb8_gradient(64, 48),
    ] {
        let bytes = E::new().job().encode(img.as_slice()).unwrap();
        tk::check_incremental_capability_honesty(D::new(), bytes.data()).unwrap();
    }
}
