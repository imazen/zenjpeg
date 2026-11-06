//! Resource limits enforced by the parser.
//!
//! These constants bound the parser's resource use against
//! adversarial input. The numbers are picked to comfortably exceed
//! every realistic shipped bake (V0_18 zensim is 93 KB, the largest
//! shipped picker is the zenavif rav1e v0.1.1 at 217 KB; the deepest
//! shipped layer count is 4) while staying small enough that a
//! gigabyte-claiming header fails fast.
//!
//! ## Hardening pattern
//!
//! Every limit is enforced **before** memory is allocated against
//! the value it bounds. `MAX_BAKE_BYTES` rejects the byte slice
//! itself before the header is read. `MAX_DIM` / `MAX_LAYERS` reject
//! the parsed-but-untrusted header values before the scratch
//! allocation in [`crate::Predictor::new`] tries to materialize them.
//! Whole-bake LZ4 decompression at load time additionally bounds
//! the decompressed payload to `MAX_BAKE_BYTES` minus header.
//!
//! ## Picking numbers
//!
//! 64 MB / 64 K / 256 are 1000x – 10000x larger than any production
//! bake. They're upper bounds that protect web / fuzz / no_std-alloc
//! consumers from runaway parsing without restricting legitimate
//! research / picker work. Adjust if a real bake ever approaches
//! the limit (none should).

/// Maximum byte length of a ZNPR `.bin` accepted by [`crate::Model::from_bytes`].
/// 64 MiB — every shipped bake is < 1 MiB; the limit exists to bound
/// fuzz / adversarial input.
pub const MAX_BAKE_BYTES: usize = 64 * 1024 * 1024;

/// Maximum value of a per-layer or scaler dimension (`n_inputs`,
/// `n_outputs`, `in_dim`, `out_dim`). 65,536 — every shipped bake's
/// largest dim is 384; the limit caps multiplications like
/// `in_dim * out_dim` against `usize` overflow on 32-bit targets.
pub const MAX_DIM: usize = 65_536;

/// Maximum layer count. 256 — every shipped bake has ≤ 4 layers;
/// the limit exists so that `layer_table` allocations are bounded.
pub const MAX_LAYERS: usize = 256;
