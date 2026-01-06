//! FFI bindings to jpegli C++ for comparison testing.
//!
//! This crate provides raw FFI bindings to the C++ jpegli library
//! for testing the Rust port against the original implementation.
//!
//! jpegli exposes a libjpeg-62 compatible API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]

use libc::{c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_void, size_t};

// Boolean type
pub type boolean = c_int;

// JPEG constants
pub const JPEG_LIB_VERSION: c_int = 62;
pub const DCTSIZE: c_int = 8;
pub const DCTSIZE2: c_int = 64;
pub const NUM_QUANT_TBLS: c_int = 4;
pub const NUM_HUFF_TBLS: c_int = 4;
pub const NUM_ARITH_TBLS: c_int = 16;
pub const MAX_COMPS_IN_SCAN: c_int = 4;
pub const MAX_SAMP_FACTOR: c_int = 4;
pub const C_MAX_BLOCKS_IN_MCU: c_int = 10;
pub const D_MAX_BLOCKS_IN_MCU: c_int = 10;

// Color space constants
pub const JCS_UNKNOWN: c_int = 0;
pub const JCS_GRAYSCALE: c_int = 1;
pub const JCS_RGB: c_int = 2;
pub const JCS_YCbCr: c_int = 3;
pub const JCS_CMYK: c_int = 4;
pub const JCS_YCCK: c_int = 5;

// DCT method constants
pub const JDCT_ISLOW: c_int = 0;
pub const JDCT_IFAST: c_int = 1;
pub const JDCT_FLOAT: c_int = 2;

/// JCOEF - DCT coefficient type
pub type JCOEF = i16;

/// JOCTET - a byte
pub type JOCTET = c_uchar;

/// JSAMPLE - a pixel sample value
pub type JSAMPLE = c_uchar;

/// JDIMENSION - dimensions
pub type JDIMENSION = c_uint;

/// J_COLOR_SPACE - color space enumeration
pub type J_COLOR_SPACE = c_int;

/// J_DCT_METHOD - DCT algorithm selector
pub type J_DCT_METHOD = c_int;

/// JSAMPROW - pointer to a row of samples
pub type JSAMPROW = *mut JSAMPLE;

/// JSAMPARRAY - pointer to an array of sample rows
pub type JSAMPARRAY = *mut JSAMPROW;

/// JSAMPIMAGE - pointer to an array of sample arrays (one per component)
pub type JSAMPIMAGE = *mut JSAMPARRAY;

/// JBLOCKROW - pointer to a row of DCT blocks
pub type JBLOCKROW = *mut [JCOEF; 64];

/// JBLOCKARRAY - pointer to an array of block rows
pub type JBLOCKARRAY = *mut JBLOCKROW;

/// Quantization table
#[repr(C)]
pub struct JQUANT_TBL {
    /// Quantization table values in natural order
    pub quantval: [u16; 64],
    /// Indicates if table has been output
    pub sent_table: boolean,
}

/// Huffman table
#[repr(C)]
pub struct JHUFF_TBL {
    /// bits[k] = # of symbols with codes of length k bits
    pub bits: [u8; 17],
    /// Symbol values
    pub huffval: [u8; 256],
    /// Indicates if table has been output
    pub sent_table: boolean,
}

/// Component info for a single component
#[repr(C)]
pub struct jpeg_component_info {
    pub component_id: c_int,
    pub component_index: c_int,
    pub h_samp_factor: c_int,
    pub v_samp_factor: c_int,
    pub quant_tbl_no: c_int,
    pub dc_tbl_no: c_int,
    pub ac_tbl_no: c_int,
    pub width_in_blocks: JDIMENSION,
    pub height_in_blocks: JDIMENSION,
    pub DCT_scaled_size: c_int,
    pub downsampled_width: JDIMENSION,
    pub downsampled_height: JDIMENSION,
    pub component_needed: boolean,
    pub MCU_width: c_int,
    pub MCU_height: c_int,
    pub MCU_blocks: c_int,
    pub MCU_sample_width: c_int,
    pub last_col_width: c_int,
    pub last_row_height: c_int,
    pub quant_table: *mut JQUANT_TBL,
    pub dct_table: *mut c_void,
}

/// Error manager (simplified)
#[repr(C)]
pub struct jpeg_error_mgr {
    pub error_exit: Option<unsafe extern "C" fn(*mut jpeg_common_struct)>,
    pub emit_message: Option<unsafe extern "C" fn(*mut jpeg_common_struct, c_int)>,
    pub output_message: Option<unsafe extern "C" fn(*mut jpeg_common_struct)>,
    pub format_message: Option<unsafe extern "C" fn(*mut jpeg_common_struct, *mut c_char)>,
    pub reset_error_mgr: Option<unsafe extern "C" fn(*mut jpeg_common_struct)>,
    pub msg_code: c_int,
    pub msg_parm: [c_int; 8],
    pub trace_level: c_int,
    pub num_warnings: c_long,
    pub jpeg_message_table: *const *const c_char,
    pub last_jpeg_message: c_int,
    pub addon_message_table: *const *const c_char,
    pub first_addon_message: c_int,
    pub last_addon_message: c_int,
}

/// Destination manager for output
#[repr(C)]
pub struct jpeg_destination_mgr {
    pub next_output_byte: *mut JOCTET,
    pub free_in_buffer: size_t,
    pub init_destination: Option<unsafe extern "C" fn(*mut jpeg_compress_struct)>,
    pub empty_output_buffer: Option<unsafe extern "C" fn(*mut jpeg_compress_struct) -> boolean>,
    pub term_destination: Option<unsafe extern "C" fn(*mut jpeg_compress_struct)>,
}

/// Source manager for input
#[repr(C)]
pub struct jpeg_source_mgr {
    pub next_input_byte: *const JOCTET,
    pub bytes_in_buffer: size_t,
    pub init_source: Option<unsafe extern "C" fn(*mut jpeg_decompress_struct)>,
    pub fill_input_buffer: Option<unsafe extern "C" fn(*mut jpeg_decompress_struct) -> boolean>,
    pub skip_input_data: Option<unsafe extern "C" fn(*mut jpeg_decompress_struct, c_long)>,
    pub resync_to_restart:
        Option<unsafe extern "C" fn(*mut jpeg_decompress_struct, c_int) -> boolean>,
    pub term_source: Option<unsafe extern "C" fn(*mut jpeg_decompress_struct)>,
}

/// Memory manager (opaque)
#[repr(C)]
pub struct jpeg_memory_mgr {
    _private: [u8; 0],
}

/// Progress manager (opaque)
#[repr(C)]
pub struct jpeg_progress_mgr {
    _private: [u8; 0],
}

/// Common fields for compress and decompress
#[repr(C)]
pub struct jpeg_common_struct {
    pub err: *mut jpeg_error_mgr,
    pub mem: *mut jpeg_memory_mgr,
    pub progress: *mut jpeg_progress_mgr,
    pub client_data: *mut c_void,
    pub is_decompressor: boolean,
    pub global_state: c_int,
}

/// Compression struct
#[repr(C)]
pub struct jpeg_compress_struct {
    // Common fields
    pub err: *mut jpeg_error_mgr,
    pub mem: *mut jpeg_memory_mgr,
    pub progress: *mut jpeg_progress_mgr,
    pub client_data: *mut c_void,
    pub is_decompressor: boolean,
    pub global_state: c_int,

    // Destination manager
    pub dest: *mut jpeg_destination_mgr,

    // Image info
    pub image_width: JDIMENSION,
    pub image_height: JDIMENSION,
    pub input_components: c_int,
    pub in_color_space: J_COLOR_SPACE,

    pub input_gamma: f64,

    // Compression parameters
    pub data_precision: c_int,
    pub num_components: c_int,
    pub jpeg_color_space: J_COLOR_SPACE,

    pub comp_info: *mut jpeg_component_info,

    pub quant_tbl_ptrs: [*mut JQUANT_TBL; NUM_QUANT_TBLS as usize],
    pub dc_huff_tbl_ptrs: [*mut JHUFF_TBL; NUM_HUFF_TBLS as usize],
    pub ac_huff_tbl_ptrs: [*mut JHUFF_TBL; NUM_HUFF_TBLS as usize],

    pub arith_dc_L: [u8; NUM_ARITH_TBLS as usize],
    pub arith_dc_U: [u8; NUM_ARITH_TBLS as usize],
    pub arith_ac_K: [u8; NUM_ARITH_TBLS as usize],

    pub num_scans: c_int,
    pub scan_info: *const c_void, // jpeg_scan_info

    pub raw_data_in: boolean,
    pub arith_code: boolean,
    pub optimize_coding: boolean,
    pub CCIR601_sampling: boolean,
    pub smoothing_factor: c_int,
    pub dct_method: J_DCT_METHOD,

    pub restart_interval: c_uint,
    pub restart_in_rows: c_int,

    pub write_JFIF_header: boolean,
    pub JFIF_major_version: u8,
    pub JFIF_minor_version: u8,
    pub density_unit: u8,
    pub X_density: u16,
    pub Y_density: u16,
    pub write_Adobe_marker: boolean,

    // State variables (partial - the actual struct is larger)
    pub next_scanline: JDIMENSION,

    // ... more fields omitted for brevity
    _padding: [u8; 256], // Padding to account for additional fields
}

/// Decompression struct
#[repr(C)]
pub struct jpeg_decompress_struct {
    // Common fields
    pub err: *mut jpeg_error_mgr,
    pub mem: *mut jpeg_memory_mgr,
    pub progress: *mut jpeg_progress_mgr,
    pub client_data: *mut c_void,
    pub is_decompressor: boolean,
    pub global_state: c_int,

    // Source manager
    pub src: *mut jpeg_source_mgr,

    // Basic image info
    pub image_width: JDIMENSION,
    pub image_height: JDIMENSION,
    pub num_components: c_int,
    pub jpeg_color_space: J_COLOR_SPACE,

    // Decompression parameters
    pub out_color_space: J_COLOR_SPACE,
    pub scale_num: c_uint,
    pub scale_denom: c_uint,
    pub output_gamma: f64,
    pub buffered_image: boolean,
    pub raw_data_out: boolean,
    pub dct_method: J_DCT_METHOD,
    pub do_fancy_upsampling: boolean,
    pub do_block_smoothing: boolean,
    pub quantize_colors: boolean,
    pub dither_mode: c_int,
    pub two_pass_quantize: boolean,
    pub desired_number_of_colors: c_int,
    pub enable_1pass_quant: boolean,
    pub enable_external_quant: boolean,
    pub enable_2pass_quant: boolean,

    // Output info
    pub output_width: JDIMENSION,
    pub output_height: JDIMENSION,
    pub out_color_components: c_int,
    pub output_components: c_int,
    pub rec_outbuf_height: c_int,

    // Color quantization info
    pub actual_number_of_colors: c_int,
    pub colormap: JSAMPARRAY,

    // State variables
    pub output_scanline: JDIMENSION,
    pub input_scan_number: c_int,
    pub input_iMCU_row: JDIMENSION,
    pub output_scan_number: c_int,
    pub output_iMCU_row: JDIMENSION,

    // ... more fields omitted
    _padding: [u8; 512], // Padding to account for additional fields
}

// External function declarations
extern "C" {
    // Error handling
    pub fn jpeg_std_error(err: *mut jpeg_error_mgr) -> *mut jpeg_error_mgr;

    // Compression
    pub fn jpeg_CreateCompress(
        cinfo: *mut jpeg_compress_struct,
        version: c_int,
        structsize: size_t,
    );
    pub fn jpeg_destroy_compress(cinfo: *mut jpeg_compress_struct);
    pub fn jpeg_set_defaults(cinfo: *mut jpeg_compress_struct);
    pub fn jpeg_set_quality(
        cinfo: *mut jpeg_compress_struct,
        quality: c_int,
        force_baseline: boolean,
    );
    pub fn jpeg_start_compress(cinfo: *mut jpeg_compress_struct, write_all_tables: boolean);
    pub fn jpeg_write_scanlines(
        cinfo: *mut jpeg_compress_struct,
        scanlines: JSAMPARRAY,
        num_lines: JDIMENSION,
    ) -> JDIMENSION;
    pub fn jpeg_finish_compress(cinfo: *mut jpeg_compress_struct);
    pub fn jpeg_mem_dest(
        cinfo: *mut jpeg_compress_struct,
        outbuffer: *mut *mut c_uchar,
        outsize: *mut c_ulong,
    );
    pub fn jpeg_simple_progression(cinfo: *mut jpeg_compress_struct);

    // Decompression
    pub fn jpeg_CreateDecompress(
        cinfo: *mut jpeg_decompress_struct,
        version: c_int,
        structsize: size_t,
    );
    pub fn jpeg_destroy_decompress(cinfo: *mut jpeg_decompress_struct);
    pub fn jpeg_read_header(cinfo: *mut jpeg_decompress_struct, require_image: boolean) -> c_int;
    pub fn jpeg_start_decompress(cinfo: *mut jpeg_decompress_struct) -> boolean;
    pub fn jpeg_read_scanlines(
        cinfo: *mut jpeg_decompress_struct,
        scanlines: JSAMPARRAY,
        max_lines: JDIMENSION,
    ) -> JDIMENSION;
    pub fn jpeg_finish_decompress(cinfo: *mut jpeg_decompress_struct) -> boolean;
    pub fn jpeg_mem_src(
        cinfo: *mut jpeg_decompress_struct,
        inbuffer: *const c_uchar,
        insize: c_ulong,
    );
}

// Helper macros
#[macro_export]
macro_rules! jpeg_create_compress {
    ($cinfo:expr) => {
        unsafe {
            jpeg_CreateCompress(
                $cinfo,
                JPEG_LIB_VERSION,
                std::mem::size_of::<jpeg_compress_struct>() as size_t,
            )
        }
    };
}

#[macro_export]
macro_rules! jpeg_create_decompress {
    ($cinfo:expr) => {
        unsafe {
            jpeg_CreateDecompress(
                $cinfo,
                JPEG_LIB_VERSION,
                std::mem::size_of::<jpeg_decompress_struct>() as size_t,
            )
        }
    };
}

// ============================================================================
// Butteraugli FFI bindings
// ============================================================================

/// Butteraugli error codes
pub const BUTTERAUGLI_OK: c_int = 0;
pub const BUTTERAUGLI_ERROR_MEMORY: c_int = 1;
pub const BUTTERAUGLI_ERROR_INVALID_INPUT: c_int = 2;
pub const BUTTERAUGLI_ERROR_INTERNAL: c_int = 3;

extern "C" {
    /// Compute butteraugli score between two linear RGB images.
    ///
    /// Both images must be linear RGB (not sRGB) with values in [0, 1].
    /// Data layout: row-major, 3 channels interleaved (RGBRGBRGB...).
    ///
    /// # Parameters
    /// - `rgb0`, `rgb1`: Linear RGB image data, width * height * 3 floats each
    /// - `width`, `height`: Image dimensions
    /// - `intensity_target`: Nits corresponding to 1.0 (default 80.0)
    /// - `out_score`: Output butteraugli score (max of diffmap)
    ///
    /// # Returns
    /// `BUTTERAUGLI_OK` on success.
    pub fn butteraugli_compare(
        rgb0: *const f32,
        rgb1: *const f32,
        width: size_t,
        height: size_t,
        intensity_target: f32,
        out_score: *mut f64,
    ) -> c_int;

    /// Compute butteraugli score with full parameters.
    ///
    /// # Parameters
    /// - `rgb0`, `rgb1`: Linear RGB image data
    /// - `width`, `height`: Image dimensions
    /// - `hf_asymmetry`: High-frequency asymmetry (default 1.0)
    /// - `xmul`: X channel multiplier (default 1.0)
    /// - `intensity_target`: Nits for 1.0 (default 80.0)
    /// - `out_score`: Output butteraugli score
    /// - `out_diffmap`: Optional output diffmap (width * height floats, or null)
    ///
    /// # Returns
    /// `BUTTERAUGLI_OK` on success.
    pub fn butteraugli_compare_full(
        rgb0: *const f32,
        rgb1: *const f32,
        width: size_t,
        height: size_t,
        hf_asymmetry: f32,
        xmul: f32,
        intensity_target: f32,
        out_score: *mut f64,
        out_diffmap: *mut f32,
    ) -> c_int;

    /// Convert sRGB u8 to linear RGB for butteraugli input.
    ///
    /// # Parameters
    /// - `srgb`: Input sRGB data (width * height * 3 bytes)
    /// - `width`, `height`: Image dimensions
    /// - `out_linear`: Output linear RGB (width * height * 3 floats, pre-allocated)
    pub fn butteraugli_srgb_to_linear(
        srgb: *const u8,
        width: size_t,
        height: size_t,
        out_linear: *mut f32,
    );

    /// Compute butteraugli Gamma function value (for testing FastLog2f).
    pub fn butteraugli_gamma(v: f32) -> f32;

    /// Compute butteraugli FastLog2f (for testing).
    pub fn butteraugli_fast_log2f(v: f32) -> f32;

    /// Compute OpsinDynamicsImage output (XYB values).
    /// NOTE: Uses simplified blur - for exact parity use butteraugli_compare_full.
    pub fn butteraugli_opsin_dynamics(
        linear_rgb: *const f32,
        width: size_t,
        height: size_t,
        intensity_target: f32,
        out_xyb: *mut f32,
    ) -> c_int;

    // ========================================================================
    // Step-by-step intermediate value extraction for divergence debugging
    // ========================================================================

    /// Compute Gaussian blur on a single plane.
    ///
    /// # Parameters
    /// - `input`: Input plane data (width * height floats)
    /// - `width`, `height`: Image dimensions
    /// - `sigma`: Blur sigma
    /// - `out_blurred`: Pre-allocated output (width * height floats)
    pub fn butteraugli_blur(
        input: *const f32,
        width: size_t,
        height: size_t,
        sigma: f32,
        out_blurred: *mut f32,
    ) -> c_int;

    /// Compute frequency separation from LINEAR RGB image.
    ///
    /// Internally applies OpsinDynamicsImage and SeparateFrequencies to produce
    /// the frequency-decomposed PsychoImage bands.
    ///
    /// Returns LF (XYB), MF (XYB), HF (XY), UHF (XY) planes.
    ///
    /// # Parameters
    /// - `linear_rgb`: Input LINEAR RGB data (width * height * 3 floats, interleaved)
    /// - `width`, `height`: Image dimensions (minimum 8x8)
    /// - `intensity_target`: Nits for 1.0
    /// - `out_*`: Pre-allocated output planes (width * height floats each), or null if not needed
    pub fn butteraugli_separate_frequencies(
        linear_rgb: *const f32,
        width: size_t,
        height: size_t,
        intensity_target: f32,
        out_lf_x: *mut f32,
        out_lf_y: *mut f32,
        out_lf_b: *mut f32,
        out_mf_x: *mut f32,
        out_mf_y: *mut f32,
        out_mf_b: *mut f32,
        out_hf_x: *mut f32,
        out_hf_y: *mut f32,
        out_uhf_x: *mut f32,
        out_uhf_y: *mut f32,
    ) -> c_int;

    /// Compute Malta filter on a single plane.
    ///
    /// # Parameters
    /// - `input`: Input plane data (width * height floats)
    /// - `width`, `height`: Image dimensions
    /// - `use_lf`: If non-zero, use MaltaUnitLF; otherwise use MaltaUnit
    /// - `out_malta`: Pre-allocated output (width * height floats)
    pub fn butteraugli_malta(
        input: *const f32,
        width: size_t,
        height: size_t,
        use_lf: c_int,
        out_malta: *mut f32,
    ) -> c_int;

    /// Compute mask from linear RGB image.
    ///
    /// Internally creates ButteraugliComparator and calls Mask() to produce
    /// the perceptual masking values.
    ///
    /// # Parameters
    /// - `linear_rgb`: Input LINEAR RGB data (width * height * 3 floats, interleaved)
    /// - `width`, `height`: Image dimensions (minimum 8x8)
    /// - `intensity_target`: Nits for 1.0
    /// - `out_mask`: Pre-allocated output (width * height floats)
    pub fn butteraugli_compute_mask(
        linear_rgb: *const f32,
        width: size_t,
        height: size_t,
        intensity_target: f32,
        out_mask: *mut f32,
    ) -> c_int;
}

/// Safe wrapper to compute butteraugli score between two sRGB images.
///
/// This handles the sRGB to linear conversion and calls the C++ butteraugli.
///
/// # Safety
/// The input slices must have length `width * height * 3`.
#[cfg(feature = "butteraugli")]
pub unsafe fn compute_butteraugli_cpp(
    srgb0: &[u8],
    srgb1: &[u8],
    width: usize,
    height: usize,
    intensity_target: f32,
) -> Result<f64, c_int> {
    let num_pixels = width * height;
    assert_eq!(srgb0.len(), num_pixels * 3);
    assert_eq!(srgb1.len(), num_pixels * 3);

    // Allocate buffers for linear RGB
    let mut linear0 = vec![0.0f32; num_pixels * 3];
    let mut linear1 = vec![0.0f32; num_pixels * 3];

    // Convert sRGB to linear
    butteraugli_srgb_to_linear(srgb0.as_ptr(), width, height, linear0.as_mut_ptr());
    butteraugli_srgb_to_linear(srgb1.as_ptr(), width, height, linear1.as_mut_ptr());

    // Compute butteraugli
    let mut score = 0.0f64;
    let result = butteraugli_compare(
        linear0.as_ptr(),
        linear1.as_ptr(),
        width,
        height,
        intensity_target,
        &mut score,
    );

    if result == BUTTERAUGLI_OK {
        Ok(score)
    } else {
        Err(result)
    }
}

// ============================================================================
// XYB Color Conversion FFI
// ============================================================================

extern "C" {
    /// Convert sRGB u8 to linear RGB float [0, 1]
    pub fn jpegli_srgb_to_linear(
        srgb: *const u8,
        width: size_t,
        height: size_t,
        out_linear: *mut f32,
    );

    /// Convert linear RGB to XYB (unscaled, as used in butteraugli)
    pub fn jpegli_linear_to_xyb(
        linear_rgb: *const f32,
        width: size_t,
        height: size_t,
        intensity_target: f32,
        out_xyb: *mut f32,
    );

    /// Convert XYB to scaled XYB [0, 1] range (as stored in JPEG)
    pub fn jpegli_scale_xyb(xyb: *mut f32, width: size_t, height: size_t);

    /// All-in-one: sRGB u8 -> scaled XYB
    pub fn jpegli_srgb_to_scaled_xyb(
        srgb: *const u8,
        width: size_t,
        height: size_t,
        intensity_target: f32,
        out_scaled_xyb: *mut f32,
    );

    /// Get XYB constants for verification
    pub fn jpegli_get_xyb_constants(
        opsin_matrix: *mut f32,
        opsin_bias: *mut f32,
        scaled_xyb_offset: *mut f32,
        scaled_xyb_scale: *mut f32,
    );

    /// PQ EOTF (display from encoded)
    pub fn jpegli_pq_eotf(encoded: f32) -> f32;

    /// PQ inverse EOTF (encoded from display)
    pub fn jpegli_pq_inv_eotf(display: f32) -> f32;

    /// HLG EOTF (display from encoded)
    pub fn jpegli_hlg_eotf(encoded: f32) -> f32;

    /// HLG inverse EOTF (encoded from display)
    pub fn jpegli_hlg_inv_eotf(display: f32) -> f32;

    /// sRGB to linear (single value)
    pub fn jpegli_srgb_to_linear_single(srgb: f32) -> f32;

    /// Linear to sRGB (single value)
    pub fn jpegli_linear_to_srgb_single(linear: f32) -> f32;

    /// Rec2408 tone mapping
    pub fn jpegli_rec2408_tone_map(
        source_nits: f32,
        target_nits: f32,
        luminances: *const f32,
        rgb: *mut f32,
    );

    /// HLG OOTF
    pub fn jpegli_hlg_ootf(
        source_nits: f32,
        target_nits: f32,
        luminances: *const f32,
        rgb: *mut f32,
    );

    /// Gamut mapping
    pub fn jpegli_gamut_map(preserve_saturation: f32, luminances: *const f32, rgb: *mut f32);

    // ========================================================================
    // Fast Math Functions (for AQ parity testing)
    // ========================================================================

    /// Fast log2 approximation (L1 error ~3.9E-6)
    pub fn jpegli_fast_log2f(x: f32) -> f32;

    /// Fast pow2 approximation (max relative error ~3e-7)
    pub fn jpegli_fast_pow2f(x: f32) -> f32;

    /// Fast power: base^exponent
    pub fn jpegli_fast_powf(base: f32, exponent: f32) -> f32;

    /// ComputeMask - perceptual masking curve
    pub fn jpegli_compute_mask(out_val: f32) -> f32;

    /// MaskingSqrt
    pub fn jpegli_masking_sqrt(v: f32) -> f32;

    /// RatioOfDerivativesOfCubicRootToSimpleGamma
    pub fn jpegli_ratio_of_derivatives(v: f32, invert: std::os::raw::c_int) -> f32;
}

// ============================================================================
// Safe Rust wrappers for XYB conversion
// ============================================================================

/// Safe wrapper for C++ sRGB to scaled XYB conversion.
#[cfg(feature = "butteraugli")]
pub fn cpp_srgb_to_scaled_xyb(
    srgb: &[u8],
    width: usize,
    height: usize,
    intensity_target: f32,
) -> Vec<f32> {
    assert_eq!(srgb.len(), width * height * 3);
    let mut out = vec![0.0f32; width * height * 3];
    unsafe {
        jpegli_srgb_to_scaled_xyb(
            srgb.as_ptr(),
            width,
            height,
            intensity_target,
            out.as_mut_ptr(),
        );
    }
    out
}

/// Safe wrapper to get XYB constants from C++.
#[cfg(feature = "butteraugli")]
pub fn cpp_get_xyb_constants() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut opsin_matrix = vec![0.0f32; 9];
    let mut opsin_bias = vec![0.0f32; 3];
    let mut scaled_xyb_offset = vec![0.0f32; 3];
    let mut scaled_xyb_scale = vec![0.0f32; 3];
    unsafe {
        jpegli_get_xyb_constants(
            opsin_matrix.as_mut_ptr(),
            opsin_bias.as_mut_ptr(),
            scaled_xyb_offset.as_mut_ptr(),
            scaled_xyb_scale.as_mut_ptr(),
        );
    }
    (
        opsin_matrix,
        opsin_bias,
        scaled_xyb_offset,
        scaled_xyb_scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(DCTSIZE, 8);
        assert_eq!(DCTSIZE2, 64);
        assert_eq!(JPEG_LIB_VERSION, 62);
    }

    #[test]
    fn test_color_space_values() {
        assert_eq!(JCS_UNKNOWN, 0);
        assert_eq!(JCS_GRAYSCALE, 1);
        assert_eq!(JCS_RGB, 2);
        assert_eq!(JCS_YCbCr, 3);
    }

    /// Validates that the instrumented C++ FFI is properly linked.
    ///
    /// If this test fails to link, the jpegli-cpp submodule is on the wrong branch.
    /// Fix with: `cd internal/jpegli-cpp && git checkout instrumented`
    #[test]
    fn test_instrumented_ffi_available() {
        // Call fast math functions - these only exist in the instrumented build
        unsafe {
            // Test fast_log2f: log2(1.0) = 0.0
            let log2_1 = jpegli_fast_log2f(1.0);
            assert!(
                log2_1.abs() < 0.001,
                "jpegli_fast_log2f(1.0) = {}, expected ~0.0",
                log2_1
            );

            // Test fast_pow2f: 2^0 = 1.0
            let pow2_0 = jpegli_fast_pow2f(0.0);
            assert!(
                (pow2_0 - 1.0).abs() < 0.001,
                "jpegli_fast_pow2f(0.0) = {}, expected ~1.0",
                pow2_0
            );

            // Test fast_log2f: log2(8.0) = 3.0
            let log2_8 = jpegli_fast_log2f(8.0);
            assert!(
                (log2_8 - 3.0).abs() < 0.01,
                "jpegli_fast_log2f(8.0) = {}, expected ~3.0",
                log2_8
            );

            // Test roundtrip: 2^(log2(x)) ≈ x
            let x = 42.5;
            let roundtrip = jpegli_fast_pow2f(jpegli_fast_log2f(x));
            assert!(
                (roundtrip - x).abs() < 0.01,
                "Roundtrip failed: 2^log2({}) = {}",
                x,
                roundtrip
            );
        }

        println!("✓ Instrumented C++ FFI is properly linked and functional");
    }

    /// Test that compute_mask produces reasonable values.
    #[test]
    fn test_compute_mask_ffi() {
        unsafe {
            // ComputeMask should produce finite, reasonable values
            for v in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
                let mask = jpegli_compute_mask(v);
                assert!(
                    mask.is_finite(),
                    "jpegli_compute_mask({}) returned non-finite: {}",
                    v,
                    mask
                );
            }
        }
        println!("✓ jpegli_compute_mask FFI works");
    }
}
