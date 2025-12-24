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

    // Note: Full FFI tests require linking with the C++ library
    // These are integration tests that would be run with the full build
}
