# Decoder Extras Specification

## Overview

This document specifies additions to jpegli-rs for preserving and accessing JPEG metadata segments and MPF (Multi-Picture Format) secondary images during decode. These features enable round-trip editing workflows where metadata must be preserved, and support for formats like UltraHDR that embed secondary images.

## Goals

1. **Preserve metadata during decode** - Buffer APP segments so they survive decode and can be re-injected on encode
2. **Support MPF secondary images** - Extract gain maps, depth maps, etc. embedded via MPF
3. **Granular control** - User chooses what to keep (memory vs. completeness tradeoff)
4. **Sensible defaults** - Keep copyright-relevant and rendering-relevant data by default
5. **Lazy parsing** - Buffer raw bytes, parse on demand
6. **No API breakage** - Existing decode methods unchanged, extras are opt-in

## JPEG Segment Reference

| Marker | Name | Contains | Default |
|--------|------|----------|---------|
| APP0 (0xE0) | JFIF | Version, DPI/density, aspect ratio, tiny thumbnail | Keep |
| APP1 (0xE1) | EXIF | Orientation, camera info, GPS, copyright, thumbnail | Keep |
| APP1 (0xE1) | XMP | Edit history, copyright, HDR gainmap metadata | Keep |
| APP2 (0xE2) | ICC | Color profile (can be chunked across multiple APP2) | Keep |
| APP2 (0xE2) | MPF | Multi-Picture Format directory (offsets to secondary images) | Keep |
| APP13 (0xED) | IPTC/IIM | Copyright, creator, caption, keywords, location | Keep |
| APP14 (0xEE) | Adobe | Color transform flag (RGB/CMYK/YCbCr) - affects decode | Keep |
| COM (0xFE) | Comment | Free text, sometimes copyright | Keep |
| APP3-12, APP15 | Various | Vendor-specific (Exif thumbnails in APP1, etc.) | Drop |

## MPF Secondary Image Types

MPF (CIPA DC-007) allows embedding multiple images in one JPEG file.

| Type Code | Name | Description | Default |
|-----------|------|-------------|---------|
| 0x000000 | Undefined | Used for gain maps (UltraHDR) | Keep |
| 0x010001 | Large Thumbnail (VGA) | ~640x480 preview | Drop |
| 0x010002 | Large Thumbnail (Full HD) | ~1920x1080 preview | Drop |
| 0x020001 | Multi-Frame Panorama | Panorama component | Drop |
| 0x020002 | Multi-Frame Disparity | Depth/disparity map | Drop |
| 0x020003 | Multi-Frame Multi-Angle | Different viewing angle | Drop |
| 0x030000 | Baseline MP Primary | Primary image marker | N/A |

## API Design

### PreserveConfig

```rust
/// Configuration for what to preserve during decode
#[derive(Clone, Debug)]
pub struct PreserveConfig {
    // === Metadata segments ===

    /// APP0 JFIF - DPI/density for print
    pub jfif: bool,

    /// APP1 EXIF - orientation, camera, GPS, copyright
    pub exif: bool,

    /// APP1 XMP - edit history, copyright, gainmap metadata
    /// Note: Extended XMP (chunked across multiple APP1) is reassembled
    pub xmp: bool,

    /// APP2 ICC - color profile
    /// Note: Chunked ICC profiles are reassembled
    pub icc: IccPreserve,

    /// APP13 IPTC/IIM - copyright, creator, caption, keywords
    pub iptc: bool,

    /// APP14 Adobe - color transform flag
    pub adobe: bool,

    /// COM - comment markers (sometimes contain copyright)
    pub com: bool,

    /// Unknown APP markers (APP3-12, APP15 excluding known types)
    pub app_unknown: bool,

    // === MPF secondary images ===

    /// Undefined type - used for gain maps (UltraHDR)
    pub mpf_gainmaps: bool,

    /// Large thumbnails (VGA, Full HD)
    pub mpf_thumbnails: bool,

    /// Multi-frame images (panorama, multi-angle)
    pub mpf_multiframe: bool,

    /// Disparity/depth maps
    pub mpf_depth: bool,

    /// Custom filter for MPF images (overrides above if set)
    /// Called with (index, image_type, size_bytes) -> should_keep
    pub mpf_filter: Option<Arc<dyn Fn(usize, MpfImageType, u32) -> bool + Send + Sync>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IccPreserve {
    /// Keep all ICC profiles
    #[default]
    All,
    /// Drop well-known standard profiles (sRGB IEC61966-2.1, Display P3)
    /// Saves space when profile is implicit
    DropStandard,
    /// Keep no ICC profiles
    None,
}

impl Default for PreserveConfig {
    fn default() -> Self {
        Self {
            // Metadata - keep by default (copyright, rendering)
            jfif: true,
            exif: true,
            xmp: true,
            icc: IccPreserve::All,
            iptc: true,
            adobe: true,
            com: true,
            app_unknown: false,

            // MPF - keep gain maps, drop redundant previews
            mpf_gainmaps: true,
            mpf_thumbnails: false,
            mpf_multiframe: false,
            mpf_depth: false,
            mpf_filter: None,
        }
    }
}

impl PreserveConfig {
    /// Preserve nothing (minimal memory)
    pub fn none() -> Self { ... }

    /// Preserve everything
    pub fn all() -> Self { ... }

    // Builder methods
    pub fn jfif(mut self, keep: bool) -> Self { ... }
    pub fn exif(mut self, keep: bool) -> Self { ... }
    pub fn xmp(mut self, keep: bool) -> Self { ... }
    pub fn icc(mut self, mode: IccPreserve) -> Self { ... }
    pub fn iptc(mut self, keep: bool) -> Self { ... }
    pub fn adobe(mut self, keep: bool) -> Self { ... }
    pub fn com(mut self, keep: bool) -> Self { ... }
    pub fn mpf_gainmaps(mut self, keep: bool) -> Self { ... }
    pub fn mpf_thumbnails(mut self, keep: bool) -> Self { ... }
    pub fn mpf_depth(mut self, keep: bool) -> Self { ... }

    /// Custom MPF filter (called for each secondary image)
    pub fn mpf_filter<F>(mut self, f: F) -> Self
    where
        F: Fn(usize, MpfImageType, u32) -> bool + Send + Sync + 'static
    { ... }
}
```

### Decoder Integration

```rust
impl Decoder {
    /// Configure what metadata/images to preserve (default: PreserveConfig::default())
    pub fn preserve(mut self, config: PreserveConfig) -> Self {
        self.config.preserve = config;
        self
    }

    /// Convenience: preserve nothing extra (minimal memory)
    pub fn preserve_none(self) -> Self {
        self.preserve(PreserveConfig::none())
    }

    /// Convenience: preserve everything
    pub fn preserve_all(self) -> Self {
        self.preserve(PreserveConfig::all())
    }
}

impl DecoderConfig {
    /// What to preserve during decode
    pub preserve: PreserveConfig,  // default: PreserveConfig::default()
}
```

### DecodedExtras

```rust
/// Preserved metadata and secondary images from decode
///
/// Raw bytes are buffered during decode. Parsing is lazy (on first access).
pub struct DecodedExtras {
    // Raw buffered data
    segments: Vec<PreservedSegment>,
    secondary_images: Vec<PreservedMpfImage>,

    // Lazy parse cache
    xmp_cache: OnceCell<Option<String>>,
    icc_cache: OnceCell<Option<Vec<u8>>>,
    mpf_cache: OnceCell<Option<MpfDirectory>>,
}

#[derive(Clone, Debug)]
pub struct PreservedSegment {
    pub marker: u8,
    pub data: Vec<u8>,
    pub segment_type: SegmentType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentType {
    Jfif,
    Exif,
    Xmp,
    XmpExtended,
    Icc,       // May be one chunk of many
    Mpf,
    Iptc,
    Adobe,
    Comment,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct PreservedMpfImage {
    pub mpf_index: usize,
    pub image_type: MpfImageType,
    pub data: Vec<u8>,
}

impl DecodedExtras {
    // === Raw segment access ===

    /// All preserved segments
    pub fn segments(&self) -> &[PreservedSegment] { ... }

    /// Segments by type
    pub fn segments_by_type(&self, typ: SegmentType) -> impl Iterator<Item = &PreservedSegment> { ... }

    /// Segments by marker
    pub fn segments_by_marker(&self, marker: u8) -> impl Iterator<Item = &PreservedSegment> { ... }

    // === Lazy-parsed metadata access ===

    /// JFIF info (density, aspect ratio)
    pub fn jfif(&self) -> Option<JfifInfo> { ... }

    /// EXIF data (returns raw bytes for external parsing)
    pub fn exif(&self) -> Option<&[u8]> { ... }

    /// XMP string (reassembled from extended XMP if needed)
    pub fn xmp(&self) -> Option<&str> { ... }

    /// ICC profile (reassembled from chunks)
    pub fn icc_profile(&self) -> Option<&[u8]> { ... }

    /// Check if ICC is a standard profile (sRGB, Display P3)
    pub fn icc_is_standard(&self) -> Option<StandardProfile> { ... }

    /// IPTC data (returns raw bytes for external parsing)
    pub fn iptc(&self) -> Option<&[u8]> { ... }

    /// Adobe segment info
    pub fn adobe(&self) -> Option<AdobeInfo> { ... }

    /// Comment strings
    pub fn comments(&self) -> impl Iterator<Item = &str> { ... }

    /// MPF directory (parsed from APP2)
    pub fn mpf(&self) -> Option<&MpfDirectory> { ... }

    // === MPF secondary images ===

    /// All preserved secondary images
    pub fn secondary_images(&self) -> &[PreservedMpfImage] { ... }

    /// Get secondary image by MPF index
    pub fn secondary_image(&self, mpf_index: usize) -> Option<&[u8]> { ... }

    /// Get first gain map (first Undefined-type secondary image)
    pub fn gainmap(&self) -> Option<&[u8]> { ... }

    /// Get depth/disparity map if present
    pub fn depth_map(&self) -> Option<&[u8]> { ... }

    // === For encoder round-trip ===

    /// Convert preserved segments to format suitable for encoder injection
    /// Maintains original order, excludes MPF (encoder regenerates it)
    pub fn to_encoder_segments(&self) -> Vec<(u8, Vec<u8>)> { ... }

    /// Get segments that should go before image data
    pub fn to_header_segments(&self) -> Vec<(u8, Vec<u8>)> { ... }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandardProfile {
    SrgbIec61966,
    DisplayP3,
}

#[derive(Clone, Debug)]
pub struct JfifInfo {
    pub version_major: u8,
    pub version_minor: u8,
    pub density_units: DensityUnits,
    pub x_density: u16,
    pub y_density: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityUnits {
    None,       // Aspect ratio only
    PixelsPerInch,
    PixelsPerCm,
}

#[derive(Clone, Debug)]
pub struct AdobeInfo {
    pub version: u16,
    pub color_transform: AdobeColorTransform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdobeColorTransform {
    Unknown,
    YCbCr,
    Ycck,
}
```

### MPF Types

```rust
/// MPF directory parsed from APP2
#[derive(Clone, Debug)]
pub struct MpfDirectory {
    pub version: [u8; 4],
    pub images: Vec<MpfEntry>,
}

#[derive(Clone, Debug)]
pub struct MpfEntry {
    pub image_type: MpfImageType,
    pub offset: u32,
    pub size: u32,
    pub dependent_image1: Option<u16>,
    pub dependent_image2: Option<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpfImageType {
    /// Used for gain maps (UltraHDR)
    Undefined,
    /// Primary baseline image
    BaselinePrimary,
    /// VGA-equivalent thumbnail
    LargeThumbnailVga,
    /// Full HD-equivalent thumbnail
    LargeThumbnailFullHd,
    /// Panorama component
    Panorama,
    /// Depth/disparity map
    Disparity,
    /// Multi-angle view
    MultiAngle,
    /// Unknown type code
    Other(u32),
}

impl MpfImageType {
    pub fn from_type_code(code: u32) -> Self { ... }
    pub fn to_type_code(self) -> u32 { ... }

    /// Is this a thumbnail/preview type?
    pub fn is_thumbnail(&self) -> bool { ... }

    /// Is this a gain map (Undefined type)?
    pub fn is_gainmap(&self) -> bool { ... }
}
```

### DecodedImage Changes

```rust
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Vec<u8>,

    /// Preserved metadata and secondary images (if requested)
    extras: Option<DecodedExtras>,
}

impl DecodedImage {
    /// Access preserved extras
    /// Returns None if preservation wasn't configured
    pub fn extras(&self) -> Option<&DecodedExtras> {
        self.extras.as_ref()
    }

    /// Take ownership of extras
    pub fn into_parts(self) -> (Vec<u8>, u32, u32, PixelFormat, Option<DecodedExtras>) {
        (self.data, self.width, self.height, self.format, self.extras)
    }
}
```

## Usage Examples

### Basic UltraHDR Decode

```rust
use jpegli::decoder::{Decoder, PixelFormat};

// Default config preserves XMP and gain maps
let result = Decoder::new()
    .output_format(PixelFormat::Rgb)
    .decode(&ultrahdr_bytes)?;

let sdr_pixels = &result.data;
let extras = result.extras().expect("extras preserved by default");

// Get gain map metadata from XMP
let xmp = extras.xmp().ok_or("no XMP")?;
let metadata = ultrahdr::parse_xmp(xmp)?;

// Get gain map JPEG
let gainmap_jpeg = extras.gainmap().ok_or("no gain map")?;

// Decode gain map
let gainmap = Decoder::new()
    .output_format(PixelFormat::Gray)
    .preserve(PreserveConfig::none())  // Don't need extras for gain map
    .decode(gainmap_jpeg)?;

// Apply (ultrahdr crate does the math)
let hdr = ultrahdr::apply_gainmap(&sdr_pixels, &gainmap.data, &metadata, 4.0);
```

### Memory-Constrained (WASM)

```rust
// Only keep essential metadata, skip large secondary images
let config = PreserveConfig::default()
    .mpf_filter(|_idx, typ, size| {
        typ.is_gainmap() && size < 500_000  // Gain maps under 500KB only
    });

let result = Decoder::new()
    .preserve(config)
    .decode(&data)?;
```

### Round-Trip Editing

```rust
// Decode with full preservation
let result = Decoder::new()
    .preserve(PreserveConfig::all())
    .decode(&original_jpeg)?;

// Edit pixels...
let edited_pixels = edit(&result.data);

// Get preserved segments for re-injection
let extras = result.extras().unwrap();
let segments = extras.to_encoder_segments();

// Re-encode with preserved metadata
let new_jpeg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(segments)
    .encode_oneshot(&edited_pixels, w, h, PixelLayout::Rgb8Srgb)?;
```

### Print Workflow (DPI matters)

```rust
let result = Decoder::new()
    .preserve(PreserveConfig::default())
    .decode(&data)?;

let extras = result.extras().unwrap();
if let Some(jfif) = extras.jfif() {
    println!("DPI: {}x{} {:?}", jfif.x_density, jfif.y_density, jfif.density_units);
}
```

## Implementation Notes

### Buffering During Parse

The parser already scans all markers. Changes needed:
1. Add `preserved_segments: Vec<PreservedSegment>` to parser state
2. Check `PreserveConfig` when encountering each APP marker
3. For MPF, parse directory during header scan, then buffer secondary images after primary EOI based on filter

### Extended XMP Handling

XMP can span multiple APP1 markers using Adobe's extended XMP spec:
1. Primary XMP in first APP1 (starts with `http://ns.adobe.com/xap/1.0/\0`)
2. Extended chunks in subsequent APP1s (start with `http://ns.adobe.com/xmp/extension/\0`)
3. Extended chunks have GUID + total length + offset for reassembly

The `xmp()` accessor should reassemble transparently.

### ICC Chunk Handling

ICC profiles > 64KB are chunked across multiple APP2 markers:
1. Each chunk starts with `ICC_PROFILE\0`
2. Followed by chunk index (1-based) and total chunks
3. Chunks may be out of order

The `icc_profile()` accessor should reassemble transparently.

### MPF Secondary Image Extraction

After primary image EOI:
1. Use MPF directory offsets to locate secondary images
2. Each secondary is a complete JPEG (SOI to EOI)
3. Apply `mpf_filter` to decide which to buffer
4. Secondary images are stored as raw JPEG bytes

### Memory Considerations

Default config buffers:
- All metadata segments (~10-100KB typical)
- Gain map JPEG (~50KB-500KB typical for UltraHDR)
- Total: ~100KB-1MB additional

`PreserveConfig::none()` buffers nothing extra.
`PreserveConfig::all()` could buffer several MB if thumbnails/previews present.

## Encoder Additions (Separate Spec)

The encoder side needs corresponding features:
- `add_segment(marker, data)` - inject APP segment
- `add_mpf_image(jpeg, type)` - add secondary image with auto-MPF
- MPF offset calculation during `finish()`

See `encoder-extras-spec.md` for details.
