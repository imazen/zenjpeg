# Encoder Extras Specification

## Overview

This document specifies additions to jpegli-rs for injecting metadata segments and assembling MPF (Multi-Picture Format) secondary images during encode. Designed to complement `decoder-extras-spec.md` with seamless data transfer between decode and encode for round-trip workflows.

## Goals

1. **Segment injection** - Insert APP segments (EXIF, XMP, ICC, IPTC, etc.) into output
2. **MPF assembly** - Append secondary images with auto-generated MPF directory
3. **Easy transfer from decode** - `DecodedExtras` → encoder with minimal friction
4. **Modification support** - Edit/filter/add segments before encoding
5. **Correct ordering** - Segments placed in standard-compliant order
6. **No API breakage** - Existing encode methods unchanged, extras are additive

## Segment Ordering

JPEG segments must appear in specific order for compatibility:

```
SOI
APP0 (JFIF) - if present, must be first
APP1 (EXIF) - orientation needed early for some decoders
APP1 (XMP)
APP2 (ICC) - chunked if > 64KB
APP2 (MPF) - references secondary images after EOI
APP13 (IPTC)
APP14 (Adobe)
DQT, SOF, DHT, SOS...
[image data]
EOI
[Secondary image 1 - complete JPEG]
[Secondary image 2 - complete JPEG]
...
```

## API Design

### EncoderSegments - Transfer Type

```rust
/// Prepared segments for encoder injection
///
/// This is the bridge type between decoder and encoder.
/// Created from `DecodedExtras::to_encoder_segments()` or built manually.
#[derive(Clone, Debug, Default)]
pub struct EncoderSegments {
    // Ordered segments to inject
    segments: Vec<EncoderSegment>,
    // Secondary images to append after EOI
    mpf_images: Vec<MpfImage>,
}

#[derive(Clone, Debug)]
pub struct EncoderSegment {
    pub marker: u8,
    pub data: Vec<u8>,
    pub segment_type: SegmentType,
}

#[derive(Clone, Debug)]
pub struct MpfImage {
    pub image_type: MpfImageType,
    pub data: Vec<u8>,
}

impl EncoderSegments {
    /// Create empty segments
    pub fn new() -> Self { ... }

    // === From DecodedExtras ===

    /// Create from decoded extras (copies relevant segments)
    /// Excludes MPF directory (encoder regenerates it)
    pub fn from_extras(extras: &DecodedExtras) -> Self { ... }

    // === Segment access ===

    /// Get segment by type (first match)
    pub fn get(&self, typ: SegmentType) -> Option<&[u8]> { ... }

    /// Get all segments of a type
    pub fn get_all(&self, typ: SegmentType) -> Vec<&[u8]> { ... }

    /// Check if segment type is present
    pub fn has(&self, typ: SegmentType) -> bool { ... }

    // === Segment modification ===

    /// Add a segment (appended to appropriate position)
    pub fn add(&mut self, marker: u8, data: Vec<u8>, typ: SegmentType) -> &mut Self { ... }

    /// Add raw segment (type inferred from marker + data)
    pub fn add_raw(&mut self, marker: u8, data: Vec<u8>) -> &mut Self { ... }

    /// Remove all segments of a type
    pub fn remove(&mut self, typ: SegmentType) -> &mut Self { ... }

    /// Remove segments matching predicate
    pub fn remove_where<F: Fn(&EncoderSegment) -> bool>(&mut self, f: F) -> &mut Self { ... }

    /// Replace segment of a type (removes existing, adds new)
    pub fn replace(&mut self, marker: u8, data: Vec<u8>, typ: SegmentType) -> &mut Self { ... }

    // === Typed segment helpers ===

    /// Set/replace EXIF data
    pub fn set_exif(&mut self, data: Vec<u8>) -> &mut Self { ... }

    /// Set/replace XMP string
    pub fn set_xmp(&mut self, xmp: &str) -> &mut Self { ... }

    /// Modify XMP in place (no-op if no XMP present)
    pub fn modify_xmp<F: FnOnce(&str) -> String>(&mut self, f: F) -> &mut Self { ... }

    /// Set/replace ICC profile (auto-chunks if > 64KB)
    pub fn set_icc(&mut self, profile: Vec<u8>) -> &mut Self { ... }

    /// Remove ICC profile
    pub fn remove_icc(&mut self) -> &mut Self { ... }

    /// Set/replace IPTC data
    pub fn set_iptc(&mut self, data: Vec<u8>) -> &mut Self { ... }

    /// Set JFIF density/DPI
    pub fn set_jfif(&mut self, info: JfifInfo) -> &mut Self { ... }

    /// Add comment
    pub fn add_comment(&mut self, comment: &str) -> &mut Self { ... }

    // === MPF secondary images ===

    /// Add secondary image (will be appended after EOI)
    pub fn add_mpf_image(&mut self, data: Vec<u8>, typ: MpfImageType) -> &mut Self { ... }

    /// Add gain map (convenience for MpfImageType::Undefined)
    pub fn add_gainmap(&mut self, jpeg_data: Vec<u8>) -> &mut Self { ... }

    /// Add depth map
    pub fn add_depth_map(&mut self, jpeg_data: Vec<u8>) -> &mut Self { ... }

    /// Get MPF images
    pub fn mpf_images(&self) -> &[MpfImage] { ... }

    /// Remove all MPF images
    pub fn clear_mpf_images(&mut self) -> &mut Self { ... }

    /// Remove MPF images by type
    pub fn remove_mpf_images(&mut self, typ: MpfImageType) -> &mut Self { ... }

    // === Bulk operations ===

    /// Merge segments from another EncoderSegments
    /// Existing segments of same type are kept (use replace for override)
    pub fn merge(&mut self, other: &EncoderSegments) -> &mut Self { ... }

    /// Clear all segments
    pub fn clear(&mut self) -> &mut Self { ... }

    /// Keep only specified segment types
    pub fn retain(&mut self, types: &[SegmentType]) -> &mut Self { ... }
}
```

### DecodedExtras Integration

```rust
impl DecodedExtras {
    /// Convert to encoder segments for round-trip
    ///
    /// Includes: JFIF, EXIF, XMP, ICC, IPTC, Adobe, Comments
    /// Excludes: MPF directory (encoder regenerates), unknown segments
    ///
    /// Secondary images are included in the result.
    pub fn to_encoder_segments(&self) -> EncoderSegments {
        let mut segments = EncoderSegments::new();

        // Copy metadata segments
        for seg in &self.segments {
            match seg.segment_type {
                SegmentType::Jfif |
                SegmentType::Exif |
                SegmentType::Xmp |
                SegmentType::XmpExtended |
                SegmentType::Icc |
                SegmentType::Iptc |
                SegmentType::Adobe |
                SegmentType::Comment => {
                    segments.add(seg.marker, seg.data.clone(), seg.segment_type);
                }
                SegmentType::Mpf => {
                    // Skip - encoder regenerates MPF
                }
                SegmentType::Unknown => {
                    // Skip unknown by default
                }
            }
        }

        // Copy secondary images
        for img in &self.secondary_images {
            segments.add_mpf_image(img.data.clone(), img.image_type);
        }

        segments
    }

    /// Convert to encoder segments with custom filter
    pub fn to_encoder_segments_filtered<F>(&self, filter: F) -> EncoderSegments
    where
        F: Fn(&PreservedSegment) -> bool
    { ... }
}
```

### EncoderConfig Integration

```rust
impl EncoderConfig {
    /// Add prepared segments to encode output
    pub fn with_segments(mut self, segments: EncoderSegments) -> Self {
        self.segments = Some(segments);
        self
    }

    /// Add single segment (convenience)
    pub fn add_segment(mut self, marker: u8, data: Vec<u8>) -> Self {
        self.segments
            .get_or_insert_with(EncoderSegments::new)
            .add_raw(marker, data);
        self
    }

    /// Add APP1 segment (EXIF or XMP)
    pub fn add_app1(self, data: Vec<u8>) -> Self {
        self.add_segment(0xE1, data)
    }

    /// Add APP2 segment (ICC or MPF)
    pub fn add_app2(self, data: Vec<u8>) -> Self {
        self.add_segment(0xE2, data)
    }

    /// Set XMP metadata
    pub fn with_xmp(mut self, xmp: &str) -> Self {
        self.segments
            .get_or_insert_with(EncoderSegments::new)
            .set_xmp(xmp);
        self
    }

    /// Set ICC profile
    pub fn with_icc(mut self, profile: Vec<u8>) -> Self {
        self.segments
            .get_or_insert_with(EncoderSegments::new)
            .set_icc(profile);
        self
    }

    /// Set EXIF data
    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.segments
            .get_or_insert_with(EncoderSegments::new)
            .set_exif(exif);
        self
    }

    /// Add MPF secondary image
    pub fn add_mpf_image(mut self, jpeg: Vec<u8>, typ: MpfImageType) -> Self {
        self.segments
            .get_or_insert_with(EncoderSegments::new)
            .add_mpf_image(jpeg, typ);
        self
    }

    /// Add gain map (convenience)
    pub fn add_gainmap(self, jpeg: Vec<u8>) -> Self {
        self.add_mpf_image(jpeg, MpfImageType::Undefined)
    }
}
```

### Encoding Implementation

```rust
impl BytesEncoder {
    pub fn finish(self) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        // 1. Write SOI
        output.extend_from_slice(&[0xFF, 0xD8]);

        // 2. Write injected segments (ordered)
        if let Some(segments) = &self.config.segments {
            // JFIF must be first if present
            for seg in segments.segments_of_type(SegmentType::Jfif) {
                write_segment(&mut output, seg.marker, &seg.data);
            }

            // EXIF before other APP1
            for seg in segments.segments_of_type(SegmentType::Exif) {
                write_segment(&mut output, seg.marker, &seg.data);
            }

            // XMP (including extended)
            for seg in segments.segments_of_type(SegmentType::Xmp) {
                write_segment(&mut output, seg.marker, &seg.data);
            }
            for seg in segments.segments_of_type(SegmentType::XmpExtended) {
                write_segment(&mut output, seg.marker, &seg.data);
            }

            // ICC chunks
            for seg in segments.segments_of_type(SegmentType::Icc) {
                write_segment(&mut output, seg.marker, &seg.data);
            }

            // MPF directory (generated, not from segments)
            if !segments.mpf_images.is_empty() {
                // Placeholder - will patch offsets later
                let mpf_offset = output.len();
                let mpf_placeholder = generate_mpf_placeholder(segments.mpf_images.len());
                write_segment(&mut output, 0xE2, &mpf_placeholder);
            }

            // IPTC
            for seg in segments.segments_of_type(SegmentType::Iptc) {
                write_segment(&mut output, seg.marker, &seg.data);
            }

            // Adobe
            for seg in segments.segments_of_type(SegmentType::Adobe) {
                write_segment(&mut output, seg.marker, &seg.data);
            }

            // Comments
            for seg in segments.segments_of_type(SegmentType::Comment) {
                write_segment(&mut output, 0xFE, &seg.data);
            }
        }

        // 3. Write DQT, SOF, DHT, SOS, image data, EOI
        self.write_image_data(&mut output)?;

        // 4. Append MPF secondary images and patch offsets
        if let Some(segments) = &self.config.segments {
            if !segments.mpf_images.is_empty() {
                let primary_size = output.len();

                // Calculate offsets and patch MPF
                let mut offsets = Vec::new();
                let mut current_offset = primary_size;
                for img in &segments.mpf_images {
                    offsets.push((current_offset, img.data.len(), img.image_type));
                    current_offset += img.data.len();
                }

                patch_mpf_offsets(&mut output, mpf_offset, &offsets);

                // Append secondary images
                for img in &segments.mpf_images {
                    output.extend_from_slice(&img.data);
                }
            }
        }

        Ok(output)
    }
}
```

## Usage Examples

### Simple Round-Trip (No Modification)

```rust
// Decode
let decoded = Decoder::new().decode(&original)?;
let extras = decoded.extras().unwrap();

// Edit pixels
let edited = apply_filter(&decoded.data);

// Encode with same metadata
let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(extras.to_encoder_segments())
    .encode_oneshot(&edited, w, h, PixelLayout::Rgb8Srgb)?;
```

### Round-Trip with XMP Modification

```rust
let decoded = Decoder::new().decode(&original)?;
let mut segments = decoded.extras().unwrap().to_encoder_segments();

// Modify XMP
segments.modify_xmp(|xmp| {
    // Add edit history, update metadata, etc.
    format!("{}\n<!-- edited -->", xmp)
});

let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(segments)
    .encode_oneshot(&pixels, w, h, PixelLayout::Rgb8Srgb)?;
```

### UltraHDR Creation

```rust
// Compute gain map (ultrahdr crate)
let (gm_pixels, metadata) = ultrahdr::compute_gainmap(&hdr, &sdr, &config);

// Encode gain map
let gm_jpeg = EncoderConfig::grayscale(75.0)
    .encode_oneshot(&gm_pixels, gm_w, gm_h, PixelLayout::Gray8Srgb)?;

// Encode primary with gain map
let ultrahdr = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_xmp(&ultrahdr::generate_xmp(&metadata))
    .with_icc(srgb_profile)
    .add_gainmap(gm_jpeg)
    .encode_oneshot(&sdr, w, h, PixelLayout::Rgb8Srgb)?;
```

### UltraHDR Round-Trip (Edit SDR, Keep Gain Map)

```rust
// Decode UltraHDR
let decoded = Decoder::new().decode(&ultrahdr_bytes)?;
let extras = decoded.extras().unwrap();

// Get existing gain map
let gainmap_jpeg = extras.gainmap().unwrap();

// Edit SDR pixels
let edited_sdr = adjust_exposure(&decoded.data);

// Re-encode with same gain map and metadata
let mut segments = extras.to_encoder_segments();
// Gain map already included from to_encoder_segments()

let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(segments)
    .encode_oneshot(&edited_sdr, w, h, PixelLayout::Rgb8Srgb)?;
```

### Selective Metadata Transfer

```rust
let decoded = Decoder::new().decode(&original)?;
let extras = decoded.extras().unwrap();

// Build custom segments - only transfer some metadata
let mut segments = EncoderSegments::new();

// Keep EXIF (orientation, camera info)
if let Some(exif) = extras.exif() {
    segments.set_exif(exif.to_vec());
}

// Keep ICC
if let Some(icc) = extras.icc_profile() {
    segments.set_icc(icc.to_vec());
}

// New XMP (don't transfer old)
segments.set_xmp(&generate_new_xmp());

// Don't transfer: IPTC, comments, gain maps

let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(segments)
    .encode_oneshot(&pixels, w, h, PixelLayout::Rgb8Srgb)?;
```

### Strip All Metadata

```rust
// Encode with no extra segments
let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .encode_oneshot(&pixels, w, h, PixelLayout::Rgb8Srgb)?;
```

### Add Multiple Secondary Images

```rust
let segments = EncoderSegments::new()
    .add_gainmap(gainmap_jpeg)
    .add_depth_map(depth_jpeg)
    .set_xmp(&metadata_xmp);

let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(segments)
    .encode_oneshot(&pixels, w, h, PixelLayout::Rgb8Srgb)?;
```

## Implementation Notes

### MPF Offset Calculation

MPF directory contains absolute file offsets. Encoder must:
1. Write primary image with MPF placeholder (zeros for offsets)
2. Record primary image size
3. Calculate secondary image offsets: `primary_size + sum(previous_secondary_sizes)`
4. Patch MPF segment with correct offsets
5. Append secondary images

### ICC Chunking

ICC profiles > 65533 bytes must be chunked:
```
Each chunk: "ICC_PROFILE\0" + seq_no (1 byte) + num_chunks (1 byte) + data
Max chunk data: 65533 - 14 = 65519 bytes
```

The `set_icc()` method handles chunking automatically.

### Extended XMP

XMP > 65502 bytes must use extended XMP:
```
Primary: "http://ns.adobe.com/xap/1.0/\0" + XMP (up to 65502 bytes)
Extended: "http://ns.adobe.com/xmp/extension/\0" + GUID + total_len + offset + data
```

The `set_xmp()` method handles splitting automatically.

### Segment Size Limits

Each APP segment: max 65535 bytes (including 2-byte length field)
Effective data limit: 65533 bytes

Segments requiring more space must be chunked (ICC, XMP) or use MPF (images).

## Compatibility

### Transfer from Third-Party Decoders

If using a non-jpegli decoder, create `EncoderSegments` manually:

```rust
let mut segments = EncoderSegments::new();

// From some other decoder's output
if let Some(exif) = other_decoder.exif() {
    segments.set_exif(exif.to_vec());
}
if let Some(icc) = other_decoder.icc_profile() {
    segments.set_icc(icc.to_vec());
}
// etc.

let output = EncoderConfig::ycbcr(90.0, sub)
    .with_segments(segments)
    .encode_oneshot(&pixels, w, h, layout)?;
```

### Integration with ultrahdr Crate

The ultrahdr crate provides:
- `parse_xmp()` / `generate_xmp()` - gain map metadata
- `apply_gainmap()` / `compute_gainmap()` - pixel math
- `tonemap()` - HDR → SDR

jpegli-rs provides:
- Decode/encode with segment preservation
- MPF assembly/extraction
- ICC/XMP chunking

Clean separation: ultrahdr knows gain map semantics, jpegli knows JPEG structure.
