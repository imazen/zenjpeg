//! Container metadata types shared across JPEG multi-image formats.
//!
//! These types model the two container mechanisms found in multi-image
//! JPEG files:
//!
//! - **GContainer** (Google/Adobe XMP): `Container:Directory` with
//!   `Item:Semantic` values like `"Primary"`, `"GainMap"`, `"DepthMap"`,
//!   `"ConfidenceMap"`. See [`ItemSemantic`] and [`ContainerItem`].
//!
//! - **MPF** (CIPA DC-007): Multi-Picture Format directory with typed
//!   image entries identified by [`MpImageType`] codes (primary,
//!   disparity, panorama, thumbnails, etc.). See [`MpfEntry`].
//!
//! The parsers that consume these types live in [`super::xmp`] and
//! [`super::mpf`].

use alloc::string::String;
use alloc::vec::Vec;

// ===========================================================================
// GContainer (XMP Container:Directory)
// ===========================================================================

/// Semantic role of an item in a GContainer directory.
///
/// These correspond to `Item:Semantic` values in the XMP
/// `Container:Directory`. Used by Ultra HDR, Android Dynamic Depth
/// Format, and Google Camera output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ItemSemantic {
    /// Primary image (`"Primary"`).
    Primary,
    /// HDR gain map (`"GainMap"`).
    GainMap,
    /// Depth map (`"DepthMap"`).
    DepthMap,
    /// Confidence / quality map for depth (`"ConfidenceMap"`).
    ConfidenceMap,
    /// Unrecognized or vendor-specific semantic.
    Other(String),
}

impl ItemSemantic {
    /// Parse from XMP `Item:Semantic` attribute value.
    #[must_use]
    pub fn from_xmp(s: &str) -> Self {
        match s {
            "Primary" => Self::Primary,
            "GainMap" => Self::GainMap,
            "DepthMap" => Self::DepthMap,
            "ConfidenceMap" => Self::ConfidenceMap,
            other => Self::Other(String::from(other)),
        }
    }

    /// XMP-facing string representation.
    #[must_use]
    pub fn as_xmp_str(&self) -> &str {
        match self {
            Self::Primary => "Primary",
            Self::GainMap => "GainMap",
            Self::DepthMap => "DepthMap",
            Self::ConfidenceMap => "ConfidenceMap",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// An item in a GContainer directory.
///
/// Parsed from `Container:Directory` → `rdf:Seq` → `rdf:li` entries in
/// XMP. Each item describes a secondary image appended after the primary
/// JPEG's EOI.
///
/// The [`padding`](Self::padding) field is from Google's Dynamic Depth
/// Format extension and not present in canonical Ultra HDR.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContainerItem {
    /// Semantic role.
    pub semantic: ItemSemantic,
    /// MIME type (e.g. `"image/jpeg"`, `"image/png"`).
    pub mime: String,
    /// Byte length of this item's payload. `None` for the primary image
    /// (its size is determined by its own EOI).
    pub length: Option<usize>,
    /// Padding bytes before this item. Used by some Dynamic Depth
    /// Format implementations.
    pub padding: Option<usize>,
}

impl ContainerItem {
    /// Create a primary image item.
    #[must_use]
    pub fn primary(mime: impl Into<String>) -> Self {
        Self {
            semantic: ItemSemantic::Primary,
            mime: mime.into(),
            length: None,
            padding: None,
        }
    }

    /// Create a secondary item with a known length.
    #[must_use]
    pub fn secondary(semantic: ItemSemantic, mime: impl Into<String>, length: usize) -> Self {
        Self {
            semantic,
            mime: mime.into(),
            length: Some(length),
            padding: None,
        }
    }
}

// ===========================================================================
// MPF (CIPA DC-007) types
// ===========================================================================

/// Multi-Picture Format image type (CIPA DC-007 Individual Image
/// Attribute).
///
/// The type code occupies bits 16–23 of the MP Entry attribute field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MpImageType {
    /// Undefined type (`0x000000`). Ultra HDR uses this for gain maps.
    #[default]
    Undefined,
    /// Baseline MP primary image (`0x030000`).
    BaselinePrimary,
    /// Large thumbnail — VGA resolution (`0x010001`).
    LargeThumbnailVga,
    /// Large thumbnail — Full HD resolution (`0x010002`).
    LargeThumbnailFullHd,
    /// Multi-frame panorama (`0x020001`).
    Panorama,
    /// Multi-frame disparity / depth map (`0x020002`).
    Disparity,
    /// Multi-frame multi-angle (`0x020003`).
    MultiAngle,
    /// Unrecognized type code.
    Other(u32),
}

impl MpImageType {
    /// Decode an MPF Individual Image Attribute type code.
    #[must_use]
    pub fn from_type_code(code: u32) -> Self {
        match code {
            0x000000 => Self::Undefined,
            0x030000 => Self::BaselinePrimary,
            0x010001 => Self::LargeThumbnailVga,
            0x010002 => Self::LargeThumbnailFullHd,
            0x020001 => Self::Panorama,
            0x020002 => Self::Disparity,
            0x020003 => Self::MultiAngle,
            other => Self::Other(other),
        }
    }

    /// Encode to MPF Individual Image Attribute type code.
    #[must_use]
    pub fn type_code(self) -> u32 {
        match self {
            Self::Undefined => 0x000000,
            Self::BaselinePrimary => 0x030000,
            Self::LargeThumbnailVga => 0x010001,
            Self::LargeThumbnailFullHd => 0x010002,
            Self::Panorama => 0x020001,
            Self::Disparity => 0x020002,
            Self::MultiAngle => 0x020003,
            Self::Other(code) => code,
        }
    }
}

/// A parsed MPF directory entry.
///
/// Represents one image in the MPF directory with its type, byte offset,
/// and size within the file. See [`super::mpf::parse_mpf_segment`] for
/// the parser that yields these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MpfEntry {
    /// Image type from the attribute field.
    pub image_type: MpImageType,
    /// Absolute byte offset of this image in the file. The primary
    /// image is conventionally `0`.
    pub offset: usize,
    /// Size of this image in bytes.
    pub size: usize,
}

/// GContainer `Container:Directory` items extracted from XMP.
///
/// Returns items in order of appearance; the first item is conventionally
/// [`ItemSemantic::Primary`].
///
/// Parsed from `rdf:Seq` inside `Container:Directory`, extracting
/// `Item:Semantic`, `Item:Mime`, `Item:Length`, and `Item:Padding`.
#[must_use]
pub fn parse_container_items(xmp: &str) -> Vec<ContainerItem> {
    let mut items = Vec::new();
    let mut search_from = 0;
    while let Some(li_start) = xmp[search_from..].find("rdf:li") {
        let abs_start = search_from + li_start;
        let block_end = xmp[abs_start..]
            .find("</rdf:li>")
            .map(|p| abs_start + p)
            .or_else(|| {
                xmp[abs_start + 6..]
                    .find("rdf:li")
                    .map(|p| abs_start + 6 + p)
            })
            .unwrap_or(xmp.len());
        let block = &xmp[abs_start..block_end];
        let semantic =
            extract_attr_from_block(block, "Item:Semantic").map(|s| ItemSemantic::from_xmp(&s));
        let mime = extract_attr_from_block(block, "Item:Mime");
        if let (Some(semantic), Some(mime)) = (semantic, mime) {
            let length =
                extract_attr_from_block(block, "Item:Length").and_then(|s| s.parse::<usize>().ok());
            let padding = extract_attr_from_block(block, "Item:Padding")
                .and_then(|s| s.parse::<usize>().ok());
            items.push(ContainerItem {
                semantic,
                mime,
                length,
                padding,
            });
        }
        search_from = block_end;
    }
    items
}

/// Generate a `Container:Directory` XML fragment for a list of items.
///
/// The result is an `rdf:Seq` block suitable for embedding inside an
/// `rdf:Description` that declares the Container and Item namespaces.
#[must_use]
pub fn generate_container_directory(items: &[ContainerItem]) -> String {
    let mut xml = String::from("      <Container:Directory>\n        <rdf:Seq>\n");
    for item in items {
        xml.push_str("          <rdf:li rdf:parseType=\"Resource\">\n");
        xml.push_str("            <Container:Item\n");
        xml.push_str(&alloc::format!(
            "                Item:Semantic=\"{}\"\n",
            item.semantic.as_xmp_str()
        ));
        xml.push_str(&alloc::format!(
            "                Item:Mime=\"{}\"",
            item.mime
        ));
        if let Some(length) = item.length {
            xml.push_str(&alloc::format!(
                "\n                Item:Length=\"{length}\""
            ));
        }
        if let Some(padding) = item.padding {
            xml.push_str(&alloc::format!(
                "\n                Item:Padding=\"{padding}\""
            ));
        }
        xml.push_str("/>\n");
        xml.push_str("          </rdf:li>\n");
    }
    xml.push_str("        </rdf:Seq>\n      </Container:Directory>");
    xml
}

fn extract_attr_from_block(block: &str, attr_name: &str) -> Option<String> {
    let pattern = alloc::format!("{attr_name}=\"");
    if let Some(start) = block.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = block[value_start..].find('"') {
            return Some(String::from(&block[value_start..value_start + end]));
        }
    }
    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn item_semantic_roundtrip_known() {
        for s in ["Primary", "GainMap", "DepthMap", "ConfidenceMap"] {
            let parsed = ItemSemantic::from_xmp(s);
            assert_eq!(parsed.as_xmp_str(), s);
        }
    }

    #[test]
    fn item_semantic_roundtrip_custom() {
        let other = ItemSemantic::from_xmp("CustomVendorSemantic");
        assert_eq!(other.as_xmp_str(), "CustomVendorSemantic");
    }

    #[test]
    fn item_semantic_empty_string_is_other() {
        let e = ItemSemantic::from_xmp("");
        assert_eq!(e.as_xmp_str(), "");
    }

    #[test]
    fn container_item_primary_constructor_has_no_length() {
        let p = ContainerItem::primary("image/jpeg");
        assert_eq!(p.semantic, ItemSemantic::Primary);
        assert_eq!(p.mime, "image/jpeg");
        assert!(p.length.is_none());
        assert!(p.padding.is_none());
    }

    #[test]
    fn container_item_secondary_constructor_sets_length() {
        let g = ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", 5000);
        assert_eq!(g.length, Some(5000));
        assert_eq!(g.semantic, ItemSemantic::GainMap);
    }

    #[test]
    fn mp_image_type_roundtrip_covers_known_codes() {
        let codes = [
            (0x000000u32, MpImageType::Undefined),
            (0x030000, MpImageType::BaselinePrimary),
            (0x010001, MpImageType::LargeThumbnailVga),
            (0x010002, MpImageType::LargeThumbnailFullHd),
            (0x020001, MpImageType::Panorama),
            (0x020002, MpImageType::Disparity),
            (0x020003, MpImageType::MultiAngle),
        ];
        for (code, expected) in codes {
            let parsed = MpImageType::from_type_code(code);
            assert_eq!(parsed, expected);
            assert_eq!(parsed.type_code(), code);
        }
    }

    #[test]
    fn mp_image_type_other_roundtrips_any_code() {
        let arbitrary = MpImageType::from_type_code(0xDEAD00);
        assert_eq!(arbitrary, MpImageType::Other(0xDEAD00));
        assert_eq!(arbitrary.type_code(), 0xDEAD00);
    }

    #[test]
    fn mp_image_type_default_is_undefined() {
        assert_eq!(MpImageType::default(), MpImageType::Undefined);
    }

    #[test]
    fn parse_container_items_ultrahdr_shape() {
        let xmp = r#"
            <Container:Directory>
                <rdf:Seq>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item
                            Item:Semantic="Primary"
                            Item:Mime="image/jpeg"/>
                    </rdf:li>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item
                            Item:Semantic="GainMap"
                            Item:Mime="image/jpeg"
                            Item:Length="5000"/>
                    </rdf:li>
                </rdf:Seq>
            </Container:Directory>"#;
        let items = parse_container_items(xmp);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].semantic, ItemSemantic::Primary);
        assert_eq!(items[0].mime, "image/jpeg");
        assert!(items[0].length.is_none());
        assert_eq!(items[1].semantic, ItemSemantic::GainMap);
        assert_eq!(items[1].length, Some(5000));
    }

    #[test]
    fn parse_container_items_android_ddf_shape() {
        let xmp = r#"
            <Container:Directory>
                <rdf:Seq>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/>
                    </rdf:li>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item Item:Semantic="DepthMap" Item:Mime="image/jpeg" Item:Length="8000"/>
                    </rdf:li>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item Item:Semantic="ConfidenceMap" Item:Mime="image/jpeg" Item:Length="3000"/>
                    </rdf:li>
                </rdf:Seq>
            </Container:Directory>"#;
        let items = parse_container_items(xmp);
        assert_eq!(items.len(), 3);
        assert_eq!(items[1].semantic, ItemSemantic::DepthMap);
        assert_eq!(items[2].semantic, ItemSemantic::ConfidenceMap);
    }

    #[test]
    fn parse_container_items_with_padding() {
        let xmp = r#"
            <Container:Directory>
                <rdf:Seq>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/>
                    </rdf:li>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="5000" Item:Padding="16"/>
                    </rdf:li>
                </rdf:Seq>
            </Container:Directory>"#;
        let items = parse_container_items(xmp);
        assert_eq!(items[1].padding, Some(16));
    }

    #[test]
    fn parse_container_items_empty_string() {
        assert!(parse_container_items("").is_empty());
    }

    #[test]
    fn parse_container_items_missing_attrs_skips_item() {
        // rdf:li with no Item:Semantic should be skipped rather than
        // erroring.
        let xmp = r#"
            <Container:Directory>
                <rdf:Seq>
                    <rdf:li rdf:parseType="Resource">
                        <Container:Item Item:Mime="image/jpeg"/>
                    </rdf:li>
                </rdf:Seq>
            </Container:Directory>"#;
        assert!(parse_container_items(xmp).is_empty());
    }

    #[test]
    fn generate_container_directory_roundtrip_two_items() {
        let items = vec![
            ContainerItem::primary("image/jpeg"),
            ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", 5000),
        ];
        let xml = generate_container_directory(&items);
        let parsed = parse_container_items(&xml);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].semantic, ItemSemantic::Primary);
        assert_eq!(parsed[1].semantic, ItemSemantic::GainMap);
        assert_eq!(parsed[1].length, Some(5000));
    }

    #[test]
    fn generate_container_directory_roundtrip_preserves_padding() {
        let items = vec![
            ContainerItem::primary("image/jpeg"),
            ContainerItem {
                semantic: ItemSemantic::GainMap,
                mime: String::from("image/jpeg"),
                length: Some(1000),
                padding: Some(32),
            },
        ];
        let xml = generate_container_directory(&items);
        let parsed = parse_container_items(&xml);
        assert_eq!(parsed[1].padding, Some(32));
    }

    #[test]
    fn generate_container_directory_includes_all_fields() {
        let items = vec![
            ContainerItem::primary("image/jpeg"),
            ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", 5000),
            ContainerItem::secondary(ItemSemantic::DepthMap, "image/png", 12000),
        ];
        let xml = generate_container_directory(&items);
        assert!(xml.contains("GainMap"));
        assert!(xml.contains("DepthMap"));
        assert!(xml.contains("image/png"));
        assert!(xml.contains("12000"));
    }

    #[test]
    fn mpf_entry_fields() {
        let e = MpfEntry {
            image_type: MpImageType::Disparity,
            offset: 50_000,
            size: 10_000,
        };
        assert_eq!(e.image_type, MpImageType::Disparity);
        assert_eq!(e.offset, 50_000);
        assert_eq!(e.size, 10_000);
    }

    // -----------------------------------------------------------------------
    // Property tests: parse(generate(x)) == x (modulo canonicalization)
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    fn arb_item_semantic() -> impl Strategy<Value = ItemSemantic> {
        prop_oneof![
            Just(ItemSemantic::Primary),
            Just(ItemSemantic::GainMap),
            Just(ItemSemantic::DepthMap),
            Just(ItemSemantic::ConfidenceMap),
            // `Other` must avoid XML-special chars to survive a naive XML
            // roundtrip. The parser doesn't decode entities, so restrict
            // to a safe ASCII subset.
            "[A-Za-z][A-Za-z0-9_-]{0,15}".prop_map(ItemSemantic::Other),
        ]
    }

    fn arb_container_item() -> impl Strategy<Value = ContainerItem> {
        (
            arb_item_semantic(),
            prop_oneof![
                Just(String::from("image/jpeg")),
                Just(String::from("image/png")),
                Just(String::from("image/heic")),
                Just(String::from("application/octet-stream")),
            ],
            proptest::option::of(1usize..10_000_000),
            proptest::option::of(0usize..1024),
        )
            .prop_map(|(semantic, mime, length, padding)| ContainerItem {
                semantic,
                mime,
                length,
                padding,
            })
    }

    proptest! {
        /// Full field-preserving roundtrip:
        /// `generate_container_directory` → `parse_container_items`
        /// preserves `semantic`, `mime`, `length`, and `padding` exactly.
        /// The emitter is a pure pass-through (no Primary-special-casing,
        /// no forced defaults); `None` length/padding is absent in the XML
        /// and parses back as `None`.
        #[test]
        fn container_directory_roundtrip(
            items in proptest::collection::vec(arb_container_item(), 1..5),
        ) {
            let xml = generate_container_directory(&items);
            let parsed = parse_container_items(&xml);
            prop_assert_eq!(parsed.len(), items.len());
            for (i, (orig, got)) in items.iter().zip(parsed.iter()).enumerate() {
                prop_assert_eq!(
                    &orig.semantic, &got.semantic,
                    "item {}: semantic", i,
                );
                prop_assert_eq!(&orig.mime, &got.mime, "item {}: mime", i);
                prop_assert_eq!(orig.length, got.length, "item {}: length", i);
                prop_assert_eq!(orig.padding, got.padding, "item {}: padding", i);
            }
        }
    }
}
