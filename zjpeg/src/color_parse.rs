use zenlayout::CanvasColor;

/// Parse a CSS color string (hex or named) into a `CanvasColor`.
///
/// Delegates to `zenlayout::riapi::parse_color`.
pub fn parse_color(s: &str) -> Option<CanvasColor> {
    zenlayout::riapi::parse_color(s)
}
