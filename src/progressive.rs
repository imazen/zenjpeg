//! Progressive JPEG scan generation and encoding support.
//!
//! Progressive JPEG encodes the image in multiple scans:
//! 1. DC coefficients (gives coarse image)
//! 2. Low-frequency AC coefficients
//! 3. High-frequency AC coefficients
//! 4. Refinement scans (successive approximation)
//!
//! This allows partial display as the image loads and can improve compression.

/// Maximum number of components in a scan.
const MAX_COMPS_IN_SCAN: usize = 4;

/// Describes a single scan in a progressive JPEG.
///
/// A scan encodes a subset of DCT coefficients for one or more components.
/// Progressive JPEG uses multiple scans to gradually build the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanInfo {
    /// Number of components in this scan (1-4)
    pub comps_in_scan: u8,
    /// Component indices (up to 4)
    pub component_index: [u8; MAX_COMPS_IN_SCAN],
    /// Spectral selection start (0 for DC, 1-63 for AC)
    pub ss: u8,
    /// Spectral selection end (0 for DC, 1-63 for AC)
    pub se: u8,
    /// Successive approximation high bit (previous Al, 0 for first pass)
    pub ah: u8,
    /// Successive approximation low bit (current precision)
    pub al: u8,
}

impl ScanInfo {
    /// Create a DC scan for the given number of components.
    ///
    /// DC scans can be interleaved (multiple components in one scan).
    pub fn dc_scan(num_components: u8) -> Self {
        Self {
            comps_in_scan: num_components.min(MAX_COMPS_IN_SCAN as u8),
            component_index: [0, 1, 2, 3],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        }
    }

    /// Create an AC scan for a single component.
    ///
    /// AC scans must be single-component in progressive mode.
    ///
    /// # Arguments
    /// * `component` - Component index (0=Y, 1=Cb, 2=Cr)
    /// * `ss` - Spectral selection start (1-63)
    /// * `se` - Spectral selection end (1-63, must be >= ss)
    /// * `ah` - Successive approximation high bit (previous Al)
    /// * `al` - Successive approximation low bit (current precision)
    pub fn ac_scan(component: u8, ss: u8, se: u8, ah: u8, al: u8) -> Self {
        Self {
            comps_in_scan: 1,
            component_index: [component, 0, 0, 0],
            ss,
            se,
            ah,
            al,
        }
    }

    /// Check if this is a DC-only scan.
    pub fn is_dc_scan(&self) -> bool {
        self.ss == 0 && self.se == 0
    }

    /// Check if this is a refinement scan (Ah > 0).
    pub fn is_refinement(&self) -> bool {
        self.ah > 0
    }
}

/// Generate a baseline (non-progressive) scan script.
///
/// This generates a single scan containing all components and all coefficients.
pub fn generate_baseline_scan(num_components: u8) -> Vec<ScanInfo> {
    vec![ScanInfo {
        comps_in_scan: num_components.min(MAX_COMPS_IN_SCAN as u8),
        component_index: [0, 1, 2, 3],
        ss: 0,
        se: 63,
        ah: 0,
        al: 0,
    }]
}

/// Generate a minimal progressive scan script.
///
/// This is the simplest progressive encoding:
/// 1. DC scan for all components
/// 2. Full AC scan (1-63) for each component
pub fn generate_minimal_progressive_scans(num_components: u8) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    // DC scan for all components (interleaved)
    scans.push(ScanInfo::dc_scan(num_components));

    // Full AC scan for each component
    for comp in 0..num_components {
        scans.push(ScanInfo::ac_scan(comp, 1, 63, 0, 0));
    }

    scans
}

/// Generate a simple progressive scan script.
///
/// 1. DC scan for all components
/// 2. AC scans for each component (bands 1-5, 6-63)
pub fn generate_simple_progressive_scans(num_components: u8) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    // DC scan for all components
    scans.push(ScanInfo::dc_scan(num_components));

    // AC scans for each component
    for comp in 0..num_components {
        // Low frequency AC (1-5)
        scans.push(ScanInfo::ac_scan(comp, 1, 5, 0, 0));
        // High frequency AC (6-63)
        scans.push(ScanInfo::ac_scan(comp, 6, 63, 0, 0));
    }

    scans
}

/// Generate a standard progressive scan script with successive approximation.
///
/// Uses successive approximation for better compression:
/// 1. DC scan with point transform (Al=1)
/// 2. AC scans with reduced precision (split into 1-5 and 6-63 like libjpeg)
/// 3. Refinement scans
pub fn generate_standard_progressive_scans(num_components: u8) -> Vec<ScanInfo> {
    let mut scans = Vec::new();

    // Initial DC scan with point transform (Al=1)
    let mut dc_scan = ScanInfo::dc_scan(num_components);
    dc_scan.al = 1;
    scans.push(dc_scan);

    // AC first scans for each component - split like libjpeg
    for comp in 0..num_components {
        // Low frequency (1-5) at reduced precision
        scans.push(ScanInfo::ac_scan(comp, 1, 5, 0, 2));
        // High frequency (6-63) at reduced precision
        scans.push(ScanInfo::ac_scan(comp, 6, 63, 0, 2));
    }

    // AC refinement scans - must cover same spectral range as first scans combined
    for comp in 0..num_components {
        // Refine bits 2->1 for full band
        scans.push(ScanInfo::ac_scan(comp, 1, 63, 2, 1));
    }

    // DC refinement scan
    let mut dc_refine = ScanInfo::dc_scan(num_components);
    dc_refine.ah = 1;
    dc_refine.al = 0;
    scans.push(dc_refine);

    // Final AC refinement (1->0)
    for comp in 0..num_components {
        scans.push(ScanInfo::ac_scan(comp, 1, 63, 1, 0));
    }

    scans
}

/// Check if a scan script uses progressive mode.
pub fn is_progressive_script(scans: &[ScanInfo]) -> bool {
    if scans.len() > 1 {
        return true;
    }
    if let Some(scan) = scans.first() {
        // Single scan covering all coefficients in one pass is baseline
        scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0
    } else {
        false
    }
}

/// Validate a scan script for correctness.
pub fn validate_scan_script(scans: &[ScanInfo], num_components: u8) -> Result<(), &'static str> {
    if scans.is_empty() {
        return Err("Scan script must have at least one scan");
    }

    for scan in scans.iter() {
        // Validate component count
        if scan.comps_in_scan == 0 || scan.comps_in_scan > MAX_COMPS_IN_SCAN as u8 {
            return Err("Invalid component count in scan");
        }

        // Validate component indices
        for j in 0..scan.comps_in_scan as usize {
            if scan.component_index[j] >= num_components {
                return Err("Component index out of range");
            }
        }

        // Validate spectral selection
        if scan.se > 63 {
            return Err("Spectral selection end (Se) must be <= 63");
        }
        if scan.ss > scan.se {
            return Err("Spectral selection start (Ss) must be <= end (Se)");
        }

        // Progressive-specific validation
        let is_progressive_scan =
            scan.ah != 0 || scan.al != 0 || (scan.ss == 0 && scan.se != 0 && scan.se != 63);

        if is_progressive_scan {
            // In progressive, interleaved DC scans must be DC-only
            if scan.ss == 0 && scan.se != 0 && scan.comps_in_scan > 1 {
                return Err("Progressive interleaved scans cannot mix DC and AC");
            }
            // Progressive AC scans must be single-component
            if scan.ss > 0 && scan.comps_in_scan > 1 {
                return Err("Progressive AC scans must be single-component");
            }
        }

        // Successive approximation validation
        if scan.ah > 13 || scan.al > 13 {
            return Err("Successive approximation bit position out of range");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_scan() {
        let scan = ScanInfo::dc_scan(3);
        assert!(scan.is_dc_scan());
        assert!(!scan.is_refinement());
        assert_eq!(scan.comps_in_scan, 3);
    }

    #[test]
    fn test_ac_scan() {
        let scan = ScanInfo::ac_scan(0, 1, 63, 0, 0);
        assert!(!scan.is_dc_scan());
        assert!(!scan.is_refinement());
        assert_eq!(scan.ss, 1);
        assert_eq!(scan.se, 63);
    }

    #[test]
    fn test_refinement_scan() {
        let scan = ScanInfo::ac_scan(0, 1, 63, 2, 1);
        assert!(scan.is_refinement());
        assert_eq!(scan.ah, 2);
        assert_eq!(scan.al, 1);
    }

    #[test]
    fn test_baseline_scan() {
        let scans = generate_baseline_scan(3);
        assert_eq!(scans.len(), 1);
        assert!(!is_progressive_script(&scans));
    }

    #[test]
    fn test_minimal_progressive() {
        let scans = generate_minimal_progressive_scans(3);
        assert_eq!(scans.len(), 4); // DC + 3 AC
        assert!(is_progressive_script(&scans));
        assert!(validate_scan_script(&scans, 3).is_ok());
    }

    #[test]
    fn test_simple_progressive() {
        let scans = generate_simple_progressive_scans(3);
        assert!(is_progressive_script(&scans));
        assert!(validate_scan_script(&scans, 3).is_ok());
    }

    #[test]
    fn test_standard_progressive() {
        let scans = generate_standard_progressive_scans(3);
        assert!(is_progressive_script(&scans));
        assert!(validate_scan_script(&scans, 3).is_ok());
    }

    #[test]
    fn test_validate_invalid() {
        // Invalid component index
        let scans = vec![ScanInfo::ac_scan(5, 1, 63, 0, 0)];
        assert!(validate_scan_script(&scans, 3).is_err());
    }
}
