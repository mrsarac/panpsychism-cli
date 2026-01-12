//! Privacy tier configuration for Project Panpsychism.
//!
//! Implements the three-tier privacy system:
//! - LOCAL: All data stays on device (€19/mo)
//! - HYBRID: User controls what to share (€29/mo)
//! - FEDERATED: Full collaboration with rewards (€49/mo)
//!
//! Default is LOCAL (GDPR Article 25 - Privacy by Design)

// External crates
use serde::{Deserialize, Serialize};

// =============================================================================
// PRIVACY TIER ENUM
// =============================================================================

/// Privacy tier levels for data handling.
///
/// Based on Board Decision #041 - Privacy-Learning Paradox Resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrivacyTier {
    /// All data stays on device. No network sync.
    /// Maximum privacy, no collective learning.
    /// Price: €19/month
    #[default]
    Local,

    /// User controls what data to share.
    /// Granular consent for each sync request.
    /// Price: €29/month
    Hybrid,

    /// Full collaboration with anonymized patterns.
    /// Differential privacy (ε=0.1) applied.
    /// Conatus Network rewards enabled.
    /// Price: €49/month
    Federated,
}

// =============================================================================
// PRIVACY CONFIG STRUCT
// =============================================================================

/// Privacy configuration with granular controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// The selected privacy tier
    pub tier: PrivacyTier,

    /// Share anonymized query patterns (HYBRID/FEDERATED only)
    #[serde(default)]
    pub share_patterns: bool,

    /// Share prompt effectiveness ratings (HYBRID/FEDERATED only)
    #[serde(default)]
    pub share_ratings: bool,

    /// Share custom prompt contributions (HYBRID/FEDERATED only)
    #[serde(default)]
    pub share_prompts: bool,

    /// Share error reports with context (HYBRID/FEDERATED only)
    #[serde(default)]
    pub share_errors: bool,

    /// Differential privacy epsilon (0-100, lower = stronger privacy)
    /// Default: 10 (ε = 0.1)
    #[serde(default = "default_anonymization_level")]
    pub anonymization_level: u8,
}

fn default_anonymization_level() -> u8 {
    10 // ε = 0.1 (strong privacy)
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            tier: PrivacyTier::Local, // GDPR: Privacy by Default
            share_patterns: false,
            share_ratings: false,
            share_prompts: false,
            share_errors: false,
            anonymization_level: default_anonymization_level(),
        }
    }
}

// =============================================================================
// PRIVACY CONFIG METHODS
// =============================================================================

impl PrivacyConfig {
    /// Create a new LOCAL (maximum privacy) configuration.
    pub fn local() -> Self {
        Self::default()
    }

    /// Create a new HYBRID configuration with all sharing disabled by default.
    pub fn hybrid() -> Self {
        Self {
            tier: PrivacyTier::Hybrid,
            ..Default::default()
        }
    }

    /// Create a new FEDERATED configuration with all sharing enabled.
    pub fn federated() -> Self {
        Self {
            tier: PrivacyTier::Federated,
            share_patterns: true,
            share_ratings: true,
            share_prompts: true,
            share_errors: true,
            anonymization_level: default_anonymization_level(),
        }
    }

    /// Check if any data sharing is enabled.
    pub fn is_sharing_enabled(&self) -> bool {
        self.share_patterns || self.share_ratings || self.share_prompts || self.share_errors
    }

    /// Check if this configuration allows network access.
    pub fn allows_network(&self) -> bool {
        self.tier != PrivacyTier::Local
    }

    /// Get the differential privacy epsilon value.
    pub fn epsilon(&self) -> f64 {
        self.anonymization_level as f64 / 100.0
    }
}

impl std::fmt::Display for PrivacyTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrivacyTier::Local => write!(f, "LOCAL (€19/mo)"),
            PrivacyTier::Hybrid => write!(f, "HYBRID (€29/mo)"),
            PrivacyTier::Federated => write!(f, "FEDERATED (€49/mo)"),
        }
    }
}

impl std::str::FromStr for PrivacyTier {
    type Err = crate::Error;

    /// Parse a privacy tier from a string identifier.
    ///
    /// Accepts various forms: "local", "LOCAL", "hybrid", "federated", etc.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::str::FromStr;
    /// use panpsychism::PrivacyTier;
    ///
    /// let tier: PrivacyTier = "local".parse().unwrap();
    /// assert_eq!(tier, PrivacyTier::Local);
    ///
    /// let tier: PrivacyTier = "FEDERATED".parse().unwrap();
    /// assert_eq!(tier, PrivacyTier::Federated);
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "local" | "private" | "offline" => Ok(PrivacyTier::Local),
            "hybrid" | "mixed" | "selective" => Ok(PrivacyTier::Hybrid),
            "federated" | "shared" | "collaborative" | "network" => Ok(PrivacyTier::Federated),
            _ => Err(crate::Error::Config(format!(
                "Unknown privacy tier: '{}'. Valid tiers: local, hybrid, federated",
                s
            ))),
        }
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_local() {
        let config = PrivacyConfig::default();
        assert_eq!(config.tier, PrivacyTier::Local);
        assert!(!config.is_sharing_enabled());
        assert!(!config.allows_network());
    }

    #[test]
    fn test_federated_enables_sharing() {
        let config = PrivacyConfig::federated();
        assert_eq!(config.tier, PrivacyTier::Federated);
        assert!(config.is_sharing_enabled());
        assert!(config.allows_network());
    }

    #[test]
    fn test_epsilon_calculation() {
        let config = PrivacyConfig::default();
        assert!((config.epsilon() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_privacy_tier_from_str() {
        assert_eq!("local".parse::<PrivacyTier>().unwrap(), PrivacyTier::Local);
        assert_eq!(
            "hybrid".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Hybrid
        );
        assert_eq!(
            "federated".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Federated
        );
    }

    #[test]
    fn test_privacy_tier_from_str_aliases() {
        // Local aliases
        assert_eq!(
            "private".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Local
        );
        assert_eq!(
            "offline".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Local
        );

        // Hybrid aliases
        assert_eq!("mixed".parse::<PrivacyTier>().unwrap(), PrivacyTier::Hybrid);
        assert_eq!(
            "selective".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Hybrid
        );

        // Federated aliases
        assert_eq!(
            "shared".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Federated
        );
        assert_eq!(
            "network".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Federated
        );
    }

    #[test]
    fn test_privacy_tier_from_str_case_insensitive() {
        assert_eq!("LOCAL".parse::<PrivacyTier>().unwrap(), PrivacyTier::Local);
        assert_eq!(
            "FEDERATED".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Federated
        );
        assert_eq!(
            "Hybrid".parse::<PrivacyTier>().unwrap(),
            PrivacyTier::Hybrid
        );
    }

    #[test]
    fn test_privacy_tier_from_str_invalid() {
        let result = "invalid".parse::<PrivacyTier>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unknown privacy tier"));
    }
}
