//! Centralized constants for Project Panpsychism.
//!
//! This module provides a single source of truth for all magic numbers,
//! default values, and configuration constants used throughout the crate.

// ============================================================================
// API & Network Constants
// ============================================================================

/// Default Gemini API endpoint (Antigravity proxy for local development).
pub const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8045";

/// Default API key for Antigravity proxy.
pub const DEFAULT_API_KEY: &str = "sk-antigravity";

/// Default model for Gemini API requests.
pub const DEFAULT_MODEL: &str = "gemini-3-flash";

/// Default timeout for API requests in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Default maximum retry attempts for failed API requests.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default rate limit: requests per minute.
pub const DEFAULT_REQUESTS_PER_MINUTE: u32 = 60;

// ============================================================================
// Prompt Selection Constants
// ============================================================================

/// Minimum number of prompts to consider in search results.
pub const MIN_PROMPTS: usize = 2;

/// Maximum number of prompts to return in search results.
pub const MAX_PROMPTS: usize = 7;

/// Default relevance threshold for prompt matching (0.0 - 1.0).
pub const DEFAULT_RELEVANCE_THRESHOLD: f64 = 0.3;

// ============================================================================
// Validation Thresholds (Spinoza Philosophy)
// ============================================================================

/// Default threshold for Conatus (self-preservation/drive) validation.
pub const DEFAULT_CONATUS_THRESHOLD: f64 = 0.6;

/// Default threshold for Ratio (logical coherence) validation.
pub const DEFAULT_RATIO_THRESHOLD: f64 = 0.5;

/// Default threshold for Laetitia (joy/satisfaction) validation.
pub const DEFAULT_LAETITIA_THRESHOLD: f64 = 0.4;

/// Default minimum keywords for philosophical validation.
pub const DEFAULT_MIN_KEYWORDS: usize = 2;

// ============================================================================
// Corrector Constants (Second Throw Pattern)
// ============================================================================

/// Default maximum correction iterations.
pub const DEFAULT_MAX_ITERATIONS: usize = 3;

/// Default ambiguity threshold for triggering correction.
pub const DEFAULT_AMBIGUITY_THRESHOLD: f64 = 0.7;

// ============================================================================
// Privacy Constants
// ============================================================================

/// Default anonymization level (epsilon * 100 for differential privacy).
/// Value 10 = epsilon 0.1 (strong privacy).
pub const DEFAULT_ANONYMIZATION_LEVEL: u8 = 10;

// ============================================================================
// Search & Indexing Constants
// ============================================================================

/// Maximum depth for recursive directory scanning.
pub const DEFAULT_MAX_SCAN_DEPTH: usize = 10;

/// Default batch size for indexing operations.
pub const DEFAULT_BATCH_SIZE: usize = 100;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_limits() {
        assert!(MIN_PROMPTS <= MAX_PROMPTS);
    }

    #[test]
    fn test_thresholds_in_range() {
        assert!((0.0..=1.0).contains(&DEFAULT_RELEVANCE_THRESHOLD));
        assert!((0.0..=1.0).contains(&DEFAULT_CONATUS_THRESHOLD));
        assert!((0.0..=1.0).contains(&DEFAULT_RATIO_THRESHOLD));
        assert!((0.0..=1.0).contains(&DEFAULT_LAETITIA_THRESHOLD));
        assert!((0.0..=1.0).contains(&DEFAULT_AMBIGUITY_THRESHOLD));
    }

    #[test]
    fn test_anonymization_epsilon() {
        let epsilon = DEFAULT_ANONYMIZATION_LEVEL as f64 / 100.0;
        assert!((epsilon - 0.1).abs() < f64::EPSILON);
    }
}
