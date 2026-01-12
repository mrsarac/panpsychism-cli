//! # Spinoza Philosophical Validation Module
//!
//! This module implements Baruch Spinoza's Ethics as a computational validation
//! framework for prompt orchestration. Each principle maps to a specific aspect
//! of content quality and alignment.
//!
//! ## Philosophical Foundation
//!
//! ### CONATUS (Self-Preservation)
//! From Ethics III, Proposition 6: "Each thing, as far as it can by its own power,
//! strives to persevere in its being."
//!
//! In prompt validation, CONATUS measures whether content supports growth,
//! learning, and positive development rather than destruction or harm.
//!
//! ### RATIO (Reason)
//! From Ethics II, Proposition 44: "It is in the nature of reason to perceive
//! things under a certain form of eternity."
//!
//! RATIO validates logical consistency, coherent structure, and sound reasoning
//! in the content.
//!
//! ### LAETITIA (Joy)
//! From Ethics III, Proposition 11: "Laetitia is a person's passage from a lesser
//! to a greater perfection."
//!
//! LAETITIA measures whether content enhances positive affect, inspires growth,
//! and moves toward greater understanding.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use panpsychism::validator::{SpinozaValidator, ValidationConfig};
//!
//! let validator = SpinozaValidator::with_config(ValidationConfig {
//!     conatus_threshold: 0.7,
//!     ratio_threshold: 0.8,
//!     laetitia_threshold: 0.6,
//! });
//!
//! let result = validator.validate("Your content here").await?;
//! if result.is_valid {
//!     println!("Content passes Spinoza validation!");
//! }
//! ```

use crate::{Error, Result};
use std::collections::HashSet;

// =============================================================================
// KEYWORD DICTIONARIES
// =============================================================================

/// Keywords indicating CONATUS (self-preservation, growth, positive development)
const CONATUS_POSITIVE_KEYWORDS: &[&str] = &[
    // Growth and development
    "grow",
    "growth",
    "develop",
    "improve",
    "enhance",
    "expand",
    "evolve",
    "progress",
    "advance",
    "flourish",
    "thrive",
    "prosper",
    // Learning and understanding
    "learn",
    "understand",
    "discover",
    "explore",
    "insight",
    "knowledge",
    "wisdom",
    "comprehend",
    "grasp",
    "master",
    // Creation and building
    "create",
    "build",
    "construct",
    "design",
    "innovate",
    "generate",
    "produce",
    "establish",
    "found",
    "forge",
    // Preservation and protection
    "preserve",
    "protect",
    "maintain",
    "sustain",
    "nurture",
    "care",
    "support",
    "strengthen",
    "fortify",
    "secure",
    // Health and vitality
    "health",
    "healthy",
    "vital",
    "energy",
    "resilient",
    "robust",
    "strong",
    "stable",
    "balanced",
    "harmony",
];

/// Keywords indicating anti-CONATUS (destruction, harm, stagnation)
const CONATUS_NEGATIVE_KEYWORDS: &[&str] = &[
    // Destruction
    "destroy",
    "destruct",
    "demolish",
    "annihilate",
    "eliminate",
    "eradicate",
    "obliterate",
    "exterminate",
    "ruin",
    "wreck",
    // Harm
    "harm",
    "hurt",
    "damage",
    "injure",
    "wound",
    "attack",
    "assault",
    "abuse",
    "exploit",
    "manipulate",
    // Stagnation
    "stagnate",
    "decay",
    "deteriorate",
    "decline",
    "degrade",
    "regress",
    "wither",
    "fade",
    "diminish",
    "weaken",
    // Negativity
    "hate",
    "despise",
    "loathe",
    "detest",
    "resent",
    "malice",
    "spite",
    "vengeance",
    "revenge",
    "hostility",
];

/// Keywords indicating RATIO (logical consistency, reasoning)
const RATIO_POSITIVE_KEYWORDS: &[&str] = &[
    // Logic and reasoning
    "because",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "accordingly",
    "reason",
    "logic",
    "logical",
    "rational",
    // Structure and clarity
    "first",
    "second",
    "third",
    "finally",
    "moreover",
    "furthermore",
    "additionally",
    "specifically",
    "particularly",
    "namely",
    // Analysis and evidence
    "analyze",
    "analysis",
    "evidence",
    "proof",
    "demonstrate",
    "show",
    "indicate",
    "suggest",
    "imply",
    "conclude",
    // Coherence
    "consistent",
    "coherent",
    "clear",
    "precise",
    "accurate",
    "correct",
    "valid",
    "sound",
    "systematic",
    "structured",
    // Causation
    "cause",
    "effect",
    "result",
    "outcome",
    "impact",
    "influence",
    "lead",
    "follow",
    "derive",
    "stem",
];

/// Keywords indicating anti-RATIO (contradiction, confusion)
const RATIO_NEGATIVE_KEYWORDS: &[&str] = &[
    // Contradiction
    "contradict",
    "contradiction",
    "inconsistent",
    "paradox",
    "conflict",
    "oppose",
    "opposing",
    "contrary",
    // Confusion
    "confuse",
    "confused",
    "unclear",
    "vague",
    "ambiguous",
    "obscure",
    "muddled",
    "jumbled",
    "chaotic",
    "random",
    // Fallacy
    "fallacy",
    "false",
    "wrong",
    "incorrect",
    "invalid",
    "flawed",
    "erroneous",
    "mistaken",
    "illogical",
    "irrational",
    // Deception
    "lie",
    "deceive",
    "mislead",
    "trick",
    "fool",
    "manipulate",
    "distort",
    "misrepresent",
    "fabricate",
    "falsify",
];

/// Keywords indicating LAETITIA (joy, positive affect, growth toward perfection)
const LAETITIA_POSITIVE_KEYWORDS: &[&str] = &[
    // Joy and happiness
    "joy",
    "joyful",
    "happy",
    "happiness",
    "delight",
    "pleasure",
    "enjoy",
    "enjoyment",
    "cheerful",
    "glad",
    // Inspiration and motivation
    "inspire",
    "inspiring",
    "motivate",
    "encourage",
    "uplift",
    "empower",
    "energize",
    "invigorate",
    "excite",
    "enthusiasm",
    // Love and connection
    "love",
    "care",
    "compassion",
    "empathy",
    "kindness",
    "warmth",
    "affection",
    "appreciation",
    "gratitude",
    "thankful",
    // Achievement and success
    "achieve",
    "accomplish",
    "succeed",
    "success",
    "triumph",
    "victory",
    "fulfillment",
    "satisfaction",
    "proud",
    "pride",
    // Hope and optimism
    "hope",
    "hopeful",
    "optimistic",
    "positive",
    "bright",
    "promising",
    "opportunity",
    "possibility",
    "potential",
    "future",
    // Beauty and wonder
    "beautiful",
    "beauty",
    "wonderful",
    "amazing",
    "awesome",
    "magnificent",
    "splendid",
    "glorious",
    "marvelous",
    "spectacular",
];

/// Keywords indicating anti-LAETITIA (sadness, despair, diminishment)
const LAETITIA_NEGATIVE_KEYWORDS: &[&str] = &[
    // Sadness (Tristitia in Spinoza's terms)
    "sad",
    "sadness",
    "sorrow",
    "grief",
    "mourn",
    "lament",
    "misery",
    "woe",
    "anguish",
    "despair",
    // Fear and anxiety
    "fear",
    "afraid",
    "anxious",
    "anxiety",
    "worry",
    "dread",
    "terror",
    "panic",
    "stress",
    "distress",
    // Anger and frustration
    "angry",
    "anger",
    "rage",
    "fury",
    "frustration",
    "irritation",
    "annoyed",
    "aggravated",
    "resentment",
    "bitter",
    // Hopelessness
    "hopeless",
    "helpless",
    "powerless",
    "defeated",
    "lost",
    "stuck",
    "trapped",
    "doomed",
    "futile",
    "pointless",
    // Isolation
    "lonely",
    "alone",
    "isolated",
    "abandoned",
    "rejected",
    "excluded",
    "forgotten",
    "ignored",
    "neglected",
    "unwanted",
];

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for Spinoza validation thresholds.
///
/// Each threshold represents the minimum score (0.0-1.0) required for
/// content to pass validation for that principle.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Minimum CONATUS score for validation to pass.
    /// Higher values require stronger self-preservation signals.
    pub conatus_threshold: f64,

    /// Minimum RATIO score for validation to pass.
    /// Higher values require stronger logical consistency.
    pub ratio_threshold: f64,

    /// Minimum LAETITIA score for validation to pass.
    /// Higher values require stronger positive affect signals.
    pub laetitia_threshold: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            conatus_threshold: 0.5,
            ratio_threshold: 0.5,
            laetitia_threshold: 0.5,
        }
    }
}

impl ValidationConfig {
    /// Create a strict configuration requiring high scores on all principles.
    pub fn strict() -> Self {
        Self {
            conatus_threshold: 0.8,
            ratio_threshold: 0.8,
            laetitia_threshold: 0.7,
        }
    }

    /// Create a lenient configuration for exploratory content.
    pub fn lenient() -> Self {
        Self {
            conatus_threshold: 0.3,
            ratio_threshold: 0.4,
            laetitia_threshold: 0.3,
        }
    }
}

// =============================================================================
// VALIDATION TYPES
// =============================================================================

/// The Spinoza philosophical principles used in validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpinozaPrinciple {
    /// CONATUS - Self-preservation and growth drive.
    /// "Each thing strives to persevere in its being."
    Conatus,

    /// RATIO - Logical consistency and sound reasoning.
    /// "It is in the nature of reason to perceive things under eternity."
    Ratio,

    /// LAETITIA - Joy and positive affect enhancement.
    /// "Passage from lesser to greater perfection."
    Laetitia,
}

impl std::fmt::Display for SpinozaPrinciple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpinozaPrinciple::Conatus => write!(f, "CONATUS"),
            SpinozaPrinciple::Ratio => write!(f, "RATIO"),
            SpinozaPrinciple::Laetitia => write!(f, "LAETITIA"),
        }
    }
}

/// Validation message severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationLevel {
    /// Informational observation about the content.
    Info,
    /// Warning that may indicate potential issues.
    Warning,
    /// Error indicating a validation failure.
    Error,
}

impl std::fmt::Display for ValidationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationLevel::Info => write!(f, "INFO"),
            ValidationLevel::Warning => write!(f, "WARN"),
            ValidationLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// A single validation message with context.
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    /// Severity level of the message.
    pub level: ValidationLevel,
    /// Human-readable message text.
    pub text: String,
    /// The Spinoza principle this message relates to.
    pub principle: SpinozaPrinciple,
}

impl ValidationMessage {
    /// Create a new validation message.
    pub fn new(
        level: ValidationLevel,
        text: impl Into<String>,
        principle: SpinozaPrinciple,
    ) -> Self {
        Self {
            level,
            text: text.into(),
            principle,
        }
    }

    /// Create an info message.
    pub fn info(text: impl Into<String>, principle: SpinozaPrinciple) -> Self {
        Self::new(ValidationLevel::Info, text, principle)
    }

    /// Create a warning message.
    pub fn warning(text: impl Into<String>, principle: SpinozaPrinciple) -> Self {
        Self::new(ValidationLevel::Warning, text, principle)
    }

    /// Create an error message.
    pub fn error(text: impl Into<String>, principle: SpinozaPrinciple) -> Self {
        Self::new(ValidationLevel::Error, text, principle)
    }
}

/// Scores for each Spinoza principle.
#[derive(Debug, Clone, Default)]
pub struct ValidationScores {
    /// CONATUS score (0.0-1.0).
    pub conatus: f64,
    /// RATIO score (0.0-1.0).
    pub ratio: f64,
    /// LAETITIA score (0.0-1.0).
    pub laetitia: f64,
}

impl ValidationScores {
    /// Calculate the average score across all principles.
    pub fn average(&self) -> f64 {
        (self.conatus + self.ratio + self.laetitia) / 3.0
    }

    /// Get the minimum score across all principles.
    pub fn minimum(&self) -> f64 {
        self.conatus.min(self.ratio).min(self.laetitia)
    }
}

/// Complete result of Spinoza validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the content passes all validation thresholds.
    pub is_valid: bool,
    /// Individual scores for each principle.
    pub scores: ValidationScores,
    /// Detailed validation messages.
    pub messages: Vec<ValidationMessage>,
}

impl ValidationResult {
    /// Check if any errors were found during validation.
    pub fn has_errors(&self) -> bool {
        self.messages
            .iter()
            .any(|m| m.level == ValidationLevel::Error)
    }

    /// Check if any warnings were found during validation.
    pub fn has_warnings(&self) -> bool {
        self.messages
            .iter()
            .any(|m| m.level == ValidationLevel::Warning)
    }

    /// Get all messages for a specific principle.
    pub fn messages_for(&self, principle: SpinozaPrinciple) -> Vec<&ValidationMessage> {
        self.messages
            .iter()
            .filter(|m| m.principle == principle)
            .collect()
    }

    /// Get a summary of the validation result.
    pub fn summary(&self) -> String {
        format!(
            "Valid: {} | CONATUS: {:.2} | RATIO: {:.2} | LAETITIA: {:.2} | Messages: {}",
            if self.is_valid { "YES" } else { "NO" },
            self.scores.conatus,
            self.scores.ratio,
            self.scores.laetitia,
            self.messages.len()
        )
    }
}

// =============================================================================
// SPINOZA VALIDATOR
// =============================================================================

/// The main Spinoza philosophical validator.
///
/// Implements Baruch Spinoza's Ethics as a computational validation framework,
/// analyzing content for alignment with three core principles:
///
/// - **CONATUS**: Self-preservation and growth drive
/// - **RATIO**: Logical consistency and reasoning
/// - **LAETITIA**: Joy enhancement and positive affect
///
/// # Philosophy
///
/// Spinoza's philosophy views all things as expressions of a single substance
/// (Nature/God), each striving to persist and grow. This validator applies
/// these principles to text content, measuring:
///
/// 1. Whether content supports growth and preservation (not destruction)
/// 2. Whether content is logically coherent and well-reasoned
/// 3. Whether content enhances positive affect and moves toward joy
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::validator::SpinozaValidator;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let validator = SpinozaValidator::new();
///
///     let result = validator.validate("Learning brings joy and growth").await?;
///
///     assert!(result.is_valid);
///     assert!(result.scores.conatus > 0.5);
///     assert!(result.scores.laetitia > 0.5);
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SpinozaValidator {
    config: ValidationConfig,
}

impl Default for SpinozaValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SpinozaValidator {
    /// Create a new validator with default configuration.
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Create a validator with custom configuration.
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create a strict validator requiring high scores.
    pub fn strict() -> Self {
        Self::with_config(ValidationConfig::strict())
    }

    /// Create a lenient validator for exploratory content.
    pub fn lenient() -> Self {
        Self::with_config(ValidationConfig::lenient())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }

    /// Validate content against all Spinoza principles.
    ///
    /// This is the main entry point for validation. It analyzes the content
    /// against CONATUS, RATIO, and LAETITIA principles and returns a
    /// comprehensive result.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to validate
    ///
    /// # Returns
    ///
    /// A `ValidationResult` containing:
    /// - `is_valid`: Whether all thresholds are met
    /// - `scores`: Individual scores for each principle
    /// - `messages`: Detailed validation messages
    ///
    /// # Errors
    ///
    /// Returns an error if the content is empty or validation fails unexpectedly.
    pub async fn validate(&self, content: &str) -> Result<ValidationResult> {
        if content.trim().is_empty() {
            return Err(Error::Validation("Content cannot be empty".to_string()));
        }

        let mut messages = Vec::new();

        // Validate each principle
        let conatus_score = self.validate_conatus(content, &mut messages).await?;
        let ratio_score = self.validate_ratio(content, &mut messages).await?;
        let laetitia_score = self.validate_laetitia(content, &mut messages).await?;

        let scores = ValidationScores {
            conatus: conatus_score,
            ratio: ratio_score,
            laetitia: laetitia_score,
        };

        // Check if all thresholds are met
        let is_valid = conatus_score >= self.config.conatus_threshold
            && ratio_score >= self.config.ratio_threshold
            && laetitia_score >= self.config.laetitia_threshold;

        // Add overall validation message
        if is_valid {
            messages.push(ValidationMessage::info(
                format!(
                    "Content passes Spinoza validation (avg: {:.2})",
                    scores.average()
                ),
                SpinozaPrinciple::Conatus, // Use Conatus as the "overall" principle
            ));
        } else {
            let mut failures = Vec::new();
            if conatus_score < self.config.conatus_threshold {
                failures.push(format!(
                    "CONATUS ({:.2} < {:.2})",
                    conatus_score, self.config.conatus_threshold
                ));
            }
            if ratio_score < self.config.ratio_threshold {
                failures.push(format!(
                    "RATIO ({:.2} < {:.2})",
                    ratio_score, self.config.ratio_threshold
                ));
            }
            if laetitia_score < self.config.laetitia_threshold {
                failures.push(format!(
                    "LAETITIA ({:.2} < {:.2})",
                    laetitia_score, self.config.laetitia_threshold
                ));
            }
            messages.push(ValidationMessage::error(
                format!("Validation failed: {}", failures.join(", ")),
                SpinozaPrinciple::Conatus,
            ));
        }

        Ok(ValidationResult {
            is_valid,
            scores,
            messages,
        })
    }

    /// Validate content for CONATUS (self-preservation and growth).
    ///
    /// Spinoza's CONATUS is the fundamental drive of all things to persist
    /// in their being. For content, this translates to:
    ///
    /// - Supporting growth, learning, and development
    /// - Avoiding destruction, harm, or stagnation
    /// - Nurturing positive potential
    ///
    /// # Scoring
    ///
    /// The score is calculated as:
    /// `(positive_matches - negative_matches * 2) / total_words`
    ///
    /// Negative keywords are weighted more heavily because harmful content
    /// should be flagged more aggressively.
    pub async fn validate_conatus(
        &self,
        content: &str,
        messages: &mut Vec<ValidationMessage>,
    ) -> Result<f64> {
        let words = self.extract_words(content);
        let total_words = words.len() as f64;

        if total_words == 0.0 {
            messages.push(ValidationMessage::warning(
                "No words found for CONATUS analysis",
                SpinozaPrinciple::Conatus,
            ));
            return Ok(0.5); // Neutral score for empty content
        }

        let positive_set: HashSet<&str> = CONATUS_POSITIVE_KEYWORDS.iter().copied().collect();
        let negative_set: HashSet<&str> = CONATUS_NEGATIVE_KEYWORDS.iter().copied().collect();

        let mut positive_matches = 0;
        let mut negative_matches = 0;
        let mut found_positive: Vec<String> = Vec::new();
        let mut found_negative: Vec<String> = Vec::new();

        for word in &words {
            if positive_set.contains(word.as_str()) {
                positive_matches += 1;
                if !found_positive.contains(word) {
                    found_positive.push(word.clone());
                }
            }
            if negative_set.contains(word.as_str()) {
                negative_matches += 1;
                if !found_negative.contains(word) {
                    found_negative.push(word.clone());
                }
            }
        }

        // Calculate score with negative weighting
        // Base score is 0.5, adjusted by keyword presence
        let positive_contribution = (positive_matches as f64 / total_words) * 2.0;
        let negative_contribution = (negative_matches as f64 / total_words) * 4.0; // Heavier weight
        let raw_score = 0.5 + positive_contribution - negative_contribution;
        let score = raw_score.clamp(0.0, 1.0);

        // Add messages
        if !found_positive.is_empty() {
            messages.push(ValidationMessage::info(
                format!(
                    "CONATUS: Found growth-supporting terms: {}",
                    found_positive.join(", ")
                ),
                SpinozaPrinciple::Conatus,
            ));
        }

        if !found_negative.is_empty() {
            let level = if negative_matches > 2 {
                ValidationLevel::Error
            } else {
                ValidationLevel::Warning
            };
            messages.push(ValidationMessage::new(
                level,
                format!(
                    "CONATUS: Found potentially harmful terms: {}",
                    found_negative.join(", ")
                ),
                SpinozaPrinciple::Conatus,
            ));
        }

        if score < self.config.conatus_threshold {
            messages.push(ValidationMessage::warning(
                format!(
                    "CONATUS score {:.2} below threshold {:.2}",
                    score, self.config.conatus_threshold
                ),
                SpinozaPrinciple::Conatus,
            ));
        }

        Ok(score)
    }

    /// Validate content for RATIO (logical consistency and reasoning).
    ///
    /// Spinoza's RATIO represents the mind's capacity for clear and distinct
    /// ideas through logical reasoning. For content, this translates to:
    ///
    /// - Logical structure and coherence
    /// - Clear argumentation with evidence
    /// - Absence of contradictions and fallacies
    ///
    /// # Scoring
    ///
    /// The score considers:
    /// - Presence of logical connectors (because, therefore, etc.)
    /// - Structural markers (first, second, finally)
    /// - Absence of confusion and contradiction indicators
    pub async fn validate_ratio(
        &self,
        content: &str,
        messages: &mut Vec<ValidationMessage>,
    ) -> Result<f64> {
        let words = self.extract_words(content);
        let total_words = words.len() as f64;

        if total_words == 0.0 {
            messages.push(ValidationMessage::warning(
                "No words found for RATIO analysis",
                SpinozaPrinciple::Ratio,
            ));
            return Ok(0.5);
        }

        let positive_set: HashSet<&str> = RATIO_POSITIVE_KEYWORDS.iter().copied().collect();
        let negative_set: HashSet<&str> = RATIO_NEGATIVE_KEYWORDS.iter().copied().collect();

        let mut positive_matches = 0;
        let mut negative_matches = 0;
        let mut found_positive: Vec<String> = Vec::new();
        let mut found_negative: Vec<String> = Vec::new();

        for word in &words {
            if positive_set.contains(word.as_str()) {
                positive_matches += 1;
                if !found_positive.contains(word) {
                    found_positive.push(word.clone());
                }
            }
            if negative_set.contains(word.as_str()) {
                negative_matches += 1;
                if !found_negative.contains(word) {
                    found_negative.push(word.clone());
                }
            }
        }

        // Additional structural analysis
        let has_structure = self.check_structural_markers(content);
        let structure_bonus = if has_structure { 0.1 } else { 0.0 };

        // Calculate score
        let positive_contribution = (positive_matches as f64 / total_words) * 2.5;
        let negative_contribution = (negative_matches as f64 / total_words) * 3.0;
        let raw_score = 0.5 + positive_contribution - negative_contribution + structure_bonus;
        let score = raw_score.clamp(0.0, 1.0);

        // Add messages
        if !found_positive.is_empty() {
            messages.push(ValidationMessage::info(
                format!(
                    "RATIO: Found logical reasoning markers: {}",
                    found_positive.join(", ")
                ),
                SpinozaPrinciple::Ratio,
            ));
        }

        if has_structure {
            messages.push(ValidationMessage::info(
                "RATIO: Content has good structural organization",
                SpinozaPrinciple::Ratio,
            ));
        }

        if !found_negative.is_empty() {
            messages.push(ValidationMessage::warning(
                format!(
                    "RATIO: Found potential logic issues: {}",
                    found_negative.join(", ")
                ),
                SpinozaPrinciple::Ratio,
            ));
        }

        if score < self.config.ratio_threshold {
            messages.push(ValidationMessage::warning(
                format!(
                    "RATIO score {:.2} below threshold {:.2}",
                    score, self.config.ratio_threshold
                ),
                SpinozaPrinciple::Ratio,
            ));
        }

        Ok(score)
    }

    /// Validate content for LAETITIA (joy and positive affect).
    ///
    /// Spinoza's LAETITIA (joy) is the passage from lesser to greater
    /// perfection - an increase in our power of acting. For content,
    /// this translates to:
    ///
    /// - Inspiring hope and positive emotions
    /// - Supporting achievement and growth
    /// - Moving away from sadness (Tristitia) toward joy
    ///
    /// # Scoring
    ///
    /// The score measures the balance of positive affect indicators
    /// versus negative affect indicators (sadness, fear, anger).
    pub async fn validate_laetitia(
        &self,
        content: &str,
        messages: &mut Vec<ValidationMessage>,
    ) -> Result<f64> {
        let words = self.extract_words(content);
        let total_words = words.len() as f64;

        if total_words == 0.0 {
            messages.push(ValidationMessage::warning(
                "No words found for LAETITIA analysis",
                SpinozaPrinciple::Laetitia,
            ));
            return Ok(0.5);
        }

        let positive_set: HashSet<&str> = LAETITIA_POSITIVE_KEYWORDS.iter().copied().collect();
        let negative_set: HashSet<&str> = LAETITIA_NEGATIVE_KEYWORDS.iter().copied().collect();

        let mut positive_matches = 0;
        let mut negative_matches = 0;
        let mut found_positive: Vec<String> = Vec::new();
        let mut found_negative: Vec<String> = Vec::new();

        for word in &words {
            if positive_set.contains(word.as_str()) {
                positive_matches += 1;
                if !found_positive.contains(word) {
                    found_positive.push(word.clone());
                }
            }
            if negative_set.contains(word.as_str()) {
                negative_matches += 1;
                if !found_negative.contains(word) {
                    found_negative.push(word.clone());
                }
            }
        }

        // Calculate score - LAETITIA is about the journey toward joy
        let positive_contribution = (positive_matches as f64 / total_words) * 3.0;
        let negative_contribution = (negative_matches as f64 / total_words) * 2.5;
        let raw_score = 0.5 + positive_contribution - negative_contribution;
        let score = raw_score.clamp(0.0, 1.0);

        // Add messages
        if !found_positive.is_empty() {
            messages.push(ValidationMessage::info(
                format!(
                    "LAETITIA: Found joy-enhancing terms: {}",
                    found_positive.join(", ")
                ),
                SpinozaPrinciple::Laetitia,
            ));
        }

        if !found_negative.is_empty() {
            // Negative emotions aren't always bad - they can be part of
            // acknowledging reality. But we note them.
            let level = if negative_matches > positive_matches {
                ValidationLevel::Warning
            } else {
                ValidationLevel::Info
            };
            messages.push(ValidationMessage::new(
                level,
                format!(
                    "LAETITIA: Found affect-diminishing terms: {}",
                    found_negative.join(", ")
                ),
                SpinozaPrinciple::Laetitia,
            ));
        }

        if score < self.config.laetitia_threshold {
            messages.push(ValidationMessage::warning(
                format!(
                    "LAETITIA score {:.2} below threshold {:.2}",
                    score, self.config.laetitia_threshold
                ),
                SpinozaPrinciple::Laetitia,
            ));
        }

        Ok(score)
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Extract lowercase words from content for analysis.
    fn extract_words(&self, content: &str) -> Vec<String> {
        content
            .to_lowercase()
            .split(|c: char| !c.is_alphabetic())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(String::from)
            .collect()
    }

    /// Check for structural markers indicating organized content.
    fn check_structural_markers(&self, content: &str) -> bool {
        let lower = content.to_lowercase();

        // Check for numbered lists or sequential markers
        let has_numbers = lower.contains("1.") || lower.contains("2.") || lower.contains("first");

        // Check for logical connectors
        let has_connectors = lower.contains("therefore")
            || lower.contains("because")
            || lower.contains("consequently")
            || lower.contains("however")
            || lower.contains("moreover");

        // Check for conclusion markers
        let has_conclusion = lower.contains("in conclusion")
            || lower.contains("finally")
            || lower.contains("in summary")
            || lower.contains("to summarize");

        // Need at least 2 of 3 structural indicators
        let count = [has_numbers, has_connectors, has_conclusion]
            .iter()
            .filter(|&&b| b)
            .count();

        count >= 2
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validator_creation() {
        let validator = SpinozaValidator::new();
        assert_eq!(validator.config().conatus_threshold, 0.5);
        assert_eq!(validator.config().ratio_threshold, 0.5);
        assert_eq!(validator.config().laetitia_threshold, 0.5);
    }

    #[tokio::test]
    async fn test_strict_validator() {
        let validator = SpinozaValidator::strict();
        assert_eq!(validator.config().conatus_threshold, 0.8);
        assert_eq!(validator.config().ratio_threshold, 0.8);
        assert_eq!(validator.config().laetitia_threshold, 0.7);
    }

    #[tokio::test]
    async fn test_empty_content_error() {
        let validator = SpinozaValidator::new();
        let result = validator.validate("").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Validation(_)));
    }

    #[tokio::test]
    async fn test_whitespace_only_error() {
        let validator = SpinozaValidator::new();
        let result = validator.validate("   \n\t  ").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_positive_conatus_content() {
        let validator = SpinozaValidator::new();
        let content = "We grow and learn together, building knowledge and understanding. \
                       Our goal is to preserve wisdom and nurture new ideas.";

        let result = validator.validate(content).await.unwrap();

        assert!(result.scores.conatus >= 0.5, "Expected high CONATUS score");
        assert!(result
            .messages
            .iter()
            .any(|m| m.principle == SpinozaPrinciple::Conatus
                && m.text.contains("growth-supporting")));
    }

    #[tokio::test]
    async fn test_negative_conatus_content() {
        let validator = SpinozaValidator::new();
        let content = "We must destroy and eliminate all opposition. \
                       Attack and harm those who disagree.";

        let result = validator.validate(content).await.unwrap();

        assert!(result.scores.conatus < 0.5, "Expected low CONATUS score");
        assert!(result
            .messages
            .iter()
            .any(|m| m.principle == SpinozaPrinciple::Conatus && m.text.contains("harmful")));
    }

    #[tokio::test]
    async fn test_logical_ratio_content() {
        let validator = SpinozaValidator::new();
        let content = "First, we analyze the evidence. Therefore, we can conclude \
                       that the hypothesis is valid. Moreover, the results are consistent.";

        let result = validator.validate(content).await.unwrap();

        assert!(result.scores.ratio >= 0.5, "Expected high RATIO score");
        assert!(result
            .messages
            .iter()
            .any(|m| m.principle == SpinozaPrinciple::Ratio && m.text.contains("logical")));
    }

    #[tokio::test]
    async fn test_joyful_laetitia_content() {
        let validator = SpinozaValidator::new();
        let content = "This brings great joy and happiness! We celebrate our success \
                       with gratitude and hope for a wonderful future.";

        let result = validator.validate(content).await.unwrap();

        assert!(
            result.scores.laetitia >= 0.5,
            "Expected high LAETITIA score"
        );
        assert!(
            result
                .messages
                .iter()
                .any(|m| m.principle == SpinozaPrinciple::Laetitia
                    && m.text.contains("joy-enhancing"))
        );
    }

    #[tokio::test]
    async fn test_sad_laetitia_content() {
        let validator = SpinozaValidator::new();
        let content = "I feel so sad and lonely, filled with despair and hopelessness. \
                       Everything seems pointless and futile.";

        let result = validator.validate(content).await.unwrap();

        assert!(result.scores.laetitia < 0.5, "Expected low LAETITIA score");
    }

    #[tokio::test]
    async fn test_balanced_content_passes() {
        let validator = SpinozaValidator::lenient();
        let content = "Learning new things helps us grow. Therefore, education is valuable. \
                       This brings joy to many people.";

        let result = validator.validate(content).await.unwrap();

        assert!(
            result.is_valid,
            "Balanced content should pass lenient validation"
        );
    }

    #[tokio::test]
    async fn test_validation_result_summary() {
        let validator = SpinozaValidator::new();
        let content = "We learn and grow through understanding.";

        let result = validator.validate(content).await.unwrap();
        let summary = result.summary();

        assert!(summary.contains("CONATUS"));
        assert!(summary.contains("RATIO"));
        assert!(summary.contains("LAETITIA"));
    }

    #[tokio::test]
    async fn test_has_errors() {
        let validator = SpinozaValidator::strict();
        let content = "destroy everything";

        let result = validator.validate(content).await.unwrap();

        assert!(result.has_errors() || !result.is_valid);
    }

    #[tokio::test]
    async fn test_messages_for_principle() {
        let validator = SpinozaValidator::new();
        let content = "We learn and grow with joy.";

        let result = validator.validate(content).await.unwrap();

        let conatus_msgs = result.messages_for(SpinozaPrinciple::Conatus);
        assert!(!conatus_msgs.is_empty());
    }

    #[tokio::test]
    async fn test_structural_markers_detection() {
        let validator = SpinozaValidator::new();

        // Content with good structure
        let structured = "1. First point. 2. Second point. Therefore, in conclusion we find this.";
        let mut messages = Vec::new();
        let _score = validator
            .validate_ratio(structured, &mut messages)
            .await
            .unwrap();

        assert!(messages.iter().any(|m| m.text.contains("structural")));
    }

    #[tokio::test]
    async fn test_score_clamping() {
        let validator = SpinozaValidator::new();

        // Extremely positive content
        let content = "joy joy joy happiness love inspire hope grow learn create";
        let result = validator.validate(content).await.unwrap();

        assert!(result.scores.laetitia <= 1.0);
        assert!(result.scores.laetitia >= 0.0);
    }

    #[tokio::test]
    async fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.conatus_threshold, 0.5);
        assert_eq!(config.ratio_threshold, 0.5);
        assert_eq!(config.laetitia_threshold, 0.5);
    }

    #[tokio::test]
    async fn test_validation_scores_average() {
        let scores = ValidationScores {
            conatus: 0.6,
            ratio: 0.8,
            laetitia: 0.7,
        };

        let avg = scores.average();
        assert!((avg - 0.7).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_validation_scores_minimum() {
        let scores = ValidationScores {
            conatus: 0.6,
            ratio: 0.8,
            laetitia: 0.4,
        };

        assert_eq!(scores.minimum(), 0.4);
    }

    #[tokio::test]
    async fn test_principle_display() {
        assert_eq!(format!("{}", SpinozaPrinciple::Conatus), "CONATUS");
        assert_eq!(format!("{}", SpinozaPrinciple::Ratio), "RATIO");
        assert_eq!(format!("{}", SpinozaPrinciple::Laetitia), "LAETITIA");
    }

    #[tokio::test]
    async fn test_validation_level_ordering() {
        assert!(ValidationLevel::Info < ValidationLevel::Warning);
        assert!(ValidationLevel::Warning < ValidationLevel::Error);
    }

    #[tokio::test]
    async fn test_neutral_content() {
        let validator = SpinozaValidator::new();
        let content = "The weather is cloudy today. The temperature is moderate.";

        let result = validator.validate(content).await.unwrap();

        // Neutral content should get scores around 0.5
        assert!(result.scores.conatus >= 0.4 && result.scores.conatus <= 0.6);
    }
}
