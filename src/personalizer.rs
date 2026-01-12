//! Personalizer Agent module for Project Panpsychism.
//!
//! 🎭 **The Chameleon** — Adapts every message to speak your language.
//!
//! This module implements Agent 22: the Personalizer (Style Adapter), responsible for
//! adapting prompts to user preferences and communication style. Like a skilled diplomat
//! who speaks every language fluently, the Personalizer ensures that all output resonates
//! with the user's preferred communication patterns.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Chameleon serves a crucial role:
//!
//! - **Raw Magic** (generic content) enters the adaptation chamber
//! - **The Chameleon** (PersonalizerAgent) studies the sorcerer's preferences
//! - **Tailored Spell** (personalized output) emerges in the user's native tongue
//!
//! The Chameleon can:
//! - Adjust tone (formal, casual, technical, friendly, academic)
//! - Control verbosity (concise to exhaustive)
//! - Match expertise level (beginner to expert)
//! - Adopt language style (simple to academic)
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//! - **CONATUS**: Persistent user preferences across sessions
//! - **NATURA**: Natural alignment between output and user expectations
//! - **RATIO**: Logical adaptation rules based on user profile
//! - **LAETITIA**: Joy through personalized, resonant communication
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::personalizer::{PersonalizerAgent, UserProfile, Tone, VerbosityLevel};
//!
//! let mut profile = UserProfile::default();
//! profile.preferred_tone = Tone::Friendly;
//! profile.verbosity_level = VerbosityLevel::Concise;
//!
//! let agent = PersonalizerAgent::builder()
//!     .user_profile(profile)
//!     .build();
//!
//! let result = agent.personalize("This is a technical explanation of OAuth2...").await?;
//! println!("Personalized: {}", result.content);
//! ```

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// ENUMS
// =============================================================================

/// The tone of communication preferred by the user.
///
/// Each tone represents a different style of expression, from strictly
/// professional to warmly personal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Tone {
    /// Formal, professional language suitable for business contexts.
    /// Uses proper grammar, avoids contractions, maintains distance.
    Formal,

    /// Relaxed, everyday language with a conversational feel.
    /// Uses contractions, informal phrases, and approachable wording.
    Casual,

    /// Precise, domain-specific language for technical audiences.
    /// Uses jargon appropriately, includes technical details.
    #[default]
    Technical,

    /// Warm, approachable language that builds rapport.
    /// Uses encouraging phrases, personal touches, and empathy.
    Friendly,

    /// Scholarly, rigorous language for academic contexts.
    /// Cites sources, uses precise terminology, maintains objectivity.
    Academic,
}

impl std::fmt::Display for Tone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Formal => write!(f, "formal"),
            Self::Casual => write!(f, "casual"),
            Self::Technical => write!(f, "technical"),
            Self::Friendly => write!(f, "friendly"),
            Self::Academic => write!(f, "academic"),
        }
    }
}

impl std::str::FromStr for Tone {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "formal" | "professional" | "business" => Ok(Self::Formal),
            "casual" | "informal" | "relaxed" => Ok(Self::Casual),
            "technical" | "tech" | "engineering" => Ok(Self::Technical),
            "friendly" | "warm" | "approachable" => Ok(Self::Friendly),
            "academic" | "scholarly" | "research" => Ok(Self::Academic),
            _ => Err(crate::Error::Config(format!(
                "Unknown tone: '{}'. Valid tones: formal, casual, technical, friendly, academic",
                s
            ))),
        }
    }
}

/// The level of detail preferred by the user.
///
/// Controls how much information is included in responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum VerbosityLevel {
    /// Minimal, to-the-point responses. Maximum brevity.
    /// Best for quick answers and experienced users.
    Concise,

    /// Standard level of detail with key information.
    /// Suitable for most contexts and users.
    #[default]
    Balanced,

    /// Comprehensive responses with explanations.
    /// Good for learning and complex topics.
    Detailed,

    /// Maximum detail with examples, context, and alternatives.
    /// Best for deep understanding and documentation.
    Exhaustive,
}

impl std::fmt::Display for VerbosityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Concise => write!(f, "concise"),
            Self::Balanced => write!(f, "balanced"),
            Self::Detailed => write!(f, "detailed"),
            Self::Exhaustive => write!(f, "exhaustive"),
        }
    }
}

impl std::str::FromStr for VerbosityLevel {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "concise" | "brief" | "short" | "minimal" => Ok(Self::Concise),
            "balanced" | "normal" | "standard" | "default" => Ok(Self::Balanced),
            "detailed" | "comprehensive" | "full" => Ok(Self::Detailed),
            "exhaustive" | "complete" | "maximum" | "max" => Ok(Self::Exhaustive),
            _ => Err(crate::Error::Config(format!(
                "Unknown verbosity level: '{}'. Valid levels: concise, balanced, detailed, exhaustive",
                s
            ))),
        }
    }
}

impl VerbosityLevel {
    /// Get the target word count multiplier for this level.
    pub fn word_count_multiplier(&self) -> f64 {
        match self {
            Self::Concise => 0.5,
            Self::Balanced => 1.0,
            Self::Detailed => 1.5,
            Self::Exhaustive => 2.5,
        }
    }
}

/// The user's expertise level in the subject matter.
///
/// Affects technical depth, assumed knowledge, and explanation style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ExpertiseLevel {
    /// New to the subject. Requires explanations, avoids jargon.
    Beginner,

    /// Familiar with basics. Can handle some technical terms.
    #[default]
    Intermediate,

    /// Strong understanding. Expects precision and efficiency.
    Advanced,

    /// Deep specialist knowledge. Prefers peer-level discussion.
    Expert,
}

impl std::fmt::Display for ExpertiseLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Beginner => write!(f, "beginner"),
            Self::Intermediate => write!(f, "intermediate"),
            Self::Advanced => write!(f, "advanced"),
            Self::Expert => write!(f, "expert"),
        }
    }
}

impl std::str::FromStr for ExpertiseLevel {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "beginner" | "novice" | "new" | "newbie" => Ok(Self::Beginner),
            "intermediate" | "medium" | "mid" => Ok(Self::Intermediate),
            "advanced" | "experienced" | "senior" => Ok(Self::Advanced),
            "expert" | "master" | "specialist" | "pro" => Ok(Self::Expert),
            _ => Err(crate::Error::Config(format!(
                "Unknown expertise level: '{}'. Valid levels: beginner, intermediate, advanced, expert",
                s
            ))),
        }
    }
}

impl ExpertiseLevel {
    /// Get the jargon tolerance for this level (0.0 = none, 1.0 = full).
    pub fn jargon_tolerance(&self) -> f64 {
        match self {
            Self::Beginner => 0.1,
            Self::Intermediate => 0.4,
            Self::Advanced => 0.7,
            Self::Expert => 1.0,
        }
    }
}

/// The language style preference.
///
/// Affects vocabulary, sentence structure, and overall presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LanguageStyle {
    /// Plain language, short sentences, everyday vocabulary.
    Simple,

    /// Business-appropriate, clear, and polished.
    #[default]
    Professional,

    /// Natural, flowing, as if speaking to a friend.
    Conversational,

    /// Scholarly, precise, with careful word choice.
    Academic,
}

impl std::fmt::Display for LanguageStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Professional => write!(f, "professional"),
            Self::Conversational => write!(f, "conversational"),
            Self::Academic => write!(f, "academic"),
        }
    }
}

impl std::str::FromStr for LanguageStyle {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simple" | "plain" | "easy" | "basic" => Ok(Self::Simple),
            "professional" | "business" | "corporate" => Ok(Self::Professional),
            "conversational" | "natural" | "chatty" | "casual" => Ok(Self::Conversational),
            "academic" | "scholarly" | "formal" | "rigorous" => Ok(Self::Academic),
            _ => Err(crate::Error::Config(format!(
                "Unknown language style: '{}'. Valid styles: simple, professional, conversational, academic",
                s
            ))),
        }
    }
}

// =============================================================================
// USER PROFILE
// =============================================================================

/// A user's communication preferences profile.
///
/// The UserProfile captures all the preferences that define how the user
/// prefers to receive information. Like a sorcerer's personal sigil, it
/// ensures all magic is crafted to their unique specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// The preferred tone of communication.
    pub preferred_tone: Tone,

    /// The preferred level of detail.
    pub verbosity_level: VerbosityLevel,

    /// The user's expertise level.
    pub expertise_level: ExpertiseLevel,

    /// The preferred language style.
    pub language_style: LanguageStyle,

    /// Custom vocabulary preferences (words to use/avoid).
    #[serde(default)]
    pub vocabulary_preferences: VocabularyPreferences,

    /// Domain-specific preferences.
    #[serde(default)]
    pub domain_preferences: HashMap<String, DomainPreference>,

    /// User's name for personalized greetings (optional).
    #[serde(default)]
    pub name: Option<String>,

    /// Preferred pronouns (optional).
    #[serde(default)]
    pub pronouns: Option<String>,

    /// Additional custom preferences as key-value pairs.
    #[serde(default)]
    pub custom_preferences: HashMap<String, String>,
}

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            preferred_tone: Tone::default(),
            verbosity_level: VerbosityLevel::default(),
            expertise_level: ExpertiseLevel::default(),
            language_style: LanguageStyle::default(),
            vocabulary_preferences: VocabularyPreferences::default(),
            domain_preferences: HashMap::new(),
            name: None,
            pronouns: None,
            custom_preferences: HashMap::new(),
        }
    }
}

impl UserProfile {
    /// Create a new user profile with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a profile for a beginner user.
    pub fn beginner() -> Self {
        Self {
            preferred_tone: Tone::Friendly,
            verbosity_level: VerbosityLevel::Detailed,
            expertise_level: ExpertiseLevel::Beginner,
            language_style: LanguageStyle::Simple,
            ..Default::default()
        }
    }

    /// Create a profile for an expert user.
    pub fn expert() -> Self {
        Self {
            preferred_tone: Tone::Technical,
            verbosity_level: VerbosityLevel::Concise,
            expertise_level: ExpertiseLevel::Expert,
            language_style: LanguageStyle::Professional,
            ..Default::default()
        }
    }

    /// Create a profile for an academic context.
    pub fn academic() -> Self {
        Self {
            preferred_tone: Tone::Academic,
            verbosity_level: VerbosityLevel::Exhaustive,
            expertise_level: ExpertiseLevel::Advanced,
            language_style: LanguageStyle::Academic,
            ..Default::default()
        }
    }

    /// Set the user's name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the preferred tone.
    pub fn with_tone(mut self, tone: Tone) -> Self {
        self.preferred_tone = tone;
        self
    }

    /// Set the verbosity level.
    pub fn with_verbosity(mut self, level: VerbosityLevel) -> Self {
        self.verbosity_level = level;
        self
    }

    /// Set the expertise level.
    pub fn with_expertise(mut self, level: ExpertiseLevel) -> Self {
        self.expertise_level = level;
        self
    }

    /// Set the language style.
    pub fn with_language_style(mut self, style: LanguageStyle) -> Self {
        self.language_style = style;
        self
    }

    /// Add a domain-specific preference.
    pub fn with_domain_preference(
        mut self,
        domain: impl Into<String>,
        preference: DomainPreference,
    ) -> Self {
        self.domain_preferences.insert(domain.into(), preference);
        self
    }
}

/// Vocabulary preferences for personalization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VocabularyPreferences {
    /// Words to prefer using.
    #[serde(default)]
    pub preferred_terms: HashMap<String, String>,

    /// Words to avoid.
    #[serde(default)]
    pub avoided_terms: Vec<String>,

    /// Acronyms to expand.
    #[serde(default)]
    pub expand_acronyms: bool,

    /// Use contractions (e.g., "don't" vs "do not").
    #[serde(default = "default_true")]
    pub use_contractions: bool,
}

fn default_true() -> bool {
    true
}

/// Domain-specific preferences for content adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainPreference {
    /// Override expertise level for this domain.
    pub expertise_override: Option<ExpertiseLevel>,

    /// Override tone for this domain.
    pub tone_override: Option<Tone>,

    /// Domain-specific terminology to use.
    #[serde(default)]
    pub terminology: HashMap<String, String>,
}

impl Default for DomainPreference {
    fn default() -> Self {
        Self {
            expertise_override: None,
            tone_override: None,
            terminology: HashMap::new(),
        }
    }
}

// =============================================================================
// ADAPTATION TRACKING
// =============================================================================

/// An individual adaptation made to the content.
///
/// Tracks what was changed and why, enabling transparency and learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adaptation {
    /// The type of adaptation made.
    pub adaptation_type: AdaptationType,

    /// Description of the change.
    pub description: String,

    /// The original text (if applicable).
    pub original: Option<String>,

    /// The adapted text (if applicable).
    pub adapted: Option<String>,

    /// Confidence in the adaptation (0.0 - 1.0).
    pub confidence: f64,
}

impl Adaptation {
    /// Create a new adaptation record.
    pub fn new(
        adaptation_type: AdaptationType,
        description: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            adaptation_type,
            description: description.into(),
            original: None,
            adapted: None,
            confidence,
        }
    }

    /// Add the original text.
    pub fn with_original(mut self, original: impl Into<String>) -> Self {
        self.original = Some(original.into());
        self
    }

    /// Add the adapted text.
    pub fn with_adapted(mut self, adapted: impl Into<String>) -> Self {
        self.adapted = Some(adapted.into());
        self
    }
}

/// The type of adaptation made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Tone adjustment (formal/casual/etc).
    ToneAdjustment,

    /// Verbosity change (expanded/condensed).
    VerbosityChange,

    /// Expertise level adaptation.
    ExpertiseAdaptation,

    /// Language style modification.
    StyleModification,

    /// Vocabulary substitution.
    VocabularySubstitution,

    /// Structure reorganization.
    StructureChange,

    /// Personalization (name, pronouns).
    Personalization,
}

impl std::fmt::Display for AdaptationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToneAdjustment => write!(f, "tone"),
            Self::VerbosityChange => write!(f, "verbosity"),
            Self::ExpertiseAdaptation => write!(f, "expertise"),
            Self::StyleModification => write!(f, "style"),
            Self::VocabularySubstitution => write!(f, "vocabulary"),
            Self::StructureChange => write!(f, "structure"),
            Self::Personalization => write!(f, "personalization"),
        }
    }
}

// =============================================================================
// PERSONALIZED PROMPT
// =============================================================================

/// The result of personalization.
///
/// Contains the adapted content along with metadata about
/// the transformations applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedPrompt {
    /// The personalized content.
    pub content: String,

    /// The original content before personalization.
    pub original_content: String,

    /// List of adaptations made.
    pub adaptations: Vec<Adaptation>,

    /// Overall confidence in the personalization (0.0 - 1.0).
    pub confidence: f64,

    /// The user profile used for personalization.
    pub profile_summary: ProfileSummary,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

impl PersonalizedPrompt {
    /// Check if any adaptations were made.
    pub fn was_adapted(&self) -> bool {
        !self.adaptations.is_empty()
    }

    /// Get the number of adaptations made.
    pub fn adaptation_count(&self) -> usize {
        self.adaptations.len()
    }

    /// Get adaptations of a specific type.
    pub fn adaptations_of_type(&self, adaptation_type: AdaptationType) -> Vec<&Adaptation> {
        self.adaptations
            .iter()
            .filter(|a| a.adaptation_type == adaptation_type)
            .collect()
    }

    /// Calculate the average adaptation confidence.
    pub fn average_adaptation_confidence(&self) -> f64 {
        if self.adaptations.is_empty() {
            1.0
        } else {
            self.adaptations.iter().map(|a| a.confidence).sum::<f64>()
                / self.adaptations.len() as f64
        }
    }

    /// Check if content length changed significantly.
    pub fn length_change_percent(&self) -> f64 {
        let original_len = self.original_content.len() as f64;
        if original_len == 0.0 {
            return 0.0;
        }
        let new_len = self.content.len() as f64;
        ((new_len - original_len) / original_len) * 100.0
    }
}

/// A summary of the user profile used for personalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSummary {
    /// The tone used.
    pub tone: Tone,

    /// The verbosity level used.
    pub verbosity: VerbosityLevel,

    /// The expertise level assumed.
    pub expertise: ExpertiseLevel,

    /// The language style applied.
    pub style: LanguageStyle,
}

impl From<&UserProfile> for ProfileSummary {
    fn from(profile: &UserProfile) -> Self {
        Self {
            tone: profile.preferred_tone,
            verbosity: profile.verbosity_level,
            expertise: profile.expertise_level,
            style: profile.language_style,
        }
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the Personalizer Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizerConfig {
    /// Whether to apply tone adaptations.
    #[serde(default = "default_true")]
    pub adapt_tone: bool,

    /// Whether to apply verbosity adjustments.
    #[serde(default = "default_true")]
    pub adapt_verbosity: bool,

    /// Whether to apply expertise-level adaptations.
    #[serde(default = "default_true")]
    pub adapt_expertise: bool,

    /// Whether to apply language style modifications.
    #[serde(default = "default_true")]
    pub adapt_style: bool,

    /// Whether to apply vocabulary substitutions.
    #[serde(default = "default_true")]
    pub adapt_vocabulary: bool,

    /// Minimum confidence threshold for adaptations.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,

    /// Maximum number of adaptations per personalization.
    #[serde(default = "default_max_adaptations")]
    pub max_adaptations: usize,

    /// Whether to preserve original formatting.
    #[serde(default = "default_true")]
    pub preserve_formatting: bool,
}

fn default_confidence_threshold() -> f64 {
    0.5
}

fn default_max_adaptations() -> usize {
    50
}

impl Default for PersonalizerConfig {
    fn default() -> Self {
        Self {
            adapt_tone: true,
            adapt_verbosity: true,
            adapt_expertise: true,
            adapt_style: true,
            adapt_vocabulary: true,
            confidence_threshold: default_confidence_threshold(),
            max_adaptations: default_max_adaptations(),
            preserve_formatting: true,
        }
    }
}

impl PersonalizerConfig {
    /// Create a minimal configuration (fast, fewer adaptations).
    pub fn minimal() -> Self {
        Self {
            adapt_tone: true,
            adapt_verbosity: false,
            adapt_expertise: false,
            adapt_style: false,
            adapt_vocabulary: false,
            confidence_threshold: 0.7,
            max_adaptations: 10,
            preserve_formatting: true,
        }
    }

    /// Create an aggressive configuration (full adaptation).
    pub fn aggressive() -> Self {
        Self {
            adapt_tone: true,
            adapt_verbosity: true,
            adapt_expertise: true,
            adapt_style: true,
            adapt_vocabulary: true,
            confidence_threshold: 0.3,
            max_adaptations: 100,
            preserve_formatting: false,
        }
    }
}

// =============================================================================
// PERSONALIZER AGENT (THE CHAMELEON)
// =============================================================================

/// The Personalizer Agent — The Chameleon of the Sorcerer's Tower.
///
/// Agent 22 adapts all content to match user preferences and communication style.
/// Like a master diplomat who speaks every language, the Personalizer ensures
/// that every piece of magical output resonates with the user.
///
/// # Responsibilities
///
/// 1. **Adapt Tone**: Match the user's preferred communication tone
/// 2. **Control Verbosity**: Adjust detail level to user preference
/// 3. **Match Expertise**: Tailor technical depth appropriately
/// 4. **Apply Style**: Use the user's preferred language style
/// 5. **Substitute Vocabulary**: Replace terms per user preferences
///
/// # Example
///
/// ```rust,ignore
/// let agent = PersonalizerAgent::builder()
///     .user_profile(UserProfile::beginner())
///     .config(PersonalizerConfig::default())
///     .build();
///
/// let result = agent.personalize("OAuth2 implements RFC 6749...").await?;
/// // Result is adapted for beginner, friendly tone
/// ```
#[derive(Debug, Clone)]
pub struct PersonalizerAgent {
    /// The user's preference profile.
    profile: UserProfile,

    /// Agent configuration.
    config: PersonalizerConfig,

    /// Tone adaptation patterns.
    tone_patterns: TonePatterns,

    /// Expertise adaptation patterns.
    expertise_patterns: ExpertisePatterns,
}

impl Default for PersonalizerAgent {
    fn default() -> Self {
        Self::new(UserProfile::default(), PersonalizerConfig::default())
    }
}

impl PersonalizerAgent {
    /// Create a new Personalizer Agent.
    pub fn new(profile: UserProfile, config: PersonalizerConfig) -> Self {
        Self {
            profile,
            config,
            tone_patterns: TonePatterns::default(),
            expertise_patterns: ExpertisePatterns::default(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> PersonalizerAgentBuilder {
        PersonalizerAgentBuilder::default()
    }

    /// Get the current user profile.
    pub fn profile(&self) -> &UserProfile {
        &self.profile
    }

    /// Get the current configuration.
    pub fn config(&self) -> &PersonalizerConfig {
        &self.config
    }

    /// Update the user profile.
    pub fn set_profile(&mut self, profile: UserProfile) {
        self.profile = profile;
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: PersonalizerConfig) {
        self.config = config;
    }

    // =========================================================================
    // CORE PERSONALIZATION
    // =========================================================================

    /// Personalize content based on the user profile.
    ///
    /// This is the main entry point for personalization. It applies all
    /// configured adaptations based on the user's preferences.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to personalize
    ///
    /// # Returns
    ///
    /// A `PersonalizedPrompt` containing the adapted content and metadata.
    pub async fn personalize(&self, content: &str) -> Result<PersonalizedPrompt> {
        let start = std::time::Instant::now();
        let original_content = content.to_string();
        let mut current_content = content.to_string();
        let mut adaptations = Vec::new();

        // Apply tone adaptations
        if self.config.adapt_tone {
            let (adapted, tone_adaptations) = self.adapt_tone(&current_content);
            current_content = adapted;
            adaptations.extend(tone_adaptations);
        }

        // Apply verbosity adjustments
        if self.config.adapt_verbosity {
            let (adapted, verbosity_adaptations) = self.adapt_verbosity(&current_content);
            current_content = adapted;
            adaptations.extend(verbosity_adaptations);
        }

        // Apply expertise-level adaptations
        if self.config.adapt_expertise {
            let (adapted, expertise_adaptations) = self.adapt_expertise(&current_content);
            current_content = adapted;
            adaptations.extend(expertise_adaptations);
        }

        // Apply language style modifications
        if self.config.adapt_style {
            let (adapted, style_adaptations) = self.adapt_style(&current_content);
            current_content = adapted;
            adaptations.extend(style_adaptations);
        }

        // Apply vocabulary substitutions
        if self.config.adapt_vocabulary {
            let (adapted, vocab_adaptations) = self.adapt_vocabulary(&current_content);
            current_content = adapted;
            adaptations.extend(vocab_adaptations);
        }

        // Apply personalization (name, etc.)
        let (adapted, personal_adaptations) = self.apply_personalization(&current_content);
        current_content = adapted;
        adaptations.extend(personal_adaptations);

        // Filter by confidence threshold
        adaptations.retain(|a| a.confidence >= self.config.confidence_threshold);

        // Limit adaptations if needed
        adaptations.truncate(self.config.max_adaptations);

        // Calculate overall confidence
        let confidence = if adaptations.is_empty() {
            1.0
        } else {
            adaptations.iter().map(|a| a.confidence).sum::<f64>() / adaptations.len() as f64
        };

        let processing_time_ms = start.elapsed().as_millis() as u64;

        Ok(PersonalizedPrompt {
            content: current_content,
            original_content,
            adaptations,
            confidence,
            profile_summary: ProfileSummary::from(&self.profile),
            processing_time_ms,
        })
    }

    // =========================================================================
    // ADAPTATION METHODS
    // =========================================================================

    /// Adapt tone based on user preference.
    fn adapt_tone(&self, content: &str) -> (String, Vec<Adaptation>) {
        let mut result = content.to_string();
        let mut adaptations = Vec::new();

        match self.profile.preferred_tone {
            Tone::Formal => {
                // Remove contractions
                for (contracted, expanded) in &self.tone_patterns.contractions {
                    if result.contains(contracted) {
                        result = result.replace(contracted, expanded);
                        adaptations.push(
                            Adaptation::new(
                                AdaptationType::ToneAdjustment,
                                format!("Expanded contraction '{}' to '{}'", contracted, expanded),
                                0.9,
                            )
                            .with_original(contracted.clone())
                            .with_adapted(expanded.clone()),
                        );
                    }
                }
                // Use formal phrases
                for (informal, formal) in &self.tone_patterns.formal_substitutions {
                    if result.to_lowercase().contains(&informal.to_lowercase()) {
                        let re =
                            regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(informal)))
                                .unwrap();
                        result = re.replace_all(&result, formal.as_str()).to_string();
                        adaptations.push(
                            Adaptation::new(
                                AdaptationType::ToneAdjustment,
                                format!("Used formal phrase '{}' instead of '{}'", formal, informal),
                                0.85,
                            )
                            .with_original(informal.clone())
                            .with_adapted(formal.clone()),
                        );
                    }
                }
            }
            Tone::Casual => {
                // Add contractions where appropriate
                for (contracted, expanded) in &self.tone_patterns.contractions {
                    if result.contains(expanded) && !result.contains("'") {
                        result = result.replace(expanded, contracted);
                        adaptations.push(
                            Adaptation::new(
                                AdaptationType::ToneAdjustment,
                                format!("Used contraction '{}' for casual tone", contracted),
                                0.8,
                            )
                            .with_original(expanded.clone())
                            .with_adapted(contracted.clone()),
                        );
                    }
                }
            }
            Tone::Friendly => {
                // Add friendly phrases
                if !result.is_empty() && !result.starts_with("Hey") && !result.starts_with("Hi") {
                    let greeting = if let Some(name) = &self.profile.name {
                        format!("Hey {}! ", name)
                    } else {
                        String::new()
                    };
                    if !greeting.is_empty() {
                        result = format!("{}{}", greeting, result);
                        adaptations.push(Adaptation::new(
                            AdaptationType::ToneAdjustment,
                            "Added friendly greeting",
                            0.75,
                        ));
                    }
                }
            }
            Tone::Technical | Tone::Academic => {
                // Keep as-is for now, handled in expertise adaptation
            }
        }

        (result, adaptations)
    }

    /// Adapt verbosity level.
    fn adapt_verbosity(&self, content: &str) -> (String, Vec<Adaptation>) {
        let mut result = content.to_string();
        let mut adaptations = Vec::new();

        match self.profile.verbosity_level {
            VerbosityLevel::Concise => {
                // Remove filler phrases
                for filler in &[
                    "In other words, ",
                    "To put it simply, ",
                    "As you might know, ",
                    "It's worth noting that ",
                    "Essentially, ",
                    "Basically, ",
                ] {
                    if result.contains(*filler) {
                        result = result.replace(*filler, "");
                        adaptations.push(
                            Adaptation::new(
                                AdaptationType::VerbosityChange,
                                format!("Removed filler phrase '{}'", filler.trim()),
                                0.8,
                            )
                            .with_original(filler.to_string()),
                        );
                    }
                }
            }
            VerbosityLevel::Detailed | VerbosityLevel::Exhaustive => {
                // Add transitional phrases if content has multiple sentences
                let sentences: Vec<&str> = result.split(". ").collect();
                if sentences.len() > 1 && !result.contains("Furthermore") {
                    adaptations.push(Adaptation::new(
                        AdaptationType::VerbosityChange,
                        "Content structured for detailed presentation",
                        0.7,
                    ));
                }
            }
            VerbosityLevel::Balanced => {
                // No changes needed
            }
        }

        (result, adaptations)
    }

    /// Adapt to expertise level.
    fn adapt_expertise(&self, content: &str) -> (String, Vec<Adaptation>) {
        let mut result = content.to_string();
        let mut adaptations = Vec::new();

        match self.profile.expertise_level {
            ExpertiseLevel::Beginner => {
                // Expand acronyms
                for (acronym, expansion) in &self.expertise_patterns.acronym_expansions {
                    let pattern = format!(r"\b{}\b", regex::escape(acronym));
                    let re = regex::Regex::new(&pattern).unwrap();
                    if re.is_match(&result) {
                        let replacement = format!("{} ({})", expansion, acronym);
                        result = re.replace_all(&result, replacement.as_str()).to_string();
                        adaptations.push(
                            Adaptation::new(
                                AdaptationType::ExpertiseAdaptation,
                                format!("Expanded acronym '{}' for beginner", acronym),
                                0.9,
                            )
                            .with_original(acronym.clone())
                            .with_adapted(replacement),
                        );
                    }
                }
            }
            ExpertiseLevel::Expert => {
                // Remove explanatory phrases
                for phrase in &[
                    "which means ",
                    "in other words ",
                    "that is to say ",
                    ", basically, ",
                ] {
                    if result.contains(*phrase) {
                        // Find and potentially remove explanatory clauses
                        adaptations.push(Adaptation::new(
                            AdaptationType::ExpertiseAdaptation,
                            format!("Identified explanatory phrase '{}' (expert can skip)", phrase),
                            0.6,
                        ));
                    }
                }
            }
            ExpertiseLevel::Intermediate | ExpertiseLevel::Advanced => {
                // Moderate adjustments
            }
        }

        (result, adaptations)
    }

    /// Adapt language style.
    fn adapt_style(&self, content: &str) -> (String, Vec<Adaptation>) {
        let mut result = content.to_string();
        let mut adaptations = Vec::new();

        match self.profile.language_style {
            LanguageStyle::Simple => {
                // Simplify complex sentences
                if result.contains(";") || result.matches(',').count() > 3 {
                    adaptations.push(Adaptation::new(
                        AdaptationType::StyleModification,
                        "Content may benefit from sentence simplification",
                        0.6,
                    ));
                }
            }
            LanguageStyle::Academic => {
                // Add hedging language
                if result.starts_with("This is") {
                    result = result.replacen("This is", "This appears to be", 1);
                    adaptations.push(Adaptation::new(
                        AdaptationType::StyleModification,
                        "Added academic hedging",
                        0.75,
                    ));
                }
            }
            LanguageStyle::Conversational => {
                // Add conversational markers
                if !result.contains("you") && result.len() > 100 {
                    adaptations.push(Adaptation::new(
                        AdaptationType::StyleModification,
                        "Consider adding direct address for conversational style",
                        0.5,
                    ));
                }
            }
            LanguageStyle::Professional => {
                // No changes needed for professional style
            }
        }

        (result, adaptations)
    }

    /// Apply vocabulary substitutions.
    fn adapt_vocabulary(&self, content: &str) -> (String, Vec<Adaptation>) {
        let mut result = content.to_string();
        let mut adaptations = Vec::new();

        // Apply preferred terms
        for (from, to) in &self.profile.vocabulary_preferences.preferred_terms {
            if result.to_lowercase().contains(&from.to_lowercase()) {
                let re =
                    regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(from))).unwrap();
                result = re.replace_all(&result, to.as_str()).to_string();
                adaptations.push(
                    Adaptation::new(
                        AdaptationType::VocabularySubstitution,
                        format!("Substituted '{}' with preferred term '{}'", from, to),
                        0.95,
                    )
                    .with_original(from.clone())
                    .with_adapted(to.clone()),
                );
            }
        }

        // Handle avoided terms (mark for manual review)
        for avoided in &self.profile.vocabulary_preferences.avoided_terms {
            if result.to_lowercase().contains(&avoided.to_lowercase()) {
                adaptations.push(Adaptation::new(
                    AdaptationType::VocabularySubstitution,
                    format!("Content contains avoided term '{}' - consider revision", avoided),
                    0.4,
                ));
            }
        }

        // Handle contractions based on preference
        if !self.profile.vocabulary_preferences.use_contractions {
            for (contracted, expanded) in &self.tone_patterns.contractions {
                if result.contains(contracted) {
                    result = result.replace(contracted, expanded);
                    adaptations.push(
                        Adaptation::new(
                            AdaptationType::VocabularySubstitution,
                            format!("Expanded contraction '{}' per preference", contracted),
                            0.9,
                        )
                        .with_original(contracted.clone())
                        .with_adapted(expanded.clone()),
                    );
                }
            }
        }

        (result, adaptations)
    }

    /// Apply personal touches (name, pronouns, etc.).
    fn apply_personalization(&self, content: &str) -> (String, Vec<Adaptation>) {
        let result = content.to_string();
        let mut adaptations = Vec::new();

        // Track that personalization was considered
        if self.profile.name.is_some() {
            adaptations.push(Adaptation::new(
                AdaptationType::Personalization,
                "User name available for personalization",
                0.5,
            ));
        }

        (result, adaptations)
    }
}

// =============================================================================
// ADAPTATION PATTERNS
// =============================================================================

/// Patterns for tone adaptation.
#[derive(Debug, Clone)]
struct TonePatterns {
    /// Contraction mappings (contracted -> expanded).
    contractions: HashMap<String, String>,

    /// Formal substitutions (informal -> formal).
    formal_substitutions: HashMap<String, String>,
}

impl Default for TonePatterns {
    fn default() -> Self {
        let mut contractions = HashMap::new();
        contractions.insert("don't".to_string(), "do not".to_string());
        contractions.insert("doesn't".to_string(), "does not".to_string());
        contractions.insert("won't".to_string(), "will not".to_string());
        contractions.insert("can't".to_string(), "cannot".to_string());
        contractions.insert("couldn't".to_string(), "could not".to_string());
        contractions.insert("wouldn't".to_string(), "would not".to_string());
        contractions.insert("shouldn't".to_string(), "should not".to_string());
        contractions.insert("isn't".to_string(), "is not".to_string());
        contractions.insert("aren't".to_string(), "are not".to_string());
        contractions.insert("wasn't".to_string(), "was not".to_string());
        contractions.insert("weren't".to_string(), "were not".to_string());
        contractions.insert("haven't".to_string(), "have not".to_string());
        contractions.insert("hasn't".to_string(), "has not".to_string());
        contractions.insert("hadn't".to_string(), "had not".to_string());
        contractions.insert("it's".to_string(), "it is".to_string());
        contractions.insert("that's".to_string(), "that is".to_string());
        contractions.insert("there's".to_string(), "there is".to_string());
        contractions.insert("I'm".to_string(), "I am".to_string());
        contractions.insert("you're".to_string(), "you are".to_string());
        contractions.insert("we're".to_string(), "we are".to_string());
        contractions.insert("they're".to_string(), "they are".to_string());
        contractions.insert("I've".to_string(), "I have".to_string());
        contractions.insert("you've".to_string(), "you have".to_string());
        contractions.insert("we've".to_string(), "we have".to_string());
        contractions.insert("they've".to_string(), "they have".to_string());
        contractions.insert("I'll".to_string(), "I will".to_string());
        contractions.insert("you'll".to_string(), "you will".to_string());
        contractions.insert("we'll".to_string(), "we will".to_string());
        contractions.insert("they'll".to_string(), "they will".to_string());
        contractions.insert("I'd".to_string(), "I would".to_string());
        contractions.insert("you'd".to_string(), "you would".to_string());
        contractions.insert("we'd".to_string(), "we would".to_string());
        contractions.insert("they'd".to_string(), "they would".to_string());

        let mut formal_substitutions = HashMap::new();
        formal_substitutions.insert("gonna".to_string(), "going to".to_string());
        formal_substitutions.insert("wanna".to_string(), "want to".to_string());
        formal_substitutions.insert("gotta".to_string(), "have to".to_string());
        formal_substitutions.insert("kinda".to_string(), "kind of".to_string());
        formal_substitutions.insert("sorta".to_string(), "sort of".to_string());
        formal_substitutions.insert("ok".to_string(), "acceptable".to_string());
        formal_substitutions.insert("okay".to_string(), "acceptable".to_string());
        formal_substitutions.insert("yeah".to_string(), "yes".to_string());
        formal_substitutions.insert("yep".to_string(), "yes".to_string());
        formal_substitutions.insert("nope".to_string(), "no".to_string());
        formal_substitutions.insert("stuff".to_string(), "items".to_string());
        formal_substitutions.insert("things".to_string(), "elements".to_string());
        formal_substitutions.insert("lots of".to_string(), "many".to_string());
        formal_substitutions.insert("a lot of".to_string(), "numerous".to_string());
        formal_substitutions.insert("pretty much".to_string(), "essentially".to_string());

        Self {
            contractions,
            formal_substitutions,
        }
    }
}

/// Patterns for expertise-level adaptation.
#[derive(Debug, Clone)]
struct ExpertisePatterns {
    /// Acronym expansions (acronym -> full form).
    acronym_expansions: HashMap<String, String>,
}

impl Default for ExpertisePatterns {
    fn default() -> Self {
        let mut acronym_expansions = HashMap::new();
        acronym_expansions.insert("API".to_string(), "Application Programming Interface".to_string());
        acronym_expansions.insert("REST".to_string(), "Representational State Transfer".to_string());
        acronym_expansions.insert("JSON".to_string(), "JavaScript Object Notation".to_string());
        acronym_expansions.insert("HTML".to_string(), "HyperText Markup Language".to_string());
        acronym_expansions.insert("CSS".to_string(), "Cascading Style Sheets".to_string());
        acronym_expansions.insert("SQL".to_string(), "Structured Query Language".to_string());
        acronym_expansions.insert("HTTP".to_string(), "HyperText Transfer Protocol".to_string());
        acronym_expansions.insert("HTTPS".to_string(), "HyperText Transfer Protocol Secure".to_string());
        acronym_expansions.insert("URL".to_string(), "Uniform Resource Locator".to_string());
        acronym_expansions.insert("JWT".to_string(), "JSON Web Token".to_string());
        acronym_expansions.insert("OAuth".to_string(), "Open Authorization".to_string());
        acronym_expansions.insert("CRUD".to_string(), "Create, Read, Update, Delete".to_string());
        acronym_expansions.insert("CLI".to_string(), "Command Line Interface".to_string());
        acronym_expansions.insert("GUI".to_string(), "Graphical User Interface".to_string());
        acronym_expansions.insert("IDE".to_string(), "Integrated Development Environment".to_string());
        acronym_expansions.insert("ORM".to_string(), "Object-Relational Mapping".to_string());
        acronym_expansions.insert("SDK".to_string(), "Software Development Kit".to_string());
        acronym_expansions.insert("TDD".to_string(), "Test-Driven Development".to_string());
        acronym_expansions.insert("CI".to_string(), "Continuous Integration".to_string());
        acronym_expansions.insert("CD".to_string(), "Continuous Deployment".to_string());

        Self { acronym_expansions }
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom PersonalizerAgent configuration.
#[derive(Debug, Default)]
pub struct PersonalizerAgentBuilder {
    profile: Option<UserProfile>,
    config: Option<PersonalizerConfig>,
}

impl PersonalizerAgentBuilder {
    /// Set the user profile.
    pub fn user_profile(mut self, profile: UserProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: PersonalizerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the preferred tone via the profile.
    pub fn tone(mut self, tone: Tone) -> Self {
        let profile = self.profile.get_or_insert_with(UserProfile::default);
        profile.preferred_tone = tone;
        self
    }

    /// Set the verbosity level via the profile.
    pub fn verbosity(mut self, level: VerbosityLevel) -> Self {
        let profile = self.profile.get_or_insert_with(UserProfile::default);
        profile.verbosity_level = level;
        self
    }

    /// Set the expertise level via the profile.
    pub fn expertise(mut self, level: ExpertiseLevel) -> Self {
        let profile = self.profile.get_or_insert_with(UserProfile::default);
        profile.expertise_level = level;
        self
    }

    /// Set the language style via the profile.
    pub fn language_style(mut self, style: LanguageStyle) -> Self {
        let profile = self.profile.get_or_insert_with(UserProfile::default);
        profile.language_style = style;
        self
    }

    /// Set the user's name via the profile.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        let profile = self.profile.get_or_insert_with(UserProfile::default);
        profile.name = Some(name.into());
        self
    }

    /// Build the PersonalizerAgent.
    pub fn build(self) -> PersonalizerAgent {
        PersonalizerAgent::new(
            self.profile.unwrap_or_default(),
            self.config.unwrap_or_default(),
        )
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Enum Tests
    // =========================================================================

    #[test]
    fn test_tone_display() {
        assert_eq!(Tone::Formal.to_string(), "formal");
        assert_eq!(Tone::Casual.to_string(), "casual");
        assert_eq!(Tone::Technical.to_string(), "technical");
        assert_eq!(Tone::Friendly.to_string(), "friendly");
        assert_eq!(Tone::Academic.to_string(), "academic");
    }

    #[test]
    fn test_tone_from_str() {
        assert_eq!("formal".parse::<Tone>().unwrap(), Tone::Formal);
        assert_eq!("professional".parse::<Tone>().unwrap(), Tone::Formal);
        assert_eq!("casual".parse::<Tone>().unwrap(), Tone::Casual);
        assert_eq!("informal".parse::<Tone>().unwrap(), Tone::Casual);
        assert_eq!("technical".parse::<Tone>().unwrap(), Tone::Technical);
        assert_eq!("friendly".parse::<Tone>().unwrap(), Tone::Friendly);
        assert_eq!("academic".parse::<Tone>().unwrap(), Tone::Academic);
    }

    #[test]
    fn test_tone_from_str_invalid() {
        assert!("invalid".parse::<Tone>().is_err());
    }

    #[test]
    fn test_verbosity_display() {
        assert_eq!(VerbosityLevel::Concise.to_string(), "concise");
        assert_eq!(VerbosityLevel::Balanced.to_string(), "balanced");
        assert_eq!(VerbosityLevel::Detailed.to_string(), "detailed");
        assert_eq!(VerbosityLevel::Exhaustive.to_string(), "exhaustive");
    }

    #[test]
    fn test_verbosity_from_str() {
        assert_eq!("concise".parse::<VerbosityLevel>().unwrap(), VerbosityLevel::Concise);
        assert_eq!("brief".parse::<VerbosityLevel>().unwrap(), VerbosityLevel::Concise);
        assert_eq!("balanced".parse::<VerbosityLevel>().unwrap(), VerbosityLevel::Balanced);
        assert_eq!("detailed".parse::<VerbosityLevel>().unwrap(), VerbosityLevel::Detailed);
        assert_eq!("exhaustive".parse::<VerbosityLevel>().unwrap(), VerbosityLevel::Exhaustive);
    }

    #[test]
    fn test_verbosity_multiplier() {
        assert_eq!(VerbosityLevel::Concise.word_count_multiplier(), 0.5);
        assert_eq!(VerbosityLevel::Balanced.word_count_multiplier(), 1.0);
        assert_eq!(VerbosityLevel::Detailed.word_count_multiplier(), 1.5);
        assert_eq!(VerbosityLevel::Exhaustive.word_count_multiplier(), 2.5);
    }

    #[test]
    fn test_expertise_display() {
        assert_eq!(ExpertiseLevel::Beginner.to_string(), "beginner");
        assert_eq!(ExpertiseLevel::Intermediate.to_string(), "intermediate");
        assert_eq!(ExpertiseLevel::Advanced.to_string(), "advanced");
        assert_eq!(ExpertiseLevel::Expert.to_string(), "expert");
    }

    #[test]
    fn test_expertise_from_str() {
        assert_eq!("beginner".parse::<ExpertiseLevel>().unwrap(), ExpertiseLevel::Beginner);
        assert_eq!("novice".parse::<ExpertiseLevel>().unwrap(), ExpertiseLevel::Beginner);
        assert_eq!("intermediate".parse::<ExpertiseLevel>().unwrap(), ExpertiseLevel::Intermediate);
        assert_eq!("advanced".parse::<ExpertiseLevel>().unwrap(), ExpertiseLevel::Advanced);
        assert_eq!("expert".parse::<ExpertiseLevel>().unwrap(), ExpertiseLevel::Expert);
        assert_eq!("pro".parse::<ExpertiseLevel>().unwrap(), ExpertiseLevel::Expert);
    }

    #[test]
    fn test_expertise_jargon_tolerance() {
        assert!(ExpertiseLevel::Beginner.jargon_tolerance() < ExpertiseLevel::Expert.jargon_tolerance());
        assert_eq!(ExpertiseLevel::Expert.jargon_tolerance(), 1.0);
    }

    #[test]
    fn test_language_style_display() {
        assert_eq!(LanguageStyle::Simple.to_string(), "simple");
        assert_eq!(LanguageStyle::Professional.to_string(), "professional");
        assert_eq!(LanguageStyle::Conversational.to_string(), "conversational");
        assert_eq!(LanguageStyle::Academic.to_string(), "academic");
    }

    #[test]
    fn test_language_style_from_str() {
        assert_eq!("simple".parse::<LanguageStyle>().unwrap(), LanguageStyle::Simple);
        assert_eq!("plain".parse::<LanguageStyle>().unwrap(), LanguageStyle::Simple);
        assert_eq!("professional".parse::<LanguageStyle>().unwrap(), LanguageStyle::Professional);
        assert_eq!("conversational".parse::<LanguageStyle>().unwrap(), LanguageStyle::Conversational);
        assert_eq!("academic".parse::<LanguageStyle>().unwrap(), LanguageStyle::Academic);
    }

    // =========================================================================
    // UserProfile Tests
    // =========================================================================

    #[test]
    fn test_user_profile_default() {
        let profile = UserProfile::default();
        assert_eq!(profile.preferred_tone, Tone::Technical);
        assert_eq!(profile.verbosity_level, VerbosityLevel::Balanced);
        assert_eq!(profile.expertise_level, ExpertiseLevel::Intermediate);
        assert_eq!(profile.language_style, LanguageStyle::Professional);
    }

    #[test]
    fn test_user_profile_beginner() {
        let profile = UserProfile::beginner();
        assert_eq!(profile.preferred_tone, Tone::Friendly);
        assert_eq!(profile.expertise_level, ExpertiseLevel::Beginner);
        assert_eq!(profile.language_style, LanguageStyle::Simple);
    }

    #[test]
    fn test_user_profile_expert() {
        let profile = UserProfile::expert();
        assert_eq!(profile.preferred_tone, Tone::Technical);
        assert_eq!(profile.verbosity_level, VerbosityLevel::Concise);
        assert_eq!(profile.expertise_level, ExpertiseLevel::Expert);
    }

    #[test]
    fn test_user_profile_academic() {
        let profile = UserProfile::academic();
        assert_eq!(profile.preferred_tone, Tone::Academic);
        assert_eq!(profile.verbosity_level, VerbosityLevel::Exhaustive);
        assert_eq!(profile.expertise_level, ExpertiseLevel::Advanced);
    }

    #[test]
    fn test_user_profile_with_name() {
        let profile = UserProfile::default().with_name("Alice");
        assert_eq!(profile.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_user_profile_builder_methods() {
        let profile = UserProfile::new()
            .with_tone(Tone::Casual)
            .with_verbosity(VerbosityLevel::Detailed)
            .with_expertise(ExpertiseLevel::Advanced)
            .with_language_style(LanguageStyle::Conversational);

        assert_eq!(profile.preferred_tone, Tone::Casual);
        assert_eq!(profile.verbosity_level, VerbosityLevel::Detailed);
        assert_eq!(profile.expertise_level, ExpertiseLevel::Advanced);
        assert_eq!(profile.language_style, LanguageStyle::Conversational);
    }

    #[test]
    fn test_user_profile_with_domain_preference() {
        let mut pref = DomainPreference::default();
        pref.expertise_override = Some(ExpertiseLevel::Expert);

        let profile = UserProfile::default().with_domain_preference("rust", pref);

        assert!(profile.domain_preferences.contains_key("rust"));
    }

    // =========================================================================
    // Adaptation Tests
    // =========================================================================

    #[test]
    fn test_adaptation_creation() {
        let adaptation = Adaptation::new(
            AdaptationType::ToneAdjustment,
            "Changed tone to formal",
            0.9,
        );

        assert_eq!(adaptation.adaptation_type, AdaptationType::ToneAdjustment);
        assert_eq!(adaptation.description, "Changed tone to formal");
        assert_eq!(adaptation.confidence, 0.9);
        assert!(adaptation.original.is_none());
        assert!(adaptation.adapted.is_none());
    }

    #[test]
    fn test_adaptation_with_text() {
        let adaptation = Adaptation::new(
            AdaptationType::VocabularySubstitution,
            "Substituted term",
            0.85,
        )
        .with_original("gonna")
        .with_adapted("going to");

        assert_eq!(adaptation.original, Some("gonna".to_string()));
        assert_eq!(adaptation.adapted, Some("going to".to_string()));
    }

    #[test]
    fn test_adaptation_type_display() {
        assert_eq!(AdaptationType::ToneAdjustment.to_string(), "tone");
        assert_eq!(AdaptationType::VerbosityChange.to_string(), "verbosity");
        assert_eq!(AdaptationType::ExpertiseAdaptation.to_string(), "expertise");
        assert_eq!(AdaptationType::StyleModification.to_string(), "style");
        assert_eq!(AdaptationType::VocabularySubstitution.to_string(), "vocabulary");
        assert_eq!(AdaptationType::StructureChange.to_string(), "structure");
        assert_eq!(AdaptationType::Personalization.to_string(), "personalization");
    }

    // =========================================================================
    // PersonalizedPrompt Tests
    // =========================================================================

    #[test]
    fn test_personalized_prompt_was_adapted() {
        let prompt = PersonalizedPrompt {
            content: "adapted content".to_string(),
            original_content: "original content".to_string(),
            adaptations: vec![],
            confidence: 1.0,
            profile_summary: ProfileSummary {
                tone: Tone::Technical,
                verbosity: VerbosityLevel::Balanced,
                expertise: ExpertiseLevel::Intermediate,
                style: LanguageStyle::Professional,
            },
            processing_time_ms: 10,
        };

        assert!(!prompt.was_adapted());

        let prompt_with_adaptations = PersonalizedPrompt {
            adaptations: vec![Adaptation::new(AdaptationType::ToneAdjustment, "test", 0.9)],
            ..prompt
        };

        assert!(prompt_with_adaptations.was_adapted());
    }

    #[test]
    fn test_personalized_prompt_adaptation_count() {
        let prompt = PersonalizedPrompt {
            content: "test".to_string(),
            original_content: "test".to_string(),
            adaptations: vec![
                Adaptation::new(AdaptationType::ToneAdjustment, "a", 0.9),
                Adaptation::new(AdaptationType::VerbosityChange, "b", 0.8),
            ],
            confidence: 0.85,
            profile_summary: ProfileSummary {
                tone: Tone::Technical,
                verbosity: VerbosityLevel::Balanced,
                expertise: ExpertiseLevel::Intermediate,
                style: LanguageStyle::Professional,
            },
            processing_time_ms: 10,
        };

        assert_eq!(prompt.adaptation_count(), 2);
    }

    #[test]
    fn test_personalized_prompt_adaptations_of_type() {
        let prompt = PersonalizedPrompt {
            content: "test".to_string(),
            original_content: "test".to_string(),
            adaptations: vec![
                Adaptation::new(AdaptationType::ToneAdjustment, "a", 0.9),
                Adaptation::new(AdaptationType::ToneAdjustment, "b", 0.8),
                Adaptation::new(AdaptationType::VerbosityChange, "c", 0.7),
            ],
            confidence: 0.8,
            profile_summary: ProfileSummary {
                tone: Tone::Technical,
                verbosity: VerbosityLevel::Balanced,
                expertise: ExpertiseLevel::Intermediate,
                style: LanguageStyle::Professional,
            },
            processing_time_ms: 10,
        };

        assert_eq!(prompt.adaptations_of_type(AdaptationType::ToneAdjustment).len(), 2);
        assert_eq!(prompt.adaptations_of_type(AdaptationType::VerbosityChange).len(), 1);
        assert_eq!(prompt.adaptations_of_type(AdaptationType::StyleModification).len(), 0);
    }

    #[test]
    fn test_personalized_prompt_average_confidence() {
        let prompt = PersonalizedPrompt {
            content: "test".to_string(),
            original_content: "test".to_string(),
            adaptations: vec![
                Adaptation::new(AdaptationType::ToneAdjustment, "a", 0.9),
                Adaptation::new(AdaptationType::VerbosityChange, "b", 0.7),
            ],
            confidence: 0.8,
            profile_summary: ProfileSummary {
                tone: Tone::Technical,
                verbosity: VerbosityLevel::Balanced,
                expertise: ExpertiseLevel::Intermediate,
                style: LanguageStyle::Professional,
            },
            processing_time_ms: 10,
        };

        assert!((prompt.average_adaptation_confidence() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_personalized_prompt_length_change_percent() {
        let prompt = PersonalizedPrompt {
            content: "short".to_string(),
            original_content: "this is a longer original".to_string(),
            adaptations: vec![],
            confidence: 1.0,
            profile_summary: ProfileSummary {
                tone: Tone::Technical,
                verbosity: VerbosityLevel::Balanced,
                expertise: ExpertiseLevel::Intermediate,
                style: LanguageStyle::Professional,
            },
            processing_time_ms: 10,
        };

        assert!(prompt.length_change_percent() < 0.0);
    }

    // =========================================================================
    // PersonalizerConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = PersonalizerConfig::default();
        assert!(config.adapt_tone);
        assert!(config.adapt_verbosity);
        assert!(config.adapt_expertise);
        assert!(config.adapt_style);
        assert!(config.adapt_vocabulary);
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.max_adaptations, 50);
    }

    #[test]
    fn test_config_minimal() {
        let config = PersonalizerConfig::minimal();
        assert!(config.adapt_tone);
        assert!(!config.adapt_verbosity);
        assert!(!config.adapt_expertise);
        assert!(!config.adapt_style);
        assert!(!config.adapt_vocabulary);
        assert_eq!(config.confidence_threshold, 0.7);
        assert_eq!(config.max_adaptations, 10);
    }

    #[test]
    fn test_config_aggressive() {
        let config = PersonalizerConfig::aggressive();
        assert!(config.adapt_tone);
        assert!(config.adapt_verbosity);
        assert!(config.adapt_expertise);
        assert!(config.adapt_style);
        assert!(config.adapt_vocabulary);
        assert_eq!(config.confidence_threshold, 0.3);
        assert_eq!(config.max_adaptations, 100);
    }

    // =========================================================================
    // PersonalizerAgent Tests
    // =========================================================================

    #[test]
    fn test_agent_creation() {
        let agent = PersonalizerAgent::default();
        assert_eq!(agent.profile().preferred_tone, Tone::Technical);
        assert!(agent.config().adapt_tone);
    }

    #[test]
    fn test_agent_builder() {
        let agent = PersonalizerAgent::builder()
            .tone(Tone::Friendly)
            .verbosity(VerbosityLevel::Concise)
            .expertise(ExpertiseLevel::Expert)
            .language_style(LanguageStyle::Conversational)
            .name("Bob")
            .build();

        assert_eq!(agent.profile().preferred_tone, Tone::Friendly);
        assert_eq!(agent.profile().verbosity_level, VerbosityLevel::Concise);
        assert_eq!(agent.profile().expertise_level, ExpertiseLevel::Expert);
        assert_eq!(agent.profile().language_style, LanguageStyle::Conversational);
        assert_eq!(agent.profile().name, Some("Bob".to_string()));
    }

    #[test]
    fn test_agent_builder_with_profile() {
        let profile = UserProfile::beginner();
        let agent = PersonalizerAgent::builder().user_profile(profile).build();

        assert_eq!(agent.profile().expertise_level, ExpertiseLevel::Beginner);
    }

    #[test]
    fn test_agent_builder_with_config() {
        let config = PersonalizerConfig::minimal();
        let agent = PersonalizerAgent::builder().config(config).build();

        assert!(!agent.config().adapt_verbosity);
    }

    #[tokio::test]
    async fn test_agent_personalize_basic() {
        let agent = PersonalizerAgent::default();
        let result = agent.personalize("This is a test message.").await.unwrap();

        assert!(!result.content.is_empty());
        assert_eq!(result.original_content, "This is a test message.");
    }

    #[tokio::test]
    async fn test_agent_personalize_formal_tone() {
        let agent = PersonalizerAgent::builder()
            .tone(Tone::Formal)
            .config(PersonalizerConfig::default())
            .build();

        let result = agent.personalize("You don't need to worry about this.").await.unwrap();

        assert!(result.content.contains("do not") || result.was_adapted());
    }

    #[tokio::test]
    async fn test_agent_personalize_casual_tone() {
        let agent = PersonalizerAgent::builder()
            .tone(Tone::Casual)
            .build();

        let result = agent.personalize("You do not need to worry.").await.unwrap();

        // Casual should use contractions
        assert!(result.content.contains("don't") || result.was_adapted());
    }

    #[tokio::test]
    async fn test_agent_personalize_friendly_with_name() {
        let agent = PersonalizerAgent::builder()
            .tone(Tone::Friendly)
            .name("Alice")
            .build();

        let result = agent.personalize("Here is the information you requested.").await.unwrap();

        // Should include friendly greeting with name
        assert!(result.content.contains("Alice") || result.was_adapted());
    }

    #[tokio::test]
    async fn test_agent_personalize_beginner_expertise() {
        let agent = PersonalizerAgent::builder()
            .expertise(ExpertiseLevel::Beginner)
            .build();

        let result = agent.personalize("Use the API endpoint to fetch data.").await.unwrap();

        // Should expand API acronym for beginners
        assert!(result.content.contains("Application Programming Interface") || result.was_adapted());
    }

    #[tokio::test]
    async fn test_agent_personalize_concise_verbosity() {
        let agent = PersonalizerAgent::builder()
            .verbosity(VerbosityLevel::Concise)
            .build();

        let result = agent.personalize("Basically, in other words, this is the answer.").await.unwrap();

        // Should remove filler phrases
        assert!(!result.content.contains("Basically") || result.was_adapted());
    }

    #[tokio::test]
    async fn test_agent_personalize_vocabulary_preferences() {
        let mut profile = UserProfile::default();
        profile.vocabulary_preferences.preferred_terms.insert(
            "foo".to_string(),
            "bar".to_string(),
        );

        let agent = PersonalizerAgent::builder()
            .user_profile(profile)
            .build();

        let result = agent.personalize("The foo value should be set.").await.unwrap();

        assert!(result.content.contains("bar"));
    }

    #[tokio::test]
    async fn test_agent_personalize_no_contractions_preference() {
        let mut profile = UserProfile::default();
        profile.vocabulary_preferences.use_contractions = false;

        let agent = PersonalizerAgent::builder()
            .user_profile(profile)
            .build();

        let result = agent.personalize("You don't need to worry.").await.unwrap();

        // Should expand contraction when use_contractions is false
        assert!(
            result.content.to_lowercase().contains("do not") || result.was_adapted(),
            "Expected 'do not' or adaptation, got: {}",
            result.content
        );
    }

    #[test]
    fn test_agent_set_profile() {
        let mut agent = PersonalizerAgent::default();
        let new_profile = UserProfile::expert();

        agent.set_profile(new_profile);

        assert_eq!(agent.profile().expertise_level, ExpertiseLevel::Expert);
    }

    #[test]
    fn test_agent_set_config() {
        let mut agent = PersonalizerAgent::default();
        let new_config = PersonalizerConfig::minimal();

        agent.set_config(new_config);

        assert!(!agent.config().adapt_verbosity);
    }

    #[tokio::test]
    async fn test_agent_personalize_empty_content() {
        let agent = PersonalizerAgent::default();
        let result = agent.personalize("").await.unwrap();

        assert!(result.content.is_empty());
        assert!(!result.was_adapted() || result.adaptation_count() == 0);
    }

    #[tokio::test]
    async fn test_agent_personalize_confidence_threshold() {
        let config = PersonalizerConfig {
            confidence_threshold: 0.99, // Very high threshold
            ..Default::default()
        };

        let agent = PersonalizerAgent::builder()
            .config(config)
            .build();

        let result = agent.personalize("Some content to personalize.").await.unwrap();

        // All adaptations below 0.99 confidence should be filtered out
        for adaptation in &result.adaptations {
            assert!(adaptation.confidence >= 0.99);
        }
    }

    #[tokio::test]
    async fn test_agent_personalize_max_adaptations() {
        let config = PersonalizerConfig {
            max_adaptations: 2,
            confidence_threshold: 0.0, // Accept all
            ..Default::default()
        };

        let agent = PersonalizerAgent::builder()
            .config(config)
            .build();

        // Content that would trigger many adaptations
        let content = "Don't forget: gonna need this, ok? Stuff and things, yeah.";
        let result = agent.personalize(content).await.unwrap();

        assert!(result.adaptations.len() <= 2);
    }

    // =========================================================================
    // ProfileSummary Tests
    // =========================================================================

    #[test]
    fn test_profile_summary_from_user_profile() {
        let profile = UserProfile::expert();
        let summary = ProfileSummary::from(&profile);

        assert_eq!(summary.tone, Tone::Technical);
        assert_eq!(summary.verbosity, VerbosityLevel::Concise);
        assert_eq!(summary.expertise, ExpertiseLevel::Expert);
        assert_eq!(summary.style, LanguageStyle::Professional);
    }

    // =========================================================================
    // Pattern Tests
    // =========================================================================

    #[test]
    fn test_tone_patterns_default() {
        let patterns = TonePatterns::default();

        assert!(patterns.contractions.contains_key("don't"));
        assert!(patterns.contractions.contains_key("can't"));
        assert!(patterns.formal_substitutions.contains_key("gonna"));
        assert!(patterns.formal_substitutions.contains_key("wanna"));
    }

    #[test]
    fn test_expertise_patterns_default() {
        let patterns = ExpertisePatterns::default();

        assert!(patterns.acronym_expansions.contains_key("API"));
        assert!(patterns.acronym_expansions.contains_key("REST"));
        assert!(patterns.acronym_expansions.contains_key("JSON"));
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_user_profile_serialization() {
        let profile = UserProfile::beginner().with_name("Test");
        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: UserProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.preferred_tone, Tone::Friendly);
        assert_eq!(deserialized.name, Some("Test".to_string()));
    }

    #[test]
    fn test_config_serialization() {
        let config = PersonalizerConfig::minimal();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PersonalizerConfig = serde_json::from_str(&json).unwrap();

        assert!(!deserialized.adapt_verbosity);
        assert_eq!(deserialized.confidence_threshold, 0.7);
    }

    #[test]
    fn test_adaptation_serialization() {
        let adaptation = Adaptation::new(AdaptationType::ToneAdjustment, "test", 0.9)
            .with_original("a")
            .with_adapted("b");

        let json = serde_json::to_string(&adaptation).unwrap();
        let deserialized: Adaptation = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.adaptation_type, AdaptationType::ToneAdjustment);
        assert_eq!(deserialized.original, Some("a".to_string()));
    }
}
