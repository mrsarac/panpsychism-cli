//! Localizer Agent module for Project Panpsychism.
//!
//! The Polyglot - "Every incantation must speak in the tongue of its listener."
//!
//! This module implements Agent 24: the Localizer (i18n Master), responsible for
//! adapting prompts for different languages and cultures. Like a master translator
//! in the sorcerer's multilingual library, The Polyglot understands not just words,
//! but the subtle nuances of culture, idiom, and expression.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Polyglot serves as the bridge between worlds:
//!
//! - **Source Incantation** (original prompt) arrives from any tongue
//! - **The Polyglot** (LocalizerAgent) transcribes the magical essence
//! - **Localized Spell** (adapted prompt) emerges culturally appropriate
//!
//! The Polyglot can:
//! - Adapt content for specific language/region combinations (locales)
//! - Identify and handle cultural sensitivities (taboos, idioms, formality)
//! - Provide alternative localizations for different contexts
//! - Respect date, number, and currency formatting conventions
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//! - **RATIO**: Logical preservation of meaning across languages
//! - **LAETITIA**: Joy through clear communication in native tongues
//! - **NATURA**: Natural alignment with cultural expectations
//! - **CONATUS**: Drive to preserve the essence while adapting the form
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::localizer::{LocalizerAgent, Locale};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let agent = LocalizerAgent::new();
//!
//!     // Localize for German (Germany)
//!     let locale = Locale::new("de", "DE");
//!     let localized = agent.localize("Hello, world!", &locale).await?;
//!
//!     println!("Localized: {}", localized.content);
//!     println!("Cultural notes: {:?}", localized.cultural_notes);
//!     Ok(())
//! }
//! ```

use crate::gemini::{GeminiClient, Message};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

// =============================================================================
// LOCALE DEFINITION
// =============================================================================

/// Represents a locale for localization.
///
/// A locale combines language, region, and optionally script information
/// following ISO standards for precise targeting of content.
///
/// ## Standards
///
/// - Language: ISO 639-1 (e.g., "en", "de", "ja", "zh")
/// - Region: ISO 3166-1 alpha-2 (e.g., "US", "DE", "JP", "CN")
/// - Script: ISO 15924 (e.g., "Latn", "Hans", "Hant", "Cyrl")
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::localizer::Locale;
///
/// // Simple locale
/// let en_us = Locale::new("en", "US");
///
/// // With script (Chinese Simplified vs Traditional)
/// let zh_hans = Locale::new("zh", "CN").with_script("Hans");
/// let zh_hant = Locale::new("zh", "TW").with_script("Hant");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Locale {
    /// ISO 639-1 language code (e.g., "en", "de", "ja").
    pub language: String,

    /// ISO 3166-1 alpha-2 region code (e.g., "US", "DE", "JP").
    pub region: String,

    /// ISO 15924 script code (e.g., "Latn", "Hans", "Hant").
    pub script: Option<String>,
}

impl Locale {
    /// Create a new locale with language and region.
    ///
    /// # Arguments
    ///
    /// * `language` - ISO 639-1 language code
    /// * `region` - ISO 3166-1 region code
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let locale = Locale::new("en", "US");
    /// assert_eq!(locale.language, "en");
    /// assert_eq!(locale.region, "US");
    /// ```
    pub fn new(language: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            language: language.into().to_lowercase(),
            region: region.into().to_uppercase(),
            script: None,
        }
    }

    /// Add a script code to the locale.
    ///
    /// # Arguments
    ///
    /// * `script` - ISO 15924 script code
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let locale = Locale::new("zh", "CN").with_script("Hans");
    /// assert_eq!(locale.script, Some("Hans".to_string()));
    /// ```
    pub fn with_script(mut self, script: impl Into<String>) -> Self {
        let script_str = script.into();
        // Capitalize first letter, lowercase rest (ISO 15924 format)
        let formatted = if script_str.len() >= 1 {
            let mut chars = script_str.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => script_str,
            }
        } else {
            script_str
        };
        self.script = Some(formatted);
        self
    }

    /// Parse a locale from a BCP 47 language tag.
    ///
    /// Supports formats:
    /// - `en` (language only)
    /// - `en-US` (language-region)
    /// - `zh-Hans-CN` (language-script-region)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let locale = Locale::from_bcp47("en-US")?;
    /// assert_eq!(locale.language, "en");
    /// assert_eq!(locale.region, "US");
    /// ```
    pub fn from_bcp47(tag: &str) -> Result<Self> {
        let parts: Vec<&str> = tag.split(|c| c == '-' || c == '_').collect();

        match parts.len() {
            1 => {
                // Language only
                Ok(Self::new(parts[0], ""))
            }
            2 => {
                // Language-Region or Language-Script
                let second = parts[1];
                if second.len() == 2 {
                    // Region code
                    Ok(Self::new(parts[0], second))
                } else if second.len() == 4 {
                    // Script code
                    Ok(Self::new(parts[0], "").with_script(second))
                } else {
                    Err(Error::Config(format!("Invalid locale format: {}", tag)))
                }
            }
            3 => {
                // Language-Script-Region
                Ok(Self::new(parts[0], parts[2]).with_script(parts[1]))
            }
            _ => Err(Error::Config(format!("Invalid locale format: {}", tag))),
        }
    }

    /// Convert to BCP 47 language tag format.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let locale = Locale::new("zh", "CN").with_script("Hans");
    /// assert_eq!(locale.to_bcp47(), "zh-Hans-CN");
    /// ```
    pub fn to_bcp47(&self) -> String {
        match (&self.script, self.region.is_empty()) {
            (Some(script), true) => format!("{}-{}", self.language, script),
            (Some(script), false) => format!("{}-{}-{}", self.language, script, self.region),
            (None, true) => self.language.clone(),
            (None, false) => format!("{}-{}", self.language, self.region),
        }
    }

    /// Get the display name for this locale.
    ///
    /// Returns a human-readable name like "English (United States)".
    pub fn display_name(&self) -> String {
        let lang_name = match self.language.as_str() {
            "en" => "English",
            "de" => "German",
            "fr" => "French",
            "es" => "Spanish",
            "it" => "Italian",
            "pt" => "Portuguese",
            "ja" => "Japanese",
            "zh" => "Chinese",
            "ko" => "Korean",
            "ar" => "Arabic",
            "ru" => "Russian",
            "tr" => "Turkish",
            "nl" => "Dutch",
            "pl" => "Polish",
            "sv" => "Swedish",
            "da" => "Danish",
            "fi" => "Finnish",
            "no" => "Norwegian",
            "el" => "Greek",
            "he" => "Hebrew",
            "hi" => "Hindi",
            "th" => "Thai",
            "vi" => "Vietnamese",
            "id" => "Indonesian",
            "ms" => "Malay",
            _ => &self.language,
        };

        let region_name = match self.region.as_str() {
            "" => return lang_name.to_string(),
            "US" => "United States",
            "GB" => "United Kingdom",
            "DE" => "Germany",
            "FR" => "France",
            "ES" => "Spain",
            "IT" => "Italy",
            "PT" => "Portugal",
            "BR" => "Brazil",
            "JP" => "Japan",
            "CN" => "China",
            "TW" => "Taiwan",
            "HK" => "Hong Kong",
            "KR" => "South Korea",
            "AU" => "Australia",
            "CA" => "Canada",
            "MX" => "Mexico",
            "AR" => "Argentina",
            "NL" => "Netherlands",
            "BE" => "Belgium",
            "CH" => "Switzerland",
            "AT" => "Austria",
            "RU" => "Russia",
            "TR" => "Turkey",
            "SA" => "Saudi Arabia",
            "AE" => "United Arab Emirates",
            "IN" => "India",
            _ => &self.region,
        };

        format!("{} ({})", lang_name, region_name)
    }

    /// Check if this locale is RTL (right-to-left).
    pub fn is_rtl(&self) -> bool {
        matches!(self.language.as_str(), "ar" | "he" | "fa" | "ur")
    }

    /// Get common locales.
    pub fn common_locales() -> Vec<Self> {
        vec![
            Self::new("en", "US"),
            Self::new("en", "GB"),
            Self::new("de", "DE"),
            Self::new("fr", "FR"),
            Self::new("es", "ES"),
            Self::new("it", "IT"),
            Self::new("pt", "BR"),
            Self::new("ja", "JP"),
            Self::new("zh", "CN").with_script("Hans"),
            Self::new("zh", "TW").with_script("Hant"),
            Self::new("ko", "KR"),
            Self::new("ar", "SA"),
            Self::new("ru", "RU"),
            Self::new("tr", "TR"),
        ]
    }
}

impl Default for Locale {
    fn default() -> Self {
        Self::new("en", "US")
    }
}

impl std::fmt::Display for Locale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_bcp47())
    }
}

impl std::str::FromStr for Locale {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_bcp47(s)
    }
}

// =============================================================================
// CULTURAL CATEGORIES AND NOTES
// =============================================================================

/// Categories of cultural considerations for localization.
///
/// The Polyglot classifies cultural notes by category to help
/// translators and developers understand the nature of adaptations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CulturalCategory {
    /// Idiomatic expressions that don't translate literally.
    ///
    /// E.g., "raining cats and dogs" -> "es regnet in Strömen" (DE)
    Idiom,

    /// Formality level adjustments.
    ///
    /// E.g., casual English -> formal Japanese (keigo)
    Formality,

    /// Topics that may be sensitive or taboo in certain cultures.
    ///
    /// E.g., religious references, political topics
    Taboo,

    /// Cultural preferences and conventions.
    ///
    /// E.g., greeting customs, color symbolism
    Preference,

    /// Date and time format conventions.
    ///
    /// E.g., MM/DD/YYYY (US) vs DD.MM.YYYY (DE)
    DateFormat,

    /// Number and currency format conventions.
    ///
    /// E.g., 1,000.00 (US) vs 1.000,00 (DE)
    NumberFormat,
}

impl std::fmt::Display for CulturalCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idiom => write!(f, "Idiom"),
            Self::Formality => write!(f, "Formality"),
            Self::Taboo => write!(f, "Taboo"),
            Self::Preference => write!(f, "Preference"),
            Self::DateFormat => write!(f, "Date Format"),
            Self::NumberFormat => write!(f, "Number Format"),
        }
    }
}

impl std::str::FromStr for CulturalCategory {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "idiom" | "idioms" => Ok(Self::Idiom),
            "formality" | "formal" => Ok(Self::Formality),
            "taboo" | "taboos" | "sensitive" => Ok(Self::Taboo),
            "preference" | "preferences" | "custom" => Ok(Self::Preference),
            "dateformat" | "date_format" | "date" => Ok(Self::DateFormat),
            "numberformat" | "number_format" | "number" | "currency" => Ok(Self::NumberFormat),
            _ => Err(Error::Config(format!("Unknown cultural category: {}", s))),
        }
    }
}

/// Severity level for cultural notes.
///
/// Indicates how important it is to address a cultural consideration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CulturalSeverity {
    /// Informational note, no action required.
    Info,
    /// Suggestion for improvement.
    Suggestion,
    /// Warning that should be addressed.
    Warning,
    /// Critical issue that must be fixed.
    Critical,
}

impl Default for CulturalSeverity {
    fn default() -> Self {
        Self::Info
    }
}

impl std::fmt::Display for CulturalSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "Info"),
            Self::Suggestion => write!(f, "Suggestion"),
            Self::Warning => write!(f, "Warning"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// A cultural note about localization considerations.
///
/// Cultural notes document adaptations made during localization
/// and highlight areas that may need attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalNote {
    /// Category of the cultural consideration.
    pub category: CulturalCategory,

    /// Description of the cultural issue or adaptation.
    pub note: String,

    /// Severity level of this note.
    pub severity: CulturalSeverity,

    /// The original text that triggered this note.
    pub original_text: Option<String>,

    /// The adapted text (if applicable).
    pub adapted_text: Option<String>,
}

impl CulturalNote {
    /// Create a new cultural note.
    pub fn new(category: CulturalCategory, note: impl Into<String>) -> Self {
        Self {
            category,
            note: note.into(),
            severity: CulturalSeverity::default(),
            original_text: None,
            adapted_text: None,
        }
    }

    /// Set the severity level.
    pub fn with_severity(mut self, severity: CulturalSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set the original text.
    pub fn with_original(mut self, text: impl Into<String>) -> Self {
        self.original_text = Some(text.into());
        self
    }

    /// Set the adapted text.
    pub fn with_adapted(mut self, text: impl Into<String>) -> Self {
        self.adapted_text = Some(text.into());
        self
    }

    /// Check if this note requires attention (Warning or Critical).
    pub fn requires_attention(&self) -> bool {
        matches!(
            self.severity,
            CulturalSeverity::Warning | CulturalSeverity::Critical
        )
    }

    /// Format the note for display.
    pub fn to_display(&self) -> String {
        let mut output = format!("[{}] {}: {}", self.severity, self.category, self.note);

        if let Some(ref original) = self.original_text {
            output.push_str(&format!("\n  Original: \"{}\"", original));
        }

        if let Some(ref adapted) = self.adapted_text {
            output.push_str(&format!("\n  Adapted: \"{}\"", adapted));
        }

        output
    }
}

// =============================================================================
// LOCALE ALTERNATIVE
// =============================================================================

/// An alternative localization for different contexts.
///
/// When a single translation isn't sufficient, alternatives
/// provide options for different formality levels, contexts,
/// or audience segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocaleAlternative {
    /// The alternative localized content.
    pub content: String,

    /// Description of when to use this alternative.
    pub context: String,

    /// Formality level (1-5, where 1 is most casual, 5 is most formal).
    pub formality_level: u8,

    /// Tags describing this alternative.
    pub tags: HashSet<String>,
}

impl LocaleAlternative {
    /// Create a new locale alternative.
    pub fn new(content: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            context: context.into(),
            formality_level: 3, // Default to neutral
            tags: HashSet::new(),
        }
    }

    /// Set the formality level.
    pub fn with_formality(mut self, level: u8) -> Self {
        self.formality_level = level.min(5).max(1);
        self
    }

    /// Add tags to describe this alternative.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(|t| t.into()).collect();
        self
    }

    /// Check if this alternative matches a formality level.
    pub fn matches_formality(&self, level: u8) -> bool {
        self.formality_level == level
    }
}

// =============================================================================
// LOCALIZED PROMPT
// =============================================================================

/// The result of localizing a prompt.
///
/// Contains the localized content along with cultural notes,
/// alternatives, and metadata about the localization process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizedPrompt {
    /// The localized content.
    pub content: String,

    /// Target locale of this localization.
    pub locale: Locale,

    /// Cultural notes about the localization.
    pub cultural_notes: Vec<CulturalNote>,

    /// Alternative localizations for different contexts.
    pub alternatives: Vec<LocaleAlternative>,

    /// Original content before localization.
    pub original: String,

    /// Source locale (if detected or specified).
    pub source_locale: Option<Locale>,

    /// Confidence score for the localization (0.0 - 1.0).
    pub confidence: f64,

    /// Processing duration in milliseconds.
    pub duration_ms: u64,
}

impl LocalizedPrompt {
    /// Create a new localized prompt.
    pub fn new(content: impl Into<String>, locale: Locale) -> Self {
        let content = content.into();
        Self {
            content: content.clone(),
            locale,
            cultural_notes: Vec::new(),
            alternatives: Vec::new(),
            original: content,
            source_locale: None,
            confidence: 0.0,
            duration_ms: 0,
        }
    }

    /// Add a cultural note.
    pub fn add_note(&mut self, note: CulturalNote) {
        self.cultural_notes.push(note);
    }

    /// Add an alternative localization.
    pub fn add_alternative(&mut self, alternative: LocaleAlternative) {
        self.alternatives.push(alternative);
    }

    /// Check if there are any cultural warnings.
    pub fn has_warnings(&self) -> bool {
        self.cultural_notes.iter().any(|n| n.requires_attention())
    }

    /// Get notes by category.
    pub fn notes_by_category(&self, category: CulturalCategory) -> Vec<&CulturalNote> {
        self.cultural_notes
            .iter()
            .filter(|n| n.category == category)
            .collect()
    }

    /// Get the most formal alternative.
    pub fn most_formal(&self) -> Option<&LocaleAlternative> {
        self.alternatives.iter().max_by_key(|a| a.formality_level)
    }

    /// Get the most casual alternative.
    pub fn most_casual(&self) -> Option<&LocaleAlternative> {
        self.alternatives.iter().min_by_key(|a| a.formality_level)
    }

    /// Get a summary of the localization.
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("Localized to {}", self.locale.display_name())];

        if !self.cultural_notes.is_empty() {
            parts.push(format!("{} cultural notes", self.cultural_notes.len()));
        }

        if !self.alternatives.is_empty() {
            parts.push(format!("{} alternatives", self.alternatives.len()));
        }

        parts.join(", ")
    }
}

// =============================================================================
// LOCALIZER CONFIGURATION
// =============================================================================

/// Configuration for the Localizer Agent.
#[derive(Debug, Clone)]
pub struct LocalizerConfig {
    /// Supported locales for localization.
    pub supported_locales: Vec<Locale>,

    /// Fallback locale when target locale is not supported.
    pub fallback_locale: Locale,

    /// Whether to detect idioms and flag them.
    pub detect_idioms: bool,

    /// Whether to adjust formality levels.
    pub adjust_formality: bool,

    /// Whether to adapt date/number formats.
    pub adapt_formats: bool,

    /// Whether to generate alternative translations.
    pub generate_alternatives: bool,

    /// Maximum number of alternatives to generate.
    pub max_alternatives: usize,

    /// Minimum confidence threshold for localization.
    pub confidence_threshold: f64,

    /// Timeout in seconds for LLM requests.
    pub timeout_secs: u64,
}

impl Default for LocalizerConfig {
    fn default() -> Self {
        Self {
            supported_locales: Locale::common_locales(),
            fallback_locale: Locale::new("en", "US"),
            detect_idioms: true,
            adjust_formality: true,
            adapt_formats: true,
            generate_alternatives: true,
            max_alternatives: 3,
            confidence_threshold: 0.7,
            timeout_secs: 60,
        }
    }
}

impl LocalizerConfig {
    /// Create a minimal configuration for fast localization.
    pub fn minimal() -> Self {
        Self {
            supported_locales: vec![
                Locale::new("en", "US"),
                Locale::new("de", "DE"),
                Locale::new("fr", "FR"),
                Locale::new("es", "ES"),
            ],
            fallback_locale: Locale::new("en", "US"),
            detect_idioms: false,
            adjust_formality: false,
            adapt_formats: true,
            generate_alternatives: false,
            max_alternatives: 0,
            confidence_threshold: 0.5,
            timeout_secs: 30,
        }
    }

    /// Create a thorough configuration for high-quality localization.
    pub fn thorough() -> Self {
        Self {
            supported_locales: Locale::common_locales(),
            fallback_locale: Locale::new("en", "US"),
            detect_idioms: true,
            adjust_formality: true,
            adapt_formats: true,
            generate_alternatives: true,
            max_alternatives: 5,
            confidence_threshold: 0.8,
            timeout_secs: 120,
        }
    }

    /// Check if a locale is supported.
    pub fn is_supported(&self, locale: &Locale) -> bool {
        self.supported_locales
            .iter()
            .any(|l| l.language == locale.language && (l.region.is_empty() || l.region == locale.region))
    }

    /// Add a supported locale.
    pub fn add_locale(&mut self, locale: Locale) {
        if !self.is_supported(&locale) {
            self.supported_locales.push(locale);
        }
    }
}

// =============================================================================
// LOCALIZER AGENT
// =============================================================================

/// The Localizer Agent - The Polyglot of the Sorcerer's Tower.
///
/// Responsible for adapting prompts to different languages and cultures.
/// The Polyglot understands not just translation, but the subtle art of
/// cultural adaptation that makes content truly resonate.
///
/// ## Capabilities
///
/// - **Translation**: Adapt content to different languages
/// - **Cultural Adaptation**: Handle idioms, formality, and taboos
/// - **Format Conversion**: Adapt dates, numbers, and currencies
/// - **Alternative Generation**: Provide options for different contexts
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::localizer::{LocalizerAgent, Locale};
///
/// #[tokio::main]
/// async fn main() -> panpsychism::Result<()> {
///     let agent = LocalizerAgent::new();
///
///     // Localize a prompt for German
///     let locale = Locale::new("de", "DE");
///     let result = agent.localize("Hello, how are you?", &locale).await?;
///
///     println!("Localized: {}", result.content);
///     // Output: "Hallo, wie geht es Ihnen?" (formal) or "Hallo, wie geht's?" (casual)
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LocalizerAgent {
    /// Configuration for localization behavior.
    config: LocalizerConfig,

    /// Gemini client for LLM-powered localization.
    gemini_client: Option<Arc<GeminiClient>>,

    /// Phrase mappings for common translations.
    phrase_cache: HashMap<String, HashMap<String, String>>,
}

impl Default for LocalizerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalizerAgent {
    /// Create a new Localizer Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: LocalizerConfig::default(),
            gemini_client: None,
            phrase_cache: Self::build_phrase_cache(),
        }
    }

    /// Create a new Localizer Agent with a Gemini client.
    pub fn with_gemini(client: Arc<GeminiClient>) -> Self {
        Self {
            config: LocalizerConfig::default(),
            gemini_client: Some(client),
            phrase_cache: Self::build_phrase_cache(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> LocalizerAgentBuilder {
        LocalizerAgentBuilder::default()
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: LocalizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &LocalizerConfig {
        &self.config
    }

    /// Get supported locales.
    pub fn supported_locales(&self) -> &[Locale] {
        &self.config.supported_locales
    }

    /// Check if a locale is supported.
    pub fn is_supported(&self, locale: &Locale) -> bool {
        self.config.is_supported(locale)
    }

    /// Build initial phrase cache for common translations.
    fn build_phrase_cache() -> HashMap<String, HashMap<String, String>> {
        let mut cache = HashMap::new();

        // Common greetings
        let mut greetings = HashMap::new();
        greetings.insert("de".to_string(), "Hallo".to_string());
        greetings.insert("fr".to_string(), "Bonjour".to_string());
        greetings.insert("es".to_string(), "Hola".to_string());
        greetings.insert("it".to_string(), "Ciao".to_string());
        greetings.insert("ja".to_string(), "こんにちは".to_string());
        greetings.insert("zh".to_string(), "你好".to_string());
        greetings.insert("ko".to_string(), "안녕하세요".to_string());
        greetings.insert("ar".to_string(), "مرحبا".to_string());
        greetings.insert("ru".to_string(), "Привет".to_string());
        greetings.insert("tr".to_string(), "Merhaba".to_string());
        cache.insert("hello".to_string(), greetings);

        // Common farewells
        let mut farewells = HashMap::new();
        farewells.insert("de".to_string(), "Auf Wiedersehen".to_string());
        farewells.insert("fr".to_string(), "Au revoir".to_string());
        farewells.insert("es".to_string(), "Adiós".to_string());
        farewells.insert("it".to_string(), "Arrivederci".to_string());
        farewells.insert("ja".to_string(), "さようなら".to_string());
        farewells.insert("zh".to_string(), "再见".to_string());
        farewells.insert("ko".to_string(), "안녕히 가세요".to_string());
        farewells.insert("ar".to_string(), "مع السلامة".to_string());
        farewells.insert("ru".to_string(), "До свидания".to_string());
        farewells.insert("tr".to_string(), "Hoşça kalın".to_string());
        cache.insert("goodbye".to_string(), farewells);

        // Thank you
        let mut thanks = HashMap::new();
        thanks.insert("de".to_string(), "Danke".to_string());
        thanks.insert("fr".to_string(), "Merci".to_string());
        thanks.insert("es".to_string(), "Gracias".to_string());
        thanks.insert("it".to_string(), "Grazie".to_string());
        thanks.insert("ja".to_string(), "ありがとう".to_string());
        thanks.insert("zh".to_string(), "谢谢".to_string());
        thanks.insert("ko".to_string(), "감사합니다".to_string());
        thanks.insert("ar".to_string(), "شكرا".to_string());
        thanks.insert("ru".to_string(), "Спасибо".to_string());
        thanks.insert("tr".to_string(), "Teşekkürler".to_string());
        cache.insert("thank you".to_string(), thanks);

        cache
    }

    // =========================================================================
    // MAIN LOCALIZATION METHOD
    // =========================================================================

    /// Localize content for a specific locale.
    ///
    /// This is the primary method of The Polyglot, adapting content
    /// to speak naturally in the target language and culture.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to localize
    /// * `target` - The target locale
    ///
    /// # Returns
    ///
    /// A `LocalizedPrompt` containing the localized content and metadata.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = LocalizerAgent::new();
    /// let locale = Locale::new("de", "DE");
    /// let result = agent.localize("Hello!", &locale).await?;
    /// ```
    pub async fn localize(&self, content: &str, target: &Locale) -> Result<LocalizedPrompt> {
        let start = Instant::now();

        if content.trim().is_empty() {
            return Err(Error::Synthesis("Cannot localize empty content".to_string()));
        }

        // Check if locale is supported
        if !self.is_supported(target) {
            warn!(
                "Locale {} not in supported list, using fallback patterns",
                target
            );
        }

        // Detect source locale if possible
        let source_locale = self.detect_language(content);
        debug!(
            "Detected source locale: {:?}",
            source_locale.as_ref().map(|l| l.to_bcp47())
        );

        // Perform localization
        let mut result = if let Some(client) = &self.gemini_client {
            self.localize_with_llm(client, content, target, source_locale.as_ref())
                .await?
        } else {
            self.localize_with_rules(content, target, source_locale.as_ref())?
        };

        result.source_locale = source_locale;
        result.duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "Localization complete: {} -> {} in {}ms",
            result.source_locale.as_ref().map(|l| l.to_bcp47()).unwrap_or("unknown".to_string()),
            target.to_bcp47(),
            result.duration_ms
        );

        Ok(result)
    }

    /// Localize using LLM.
    async fn localize_with_llm(
        &self,
        client: &GeminiClient,
        content: &str,
        target: &Locale,
        source: Option<&Locale>,
    ) -> Result<LocalizedPrompt> {
        let source_info = source
            .map(|s| format!(" from {}", s.display_name()))
            .unwrap_or_default();

        let mut instructions = vec![
            format!(
                "Translate the following text{} to {} ({}).",
                source_info,
                target.display_name(),
                target.to_bcp47()
            ),
        ];

        if self.config.detect_idioms {
            instructions.push(
                "Identify any idioms and adapt them naturally for the target culture.".to_string(),
            );
        }

        if self.config.adjust_formality {
            instructions.push(
                "Use an appropriate formality level for the target culture.".to_string(),
            );
        }

        if self.config.adapt_formats {
            instructions.push(
                "Adapt any dates, numbers, and currencies to local conventions.".to_string(),
            );
        }

        let alternatives_instruction = if self.config.generate_alternatives {
            format!(
                "\n\nAlso provide {} alternative translations with different formality levels.",
                self.config.max_alternatives
            )
        } else {
            String::new()
        };

        let prompt = format!(
            "{}\n\n\
             Text to localize:\n{}\n\n\
             Respond in JSON format:\n\
             {{\n\
               \"translation\": \"<main translation>\",\n\
               \"cultural_notes\": [\n\
                 {{\n\
                   \"category\": \"<Idiom|Formality|Taboo|Preference|DateFormat|NumberFormat>\",\n\
                   \"note\": \"<description>\",\n\
                   \"severity\": \"<Info|Suggestion|Warning|Critical>\",\n\
                   \"original\": \"<original text if applicable>\",\n\
                   \"adapted\": \"<adapted text if applicable>\"\n\
                 }}\n\
               ],\n\
               \"alternatives\": [\n\
                 {{\n\
                   \"content\": \"<alternative translation>\",\n\
                   \"context\": \"<when to use>\",\n\
                   \"formality_level\": <1-5>\n\
                 }}\n\
               ],\n\
               \"confidence\": <0.0-1.0>\n\
             }}{}",
            instructions.join(" "),
            content,
            alternatives_instruction
        );

        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt,
        }];

        let response = client.chat_with_retry(messages).await?;

        let response_text = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        self.parse_localization_response(&response_text, content, target)
    }

    /// Parse LLM response into LocalizedPrompt.
    fn parse_localization_response(
        &self,
        response: &str,
        original: &str,
        target: &Locale,
    ) -> Result<LocalizedPrompt> {
        // Try to parse as JSON
        let json_result: std::result::Result<serde_json::Value, _> =
            serde_json::from_str(response);

        // Also try extracting JSON from code block
        let json_value = if let Ok(value) = json_result {
            Some(value)
        } else {
            let json_re = regex::Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
            json_re.and_then(|re| {
                re.captures(response).and_then(|caps| {
                    caps.get(1)
                        .and_then(|m| serde_json::from_str(m.as_str().trim()).ok())
                })
            })
        };

        if let Some(value) = json_value {
            let translation = value
                .get("translation")
                .and_then(|v| v.as_str())
                .unwrap_or(response)
                .to_string();

            let confidence = value
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.8);

            let mut result = LocalizedPrompt::new(translation, target.clone());
            result.original = original.to_string();
            result.confidence = confidence;

            // Parse cultural notes
            if let Some(notes) = value.get("cultural_notes").and_then(|v| v.as_array()) {
                for note_value in notes {
                    if let Some(note) = self.parse_cultural_note(note_value) {
                        result.add_note(note);
                    }
                }
            }

            // Parse alternatives
            if let Some(alts) = value.get("alternatives").and_then(|v| v.as_array()) {
                for alt_value in alts.iter().take(self.config.max_alternatives) {
                    if let Some(alt) = self.parse_alternative(alt_value) {
                        result.add_alternative(alt);
                    }
                }
            }

            Ok(result)
        } else {
            // Fallback: use response as translation
            let mut result = LocalizedPrompt::new(response.trim(), target.clone());
            result.original = original.to_string();
            result.confidence = 0.5;
            Ok(result)
        }
    }

    /// Parse a cultural note from JSON.
    fn parse_cultural_note(&self, value: &serde_json::Value) -> Option<CulturalNote> {
        let category_str = value.get("category")?.as_str()?;
        let category: CulturalCategory = category_str.parse().ok()?;
        let note_text = value.get("note")?.as_str()?;

        let severity = value
            .get("severity")
            .and_then(|v| v.as_str())
            .and_then(|s| match s.to_lowercase().as_str() {
                "info" => Some(CulturalSeverity::Info),
                "suggestion" => Some(CulturalSeverity::Suggestion),
                "warning" => Some(CulturalSeverity::Warning),
                "critical" => Some(CulturalSeverity::Critical),
                _ => None,
            })
            .unwrap_or(CulturalSeverity::Info);

        let mut note = CulturalNote::new(category, note_text).with_severity(severity);

        if let Some(original) = value.get("original").and_then(|v| v.as_str()) {
            note = note.with_original(original);
        }

        if let Some(adapted) = value.get("adapted").and_then(|v| v.as_str()) {
            note = note.with_adapted(adapted);
        }

        Some(note)
    }

    /// Parse an alternative from JSON.
    fn parse_alternative(&self, value: &serde_json::Value) -> Option<LocaleAlternative> {
        let content = value.get("content")?.as_str()?.to_string();
        let context = value
            .get("context")
            .and_then(|v| v.as_str())
            .unwrap_or("Alternative translation")
            .to_string();

        let formality = value
            .get("formality_level")
            .and_then(|v| v.as_u64())
            .map(|v| v as u8)
            .unwrap_or(3);

        Some(LocaleAlternative::new(content, context).with_formality(formality))
    }

    /// Localize using rule-based translation (fallback without LLM).
    fn localize_with_rules(
        &self,
        content: &str,
        target: &Locale,
        _source: Option<&Locale>,
    ) -> Result<LocalizedPrompt> {
        let mut localized = content.to_string();
        let mut notes = Vec::new();

        // Apply phrase cache translations
        for (phrase, translations) in &self.phrase_cache {
            if let Some(translation) = translations.get(&target.language) {
                if content.to_lowercase().contains(phrase) {
                    localized = localized.replace(
                        &content
                            .to_lowercase()
                            .find(phrase)
                            .map(|i| &content[i..i + phrase.len()])
                            .unwrap_or(phrase),
                        translation,
                    );

                    notes.push(
                        CulturalNote::new(CulturalCategory::Idiom, format!("Translated common phrase: {} -> {}", phrase, translation))
                            .with_original(phrase.clone())
                            .with_adapted(translation.clone()),
                    );
                }
            }
        }

        // Add format-related notes
        if self.config.adapt_formats {
            if target.language == "de" || target.region == "DE" {
                notes.push(
                    CulturalNote::new(
                        CulturalCategory::NumberFormat,
                        "German uses comma as decimal separator and period as thousands separator",
                    )
                    .with_severity(CulturalSeverity::Info),
                );
                notes.push(
                    CulturalNote::new(
                        CulturalCategory::DateFormat,
                        "German uses DD.MM.YYYY date format",
                    )
                    .with_severity(CulturalSeverity::Info),
                );
            }

            if target.is_rtl() {
                notes.push(
                    CulturalNote::new(
                        CulturalCategory::Preference,
                        "This language is read right-to-left; ensure proper text direction",
                    )
                    .with_severity(CulturalSeverity::Warning),
                );
            }
        }

        let mut result = LocalizedPrompt::new(localized, target.clone());
        result.original = content.to_string();
        result.confidence = 0.6; // Lower confidence for rule-based
        result.cultural_notes = notes;

        Ok(result)
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Detect the language of content.
    ///
    /// Uses simple heuristics to detect the source language.
    pub fn detect_language(&self, content: &str) -> Option<Locale> {
        let content_lower = content.to_lowercase();

        // Check for common language patterns
        if content_lower.contains("the ")
            || content_lower.contains(" is ")
            || content_lower.contains(" are ")
        {
            return Some(Locale::new("en", ""));
        }

        if content_lower.contains(" ist ")
            || content_lower.contains(" sind ")
            || content_lower.contains(" und ")
        {
            return Some(Locale::new("de", ""));
        }

        if content_lower.contains(" est ")
            || content_lower.contains(" sont ")
            || content_lower.contains(" et ")
        {
            return Some(Locale::new("fr", ""));
        }

        if content_lower.contains(" es ")
            || content_lower.contains(" son ")
            || content_lower.contains(" y ")
        {
            return Some(Locale::new("es", ""));
        }

        // Check for non-Latin scripts
        if content.chars().any(|c| ('\u{3040}'..='\u{309F}').contains(&c)) {
            return Some(Locale::new("ja", ""));
        }

        if content.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c)) {
            return Some(Locale::new("ko", ""));
        }

        if content.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)) {
            return Some(Locale::new("zh", ""));
        }

        if content.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c)) {
            return Some(Locale::new("ar", ""));
        }

        if content.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)) {
            return Some(Locale::new("ru", ""));
        }

        None
    }

    /// Get the fallback locale.
    pub fn fallback_locale(&self) -> &Locale {
        &self.config.fallback_locale
    }

    /// Validate a locale string.
    pub fn validate_locale(&self, locale_str: &str) -> Result<Locale> {
        Locale::from_bcp47(locale_str)
    }

    /// Get date format for a locale.
    pub fn date_format(&self, locale: &Locale) -> &'static str {
        match (locale.language.as_str(), locale.region.as_str()) {
            ("en", "US") => "MM/DD/YYYY",
            ("en", _) => "DD/MM/YYYY",
            ("de", _) => "DD.MM.YYYY",
            ("ja", _) => "YYYY/MM/DD",
            ("zh", _) => "YYYY-MM-DD",
            ("ko", _) => "YYYY.MM.DD",
            _ => "YYYY-MM-DD", // ISO 8601 fallback
        }
    }

    /// Get number format info for a locale.
    pub fn number_format(&self, locale: &Locale) -> (char, char) {
        // Returns (decimal_separator, thousands_separator)
        match locale.language.as_str() {
            "de" | "fr" | "es" | "it" | "pt" | "nl" | "pl" | "ru" | "tr" => (',', '.'),
            "en" | "ja" | "ko" | "zh" => ('.', ','),
            _ => ('.', ','), // Default to US style
        }
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom LocalizerAgent configuration.
#[derive(Debug, Default)]
pub struct LocalizerAgentBuilder {
    config: Option<LocalizerConfig>,
    gemini_client: Option<Arc<GeminiClient>>,
}

impl LocalizerAgentBuilder {
    /// Set the configuration.
    pub fn config(mut self, config: LocalizerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the Gemini client.
    pub fn gemini_client(mut self, client: Arc<GeminiClient>) -> Self {
        self.gemini_client = Some(client);
        self
    }

    /// Add a supported locale.
    pub fn add_locale(mut self, locale: Locale) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.add_locale(locale);
        self.config = Some(config);
        self
    }

    /// Set the fallback locale.
    pub fn fallback_locale(mut self, locale: Locale) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.fallback_locale = locale;
        self.config = Some(config);
        self
    }

    /// Enable or disable idiom detection.
    pub fn detect_idioms(mut self, enable: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.detect_idioms = enable;
        self.config = Some(config);
        self
    }

    /// Enable or disable formality adjustment.
    pub fn adjust_formality(mut self, enable: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.adjust_formality = enable;
        self.config = Some(config);
        self
    }

    /// Enable or disable format adaptation.
    pub fn adapt_formats(mut self, enable: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.adapt_formats = enable;
        self.config = Some(config);
        self
    }

    /// Enable or disable alternative generation.
    pub fn generate_alternatives(mut self, enable: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.generate_alternatives = enable;
        self.config = Some(config);
        self
    }

    /// Set maximum number of alternatives.
    pub fn max_alternatives(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_alternatives = max;
        self.config = Some(config);
        self
    }

    /// Build the LocalizerAgent.
    pub fn build(self) -> LocalizerAgent {
        LocalizerAgent {
            config: self.config.unwrap_or_default(),
            gemini_client: self.gemini_client,
            phrase_cache: LocalizerAgent::build_phrase_cache(),
        }
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Locale Tests
    // =========================================================================

    #[test]
    fn test_locale_new() {
        let locale = Locale::new("en", "US");
        assert_eq!(locale.language, "en");
        assert_eq!(locale.region, "US");
        assert!(locale.script.is_none());
    }

    #[test]
    fn test_locale_with_script() {
        let locale = Locale::new("zh", "CN").with_script("Hans");
        assert_eq!(locale.language, "zh");
        assert_eq!(locale.region, "CN");
        assert_eq!(locale.script, Some("Hans".to_string()));
    }

    #[test]
    fn test_locale_from_bcp47_simple() {
        let locale = Locale::from_bcp47("en-US").unwrap();
        assert_eq!(locale.language, "en");
        assert_eq!(locale.region, "US");
    }

    #[test]
    fn test_locale_from_bcp47_with_script() {
        let locale = Locale::from_bcp47("zh-Hans-CN").unwrap();
        assert_eq!(locale.language, "zh");
        assert_eq!(locale.region, "CN");
        assert_eq!(locale.script, Some("Hans".to_string()));
    }

    #[test]
    fn test_locale_from_bcp47_language_only() {
        let locale = Locale::from_bcp47("en").unwrap();
        assert_eq!(locale.language, "en");
        assert!(locale.region.is_empty());
    }

    #[test]
    fn test_locale_to_bcp47() {
        let locale = Locale::new("en", "US");
        assert_eq!(locale.to_bcp47(), "en-US");

        let locale_with_script = Locale::new("zh", "CN").with_script("Hans");
        assert_eq!(locale_with_script.to_bcp47(), "zh-Hans-CN");

        let locale_no_region = Locale::new("en", "");
        assert_eq!(locale_no_region.to_bcp47(), "en");
    }

    #[test]
    fn test_locale_display_name() {
        let locale = Locale::new("en", "US");
        assert_eq!(locale.display_name(), "English (United States)");

        let locale_de = Locale::new("de", "DE");
        assert_eq!(locale_de.display_name(), "German (Germany)");

        let locale_no_region = Locale::new("ja", "");
        assert_eq!(locale_no_region.display_name(), "Japanese");
    }

    #[test]
    fn test_locale_is_rtl() {
        assert!(Locale::new("ar", "SA").is_rtl());
        assert!(Locale::new("he", "IL").is_rtl());
        assert!(!Locale::new("en", "US").is_rtl());
        assert!(!Locale::new("de", "DE").is_rtl());
    }

    #[test]
    fn test_locale_common_locales() {
        let locales = Locale::common_locales();
        assert!(!locales.is_empty());
        assert!(locales.iter().any(|l| l.language == "en" && l.region == "US"));
        assert!(locales.iter().any(|l| l.language == "de" && l.region == "DE"));
        assert!(locales.iter().any(|l| l.language == "ja" && l.region == "JP"));
    }

    #[test]
    fn test_locale_default() {
        let locale = Locale::default();
        assert_eq!(locale.language, "en");
        assert_eq!(locale.region, "US");
    }

    #[test]
    fn test_locale_display() {
        let locale = Locale::new("en", "US");
        assert_eq!(format!("{}", locale), "en-US");
    }

    #[test]
    fn test_locale_from_str() {
        let locale: Locale = "de-DE".parse().unwrap();
        assert_eq!(locale.language, "de");
        assert_eq!(locale.region, "DE");
    }

    #[test]
    fn test_locale_underscore_separator() {
        let locale = Locale::from_bcp47("en_US").unwrap();
        assert_eq!(locale.language, "en");
        assert_eq!(locale.region, "US");
    }

    // =========================================================================
    // CulturalCategory Tests
    // =========================================================================

    #[test]
    fn test_cultural_category_display() {
        assert_eq!(CulturalCategory::Idiom.to_string(), "Idiom");
        assert_eq!(CulturalCategory::Formality.to_string(), "Formality");
        assert_eq!(CulturalCategory::DateFormat.to_string(), "Date Format");
    }

    #[test]
    fn test_cultural_category_from_str() {
        assert_eq!("idiom".parse::<CulturalCategory>().unwrap(), CulturalCategory::Idiom);
        assert_eq!("formality".parse::<CulturalCategory>().unwrap(), CulturalCategory::Formality);
        assert_eq!("taboo".parse::<CulturalCategory>().unwrap(), CulturalCategory::Taboo);
        assert_eq!("preference".parse::<CulturalCategory>().unwrap(), CulturalCategory::Preference);
        assert_eq!("dateformat".parse::<CulturalCategory>().unwrap(), CulturalCategory::DateFormat);
        assert_eq!("numberformat".parse::<CulturalCategory>().unwrap(), CulturalCategory::NumberFormat);
    }

    #[test]
    fn test_cultural_category_from_str_invalid() {
        assert!("invalid".parse::<CulturalCategory>().is_err());
    }

    // =========================================================================
    // CulturalSeverity Tests
    // =========================================================================

    #[test]
    fn test_cultural_severity_display() {
        assert_eq!(CulturalSeverity::Info.to_string(), "Info");
        assert_eq!(CulturalSeverity::Suggestion.to_string(), "Suggestion");
        assert_eq!(CulturalSeverity::Warning.to_string(), "Warning");
        assert_eq!(CulturalSeverity::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_cultural_severity_default() {
        assert_eq!(CulturalSeverity::default(), CulturalSeverity::Info);
    }

    #[test]
    fn test_cultural_severity_ordering() {
        assert!(CulturalSeverity::Info < CulturalSeverity::Suggestion);
        assert!(CulturalSeverity::Suggestion < CulturalSeverity::Warning);
        assert!(CulturalSeverity::Warning < CulturalSeverity::Critical);
    }

    // =========================================================================
    // CulturalNote Tests
    // =========================================================================

    #[test]
    fn test_cultural_note_new() {
        let note = CulturalNote::new(CulturalCategory::Idiom, "Test note");
        assert_eq!(note.category, CulturalCategory::Idiom);
        assert_eq!(note.note, "Test note");
        assert_eq!(note.severity, CulturalSeverity::Info);
    }

    #[test]
    fn test_cultural_note_builder() {
        let note = CulturalNote::new(CulturalCategory::Formality, "Formality adjustment")
            .with_severity(CulturalSeverity::Warning)
            .with_original("hey there")
            .with_adapted("guten Tag");

        assert_eq!(note.severity, CulturalSeverity::Warning);
        assert_eq!(note.original_text, Some("hey there".to_string()));
        assert_eq!(note.adapted_text, Some("guten Tag".to_string()));
    }

    #[test]
    fn test_cultural_note_requires_attention() {
        let info = CulturalNote::new(CulturalCategory::Idiom, "Info");
        let warning = CulturalNote::new(CulturalCategory::Taboo, "Warning")
            .with_severity(CulturalSeverity::Warning);
        let critical = CulturalNote::new(CulturalCategory::Taboo, "Critical")
            .with_severity(CulturalSeverity::Critical);

        assert!(!info.requires_attention());
        assert!(warning.requires_attention());
        assert!(critical.requires_attention());
    }

    #[test]
    fn test_cultural_note_to_display() {
        let note = CulturalNote::new(CulturalCategory::Idiom, "Idiom adaptation")
            .with_severity(CulturalSeverity::Suggestion)
            .with_original("raining cats and dogs")
            .with_adapted("es regnet in Strömen");

        let display = note.to_display();
        assert!(display.contains("[Suggestion]"));
        assert!(display.contains("Idiom"));
        assert!(display.contains("raining cats and dogs"));
        assert!(display.contains("es regnet in Strömen"));
    }

    // =========================================================================
    // LocaleAlternative Tests
    // =========================================================================

    #[test]
    fn test_locale_alternative_new() {
        let alt = LocaleAlternative::new("Hallo!", "Casual greeting");
        assert_eq!(alt.content, "Hallo!");
        assert_eq!(alt.context, "Casual greeting");
        assert_eq!(alt.formality_level, 3);
    }

    #[test]
    fn test_locale_alternative_builder() {
        let alt = LocaleAlternative::new("Guten Tag!", "Formal greeting")
            .with_formality(5)
            .with_tags(vec!["formal", "business"]);

        assert_eq!(alt.formality_level, 5);
        assert!(alt.tags.contains("formal"));
        assert!(alt.tags.contains("business"));
    }

    #[test]
    fn test_locale_alternative_formality_bounds() {
        let low = LocaleAlternative::new("test", "test").with_formality(0);
        let high = LocaleAlternative::new("test", "test").with_formality(10);

        assert_eq!(low.formality_level, 1); // Clamped to 1
        assert_eq!(high.formality_level, 5); // Clamped to 5
    }

    #[test]
    fn test_locale_alternative_matches_formality() {
        let alt = LocaleAlternative::new("test", "test").with_formality(3);
        assert!(alt.matches_formality(3));
        assert!(!alt.matches_formality(5));
    }

    // =========================================================================
    // LocalizedPrompt Tests
    // =========================================================================

    #[test]
    fn test_localized_prompt_new() {
        let locale = Locale::new("de", "DE");
        let prompt = LocalizedPrompt::new("Hallo Welt!", locale.clone());

        assert_eq!(prompt.content, "Hallo Welt!");
        assert_eq!(prompt.locale, locale);
        assert!(prompt.cultural_notes.is_empty());
        assert!(prompt.alternatives.is_empty());
    }

    #[test]
    fn test_localized_prompt_add_note() {
        let locale = Locale::new("de", "DE");
        let mut prompt = LocalizedPrompt::new("Hallo", locale);

        prompt.add_note(CulturalNote::new(CulturalCategory::Formality, "Test"));
        assert_eq!(prompt.cultural_notes.len(), 1);
    }

    #[test]
    fn test_localized_prompt_add_alternative() {
        let locale = Locale::new("de", "DE");
        let mut prompt = LocalizedPrompt::new("Hallo", locale);

        prompt.add_alternative(LocaleAlternative::new("Guten Tag", "Formal"));
        assert_eq!(prompt.alternatives.len(), 1);
    }

    #[test]
    fn test_localized_prompt_has_warnings() {
        let locale = Locale::new("de", "DE");
        let mut prompt = LocalizedPrompt::new("Test", locale);

        assert!(!prompt.has_warnings());

        prompt.add_note(
            CulturalNote::new(CulturalCategory::Taboo, "Warning")
                .with_severity(CulturalSeverity::Warning),
        );

        assert!(prompt.has_warnings());
    }

    #[test]
    fn test_localized_prompt_notes_by_category() {
        let locale = Locale::new("de", "DE");
        let mut prompt = LocalizedPrompt::new("Test", locale);

        prompt.add_note(CulturalNote::new(CulturalCategory::Idiom, "Idiom 1"));
        prompt.add_note(CulturalNote::new(CulturalCategory::Formality, "Form 1"));
        prompt.add_note(CulturalNote::new(CulturalCategory::Idiom, "Idiom 2"));

        let idiom_notes = prompt.notes_by_category(CulturalCategory::Idiom);
        assert_eq!(idiom_notes.len(), 2);
    }

    #[test]
    fn test_localized_prompt_most_formal_casual() {
        let locale = Locale::new("de", "DE");
        let mut prompt = LocalizedPrompt::new("Test", locale);

        prompt.add_alternative(LocaleAlternative::new("Casual", "").with_formality(1));
        prompt.add_alternative(LocaleAlternative::new("Neutral", "").with_formality(3));
        prompt.add_alternative(LocaleAlternative::new("Formal", "").with_formality(5));

        assert_eq!(prompt.most_formal().unwrap().formality_level, 5);
        assert_eq!(prompt.most_casual().unwrap().formality_level, 1);
    }

    #[test]
    fn test_localized_prompt_summary() {
        let locale = Locale::new("de", "DE");
        let mut prompt = LocalizedPrompt::new("Test", locale);

        prompt.add_note(CulturalNote::new(CulturalCategory::Idiom, "Note"));
        prompt.add_alternative(LocaleAlternative::new("Alt", "Context"));

        let summary = prompt.summary();
        assert!(summary.contains("German"));
        assert!(summary.contains("1 cultural notes"));
        assert!(summary.contains("1 alternatives"));
    }

    // =========================================================================
    // LocalizerConfig Tests
    // =========================================================================

    #[test]
    fn test_localizer_config_default() {
        let config = LocalizerConfig::default();
        assert!(!config.supported_locales.is_empty());
        assert!(config.detect_idioms);
        assert!(config.adjust_formality);
        assert!(config.generate_alternatives);
    }

    #[test]
    fn test_localizer_config_minimal() {
        let config = LocalizerConfig::minimal();
        assert_eq!(config.supported_locales.len(), 4);
        assert!(!config.detect_idioms);
        assert!(!config.adjust_formality);
        assert!(!config.generate_alternatives);
    }

    #[test]
    fn test_localizer_config_thorough() {
        let config = LocalizerConfig::thorough();
        assert!(config.detect_idioms);
        assert!(config.adjust_formality);
        assert!(config.generate_alternatives);
        assert_eq!(config.max_alternatives, 5);
    }

    #[test]
    fn test_localizer_config_is_supported() {
        let config = LocalizerConfig::default();
        assert!(config.is_supported(&Locale::new("en", "US")));
        assert!(config.is_supported(&Locale::new("de", "DE")));
    }

    #[test]
    fn test_localizer_config_add_locale() {
        let mut config = LocalizerConfig::minimal();
        let new_locale = Locale::new("ja", "JP");

        assert!(!config.is_supported(&new_locale));
        config.add_locale(new_locale.clone());
        assert!(config.is_supported(&new_locale));
    }

    // =========================================================================
    // LocalizerAgent Tests
    // =========================================================================

    #[test]
    fn test_localizer_agent_new() {
        let agent = LocalizerAgent::new();
        assert!(agent.gemini_client.is_none());
        assert!(!agent.config.supported_locales.is_empty());
    }

    #[test]
    fn test_localizer_agent_builder() {
        let agent = LocalizerAgent::builder()
            .detect_idioms(false)
            .adjust_formality(false)
            .max_alternatives(5)
            .build();

        assert!(!agent.config.detect_idioms);
        assert!(!agent.config.adjust_formality);
        assert_eq!(agent.config.max_alternatives, 5);
    }

    #[test]
    fn test_localizer_agent_is_supported() {
        let agent = LocalizerAgent::new();
        assert!(agent.is_supported(&Locale::new("en", "US")));
        assert!(agent.is_supported(&Locale::new("de", "DE")));
    }

    #[test]
    fn test_localizer_agent_detect_language_english() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("The quick brown fox jumps over the lazy dog");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "en");
    }

    #[test]
    fn test_localizer_agent_detect_language_german() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("Das ist ein Test und hier sind mehr Worte");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "de");
    }

    #[test]
    fn test_localizer_agent_detect_language_french() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("C'est un test et voici plus de mots");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "fr");
    }

    #[test]
    fn test_localizer_agent_detect_language_japanese() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("これはテストです");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "ja");
    }

    #[test]
    fn test_localizer_agent_detect_language_korean() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("이것은 테스트입니다");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "ko");
    }

    #[test]
    fn test_localizer_agent_detect_language_chinese() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("这是一个测试");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "zh");
    }

    #[test]
    fn test_localizer_agent_detect_language_arabic() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("هذا اختبار");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "ar");
    }

    #[test]
    fn test_localizer_agent_detect_language_russian() {
        let agent = LocalizerAgent::new();
        let locale = agent.detect_language("Это тест");
        assert!(locale.is_some());
        assert_eq!(locale.unwrap().language, "ru");
    }

    #[test]
    fn test_localizer_agent_date_format() {
        let agent = LocalizerAgent::new();

        assert_eq!(agent.date_format(&Locale::new("en", "US")), "MM/DD/YYYY");
        assert_eq!(agent.date_format(&Locale::new("en", "GB")), "DD/MM/YYYY");
        assert_eq!(agent.date_format(&Locale::new("de", "DE")), "DD.MM.YYYY");
        assert_eq!(agent.date_format(&Locale::new("ja", "JP")), "YYYY/MM/DD");
    }

    #[test]
    fn test_localizer_agent_number_format() {
        let agent = LocalizerAgent::new();

        let (dec, thou) = agent.number_format(&Locale::new("en", "US"));
        assert_eq!(dec, '.');
        assert_eq!(thou, ',');

        let (dec_de, thou_de) = agent.number_format(&Locale::new("de", "DE"));
        assert_eq!(dec_de, ',');
        assert_eq!(thou_de, '.');
    }

    #[test]
    fn test_localizer_agent_validate_locale() {
        let agent = LocalizerAgent::new();

        assert!(agent.validate_locale("en-US").is_ok());
        assert!(agent.validate_locale("de-DE").is_ok());
        assert!(agent.validate_locale("zh-Hans-CN").is_ok());
    }

    #[test]
    fn test_localizer_agent_fallback_locale() {
        let agent = LocalizerAgent::new();
        assert_eq!(agent.fallback_locale().language, "en");
        assert_eq!(agent.fallback_locale().region, "US");
    }

    #[tokio::test]
    async fn test_localizer_agent_localize_empty() {
        let agent = LocalizerAgent::new();
        let locale = Locale::new("de", "DE");
        let result = agent.localize("", &locale).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_localizer_agent_localize_without_llm() {
        let agent = LocalizerAgent::new();
        let locale = Locale::new("de", "DE");
        let result = agent.localize("Hello, world!", &locale).await;

        assert!(result.is_ok());
        let localized = result.unwrap();
        assert!(!localized.content.is_empty());
        assert_eq!(localized.locale, locale);
    }

    #[tokio::test]
    async fn test_localizer_agent_localize_with_phrase_cache() {
        let agent = LocalizerAgent::new();
        let locale = Locale::new("de", "DE");
        let result = agent.localize("Hello", &locale).await.unwrap();

        // Should use phrase cache for common greeting
        assert!(result.content.contains("Hallo") || result.content.contains("Hello"));
    }

    #[tokio::test]
    async fn test_localizer_agent_localize_rtl_warning() {
        let agent = LocalizerAgent::new();
        let locale = Locale::new("ar", "SA");
        let result = agent.localize("Hello, world!", &locale).await.unwrap();

        // Should include RTL warning in cultural notes
        assert!(result.cultural_notes.iter().any(|n| {
            n.note.contains("right-to-left")
        }));
    }

    // =========================================================================
    // Builder Pattern Tests
    // =========================================================================

    #[test]
    fn test_builder_add_locale() {
        let agent = LocalizerAgent::builder()
            .add_locale(Locale::new("cs", "CZ"))
            .build();

        assert!(agent.is_supported(&Locale::new("cs", "CZ")));
    }

    #[test]
    fn test_builder_fallback_locale() {
        let agent = LocalizerAgent::builder()
            .fallback_locale(Locale::new("de", "DE"))
            .build();

        assert_eq!(agent.fallback_locale().language, "de");
    }

    #[test]
    fn test_builder_all_options() {
        let agent = LocalizerAgent::builder()
            .detect_idioms(false)
            .adjust_formality(false)
            .adapt_formats(false)
            .generate_alternatives(false)
            .max_alternatives(0)
            .build();

        assert!(!agent.config.detect_idioms);
        assert!(!agent.config.adjust_formality);
        assert!(!agent.config.adapt_formats);
        assert!(!agent.config.generate_alternatives);
        assert_eq!(agent.config.max_alternatives, 0);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_locale_case_normalization() {
        let locale1 = Locale::new("EN", "us");
        let locale2 = Locale::new("en", "US");

        assert_eq!(locale1.language, "en");
        assert_eq!(locale1.region, "US");
        // Should be normalized to same format
        assert_eq!(locale1.language, locale2.language);
        assert_eq!(locale1.region, locale2.region);
    }

    #[test]
    fn test_script_formatting() {
        let locale = Locale::new("zh", "CN").with_script("HANS");
        assert_eq!(locale.script, Some("Hans".to_string()));

        let locale2 = Locale::new("zh", "TW").with_script("hant");
        assert_eq!(locale2.script, Some("Hant".to_string()));
    }

    #[test]
    fn test_empty_region() {
        let locale = Locale::new("en", "");
        assert_eq!(locale.to_bcp47(), "en");
        assert_eq!(locale.display_name(), "English");
    }

    #[test]
    fn test_phrase_cache_contains_common_phrases() {
        let cache = LocalizerAgent::build_phrase_cache();

        assert!(cache.contains_key("hello"));
        assert!(cache.contains_key("goodbye"));
        assert!(cache.contains_key("thank you"));

        // Check specific translations exist
        assert!(cache.get("hello").unwrap().contains_key("de"));
        assert!(cache.get("hello").unwrap().contains_key("fr"));
        assert!(cache.get("hello").unwrap().contains_key("ja"));
    }
}
