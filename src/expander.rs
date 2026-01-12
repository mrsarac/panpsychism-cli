//! Expander Agent module for Project Panpsychism.
//!
//! Implements "The Detail Conjurer" — an agent that enriches content with
//! additional information, examples, definitions, and context. Like a sorcerer
//! who weaves threads of knowledge into a tapestry, the Expander Agent takes
//! sparse content and conjures forth rich, detailed material.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In the arcane arts of knowledge synthesis, the Expander Agent serves as
//! the **Detail Conjurer** — a specialist in the magical art of expansion:
//!
//! - **Examples** are like demonstration spells, showing magic in action
//! - **Definitions** are naming rituals that bind meaning to terms
//! - **Context** is the ambient magical field that gives spells their power
//! - **Alternatives** are parallel incantations achieving similar effects
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Drive to enrich understanding through comprehensive detail
//! - **RATIO**: Logical extraction of terms that require definition
//! - **LAETITIA**: Joy through clarity and illuminating examples
//! - **NATURA**: Natural flow from abstract concepts to concrete instances
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::expander::{ExpanderAgent, ExpansionType};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let agent = ExpanderAgent::new();
//!
//!     let content = "Implement OAuth2 authentication with JWT tokens.";
//!     let expanded = agent.expand(content, ExpansionType::Examples).await?;
//!
//!     println!("Expanded content: {}", expanded.content);
//!     println!("Added {} examples", expanded.examples.len());
//!     Ok(())
//! }
//! ```

use crate::gemini::{GeminiClient, Message};
use crate::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

// =============================================================================
// REGEX PATTERNS FOR TERM EXTRACTION
// =============================================================================

/// Pattern for technical terms (CamelCase, snake_case, acronyms).
/// Matches:
/// - CamelCase: OAuth2, PostgreSQL, JavaScript
/// - snake_case: api_key, access_token
/// - Acronyms: API, JWT, HTTP
/// - Mixed: OAuth2, ES6
const TECHNICAL_TERM_PATTERN: &str =
    r"(?x)
    \b[A-Z][a-zA-Z]+\d*\b |                  # CamelCase/PascalCase: OAuth2, PostgreSQL, JavaScript
    \b[a-z]+(?:_[a-z0-9]+)+\b |              # snake_case: api_key, access_token
    \b[A-Z]{2,}\d*s?\b                       # Acronyms: API, JWT, HTTP, APIs
";

/// Pattern for code-like terms (functions, methods, classes).
const CODE_TERM_PATTERN: &str =
    r"(?x)
    \b\w+\(\) |                        # Functions: authenticate()
    \b\w+\.\w+ |                       # Methods: user.login
    `[^`]+`                            # Inline code: `Bearer token`
";

/// Pattern for domain-specific terms (common tech vocabulary).
const DOMAIN_TERM_PATTERN: &str =
    r"(?xi)
    \b(?:
        authentication | authorization | middleware | endpoint |
        callback | webhook | payload | schema | migration |
        repository | singleton | factory | adapter | decorator |
        async | await | promise | future | stream | buffer |
        encryption | hashing | salting | tokenization |
        rate.?limiting | load.?balancing | caching | sharding
    )\b
";

// =============================================================================
// EXPANSION TYPES
// =============================================================================

/// Types of content expansion the Detail Conjurer can perform.
///
/// Each expansion type represents a different magical enhancement:
/// - Examples illuminate through demonstration
/// - Definitions clarify through naming
/// - Context enriches through background
/// - Alternatives expand through options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExpansionType {
    /// Add code or usage examples to illustrate concepts.
    ///
    /// Like casting a demonstration spell, examples show
    /// abstract concepts manifested in concrete form.
    #[default]
    Examples,

    /// Define technical terms and jargon.
    ///
    /// A naming ritual that binds precise meaning to
    /// potentially ambiguous terminology.
    Definitions,

    /// Add background information and context.
    ///
    /// The ambient magical field that gives meaning its
    /// proper setting and significance.
    Context,

    /// Suggest alternative approaches or implementations.
    ///
    /// Parallel incantations that achieve similar effects
    /// through different magical paths.
    Alternatives,
}

impl std::fmt::Display for ExpansionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Examples => write!(f, "Examples"),
            Self::Definitions => write!(f, "Definitions"),
            Self::Context => write!(f, "Context"),
            Self::Alternatives => write!(f, "Alternatives"),
        }
    }
}

impl std::str::FromStr for ExpansionType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "examples" | "example" | "demo" | "demos" => Ok(Self::Examples),
            "definitions" | "definition" | "define" | "terms" => Ok(Self::Definitions),
            "context" | "background" | "info" => Ok(Self::Context),
            "alternatives" | "alternative" | "options" | "other" => Ok(Self::Alternatives),
            _ => Err(Error::Config(format!(
                "Unknown expansion type: '{}'. Valid types: examples, definitions, context, alternatives",
                s
            ))),
        }
    }
}

impl ExpansionType {
    /// Get all expansion types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Examples,
            Self::Definitions,
            Self::Context,
            Self::Alternatives,
        ]
    }

    /// Get the prompt instruction for this expansion type.
    pub fn instruction(&self) -> &'static str {
        match self {
            Self::Examples => {
                "Add practical code examples that demonstrate the concepts. \
                 Include both simple and complex examples where appropriate. \
                 Use proper syntax highlighting with language tags."
            }
            Self::Definitions => {
                "Identify and define technical terms, acronyms, and jargon. \
                 Provide clear, concise definitions that a newcomer would understand. \
                 Include etymology or origin where it aids understanding."
            }
            Self::Context => {
                "Add relevant background information and context. \
                 Explain why something matters, its history, and how it fits \
                 into the broader technical landscape."
            }
            Self::Alternatives => {
                "Suggest alternative approaches, tools, or implementations. \
                 Compare trade-offs between options. Help the reader understand \
                 when to choose one approach over another."
            }
        }
    }
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// A code or usage example generated by the Expander.
///
/// Examples are demonstration spells — they show abstract
/// concepts manifested in concrete, executable form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Title or description of the example.
    pub title: String,
    /// Programming language (for syntax highlighting).
    pub language: Option<String>,
    /// The example code or content.
    pub code: String,
    /// Explanation of what the example demonstrates.
    pub explanation: Option<String>,
    /// Complexity level (beginner, intermediate, advanced).
    pub complexity: ExampleComplexity,
}

impl Example {
    /// Create a new example.
    pub fn new(title: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            language: None,
            code: code.into(),
            explanation: None,
            complexity: ExampleComplexity::Intermediate,
        }
    }

    /// Set the programming language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set the explanation.
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }

    /// Set the complexity level.
    pub fn with_complexity(mut self, complexity: ExampleComplexity) -> Self {
        self.complexity = complexity;
        self
    }

    /// Format the example as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!("### {}\n\n", self.title);

        if let Some(ref explanation) = self.explanation {
            output.push_str(explanation);
            output.push_str("\n\n");
        }

        let lang = self.language.as_deref().unwrap_or("text");
        output.push_str(&format!("```{}\n{}\n```\n", lang, self.code));

        output
    }
}

/// Complexity level for examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExampleComplexity {
    /// Simple, introductory example.
    Beginner,
    /// Standard complexity example.
    #[default]
    Intermediate,
    /// Complex, advanced example.
    Advanced,
}

impl std::fmt::Display for ExampleComplexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Beginner => write!(f, "beginner"),
            Self::Intermediate => write!(f, "intermediate"),
            Self::Advanced => write!(f, "advanced"),
        }
    }
}

/// A technical term definition.
///
/// Definitions are naming rituals that bind precise meaning
/// to terminology, dispelling the fog of ambiguity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Definition {
    /// The term being defined.
    pub term: String,
    /// The definition text.
    pub definition: String,
    /// Category (e.g., "Security", "Database", "Protocol").
    pub category: Option<String>,
    /// Related terms for cross-reference.
    pub related_terms: Vec<String>,
    /// Example usage of the term.
    pub usage_example: Option<String>,
}

impl Definition {
    /// Create a new definition.
    pub fn new(term: impl Into<String>, definition: impl Into<String>) -> Self {
        Self {
            term: term.into(),
            definition: definition.into(),
            category: None,
            related_terms: Vec::new(),
            usage_example: None,
        }
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Add related terms.
    pub fn with_related_terms(mut self, terms: Vec<String>) -> Self {
        self.related_terms = terms;
        self
    }

    /// Format the definition as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!("**{}**", self.term);

        if let Some(ref category) = self.category {
            output.push_str(&format!(" _{}_", category));
        }

        output.push_str(&format!(": {}\n", self.definition));

        if !self.related_terms.is_empty() {
            output.push_str(&format!("  - Related: {}\n", self.related_terms.join(", ")));
        }

        if let Some(ref example) = self.usage_example {
            output.push_str(&format!("  - Example: {}\n", example));
        }

        output
    }
}

/// An alternative approach or implementation.
///
/// Alternatives represent parallel magical paths — different
/// incantations that achieve similar effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    /// Name of the alternative approach.
    pub name: String,
    /// Description of the approach.
    pub description: String,
    /// Pros of this approach.
    pub pros: Vec<String>,
    /// Cons of this approach.
    pub cons: Vec<String>,
    /// When to use this approach.
    pub use_when: Option<String>,
}

impl Alternative {
    /// Create a new alternative.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            pros: Vec::new(),
            cons: Vec::new(),
            use_when: None,
        }
    }

    /// Set pros.
    pub fn with_pros(mut self, pros: Vec<String>) -> Self {
        self.pros = pros;
        self
    }

    /// Set cons.
    pub fn with_cons(mut self, cons: Vec<String>) -> Self {
        self.cons = cons;
        self
    }

    /// Format the alternative as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!("### {}\n\n{}\n\n", self.name, self.description);

        if !self.pros.is_empty() {
            output.push_str("**Pros:**\n");
            for pro in &self.pros {
                output.push_str(&format!("- {}\n", pro));
            }
            output.push('\n');
        }

        if !self.cons.is_empty() {
            output.push_str("**Cons:**\n");
            for con in &self.cons {
                output.push_str(&format!("- {}\n", con));
            }
            output.push('\n');
        }

        if let Some(ref use_when) = self.use_when {
            output.push_str(&format!("**Use when:** {}\n", use_when));
        }

        output
    }
}

/// Contextual background information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInfo {
    /// Topic of the context.
    pub topic: String,
    /// Background information text.
    pub background: String,
    /// Historical context if relevant.
    pub history: Option<String>,
    /// Related concepts.
    pub related_concepts: Vec<String>,
    /// External resources for further reading.
    pub resources: Vec<String>,
}

impl ContextInfo {
    /// Create new context info.
    pub fn new(topic: impl Into<String>, background: impl Into<String>) -> Self {
        Self {
            topic: topic.into(),
            background: background.into(),
            history: None,
            related_concepts: Vec::new(),
            resources: Vec::new(),
        }
    }

    /// Format the context as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!("## Background: {}\n\n{}\n\n", self.topic, self.background);

        if let Some(ref history) = self.history {
            output.push_str(&format!("### History\n\n{}\n\n", history));
        }

        if !self.related_concepts.is_empty() {
            output.push_str("### Related Concepts\n\n");
            for concept in &self.related_concepts {
                output.push_str(&format!("- {}\n", concept));
            }
            output.push('\n');
        }

        if !self.resources.is_empty() {
            output.push_str("### Further Reading\n\n");
            for resource in &self.resources {
                output.push_str(&format!("- {}\n", resource));
            }
        }

        output
    }
}

// =============================================================================
// EXPANDED CONTENT
// =============================================================================

/// The result of content expansion.
///
/// Contains the enriched content along with metadata about
/// the expansion process — the fruits of the Detail Conjurer's work.
#[derive(Debug, Clone)]
pub struct ExpandedContent {
    /// Original content before expansion.
    pub original: String,
    /// Expanded/enriched content.
    pub content: String,
    /// Type of expansion performed.
    pub expansion_type: ExpansionType,
    /// Generated examples (if expansion type was Examples).
    pub examples: Vec<Example>,
    /// Generated definitions (if expansion type was Definitions).
    pub definitions: Vec<Definition>,
    /// Generated alternatives (if expansion type was Alternatives).
    pub alternatives: Vec<Alternative>,
    /// Context information (if expansion type was Context).
    pub context: Option<ContextInfo>,
    /// Terms extracted from the original content.
    pub extracted_terms: Vec<String>,
    /// Processing duration in milliseconds.
    pub duration_ms: u64,
    /// Confidence score for the expansion (0.0 - 1.0).
    pub confidence: f64,
}

impl ExpandedContent {
    /// Create a new expanded content result.
    pub fn new(original: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            original: original.into(),
            content: content.into(),
            expansion_type: ExpansionType::default(),
            examples: Vec::new(),
            definitions: Vec::new(),
            alternatives: Vec::new(),
            context: None,
            extracted_terms: Vec::new(),
            duration_ms: 0,
            confidence: 0.0,
        }
    }

    /// Check if the expansion produced substantial content.
    pub fn is_substantial(&self) -> bool {
        !self.examples.is_empty()
            || !self.definitions.is_empty()
            || !self.alternatives.is_empty()
            || self.context.is_some()
    }

    /// Get a summary of what was expanded.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if !self.examples.is_empty() {
            parts.push(format!("{} examples", self.examples.len()));
        }
        if !self.definitions.is_empty() {
            parts.push(format!("{} definitions", self.definitions.len()));
        }
        if !self.alternatives.is_empty() {
            parts.push(format!("{} alternatives", self.alternatives.len()));
        }
        if self.context.is_some() {
            parts.push("context".to_string());
        }

        if parts.is_empty() {
            "No expansions".to_string()
        } else {
            format!("Expanded with: {}", parts.join(", "))
        }
    }

    /// Format all expansions as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Expanded Content\n\n");
        output.push_str(&self.content);
        output.push_str("\n\n---\n\n");

        if !self.definitions.is_empty() {
            output.push_str("## Glossary\n\n");
            for def in &self.definitions {
                output.push_str(&def.to_markdown());
                output.push('\n');
            }
            output.push('\n');
        }

        if !self.examples.is_empty() {
            output.push_str("## Examples\n\n");
            for example in &self.examples {
                output.push_str(&example.to_markdown());
                output.push('\n');
            }
        }

        if !self.alternatives.is_empty() {
            output.push_str("## Alternatives\n\n");
            for alt in &self.alternatives {
                output.push_str(&alt.to_markdown());
            }
        }

        if let Some(ref ctx) = self.context {
            output.push_str(&ctx.to_markdown());
        }

        output
    }
}

// =============================================================================
// EXPANDER AGENT CONFIGURATION
// =============================================================================

/// Configuration for the Expander Agent.
#[derive(Debug, Clone)]
pub struct ExpanderConfig {
    /// Maximum number of examples to generate.
    pub max_examples: usize,
    /// Maximum number of definitions to generate.
    pub max_definitions: usize,
    /// Maximum number of alternatives to suggest.
    pub max_alternatives: usize,
    /// Minimum confidence threshold for including expansions.
    pub confidence_threshold: f64,
    /// Include code detection for example generation.
    pub detect_code: bool,
    /// Timeout in seconds for LLM requests.
    pub timeout_secs: u64,
}

impl Default for ExpanderConfig {
    fn default() -> Self {
        Self {
            max_examples: 5,
            max_definitions: 10,
            max_alternatives: 4,
            confidence_threshold: 0.5,
            detect_code: true,
            timeout_secs: 60,
        }
    }
}

impl ExpanderConfig {
    /// Create a fast configuration with fewer expansions.
    pub fn fast() -> Self {
        Self {
            max_examples: 2,
            max_definitions: 5,
            max_alternatives: 2,
            confidence_threshold: 0.6,
            detect_code: true,
            timeout_secs: 30,
        }
    }

    /// Create a thorough configuration with more expansions.
    pub fn thorough() -> Self {
        Self {
            max_examples: 10,
            max_definitions: 20,
            max_alternatives: 6,
            confidence_threshold: 0.4,
            detect_code: true,
            timeout_secs: 120,
        }
    }
}

// =============================================================================
// TERM EXTRACTOR
// =============================================================================

/// Extracts technical terms from content using regex patterns.
#[derive(Debug, Clone)]
struct TermExtractor {
    technical_pattern: Regex,
    code_pattern: Regex,
    domain_pattern: Regex,
}

impl Default for TermExtractor {
    fn default() -> Self {
        Self {
            technical_pattern: Regex::new(TECHNICAL_TERM_PATTERN)
                .expect("Invalid technical term pattern"),
            code_pattern: Regex::new(CODE_TERM_PATTERN).expect("Invalid code term pattern"),
            domain_pattern: Regex::new(DOMAIN_TERM_PATTERN).expect("Invalid domain term pattern"),
        }
    }
}

impl TermExtractor {
    /// Extract all technical terms from content.
    pub fn extract(&self, content: &str) -> Vec<String> {
        let mut terms = HashSet::new();

        // Extract technical terms (CamelCase, snake_case, acronyms)
        for cap in self.technical_pattern.find_iter(content) {
            let term = cap.as_str().to_string();
            if self.is_valid_term(&term) {
                terms.insert(term);
            }
        }

        // Extract code-like terms
        for cap in self.code_pattern.find_iter(content) {
            let term = cap.as_str().trim_matches('`').to_string();
            if self.is_valid_term(&term) {
                terms.insert(term);
            }
        }

        // Extract domain-specific terms
        for cap in self.domain_pattern.find_iter(content) {
            terms.insert(cap.as_str().to_lowercase());
        }

        let mut result: Vec<_> = terms.into_iter().collect();
        result.sort();
        result
    }

    /// Check if a term is valid (not too short, not common English).
    fn is_valid_term(&self, term: &str) -> bool {
        // Minimum length
        if term.len() < 2 {
            return false;
        }

        // Filter common English words that match patterns (including capitalized forms)
        let common_words = [
            // Common lowercase
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was",
            "one", "our", "out", "has", "his", "how", "its", "may", "new", "now", "old", "see",
            "two", "way", "who", "did", "get", "let", "put", "say", "she", "too", "use",
            // Common capitalized words that might appear in sentences
            "implement", "with", "tokens", "database", "using", "from", "this", "that",
            "when", "where", "which", "what", "they", "them", "then", "than", "some",
            "these", "those", "each", "every", "other", "such", "only", "also", "just",
            "like", "make", "made", "into", "over", "most", "more", "many", "much",
            "very", "well", "back", "been", "being", "both", "come", "could", "down",
            "even", "first", "good", "great", "here", "know", "last", "long", "look",
            "must", "need", "next", "part", "same", "should", "still", "take", "there",
            "think", "through", "time", "under", "want", "will", "work", "would", "year",
        ];

        !common_words.contains(&term.to_lowercase().as_str())
    }

    /// Detect if content contains code blocks.
    pub fn has_code_blocks(&self, content: &str) -> bool {
        content.contains("```") || content.contains("    ") && content.lines().count() > 3
    }

    /// Detect the likely programming language from content.
    pub fn detect_language(&self, content: &str) -> Option<String> {
        // Check for explicit language hints
        let code_fence_re = Regex::new(r"```(\w+)").ok()?;
        if let Some(cap) = code_fence_re.captures(content) {
            return cap.get(1).map(|m| m.as_str().to_string());
        }

        // Heuristic detection based on keywords
        let content_lower = content.to_lowercase();

        if content_lower.contains("fn ") && content_lower.contains("let ") {
            return Some("rust".to_string());
        }
        if content_lower.contains("def ") && content_lower.contains("import ") {
            return Some("python".to_string());
        }
        if content_lower.contains("function ") || content_lower.contains("const ") {
            return Some("javascript".to_string());
        }
        if content_lower.contains("func ") && content_lower.contains("package ") {
            return Some("go".to_string());
        }

        None
    }
}

// =============================================================================
// EXPANDER AGENT
// =============================================================================

/// The Detail Conjurer — expands content with examples, definitions, and context.
///
/// The Expander Agent is a specialized component in the Sorcerer's Wand system
/// that enriches sparse content with detailed information. Like a master conjurer
/// who summons forth knowledge from the aether, this agent transforms brief
/// descriptions into comprehensive documentation.
///
/// ## Capabilities
///
/// - **Example Generation**: Creates code examples demonstrating concepts
/// - **Term Definition**: Identifies and defines technical terminology
/// - **Context Addition**: Provides background and historical information
/// - **Alternative Suggestion**: Proposes different approaches and trade-offs
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::expander::{ExpanderAgent, ExpansionType};
///
/// let agent = ExpanderAgent::new();
///
/// // Expand with examples
/// let expanded = agent.expand("Implement rate limiting", ExpansionType::Examples).await?;
///
/// // Extract and define terms
/// let definitions = agent.define_terms("Use JWT for authentication").await?;
///
/// // Generate examples with code detection
/// let examples = agent.add_examples("Async/await in Rust").await?;
/// ```
#[derive(Debug, Clone)]
pub struct ExpanderAgent {
    /// Configuration for expansion behavior.
    config: ExpanderConfig,
    /// Term extraction engine.
    term_extractor: TermExtractor,
    /// Gemini client for LLM calls (optional).
    gemini_client: Option<Arc<GeminiClient>>,
}

impl Default for ExpanderAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpanderAgent {
    /// Create a new Expander Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExpanderConfig::default(),
            term_extractor: TermExtractor::default(),
            gemini_client: None,
        }
    }

    /// Create a new Expander Agent with a Gemini client.
    pub fn with_gemini(client: Arc<GeminiClient>) -> Self {
        Self {
            config: ExpanderConfig::default(),
            term_extractor: TermExtractor::default(),
            gemini_client: Some(client),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> ExpanderAgentBuilder {
        ExpanderAgentBuilder::default()
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: ExpanderConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ExpanderConfig {
        &self.config
    }

    // =========================================================================
    // MAIN EXPANSION METHOD
    // =========================================================================

    /// Expand content with the specified expansion type.
    ///
    /// This is the primary method of the Detail Conjurer, channeling magical
    /// energy to enrich content based on the requested expansion type.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to expand
    /// * `expansion` - The type of expansion to perform
    ///
    /// # Returns
    ///
    /// An `ExpandedContent` containing the enriched content and metadata.
    ///
    /// # Errors
    ///
    /// Returns `Error::Synthesis` if LLM generation fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = ExpanderAgent::new();
    /// let expanded = agent.expand(
    ///     "Implement OAuth2 authentication",
    ///     ExpansionType::Examples
    /// ).await?;
    /// ```
    pub async fn expand(&self, content: &str, expansion: ExpansionType) -> Result<ExpandedContent> {
        let start = Instant::now();

        if content.trim().is_empty() {
            return Err(Error::Synthesis("Cannot expand empty content".to_string()));
        }

        // Extract terms for context
        let extracted_terms = self.term_extractor.extract(content);
        debug!("Extracted {} terms from content", extracted_terms.len());

        // Generate expanded content based on type
        let mut result = match expansion {
            ExpansionType::Examples => self.expand_with_examples(content).await?,
            ExpansionType::Definitions => self.expand_with_definitions(content).await?,
            ExpansionType::Context => self.expand_with_context(content).await?,
            ExpansionType::Alternatives => self.expand_with_alternatives(content).await?,
        };

        result.expansion_type = expansion;
        result.extracted_terms = extracted_terms;
        result.duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "Expansion complete: {} in {}ms",
            result.summary(),
            result.duration_ms
        );

        Ok(result)
    }

    // =========================================================================
    // EXAMPLE GENERATION
    // =========================================================================

    /// Add examples to content.
    ///
    /// Generates practical code examples that demonstrate the concepts
    /// in the provided content. Detects programming languages and
    /// creates examples at varying complexity levels.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to add examples for
    ///
    /// # Returns
    ///
    /// A vector of generated examples.
    pub async fn add_examples(&self, content: &str) -> Result<Vec<Example>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        let detected_language = self.term_extractor.detect_language(content);
        let has_code = self.term_extractor.has_code_blocks(content);

        if let Some(client) = &self.gemini_client {
            self.generate_examples_with_llm(client, content, detected_language.as_deref())
                .await
        } else {
            // Fallback: generate placeholder examples
            Ok(self.generate_placeholder_examples(content, detected_language.as_deref(), has_code))
        }
    }

    /// Generate examples using LLM.
    async fn generate_examples_with_llm(
        &self,
        client: &GeminiClient,
        content: &str,
        language: Option<&str>,
    ) -> Result<Vec<Example>> {
        let lang_hint = language
            .map(|l| format!(" Use {} syntax.", l))
            .unwrap_or_default();

        let prompt = format!(
            "Generate {} practical code examples for the following content.{}\n\n\
             Content: {}\n\n\
             For each example, provide:\n\
             1. A descriptive title\n\
             2. The code\n\
             3. A brief explanation\n\n\
             Format as JSON array with objects containing: title, language, code, explanation, complexity (beginner/intermediate/advanced)",
            self.config.max_examples, lang_hint, content
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

        self.parse_examples_response(&response_text)
    }

    /// Parse LLM response into Example structs.
    fn parse_examples_response(&self, response: &str) -> Result<Vec<Example>> {
        // Try to parse as JSON
        if let Ok(examples) = serde_json::from_str::<Vec<serde_json::Value>>(response) {
            return Ok(examples
                .into_iter()
                .filter_map(|v| self.value_to_example(v))
                .take(self.config.max_examples)
                .collect());
        }

        // Try to extract JSON from markdown code block
        let json_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
        if let Some(re) = json_re {
            if let Some(caps) = re.captures(response) {
                if let Some(json_text) = caps.get(1) {
                    if let Ok(examples) =
                        serde_json::from_str::<Vec<serde_json::Value>>(json_text.as_str().trim())
                    {
                        return Ok(examples
                            .into_iter()
                            .filter_map(|v| self.value_to_example(v))
                            .take(self.config.max_examples)
                            .collect());
                    }
                }
            }
        }

        // Fallback: extract code blocks from response
        let code_block_re = Regex::new(r"```(\w+)?\s*([\s\S]*?)```").ok();
        let mut examples = Vec::new();

        if let Some(re) = code_block_re {
            for (idx, caps) in re.captures_iter(response).enumerate() {
                let language = caps.get(1).map(|m| m.as_str().to_string());
                let code = caps.get(2).map(|m| m.as_str().trim().to_string());

                if let Some(code) = code {
                    let mut example = Example::new(format!("Example {}", idx + 1), code);
                    if let Some(lang) = language {
                        example = example.with_language(lang);
                    }
                    examples.push(example);

                    if examples.len() >= self.config.max_examples {
                        break;
                    }
                }
            }
        }

        Ok(examples)
    }

    /// Convert JSON value to Example.
    fn value_to_example(&self, value: serde_json::Value) -> Option<Example> {
        let title = value.get("title")?.as_str()?.to_string();
        let code = value.get("code")?.as_str()?.to_string();

        let mut example = Example::new(title, code);

        if let Some(lang) = value.get("language").and_then(|v| v.as_str()) {
            example = example.with_language(lang);
        }

        if let Some(explanation) = value.get("explanation").and_then(|v| v.as_str()) {
            example = example.with_explanation(explanation);
        }

        if let Some(complexity) = value.get("complexity").and_then(|v| v.as_str()) {
            example = example.with_complexity(match complexity {
                "beginner" => ExampleComplexity::Beginner,
                "advanced" => ExampleComplexity::Advanced,
                _ => ExampleComplexity::Intermediate,
            });
        }

        Some(example)
    }

    /// Generate placeholder examples without LLM.
    fn generate_placeholder_examples(
        &self,
        content: &str,
        language: Option<&str>,
        _has_code: bool,
    ) -> Vec<Example> {
        let lang = language.unwrap_or("text");
        let terms = self.term_extractor.extract(content);

        let mut examples = Vec::new();

        // Generate basic example
        if !terms.is_empty() {
            let example = Example::new(
                format!("Basic {} usage", terms.first().unwrap_or(&"concept".to_string())),
                format!("// Example demonstrating: {}", content),
            )
            .with_language(lang)
            .with_complexity(ExampleComplexity::Beginner);

            examples.push(example);
        }

        examples
    }

    // =========================================================================
    // TERM DEFINITION
    // =========================================================================

    /// Define technical terms in content.
    ///
    /// Identifies technical terms, acronyms, and jargon, then generates
    /// clear definitions for each. Uses regex-based term extraction
    /// followed by LLM-powered definition generation.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to extract and define terms from
    ///
    /// # Returns
    ///
    /// A vector of term definitions.
    pub async fn define_terms(&self, content: &str) -> Result<Vec<Definition>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Extract terms using regex patterns
        let terms = self.term_extractor.extract(content);

        if terms.is_empty() {
            debug!("No technical terms found in content");
            return Ok(Vec::new());
        }

        debug!("Found {} terms to define: {:?}", terms.len(), terms);

        if let Some(client) = &self.gemini_client {
            self.generate_definitions_with_llm(client, &terms).await
        } else {
            // Fallback: generate placeholder definitions
            Ok(self.generate_placeholder_definitions(&terms))
        }
    }

    /// Generate definitions using LLM.
    async fn generate_definitions_with_llm(
        &self,
        client: &GeminiClient,
        terms: &[String],
    ) -> Result<Vec<Definition>> {
        let terms_list = terms
            .iter()
            .take(self.config.max_definitions)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");

        let prompt = format!(
            "Define the following technical terms concisely:\n\n{}\n\n\
             For each term, provide:\n\
             1. A clear definition (1-2 sentences)\n\
             2. Category (e.g., Security, Database, Protocol)\n\
             3. Related terms\n\n\
             Format as JSON array with objects containing: term, definition, category, related_terms",
            terms_list
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

        self.parse_definitions_response(&response_text)
    }

    /// Parse LLM response into Definition structs.
    fn parse_definitions_response(&self, response: &str) -> Result<Vec<Definition>> {
        // Try to parse as JSON
        if let Ok(definitions) = serde_json::from_str::<Vec<serde_json::Value>>(response) {
            return Ok(definitions
                .into_iter()
                .filter_map(|v| self.value_to_definition(v))
                .take(self.config.max_definitions)
                .collect());
        }

        // Try to extract JSON from markdown code block
        let json_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
        if let Some(re) = json_re {
            if let Some(caps) = re.captures(response) {
                if let Some(json_text) = caps.get(1) {
                    if let Ok(definitions) = serde_json::from_str::<Vec<serde_json::Value>>(
                        json_text.as_str().trim(),
                    ) {
                        return Ok(definitions
                            .into_iter()
                            .filter_map(|v| self.value_to_definition(v))
                            .take(self.config.max_definitions)
                            .collect());
                    }
                }
            }
        }

        // Fallback: parse as text
        warn!("Could not parse definitions as JSON, using fallback");
        Ok(Vec::new())
    }

    /// Convert JSON value to Definition.
    fn value_to_definition(&self, value: serde_json::Value) -> Option<Definition> {
        let term = value.get("term")?.as_str()?.to_string();
        let definition = value.get("definition")?.as_str()?.to_string();

        let mut def = Definition::new(term, definition);

        if let Some(category) = value.get("category").and_then(|v| v.as_str()) {
            def = def.with_category(category);
        }

        if let Some(related) = value.get("related_terms").and_then(|v| v.as_array()) {
            let related_terms: Vec<String> = related
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            def = def.with_related_terms(related_terms);
        }

        Some(def)
    }

    /// Generate placeholder definitions without LLM.
    fn generate_placeholder_definitions(&self, terms: &[String]) -> Vec<Definition> {
        terms
            .iter()
            .take(self.config.max_definitions)
            .map(|term| {
                Definition::new(
                    term.clone(),
                    format!("A technical term related to software development: {}", term),
                )
            })
            .collect()
    }

    // =========================================================================
    // EXPANSION IMPLEMENTATIONS
    // =========================================================================

    /// Expand content with examples.
    async fn expand_with_examples(&self, content: &str) -> Result<ExpandedContent> {
        let examples = self.add_examples(content).await?;

        let mut expanded_content = content.to_string();
        if !examples.is_empty() {
            expanded_content.push_str("\n\n## Examples\n\n");
            for example in &examples {
                expanded_content.push_str(&example.to_markdown());
                expanded_content.push('\n');
            }
        }

        let confidence = if examples.is_empty() { 0.3 } else { 0.8 };

        Ok(ExpandedContent {
            original: content.to_string(),
            content: expanded_content,
            expansion_type: ExpansionType::Examples,
            examples,
            definitions: Vec::new(),
            alternatives: Vec::new(),
            context: None,
            extracted_terms: Vec::new(),
            duration_ms: 0,
            confidence,
        })
    }

    /// Expand content with definitions.
    async fn expand_with_definitions(&self, content: &str) -> Result<ExpandedContent> {
        let definitions = self.define_terms(content).await?;

        let mut expanded_content = content.to_string();
        if !definitions.is_empty() {
            expanded_content.push_str("\n\n## Glossary\n\n");
            for def in &definitions {
                expanded_content.push_str(&def.to_markdown());
            }
        }

        let confidence = if definitions.is_empty() { 0.3 } else { 0.8 };

        Ok(ExpandedContent {
            original: content.to_string(),
            content: expanded_content,
            expansion_type: ExpansionType::Definitions,
            examples: Vec::new(),
            definitions,
            alternatives: Vec::new(),
            context: None,
            extracted_terms: Vec::new(),
            duration_ms: 0,
            confidence,
        })
    }

    /// Expand content with context.
    async fn expand_with_context(&self, content: &str) -> Result<ExpandedContent> {
        let context = if let Some(client) = &self.gemini_client {
            self.generate_context_with_llm(client, content).await?
        } else {
            Some(ContextInfo::new(
                "Background",
                format!("Background information for: {}", content),
            ))
        };

        let mut expanded_content = content.to_string();
        if let Some(ref ctx) = context {
            expanded_content.push_str("\n\n");
            expanded_content.push_str(&ctx.to_markdown());
        }

        let confidence = if context.is_some() { 0.7 } else { 0.3 };

        Ok(ExpandedContent {
            original: content.to_string(),
            content: expanded_content,
            expansion_type: ExpansionType::Context,
            examples: Vec::new(),
            definitions: Vec::new(),
            alternatives: Vec::new(),
            context,
            extracted_terms: Vec::new(),
            duration_ms: 0,
            confidence,
        })
    }

    /// Expand content with alternatives.
    async fn expand_with_alternatives(&self, content: &str) -> Result<ExpandedContent> {
        let alternatives = if let Some(client) = &self.gemini_client {
            self.generate_alternatives_with_llm(client, content).await?
        } else {
            vec![Alternative::new(
                "Default Approach",
                format!("Standard implementation for: {}", content),
            )]
        };

        let mut expanded_content = content.to_string();
        if !alternatives.is_empty() {
            expanded_content.push_str("\n\n## Alternatives\n\n");
            for alt in &alternatives {
                expanded_content.push_str(&alt.to_markdown());
            }
        }

        let confidence = if alternatives.is_empty() { 0.3 } else { 0.7 };

        Ok(ExpandedContent {
            original: content.to_string(),
            content: expanded_content,
            expansion_type: ExpansionType::Alternatives,
            examples: Vec::new(),
            definitions: Vec::new(),
            alternatives,
            context: None,
            extracted_terms: Vec::new(),
            duration_ms: 0,
            confidence,
        })
    }

    /// Generate context using LLM.
    async fn generate_context_with_llm(
        &self,
        client: &GeminiClient,
        content: &str,
    ) -> Result<Option<ContextInfo>> {
        let prompt = format!(
            "Provide background context for the following content:\n\n{}\n\n\
             Include:\n\
             1. Why this topic matters\n\
             2. Brief history or origin\n\
             3. Related concepts\n\
             4. Recommended resources\n\n\
             Format as JSON with: topic, background, history, related_concepts (array), resources (array)",
            content
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

        self.parse_context_response(&response_text)
    }

    /// Parse context response.
    fn parse_context_response(&self, response: &str) -> Result<Option<ContextInfo>> {
        // Try to parse as JSON
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(response) {
            return Ok(self.value_to_context(&value));
        }

        // Try to extract JSON from code block
        let json_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
        if let Some(re) = json_re {
            if let Some(caps) = re.captures(response) {
                if let Some(json_text) = caps.get(1) {
                    if let Ok(value) =
                        serde_json::from_str::<serde_json::Value>(json_text.as_str().trim())
                    {
                        return Ok(self.value_to_context(&value));
                    }
                }
            }
        }

        // Fallback: use response as background
        Ok(Some(ContextInfo::new("Background", response)))
    }

    /// Convert JSON value to ContextInfo.
    fn value_to_context(&self, value: &serde_json::Value) -> Option<ContextInfo> {
        let topic = value
            .get("topic")
            .and_then(|v| v.as_str())
            .unwrap_or("Background")
            .to_string();

        let background = value
            .get("background")
            .and_then(|v| v.as_str())?
            .to_string();

        let mut ctx = ContextInfo::new(topic, background);

        if let Some(history) = value.get("history").and_then(|v| v.as_str()) {
            ctx.history = Some(history.to_string());
        }

        if let Some(related) = value.get("related_concepts").and_then(|v| v.as_array()) {
            ctx.related_concepts = related
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        if let Some(resources) = value.get("resources").and_then(|v| v.as_array()) {
            ctx.resources = resources
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        Some(ctx)
    }

    /// Generate alternatives using LLM.
    async fn generate_alternatives_with_llm(
        &self,
        client: &GeminiClient,
        content: &str,
    ) -> Result<Vec<Alternative>> {
        let prompt = format!(
            "Suggest {} alternative approaches for:\n\n{}\n\n\
             For each alternative, provide:\n\
             1. Name of the approach\n\
             2. Description\n\
             3. Pros (as array)\n\
             4. Cons (as array)\n\
             5. When to use this approach\n\n\
             Format as JSON array with: name, description, pros, cons, use_when",
            self.config.max_alternatives, content
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

        self.parse_alternatives_response(&response_text)
    }

    /// Parse alternatives response.
    fn parse_alternatives_response(&self, response: &str) -> Result<Vec<Alternative>> {
        // Try to parse as JSON
        if let Ok(alternatives) = serde_json::from_str::<Vec<serde_json::Value>>(response) {
            return Ok(alternatives
                .into_iter()
                .filter_map(|v| self.value_to_alternative(&v))
                .take(self.config.max_alternatives)
                .collect());
        }

        // Try to extract JSON from code block
        let json_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
        if let Some(re) = json_re {
            if let Some(caps) = re.captures(response) {
                if let Some(json_text) = caps.get(1) {
                    if let Ok(alternatives) = serde_json::from_str::<Vec<serde_json::Value>>(
                        json_text.as_str().trim(),
                    ) {
                        return Ok(alternatives
                            .into_iter()
                            .filter_map(|v| self.value_to_alternative(&v))
                            .take(self.config.max_alternatives)
                            .collect());
                    }
                }
            }
        }

        Ok(Vec::new())
    }

    /// Convert JSON value to Alternative.
    fn value_to_alternative(&self, value: &serde_json::Value) -> Option<Alternative> {
        let name = value.get("name")?.as_str()?.to_string();
        let description = value.get("description")?.as_str()?.to_string();

        let mut alt = Alternative::new(name, description);

        if let Some(pros) = value.get("pros").and_then(|v| v.as_array()) {
            alt.pros = pros
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        if let Some(cons) = value.get("cons").and_then(|v| v.as_array()) {
            alt.cons = cons
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        if let Some(use_when) = value.get("use_when").and_then(|v| v.as_str()) {
            alt.use_when = Some(use_when.to_string());
        }

        Some(alt)
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Extract terms from content (public wrapper for term extraction).
    pub fn extract_terms(&self, content: &str) -> Vec<String> {
        self.term_extractor.extract(content)
    }

    /// Check if content contains code blocks.
    pub fn has_code(&self, content: &str) -> bool {
        self.term_extractor.has_code_blocks(content)
    }

    /// Detect the programming language in content.
    pub fn detect_language(&self, content: &str) -> Option<String> {
        self.term_extractor.detect_language(content)
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom ExpanderAgent configuration.
#[derive(Debug, Default)]
pub struct ExpanderAgentBuilder {
    config: Option<ExpanderConfig>,
    gemini_client: Option<Arc<GeminiClient>>,
}

impl ExpanderAgentBuilder {
    /// Set the configuration.
    pub fn config(mut self, config: ExpanderConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the Gemini client.
    pub fn gemini_client(mut self, client: Arc<GeminiClient>) -> Self {
        self.gemini_client = Some(client);
        self
    }

    /// Set max examples.
    pub fn max_examples(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_examples = max;
        self.config = Some(config);
        self
    }

    /// Set max definitions.
    pub fn max_definitions(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_definitions = max;
        self.config = Some(config);
        self
    }

    /// Build the ExpanderAgent.
    pub fn build(self) -> ExpanderAgent {
        ExpanderAgent {
            config: self.config.unwrap_or_default(),
            term_extractor: TermExtractor::default(),
            gemini_client: self.gemini_client,
        }
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ExpansionType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_expansion_type_display() {
        assert_eq!(ExpansionType::Examples.to_string(), "Examples");
        assert_eq!(ExpansionType::Definitions.to_string(), "Definitions");
        assert_eq!(ExpansionType::Context.to_string(), "Context");
        assert_eq!(ExpansionType::Alternatives.to_string(), "Alternatives");
    }

    #[test]
    fn test_expansion_type_from_str() {
        assert_eq!(
            "examples".parse::<ExpansionType>().unwrap(),
            ExpansionType::Examples
        );
        assert_eq!(
            "definitions".parse::<ExpansionType>().unwrap(),
            ExpansionType::Definitions
        );
        assert_eq!(
            "context".parse::<ExpansionType>().unwrap(),
            ExpansionType::Context
        );
        assert_eq!(
            "alternatives".parse::<ExpansionType>().unwrap(),
            ExpansionType::Alternatives
        );
    }

    #[test]
    fn test_expansion_type_from_str_aliases() {
        assert_eq!(
            "demo".parse::<ExpansionType>().unwrap(),
            ExpansionType::Examples
        );
        assert_eq!(
            "define".parse::<ExpansionType>().unwrap(),
            ExpansionType::Definitions
        );
        assert_eq!(
            "background".parse::<ExpansionType>().unwrap(),
            ExpansionType::Context
        );
        assert_eq!(
            "options".parse::<ExpansionType>().unwrap(),
            ExpansionType::Alternatives
        );
    }

    #[test]
    fn test_expansion_type_from_str_invalid() {
        let result = "invalid".parse::<ExpansionType>();
        assert!(result.is_err());
    }

    #[test]
    fn test_expansion_type_all() {
        let all = ExpansionType::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&ExpansionType::Examples));
        assert!(all.contains(&ExpansionType::Definitions));
        assert!(all.contains(&ExpansionType::Context));
        assert!(all.contains(&ExpansionType::Alternatives));
    }

    #[test]
    fn test_expansion_type_instruction() {
        let examples_instruction = ExpansionType::Examples.instruction();
        assert!(examples_instruction.contains("code examples"));

        let definitions_instruction = ExpansionType::Definitions.instruction();
        assert!(definitions_instruction.contains("technical terms"));
    }

    // -------------------------------------------------------------------------
    // Example Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_example_new() {
        let example = Example::new("Test Example", "println!(\"Hello\");");
        assert_eq!(example.title, "Test Example");
        assert_eq!(example.code, "println!(\"Hello\");");
        assert!(example.language.is_none());
        assert!(example.explanation.is_none());
    }

    #[test]
    fn test_example_builder() {
        let example = Example::new("Rust Example", "fn main() {}")
            .with_language("rust")
            .with_explanation("A simple main function")
            .with_complexity(ExampleComplexity::Beginner);

        assert_eq!(example.language, Some("rust".to_string()));
        assert_eq!(
            example.explanation,
            Some("A simple main function".to_string())
        );
        assert_eq!(example.complexity, ExampleComplexity::Beginner);
    }

    #[test]
    fn test_example_to_markdown() {
        let example = Example::new("Hello World", "println!(\"Hello\");")
            .with_language("rust")
            .with_explanation("Print greeting");

        let md = example.to_markdown();
        assert!(md.contains("### Hello World"));
        assert!(md.contains("```rust"));
        assert!(md.contains("println!"));
        assert!(md.contains("Print greeting"));
    }

    // -------------------------------------------------------------------------
    // Definition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_definition_new() {
        let def = Definition::new("OAuth", "Open Authorization protocol");
        assert_eq!(def.term, "OAuth");
        assert_eq!(def.definition, "Open Authorization protocol");
    }

    #[test]
    fn test_definition_builder() {
        let def = Definition::new("JWT", "JSON Web Token")
            .with_category("Security")
            .with_related_terms(vec!["OAuth".to_string(), "Bearer Token".to_string()]);

        assert_eq!(def.category, Some("Security".to_string()));
        assert_eq!(def.related_terms.len(), 2);
    }

    #[test]
    fn test_definition_to_markdown() {
        let def = Definition::new("API", "Application Programming Interface")
            .with_category("General")
            .with_related_terms(vec!["REST".to_string()]);

        let md = def.to_markdown();
        assert!(md.contains("**API**"));
        assert!(md.contains("General"));
        assert!(md.contains("Application Programming Interface"));
        assert!(md.contains("REST"));
    }

    // -------------------------------------------------------------------------
    // Alternative Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alternative_new() {
        let alt = Alternative::new("Session-based Auth", "Traditional session cookies");
        assert_eq!(alt.name, "Session-based Auth");
        assert!(alt.pros.is_empty());
        assert!(alt.cons.is_empty());
    }

    #[test]
    fn test_alternative_builder() {
        let alt = Alternative::new("JWT Auth", "Token-based authentication")
            .with_pros(vec!["Stateless".to_string(), "Scalable".to_string()])
            .with_cons(vec!["Token size".to_string()]);

        assert_eq!(alt.pros.len(), 2);
        assert_eq!(alt.cons.len(), 1);
    }

    #[test]
    fn test_alternative_to_markdown() {
        let alt = Alternative::new("REST API", "RESTful architecture")
            .with_pros(vec!["Simple".to_string()])
            .with_cons(vec!["Over-fetching".to_string()]);

        let md = alt.to_markdown();
        assert!(md.contains("### REST API"));
        assert!(md.contains("**Pros:**"));
        assert!(md.contains("**Cons:**"));
    }

    // -------------------------------------------------------------------------
    // ExpandedContent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_expanded_content_new() {
        let content = ExpandedContent::new("original", "expanded");
        assert_eq!(content.original, "original");
        assert_eq!(content.content, "expanded");
        assert!(!content.is_substantial());
    }

    #[test]
    fn test_expanded_content_is_substantial() {
        let mut content = ExpandedContent::new("original", "expanded");
        assert!(!content.is_substantial());

        content
            .examples
            .push(Example::new("Test", "code"));
        assert!(content.is_substantial());
    }

    #[test]
    fn test_expanded_content_summary() {
        let mut content = ExpandedContent::new("original", "expanded");
        assert_eq!(content.summary(), "No expansions");

        content
            .examples
            .push(Example::new("Test", "code"));
        content.definitions.push(Definition::new("Term", "Def"));
        let summary = content.summary();
        assert!(summary.contains("1 examples"));
        assert!(summary.contains("1 definitions"));
    }

    // -------------------------------------------------------------------------
    // ExpanderConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_expander_config_default() {
        let config = ExpanderConfig::default();
        assert_eq!(config.max_examples, 5);
        assert_eq!(config.max_definitions, 10);
        assert_eq!(config.max_alternatives, 4);
    }

    #[test]
    fn test_expander_config_fast() {
        let config = ExpanderConfig::fast();
        assert_eq!(config.max_examples, 2);
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_expander_config_thorough() {
        let config = ExpanderConfig::thorough();
        assert_eq!(config.max_examples, 10);
        assert_eq!(config.max_definitions, 20);
    }

    // -------------------------------------------------------------------------
    // TermExtractor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_term_extractor_technical_terms() {
        let extractor = TermExtractor::default();
        let content = "Implement OAuth2 with JWT tokens and use PostgreSQL database";

        let terms = extractor.extract(content);
        assert!(terms.iter().any(|t| t.contains("OAuth")));
        assert!(terms.iter().any(|t| t.contains("JWT")));
        assert!(terms.iter().any(|t| t.contains("PostgreSQL")));
    }

    #[test]
    fn test_term_extractor_snake_case() {
        let extractor = TermExtractor::default();
        let content = "Use api_key and access_token for authentication";

        let terms = extractor.extract(content);
        assert!(terms.contains(&"api_key".to_string()));
        assert!(terms.contains(&"access_token".to_string()));
    }

    #[test]
    fn test_term_extractor_domain_terms() {
        let extractor = TermExtractor::default();
        let content = "Implement authentication and middleware with caching";

        let terms = extractor.extract(content);
        assert!(terms.contains(&"authentication".to_string()));
        assert!(terms.contains(&"middleware".to_string()));
        assert!(terms.contains(&"caching".to_string()));
    }

    #[test]
    fn test_term_extractor_has_code_blocks() {
        let extractor = TermExtractor::default();

        let with_code = "Some text ```rust\nfn main() {}\n```";
        assert!(extractor.has_code_blocks(with_code));

        let without_code = "Just plain text here";
        assert!(!extractor.has_code_blocks(without_code));
    }

    #[test]
    fn test_term_extractor_detect_language() {
        let extractor = TermExtractor::default();

        let rust_code = "```rust\nfn main() {}\n```";
        assert_eq!(extractor.detect_language(rust_code), Some("rust".to_string()));

        let rust_hints = "fn main() { let x = 5; }";
        assert_eq!(
            extractor.detect_language(rust_hints),
            Some("rust".to_string())
        );

        let python_hints = "def main():\n    import os";
        assert_eq!(
            extractor.detect_language(python_hints),
            Some("python".to_string())
        );
    }

    // -------------------------------------------------------------------------
    // ExpanderAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_expander_agent_new() {
        let agent = ExpanderAgent::new();
        assert_eq!(agent.config.max_examples, 5);
        assert!(agent.gemini_client.is_none());
    }

    #[test]
    fn test_expander_agent_builder() {
        let agent = ExpanderAgent::builder()
            .max_examples(10)
            .max_definitions(15)
            .build();

        assert_eq!(agent.config.max_examples, 10);
        assert_eq!(agent.config.max_definitions, 15);
    }

    #[test]
    fn test_expander_agent_extract_terms() {
        let agent = ExpanderAgent::new();
        let terms = agent.extract_terms("Use OAuth2 for authentication with JWT");

        assert!(!terms.is_empty());
        assert!(terms.iter().any(|t| t.contains("OAuth")));
    }

    #[test]
    fn test_expander_agent_has_code() {
        let agent = ExpanderAgent::new();

        assert!(agent.has_code("```rust\ncode\n```"));
        assert!(!agent.has_code("no code here"));
    }

    #[test]
    fn test_expander_agent_detect_language() {
        let agent = ExpanderAgent::new();

        assert_eq!(
            agent.detect_language("```python\nprint()\n```"),
            Some("python".to_string())
        );
    }

    #[tokio::test]
    async fn test_expander_agent_expand_empty_content() {
        let agent = ExpanderAgent::new();
        let result = agent.expand("", ExpansionType::Examples).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_expander_agent_expand_without_llm() {
        let agent = ExpanderAgent::new();
        let result = agent
            .expand("Implement OAuth2 authentication", ExpansionType::Examples)
            .await;

        assert!(result.is_ok());
        let expanded = result.unwrap();
        assert_eq!(expanded.expansion_type, ExpansionType::Examples);
        assert!(!expanded.extracted_terms.is_empty());
    }

    #[tokio::test]
    async fn test_expander_agent_add_examples_empty() {
        let agent = ExpanderAgent::new();
        let examples = agent.add_examples("").await.unwrap();
        assert!(examples.is_empty());
    }

    #[tokio::test]
    async fn test_expander_agent_define_terms_empty() {
        let agent = ExpanderAgent::new();
        let definitions = agent.define_terms("").await.unwrap();
        assert!(definitions.is_empty());
    }

    #[tokio::test]
    async fn test_expander_agent_define_terms_no_terms() {
        let agent = ExpanderAgent::new();
        // Content with no technical terms
        let definitions = agent.define_terms("the and for").await.unwrap();
        assert!(definitions.is_empty());
    }

    // -------------------------------------------------------------------------
    // JSON Parsing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_examples_json() {
        let agent = ExpanderAgent::new();
        let json = r#"[
            {"title": "Test", "code": "fn main() {}", "language": "rust", "complexity": "beginner"},
            {"title": "Test2", "code": "print()", "language": "python", "complexity": "advanced"}
        ]"#;

        let examples = agent.parse_examples_response(json).unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].title, "Test");
        assert_eq!(examples[0].complexity, ExampleComplexity::Beginner);
        assert_eq!(examples[1].complexity, ExampleComplexity::Advanced);
    }

    #[test]
    fn test_parse_examples_code_block() {
        let agent = ExpanderAgent::new();
        let response = "Here's an example:\n\n```rust\nfn main() {\n    println!(\"Hello\");\n}\n```";

        let examples = agent.parse_examples_response(response).unwrap();
        assert!(!examples.is_empty());
        assert!(examples[0].code.contains("println"));
    }

    #[test]
    fn test_parse_definitions_json() {
        let agent = ExpanderAgent::new();
        let json = r#"[
            {"term": "OAuth", "definition": "Open Authorization", "category": "Security"},
            {"term": "JWT", "definition": "JSON Web Token", "related_terms": ["OAuth"]}
        ]"#;

        let definitions = agent.parse_definitions_response(json).unwrap();
        assert_eq!(definitions.len(), 2);
        assert_eq!(definitions[0].term, "OAuth");
        assert_eq!(definitions[0].category, Some("Security".to_string()));
    }

    #[test]
    fn test_parse_alternatives_json() {
        let agent = ExpanderAgent::new();
        let json = r#"[
            {
                "name": "Session Auth",
                "description": "Cookie-based sessions",
                "pros": ["Simple", "Familiar"],
                "cons": ["Stateful"],
                "use_when": "Traditional web apps"
            }
        ]"#;

        let alternatives = agent.parse_alternatives_response(json).unwrap();
        assert_eq!(alternatives.len(), 1);
        assert_eq!(alternatives[0].name, "Session Auth");
        assert_eq!(alternatives[0].pros.len(), 2);
        assert_eq!(alternatives[0].use_when, Some("Traditional web apps".to_string()));
    }

    // -------------------------------------------------------------------------
    // ExampleComplexity Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_example_complexity_display() {
        assert_eq!(ExampleComplexity::Beginner.to_string(), "beginner");
        assert_eq!(ExampleComplexity::Intermediate.to_string(), "intermediate");
        assert_eq!(ExampleComplexity::Advanced.to_string(), "advanced");
    }

    #[test]
    fn test_example_complexity_default() {
        assert_eq!(ExampleComplexity::default(), ExampleComplexity::Intermediate);
    }

    // -------------------------------------------------------------------------
    // ContextInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_context_info_new() {
        let ctx = ContextInfo::new("OAuth", "Background on OAuth protocol");
        assert_eq!(ctx.topic, "OAuth");
        assert!(ctx.history.is_none());
        assert!(ctx.related_concepts.is_empty());
    }

    #[test]
    fn test_context_info_to_markdown() {
        let mut ctx = ContextInfo::new("API Design", "RESTful API principles");
        ctx.history = Some("Started in 2000".to_string());
        ctx.related_concepts = vec!["HTTP".to_string(), "REST".to_string()];

        let md = ctx.to_markdown();
        assert!(md.contains("## Background: API Design"));
        assert!(md.contains("### History"));
        assert!(md.contains("### Related Concepts"));
    }
}
