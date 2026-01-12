//! Enricher Agent module for Project Panpsychism.
//!
//! Implements "The Content Amplifier" — an agent that enriches prompts with
//! additional context, citations, examples, and cross-references. Like a scholar
//! who weaves threads of knowledge into a rich tapestry, the Enricher Agent takes
//! prompts and amplifies them with supporting material.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In the arcane arts of knowledge synthesis, the Enricher Agent serves as
//! the **Content Amplifier** — a specialist in the magical art of enrichment:
//!
//! - **Citations** are references to authoritative sources that ground the spell
//! - **Examples** are demonstration spells, showing magic in action
//! - **Cross-References** are connections to related incantations in the grimoire
//! - **Metadata** is the ambient magical field that gives context to the work
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Drive to enrich understanding through comprehensive support
//! - **RATIO**: Logical selection of relevant citations and cross-references
//! - **LAETITIA**: Joy through depth and illuminating connections
//! - **NATURA**: Natural flow from core concepts to supporting evidence
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::enricher::{EnricherAgent, EnrichmentStrategy};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let agent = EnricherAgent::new();
//!
//!     let prompt = "Implement OAuth2 authentication with JWT tokens.";
//!     let enriched = agent.enrich(prompt, EnrichmentStrategy::Comprehensive).await?;
//!
//!     println!("Enriched prompt: {}", enriched.content);
//!     println!("Added {} citations", enriched.citations.len());
//!     println!("Added {} cross-references", enriched.cross_references.len());
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
// ENRICHMENT STRATEGY
// =============================================================================

/// Strategy for content enrichment.
///
/// Each strategy represents a different approach to enrichment:
///
/// - **Academic**: Focus on citations and authoritative sources
/// - **Practical**: Focus on examples and implementation guidance
/// - **Comprehensive**: Full enrichment with all elements
/// - **Minimal**: Light enrichment with only essential additions
///
/// # The Scholar's Choice
///
/// Like a scholar choosing the depth of research, the enrichment strategy
/// determines how much supporting material is gathered for the prompt.
/// Academic enrichment prioritizes citations and references.
/// Practical enrichment prioritizes examples and implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum EnrichmentStrategy {
    /// Focus on academic citations and authoritative sources.
    /// Best for research-oriented prompts and documentation.
    Academic,

    /// Focus on practical examples and implementation guidance.
    /// Best for coding and hands-on tasks.
    Practical,

    /// Full enrichment with citations, examples, and cross-references.
    /// Best when comprehensive support is needed.
    #[default]
    Comprehensive,

    /// Light enrichment with only essential additions.
    /// Best for simple prompts that need minimal enhancement.
    Minimal,
}

impl EnrichmentStrategy {
    /// Get all enrichment strategies.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Academic,
            Self::Practical,
            Self::Comprehensive,
            Self::Minimal,
        ]
    }

    /// Get the description for this strategy.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Academic => "Focus on citations and authoritative sources",
            Self::Practical => "Focus on examples and implementation guidance",
            Self::Comprehensive => "Full enrichment with all elements",
            Self::Minimal => "Light enrichment with essential additions only",
        }
    }

    /// Get the label for this strategy.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Academic => "Academic",
            Self::Practical => "Practical",
            Self::Comprehensive => "Comprehensive",
            Self::Minimal => "Minimal",
        }
    }

    /// Check if citations should be included.
    pub fn include_citations(&self) -> bool {
        matches!(self, Self::Academic | Self::Comprehensive)
    }

    /// Check if examples should be included.
    pub fn include_examples(&self) -> bool {
        matches!(self, Self::Practical | Self::Comprehensive)
    }

    /// Check if cross-references should be included.
    pub fn include_cross_refs(&self) -> bool {
        matches!(self, Self::Academic | Self::Comprehensive)
    }

    /// Get the maximum number of citations for this strategy.
    pub fn max_citations(&self) -> usize {
        match self {
            Self::Academic => 10,
            Self::Practical => 3,
            Self::Comprehensive => 8,
            Self::Minimal => 2,
        }
    }

    /// Get the maximum number of examples for this strategy.
    pub fn max_examples(&self) -> usize {
        match self {
            Self::Academic => 2,
            Self::Practical => 6,
            Self::Comprehensive => 5,
            Self::Minimal => 1,
        }
    }

    /// Get the maximum number of cross-references for this strategy.
    pub fn max_cross_refs(&self) -> usize {
        match self {
            Self::Academic => 8,
            Self::Practical => 3,
            Self::Comprehensive => 6,
            Self::Minimal => 2,
        }
    }
}

impl std::fmt::Display for EnrichmentStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl std::str::FromStr for EnrichmentStrategy {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "academic" | "research" | "scholarly" => Ok(Self::Academic),
            "practical" | "hands-on" | "implementation" => Ok(Self::Practical),
            "comprehensive" | "full" | "complete" => Ok(Self::Comprehensive),
            "minimal" | "light" | "simple" | "basic" => Ok(Self::Minimal),
            _ => Err(Error::Config(format!(
                "Unknown enrichment strategy: '{}'. Valid strategies: academic, practical, comprehensive, minimal",
                s
            ))),
        }
    }
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// A citation from an authoritative source.
///
/// Citations ground the prompt in established knowledge,
/// providing evidence and authority to the content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// The source of the citation (e.g., "RFC 6749", "Rust Book Chapter 4").
    pub source: String,

    /// Relevance score (0.0 to 1.0), higher means more relevant.
    pub relevance: f64,

    /// Optional quote from the source.
    pub quote: Option<String>,

    /// URL or reference link (optional).
    pub url: Option<String>,

    /// Author or organization (optional).
    pub author: Option<String>,

    /// Year or date of the source (optional).
    pub year: Option<String>,
}

impl Citation {
    /// Create a new citation with just a source.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            relevance: 0.5,
            quote: None,
            url: None,
            author: None,
            year: None,
        }
    }

    /// Set the relevance score.
    pub fn with_relevance(mut self, relevance: f64) -> Self {
        self.relevance = relevance.clamp(0.0, 1.0);
        self
    }

    /// Set a quote from the source.
    pub fn with_quote(mut self, quote: impl Into<String>) -> Self {
        self.quote = Some(quote.into());
        self
    }

    /// Set the URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set the author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the year.
    pub fn with_year(mut self, year: impl Into<String>) -> Self {
        self.year = Some(year.into());
        self
    }

    /// Check if this is a high-relevance citation (>= 0.7).
    pub fn is_highly_relevant(&self) -> bool {
        self.relevance >= 0.7
    }

    /// Format the citation as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Source with optional author and year
        output.push_str("- ");
        if let Some(ref author) = self.author {
            output.push_str(author);
            output.push_str(". ");
        }
        output.push_str(&format!("**{}**", self.source));
        if let Some(ref year) = self.year {
            output.push_str(&format!(" ({})", year));
        }

        // URL if available
        if let Some(ref url) = self.url {
            output.push_str(&format!(" - [Link]({})", url));
        }

        // Quote if available
        if let Some(ref quote) = self.quote {
            output.push_str(&format!("\n  > \"{}\"", quote));
        }

        output.push('\n');
        output
    }

    /// Format as a brief inline citation.
    pub fn to_inline(&self) -> String {
        if let Some(ref author) = self.author {
            if let Some(ref year) = self.year {
                return format!("({}, {})", author, year);
            }
            return format!("({})", author);
        }
        format!("({})", self.source)
    }
}

impl std::fmt::Display for Citation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source)
    }
}

/// An enrichment example demonstrating the concept.
///
/// Examples show concrete applications of the prompt's concepts,
/// making abstract ideas tangible and actionable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentExample {
    /// Title or description of the example.
    pub title: String,

    /// The example content (code, text, or explanation).
    pub content: String,

    /// Programming language (for code examples).
    pub language: Option<String>,

    /// Brief explanation of what the example demonstrates.
    pub explanation: Option<String>,

    /// Context for when this example applies.
    pub context: Option<String>,
}

impl EnrichmentExample {
    /// Create a new example.
    pub fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            language: None,
            explanation: None,
            context: None,
        }
    }

    /// Set the programming language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the explanation.
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }

    /// Set the context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Format the example as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!("### {}\n\n", self.title);

        if let Some(ref explanation) = self.explanation {
            output.push_str(explanation);
            output.push_str("\n\n");
        }

        if let Some(ref context) = self.context {
            output.push_str(&format!("*Context: {}*\n\n", context));
        }

        let lang = self.language.as_deref().unwrap_or("text");
        output.push_str(&format!("```{}\n{}\n```\n", lang, self.content));

        output
    }
}

impl std::fmt::Display for EnrichmentExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.title)
    }
}

/// A cross-reference to related content.
///
/// Cross-references connect the prompt to related concepts,
/// creating a web of knowledge that enriches understanding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRef {
    /// Title or name of the referenced content.
    pub title: String,

    /// Brief description of how it relates.
    pub description: String,

    /// Type of relationship (e.g., "prerequisite", "alternative", "extension").
    pub relationship: String,

    /// Relevance score (0.0 to 1.0).
    pub relevance: f64,

    /// Optional path or identifier to the referenced content.
    pub path: Option<String>,

    /// Optional tags for categorization.
    pub tags: Vec<String>,
}

impl CrossRef {
    /// Create a new cross-reference.
    pub fn new(
        title: impl Into<String>,
        description: impl Into<String>,
        relationship: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            relationship: relationship.into(),
            relevance: 0.5,
            path: None,
            tags: Vec::new(),
        }
    }

    /// Set the relevance score.
    pub fn with_relevance(mut self, relevance: f64) -> Self {
        self.relevance = relevance.clamp(0.0, 1.0);
        self
    }

    /// Set the path.
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add a tag.
    pub fn add_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Check if this is a high-relevance cross-reference (>= 0.7).
    pub fn is_highly_relevant(&self) -> bool {
        self.relevance >= 0.7
    }

    /// Format the cross-reference as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!(
            "- **{}** _{}_: {}\n",
            self.title, self.relationship, self.description
        );

        if let Some(ref path) = self.path {
            output.push_str(&format!("  - Path: `{}`\n", path));
        }

        if !self.tags.is_empty() {
            output.push_str(&format!("  - Tags: {}\n", self.tags.join(", ")));
        }

        output
    }
}

impl std::fmt::Display for CrossRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.title, self.relationship)
    }
}

/// Metadata about the enrichment process.
///
/// Captures information about how the enrichment was performed,
/// including timing, sources consulted, and confidence levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentMetadata {
    /// Time taken for enrichment in milliseconds.
    pub enrichment_time_ms: u64,

    /// Number of sources consulted.
    pub sources_consulted: usize,

    /// Overall confidence in the enrichment (0.0 to 1.0).
    pub confidence: f64,

    /// The strategy used for enrichment.
    pub strategy: String,

    /// Timestamp of enrichment (ISO 8601 format).
    pub timestamp: String,

    /// Version of the enricher agent.
    pub agent_version: String,

    /// Number of tokens used (if applicable).
    pub tokens_used: Option<usize>,

    /// Any warnings or notes from the enrichment process.
    pub notes: Vec<String>,
}

impl EnrichmentMetadata {
    /// Create new metadata with required fields.
    pub fn new(enrichment_time_ms: u64, strategy: impl Into<String>) -> Self {
        Self {
            enrichment_time_ms,
            sources_consulted: 0,
            confidence: 0.5,
            strategy: strategy.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            agent_version: env!("CARGO_PKG_VERSION").to_string(),
            tokens_used: None,
            notes: Vec::new(),
        }
    }

    /// Set sources consulted.
    pub fn with_sources_consulted(mut self, count: usize) -> Self {
        self.sources_consulted = count;
        self
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set tokens used.
    pub fn with_tokens_used(mut self, tokens: usize) -> Self {
        self.tokens_used = Some(tokens);
        self
    }

    /// Add a note.
    pub fn add_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Check if enrichment was high-confidence.
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.7
    }
}

impl Default for EnrichmentMetadata {
    fn default() -> Self {
        Self {
            enrichment_time_ms: 0,
            sources_consulted: 0,
            confidence: 0.5,
            strategy: EnrichmentStrategy::default().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            agent_version: env!("CARGO_PKG_VERSION").to_string(),
            tokens_used: None,
            notes: Vec::new(),
        }
    }
}

// =============================================================================
// ENRICHED PROMPT
// =============================================================================

/// The result of prompt enrichment.
///
/// Contains the enriched content along with all supporting material
/// gathered during the enrichment process.
#[derive(Debug, Clone)]
pub struct EnrichedPrompt {
    /// Original prompt before enrichment.
    pub original: String,

    /// Enriched content (may include inline citations).
    pub content: String,

    /// Citations gathered for this prompt.
    pub citations: Vec<Citation>,

    /// Examples demonstrating the concepts.
    pub examples: Vec<EnrichmentExample>,

    /// Cross-references to related content.
    pub cross_references: Vec<CrossRef>,

    /// Metadata about the enrichment process.
    pub metadata: EnrichmentMetadata,

    /// Key terms extracted from the prompt.
    pub key_terms: Vec<String>,

    /// Summary of what the prompt is about.
    pub summary: Option<String>,
}

impl EnrichedPrompt {
    /// Create a new enriched prompt with just the original and content.
    pub fn new(original: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            original: original.into(),
            content: content.into(),
            citations: Vec::new(),
            examples: Vec::new(),
            cross_references: Vec::new(),
            metadata: EnrichmentMetadata::default(),
            key_terms: Vec::new(),
            summary: None,
        }
    }

    /// Check if the enrichment produced substantial content.
    pub fn is_substantial(&self) -> bool {
        !self.citations.is_empty()
            || !self.examples.is_empty()
            || !self.cross_references.is_empty()
    }

    /// Get the number of enrichment elements added.
    pub fn enrichment_count(&self) -> usize {
        self.citations.len() + self.examples.len() + self.cross_references.len()
    }

    /// Get a summary of the enrichment.
    pub fn enrichment_summary(&self) -> String {
        let mut parts = Vec::new();

        if !self.citations.is_empty() {
            parts.push(format!("{} citations", self.citations.len()));
        }
        if !self.examples.is_empty() {
            parts.push(format!("{} examples", self.examples.len()));
        }
        if !self.cross_references.is_empty() {
            parts.push(format!("{} cross-references", self.cross_references.len()));
        }

        if parts.is_empty() {
            "No enrichments".to_string()
        } else {
            format!("Enriched with: {}", parts.join(", "))
        }
    }

    /// Get high-relevance citations only.
    pub fn high_relevance_citations(&self) -> Vec<&Citation> {
        self.citations.iter().filter(|c| c.is_highly_relevant()).collect()
    }

    /// Get high-relevance cross-references only.
    pub fn high_relevance_cross_refs(&self) -> Vec<&CrossRef> {
        self.cross_references.iter().filter(|c| c.is_highly_relevant()).collect()
    }

    /// Format the enriched prompt as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("# Enriched Prompt\n\n");

        // Summary if available
        if let Some(ref summary) = self.summary {
            output.push_str(&format!("**Summary:** {}\n\n", summary));
        }

        // Main content
        output.push_str("## Content\n\n");
        output.push_str(&self.content);
        output.push_str("\n\n---\n\n");

        // Citations
        if !self.citations.is_empty() {
            output.push_str("## Citations\n\n");
            for citation in &self.citations {
                output.push_str(&citation.to_markdown());
            }
            output.push('\n');
        }

        // Examples
        if !self.examples.is_empty() {
            output.push_str("## Examples\n\n");
            for example in &self.examples {
                output.push_str(&example.to_markdown());
                output.push('\n');
            }
        }

        // Cross-references
        if !self.cross_references.is_empty() {
            output.push_str("## Related Content\n\n");
            for cross_ref in &self.cross_references {
                output.push_str(&cross_ref.to_markdown());
            }
            output.push('\n');
        }

        // Key terms
        if !self.key_terms.is_empty() {
            output.push_str("## Key Terms\n\n");
            output.push_str(&self.key_terms.join(", "));
            output.push_str("\n\n");
        }

        // Metadata
        output.push_str("---\n\n");
        output.push_str(&format!(
            "*Enriched using {} strategy in {}ms (confidence: {:.0}%)*\n",
            self.metadata.strategy,
            self.metadata.enrichment_time_ms,
            self.metadata.confidence * 100.0
        ));

        output
    }
}

impl std::fmt::Display for EnrichedPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

// =============================================================================
// ENRICHER CONFIGURATION
// =============================================================================

/// Configuration for the Enricher Agent.
#[derive(Debug, Clone)]
pub struct EnricherConfig {
    /// Maximum number of citations to include.
    pub max_citations: usize,

    /// Maximum number of examples to include.
    pub max_examples: usize,

    /// Maximum number of cross-references to include.
    pub max_cross_refs: usize,

    /// Whether to include examples.
    pub include_examples: bool,

    /// Whether to include citations.
    pub include_citations: bool,

    /// Whether to include cross-references.
    pub include_cross_refs: bool,

    /// Minimum relevance threshold for including citations.
    pub min_citation_relevance: f64,

    /// Minimum relevance threshold for including cross-references.
    pub min_cross_ref_relevance: f64,

    /// Timeout in seconds for LLM requests.
    pub timeout_secs: u64,

    /// Maximum retries for LLM calls.
    pub max_retries: u32,

    /// Whether to generate a summary.
    pub generate_summary: bool,

    /// Whether to extract key terms.
    pub extract_key_terms: bool,
}

impl Default for EnricherConfig {
    fn default() -> Self {
        Self {
            max_citations: 8,
            max_examples: 5,
            max_cross_refs: 6,
            include_examples: true,
            include_citations: true,
            include_cross_refs: true,
            min_citation_relevance: 0.4,
            min_cross_ref_relevance: 0.4,
            timeout_secs: 60,
            max_retries: 3,
            generate_summary: true,
            extract_key_terms: true,
        }
    }
}

impl EnricherConfig {
    /// Create a fast configuration with fewer enrichments.
    pub fn fast() -> Self {
        Self {
            max_citations: 3,
            max_examples: 2,
            max_cross_refs: 2,
            include_examples: true,
            include_citations: true,
            include_cross_refs: false,
            min_citation_relevance: 0.6,
            min_cross_ref_relevance: 0.6,
            timeout_secs: 30,
            max_retries: 2,
            generate_summary: false,
            extract_key_terms: true,
        }
    }

    /// Create a thorough configuration with more enrichments.
    pub fn thorough() -> Self {
        Self {
            max_citations: 15,
            max_examples: 8,
            max_cross_refs: 10,
            include_examples: true,
            include_citations: true,
            include_cross_refs: true,
            min_citation_relevance: 0.3,
            min_cross_ref_relevance: 0.3,
            timeout_secs: 120,
            max_retries: 4,
            generate_summary: true,
            extract_key_terms: true,
        }
    }

    /// Create config from a strategy.
    pub fn from_strategy(strategy: EnrichmentStrategy) -> Self {
        Self {
            max_citations: strategy.max_citations(),
            max_examples: strategy.max_examples(),
            max_cross_refs: strategy.max_cross_refs(),
            include_examples: strategy.include_examples(),
            include_citations: strategy.include_citations(),
            include_cross_refs: strategy.include_cross_refs(),
            ..Default::default()
        }
    }
}

// =============================================================================
// TERM EXTRACTOR (INTERNAL)
// =============================================================================

/// Pattern for technical terms (CamelCase, snake_case, acronyms).
const TECHNICAL_TERM_PATTERN: &str =
    r"(?x)
    \b[A-Z][a-zA-Z]+\d*\b |
    \b[a-z]+(?:_[a-z0-9]+)+\b |
    \b[A-Z]{2,}\d*s?\b
";

/// Pattern for domain-specific terms.
const DOMAIN_TERM_PATTERN: &str =
    r"(?xi)
    \b(?:
        authentication | authorization | middleware | endpoint |
        callback | webhook | payload | schema | migration |
        repository | singleton | factory | adapter | decorator |
        async | await | promise | future | stream | buffer |
        encryption | hashing | salting | tokenization |
        rate.?limiting | load.?balancing | caching | sharding |
        oauth | jwt | rest | api | http | https | ssl | tls
    )\b
";

/// Extracts technical terms from content.
#[derive(Debug, Clone)]
struct TermExtractor {
    technical_pattern: Regex,
    domain_pattern: Regex,
}

impl Default for TermExtractor {
    fn default() -> Self {
        Self {
            technical_pattern: Regex::new(TECHNICAL_TERM_PATTERN)
                .expect("Invalid technical term pattern"),
            domain_pattern: Regex::new(DOMAIN_TERM_PATTERN)
                .expect("Invalid domain term pattern"),
        }
    }
}

impl TermExtractor {
    /// Extract all technical terms from content.
    pub fn extract(&self, content: &str) -> Vec<String> {
        let mut terms = HashSet::new();

        // Extract technical terms
        for cap in self.technical_pattern.find_iter(content) {
            let term = cap.as_str().to_string();
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

    /// Check if a term is valid.
    fn is_valid_term(&self, term: &str) -> bool {
        if term.len() < 2 {
            return false;
        }

        let common_words = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "his", "how",
            "its", "may", "new", "now", "old", "see", "two", "way", "who",
            "implement", "with", "tokens", "using", "from", "this", "that",
            "when", "where", "which", "what", "they", "them", "then", "than",
        ];

        !common_words.contains(&term.to_lowercase().as_str())
    }
}

// =============================================================================
// ENRICHER AGENT
// =============================================================================

/// The Content Amplifier — Agent 21 of Project Panpsychism.
///
/// This agent enriches prompts with citations, examples, and cross-references,
/// amplifying the content with supporting material. Like a master researcher
/// who gathers evidence and connections, this agent transforms simple prompts
/// into richly annotated knowledge.
///
/// ## Capabilities
///
/// - **Citation Generation**: Adds relevant citations and authoritative sources
/// - **Example Creation**: Generates practical examples demonstrating concepts
/// - **Cross-Referencing**: Links to related content in the knowledge base
/// - **Term Extraction**: Identifies and highlights key technical terms
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::enricher::{EnricherAgent, EnrichmentStrategy};
///
/// let agent = EnricherAgent::new();
///
/// // Enrich with comprehensive strategy
/// let enriched = agent.enrich(
///     "Implement rate limiting for API",
///     EnrichmentStrategy::Comprehensive
/// ).await?;
///
/// // Access enrichment elements
/// for citation in &enriched.citations {
///     println!("Citation: {} (relevance: {:.2})", citation.source, citation.relevance);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EnricherAgent {
    /// Configuration for enrichment behavior.
    config: EnricherConfig,

    /// Term extraction engine.
    term_extractor: TermExtractor,

    /// Gemini client for LLM calls (optional).
    gemini_client: Option<Arc<GeminiClient>>,
}

impl Default for EnricherAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl EnricherAgent {
    /// Create a new Enricher Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: EnricherConfig::default(),
            term_extractor: TermExtractor::default(),
            gemini_client: None,
        }
    }

    /// Create a new Enricher Agent with a Gemini client.
    pub fn with_gemini(client: Arc<GeminiClient>) -> Self {
        Self {
            config: EnricherConfig::default(),
            term_extractor: TermExtractor::default(),
            gemini_client: Some(client),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> EnricherAgentBuilder {
        EnricherAgentBuilder::default()
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: EnricherConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &EnricherConfig {
        &self.config
    }

    // =========================================================================
    // MAIN ENRICHMENT METHOD
    // =========================================================================

    /// Enrich a prompt with the specified strategy.
    ///
    /// This is the primary method of the Content Amplifier, gathering
    /// supporting material based on the requested enrichment strategy.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt to enrich
    /// * `strategy` - The enrichment strategy to use
    ///
    /// # Returns
    ///
    /// An `EnrichedPrompt` containing the enriched content and metadata.
    ///
    /// # Errors
    ///
    /// Returns `Error::Synthesis` if enrichment fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = EnricherAgent::new();
    /// let enriched = agent.enrich(
    ///     "Implement OAuth2 authentication",
    ///     EnrichmentStrategy::Academic
    /// ).await?;
    /// ```
    pub async fn enrich(
        &self,
        prompt: &str,
        strategy: EnrichmentStrategy,
    ) -> Result<EnrichedPrompt> {
        let start = Instant::now();

        if prompt.trim().is_empty() {
            return Err(Error::Synthesis("Cannot enrich empty prompt".to_string()));
        }

        // Extract key terms
        let key_terms = self.term_extractor.extract(prompt);
        debug!("Extracted {} key terms from prompt", key_terms.len());

        // Create effective config based on strategy
        let effective_config = EnricherConfig::from_strategy(strategy);

        // Generate enriched content
        let mut enriched = EnrichedPrompt::new(prompt, prompt);
        enriched.key_terms = key_terms.clone();
        enriched.metadata = EnrichmentMetadata::new(0, strategy.to_string());

        // Generate citations if enabled
        if effective_config.include_citations {
            let citations = self.generate_citations(prompt, &key_terms, effective_config.max_citations).await?;
            enriched.citations = citations;
        }

        // Generate examples if enabled
        if effective_config.include_examples {
            let examples = self.generate_examples(prompt, effective_config.max_examples).await?;
            enriched.examples = examples;
        }

        // Generate cross-references if enabled
        if effective_config.include_cross_refs {
            let cross_refs = self.generate_cross_refs(prompt, &key_terms, effective_config.max_cross_refs).await?;
            enriched.cross_references = cross_refs;
        }

        // Generate summary if enabled
        if self.config.generate_summary {
            enriched.summary = self.generate_summary(prompt).await?;
        }

        // Update metadata
        let duration_ms = start.elapsed().as_millis() as u64;
        enriched.metadata = EnrichmentMetadata::new(duration_ms, strategy.to_string())
            .with_sources_consulted(enriched.citations.len() + enriched.cross_references.len())
            .with_confidence(self.calculate_confidence(&enriched));

        info!(
            "Enrichment complete: {} in {}ms",
            enriched.enrichment_summary(),
            duration_ms
        );

        Ok(enriched)
    }

    // =========================================================================
    // CITATION GENERATION
    // =========================================================================

    /// Generate citations for the prompt.
    async fn generate_citations(
        &self,
        prompt: &str,
        key_terms: &[String],
        max_citations: usize,
    ) -> Result<Vec<Citation>> {
        if let Some(client) = &self.gemini_client {
            self.generate_citations_with_llm(client, prompt, key_terms, max_citations).await
        } else {
            Ok(self.generate_placeholder_citations(key_terms, max_citations))
        }
    }

    /// Generate citations using LLM.
    async fn generate_citations_with_llm(
        &self,
        client: &GeminiClient,
        prompt: &str,
        key_terms: &[String],
        max_citations: usize,
    ) -> Result<Vec<Citation>> {
        let terms_list = key_terms.join(", ");

        let llm_prompt = format!(
            r#"Generate {} relevant citations for the following prompt and key terms.

Prompt: {}
Key Terms: {}

For each citation, provide:
1. Source name (e.g., "RFC 6749", "Rust Book Chapter 4", "MDN Web Docs")
2. Relevance score (0.0 to 1.0)
3. A brief quote if applicable
4. URL if available
5. Author if known
6. Year if known

Format as JSON array with objects containing: source, relevance, quote, url, author, year"#,
            max_citations, prompt, terms_list
        );

        let messages = vec![Message {
            role: "user".to_string(),
            content: llm_prompt,
        }];

        let response = client.chat_with_retry(messages).await?;

        let response_text = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        self.parse_citations_response(&response_text, max_citations)
    }

    /// Parse LLM response into Citation structs.
    fn parse_citations_response(&self, response: &str, max_citations: usize) -> Result<Vec<Citation>> {
        // Try to parse as JSON
        if let Ok(citations) = serde_json::from_str::<Vec<serde_json::Value>>(response) {
            return Ok(citations
                .into_iter()
                .filter_map(|v| self.value_to_citation(&v))
                .take(max_citations)
                .collect());
        }

        // Try to extract JSON from markdown code block
        let json_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
        if let Some(re) = json_re {
            if let Some(caps) = re.captures(response) {
                if let Some(json_text) = caps.get(1) {
                    if let Ok(citations) = serde_json::from_str::<Vec<serde_json::Value>>(
                        json_text.as_str().trim(),
                    ) {
                        return Ok(citations
                            .into_iter()
                            .filter_map(|v| self.value_to_citation(&v))
                            .take(max_citations)
                            .collect());
                    }
                }
            }
        }

        warn!("Could not parse citations as JSON, using fallback");
        Ok(Vec::new())
    }

    /// Convert JSON value to Citation.
    fn value_to_citation(&self, value: &serde_json::Value) -> Option<Citation> {
        let source = value.get("source")?.as_str()?.to_string();

        let mut citation = Citation::new(source);

        if let Some(relevance) = value.get("relevance").and_then(|v| v.as_f64()) {
            citation = citation.with_relevance(relevance);
        }

        if let Some(quote) = value.get("quote").and_then(|v| v.as_str()) {
            if !quote.is_empty() {
                citation = citation.with_quote(quote);
            }
        }

        if let Some(url) = value.get("url").and_then(|v| v.as_str()) {
            if !url.is_empty() {
                citation = citation.with_url(url);
            }
        }

        if let Some(author) = value.get("author").and_then(|v| v.as_str()) {
            if !author.is_empty() {
                citation = citation.with_author(author);
            }
        }

        if let Some(year) = value.get("year").and_then(|v| v.as_str()) {
            if !year.is_empty() {
                citation = citation.with_year(year);
            }
        }

        Some(citation)
    }

    /// Generate placeholder citations without LLM.
    fn generate_placeholder_citations(&self, key_terms: &[String], max: usize) -> Vec<Citation> {
        key_terms
            .iter()
            .take(max)
            .map(|term| {
                Citation::new(format!("{} Documentation", term))
                    .with_relevance(0.6)
            })
            .collect()
    }

    // =========================================================================
    // EXAMPLE GENERATION
    // =========================================================================

    /// Generate examples for the prompt.
    async fn generate_examples(
        &self,
        prompt: &str,
        max_examples: usize,
    ) -> Result<Vec<EnrichmentExample>> {
        if let Some(client) = &self.gemini_client {
            self.generate_examples_with_llm(client, prompt, max_examples).await
        } else {
            Ok(self.generate_placeholder_examples(prompt, max_examples))
        }
    }

    /// Generate examples using LLM.
    async fn generate_examples_with_llm(
        &self,
        client: &GeminiClient,
        prompt: &str,
        max_examples: usize,
    ) -> Result<Vec<EnrichmentExample>> {
        let llm_prompt = format!(
            r#"Generate {} practical examples for the following prompt.

Prompt: {}

For each example, provide:
1. A descriptive title
2. The example content (code or explanation)
3. Programming language (if code)
4. Brief explanation of what it demonstrates
5. Context for when it applies

Format as JSON array with objects containing: title, content, language, explanation, context"#,
            max_examples, prompt
        );

        let messages = vec![Message {
            role: "user".to_string(),
            content: llm_prompt,
        }];

        let response = client.chat_with_retry(messages).await?;

        let response_text = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        self.parse_examples_response(&response_text, max_examples)
    }

    /// Parse LLM response into EnrichmentExample structs.
    fn parse_examples_response(
        &self,
        response: &str,
        max_examples: usize,
    ) -> Result<Vec<EnrichmentExample>> {
        // Try to parse as JSON
        if let Ok(examples) = serde_json::from_str::<Vec<serde_json::Value>>(response) {
            return Ok(examples
                .into_iter()
                .filter_map(|v| self.value_to_example(&v))
                .take(max_examples)
                .collect());
        }

        // Try to extract JSON from code block
        let json_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
        if let Some(re) = json_re {
            if let Some(caps) = re.captures(response) {
                if let Some(json_text) = caps.get(1) {
                    if let Ok(examples) = serde_json::from_str::<Vec<serde_json::Value>>(
                        json_text.as_str().trim(),
                    ) {
                        return Ok(examples
                            .into_iter()
                            .filter_map(|v| self.value_to_example(&v))
                            .take(max_examples)
                            .collect());
                    }
                }
            }
        }

        Ok(Vec::new())
    }

    /// Convert JSON value to EnrichmentExample.
    fn value_to_example(&self, value: &serde_json::Value) -> Option<EnrichmentExample> {
        let title = value.get("title")?.as_str()?.to_string();
        let content = value.get("content")?.as_str()?.to_string();

        let mut example = EnrichmentExample::new(title, content);

        if let Some(language) = value.get("language").and_then(|v| v.as_str()) {
            if !language.is_empty() {
                example = example.with_language(language);
            }
        }

        if let Some(explanation) = value.get("explanation").and_then(|v| v.as_str()) {
            if !explanation.is_empty() {
                example = example.with_explanation(explanation);
            }
        }

        if let Some(context) = value.get("context").and_then(|v| v.as_str()) {
            if !context.is_empty() {
                example = example.with_context(context);
            }
        }

        Some(example)
    }

    /// Generate placeholder examples without LLM.
    fn generate_placeholder_examples(&self, prompt: &str, max: usize) -> Vec<EnrichmentExample> {
        if max == 0 {
            return Vec::new();
        }

        vec![EnrichmentExample::new(
            "Basic Example",
            format!("// Example for: {}", prompt),
        )
        .with_language("text")
        .with_explanation("A placeholder example demonstrating the concept")]
        .into_iter()
        .take(max)
        .collect()
    }

    // =========================================================================
    // CROSS-REFERENCE GENERATION
    // =========================================================================

    /// Generate cross-references for the prompt.
    async fn generate_cross_refs(
        &self,
        prompt: &str,
        key_terms: &[String],
        max_cross_refs: usize,
    ) -> Result<Vec<CrossRef>> {
        if let Some(client) = &self.gemini_client {
            self.generate_cross_refs_with_llm(client, prompt, key_terms, max_cross_refs).await
        } else {
            Ok(self.generate_placeholder_cross_refs(key_terms, max_cross_refs))
        }
    }

    /// Generate cross-references using LLM.
    async fn generate_cross_refs_with_llm(
        &self,
        client: &GeminiClient,
        prompt: &str,
        key_terms: &[String],
        max_cross_refs: usize,
    ) -> Result<Vec<CrossRef>> {
        let terms_list = key_terms.join(", ");

        let llm_prompt = format!(
            r#"Generate {} cross-references to related content for the following prompt.

Prompt: {}
Key Terms: {}

For each cross-reference, provide:
1. Title of the related content
2. Description of how it relates
3. Relationship type (prerequisite, alternative, extension, related, see-also)
4. Relevance score (0.0 to 1.0)
5. Tags for categorization

Format as JSON array with objects containing: title, description, relationship, relevance, tags"#,
            max_cross_refs, prompt, terms_list
        );

        let messages = vec![Message {
            role: "user".to_string(),
            content: llm_prompt,
        }];

        let response = client.chat_with_retry(messages).await?;

        let response_text = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        self.parse_cross_refs_response(&response_text, max_cross_refs)
    }

    /// Parse LLM response into CrossRef structs.
    fn parse_cross_refs_response(
        &self,
        response: &str,
        max_cross_refs: usize,
    ) -> Result<Vec<CrossRef>> {
        // Try to parse as JSON
        if let Ok(cross_refs) = serde_json::from_str::<Vec<serde_json::Value>>(response) {
            return Ok(cross_refs
                .into_iter()
                .filter_map(|v| self.value_to_cross_ref(&v))
                .take(max_cross_refs)
                .collect());
        }

        // Try to extract JSON from code block
        let json_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)```").ok();
        if let Some(re) = json_re {
            if let Some(caps) = re.captures(response) {
                if let Some(json_text) = caps.get(1) {
                    if let Ok(cross_refs) = serde_json::from_str::<Vec<serde_json::Value>>(
                        json_text.as_str().trim(),
                    ) {
                        return Ok(cross_refs
                            .into_iter()
                            .filter_map(|v| self.value_to_cross_ref(&v))
                            .take(max_cross_refs)
                            .collect());
                    }
                }
            }
        }

        Ok(Vec::new())
    }

    /// Convert JSON value to CrossRef.
    fn value_to_cross_ref(&self, value: &serde_json::Value) -> Option<CrossRef> {
        let title = value.get("title")?.as_str()?.to_string();
        let description = value.get("description")?.as_str()?.to_string();
        let relationship = value
            .get("relationship")
            .and_then(|v| v.as_str())
            .unwrap_or("related")
            .to_string();

        let mut cross_ref = CrossRef::new(title, description, relationship);

        if let Some(relevance) = value.get("relevance").and_then(|v| v.as_f64()) {
            cross_ref = cross_ref.with_relevance(relevance);
        }

        if let Some(tags) = value.get("tags").and_then(|v| v.as_array()) {
            let tag_list: Vec<String> = tags
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            cross_ref = cross_ref.with_tags(tag_list);
        }

        Some(cross_ref)
    }

    /// Generate placeholder cross-references without LLM.
    fn generate_placeholder_cross_refs(&self, key_terms: &[String], max: usize) -> Vec<CrossRef> {
        key_terms
            .iter()
            .take(max)
            .map(|term| {
                CrossRef::new(
                    format!("{} Guide", term),
                    format!("Related documentation for {}", term),
                    "related",
                )
                .with_relevance(0.5)
            })
            .collect()
    }

    // =========================================================================
    // SUMMARY GENERATION
    // =========================================================================

    /// Generate a summary of the prompt.
    async fn generate_summary(&self, prompt: &str) -> Result<Option<String>> {
        if let Some(client) = &self.gemini_client {
            let llm_prompt = format!(
                "Provide a one-sentence summary of what this prompt is asking for:\n\n{}",
                prompt
            );

            let messages = vec![Message {
                role: "user".to_string(),
                content: llm_prompt,
            }];

            match client.chat_with_retry(messages).await {
                Ok(response) => {
                    let summary = response
                        .choices
                        .first()
                        .map(|c| c.message.content.trim().to_string());
                    Ok(summary)
                }
                Err(_) => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Extract key terms from content.
    pub fn extract_terms(&self, content: &str) -> Vec<String> {
        self.term_extractor.extract(content)
    }

    /// Calculate confidence score based on enrichment results.
    fn calculate_confidence(&self, enriched: &EnrichedPrompt) -> f64 {
        if enriched.enrichment_count() == 0 {
            return 0.3;
        }

        let citation_avg = if enriched.citations.is_empty() {
            0.5
        } else {
            enriched.citations.iter().map(|c| c.relevance).sum::<f64>()
                / enriched.citations.len() as f64
        };

        let cross_ref_avg = if enriched.cross_references.is_empty() {
            0.5
        } else {
            enriched.cross_references.iter().map(|c| c.relevance).sum::<f64>()
                / enriched.cross_references.len() as f64
        };

        let example_bonus = if enriched.examples.is_empty() {
            0.0
        } else {
            0.1
        };

        let base = (citation_avg + cross_ref_avg) / 2.0 + example_bonus;
        base.clamp(0.0, 1.0)
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom EnricherAgent configuration.
#[derive(Debug, Default)]
pub struct EnricherAgentBuilder {
    config: Option<EnricherConfig>,
    gemini_client: Option<Arc<GeminiClient>>,
}

impl EnricherAgentBuilder {
    /// Set the configuration.
    pub fn config(mut self, config: EnricherConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the Gemini client.
    pub fn gemini_client(mut self, client: Arc<GeminiClient>) -> Self {
        self.gemini_client = Some(client);
        self
    }

    /// Set max citations.
    pub fn max_citations(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_citations = max;
        self.config = Some(config);
        self
    }

    /// Set max examples.
    pub fn max_examples(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_examples = max;
        self.config = Some(config);
        self
    }

    /// Set max cross-references.
    pub fn max_cross_refs(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_cross_refs = max;
        self.config = Some(config);
        self
    }

    /// Enable or disable examples.
    pub fn include_examples(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_examples = include;
        self.config = Some(config);
        self
    }

    /// Enable or disable citations.
    pub fn include_citations(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_citations = include;
        self.config = Some(config);
        self
    }

    /// Enable or disable cross-references.
    pub fn include_cross_refs(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_cross_refs = include;
        self.config = Some(config);
        self
    }

    /// Set minimum citation relevance.
    pub fn min_citation_relevance(mut self, min: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.min_citation_relevance = min;
        self.config = Some(config);
        self
    }

    /// Enable or disable summary generation.
    pub fn generate_summary(mut self, generate: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.generate_summary = generate;
        self.config = Some(config);
        self
    }

    /// Build the EnricherAgent.
    pub fn build(self) -> EnricherAgent {
        EnricherAgent {
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
    // EnrichmentStrategy Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enrichment_strategy_display() {
        assert_eq!(EnrichmentStrategy::Academic.to_string(), "Academic");
        assert_eq!(EnrichmentStrategy::Practical.to_string(), "Practical");
        assert_eq!(EnrichmentStrategy::Comprehensive.to_string(), "Comprehensive");
        assert_eq!(EnrichmentStrategy::Minimal.to_string(), "Minimal");
    }

    #[test]
    fn test_enrichment_strategy_from_str() {
        assert_eq!(
            "academic".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Academic
        );
        assert_eq!(
            "practical".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Practical
        );
        assert_eq!(
            "comprehensive".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Comprehensive
        );
        assert_eq!(
            "minimal".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Minimal
        );
    }

    #[test]
    fn test_enrichment_strategy_from_str_aliases() {
        assert_eq!(
            "research".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Academic
        );
        assert_eq!(
            "hands-on".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Practical
        );
        assert_eq!(
            "full".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Comprehensive
        );
        assert_eq!(
            "light".parse::<EnrichmentStrategy>().unwrap(),
            EnrichmentStrategy::Minimal
        );
    }

    #[test]
    fn test_enrichment_strategy_from_str_invalid() {
        let result = "invalid".parse::<EnrichmentStrategy>();
        assert!(result.is_err());
    }

    #[test]
    fn test_enrichment_strategy_all() {
        let all = EnrichmentStrategy::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&EnrichmentStrategy::Academic));
        assert!(all.contains(&EnrichmentStrategy::Practical));
        assert!(all.contains(&EnrichmentStrategy::Comprehensive));
        assert!(all.contains(&EnrichmentStrategy::Minimal));
    }

    #[test]
    fn test_enrichment_strategy_include_flags() {
        assert!(EnrichmentStrategy::Academic.include_citations());
        assert!(!EnrichmentStrategy::Academic.include_examples());
        assert!(EnrichmentStrategy::Academic.include_cross_refs());

        assert!(!EnrichmentStrategy::Practical.include_citations());
        assert!(EnrichmentStrategy::Practical.include_examples());
        assert!(!EnrichmentStrategy::Practical.include_cross_refs());

        assert!(EnrichmentStrategy::Comprehensive.include_citations());
        assert!(EnrichmentStrategy::Comprehensive.include_examples());
        assert!(EnrichmentStrategy::Comprehensive.include_cross_refs());
    }

    #[test]
    fn test_enrichment_strategy_max_values() {
        assert_eq!(EnrichmentStrategy::Academic.max_citations(), 10);
        assert_eq!(EnrichmentStrategy::Academic.max_examples(), 2);
        assert_eq!(EnrichmentStrategy::Academic.max_cross_refs(), 8);

        assert_eq!(EnrichmentStrategy::Practical.max_citations(), 3);
        assert_eq!(EnrichmentStrategy::Practical.max_examples(), 6);

        assert_eq!(EnrichmentStrategy::Minimal.max_citations(), 2);
        assert_eq!(EnrichmentStrategy::Minimal.max_examples(), 1);
    }

    #[test]
    fn test_enrichment_strategy_default() {
        assert_eq!(EnrichmentStrategy::default(), EnrichmentStrategy::Comprehensive);
    }

    #[test]
    fn test_enrichment_strategy_description() {
        assert!(EnrichmentStrategy::Academic.description().contains("citations"));
        assert!(EnrichmentStrategy::Practical.description().contains("examples"));
        assert!(EnrichmentStrategy::Comprehensive.description().contains("Full"));
        assert!(EnrichmentStrategy::Minimal.description().contains("Light"));
    }

    // -------------------------------------------------------------------------
    // Citation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_citation_new() {
        let citation = Citation::new("RFC 6749");
        assert_eq!(citation.source, "RFC 6749");
        assert!((citation.relevance - 0.5).abs() < f64::EPSILON);
        assert!(citation.quote.is_none());
        assert!(citation.url.is_none());
    }

    #[test]
    fn test_citation_builder() {
        let citation = Citation::new("OAuth 2.0 Spec")
            .with_relevance(0.9)
            .with_quote("Authorization framework")
            .with_url("https://oauth.net/2/")
            .with_author("IETF")
            .with_year("2012");

        assert_eq!(citation.relevance, 0.9);
        assert_eq!(citation.quote, Some("Authorization framework".to_string()));
        assert_eq!(citation.url, Some("https://oauth.net/2/".to_string()));
        assert_eq!(citation.author, Some("IETF".to_string()));
        assert_eq!(citation.year, Some("2012".to_string()));
    }

    #[test]
    fn test_citation_relevance_clamping() {
        let high = Citation::new("Test").with_relevance(1.5);
        assert_eq!(high.relevance, 1.0);

        let low = Citation::new("Test").with_relevance(-0.5);
        assert_eq!(low.relevance, 0.0);
    }

    #[test]
    fn test_citation_is_highly_relevant() {
        let high = Citation::new("Test").with_relevance(0.8);
        assert!(high.is_highly_relevant());

        let low = Citation::new("Test").with_relevance(0.5);
        assert!(!low.is_highly_relevant());
    }

    #[test]
    fn test_citation_to_markdown() {
        let citation = Citation::new("RFC 6749")
            .with_author("IETF")
            .with_year("2012")
            .with_quote("OAuth 2.0 authorization framework");

        let md = citation.to_markdown();
        assert!(md.contains("IETF"));
        assert!(md.contains("**RFC 6749**"));
        assert!(md.contains("(2012)"));
        assert!(md.contains("> \"OAuth 2.0 authorization framework\""));
    }

    #[test]
    fn test_citation_to_inline() {
        let citation = Citation::new("RFC 6749")
            .with_author("IETF")
            .with_year("2012");

        assert_eq!(citation.to_inline(), "(IETF, 2012)");

        let citation_no_year = Citation::new("Test").with_author("Author");
        assert_eq!(citation_no_year.to_inline(), "(Author)");

        let citation_no_author = Citation::new("Test Source");
        assert_eq!(citation_no_author.to_inline(), "(Test Source)");
    }

    #[test]
    fn test_citation_display() {
        let citation = Citation::new("Test Source");
        assert_eq!(format!("{}", citation), "Test Source");
    }

    // -------------------------------------------------------------------------
    // EnrichmentExample Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enrichment_example_new() {
        let example = EnrichmentExample::new("Test Example", "fn main() {}");
        assert_eq!(example.title, "Test Example");
        assert_eq!(example.content, "fn main() {}");
        assert!(example.language.is_none());
    }

    #[test]
    fn test_enrichment_example_builder() {
        let example = EnrichmentExample::new("Rust Example", "fn main() {}")
            .with_language("rust")
            .with_explanation("A simple main function")
            .with_context("Use for entry points");

        assert_eq!(example.language, Some("rust".to_string()));
        assert_eq!(example.explanation, Some("A simple main function".to_string()));
        assert_eq!(example.context, Some("Use for entry points".to_string()));
    }

    #[test]
    fn test_enrichment_example_to_markdown() {
        let example = EnrichmentExample::new("Hello World", "println!(\"Hello\");")
            .with_language("rust")
            .with_explanation("Print greeting")
            .with_context("Terminal output");

        let md = example.to_markdown();
        assert!(md.contains("### Hello World"));
        assert!(md.contains("```rust"));
        assert!(md.contains("println!"));
        assert!(md.contains("Print greeting"));
        assert!(md.contains("*Context: Terminal output*"));
    }

    #[test]
    fn test_enrichment_example_display() {
        let example = EnrichmentExample::new("Test Title", "content");
        assert_eq!(format!("{}", example), "Test Title");
    }

    // -------------------------------------------------------------------------
    // CrossRef Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cross_ref_new() {
        let cross_ref = CrossRef::new("OAuth Guide", "Authentication reference", "related");
        assert_eq!(cross_ref.title, "OAuth Guide");
        assert_eq!(cross_ref.description, "Authentication reference");
        assert_eq!(cross_ref.relationship, "related");
        assert!((cross_ref.relevance - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cross_ref_builder() {
        let cross_ref = CrossRef::new("JWT Tutorial", "Token handling", "prerequisite")
            .with_relevance(0.8)
            .with_path("/tutorials/jwt")
            .with_tags(vec!["auth".to_string(), "security".to_string()]);

        assert_eq!(cross_ref.relevance, 0.8);
        assert_eq!(cross_ref.path, Some("/tutorials/jwt".to_string()));
        assert_eq!(cross_ref.tags.len(), 2);
    }

    #[test]
    fn test_cross_ref_add_tag() {
        let cross_ref = CrossRef::new("Test", "Desc", "related")
            .add_tag("tag1")
            .add_tag("tag2");

        assert_eq!(cross_ref.tags, vec!["tag1", "tag2"]);
    }

    #[test]
    fn test_cross_ref_relevance_clamping() {
        let high = CrossRef::new("Test", "Desc", "related").with_relevance(1.5);
        assert_eq!(high.relevance, 1.0);

        let low = CrossRef::new("Test", "Desc", "related").with_relevance(-0.5);
        assert_eq!(low.relevance, 0.0);
    }

    #[test]
    fn test_cross_ref_is_highly_relevant() {
        let high = CrossRef::new("Test", "Desc", "related").with_relevance(0.8);
        assert!(high.is_highly_relevant());

        let low = CrossRef::new("Test", "Desc", "related").with_relevance(0.5);
        assert!(!low.is_highly_relevant());
    }

    #[test]
    fn test_cross_ref_to_markdown() {
        let cross_ref = CrossRef::new("OAuth Guide", "Auth reference", "prerequisite")
            .with_path("/guides/oauth")
            .with_tags(vec!["auth".to_string()]);

        let md = cross_ref.to_markdown();
        assert!(md.contains("**OAuth Guide**"));
        assert!(md.contains("_prerequisite_"));
        assert!(md.contains("Auth reference"));
        assert!(md.contains("`/guides/oauth`"));
        assert!(md.contains("Tags: auth"));
    }

    #[test]
    fn test_cross_ref_display() {
        let cross_ref = CrossRef::new("Test Title", "Desc", "related");
        assert_eq!(format!("{}", cross_ref), "Test Title (related)");
    }

    // -------------------------------------------------------------------------
    // EnrichmentMetadata Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enrichment_metadata_new() {
        let metadata = EnrichmentMetadata::new(100, "Comprehensive");
        assert_eq!(metadata.enrichment_time_ms, 100);
        assert_eq!(metadata.strategy, "Comprehensive");
        assert!(!metadata.timestamp.is_empty());
    }

    #[test]
    fn test_enrichment_metadata_builder() {
        let metadata = EnrichmentMetadata::new(150, "Academic")
            .with_sources_consulted(5)
            .with_confidence(0.85)
            .with_tokens_used(1000)
            .add_note("Used fallback for examples");

        assert_eq!(metadata.sources_consulted, 5);
        assert_eq!(metadata.confidence, 0.85);
        assert_eq!(metadata.tokens_used, Some(1000));
        assert_eq!(metadata.notes.len(), 1);
    }

    #[test]
    fn test_enrichment_metadata_confidence_clamping() {
        let high = EnrichmentMetadata::new(100, "Test").with_confidence(1.5);
        assert_eq!(high.confidence, 1.0);

        let low = EnrichmentMetadata::new(100, "Test").with_confidence(-0.5);
        assert_eq!(low.confidence, 0.0);
    }

    #[test]
    fn test_enrichment_metadata_is_high_confidence() {
        let high = EnrichmentMetadata::new(100, "Test").with_confidence(0.8);
        assert!(high.is_high_confidence());

        let low = EnrichmentMetadata::new(100, "Test").with_confidence(0.5);
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_enrichment_metadata_default() {
        let metadata = EnrichmentMetadata::default();
        assert_eq!(metadata.enrichment_time_ms, 0);
        assert!((metadata.confidence - 0.5).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // EnrichedPrompt Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enriched_prompt_new() {
        let enriched = EnrichedPrompt::new("original prompt", "enriched content");
        assert_eq!(enriched.original, "original prompt");
        assert_eq!(enriched.content, "enriched content");
        assert!(enriched.citations.is_empty());
        assert!(enriched.examples.is_empty());
        assert!(enriched.cross_references.is_empty());
    }

    #[test]
    fn test_enriched_prompt_is_substantial() {
        let mut enriched = EnrichedPrompt::new("original", "content");
        assert!(!enriched.is_substantial());

        enriched.citations.push(Citation::new("Test Source"));
        assert!(enriched.is_substantial());
    }

    #[test]
    fn test_enriched_prompt_enrichment_count() {
        let mut enriched = EnrichedPrompt::new("original", "content");
        assert_eq!(enriched.enrichment_count(), 0);

        enriched.citations.push(Citation::new("Source 1"));
        enriched.citations.push(Citation::new("Source 2"));
        enriched.examples.push(EnrichmentExample::new("Ex", "code"));
        assert_eq!(enriched.enrichment_count(), 3);
    }

    #[test]
    fn test_enriched_prompt_enrichment_summary() {
        let mut enriched = EnrichedPrompt::new("original", "content");
        assert_eq!(enriched.enrichment_summary(), "No enrichments");

        enriched.citations.push(Citation::new("Source"));
        enriched.examples.push(EnrichmentExample::new("Ex", "code"));
        let summary = enriched.enrichment_summary();
        assert!(summary.contains("1 citations"));
        assert!(summary.contains("1 examples"));
    }

    #[test]
    fn test_enriched_prompt_high_relevance_citations() {
        let mut enriched = EnrichedPrompt::new("original", "content");
        enriched.citations.push(Citation::new("High").with_relevance(0.9));
        enriched.citations.push(Citation::new("Low").with_relevance(0.3));

        let high = enriched.high_relevance_citations();
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].source, "High");
    }

    #[test]
    fn test_enriched_prompt_high_relevance_cross_refs() {
        let mut enriched = EnrichedPrompt::new("original", "content");
        enriched.cross_references.push(
            CrossRef::new("High", "Desc", "related").with_relevance(0.8)
        );
        enriched.cross_references.push(
            CrossRef::new("Low", "Desc", "related").with_relevance(0.4)
        );

        let high = enriched.high_relevance_cross_refs();
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].title, "High");
    }

    #[test]
    fn test_enriched_prompt_to_markdown() {
        let mut enriched = EnrichedPrompt::new("original prompt", "enriched content");
        enriched.summary = Some("Test summary".to_string());
        enriched.citations.push(Citation::new("Test Source"));
        enriched.key_terms = vec!["OAuth".to_string(), "JWT".to_string()];

        let md = enriched.to_markdown();
        assert!(md.contains("# Enriched Prompt"));
        assert!(md.contains("**Summary:** Test summary"));
        assert!(md.contains("## Content"));
        assert!(md.contains("## Citations"));
        assert!(md.contains("## Key Terms"));
        assert!(md.contains("OAuth, JWT"));
    }

    #[test]
    fn test_enriched_prompt_display() {
        let enriched = EnrichedPrompt::new("original", "displayed content");
        assert_eq!(format!("{}", enriched), "displayed content");
    }

    // -------------------------------------------------------------------------
    // EnricherConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enricher_config_default() {
        let config = EnricherConfig::default();
        assert_eq!(config.max_citations, 8);
        assert_eq!(config.max_examples, 5);
        assert_eq!(config.max_cross_refs, 6);
        assert!(config.include_examples);
        assert!(config.include_citations);
        assert!(config.include_cross_refs);
    }

    #[test]
    fn test_enricher_config_fast() {
        let config = EnricherConfig::fast();
        assert_eq!(config.max_citations, 3);
        assert_eq!(config.max_examples, 2);
        assert!(!config.include_cross_refs);
        assert!(!config.generate_summary);
    }

    #[test]
    fn test_enricher_config_thorough() {
        let config = EnricherConfig::thorough();
        assert_eq!(config.max_citations, 15);
        assert_eq!(config.max_examples, 8);
        assert_eq!(config.max_cross_refs, 10);
        assert!(config.generate_summary);
    }

    #[test]
    fn test_enricher_config_from_strategy() {
        let academic = EnricherConfig::from_strategy(EnrichmentStrategy::Academic);
        assert_eq!(academic.max_citations, 10);
        assert!(academic.include_citations);
        assert!(!academic.include_examples);

        let practical = EnricherConfig::from_strategy(EnrichmentStrategy::Practical);
        assert!(practical.include_examples);
        assert!(!practical.include_citations);
    }

    // -------------------------------------------------------------------------
    // TermExtractor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_term_extractor_technical_terms() {
        let extractor = TermExtractor::default();
        let content = "Implement OAuth2 with JWT tokens using PostgreSQL";

        let terms = extractor.extract(content);
        assert!(terms.iter().any(|t| t.contains("OAuth")));
        assert!(terms.iter().any(|t| t.contains("JWT")));
        assert!(terms.iter().any(|t| t.contains("PostgreSQL")));
    }

    #[test]
    fn test_term_extractor_domain_terms() {
        let extractor = TermExtractor::default();
        let content = "Implement authentication with middleware and caching";

        let terms = extractor.extract(content);
        assert!(terms.contains(&"authentication".to_string()));
        assert!(terms.contains(&"middleware".to_string()));
        assert!(terms.contains(&"caching".to_string()));
    }

    #[test]
    fn test_term_extractor_filters_common_words() {
        let extractor = TermExtractor::default();
        let content = "the and for are but not";

        let terms = extractor.extract(content);
        assert!(terms.is_empty());
    }

    // -------------------------------------------------------------------------
    // EnricherAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enricher_agent_new() {
        let agent = EnricherAgent::new();
        assert_eq!(agent.config.max_citations, 8);
        assert!(agent.gemini_client.is_none());
    }

    #[test]
    fn test_enricher_agent_builder() {
        let agent = EnricherAgent::builder()
            .max_citations(10)
            .max_examples(8)
            .max_cross_refs(5)
            .include_examples(false)
            .generate_summary(false)
            .build();

        assert_eq!(agent.config.max_citations, 10);
        assert_eq!(agent.config.max_examples, 8);
        assert_eq!(agent.config.max_cross_refs, 5);
        assert!(!agent.config.include_examples);
        assert!(!agent.config.generate_summary);
    }

    #[test]
    fn test_enricher_agent_builder_min_citation_relevance() {
        let agent = EnricherAgent::builder()
            .min_citation_relevance(0.7)
            .include_citations(false)
            .include_cross_refs(false)
            .build();

        assert_eq!(agent.config.min_citation_relevance, 0.7);
        assert!(!agent.config.include_citations);
        assert!(!agent.config.include_cross_refs);
    }

    #[test]
    fn test_enricher_agent_extract_terms() {
        let agent = EnricherAgent::new();
        let terms = agent.extract_terms("Use OAuth2 for authentication with JWT");

        assert!(!terms.is_empty());
        assert!(terms.iter().any(|t| t.contains("OAuth") || t.contains("oauth")));
    }

    #[tokio::test]
    async fn test_enricher_agent_enrich_empty_content() {
        let agent = EnricherAgent::new();
        let result = agent.enrich("", EnrichmentStrategy::Comprehensive).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_enricher_agent_enrich_whitespace() {
        let agent = EnricherAgent::new();
        let result = agent.enrich("   \n\t  ", EnrichmentStrategy::Minimal).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_enricher_agent_enrich_without_llm() {
        let agent = EnricherAgent::new();
        let result = agent
            .enrich("Implement OAuth2 authentication", EnrichmentStrategy::Comprehensive)
            .await;

        assert!(result.is_ok());
        let enriched = result.unwrap();
        assert_eq!(enriched.original, "Implement OAuth2 authentication");
        assert!(!enriched.key_terms.is_empty());
    }

    #[tokio::test]
    async fn test_enricher_agent_enrich_academic_strategy() {
        let agent = EnricherAgent::new();
        let result = agent
            .enrich("Explain OAuth 2.0 protocol", EnrichmentStrategy::Academic)
            .await;

        assert!(result.is_ok());
        let enriched = result.unwrap();
        assert!(enriched.metadata.strategy.contains("Academic"));
    }

    #[tokio::test]
    async fn test_enricher_agent_enrich_practical_strategy() {
        let agent = EnricherAgent::new();
        let result = agent
            .enrich("Implement rate limiting", EnrichmentStrategy::Practical)
            .await;

        assert!(result.is_ok());
        let enriched = result.unwrap();
        assert!(enriched.metadata.strategy.contains("Practical"));
    }

    #[tokio::test]
    async fn test_enricher_agent_enrich_minimal_strategy() {
        let agent = EnricherAgent::new();
        let result = agent
            .enrich("Simple test prompt", EnrichmentStrategy::Minimal)
            .await;

        assert!(result.is_ok());
        let enriched = result.unwrap();
        assert!(enriched.metadata.strategy.contains("Minimal"));
    }

    // -------------------------------------------------------------------------
    // JSON Parsing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_citations_json() {
        let agent = EnricherAgent::new();
        let json = r#"[
            {"source": "RFC 6749", "relevance": 0.9, "author": "IETF", "year": "2012"},
            {"source": "Rust Book", "relevance": 0.7, "quote": "Memory safety"}
        ]"#;

        let citations = agent.parse_citations_response(json, 10).unwrap();
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0].source, "RFC 6749");
        assert_eq!(citations[0].relevance, 0.9);
        assert_eq!(citations[0].author, Some("IETF".to_string()));
    }

    #[test]
    fn test_parse_citations_json_in_code_block() {
        let agent = EnricherAgent::new();
        let response = r#"Here are citations:
```json
[{"source": "Test Source", "relevance": 0.8}]
```"#;

        let citations = agent.parse_citations_response(response, 10).unwrap();
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].source, "Test Source");
    }

    #[test]
    fn test_parse_examples_json() {
        let agent = EnricherAgent::new();
        let json = r#"[
            {"title": "Basic Example", "content": "fn main() {}", "language": "rust", "explanation": "Entry point"}
        ]"#;

        let examples = agent.parse_examples_response(json, 10).unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].title, "Basic Example");
        assert_eq!(examples[0].language, Some("rust".to_string()));
    }

    #[test]
    fn test_parse_cross_refs_json() {
        let agent = EnricherAgent::new();
        let json = r#"[
            {"title": "Auth Guide", "description": "OAuth reference", "relationship": "prerequisite", "relevance": 0.85, "tags": ["auth", "security"]}
        ]"#;

        let cross_refs = agent.parse_cross_refs_response(json, 10).unwrap();
        assert_eq!(cross_refs.len(), 1);
        assert_eq!(cross_refs[0].title, "Auth Guide");
        assert_eq!(cross_refs[0].relationship, "prerequisite");
        assert_eq!(cross_refs[0].tags.len(), 2);
    }

    #[test]
    fn test_parse_invalid_json() {
        let agent = EnricherAgent::new();

        let citations = agent.parse_citations_response("not json", 10).unwrap();
        assert!(citations.is_empty());

        let examples = agent.parse_examples_response("not json", 10).unwrap();
        assert!(examples.is_empty());

        let cross_refs = agent.parse_cross_refs_response("not json", 10).unwrap();
        assert!(cross_refs.is_empty());
    }

    // -------------------------------------------------------------------------
    // Placeholder Generation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_placeholder_citations() {
        let agent = EnricherAgent::new();
        let terms = vec!["OAuth".to_string(), "JWT".to_string()];

        let citations = agent.generate_placeholder_citations(&terms, 5);
        assert_eq!(citations.len(), 2);
        assert!(citations[0].source.contains("OAuth"));
    }

    #[test]
    fn test_generate_placeholder_examples() {
        let agent = EnricherAgent::new();

        let examples = agent.generate_placeholder_examples("Test prompt", 5);
        assert_eq!(examples.len(), 1);
        assert!(examples[0].content.contains("Test prompt"));
    }

    #[test]
    fn test_generate_placeholder_examples_max_zero() {
        let agent = EnricherAgent::new();

        let examples = agent.generate_placeholder_examples("Test", 0);
        assert!(examples.is_empty());
    }

    #[test]
    fn test_generate_placeholder_cross_refs() {
        let agent = EnricherAgent::new();
        let terms = vec!["OAuth".to_string(), "JWT".to_string()];

        let cross_refs = agent.generate_placeholder_cross_refs(&terms, 5);
        assert_eq!(cross_refs.len(), 2);
        assert!(cross_refs[0].title.contains("OAuth"));
    }

    // -------------------------------------------------------------------------
    // Confidence Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_calculate_confidence_no_enrichments() {
        let agent = EnricherAgent::new();
        let enriched = EnrichedPrompt::new("test", "test");

        let confidence = agent.calculate_confidence(&enriched);
        assert!((confidence - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_confidence_with_enrichments() {
        let agent = EnricherAgent::new();
        let mut enriched = EnrichedPrompt::new("test", "test");
        enriched.citations.push(Citation::new("Source").with_relevance(0.9));
        enriched.cross_references.push(
            CrossRef::new("Ref", "Desc", "related").with_relevance(0.8)
        );
        enriched.examples.push(EnrichmentExample::new("Ex", "code"));

        let confidence = agent.calculate_confidence(&enriched);
        assert!(confidence > 0.5);
        assert!(confidence <= 1.0);
    }

    // -------------------------------------------------------------------------
    // Builder Pattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_chaining() {
        let agent = EnricherAgent::builder()
            .max_citations(5)
            .max_examples(3)
            .max_cross_refs(2)
            .include_examples(true)
            .include_citations(true)
            .include_cross_refs(false)
            .min_citation_relevance(0.5)
            .generate_summary(true)
            .build();

        assert_eq!(agent.config.max_citations, 5);
        assert_eq!(agent.config.max_examples, 3);
        assert_eq!(agent.config.max_cross_refs, 2);
        assert!(agent.config.include_examples);
        assert!(agent.config.include_citations);
        assert!(!agent.config.include_cross_refs);
        assert!((agent.config.min_citation_relevance - 0.5).abs() < f64::EPSILON);
        assert!(agent.config.generate_summary);
    }

    #[test]
    fn test_builder_with_config() {
        let config = EnricherConfig::thorough();
        let agent = EnricherAgent::builder()
            .config(config)
            .build();

        assert_eq!(agent.config.max_citations, 15);
    }

    #[test]
    fn test_builder_default() {
        let builder = EnricherAgentBuilder::default();
        let agent = builder.build();

        assert_eq!(agent.config.max_citations, 8);
    }
}
