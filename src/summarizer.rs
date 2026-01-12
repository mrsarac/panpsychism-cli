//! Summarizer Agent module for Project Panpsychism.
//!
//! The Essence Distiller - compresses content while preserving meaning.
//! Like a sorcerer distilling the essence of a potion, this agent extracts
//! the core meaning from verbose content while maintaining its magical properties.
//!
//! # The Sorcerer's Wand Metaphor
//!
//! In the realm of Project Panpsychism, the Summarizer Agent acts as the
//! **Essence Distiller** - a powerful spell that condenses lengthy scrolls
//! into concentrated wisdom. Just as an alchemist reduces base materials
//! to their purest form, this agent distills verbose text into essential meaning.
//!
//! ## Compression Levels
//!
//! The distillation process offers three levels of concentration:
//!
//! - **Brief**: Maximum compression (~25% of original) - essence drops
//! - **Standard**: Balanced compression (~50% of original) - concentrated elixir
//! - **Detailed**: Light compression (~75% of original) - refined tincture
//!
//! # Philosophical Foundation
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Preserve the self-sustaining core of the message
//! - **RATIO**: Apply logical reduction while maintaining coherence
//! - **LAETITIA**: Enhance clarity and joy through conciseness
//! - **NATURA**: Respect the natural structure of knowledge
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::summarizer::{SummarizerAgent, CompressionLevel};
//! use panpsychism::gemini::GeminiClient;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let client = Arc::new(GeminiClient::new());
//!     let summarizer = SummarizerAgent::new(client);
//!
//!     let text = "A very long document that needs summarization...";
//!     let summary = summarizer.summarize(text, CompressionLevel::Standard).await?;
//!
//!     println!("Summary: {}", summary.content);
//!     println!("Compression: {:.1}%", summary.compression_ratio * 100.0);
//!     Ok(())
//! }
//! ```

use crate::gemini::{GeminiClient, Message};
use crate::{Error, Result};
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// COMPRESSION LEVEL
// =============================================================================

/// Compression level for summarization.
///
/// Each level represents a different target size relative to the original:
///
/// - **Brief**: ~25% of original - maximum essence extraction
/// - **Standard**: ~50% of original - balanced distillation
/// - **Detailed**: ~75% of original - light refinement
///
/// # The Alchemist's Choice
///
/// Like choosing the concentration of a potion, the compression level
/// determines how much of the original substance remains after distillation.
/// Brief summaries are like essence drops - potent but minimal.
/// Detailed summaries preserve more of the original body while removing impurities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionLevel {
    /// Maximum compression - approximately 25% of original length.
    /// Use when you need the absolute essence of the content.
    Brief,

    /// Balanced compression - approximately 50% of original length.
    /// The default choice for most summarization needs.
    #[default]
    Standard,

    /// Light compression - approximately 75% of original length.
    /// Preserves more detail while still reducing verbosity.
    Detailed,
}

impl CompressionLevel {
    /// Get the target ratio (0.0 to 1.0) for this compression level.
    ///
    /// # Returns
    ///
    /// The target length as a fraction of the original:
    /// - Brief: 0.25 (25%)
    /// - Standard: 0.50 (50%)
    /// - Detailed: 0.75 (75%)
    pub fn target_ratio(&self) -> f64 {
        match self {
            CompressionLevel::Brief => 0.25,
            CompressionLevel::Standard => 0.50,
            CompressionLevel::Detailed => 0.75,
        }
    }

    /// Calculate target word count based on original word count.
    ///
    /// # Arguments
    ///
    /// * `original_words` - Number of words in the original content
    ///
    /// # Returns
    ///
    /// Target number of words for the summary.
    pub fn target_words(&self, original_words: usize) -> usize {
        ((original_words as f64) * self.target_ratio()).round() as usize
    }

    /// Calculate target token count based on original token count.
    ///
    /// Uses a rough approximation of 0.75 words per token.
    ///
    /// # Arguments
    ///
    /// * `original_tokens` - Estimated number of tokens in original
    ///
    /// # Returns
    ///
    /// Target number of tokens for the summary.
    pub fn target_tokens(&self, original_tokens: usize) -> usize {
        ((original_tokens as f64) * self.target_ratio()).round() as usize
    }

    /// Get a human-readable description of this compression level.
    pub fn description(&self) -> &'static str {
        match self {
            CompressionLevel::Brief => "Brief (25% - essence extraction)",
            CompressionLevel::Standard => "Standard (50% - balanced distillation)",
            CompressionLevel::Detailed => "Detailed (75% - light refinement)",
        }
    }

    /// Get the label for this compression level.
    pub fn label(&self) -> &'static str {
        match self {
            CompressionLevel::Brief => "Brief",
            CompressionLevel::Standard => "Standard",
            CompressionLevel::Detailed => "Detailed",
        }
    }
}

impl std::fmt::Display for CompressionLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl std::str::FromStr for CompressionLevel {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "brief" | "short" | "minimal" | "25" => Ok(CompressionLevel::Brief),
            "standard" | "normal" | "medium" | "50" => Ok(CompressionLevel::Standard),
            "detailed" | "long" | "verbose" | "75" => Ok(CompressionLevel::Detailed),
            _ => Err(Error::Config(format!(
                "Unknown compression level: '{}'. Valid levels: brief, standard, detailed",
                s
            ))),
        }
    }
}

// =============================================================================
// KEY POINT
// =============================================================================

/// A key point extracted from the content.
///
/// Represents a distilled insight or important concept from the original text.
/// Each key point captures an atomic unit of meaning that stands on its own.
///
/// # The Wisdom Crystals
///
/// Like crystals formed from a solution, key points are concentrated
/// nuggets of wisdom extracted from the primordial soup of verbose text.
#[derive(Debug, Clone)]
pub struct KeyPoint {
    /// The main statement of this key point.
    pub statement: String,

    /// Importance score (0.0 to 1.0), higher means more critical.
    pub importance: f64,

    /// Category or theme this key point belongs to (optional).
    pub category: Option<String>,

    /// Supporting evidence or context (optional).
    pub evidence: Option<String>,
}

impl KeyPoint {
    /// Create a new key point with just a statement.
    pub fn new(statement: impl Into<String>) -> Self {
        Self {
            statement: statement.into(),
            importance: 0.5,
            category: None,
            evidence: None,
        }
    }

    /// Set the importance score.
    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set supporting evidence.
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }

    /// Check if this is a high-importance key point (>= 0.7).
    pub fn is_critical(&self) -> bool {
        self.importance >= 0.7
    }

    /// Get a brief representation of this key point.
    pub fn brief(&self) -> String {
        if self.statement.len() <= 100 {
            self.statement.clone()
        } else {
            format!("{}...", &self.statement[..97])
        }
    }
}

impl std::fmt::Display for KeyPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.statement)
    }
}

// =============================================================================
// SUMMARY
// =============================================================================

/// A complete summary with metadata.
///
/// Contains the distilled content along with statistics about
/// the summarization process.
///
/// # The Distillation Report
///
/// Like an alchemist's log, this structure records not just the result
/// of the distillation but also the process metrics - how much was
/// compressed, how long it took, and what key elements were preserved.
#[derive(Debug, Clone)]
pub struct Summary {
    /// The summarized content.
    pub content: String,

    /// Original word count.
    pub original_words: usize,

    /// Summary word count.
    pub summary_words: usize,

    /// Compression ratio achieved (0.0 to 1.0).
    /// Lower values mean more compression.
    pub compression_ratio: f64,

    /// Compression level that was requested.
    pub level: CompressionLevel,

    /// Processing time in milliseconds.
    pub duration_ms: u64,

    /// Key points extracted (if any).
    pub key_points: Vec<KeyPoint>,

    /// Token usage for the LLM call.
    pub tokens_used: usize,
}

impl Summary {
    /// Check if the summary achieved the target compression.
    ///
    /// Returns true if the actual compression is within 10% of target.
    pub fn achieved_target(&self) -> bool {
        let target = self.level.target_ratio();
        (self.compression_ratio - target).abs() <= 0.10
    }

    /// Get the compression percentage as a readable string.
    pub fn compression_percentage(&self) -> String {
        format!("{:.1}%", self.compression_ratio * 100.0)
    }

    /// Get words saved by summarization.
    pub fn words_saved(&self) -> usize {
        self.original_words.saturating_sub(self.summary_words)
    }

    /// Get a brief stats summary.
    pub fn stats(&self) -> String {
        format!(
            "{} -> {} words ({} compression) in {}ms",
            self.original_words,
            self.summary_words,
            self.compression_percentage(),
            self.duration_ms
        )
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

// =============================================================================
// SUMMARIZER CONFIGURATION
// =============================================================================

/// Configuration for the summarizer agent.
#[derive(Debug, Clone)]
pub struct SummarizerConfig {
    /// Maximum retries for LLM calls.
    pub max_retries: u32,

    /// Whether to extract key points during summarization.
    pub extract_key_points: bool,

    /// Maximum key points to extract.
    pub max_key_points: usize,

    /// Minimum content length (in words) for summarization.
    /// Content below this threshold is returned as-is.
    pub min_content_words: usize,

    /// Whether to preserve formatting (headers, lists, etc.).
    pub preserve_formatting: bool,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            extract_key_points: false,
            max_key_points: 10,
            min_content_words: 50,
            preserve_formatting: false,
        }
    }
}

impl SummarizerConfig {
    /// Create a configuration that extracts key points.
    pub fn with_key_points() -> Self {
        Self {
            extract_key_points: true,
            ..Default::default()
        }
    }

    /// Create a configuration that preserves formatting.
    pub fn preserve_format() -> Self {
        Self {
            preserve_formatting: true,
            ..Default::default()
        }
    }
}

// =============================================================================
// SUMMARIZER AGENT
// =============================================================================

/// The Essence Distiller - Agent 14 of Project Panpsychism.
///
/// This agent compresses content while preserving meaning, using the
/// Sorcerer's Wand metaphor. It distills verbose text into concentrated
/// wisdom at various compression levels.
///
/// # Architecture
///
/// The SummarizerAgent uses an `Arc<GeminiClient>` for LLM integration,
/// allowing shared access across multiple operations. It supports:
///
/// - Three compression levels (Brief, Standard, Detailed)
/// - Key point extraction
/// - TL;DR generation
/// - Batch summarization
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::summarizer::{SummarizerAgent, CompressionLevel};
/// use panpsychism::gemini::GeminiClient;
/// use std::sync::Arc;
///
/// let client = Arc::new(GeminiClient::new());
/// let summarizer = SummarizerAgent::new(client);
///
/// // Summarize with standard compression
/// let summary = summarizer.summarize("Long text...", CompressionLevel::Standard).await?;
///
/// // Extract key points
/// let points = summarizer.extract_key_points("Long text...", 5).await?;
///
/// // Quick TL;DR
/// let tldr = summarizer.tldr("Long text...").await?;
/// ```
#[derive(Debug, Clone)]
pub struct SummarizerAgent {
    /// The Gemini client for LLM calls.
    client: Arc<GeminiClient>,

    /// Configuration for summarization behavior.
    config: SummarizerConfig,
}

impl SummarizerAgent {
    /// Create a new SummarizerAgent with the given Gemini client.
    ///
    /// # Arguments
    ///
    /// * `client` - Arc-wrapped GeminiClient for LLM integration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = Arc::new(GeminiClient::new());
    /// let summarizer = SummarizerAgent::new(client);
    /// ```
    pub fn new(client: Arc<GeminiClient>) -> Self {
        Self {
            client,
            config: SummarizerConfig::default(),
        }
    }

    /// Create a new SummarizerAgent with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `client` - Arc-wrapped GeminiClient for LLM integration
    /// * `config` - Custom configuration for summarization
    pub fn with_config(client: Arc<GeminiClient>, config: SummarizerConfig) -> Self {
        Self { client, config }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SummarizerConfig {
        &self.config
    }

    // =========================================================================
    // CORE SUMMARIZATION
    // =========================================================================

    /// Summarize content at the specified compression level.
    ///
    /// This is the primary summarization method. It distills the input content
    /// to approximately the target size based on the compression level.
    ///
    /// # The Distillation Spell
    ///
    /// Like a master alchemist reducing a complex mixture to its essence,
    /// this method applies careful reduction while preserving the core meaning.
    ///
    /// # Arguments
    ///
    /// * `content` - The text to summarize
    /// * `level` - Desired compression level (Brief, Standard, or Detailed)
    ///
    /// # Returns
    ///
    /// A `Summary` containing the compressed content and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Content is empty
    /// - LLM call fails after retries
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let summary = summarizer.summarize(
    ///     "A long document about machine learning...",
    ///     CompressionLevel::Brief
    /// ).await?;
    ///
    /// println!("Original: {} words", summary.original_words);
    /// println!("Summary: {} words", summary.summary_words);
    /// println!("{}", summary.content);
    /// ```
    pub async fn summarize(&self, content: &str, level: CompressionLevel) -> Result<Summary> {
        let start = Instant::now();

        // Validate input
        if content.trim().is_empty() {
            return Err(Error::Synthesis(
                "Cannot summarize empty content".to_string(),
            ));
        }

        let original_words = Self::count_words(content);

        // If content is too short, return as-is
        if original_words < self.config.min_content_words {
            return Ok(Summary {
                content: content.to_string(),
                original_words,
                summary_words: original_words,
                compression_ratio: 1.0,
                level,
                duration_ms: start.elapsed().as_millis() as u64,
                key_points: Vec::new(),
                tokens_used: 0,
            });
        }

        let target_words = level.target_words(original_words);
        let prompt = self.build_summarize_prompt(content, level, target_words);

        let response = self.call_llm(&prompt).await?;
        let summary_words = Self::count_words(&response);
        let compression_ratio = if original_words > 0 {
            summary_words as f64 / original_words as f64
        } else {
            1.0
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        // Extract key points if configured
        let key_points = if self.config.extract_key_points {
            self.extract_key_points_internal(&response, self.config.max_key_points)
                .await
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(Summary {
            content: response,
            original_words,
            summary_words,
            compression_ratio,
            level,
            duration_ms,
            key_points,
            tokens_used: Self::estimate_tokens(content) + Self::estimate_tokens(&prompt),
        })
    }

    /// Extract key points from content.
    ///
    /// Identifies and extracts the most important points from the given content.
    /// Each key point is a standalone insight that captures essential meaning.
    ///
    /// # The Crystal Extraction
    ///
    /// Like extracting precious crystals from raw ore, this method identifies
    /// the concentrated wisdom nuggets embedded within verbose prose.
    ///
    /// # Arguments
    ///
    /// * `content` - The text to analyze
    /// * `max_points` - Maximum number of key points to extract
    ///
    /// # Returns
    ///
    /// A vector of `KeyPoint` instances, sorted by importance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Content is empty
    /// - LLM call fails after retries
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let points = summarizer.extract_key_points(
    ///     "A detailed report on climate change...",
    ///     5
    /// ).await?;
    ///
    /// for (i, point) in points.iter().enumerate() {
    ///     println!("{}. {} (importance: {:.2})", i + 1, point.statement, point.importance);
    /// }
    /// ```
    pub async fn extract_key_points(
        &self,
        content: &str,
        max_points: usize,
    ) -> Result<Vec<KeyPoint>> {
        if content.trim().is_empty() {
            return Err(Error::Synthesis(
                "Cannot extract key points from empty content".to_string(),
            ));
        }

        self.extract_key_points_internal(content, max_points).await
    }

    /// Generate a TL;DR (Too Long; Didn't Read) summary.
    ///
    /// Creates an ultra-brief, one to two sentence summary that captures
    /// the absolute essence of the content. This is the most aggressive
    /// form of compression.
    ///
    /// # The Essence Drop
    ///
    /// Like distilling an entire potion down to a single drop of essence,
    /// this method produces the most concentrated form of the content's meaning.
    ///
    /// # Arguments
    ///
    /// * `content` - The text to summarize
    ///
    /// # Returns
    ///
    /// A string containing the TL;DR summary (typically 1-2 sentences).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Content is empty
    /// - LLM call fails after retries
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tldr = summarizer.tldr(
    ///     "A 10-page document about quantum computing..."
    /// ).await?;
    ///
    /// println!("TL;DR: {}", tldr);
    /// // Output: "TL;DR: Quantum computers use qubits to solve certain problems exponentially faster than classical computers."
    /// ```
    pub async fn tldr(&self, content: &str) -> Result<String> {
        if content.trim().is_empty() {
            return Err(Error::Synthesis(
                "Cannot generate TL;DR for empty content".to_string(),
            ));
        }

        let prompt = format!(
            r#"You are the Essence Distiller, a master of extracting core meaning.

Generate a TL;DR (Too Long; Didn't Read) for the following content.
The TL;DR should be 1-2 sentences maximum that capture the absolute essence.

Guidelines:
- Be extremely concise
- Capture the single most important takeaway
- Use clear, direct language
- Do not start with "TL;DR:" - just provide the summary

Content to distill:
---
{}
---

TL;DR:"#,
            content
        );

        let response = self.call_llm(&prompt).await?;

        // Clean up the response
        let tldr = response
            .trim()
            .trim_start_matches("TL;DR:")
            .trim_start_matches("TLDR:")
            .trim();

        Ok(tldr.to_string())
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Build the summarization prompt for the LLM.
    fn build_summarize_prompt(
        &self,
        content: &str,
        level: CompressionLevel,
        target_words: usize,
    ) -> String {
        let formatting_instruction = if self.config.preserve_formatting {
            "Preserve any headers, lists, or structural formatting from the original."
        } else {
            "Use plain prose without special formatting."
        };

        format!(
            r#"You are the Essence Distiller, Agent 14 of Project Panpsychism.
Your role is to compress content while preserving its essential meaning.

Following Spinoza's principles:
- CONATUS: Preserve the self-sustaining core of the message
- RATIO: Apply logical reduction while maintaining coherence
- LAETITIA: Enhance clarity and joy through conciseness
- NATURA: Respect the natural structure of knowledge

Compression Level: {} ({})
Target Length: approximately {} words

Guidelines:
- Focus on the most important information
- Remove redundancy and filler
- Maintain logical flow and coherence
- Preserve key facts, figures, and conclusions
- {}

Content to summarize:
---
{}
---

Provide only the summary, without preamble or explanation:"#,
            level.label(),
            level.description(),
            target_words,
            formatting_instruction,
            content
        )
    }

    /// Internal method to extract key points.
    async fn extract_key_points_internal(
        &self,
        content: &str,
        max_points: usize,
    ) -> Result<Vec<KeyPoint>> {
        let prompt = format!(
            r#"You are the Essence Distiller, extracting wisdom crystals from text.

Extract exactly {} key points from the following content.
Each key point should be a standalone insight that captures essential meaning.

For each key point, provide:
1. The main statement (one clear sentence)
2. Importance (high/medium/low)
3. Category (if apparent)

Format each point as:
POINT: [statement]
IMPORTANCE: [high/medium/low]
CATEGORY: [category or "general"]

Content:
---
{}
---

Key Points:"#,
            max_points, content
        );

        let response = self.call_llm(&prompt).await?;
        let points = self.parse_key_points(&response);

        Ok(points)
    }

    /// Parse key points from LLM response.
    fn parse_key_points(&self, response: &str) -> Vec<KeyPoint> {
        let mut points = Vec::new();
        let mut current_statement = String::new();
        let mut current_importance = 0.5;
        let mut current_category: Option<String> = None;

        for line in response.lines() {
            let line = line.trim();

            if let Some(statement) = line.strip_prefix("POINT:") {
                // Save previous point if exists
                if !current_statement.is_empty() {
                    let mut point = KeyPoint::new(&current_statement)
                        .with_importance(current_importance);
                    if let Some(ref cat) = current_category {
                        point = point.with_category(cat);
                    }
                    points.push(point);
                }

                current_statement = statement.trim().to_string();
                current_importance = 0.5;
                current_category = None;
            } else if let Some(importance) = line.strip_prefix("IMPORTANCE:") {
                current_importance = match importance.trim().to_lowercase().as_str() {
                    "high" => 0.9,
                    "medium" => 0.6,
                    "low" => 0.3,
                    _ => 0.5,
                };
            } else if let Some(category) = line.strip_prefix("CATEGORY:") {
                let cat = category.trim();
                if !cat.eq_ignore_ascii_case("general") && !cat.is_empty() {
                    current_category = Some(cat.to_string());
                }
            }
        }

        // Don't forget the last point
        if !current_statement.is_empty() {
            let mut point =
                KeyPoint::new(&current_statement).with_importance(current_importance);
            if let Some(ref cat) = current_category {
                point = point.with_category(cat);
            }
            points.push(point);
        }

        // Sort by importance (descending)
        points.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

        points
    }

    /// Call the LLM with retry logic.
    async fn call_llm(&self, prompt: &str) -> Result<String> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let backoff = std::time::Duration::from_secs(1 << (attempt - 1));
                tokio::time::sleep(backoff).await;
            }

            let messages = vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }];

            match self.client.chat(messages).await {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        return Ok(choice.message.content.clone());
                    }
                    last_error = Some(Error::Synthesis("Empty response from LLM".to_string()));
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::Synthesis("LLM call failed".to_string())))
    }

    /// Count words in text.
    fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Estimate token count (rough approximation: ~4 chars per token).
    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // COMPRESSION LEVEL TESTS
    // =========================================================================

    #[test]
    fn test_compression_level_target_ratio() {
        assert!((CompressionLevel::Brief.target_ratio() - 0.25).abs() < f64::EPSILON);
        assert!((CompressionLevel::Standard.target_ratio() - 0.50).abs() < f64::EPSILON);
        assert!((CompressionLevel::Detailed.target_ratio() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compression_level_target_words() {
        assert_eq!(CompressionLevel::Brief.target_words(100), 25);
        assert_eq!(CompressionLevel::Standard.target_words(100), 50);
        assert_eq!(CompressionLevel::Detailed.target_words(100), 75);
    }

    #[test]
    fn test_compression_level_target_tokens() {
        assert_eq!(CompressionLevel::Brief.target_tokens(400), 100);
        assert_eq!(CompressionLevel::Standard.target_tokens(400), 200);
        assert_eq!(CompressionLevel::Detailed.target_tokens(400), 300);
    }

    #[test]
    fn test_compression_level_display() {
        assert_eq!(format!("{}", CompressionLevel::Brief), "Brief");
        assert_eq!(format!("{}", CompressionLevel::Standard), "Standard");
        assert_eq!(format!("{}", CompressionLevel::Detailed), "Detailed");
    }

    #[test]
    fn test_compression_level_from_str() {
        assert_eq!(
            "brief".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Brief
        );
        assert_eq!(
            "standard".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Standard
        );
        assert_eq!(
            "detailed".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Detailed
        );
        assert_eq!(
            "short".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Brief
        );
        assert_eq!(
            "verbose".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Detailed
        );
    }

    #[test]
    fn test_compression_level_from_str_invalid() {
        let result = "invalid".parse::<CompressionLevel>();
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_level_default() {
        assert_eq!(CompressionLevel::default(), CompressionLevel::Standard);
    }

    #[test]
    fn test_compression_level_description() {
        assert!(CompressionLevel::Brief.description().contains("25%"));
        assert!(CompressionLevel::Standard.description().contains("50%"));
        assert!(CompressionLevel::Detailed.description().contains("75%"));
    }

    // =========================================================================
    // KEY POINT TESTS
    // =========================================================================

    #[test]
    fn test_key_point_creation() {
        let point = KeyPoint::new("Test statement");
        assert_eq!(point.statement, "Test statement");
        assert!((point.importance - 0.5).abs() < f64::EPSILON);
        assert!(point.category.is_none());
        assert!(point.evidence.is_none());
    }

    #[test]
    fn test_key_point_builder() {
        let point = KeyPoint::new("Test")
            .with_importance(0.9)
            .with_category("Tech")
            .with_evidence("Because reasons");

        assert_eq!(point.importance, 0.9);
        assert_eq!(point.category, Some("Tech".to_string()));
        assert_eq!(point.evidence, Some("Because reasons".to_string()));
    }

    #[test]
    fn test_key_point_importance_clamping() {
        let point_high = KeyPoint::new("Test").with_importance(1.5);
        assert_eq!(point_high.importance, 1.0);

        let point_low = KeyPoint::new("Test").with_importance(-0.5);
        assert_eq!(point_low.importance, 0.0);
    }

    #[test]
    fn test_key_point_is_critical() {
        let critical = KeyPoint::new("Test").with_importance(0.8);
        assert!(critical.is_critical());

        let not_critical = KeyPoint::new("Test").with_importance(0.5);
        assert!(!not_critical.is_critical());
    }

    #[test]
    fn test_key_point_brief() {
        let short = KeyPoint::new("Short statement");
        assert_eq!(short.brief(), "Short statement");

        let long = KeyPoint::new("This is a very long statement that exceeds one hundred characters and should be truncated with an ellipsis at the end");
        assert!(long.brief().ends_with("..."));
        assert!(long.brief().len() <= 100);
    }

    #[test]
    fn test_key_point_display() {
        let point = KeyPoint::new("Display test");
        assert_eq!(format!("{}", point), "Display test");
    }

    // =========================================================================
    // SUMMARY TESTS
    // =========================================================================

    #[test]
    fn test_summary_achieved_target() {
        let achieved = Summary {
            content: "Test".to_string(),
            original_words: 100,
            summary_words: 50,
            compression_ratio: 0.50,
            level: CompressionLevel::Standard,
            duration_ms: 100,
            key_points: Vec::new(),
            tokens_used: 0,
        };
        assert!(achieved.achieved_target());

        let not_achieved = Summary {
            content: "Test".to_string(),
            original_words: 100,
            summary_words: 80,
            compression_ratio: 0.80,
            level: CompressionLevel::Standard,
            duration_ms: 100,
            key_points: Vec::new(),
            tokens_used: 0,
        };
        assert!(!not_achieved.achieved_target());
    }

    #[test]
    fn test_summary_compression_percentage() {
        let summary = Summary {
            content: "Test".to_string(),
            original_words: 100,
            summary_words: 25,
            compression_ratio: 0.25,
            level: CompressionLevel::Brief,
            duration_ms: 100,
            key_points: Vec::new(),
            tokens_used: 0,
        };
        assert_eq!(summary.compression_percentage(), "25.0%");
    }

    #[test]
    fn test_summary_words_saved() {
        let summary = Summary {
            content: "Test".to_string(),
            original_words: 100,
            summary_words: 25,
            compression_ratio: 0.25,
            level: CompressionLevel::Brief,
            duration_ms: 100,
            key_points: Vec::new(),
            tokens_used: 0,
        };
        assert_eq!(summary.words_saved(), 75);
    }

    #[test]
    fn test_summary_stats() {
        let summary = Summary {
            content: "Test".to_string(),
            original_words: 100,
            summary_words: 50,
            compression_ratio: 0.50,
            level: CompressionLevel::Standard,
            duration_ms: 250,
            key_points: Vec::new(),
            tokens_used: 0,
        };
        let stats = summary.stats();
        assert!(stats.contains("100"));
        assert!(stats.contains("50"));
        assert!(stats.contains("250ms"));
    }

    #[test]
    fn test_summary_display() {
        let summary = Summary {
            content: "Summary content here".to_string(),
            original_words: 100,
            summary_words: 50,
            compression_ratio: 0.50,
            level: CompressionLevel::Standard,
            duration_ms: 100,
            key_points: Vec::new(),
            tokens_used: 0,
        };
        assert_eq!(format!("{}", summary), "Summary content here");
    }

    // =========================================================================
    // CONFIGURATION TESTS
    // =========================================================================

    #[test]
    fn test_summarizer_config_default() {
        let config = SummarizerConfig::default();
        assert_eq!(config.max_retries, 3);
        assert!(!config.extract_key_points);
        assert_eq!(config.max_key_points, 10);
        assert_eq!(config.min_content_words, 50);
        assert!(!config.preserve_formatting);
    }

    #[test]
    fn test_summarizer_config_with_key_points() {
        let config = SummarizerConfig::with_key_points();
        assert!(config.extract_key_points);
    }

    #[test]
    fn test_summarizer_config_preserve_format() {
        let config = SummarizerConfig::preserve_format();
        assert!(config.preserve_formatting);
    }

    // =========================================================================
    // SUMMARIZER AGENT TESTS
    // =========================================================================

    #[test]
    fn test_count_words() {
        assert_eq!(SummarizerAgent::count_words("hello world"), 2);
        assert_eq!(SummarizerAgent::count_words("one two three four five"), 5);
        assert_eq!(SummarizerAgent::count_words(""), 0);
        assert_eq!(SummarizerAgent::count_words("   spaced   out   "), 2);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(SummarizerAgent::estimate_tokens("12345678"), 2);
        assert_eq!(SummarizerAgent::estimate_tokens(""), 0);
    }

    #[test]
    fn test_summarizer_agent_creation() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);
        assert_eq!(agent.config().max_retries, 3);
    }

    #[test]
    fn test_summarizer_agent_with_config() {
        let client = Arc::new(GeminiClient::new());
        let config = SummarizerConfig {
            max_retries: 5,
            ..Default::default()
        };
        let agent = SummarizerAgent::with_config(client, config);
        assert_eq!(agent.config().max_retries, 5);
    }

    #[test]
    fn test_parse_key_points() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);

        let response = r#"POINT: First important point
IMPORTANCE: high
CATEGORY: Technology

POINT: Second point here
IMPORTANCE: medium
CATEGORY: general

POINT: Third point
IMPORTANCE: low
CATEGORY: Science"#;

        let points = agent.parse_key_points(response);

        assert_eq!(points.len(), 3);

        // Should be sorted by importance (descending)
        assert!(points[0].importance > points[1].importance);
        assert!(points[1].importance > points[2].importance);

        assert_eq!(points[0].statement, "First important point");
        assert!(points[0].is_critical());

        assert_eq!(points[2].category, Some("Science".to_string()));
    }

    #[test]
    fn test_parse_key_points_empty() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);

        let points = agent.parse_key_points("");
        assert!(points.is_empty());
    }

    #[tokio::test]
    async fn test_summarize_empty_content() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);

        let result = agent.summarize("", CompressionLevel::Standard).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot summarize empty content"));
    }

    #[tokio::test]
    async fn test_summarize_whitespace_only() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);

        let result = agent.summarize("   \n\t  ", CompressionLevel::Standard).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_key_points_empty_content() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);

        let result = agent.extract_key_points("", 5).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot extract key points from empty content"));
    }

    #[tokio::test]
    async fn test_tldr_empty_content() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);

        let result = agent.tldr("").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot generate TL;DR for empty content"));
    }

    #[tokio::test]
    async fn test_summarize_short_content() {
        let client = Arc::new(GeminiClient::new());
        let config = SummarizerConfig {
            min_content_words: 100,
            ..Default::default()
        };
        let agent = SummarizerAgent::with_config(client, config);

        // Content with fewer words than min_content_words should be returned as-is
        let short_content = "This is a short sentence.";
        let result = agent.summarize(short_content, CompressionLevel::Standard).await;

        assert!(result.is_ok());
        let summary = result.unwrap();
        assert_eq!(summary.content, short_content);
        assert_eq!(summary.compression_ratio, 1.0);
    }

    #[test]
    fn test_build_summarize_prompt() {
        let client = Arc::new(GeminiClient::new());
        let agent = SummarizerAgent::new(client);

        let prompt = agent.build_summarize_prompt("Test content", CompressionLevel::Brief, 25);

        assert!(prompt.contains("Essence Distiller"));
        assert!(prompt.contains("Agent 14"));
        assert!(prompt.contains("Brief"));
        assert!(prompt.contains("25 words"));
        assert!(prompt.contains("CONATUS"));
        assert!(prompt.contains("RATIO"));
        assert!(prompt.contains("LAETITIA"));
        assert!(prompt.contains("NATURA"));
    }

    #[test]
    fn test_build_summarize_prompt_preserves_formatting() {
        let client = Arc::new(GeminiClient::new());
        let config = SummarizerConfig::preserve_format();
        let agent = SummarizerAgent::with_config(client, config);

        let prompt = agent.build_summarize_prompt("Test", CompressionLevel::Standard, 50);
        assert!(prompt.contains("Preserve any headers"));
    }
}
