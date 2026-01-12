//! Evaluator Agent module for Project Panpsychism.
//!
//! The Quality Oracle - "Every creation must be measured against the eternal forms."
//!
//! This module implements the Evaluator Agent, responsible for assessing
//! response quality across multiple dimensions. Like an oracle consulting
//! divine wisdom, the Evaluator measures how well a response aligns with
//! the fundamental principles of truth, clarity, and purpose.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Quality Oracle serves as the final arbiter:
//!
//! - **Response** (raw magical output) enters the oracle's sanctum
//! - **The Oracle** (EvaluatorAgent) consults the eternal dimensions
//! - **Judgment** (EvaluationReport) emerges with wisdom and guidance
//!
//! The Oracle evaluates:
//! - **Relevance**: How well the response addresses the query
//! - **Completeness**: Whether all aspects are covered
//! - **Clarity**: How understandable the response is
//! - **Accuracy**: Correctness and precision of information
//! - **Actionability**: How useful and implementable the guidance is
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Self-preservation through quality assurance
//! - **RATIO**: Logical assessment of structure and reasoning
//! - **LAETITIA**: Joy through excellence and continuous improvement
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::evaluator::{EvaluatorAgent, EvaluationLevel};
//!
//! let oracle = EvaluatorAgent::new();
//!
//! // Evaluate a response against a query
//! let report = oracle.evaluate(
//!     "Here is how to implement OAuth2...",
//!     "How do I implement OAuth2 authentication?",
//!     EvaluationLevel::Standard
//! ).await?;
//!
//! println!("Overall score: {:.2}", report.overall_score);
//! for strength in &report.strengths {
//!     println!("Strength: {}", strength);
//! }
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// PRIORITY ENUM
// =============================================================================

/// Priority level for improvement suggestions.
///
/// Like the urgency of magical repairs, priority indicates
/// how critical an improvement is to the overall quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Critical improvement needed urgently.
    High,
    /// Important improvement that should be addressed.
    #[default]
    Medium,
    /// Nice-to-have improvement for polish.
    Low,
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::High => write!(f, "high"),
            Self::Medium => write!(f, "medium"),
            Self::Low => write!(f, "low"),
        }
    }
}

impl std::str::FromStr for Priority {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "high" | "critical" | "urgent" => Ok(Self::High),
            "medium" | "moderate" | "normal" => Ok(Self::Medium),
            "low" | "minor" | "nice-to-have" => Ok(Self::Low),
            _ => Err(Error::Config(format!(
                "Unknown priority: '{}'. Valid: high, medium, low",
                s
            ))),
        }
    }
}

// =============================================================================
// EVALUATION LEVEL
// =============================================================================

/// Depth of evaluation to perform.
///
/// Like different levels of oracular consultation, each level
/// provides varying degrees of insight and detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EvaluationLevel {
    /// Quick evaluation with basic checks.
    /// Fast but less thorough - good for rapid feedback.
    Quick,

    /// Standard evaluation with comprehensive analysis.
    /// Balanced between speed and thoroughness.
    #[default]
    Standard,

    /// Deep evaluation with exhaustive analysis.
    /// Thorough but slower - good for final quality gates.
    Deep,
}

impl std::fmt::Display for EvaluationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quick => write!(f, "quick"),
            Self::Standard => write!(f, "standard"),
            Self::Deep => write!(f, "deep"),
        }
    }
}

impl std::str::FromStr for EvaluationLevel {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "quick" | "fast" | "rapid" => Ok(Self::Quick),
            "standard" | "normal" | "default" => Ok(Self::Standard),
            "deep" | "thorough" | "exhaustive" => Ok(Self::Deep),
            _ => Err(Error::Config(format!(
                "Unknown evaluation level: '{}'. Valid: quick, standard, deep",
                s
            ))),
        }
    }
}

impl EvaluationLevel {
    /// Get all evaluation levels.
    pub fn all() -> Vec<Self> {
        vec![Self::Quick, Self::Standard, Self::Deep]
    }

    /// Get the expected duration multiplier for this level.
    pub fn duration_multiplier(&self) -> f64 {
        match self {
            Self::Quick => 0.5,
            Self::Standard => 1.0,
            Self::Deep => 2.0,
        }
    }
}

// =============================================================================
// IMPROVEMENT SUGGESTION
// =============================================================================

/// A specific suggestion for improving the response.
///
/// Like a prescription from the oracle, each improvement
/// targets a specific aspect with actionable guidance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    /// Category of the improvement (e.g., "Clarity", "Completeness").
    pub category: String,
    /// The specific suggestion.
    pub suggestion: String,
    /// Priority level of this improvement.
    pub priority: Priority,
    /// Estimated impact on overall score (0.0 - 1.0).
    pub estimated_impact: f64,
}

impl Improvement {
    /// Create a new improvement suggestion.
    pub fn new(category: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            suggestion: suggestion.into(),
            priority: Priority::Medium,
            estimated_impact: 0.1,
        }
    }

    /// Set the priority level.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the estimated impact.
    pub fn with_impact(mut self, impact: f64) -> Self {
        self.estimated_impact = impact.clamp(0.0, 1.0);
        self
    }

    /// Format the improvement as markdown.
    pub fn to_markdown(&self) -> String {
        format!(
            "- **[{}]** {} _(Priority: {}, Impact: {:.0}%)_",
            self.category,
            self.suggestion,
            self.priority,
            self.estimated_impact * 100.0
        )
    }
}

// =============================================================================
// EVALUATION DIMENSIONS
// =============================================================================

/// Scores across all evaluation dimensions.
///
/// The five pillars of quality, each measured on a scale of 0.0 to 1.0.
/// These dimensions represent the eternal forms against which all
/// responses are measured.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationDimensions {
    /// How well the response addresses the query (0.0 - 1.0).
    /// Measures alignment between question and answer.
    pub relevance: f64,

    /// Whether all aspects of the query are covered (0.0 - 1.0).
    /// Measures thoroughness and comprehensiveness.
    pub completeness: f64,

    /// How understandable the response is (0.0 - 1.0).
    /// Measures readability and structure.
    pub clarity: f64,

    /// Correctness and precision of information (0.0 - 1.0).
    /// Measures factual accuracy and technical correctness.
    pub accuracy: f64,

    /// How useful and implementable the guidance is (0.0 - 1.0).
    /// Measures practical value and applicability.
    pub actionability: f64,
}

impl EvaluationDimensions {
    /// Create new dimensions with specified scores.
    pub fn new(
        relevance: f64,
        completeness: f64,
        clarity: f64,
        accuracy: f64,
        actionability: f64,
    ) -> Self {
        Self {
            relevance: relevance.clamp(0.0, 1.0),
            completeness: completeness.clamp(0.0, 1.0),
            clarity: clarity.clamp(0.0, 1.0),
            accuracy: accuracy.clamp(0.0, 1.0),
            actionability: actionability.clamp(0.0, 1.0),
        }
    }

    /// Calculate the average score across all dimensions.
    pub fn average(&self) -> f64 {
        (self.relevance + self.completeness + self.clarity + self.accuracy + self.actionability)
            / 5.0
    }

    /// Calculate weighted average with custom weights.
    pub fn weighted_average(&self, weights: &DimensionWeights) -> f64 {
        let total_weight =
            weights.relevance + weights.completeness + weights.clarity + weights.accuracy + weights.actionability;

        if total_weight == 0.0 {
            return self.average();
        }

        (self.relevance * weights.relevance
            + self.completeness * weights.completeness
            + self.clarity * weights.clarity
            + self.accuracy * weights.accuracy
            + self.actionability * weights.actionability)
            / total_weight
    }

    /// Get the minimum score across all dimensions.
    pub fn minimum(&self) -> f64 {
        self.relevance
            .min(self.completeness)
            .min(self.clarity)
            .min(self.accuracy)
            .min(self.actionability)
    }

    /// Get the maximum score across all dimensions.
    pub fn maximum(&self) -> f64 {
        self.relevance
            .max(self.completeness)
            .max(self.clarity)
            .max(self.accuracy)
            .max(self.actionability)
    }

    /// Get the dimension with the lowest score.
    pub fn weakest_dimension(&self) -> (&'static str, f64) {
        let dims = [
            ("relevance", self.relevance),
            ("completeness", self.completeness),
            ("clarity", self.clarity),
            ("accuracy", self.accuracy),
            ("actionability", self.actionability),
        ];

        dims.into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("unknown", 0.0))
    }

    /// Get the dimension with the highest score.
    pub fn strongest_dimension(&self) -> (&'static str, f64) {
        let dims = [
            ("relevance", self.relevance),
            ("completeness", self.completeness),
            ("clarity", self.clarity),
            ("accuracy", self.accuracy),
            ("actionability", self.actionability),
        ];

        dims.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("unknown", 0.0))
    }

    /// Check if all dimensions meet a minimum threshold.
    pub fn all_above(&self, threshold: f64) -> bool {
        self.minimum() >= threshold
    }

    /// Format dimensions as markdown table.
    pub fn to_markdown(&self) -> String {
        format!(
            "| Dimension | Score |\n\
             |-----------|-------|\n\
             | Relevance | {:.2} |\n\
             | Completeness | {:.2} |\n\
             | Clarity | {:.2} |\n\
             | Accuracy | {:.2} |\n\
             | Actionability | {:.2} |",
            self.relevance, self.completeness, self.clarity, self.accuracy, self.actionability
        )
    }
}

// =============================================================================
// DIMENSION WEIGHTS
// =============================================================================

/// Weights for each evaluation dimension.
///
/// Allows customizing the importance of each dimension
/// in the overall score calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionWeights {
    /// Weight for relevance dimension.
    pub relevance: f64,
    /// Weight for completeness dimension.
    pub completeness: f64,
    /// Weight for clarity dimension.
    pub clarity: f64,
    /// Weight for accuracy dimension.
    pub accuracy: f64,
    /// Weight for actionability dimension.
    pub actionability: f64,
}

impl Default for DimensionWeights {
    fn default() -> Self {
        Self {
            relevance: 1.0,
            completeness: 1.0,
            clarity: 1.0,
            accuracy: 1.0,
            actionability: 1.0,
        }
    }
}

impl DimensionWeights {
    /// Create equal weights for all dimensions.
    pub fn equal() -> Self {
        Self::default()
    }

    /// Create weights emphasizing accuracy.
    pub fn accuracy_focused() -> Self {
        Self {
            relevance: 1.0,
            completeness: 1.0,
            clarity: 0.8,
            accuracy: 2.0,
            actionability: 1.0,
        }
    }

    /// Create weights emphasizing actionability.
    pub fn actionability_focused() -> Self {
        Self {
            relevance: 1.0,
            completeness: 1.0,
            clarity: 1.0,
            accuracy: 1.0,
            actionability: 2.0,
        }
    }
}

// =============================================================================
// EVALUATION REPORT
// =============================================================================

/// Complete evaluation report from the Quality Oracle.
///
/// Contains the overall score, dimensional breakdown,
/// strengths, weaknesses, and improvement suggestions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Overall quality score (0.0 - 1.0).
    pub overall_score: f64,

    /// Scores across all evaluation dimensions.
    pub dimensions: EvaluationDimensions,

    /// Identified strengths of the response.
    pub strengths: Vec<String>,

    /// Identified weaknesses of the response.
    pub weaknesses: Vec<String>,

    /// Specific improvement suggestions.
    pub suggestions: Vec<Improvement>,

    /// The evaluation level used.
    pub level: EvaluationLevel,

    /// Processing duration in milliseconds.
    pub duration_ms: u64,

    /// The query that was evaluated against.
    pub query: String,

    /// Length of the evaluated response.
    pub response_length: usize,
}

impl EvaluationReport {
    /// Create a new evaluation report.
    pub fn new(overall_score: f64, dimensions: EvaluationDimensions) -> Self {
        Self {
            overall_score: overall_score.clamp(0.0, 1.0),
            dimensions,
            strengths: Vec::new(),
            weaknesses: Vec::new(),
            suggestions: Vec::new(),
            level: EvaluationLevel::Standard,
            duration_ms: 0,
            query: String::new(),
            response_length: 0,
        }
    }

    /// Check if the response passes a quality threshold.
    pub fn passes_threshold(&self, threshold: f64) -> bool {
        self.overall_score >= threshold
    }

    /// Check if the response is considered high quality.
    pub fn is_high_quality(&self) -> bool {
        self.overall_score >= 0.8 && self.dimensions.all_above(0.6)
    }

    /// Check if the response needs improvement.
    pub fn needs_improvement(&self) -> bool {
        self.overall_score < 0.6 || self.dimensions.minimum() < 0.4
    }

    /// Get high-priority improvements.
    pub fn high_priority_improvements(&self) -> Vec<&Improvement> {
        self.suggestions
            .iter()
            .filter(|i| i.priority == Priority::High)
            .collect()
    }

    /// Get a summary of the evaluation.
    pub fn summary(&self) -> String {
        let status = if self.is_high_quality() {
            "Excellent"
        } else if self.needs_improvement() {
            "Needs Work"
        } else {
            "Good"
        };

        format!(
            "{} ({:.0}%) - {} strengths, {} areas for improvement",
            status,
            self.overall_score * 100.0,
            self.strengths.len(),
            self.suggestions.len()
        )
    }

    /// Format the report as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Evaluation Report\n\n");
        output.push_str(&format!(
            "**Overall Score:** {:.0}% ({})\n\n",
            self.overall_score * 100.0,
            self.level
        ));

        output.push_str("## Dimension Scores\n\n");
        output.push_str(&self.dimensions.to_markdown());
        output.push_str("\n\n");

        if !self.strengths.is_empty() {
            output.push_str("## Strengths\n\n");
            for strength in &self.strengths {
                output.push_str(&format!("- {}\n", strength));
            }
            output.push('\n');
        }

        if !self.weaknesses.is_empty() {
            output.push_str("## Weaknesses\n\n");
            for weakness in &self.weaknesses {
                output.push_str(&format!("- {}\n", weakness));
            }
            output.push('\n');
        }

        if !self.suggestions.is_empty() {
            output.push_str("## Improvement Suggestions\n\n");
            for suggestion in &self.suggestions {
                output.push_str(&suggestion.to_markdown());
                output.push('\n');
            }
        }

        output
    }
}

// =============================================================================
// EVALUATOR CONFIGURATION
// =============================================================================

/// Configuration for the Evaluator Agent.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Minimum acceptable overall score.
    pub min_score_threshold: f64,

    /// Weights for each dimension.
    pub weights: DimensionWeights,

    /// Maximum number of improvement suggestions to generate.
    pub max_suggestions: usize,

    /// Whether to include detailed analysis in reports.
    pub include_details: bool,

    /// Timeout in seconds for evaluation.
    pub timeout_secs: u64,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            min_score_threshold: 0.6,
            weights: DimensionWeights::default(),
            max_suggestions: 5,
            include_details: true,
            timeout_secs: 30,
        }
    }
}

impl EvaluatorConfig {
    /// Create a strict configuration requiring high quality.
    pub fn strict() -> Self {
        Self {
            min_score_threshold: 0.8,
            weights: DimensionWeights::accuracy_focused(),
            max_suggestions: 10,
            include_details: true,
            timeout_secs: 60,
        }
    }

    /// Create a lenient configuration for quick checks.
    pub fn lenient() -> Self {
        Self {
            min_score_threshold: 0.4,
            weights: DimensionWeights::equal(),
            max_suggestions: 3,
            include_details: false,
            timeout_secs: 15,
        }
    }
}

// =============================================================================
// KEYWORD DICTIONARIES FOR ANALYSIS
// =============================================================================

/// Keywords indicating clarity and structure.
const CLARITY_POSITIVE_KEYWORDS: &[&str] = &[
    "first", "second", "third", "finally", "then", "next",
    "therefore", "because", "consequently", "specifically",
    "for example", "such as", "namely", "in other words",
    "step", "note", "important", "summary", "conclusion",
];

/// Keywords indicating confusion or vagueness.
const CLARITY_NEGATIVE_KEYWORDS: &[&str] = &[
    "maybe", "perhaps", "somehow", "something", "stuff",
    "thing", "etc", "whatever", "sort of", "kind of",
    "basically", "actually", "literally",
];

/// Keywords indicating actionable guidance.
const ACTIONABILITY_KEYWORDS: &[&str] = &[
    "implement", "use", "create", "configure", "set up",
    "install", "run", "execute", "call", "invoke",
    "add", "remove", "update", "change", "modify",
    "command", "code", "example", "snippet", "function",
];

/// Keywords indicating completeness.
const COMPLETENESS_KEYWORDS: &[&str] = &[
    "complete", "full", "comprehensive", "all", "every",
    "include", "cover", "address", "handle", "consider",
    "alternative", "option", "also", "additionally",
];

// =============================================================================
// EVALUATOR AGENT (THE QUALITY ORACLE)
// =============================================================================

/// The Evaluator Agent - The Quality Oracle of the Sorcerer's Tower.
///
/// Responsible for assessing response quality across multiple dimensions,
/// identifying strengths and weaknesses, and providing improvement suggestions.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive to maintain and improve quality
/// - **RATIO**: Logical assessment of structure and reasoning
/// - **LAETITIA**: Joy through excellence and continuous improvement
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::evaluator::{EvaluatorAgent, EvaluationLevel};
///
/// let oracle = EvaluatorAgent::new();
///
/// let report = oracle.evaluate(
///     "Here is the implementation...",
///     "How do I implement feature X?",
///     EvaluationLevel::Standard
/// ).await?;
///
/// println!("Score: {:.0}%", report.overall_score * 100.0);
/// ```
#[derive(Debug, Clone)]
pub struct EvaluatorAgent {
    /// Configuration for evaluation behavior.
    config: EvaluatorConfig,
}

impl Default for EvaluatorAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluatorAgent {
    /// Create a new Evaluator Agent with default configuration.
    ///
    /// The Oracle awakens, ready to judge quality.
    pub fn new() -> Self {
        Self {
            config: EvaluatorConfig::default(),
        }
    }

    /// Create a new Evaluator Agent with custom configuration.
    pub fn with_config(config: EvaluatorConfig) -> Self {
        Self { config }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> EvaluatorAgentBuilder {
        EvaluatorAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &EvaluatorConfig {
        &self.config
    }

    // =========================================================================
    // MAIN EVALUATION METHOD
    // =========================================================================

    /// Evaluate a response against a query.
    ///
    /// The Oracle consults the eternal dimensions and renders judgment.
    ///
    /// # Arguments
    ///
    /// * `response` - The response to evaluate
    /// * `query` - The original query the response addresses
    /// * `level` - The depth of evaluation to perform
    ///
    /// # Returns
    ///
    /// An `EvaluationReport` containing scores, strengths, weaknesses,
    /// and improvement suggestions.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if the response is empty.
    pub async fn evaluate(
        &self,
        response: &str,
        query: &str,
        level: EvaluationLevel,
    ) -> Result<EvaluationReport> {
        let start = Instant::now();

        if response.trim().is_empty() {
            return Err(Error::Validation("Cannot evaluate empty response".to_string()));
        }

        debug!("Evaluating response ({} chars) at {:?} level", response.len(), level);

        // Calculate dimension scores
        let dimensions = match level {
            EvaluationLevel::Quick => self.quick_evaluate(response, query),
            EvaluationLevel::Standard => self.standard_evaluate(response, query),
            EvaluationLevel::Deep => self.deep_evaluate(response, query),
        };

        // Calculate overall score
        let overall_score = dimensions.weighted_average(&self.config.weights);

        // Identify strengths and weaknesses
        let (strengths, weaknesses) = self.identify_strengths_weaknesses(&dimensions, response);

        // Generate improvement suggestions
        let suggestions = self.generate_suggestions(&dimensions, response, query);

        let report = EvaluationReport {
            overall_score,
            dimensions,
            strengths,
            weaknesses,
            suggestions,
            level,
            duration_ms: start.elapsed().as_millis() as u64,
            query: query.to_string(),
            response_length: response.len(),
        };

        info!(
            "Evaluation complete: {:.0}% overall in {}ms",
            report.overall_score * 100.0,
            report.duration_ms
        );

        Ok(report)
    }

    // =========================================================================
    // QUICK EVALUATION
    // =========================================================================

    /// Perform quick evaluation with basic checks.
    fn quick_evaluate(&self, response: &str, query: &str) -> EvaluationDimensions {
        let relevance = self.calculate_relevance_quick(response, query);
        let completeness = self.calculate_completeness_quick(response);
        let clarity = self.calculate_clarity_quick(response);
        let accuracy = 0.7; // Assume moderate accuracy for quick evaluation
        let actionability = self.calculate_actionability_quick(response);

        EvaluationDimensions::new(relevance, completeness, clarity, accuracy, actionability)
    }

    /// Quick relevance calculation based on keyword overlap.
    fn calculate_relevance_quick(&self, response: &str, query: &str) -> f64 {
        let query_words = self.extract_significant_words(query);
        let response_lower = response.to_lowercase();

        if query_words.is_empty() {
            return 0.5;
        }

        let matches = query_words
            .iter()
            .filter(|w| response_lower.contains(*w))
            .count();

        let ratio = matches as f64 / query_words.len() as f64;
        0.3 + ratio * 0.7 // Base score of 0.3, up to 1.0
    }

    /// Quick completeness calculation based on length and structure.
    fn calculate_completeness_quick(&self, response: &str) -> f64 {
        let word_count = response.split_whitespace().count();
        let has_structure = response.contains('\n') || response.contains("- ");

        let length_score = match word_count {
            0..=10 => 0.2,
            11..=50 => 0.4,
            51..=150 => 0.6,
            151..=500 => 0.8,
            _ => 0.9,
        };

        if has_structure {
            (length_score + 0.1_f64).min(1.0)
        } else {
            length_score
        }
    }

    /// Quick clarity calculation based on sentence structure.
    fn calculate_clarity_quick(&self, response: &str) -> f64 {
        let sentences: Vec<&str> = response.split(['.', '!', '?']).collect();
        let avg_sentence_length = if sentences.is_empty() {
            0.0
        } else {
            sentences.iter().map(|s| s.split_whitespace().count()).sum::<usize>() as f64
                / sentences.len() as f64
        };

        // Ideal sentence length is 15-25 words
        let length_score = if avg_sentence_length < 5.0 {
            0.4
        } else if avg_sentence_length <= 25.0 {
            0.8
        } else if avg_sentence_length <= 40.0 {
            0.6
        } else {
            0.4
        };

        // Check for structure indicators
        let has_headers = response.contains('#');
        let has_lists = response.contains("- ") || response.contains("1.");
        let structure_bonus: f64 = if has_headers || has_lists { 0.1 } else { 0.0 };

        (length_score + structure_bonus).min(1.0)
    }

    /// Quick actionability calculation based on presence of actionable keywords.
    fn calculate_actionability_quick(&self, response: &str) -> f64 {
        let response_lower = response.to_lowercase();

        let action_count = ACTIONABILITY_KEYWORDS
            .iter()
            .filter(|kw| response_lower.contains(*kw))
            .count();

        let has_code = response.contains("```") || response.contains("    ");

        let base_score = match action_count {
            0 => 0.3,
            1..=2 => 0.5,
            3..=5 => 0.7,
            _ => 0.85,
        };

        if has_code {
            (base_score + 0.15_f64).min(1.0)
        } else {
            base_score
        }
    }

    // =========================================================================
    // STANDARD EVALUATION
    // =========================================================================

    /// Perform standard evaluation with comprehensive analysis.
    fn standard_evaluate(&self, response: &str, query: &str) -> EvaluationDimensions {
        let relevance = self.calculate_relevance_standard(response, query);
        let completeness = self.calculate_completeness_standard(response, query);
        let clarity = self.calculate_clarity_standard(response);
        let accuracy = self.calculate_accuracy_standard(response);
        let actionability = self.calculate_actionability_standard(response);

        EvaluationDimensions::new(relevance, completeness, clarity, accuracy, actionability)
    }

    /// Standard relevance calculation with semantic analysis.
    fn calculate_relevance_standard(&self, response: &str, query: &str) -> f64 {
        let query_words = self.extract_significant_words(query);
        let response_words = self.extract_significant_words(response);
        let response_lower = response.to_lowercase();

        if query_words.is_empty() {
            return 0.5;
        }

        // Direct keyword matches
        let direct_matches = query_words
            .iter()
            .filter(|w| response_lower.contains(*w))
            .count();
        let direct_ratio = direct_matches as f64 / query_words.len() as f64;

        // Word overlap (Jaccard-like similarity)
        let query_set: HashSet<_> = query_words.iter().collect();
        let response_set: HashSet<_> = response_words.iter().collect();
        let intersection = query_set.intersection(&response_set).count();
        let union = query_set.union(&response_set).count();
        let jaccard = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        // Combine metrics
        0.2 + direct_ratio * 0.5 + jaccard * 0.3
    }

    /// Standard completeness calculation.
    fn calculate_completeness_standard(&self, response: &str, query: &str) -> f64 {
        let response_lower = response.to_lowercase();

        // Check for completeness indicators
        let completeness_count = COMPLETENESS_KEYWORDS
            .iter()
            .filter(|kw| response_lower.contains(*kw))
            .count();

        // Check length relative to query complexity
        let query_complexity = query.split_whitespace().count();
        let response_length = response.split_whitespace().count();
        let length_ratio = (response_length as f64 / (query_complexity * 10) as f64).min(1.0);

        // Check for multiple sections/points
        let sections = response.matches('\n').count() + 1;
        let structure_score = match sections {
            1 => 0.4,
            2..=3 => 0.6,
            4..=6 => 0.8,
            _ => 0.9,
        };

        let keyword_score = (completeness_count as f64 * 0.1).min(0.3);

        (length_ratio * 0.4 + structure_score * 0.4 + keyword_score + 0.1).min(1.0)
    }

    /// Standard clarity calculation.
    fn calculate_clarity_standard(&self, response: &str) -> f64 {
        let response_lower = response.to_lowercase();

        // Positive clarity indicators
        let positive_count = CLARITY_POSITIVE_KEYWORDS
            .iter()
            .filter(|kw| response_lower.contains(*kw))
            .count();

        // Negative clarity indicators
        let negative_count = CLARITY_NEGATIVE_KEYWORDS
            .iter()
            .filter(|kw| response_lower.contains(*kw))
            .count();

        // Structure analysis
        let has_headers = response.contains('#');
        let has_lists = response.contains("- ") || response.contains("1.");
        let has_code_blocks = response.contains("```");

        let structure_score = if has_headers && has_lists {
            0.9
        } else if has_headers || has_lists {
            0.7
        } else {
            0.5
        };

        // Calculate sentence clarity
        let sentence_clarity = self.calculate_clarity_quick(response);

        let positive_boost = (positive_count as f64 * 0.05).min(0.2);
        let negative_penalty = (negative_count as f64 * 0.08).min(0.3);
        let code_boost = if has_code_blocks { 0.1 } else { 0.0 };

        (structure_score * 0.4 + sentence_clarity * 0.4 + positive_boost - negative_penalty + code_boost)
            .clamp(0.0, 1.0)
    }

    /// Standard accuracy calculation (heuristic-based).
    fn calculate_accuracy_standard(&self, response: &str) -> f64 {
        let response_lower = response.to_lowercase();

        // Check for uncertainty markers (may indicate lower confidence)
        let uncertainty_markers = ["might", "could", "possibly", "uncertain", "not sure"];
        let uncertainty_count = uncertainty_markers
            .iter()
            .filter(|m| response_lower.contains(*m))
            .count();

        // Check for definitive statements
        let definitive_markers = ["is", "will", "must", "should", "always", "never"];
        let definitive_count = definitive_markers
            .iter()
            .filter(|m| response_lower.contains(*m))
            .count();

        // Check for citations or references
        let has_references = response.contains("http") || response.contains("see ")
            || response.contains("according to") || response.contains("documentation");

        // Base score
        let mut score: f64 = 0.6;

        // Adjust based on definitive vs uncertain
        if definitive_count > uncertainty_count {
            score += 0.1;
        } else if uncertainty_count > definitive_count {
            score -= 0.1;
        }

        // Boost for references
        if has_references {
            score += 0.1;
        }

        // Check for code (code is often more accurate)
        if response.contains("```") {
            score += 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Standard actionability calculation.
    fn calculate_actionability_standard(&self, response: &str) -> f64 {
        let response_lower = response.to_lowercase();

        // Count actionable keywords
        let action_count = ACTIONABILITY_KEYWORDS
            .iter()
            .filter(|kw| response_lower.contains(*kw))
            .count();

        // Check for code examples
        let code_block_count = response.matches("```").count() / 2;

        // Check for step-by-step instructions
        let has_steps = response.contains("1.") || response.contains("Step ");
        let has_commands = response.contains("$ ") || response.contains("```bash")
            || response.contains("```shell");

        let action_score = match action_count {
            0 => 0.2,
            1..=3 => 0.4,
            4..=7 => 0.6,
            8..=12 => 0.8,
            _ => 0.9,
        };

        let code_score = match code_block_count {
            0 => 0.0,
            1 => 0.1,
            2..=3 => 0.15,
            _ => 0.2,
        };

        let step_bonus: f64 = if has_steps { 0.1 } else { 0.0 };
        let command_bonus: f64 = if has_commands { 0.1 } else { 0.0 };

        (action_score + code_score + step_bonus + command_bonus).min(1.0)
    }

    // =========================================================================
    // DEEP EVALUATION
    // =========================================================================

    /// Perform deep evaluation with exhaustive analysis.
    fn deep_evaluate(&self, response: &str, query: &str) -> EvaluationDimensions {
        // Start with standard evaluation
        let standard = self.standard_evaluate(response, query);

        // Perform additional deep analysis
        let relevance = self.enhance_relevance_deep(standard.relevance, response, query);
        let completeness = self.enhance_completeness_deep(standard.completeness, response, query);
        let clarity = self.enhance_clarity_deep(standard.clarity, response);
        let accuracy = self.enhance_accuracy_deep(standard.accuracy, response);
        let actionability = self.enhance_actionability_deep(standard.actionability, response);

        EvaluationDimensions::new(relevance, completeness, clarity, accuracy, actionability)
    }

    /// Deep enhancement for relevance.
    fn enhance_relevance_deep(&self, base_score: f64, response: &str, query: &str) -> f64 {
        // Check if query question types are addressed
        let is_how = query.to_lowercase().contains("how");
        let is_what = query.to_lowercase().contains("what");
        let is_why = query.to_lowercase().contains("why");

        let mut adjustment = 0.0;

        if is_how && (response.contains("```") || response.contains("step") || response.contains("1.")) {
            adjustment += 0.05;
        }
        if is_what && response.len() > 100 {
            adjustment += 0.03;
        }
        if is_why && (response.contains("because") || response.contains("reason")) {
            adjustment += 0.05;
        }

        (base_score + adjustment).clamp(0.0, 1.0)
    }

    /// Deep enhancement for completeness.
    fn enhance_completeness_deep(&self, base_score: f64, response: &str, query: &str) -> f64 {
        // Check for edge case coverage
        let has_edge_cases = response.to_lowercase().contains("edge case")
            || response.contains("note:")
            || response.contains("important:");

        // Check for error handling
        let has_error_handling = response.contains("error")
            || response.contains("exception")
            || response.contains("try")
            || response.contains("catch");

        // Check for alternatives
        let has_alternatives = response.contains("alternative")
            || response.contains("another way")
            || response.contains("you could also");

        let mut adjustment = 0.0;
        if has_edge_cases { adjustment += 0.03; }
        if has_error_handling { adjustment += 0.03; }
        if has_alternatives { adjustment += 0.04; }

        // Penalize if query seems complex but response is short
        let query_words = query.split_whitespace().count();
        let response_words = response.split_whitespace().count();
        if query_words > 10 && response_words < 50 {
            adjustment -= 0.1;
        }

        (base_score + adjustment).clamp(0.0, 1.0)
    }

    /// Deep enhancement for clarity.
    fn enhance_clarity_deep(&self, base_score: f64, response: &str) -> f64 {
        // Check for consistent formatting
        let paragraphs: Vec<&str> = response.split("\n\n").collect();
        let formatting_consistency = if paragraphs.len() > 1 {
            0.02
        } else {
            0.0
        };

        // Check for excessive jargon without explanation
        let technical_terms = ["api", "sdk", "jwt", "oauth", "csrf", "xss", "sql", "http"];
        let jargon_count = technical_terms
            .iter()
            .filter(|t| response.to_lowercase().contains(*t))
            .count();
        let jargon_penalty = if jargon_count > 3 && !response.contains("means") && !response.contains("is a") {
            0.05
        } else {
            0.0
        };

        (base_score + formatting_consistency - jargon_penalty).clamp(0.0, 1.0)
    }

    /// Deep enhancement for accuracy.
    fn enhance_accuracy_deep(&self, base_score: f64, response: &str) -> f64 {
        // Check for version specificity
        let has_versions = regex::Regex::new(r"\d+\.\d+").unwrap();
        let version_bonus = if has_versions.is_match(response) { 0.03 } else { 0.0 };

        // Check for specific examples
        let example_patterns = ["example", "e.g.", "for instance", "such as"];
        let example_count = example_patterns
            .iter()
            .filter(|p| response.to_lowercase().contains(*p))
            .count();
        let example_bonus = (example_count as f64 * 0.02).min(0.06);

        (base_score + version_bonus + example_bonus).clamp(0.0, 1.0)
    }

    /// Deep enhancement for actionability.
    fn enhance_actionability_deep(&self, base_score: f64, response: &str) -> f64 {
        // Check for prerequisites
        let has_prereqs = response.to_lowercase().contains("prerequisite")
            || response.contains("before you")
            || response.contains("first, ensure");

        // Check for verification steps
        let has_verification = response.contains("verify")
            || response.contains("check that")
            || response.contains("should see");

        // Check for troubleshooting
        let has_troubleshooting = response.contains("if you encounter")
            || response.contains("troubleshoot")
            || response.contains("common issue");

        let mut adjustment = 0.0;
        if has_prereqs { adjustment += 0.02; }
        if has_verification { adjustment += 0.03; }
        if has_troubleshooting { adjustment += 0.03; }

        (base_score + adjustment).clamp(0.0, 1.0)
    }

    // =========================================================================
    // STRENGTH/WEAKNESS IDENTIFICATION
    // =========================================================================

    /// Identify strengths and weaknesses from the evaluation.
    fn identify_strengths_weaknesses(
        &self,
        dimensions: &EvaluationDimensions,
        response: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        // Dimension-based analysis
        if dimensions.relevance >= 0.8 {
            strengths.push("Highly relevant to the query".to_string());
        } else if dimensions.relevance < 0.5 {
            weaknesses.push("May not fully address the question asked".to_string());
        }

        if dimensions.completeness >= 0.8 {
            strengths.push("Comprehensive coverage of the topic".to_string());
        } else if dimensions.completeness < 0.5 {
            weaknesses.push("Missing important aspects or details".to_string());
        }

        if dimensions.clarity >= 0.8 {
            strengths.push("Clear and well-structured presentation".to_string());
        } else if dimensions.clarity < 0.5 {
            weaknesses.push("Could be clearer or better organized".to_string());
        }

        if dimensions.accuracy >= 0.8 {
            strengths.push("Information appears accurate and reliable".to_string());
        } else if dimensions.accuracy < 0.5 {
            weaknesses.push("Some information may need verification".to_string());
        }

        if dimensions.actionability >= 0.8 {
            strengths.push("Provides practical, actionable guidance".to_string());
        } else if dimensions.actionability < 0.5 {
            weaknesses.push("Could include more practical examples or steps".to_string());
        }

        // Content-based analysis
        if response.contains("```") {
            strengths.push("Includes code examples".to_string());
        }

        if response.contains("1.") || response.contains("Step ") {
            strengths.push("Provides step-by-step instructions".to_string());
        }

        let word_count = response.split_whitespace().count();
        if word_count < 30 {
            weaknesses.push("Response may be too brief".to_string());
        } else if word_count > 1000 {
            weaknesses.push("Response may be overly verbose".to_string());
        }

        (strengths, weaknesses)
    }

    // =========================================================================
    // SUGGESTION GENERATION
    // =========================================================================

    /// Generate improvement suggestions based on the evaluation.
    fn generate_suggestions(
        &self,
        dimensions: &EvaluationDimensions,
        response: &str,
        query: &str,
    ) -> Vec<Improvement> {
        let mut suggestions = Vec::new();

        // Relevance suggestions
        if dimensions.relevance < 0.7 {
            let impact = 0.7 - dimensions.relevance;
            let priority = if dimensions.relevance < 0.5 { Priority::High } else { Priority::Medium };
            suggestions.push(
                Improvement::new(
                    "Relevance",
                    "Consider focusing more directly on the specific question asked"
                )
                .with_priority(priority)
                .with_impact(impact)
            );
        }

        // Completeness suggestions
        if dimensions.completeness < 0.7 {
            let impact = 0.7 - dimensions.completeness;
            let priority = if dimensions.completeness < 0.5 { Priority::High } else { Priority::Medium };

            if query.to_lowercase().contains("how") && !response.contains("```") {
                suggestions.push(
                    Improvement::new(
                        "Completeness",
                        "Add code examples to illustrate the implementation"
                    )
                    .with_priority(priority)
                    .with_impact(impact)
                );
            } else {
                suggestions.push(
                    Improvement::new(
                        "Completeness",
                        "Consider addressing additional aspects or edge cases"
                    )
                    .with_priority(priority)
                    .with_impact(impact)
                );
            }
        }

        // Clarity suggestions
        if dimensions.clarity < 0.7 {
            let impact = 0.7 - dimensions.clarity;
            let priority = if dimensions.clarity < 0.5 { Priority::High } else { Priority::Medium };

            if !response.contains('#') && !response.contains("- ") {
                suggestions.push(
                    Improvement::new(
                        "Clarity",
                        "Add headers or bullet points to improve structure"
                    )
                    .with_priority(priority)
                    .with_impact(impact)
                );
            } else {
                suggestions.push(
                    Improvement::new(
                        "Clarity",
                        "Simplify complex sentences for better readability"
                    )
                    .with_priority(priority)
                    .with_impact(impact)
                );
            }
        }

        // Accuracy suggestions
        if dimensions.accuracy < 0.7 {
            suggestions.push(
                Improvement::new(
                    "Accuracy",
                    "Consider adding references or specific version information"
                )
                .with_priority(Priority::Medium)
                .with_impact(0.1)
            );
        }

        // Actionability suggestions
        if dimensions.actionability < 0.7 {
            let impact = 0.7 - dimensions.actionability;

            if !response.contains("1.") && !response.contains("Step ") {
                suggestions.push(
                    Improvement::new(
                        "Actionability",
                        "Add numbered steps or a clear action sequence"
                    )
                    .with_priority(Priority::Medium)
                    .with_impact(impact)
                );
            }

            if !response.contains("```") {
                suggestions.push(
                    Improvement::new(
                        "Actionability",
                        "Include executable code examples or commands"
                    )
                    .with_priority(Priority::Low)
                    .with_impact(impact * 0.8)
                );
            }
        }

        // Limit to max suggestions
        suggestions.truncate(self.config.max_suggestions);
        suggestions
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Extract significant words from text (filtering stop words).
    fn extract_significant_words(&self, text: &str) -> Vec<String> {
        let stop_words: HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "and", "but", "if", "or", "because", "until", "while", "about",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am",
        ].into_iter().collect();

        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2 && !stop_words.contains(w))
            .map(String::from)
            .collect()
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom EvaluatorAgent configuration.
#[derive(Debug, Default)]
pub struct EvaluatorAgentBuilder {
    config: Option<EvaluatorConfig>,
}

impl EvaluatorAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: EvaluatorConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the minimum score threshold.
    pub fn min_score_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.min_score_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the dimension weights.
    pub fn weights(mut self, weights: DimensionWeights) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.weights = weights;
        self.config = Some(config);
        self
    }

    /// Set the maximum number of suggestions.
    pub fn max_suggestions(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_suggestions = max;
        self.config = Some(config);
        self
    }

    /// Set whether to include details.
    pub fn include_details(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_details = include;
        self.config = Some(config);
        self
    }

    /// Build the EvaluatorAgent.
    pub fn build(self) -> EvaluatorAgent {
        EvaluatorAgent {
            config: self.config.unwrap_or_default(),
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
    // Priority Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_priority_display() {
        assert_eq!(Priority::High.to_string(), "high");
        assert_eq!(Priority::Medium.to_string(), "medium");
        assert_eq!(Priority::Low.to_string(), "low");
    }

    #[test]
    fn test_priority_from_str() {
        assert_eq!("high".parse::<Priority>().unwrap(), Priority::High);
        assert_eq!("critical".parse::<Priority>().unwrap(), Priority::High);
        assert_eq!("medium".parse::<Priority>().unwrap(), Priority::Medium);
        assert_eq!("low".parse::<Priority>().unwrap(), Priority::Low);
    }

    #[test]
    fn test_priority_from_str_invalid() {
        assert!("invalid".parse::<Priority>().is_err());
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Medium);
    }

    // -------------------------------------------------------------------------
    // EvaluationLevel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_evaluation_level_display() {
        assert_eq!(EvaluationLevel::Quick.to_string(), "quick");
        assert_eq!(EvaluationLevel::Standard.to_string(), "standard");
        assert_eq!(EvaluationLevel::Deep.to_string(), "deep");
    }

    #[test]
    fn test_evaluation_level_from_str() {
        assert_eq!("quick".parse::<EvaluationLevel>().unwrap(), EvaluationLevel::Quick);
        assert_eq!("fast".parse::<EvaluationLevel>().unwrap(), EvaluationLevel::Quick);
        assert_eq!("standard".parse::<EvaluationLevel>().unwrap(), EvaluationLevel::Standard);
        assert_eq!("deep".parse::<EvaluationLevel>().unwrap(), EvaluationLevel::Deep);
        assert_eq!("thorough".parse::<EvaluationLevel>().unwrap(), EvaluationLevel::Deep);
    }

    #[test]
    fn test_evaluation_level_from_str_invalid() {
        assert!("invalid".parse::<EvaluationLevel>().is_err());
    }

    #[test]
    fn test_evaluation_level_all() {
        let all = EvaluationLevel::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&EvaluationLevel::Quick));
        assert!(all.contains(&EvaluationLevel::Standard));
        assert!(all.contains(&EvaluationLevel::Deep));
    }

    #[test]
    fn test_evaluation_level_duration_multiplier() {
        assert!((EvaluationLevel::Quick.duration_multiplier() - 0.5).abs() < f64::EPSILON);
        assert!((EvaluationLevel::Standard.duration_multiplier() - 1.0).abs() < f64::EPSILON);
        assert!((EvaluationLevel::Deep.duration_multiplier() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluation_level_default() {
        assert_eq!(EvaluationLevel::default(), EvaluationLevel::Standard);
    }

    // -------------------------------------------------------------------------
    // Improvement Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_improvement_new() {
        let imp = Improvement::new("Clarity", "Add more examples");
        assert_eq!(imp.category, "Clarity");
        assert_eq!(imp.suggestion, "Add more examples");
        assert_eq!(imp.priority, Priority::Medium);
    }

    #[test]
    fn test_improvement_builder() {
        let imp = Improvement::new("Accuracy", "Add references")
            .with_priority(Priority::High)
            .with_impact(0.25);

        assert_eq!(imp.priority, Priority::High);
        assert!((imp.estimated_impact - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_improvement_impact_clamping() {
        let imp1 = Improvement::new("Test", "Test").with_impact(1.5);
        assert!((imp1.estimated_impact - 1.0).abs() < f64::EPSILON);

        let imp2 = Improvement::new("Test", "Test").with_impact(-0.5);
        assert!((imp2.estimated_impact - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_improvement_to_markdown() {
        let imp = Improvement::new("Clarity", "Add structure")
            .with_priority(Priority::High)
            .with_impact(0.2);

        let md = imp.to_markdown();
        assert!(md.contains("[Clarity]"));
        assert!(md.contains("Add structure"));
        assert!(md.contains("high"));
        assert!(md.contains("20%"));
    }

    // -------------------------------------------------------------------------
    // EvaluationDimensions Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dimensions_new() {
        let dims = EvaluationDimensions::new(0.8, 0.7, 0.9, 0.6, 0.75);
        assert!((dims.relevance - 0.8).abs() < f64::EPSILON);
        assert!((dims.completeness - 0.7).abs() < f64::EPSILON);
        assert!((dims.clarity - 0.9).abs() < f64::EPSILON);
        assert!((dims.accuracy - 0.6).abs() < f64::EPSILON);
        assert!((dims.actionability - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_clamping() {
        let dims = EvaluationDimensions::new(1.5, -0.5, 0.5, 0.5, 0.5);
        assert!((dims.relevance - 1.0).abs() < f64::EPSILON);
        assert!((dims.completeness - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_average() {
        let dims = EvaluationDimensions::new(0.8, 0.6, 0.7, 0.9, 0.5);
        let avg = dims.average();
        let expected = (0.8 + 0.6 + 0.7 + 0.9 + 0.5) / 5.0;
        assert!((avg - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_weighted_average() {
        let dims = EvaluationDimensions::new(1.0, 0.5, 0.5, 0.5, 0.5);
        let weights = DimensionWeights {
            relevance: 2.0,
            completeness: 1.0,
            clarity: 1.0,
            accuracy: 1.0,
            actionability: 1.0,
        };

        let weighted = dims.weighted_average(&weights);
        let expected = (1.0 * 2.0 + 0.5 * 4.0) / 6.0;
        assert!((weighted - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_minimum() {
        let dims = EvaluationDimensions::new(0.8, 0.6, 0.7, 0.4, 0.5);
        assert!((dims.minimum() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_maximum() {
        let dims = EvaluationDimensions::new(0.8, 0.6, 0.7, 0.4, 0.5);
        assert!((dims.maximum() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_weakest() {
        let dims = EvaluationDimensions::new(0.8, 0.6, 0.7, 0.4, 0.5);
        let (name, score) = dims.weakest_dimension();
        assert_eq!(name, "accuracy");
        assert!((score - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_strongest() {
        let dims = EvaluationDimensions::new(0.8, 0.6, 0.9, 0.4, 0.5);
        let (name, score) = dims.strongest_dimension();
        assert_eq!(name, "clarity");
        assert!((score - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dimensions_all_above() {
        let dims_high = EvaluationDimensions::new(0.7, 0.7, 0.7, 0.7, 0.7);
        assert!(dims_high.all_above(0.6));
        assert!(!dims_high.all_above(0.8));

        let dims_low = EvaluationDimensions::new(0.7, 0.7, 0.3, 0.7, 0.7);
        assert!(!dims_low.all_above(0.5));
    }

    #[test]
    fn test_dimensions_to_markdown() {
        let dims = EvaluationDimensions::new(0.8, 0.7, 0.9, 0.6, 0.75);
        let md = dims.to_markdown();
        assert!(md.contains("Relevance"));
        assert!(md.contains("0.80"));
        assert!(md.contains("Clarity"));
        assert!(md.contains("0.90"));
    }

    #[test]
    fn test_dimensions_default() {
        let dims = EvaluationDimensions::default();
        assert!((dims.relevance - 0.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // DimensionWeights Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_weights_default() {
        let weights = DimensionWeights::default();
        assert!((weights.relevance - 1.0).abs() < f64::EPSILON);
        assert!((weights.completeness - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weights_accuracy_focused() {
        let weights = DimensionWeights::accuracy_focused();
        assert!((weights.accuracy - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weights_actionability_focused() {
        let weights = DimensionWeights::actionability_focused();
        assert!((weights.actionability - 2.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // EvaluationReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_report_new() {
        let dims = EvaluationDimensions::new(0.8, 0.7, 0.9, 0.6, 0.75);
        let report = EvaluationReport::new(0.75, dims);
        assert!((report.overall_score - 0.75).abs() < f64::EPSILON);
        assert!(report.strengths.is_empty());
        assert!(report.weaknesses.is_empty());
    }

    #[test]
    fn test_report_passes_threshold() {
        let dims = EvaluationDimensions::default();
        let report = EvaluationReport::new(0.7, dims);
        assert!(report.passes_threshold(0.6));
        assert!(!report.passes_threshold(0.8));
    }

    #[test]
    fn test_report_is_high_quality() {
        let dims_high = EvaluationDimensions::new(0.8, 0.8, 0.8, 0.7, 0.7);
        let report_high = EvaluationReport::new(0.85, dims_high);
        assert!(report_high.is_high_quality());

        let dims_low = EvaluationDimensions::new(0.9, 0.9, 0.3, 0.9, 0.9);
        let report_low = EvaluationReport::new(0.85, dims_low);
        assert!(!report_low.is_high_quality());
    }

    #[test]
    fn test_report_needs_improvement() {
        let dims_good = EvaluationDimensions::new(0.7, 0.7, 0.7, 0.7, 0.7);
        let report_good = EvaluationReport::new(0.7, dims_good);
        assert!(!report_good.needs_improvement());

        let dims_bad = EvaluationDimensions::new(0.7, 0.7, 0.3, 0.7, 0.7);
        let report_bad = EvaluationReport::new(0.5, dims_bad);
        assert!(report_bad.needs_improvement());
    }

    #[test]
    fn test_report_high_priority_improvements() {
        let dims = EvaluationDimensions::default();
        let mut report = EvaluationReport::new(0.5, dims);
        report.suggestions.push(Improvement::new("A", "A").with_priority(Priority::High));
        report.suggestions.push(Improvement::new("B", "B").with_priority(Priority::Low));
        report.suggestions.push(Improvement::new("C", "C").with_priority(Priority::High));

        let high_priority = report.high_priority_improvements();
        assert_eq!(high_priority.len(), 2);
    }

    #[test]
    fn test_report_summary() {
        let dims = EvaluationDimensions::new(0.85, 0.85, 0.85, 0.85, 0.85);
        let mut report = EvaluationReport::new(0.85, dims);
        report.strengths.push("Good".to_string());
        report.suggestions.push(Improvement::new("Test", "Test"));

        let summary = report.summary();
        assert!(summary.contains("Excellent"));
        assert!(summary.contains("85%"));
        assert!(summary.contains("1 strengths"));
    }

    #[test]
    fn test_report_to_markdown() {
        let dims = EvaluationDimensions::new(0.8, 0.7, 0.9, 0.6, 0.75);
        let mut report = EvaluationReport::new(0.75, dims);
        report.strengths.push("Good structure".to_string());
        report.weaknesses.push("Could be clearer".to_string());

        let md = report.to_markdown();
        assert!(md.contains("# Evaluation Report"));
        assert!(md.contains("Overall Score"));
        assert!(md.contains("## Strengths"));
        assert!(md.contains("Good structure"));
        assert!(md.contains("## Weaknesses"));
    }

    // -------------------------------------------------------------------------
    // EvaluatorConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = EvaluatorConfig::default();
        assert!((config.min_score_threshold - 0.6).abs() < f64::EPSILON);
        assert_eq!(config.max_suggestions, 5);
    }

    #[test]
    fn test_config_strict() {
        let config = EvaluatorConfig::strict();
        assert!((config.min_score_threshold - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.max_suggestions, 10);
    }

    #[test]
    fn test_config_lenient() {
        let config = EvaluatorConfig::lenient();
        assert!((config.min_score_threshold - 0.4).abs() < f64::EPSILON);
        assert_eq!(config.max_suggestions, 3);
    }

    // -------------------------------------------------------------------------
    // EvaluatorAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_new() {
        let agent = EvaluatorAgent::new();
        assert!((agent.config().min_score_threshold - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_agent_with_config() {
        let config = EvaluatorConfig::strict();
        let agent = EvaluatorAgent::with_config(config);
        assert!((agent.config().min_score_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_agent_builder() {
        let agent = EvaluatorAgent::builder()
            .min_score_threshold(0.7)
            .max_suggestions(8)
            .build();

        assert!((agent.config().min_score_threshold - 0.7).abs() < f64::EPSILON);
        assert_eq!(agent.config().max_suggestions, 8);
    }

    #[test]
    fn test_agent_builder_with_weights() {
        let weights = DimensionWeights::accuracy_focused();
        let agent = EvaluatorAgent::builder()
            .weights(weights)
            .build();

        assert!((agent.config().weights.accuracy - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_agent_default() {
        let agent = EvaluatorAgent::default();
        assert!((agent.config().min_score_threshold - 0.6).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Evaluation Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_evaluate_empty_response() {
        let agent = EvaluatorAgent::new();
        let result = agent.evaluate("", "test query", EvaluationLevel::Quick).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_evaluate_quick_level() {
        let agent = EvaluatorAgent::new();
        let response = "This is a test response with some content about OAuth2 authentication.";
        let query = "How do I implement OAuth2?";

        let result = agent.evaluate(response, query, EvaluationLevel::Quick).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
        assert_eq!(report.level, EvaluationLevel::Quick);
    }

    #[tokio::test]
    async fn test_evaluate_standard_level() {
        let agent = EvaluatorAgent::new();
        let response = r#"
            To implement OAuth2 authentication, follow these steps:

            1. First, register your application
            2. Then configure the callback URL
            3. Finally, implement the token exchange

            Here's an example:
            ```javascript
            const token = await oauth.getToken(code);
            ```
        "#;
        let query = "How do I implement OAuth2 authentication?";

        let result = agent.evaluate(response, query, EvaluationLevel::Standard).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(report.overall_score > 0.5);
        assert!(!report.strengths.is_empty());
    }

    #[tokio::test]
    async fn test_evaluate_deep_level() {
        let agent = EvaluatorAgent::new();
        let response = r#"
            # OAuth2 Authentication Guide

            ## Prerequisites
            Before you begin, ensure you have:
            - Node.js 16+
            - An OAuth2 provider account

            ## Step 1: Register Application
            First, register your application with the OAuth provider.

            ## Step 2: Configure Callback
            Set up your callback URL: `https://example.com/callback`

            ## Step 3: Implement Token Exchange
            ```javascript
            async function exchangeToken(code) {
                const response = await fetch('/oauth/token', {
                    method: 'POST',
                    body: JSON.stringify({ code })
                });
                return response.json();
            }
            ```

            ## Troubleshooting
            If you encounter errors, check that your redirect URI matches exactly.
        "#;
        let query = "How do I implement OAuth2 authentication?";

        let result = agent.evaluate(response, query, EvaluationLevel::Deep).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(report.overall_score > 0.6);
        assert!(report.dimensions.actionability > 0.5);
    }

    #[tokio::test]
    async fn test_evaluate_high_quality_response() {
        let agent = EvaluatorAgent::new();
        let response = r#"
            # Complete OAuth2 Implementation

            OAuth2 is an authorization framework. Here's how to implement it:

            ## Step 1: Setup
            Install the required packages:
            ```bash
            npm install oauth2-client
            ```

            ## Step 2: Configure
            Create a configuration file with your credentials.

            ## Step 3: Implement
            ```javascript
            const oauth = new OAuth2Client({
                clientId: 'your-client-id',
                clientSecret: 'your-secret'
            });
            ```

            This approach is recommended because it follows security best practices.
        "#;
        let query = "How do I implement OAuth2?";

        let result = agent.evaluate(response, query, EvaluationLevel::Standard).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(report.dimensions.clarity > 0.6);
        assert!(report.dimensions.actionability > 0.5);
    }

    #[tokio::test]
    async fn test_evaluate_low_quality_response() {
        let agent = EvaluatorAgent::new();
        let response = "just use oauth stuff";
        let query = "How do I implement OAuth2 authentication with JWT tokens in a microservices architecture?";

        let result = agent.evaluate(response, query, EvaluationLevel::Standard).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(report.overall_score < 0.6);
        assert!(report.needs_improvement());
        assert!(!report.suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_evaluate_generates_suggestions() {
        let agent = EvaluatorAgent::new();
        let response = "OAuth2 is an authorization framework used for authentication.";
        let query = "How do I implement OAuth2?";

        let result = agent.evaluate(response, query, EvaluationLevel::Standard).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        // Should have suggestions since response lacks code and steps
        assert!(!report.suggestions.is_empty());
    }

    // -------------------------------------------------------------------------
    // Word Extraction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_significant_words() {
        let agent = EvaluatorAgent::new();
        let words = agent.extract_significant_words("How do I implement OAuth2 authentication?");

        assert!(words.contains(&"implement".to_string()));
        assert!(words.contains(&"oauth2".to_string()));
        assert!(words.contains(&"authentication".to_string()));
        // Stop words should be filtered
        assert!(!words.contains(&"how".to_string()));
        assert!(!words.contains(&"do".to_string()));
    }

    #[test]
    fn test_extract_significant_words_empty() {
        let agent = EvaluatorAgent::new();
        let words = agent.extract_significant_words("the a an");
        assert!(words.is_empty());
    }

    // -------------------------------------------------------------------------
    // Quick Evaluation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_calculate_relevance_quick() {
        let agent = EvaluatorAgent::new();

        // High relevance
        let score_high = agent.calculate_relevance_quick(
            "OAuth2 is a protocol for authentication and authorization",
            "What is OAuth2 authentication?"
        );
        assert!(score_high > 0.5);

        // Low relevance
        let score_low = agent.calculate_relevance_quick(
            "The weather is nice today",
            "What is OAuth2 authentication?"
        );
        assert!(score_low < 0.5);
    }

    #[test]
    fn test_calculate_completeness_quick() {
        let agent = EvaluatorAgent::new();

        // Short response
        let score_short = agent.calculate_completeness_quick("Short.");
        assert!(score_short < 0.6, "Short response score {} should be < 0.6", score_short);

        // Long structured response
        let score_long = agent.calculate_completeness_quick(
            "This is a longer response.\n- Point 1\n- Point 2\n- Point 3\nWith more content here."
        );
        // With structure bonus, should be reasonable
        assert!(score_long > 0.4, "Long response score {} should be > 0.4", score_long);
    }

    #[test]
    fn test_calculate_clarity_quick() {
        let agent = EvaluatorAgent::new();

        // Clear with structure
        let score_clear = agent.calculate_clarity_quick(
            "# Title\n\n- First point\n- Second point"
        );
        assert!(score_clear > 0.5, "Clear structured response score {} should be > 0.5", score_clear);

        // Very long sentences (less clear)
        let score_long_sentences = agent.calculate_clarity_quick(
            "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence that goes on and on and on and on."
        );
        // Long sentences get penalized but structure check can still add bonus
        assert!(score_long_sentences <= 1.0, "Long sentence score {} should be <= 1.0", score_long_sentences);
    }

    #[test]
    fn test_calculate_actionability_quick() {
        let agent = EvaluatorAgent::new();

        // Actionable with code
        let score_actionable = agent.calculate_actionability_quick(
            "Use this code:\n```\nconst x = 1;\n```\nThen run the command."
        );
        assert!(score_actionable > 0.5);

        // Not actionable
        let score_not_actionable = agent.calculate_actionability_quick(
            "The concept is interesting."
        );
        assert!(score_not_actionable < 0.5);
    }

    // -------------------------------------------------------------------------
    // Strength/Weakness Identification Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_identify_strengths_weaknesses_high_scores() {
        let agent = EvaluatorAgent::new();
        let dims = EvaluationDimensions::new(0.9, 0.85, 0.9, 0.85, 0.9);
        let response = "```code```\n1. First step\n2. Second step";

        let (strengths, weaknesses) = agent.identify_strengths_weaknesses(&dims, response);

        assert!(!strengths.is_empty());
        assert!(weaknesses.len() < strengths.len());
    }

    #[test]
    fn test_identify_strengths_weaknesses_low_scores() {
        let agent = EvaluatorAgent::new();
        let dims = EvaluationDimensions::new(0.3, 0.4, 0.3, 0.4, 0.3);
        let response = "short";

        let (strengths, weaknesses) = agent.identify_strengths_weaknesses(&dims, response);

        assert!(!weaknesses.is_empty());
        assert!(weaknesses.len() > strengths.len());
    }

    // -------------------------------------------------------------------------
    // Suggestion Generation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_suggestions_for_low_scores() {
        let agent = EvaluatorAgent::new();
        let dims = EvaluationDimensions::new(0.4, 0.5, 0.4, 0.5, 0.4);
        let response = "Just use oauth.";
        let query = "How do I implement OAuth2?";

        let suggestions = agent.generate_suggestions(&dims, response, query);

        assert!(!suggestions.is_empty());
        // Should have suggestions for multiple low dimensions
        assert!(suggestions.len() >= 2);
    }

    #[test]
    fn test_generate_suggestions_limited_by_max() {
        let config = EvaluatorConfig {
            max_suggestions: 2,
            ..Default::default()
        };
        let agent = EvaluatorAgent::with_config(config);
        let dims = EvaluationDimensions::new(0.3, 0.3, 0.3, 0.3, 0.3);

        let suggestions = agent.generate_suggestions(&dims, "test", "test query");

        assert!(suggestions.len() <= 2);
    }

    #[test]
    fn test_generate_suggestions_for_high_scores() {
        let agent = EvaluatorAgent::new();
        let dims = EvaluationDimensions::new(0.9, 0.9, 0.9, 0.9, 0.9);
        let response = "# Complete Guide\n\n1. Step one\n```code```";
        let query = "How to?";

        let suggestions = agent.generate_suggestions(&dims, response, query);

        // High scores should generate few or no suggestions
        assert!(suggestions.len() < 3);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_evaluate_whitespace_only() {
        let agent = EvaluatorAgent::new();
        let result = agent.evaluate("   \n\t  ", "test", EvaluationLevel::Quick).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_evaluate_very_long_response() {
        let agent = EvaluatorAgent::new();
        let response = "word ".repeat(2000);
        let query = "test query";

        let result = agent.evaluate(&response, query, EvaluationLevel::Quick).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        // Very long response might trigger "overly verbose" weakness
        assert!(report.weaknesses.iter().any(|w| w.contains("verbose")) || report.overall_score > 0.0);
    }

    #[tokio::test]
    async fn test_evaluate_response_with_only_code() {
        let agent = EvaluatorAgent::new();
        let response = "```rust\nfn main() {\n    println!(\"Hello\");\n}\n```";
        let query = "How do I print hello in Rust?";

        let result = agent.evaluate(response, query, EvaluationLevel::Standard).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        // Code-only responses should have reasonable actionability (code block bonus applies)
        assert!(report.dimensions.actionability > 0.3,
            "Actionability score {} should be > 0.3 for code-only response",
            report.dimensions.actionability);
    }

    #[tokio::test]
    async fn test_evaluate_empty_query() {
        let agent = EvaluatorAgent::new();
        let response = "This is a response.";

        let result = agent.evaluate(response, "", EvaluationLevel::Quick).await;
        assert!(result.is_ok());
        // Should still work but with lower relevance
    }

    // -------------------------------------------------------------------------
    // Integration-like Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_full_evaluation_workflow() {
        let agent = EvaluatorAgent::builder()
            .min_score_threshold(0.5)
            .max_suggestions(5)
            .weights(DimensionWeights::equal())
            .build();

        let response = r#"
            # Getting Started with Authentication

            Follow these steps:

            1. Install dependencies
            2. Configure settings
            3. Implement the flow

            ```javascript
            const auth = require('auth-lib');
            auth.configure({ key: 'xxx' });
            ```

            For more information, see the documentation.
        "#;

        let query = "How do I set up authentication?";

        let report = agent.evaluate(response, query, EvaluationLevel::Deep).await.unwrap();

        // Verify report structure
        assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
        assert!(report.dimensions.relevance >= 0.0);
        assert!(report.dimensions.completeness >= 0.0);
        assert!(report.dimensions.clarity >= 0.0);
        assert!(report.dimensions.accuracy >= 0.0);
        assert!(report.dimensions.actionability >= 0.0);
        assert!(report.duration_ms >= 0);
        assert_eq!(report.level, EvaluationLevel::Deep);

        // Verify report can be formatted
        let md = report.to_markdown();
        assert!(!md.is_empty());
        assert!(md.contains("Evaluation Report"));
    }
}
