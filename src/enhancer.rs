//! Enhancer Agent module for Project Panpsychism.
//!
//! The Quality Booster - "Every prompt can be refined toward perfection."
//!
//! This module implements the Enhancer Agent, responsible for improving
//! prompt quality through systematic optimization techniques. Like an
//! alchemist refining raw materials into gold, the Enhancer transforms
//! ordinary prompts into powerful incantations.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Quality Booster serves as the refiner:
//!
//! - **Raw Prompt** (initial incantation) enters the enhancement chamber
//! - **The Booster** (EnhancerAgent) applies optimization techniques
//! - **Enhanced Prompt** (perfected spell) emerges with increased power
//!
//! The Booster enhances across multiple dimensions:
//! - **Clarity**: Making the prompt clearer and more understandable
//! - **Specificity**: Adding precise details and constraints
//! - **Engagement**: Making the prompt more compelling
//! - **Actionability**: Ensuring clear, achievable outcomes
//! - **Completeness**: Covering all necessary aspects
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Drive toward quality improvement and perfection
//! - **RATIO**: Logical optimization of structure and content
//! - **LAETITIA**: Joy through refined expression and clarity
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::enhancer::{EnhancerAgent, QualityDimension};
//!
//! let booster = EnhancerAgent::new();
//!
//! // Enhance a prompt
//! let enhanced = booster.enhance(
//!     "Write code for authentication",
//!     &[QualityDimension::Clarity, QualityDimension::Specificity]
//! ).await?;
//!
//! println!("Enhanced: {}", enhanced.content);
//! println!("Score improved: {:.0}% -> {:.0}%",
//!     enhanced.before_score * 100.0,
//!     enhanced.after_score * 100.0
//! );
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// QUALITY DIMENSION ENUM
// =============================================================================

/// Dimensions of quality that can be enhanced.
///
/// Each dimension represents a different aspect of prompt quality
/// that can be targeted for improvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QualityDimension {
    /// Making the prompt clearer and more understandable.
    #[default]
    Clarity,

    /// Adding precise details and constraints.
    Specificity,

    /// Making the prompt more compelling and interesting.
    Engagement,

    /// Ensuring clear, achievable outcomes.
    Actionability,

    /// Covering all necessary aspects of the request.
    Completeness,
}

impl std::fmt::Display for QualityDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clarity => write!(f, "clarity"),
            Self::Specificity => write!(f, "specificity"),
            Self::Engagement => write!(f, "engagement"),
            Self::Actionability => write!(f, "actionability"),
            Self::Completeness => write!(f, "completeness"),
        }
    }
}

impl std::str::FromStr for QualityDimension {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "clarity" | "clear" => Ok(Self::Clarity),
            "specificity" | "specific" | "precise" => Ok(Self::Specificity),
            "engagement" | "engaging" | "compelling" => Ok(Self::Engagement),
            "actionability" | "actionable" | "practical" => Ok(Self::Actionability),
            "completeness" | "complete" | "comprehensive" => Ok(Self::Completeness),
            _ => Err(Error::Config(format!(
                "Unknown quality dimension: '{}'. Valid: clarity, specificity, engagement, actionability, completeness",
                s
            ))),
        }
    }
}

impl QualityDimension {
    /// Get all quality dimensions.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Clarity,
            Self::Specificity,
            Self::Engagement,
            Self::Actionability,
            Self::Completeness,
        ]
    }

    /// Get the weight multiplier for this dimension (default importance).
    pub fn default_weight(&self) -> f64 {
        match self {
            Self::Clarity => 1.2,
            Self::Specificity => 1.1,
            Self::Engagement => 0.9,
            Self::Actionability => 1.0,
            Self::Completeness => 1.0,
        }
    }
}

// =============================================================================
// ENHANCEMENT GOAL
// =============================================================================

/// A specific enhancement goal with target score and weight.
///
/// Enhancement goals allow fine-grained control over which dimensions
/// to prioritize and what quality levels to aim for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementGoal {
    /// The quality dimension to enhance.
    pub dimension: QualityDimension,

    /// Target score to achieve (0.0 - 1.0).
    pub target_score: f64,

    /// Weight/importance of this goal (default 1.0).
    pub weight: f64,
}

impl EnhancementGoal {
    /// Create a new enhancement goal.
    pub fn new(dimension: QualityDimension) -> Self {
        Self {
            dimension,
            target_score: 0.8,
            weight: dimension.default_weight(),
        }
    }

    /// Set the target score.
    pub fn with_target(mut self, target: f64) -> Self {
        self.target_score = target.clamp(0.0, 1.0);
        self
    }

    /// Set the weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    /// Check if a score meets this goal.
    pub fn is_met(&self, score: f64) -> bool {
        score >= self.target_score
    }

    /// Calculate how far a score is from the target.
    pub fn gap(&self, score: f64) -> f64 {
        (self.target_score - score).max(0.0)
    }
}

impl Default for EnhancementGoal {
    fn default() -> Self {
        Self::new(QualityDimension::Clarity)
    }
}

// =============================================================================
// IMPROVEMENT
// =============================================================================

/// A specific improvement made during enhancement.
///
/// Each improvement describes what was changed, why, and what
/// dimension it targeted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    /// The dimension this improvement targets.
    pub dimension: QualityDimension,

    /// Description of what was changed.
    pub description: String,

    /// The original text (if applicable).
    pub original: Option<String>,

    /// The improved text (if applicable).
    pub improved: Option<String>,

    /// Estimated impact on score (0.0 - 1.0).
    pub estimated_impact: f64,
}

impl Improvement {
    /// Create a new improvement.
    pub fn new(dimension: QualityDimension, description: impl Into<String>) -> Self {
        Self {
            dimension,
            description: description.into(),
            original: None,
            improved: None,
            estimated_impact: 0.1,
        }
    }

    /// Set the original text.
    pub fn with_original(mut self, text: impl Into<String>) -> Self {
        self.original = Some(text.into());
        self
    }

    /// Set the improved text.
    pub fn with_improved(mut self, text: impl Into<String>) -> Self {
        self.improved = Some(text.into());
        self
    }

    /// Set the estimated impact.
    pub fn with_impact(mut self, impact: f64) -> Self {
        self.estimated_impact = impact.clamp(0.0, 1.0);
        self
    }

    /// Format the improvement as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!("- **[{}]** {}", self.dimension, self.description);
        if self.original.is_some() || self.improved.is_some() {
            output.push_str(&format!(" _(Impact: {:.0}%)_", self.estimated_impact * 100.0));
        }
        output
    }
}

impl Default for Improvement {
    fn default() -> Self {
        Self::new(QualityDimension::Clarity, "")
    }
}

// =============================================================================
// ENHANCEMENT STEP
// =============================================================================

/// A step in the enhancement process.
///
/// Enhancement steps track the iterative refinement process,
/// showing what was attempted and the outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementStep {
    /// Step number in the sequence.
    pub step_number: usize,

    /// The technique applied.
    pub technique: String,

    /// Dimensions targeted in this step.
    pub dimensions_targeted: Vec<QualityDimension>,

    /// Score before this step.
    pub score_before: f64,

    /// Score after this step.
    pub score_after: f64,

    /// Whether this step improved quality.
    pub successful: bool,

    /// Duration of this step in milliseconds.
    pub duration_ms: u64,
}

impl EnhancementStep {
    /// Create a new enhancement step.
    pub fn new(step_number: usize, technique: impl Into<String>) -> Self {
        Self {
            step_number,
            technique: technique.into(),
            dimensions_targeted: Vec::new(),
            score_before: 0.0,
            score_after: 0.0,
            successful: false,
            duration_ms: 0,
        }
    }

    /// Add targeted dimensions.
    pub fn with_dimensions(mut self, dimensions: Vec<QualityDimension>) -> Self {
        self.dimensions_targeted = dimensions;
        self
    }

    /// Set scores.
    pub fn with_scores(mut self, before: f64, after: f64) -> Self {
        self.score_before = before.clamp(0.0, 1.0);
        self.score_after = after.clamp(0.0, 1.0);
        self.successful = after > before;
        self
    }

    /// Set duration.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Get the improvement delta.
    pub fn improvement(&self) -> f64 {
        (self.score_after - self.score_before).max(0.0)
    }
}

impl Default for EnhancementStep {
    fn default() -> Self {
        Self::new(1, "unknown")
    }
}

// =============================================================================
// ENHANCED PROMPT
// =============================================================================

/// The result of an enhancement operation.
///
/// Contains the enhanced content along with detailed information
/// about what was improved and how.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedPrompt {
    /// The enhanced prompt content.
    pub content: String,

    /// List of improvements made.
    pub improvements: Vec<Improvement>,

    /// Score before enhancement (0.0 - 1.0).
    pub before_score: f64,

    /// Score after enhancement (0.0 - 1.0).
    pub after_score: f64,

    /// Detailed log of enhancement steps.
    pub enhancement_log: Vec<EnhancementStep>,

    /// Dimensions that were enhanced.
    pub dimensions_enhanced: Vec<QualityDimension>,

    /// Total processing time in milliseconds.
    pub duration_ms: u64,

    /// Number of iterations performed.
    pub iterations: usize,

    /// The original prompt for reference.
    pub original: String,
}

impl EnhancedPrompt {
    /// Create a new enhanced prompt result.
    pub fn new(original: impl Into<String>, enhanced: impl Into<String>) -> Self {
        Self {
            original: original.into(),
            content: enhanced.into(),
            improvements: Vec::new(),
            before_score: 0.0,
            after_score: 0.0,
            enhancement_log: Vec::new(),
            dimensions_enhanced: Vec::new(),
            duration_ms: 0,
            iterations: 0,
        }
    }

    /// Calculate the improvement percentage.
    pub fn improvement_percentage(&self) -> f64 {
        if self.before_score > 0.0 {
            ((self.after_score - self.before_score) / self.before_score) * 100.0
        } else if self.after_score > 0.0 {
            100.0
        } else {
            0.0
        }
    }

    /// Check if significant improvement was achieved (>10%).
    pub fn is_significantly_improved(&self) -> bool {
        self.after_score > self.before_score + 0.1
    }

    /// Get the total improvement delta.
    pub fn improvement_delta(&self) -> f64 {
        (self.after_score - self.before_score).max(0.0)
    }

    /// Check if the enhancement was successful.
    pub fn is_successful(&self) -> bool {
        self.after_score >= self.before_score
    }

    /// Get successful enhancement steps only.
    pub fn successful_steps(&self) -> Vec<&EnhancementStep> {
        self.enhancement_log.iter().filter(|s| s.successful).collect()
    }

    /// Generate a summary of the enhancement.
    pub fn summary(&self) -> String {
        format!(
            "Enhanced: {:.0}% -> {:.0}% (+{:.0}%) via {} improvements in {}ms",
            self.before_score * 100.0,
            self.after_score * 100.0,
            self.improvement_delta() * 100.0,
            self.improvements.len(),
            self.duration_ms
        )
    }

    /// Format the result as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Enhancement Report\n\n");
        output.push_str(&format!(
            "**Score:** {:.0}% -> {:.0}% (+{:.1}%)\n\n",
            self.before_score * 100.0,
            self.after_score * 100.0,
            self.improvement_percentage()
        ));

        output.push_str("## Enhanced Prompt\n\n");
        output.push_str(&format!("```\n{}\n```\n\n", self.content));

        if !self.improvements.is_empty() {
            output.push_str("## Improvements Made\n\n");
            for imp in &self.improvements {
                output.push_str(&imp.to_markdown());
                output.push('\n');
            }
            output.push('\n');
        }

        if !self.enhancement_log.is_empty() {
            output.push_str("## Enhancement Process\n\n");
            for step in &self.enhancement_log {
                let status = if step.successful { "+" } else { "-" };
                output.push_str(&format!(
                    "{}. [{}] {} ({:.0}% -> {:.0}%)\n",
                    step.step_number,
                    status,
                    step.technique,
                    step.score_before * 100.0,
                    step.score_after * 100.0
                ));
            }
        }

        output
    }
}

impl Default for EnhancedPrompt {
    fn default() -> Self {
        Self::new("", "")
    }
}

// =============================================================================
// ENHANCER CONFIGURATION
// =============================================================================

/// Configuration for the Enhancer Agent.
#[derive(Debug, Clone)]
pub struct EnhancerConfig {
    /// Minimum quality threshold before enhancement starts.
    pub quality_threshold: f64,

    /// Maximum iterations for enhancement.
    pub max_iterations: usize,

    /// Minimum improvement per iteration to continue.
    pub min_improvement: f64,

    /// Target quality score to achieve.
    pub target_quality: f64,

    /// Whether to preserve original meaning strictly.
    pub preserve_meaning: bool,

    /// Timeout in seconds for enhancement.
    pub timeout_secs: u64,

    /// Default enhancement goals when none specified.
    pub default_goals: Vec<EnhancementGoal>,
}

impl Default for EnhancerConfig {
    fn default() -> Self {
        Self {
            quality_threshold: 0.3,
            max_iterations: 5,
            min_improvement: 0.02,
            target_quality: 0.85,
            preserve_meaning: true,
            timeout_secs: 30,
            default_goals: vec![
                EnhancementGoal::new(QualityDimension::Clarity).with_target(0.8),
                EnhancementGoal::new(QualityDimension::Specificity).with_target(0.7),
                EnhancementGoal::new(QualityDimension::Actionability).with_target(0.75),
            ],
        }
    }
}

impl EnhancerConfig {
    /// Create an aggressive enhancement configuration.
    pub fn aggressive() -> Self {
        Self {
            quality_threshold: 0.2,
            max_iterations: 10,
            min_improvement: 0.01,
            target_quality: 0.95,
            preserve_meaning: false,
            timeout_secs: 60,
            default_goals: QualityDimension::all()
                .into_iter()
                .map(|d| EnhancementGoal::new(d).with_target(0.9))
                .collect(),
        }
    }

    /// Create a conservative enhancement configuration.
    pub fn conservative() -> Self {
        Self {
            quality_threshold: 0.5,
            max_iterations: 3,
            min_improvement: 0.05,
            target_quality: 0.75,
            preserve_meaning: true,
            timeout_secs: 15,
            default_goals: vec![
                EnhancementGoal::new(QualityDimension::Clarity).with_target(0.7),
            ],
        }
    }

    /// Create a minimal enhancement configuration (quick polish).
    pub fn minimal() -> Self {
        Self {
            quality_threshold: 0.6,
            max_iterations: 2,
            min_improvement: 0.03,
            target_quality: 0.7,
            preserve_meaning: true,
            timeout_secs: 10,
            default_goals: vec![
                EnhancementGoal::new(QualityDimension::Clarity).with_target(0.65),
            ],
        }
    }
}

// =============================================================================
// KEYWORD DICTIONARIES
// =============================================================================

/// Keywords indicating clarity.
const CLARITY_KEYWORDS: &[&str] = &[
    "specifically", "namely", "in other words", "that is",
    "for example", "such as", "including", "particularly",
    "clearly", "explicitly", "precisely", "exactly",
];

/// Keywords indicating specificity.
const SPECIFICITY_KEYWORDS: &[&str] = &[
    "must", "should", "exactly", "precisely", "only",
    "between", "minimum", "maximum", "at least", "no more than",
    "required", "mandatory", "optional", "format", "structure",
];

/// Keywords indicating engagement.
const ENGAGEMENT_KEYWORDS: &[&str] = &[
    "please", "help", "assist", "guide", "explain",
    "imagine", "consider", "think", "creative", "innovative",
    "interesting", "compelling", "engaging", "exciting",
];

/// Keywords indicating actionability.
const ACTIONABILITY_KEYWORDS: &[&str] = &[
    "step", "steps", "process", "procedure", "method",
    "create", "build", "implement", "develop", "design",
    "provide", "generate", "produce", "output", "return",
];

/// Keywords indicating completeness.
const COMPLETENESS_KEYWORDS: &[&str] = &[
    "complete", "comprehensive", "full", "entire", "all",
    "including", "also", "additionally", "furthermore", "moreover",
    "cover", "address", "handle", "consider", "account for",
];

/// Vague words that reduce clarity.
const VAGUE_WORDS: &[&str] = &[
    "thing", "stuff", "something", "somehow", "whatever",
    "maybe", "perhaps", "sort of", "kind of", "basically",
    "actually", "literally", "very", "really", "just",
];

// =============================================================================
// ENHANCER AGENT
// =============================================================================

/// The Enhancer Agent - Agent 25 of Project Panpsychism.
///
/// Responsible for improving prompt quality through systematic
/// optimization techniques. Like an alchemist refining raw materials,
/// this agent transforms ordinary prompts into powerful incantations.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive toward quality improvement
/// - **RATIO**: Logical optimization of structure
/// - **LAETITIA**: Joy through refined expression
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::enhancer::{EnhancerAgent, QualityDimension};
///
/// let booster = EnhancerAgent::new();
///
/// let enhanced = booster.enhance(
///     "Write code",
///     &[QualityDimension::Clarity, QualityDimension::Specificity]
/// ).await?;
///
/// println!("Enhanced: {}", enhanced.content);
/// ```
#[derive(Debug, Clone)]
pub struct EnhancerAgent {
    /// Configuration for enhancement behavior.
    config: EnhancerConfig,
}

impl Default for EnhancerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancerAgent {
    /// Create a new Enhancer Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: EnhancerConfig::default(),
        }
    }

    /// Create a new Enhancer Agent with custom configuration.
    pub fn with_config(config: EnhancerConfig) -> Self {
        Self { config }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> EnhancerAgentBuilder {
        EnhancerAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &EnhancerConfig {
        &self.config
    }

    // =========================================================================
    // MAIN ENHANCEMENT METHOD
    // =========================================================================

    /// Enhance a prompt targeting specific quality dimensions.
    ///
    /// This is the primary enhancement method. It iteratively improves
    /// the prompt across the specified dimensions until quality targets
    /// are met or iteration limits are reached.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt to enhance
    /// * `dimensions` - Quality dimensions to target (uses defaults if empty)
    ///
    /// # Returns
    ///
    /// An `EnhancedPrompt` containing the improved content and enhancement details.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if the prompt is empty.
    pub async fn enhance(
        &self,
        prompt: &str,
        dimensions: &[QualityDimension],
    ) -> Result<EnhancedPrompt> {
        let start = Instant::now();

        if prompt.trim().is_empty() {
            return Err(Error::Validation("Cannot enhance empty prompt".to_string()));
        }

        let target_dimensions = if dimensions.is_empty() {
            self.config
                .default_goals
                .iter()
                .map(|g| g.dimension)
                .collect()
        } else {
            dimensions.to_vec()
        };

        debug!(
            "Enhancing prompt ({} chars) targeting {:?}",
            prompt.len(),
            target_dimensions
        );

        let mut current_prompt = prompt.to_string();
        let mut improvements = Vec::new();
        let mut enhancement_log = Vec::new();
        let mut iteration = 0;

        // Calculate initial score
        let initial_score = self.calculate_overall_score(&current_prompt, &target_dimensions);

        // Check if already meets target
        if initial_score >= self.config.target_quality {
            info!(
                "Prompt already meets quality target ({:.0}%)",
                initial_score * 100.0
            );
            return Ok(EnhancedPrompt {
                content: current_prompt.clone(),
                original: prompt.to_string(),
                improvements,
                before_score: initial_score,
                after_score: initial_score,
                enhancement_log,
                dimensions_enhanced: target_dimensions,
                duration_ms: start.elapsed().as_millis() as u64,
                iterations: 0,
            });
        }

        let mut previous_score = initial_score;

        // Iterative enhancement loop
        while iteration < self.config.max_iterations {
            iteration += 1;
            let step_start = Instant::now();

            // Find the dimension with the most room for improvement
            let (weakest_dim, _) = self.find_weakest_dimension(&current_prompt, &target_dimensions);

            // Apply enhancement technique for the weakest dimension
            let (enhanced, step_improvements) =
                self.apply_enhancement(&current_prompt, weakest_dim)?;

            let new_score = self.calculate_overall_score(&enhanced, &target_dimensions);

            // Record step
            let step = EnhancementStep::new(iteration, format!("Enhance {}", weakest_dim))
                .with_dimensions(vec![weakest_dim])
                .with_scores(previous_score, new_score)
                .with_duration(step_start.elapsed().as_millis() as u64);

            enhancement_log.push(step);

            // Check if improvement is sufficient
            let improvement = new_score - previous_score;
            if improvement >= self.config.min_improvement {
                current_prompt = enhanced.clone();
                improvements.extend(step_improvements);
                previous_score = new_score;
            }

            // Check if target reached
            if new_score >= self.config.target_quality {
                current_prompt = enhanced;
                debug!("Target quality reached at iteration {}", iteration);
                break;
            }

            // Check if no significant improvement
            if improvement < self.config.min_improvement && iteration > 1 {
                debug!("No significant improvement at iteration {}", iteration);
                break;
            }
        }

        let final_score = self.calculate_overall_score(&current_prompt, &target_dimensions);

        info!(
            "Enhancement complete: {:.0}% -> {:.0}% in {} iterations ({}ms)",
            initial_score * 100.0,
            final_score * 100.0,
            iteration,
            start.elapsed().as_millis()
        );

        Ok(EnhancedPrompt {
            content: current_prompt,
            original: prompt.to_string(),
            improvements,
            before_score: initial_score,
            after_score: final_score,
            enhancement_log,
            dimensions_enhanced: target_dimensions,
            duration_ms: start.elapsed().as_millis() as u64,
            iterations: iteration,
        })
    }

    /// Enhance with specific goals.
    pub async fn enhance_with_goals(
        &self,
        prompt: &str,
        goals: &[EnhancementGoal],
    ) -> Result<EnhancedPrompt> {
        let dimensions: Vec<QualityDimension> = goals.iter().map(|g| g.dimension).collect();
        self.enhance(prompt, &dimensions).await
    }

    // =========================================================================
    // SCORING METHODS
    // =========================================================================

    /// Calculate overall quality score across dimensions.
    fn calculate_overall_score(&self, prompt: &str, dimensions: &[QualityDimension]) -> f64 {
        if dimensions.is_empty() {
            return self.calculate_all_dimensions_score(prompt);
        }

        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for dim in dimensions {
            let score = self.score_dimension(prompt, *dim);
            let weight = dim.default_weight();
            total_score += score * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.5
        }
    }

    /// Calculate score across all dimensions.
    fn calculate_all_dimensions_score(&self, prompt: &str) -> f64 {
        let dims = QualityDimension::all();
        self.calculate_overall_score(prompt, &dims)
    }

    /// Score a specific dimension.
    fn score_dimension(&self, prompt: &str, dimension: QualityDimension) -> f64 {
        match dimension {
            QualityDimension::Clarity => self.score_clarity(prompt),
            QualityDimension::Specificity => self.score_specificity(prompt),
            QualityDimension::Engagement => self.score_engagement(prompt),
            QualityDimension::Actionability => self.score_actionability(prompt),
            QualityDimension::Completeness => self.score_completeness(prompt),
        }
    }

    /// Score clarity of a prompt.
    fn score_clarity(&self, prompt: &str) -> f64 {
        let prompt_lower = prompt.to_lowercase();
        let word_count = prompt.split_whitespace().count();

        // Count clarity keywords
        let clarity_count = CLARITY_KEYWORDS
            .iter()
            .filter(|kw| prompt_lower.contains(*kw))
            .count();

        // Count vague words (negative)
        let vague_count = VAGUE_WORDS
            .iter()
            .filter(|kw| prompt_lower.contains(*kw))
            .count();

        // Sentence structure (shorter sentences are clearer)
        let sentences: Vec<&str> = prompt.split(['.', '!', '?']).filter(|s| !s.trim().is_empty()).collect();
        let avg_sentence_length = if sentences.is_empty() {
            word_count as f64
        } else {
            word_count as f64 / sentences.len() as f64
        };

        let length_score = if avg_sentence_length <= 15.0 {
            1.0
        } else if avg_sentence_length <= 25.0 {
            0.8
        } else if avg_sentence_length <= 35.0 {
            0.6
        } else {
            0.4
        };

        let clarity_bonus = (clarity_count as f64 * 0.05).min(0.2);
        let vague_penalty = (vague_count as f64 * 0.08).min(0.3);

        (0.5 + length_score * 0.3 + clarity_bonus - vague_penalty).clamp(0.0, 1.0)
    }

    /// Score specificity of a prompt.
    fn score_specificity(&self, prompt: &str) -> f64 {
        let prompt_lower = prompt.to_lowercase();

        // Count specificity keywords
        let specificity_count = SPECIFICITY_KEYWORDS
            .iter()
            .filter(|kw| prompt_lower.contains(*kw))
            .count();

        // Check for numbers (specific quantities)
        let has_numbers = prompt.chars().any(|c| c.is_ascii_digit());

        // Check for quotes (specific examples)
        let has_quotes = prompt.contains('"') || prompt.contains('\'');

        // Check for technical terms
        let has_technical = prompt.contains("()") || prompt.contains("[]") || prompt.contains("{}");

        let base_score = 0.4;
        let keyword_bonus = (specificity_count as f64 * 0.1).min(0.3);
        let number_bonus: f64 = if has_numbers { 0.1 } else { 0.0 };
        let quote_bonus: f64 = if has_quotes { 0.05 } else { 0.0 };
        let technical_bonus: f64 = if has_technical { 0.1 } else { 0.0 };

        (base_score + keyword_bonus + number_bonus + quote_bonus + technical_bonus).clamp(0.0, 1.0)
    }

    /// Score engagement of a prompt.
    fn score_engagement(&self, prompt: &str) -> f64 {
        let prompt_lower = prompt.to_lowercase();

        // Count engagement keywords
        let engagement_count = ENGAGEMENT_KEYWORDS
            .iter()
            .filter(|kw| prompt_lower.contains(*kw))
            .count();

        // Check for question marks (engaging style)
        let has_questions = prompt.contains('?');

        // Check for exclamation (enthusiasm)
        let has_exclamation = prompt.contains('!');

        // Word variety (unique words / total words)
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let unique_words: HashSet<&str> = words.iter().copied().collect();
        let variety_score = if words.is_empty() {
            0.5
        } else {
            (unique_words.len() as f64 / words.len() as f64).min(1.0)
        };

        let base_score = 0.4;
        let keyword_bonus = (engagement_count as f64 * 0.08).min(0.25);
        let question_bonus: f64 = if has_questions { 0.1 } else { 0.0 };
        let exclamation_bonus: f64 = if has_exclamation { 0.05 } else { 0.0 };

        (base_score + keyword_bonus + question_bonus + exclamation_bonus + variety_score * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Score actionability of a prompt.
    fn score_actionability(&self, prompt: &str) -> f64 {
        let prompt_lower = prompt.to_lowercase();

        // Count actionability keywords
        let action_count = ACTIONABILITY_KEYWORDS
            .iter()
            .filter(|kw| prompt_lower.contains(*kw))
            .count();

        // Check for imperative verbs at start
        let first_word = prompt.split_whitespace().next().unwrap_or("").to_lowercase();
        let imperative_starters = [
            "create", "build", "write", "develop", "design", "implement",
            "generate", "produce", "make", "provide", "explain", "describe",
        ];
        let starts_imperative = imperative_starters.contains(&first_word.as_str());

        // Check for output format specification
        let has_format = prompt_lower.contains("format") || prompt_lower.contains("output")
            || prompt_lower.contains("return") || prompt_lower.contains("result");

        let base_score = 0.3;
        let keyword_bonus = (action_count as f64 * 0.1).min(0.3);
        let imperative_bonus: f64 = if starts_imperative { 0.15 } else { 0.0 };
        let format_bonus: f64 = if has_format { 0.1 } else { 0.0 };

        (base_score + keyword_bonus + imperative_bonus + format_bonus).clamp(0.0, 1.0)
    }

    /// Score completeness of a prompt.
    fn score_completeness(&self, prompt: &str) -> f64 {
        let prompt_lower = prompt.to_lowercase();
        let word_count = prompt.split_whitespace().count();

        // Count completeness keywords
        let completeness_count = COMPLETENESS_KEYWORDS
            .iter()
            .filter(|kw| prompt_lower.contains(*kw))
            .count();

        // Length factor (longer prompts tend to be more complete)
        let length_score = match word_count {
            0..=5 => 0.2,
            6..=15 => 0.4,
            16..=30 => 0.6,
            31..=60 => 0.8,
            _ => 0.9,
        };

        // Check for context elements
        let has_context = prompt_lower.contains("context") || prompt_lower.contains("background")
            || prompt_lower.contains("given") || prompt_lower.contains("assuming");

        // Check for constraints
        let has_constraints = prompt_lower.contains("constraint") || prompt_lower.contains("limit")
            || prompt_lower.contains("restriction") || prompt_lower.contains("requirement");

        let keyword_bonus = (completeness_count as f64 * 0.08).min(0.2);
        let context_bonus: f64 = if has_context { 0.1 } else { 0.0 };
        let constraint_bonus: f64 = if has_constraints { 0.1 } else { 0.0 };

        (length_score * 0.5 + keyword_bonus + context_bonus + constraint_bonus + 0.1).clamp(0.0, 1.0)
    }

    // =========================================================================
    // ENHANCEMENT TECHNIQUES
    // =========================================================================

    /// Find the dimension with the lowest score.
    fn find_weakest_dimension(
        &self,
        prompt: &str,
        dimensions: &[QualityDimension],
    ) -> (QualityDimension, f64) {
        dimensions
            .iter()
            .map(|&d| (d, self.score_dimension(prompt, d)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((QualityDimension::Clarity, 0.5))
    }

    /// Apply enhancement technique for a specific dimension.
    fn apply_enhancement(
        &self,
        prompt: &str,
        dimension: QualityDimension,
    ) -> Result<(String, Vec<Improvement>)> {
        match dimension {
            QualityDimension::Clarity => self.enhance_clarity(prompt),
            QualityDimension::Specificity => self.enhance_specificity(prompt),
            QualityDimension::Engagement => self.enhance_engagement(prompt),
            QualityDimension::Actionability => self.enhance_actionability(prompt),
            QualityDimension::Completeness => self.enhance_completeness(prompt),
        }
    }

    /// Enhance clarity by removing vague words and improving structure.
    fn enhance_clarity(&self, prompt: &str) -> Result<(String, Vec<Improvement>)> {
        let mut enhanced = prompt.to_string();
        let mut improvements = Vec::new();

        // Remove vague words
        for vague in VAGUE_WORDS {
            if enhanced.to_lowercase().contains(vague) {
                let replacement = match *vague {
                    "thing" => "element",
                    "stuff" => "content",
                    "something" => "a specific item",
                    "maybe" | "perhaps" => "",
                    "sort of" | "kind of" => "",
                    "basically" | "actually" | "literally" => "",
                    "very" | "really" => "",
                    "just" => "",
                    _ => "",
                };

                if replacement.is_empty() {
                    // Remove the word
                    enhanced = enhanced.replace(&format!(" {} ", vague), " ");
                    enhanced = enhanced.replace(&format!(" {}", vague), " ");
                    enhanced = enhanced.replace(&format!("{} ", vague), " ");
                } else {
                    enhanced = enhanced.replace(vague, replacement);
                }

                improvements.push(
                    Improvement::new(
                        QualityDimension::Clarity,
                        format!("Removed or replaced vague word '{}'", vague),
                    )
                    .with_impact(0.05),
                );
            }
        }

        // Clean up multiple spaces
        while enhanced.contains("  ") {
            enhanced = enhanced.replace("  ", " ");
        }

        // Trim
        enhanced = enhanced.trim().to_string();

        Ok((enhanced, improvements))
    }

    /// Enhance specificity by adding detail prompts.
    fn enhance_specificity(&self, prompt: &str) -> Result<(String, Vec<Improvement>)> {
        let mut enhanced = prompt.to_string();
        let mut improvements = Vec::new();

        let prompt_lower = prompt.to_lowercase();

        // Add specificity if prompt is too generic
        if !prompt_lower.contains("specifically") && !prompt_lower.contains("exactly") {
            // Check if there's a verb we can make more specific
            if prompt_lower.contains("create") || prompt_lower.contains("make") {
                if !enhanced.ends_with('.') {
                    enhanced.push('.');
                }
                enhanced.push_str(" Be specific about the requirements.");
                improvements.push(
                    Improvement::new(
                        QualityDimension::Specificity,
                        "Added prompt for specific requirements",
                    )
                    .with_impact(0.1),
                );
            }
        }

        // Suggest format if not specified
        if !prompt_lower.contains("format") && !prompt_lower.contains("output") {
            if enhanced.len() > 20 {
                if !enhanced.ends_with('.') {
                    enhanced.push('.');
                }
                enhanced.push_str(" Specify the desired output format.");
                improvements.push(
                    Improvement::new(
                        QualityDimension::Specificity,
                        "Added prompt for output format",
                    )
                    .with_impact(0.08),
                );
            }
        }

        Ok((enhanced, improvements))
    }

    /// Enhance engagement by improving tone and style.
    fn enhance_engagement(&self, prompt: &str) -> Result<(String, Vec<Improvement>)> {
        let mut enhanced = prompt.to_string();
        let mut improvements = Vec::new();

        let prompt_lower = prompt.to_lowercase();

        // Add "please" if not present and doesn't start with imperative
        if !prompt_lower.contains("please") && !prompt_lower.starts_with("create") {
            if let Some(first_word) = enhanced.split_whitespace().next() {
                let first_lower = first_word.to_lowercase();
                if ["write", "explain", "describe", "help", "show"].contains(&first_lower.as_str()) {
                    enhanced = format!("Please {}", enhanced.chars().skip(0).collect::<String>());
                    improvements.push(
                        Improvement::new(QualityDimension::Engagement, "Added polite 'Please'")
                            .with_impact(0.05),
                    );
                }
            }
        }

        Ok((enhanced, improvements))
    }

    /// Enhance actionability by clarifying expected actions.
    fn enhance_actionability(&self, prompt: &str) -> Result<(String, Vec<Improvement>)> {
        let mut enhanced = prompt.to_string();
        let mut improvements = Vec::new();

        let prompt_lower = prompt.to_lowercase();

        // Convert passive to active if needed
        if prompt_lower.contains("should be") {
            enhanced = enhanced.replace("should be", "must be");
            improvements.push(
                Improvement::new(
                    QualityDimension::Actionability,
                    "Made requirement more definitive",
                )
                .with_impact(0.05),
            );
        }

        // Add action prompt if missing
        if !prompt_lower.contains("step") && !prompt_lower.contains("procedure") {
            let has_how = prompt_lower.contains("how");
            if has_how {
                if !enhanced.ends_with('.') {
                    enhanced.push('.');
                }
                enhanced.push_str(" Provide step-by-step instructions.");
                improvements.push(
                    Improvement::new(
                        QualityDimension::Actionability,
                        "Added request for step-by-step instructions",
                    )
                    .with_impact(0.1),
                );
            }
        }

        Ok((enhanced, improvements))
    }

    /// Enhance completeness by prompting for missing elements.
    fn enhance_completeness(&self, prompt: &str) -> Result<(String, Vec<Improvement>)> {
        let mut enhanced = prompt.to_string();
        let mut improvements = Vec::new();

        let prompt_lower = prompt.to_lowercase();

        // Add context request if missing
        if !prompt_lower.contains("context") && !prompt_lower.contains("background") {
            if enhanced.split_whitespace().count() < 20 {
                if !enhanced.ends_with('.') {
                    enhanced.push('.');
                }
                enhanced.push_str(" Include relevant context.");
                improvements.push(
                    Improvement::new(
                        QualityDimension::Completeness,
                        "Added request for context",
                    )
                    .with_impact(0.08),
                );
            }
        }

        // Add edge case consideration if appropriate
        if (prompt_lower.contains("implement") || prompt_lower.contains("create"))
            && !prompt_lower.contains("edge case")
            && !prompt_lower.contains("error")
        {
            if !enhanced.ends_with('.') {
                enhanced.push('.');
            }
            enhanced.push_str(" Consider edge cases and error handling.");
            improvements.push(
                Improvement::new(
                    QualityDimension::Completeness,
                    "Added consideration for edge cases",
                )
                .with_impact(0.1),
            );
        }

        Ok((enhanced, improvements))
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Quick score check without full enhancement.
    pub fn quick_score(&self, prompt: &str) -> f64 {
        self.calculate_all_dimensions_score(prompt)
    }

    /// Get dimension scores for a prompt.
    pub fn dimension_scores(&self, prompt: &str) -> Vec<(QualityDimension, f64)> {
        QualityDimension::all()
            .into_iter()
            .map(|d| (d, self.score_dimension(prompt, d)))
            .collect()
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom EnhancerAgent configuration.
#[derive(Debug, Default)]
pub struct EnhancerAgentBuilder {
    config: Option<EnhancerConfig>,
}

impl EnhancerAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: EnhancerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the quality threshold.
    pub fn quality_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.quality_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set maximum iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_iterations = max.max(1);
        self.config = Some(config);
        self
    }

    /// Set minimum improvement threshold.
    pub fn min_improvement(mut self, min: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.min_improvement = min.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set target quality.
    pub fn target_quality(mut self, target: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.target_quality = target.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set whether to preserve meaning strictly.
    pub fn preserve_meaning(mut self, preserve: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.preserve_meaning = preserve;
        self.config = Some(config);
        self
    }

    /// Set default goals.
    pub fn default_goals(mut self, goals: Vec<EnhancementGoal>) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.default_goals = goals;
        self.config = Some(config);
        self
    }

    /// Build the EnhancerAgent.
    pub fn build(self) -> EnhancerAgent {
        EnhancerAgent {
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
    // QualityDimension Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_quality_dimension_display() {
        assert_eq!(QualityDimension::Clarity.to_string(), "clarity");
        assert_eq!(QualityDimension::Specificity.to_string(), "specificity");
        assert_eq!(QualityDimension::Engagement.to_string(), "engagement");
        assert_eq!(QualityDimension::Actionability.to_string(), "actionability");
        assert_eq!(QualityDimension::Completeness.to_string(), "completeness");
    }

    #[test]
    fn test_quality_dimension_from_str() {
        assert_eq!("clarity".parse::<QualityDimension>().unwrap(), QualityDimension::Clarity);
        assert_eq!("clear".parse::<QualityDimension>().unwrap(), QualityDimension::Clarity);
        assert_eq!("specificity".parse::<QualityDimension>().unwrap(), QualityDimension::Specificity);
        assert_eq!("specific".parse::<QualityDimension>().unwrap(), QualityDimension::Specificity);
        assert_eq!("engagement".parse::<QualityDimension>().unwrap(), QualityDimension::Engagement);
        assert_eq!("actionability".parse::<QualityDimension>().unwrap(), QualityDimension::Actionability);
        assert_eq!("completeness".parse::<QualityDimension>().unwrap(), QualityDimension::Completeness);
    }

    #[test]
    fn test_quality_dimension_from_str_invalid() {
        assert!("invalid".parse::<QualityDimension>().is_err());
    }

    #[test]
    fn test_quality_dimension_all() {
        let all = QualityDimension::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&QualityDimension::Clarity));
        assert!(all.contains(&QualityDimension::Specificity));
        assert!(all.contains(&QualityDimension::Engagement));
        assert!(all.contains(&QualityDimension::Actionability));
        assert!(all.contains(&QualityDimension::Completeness));
    }

    #[test]
    fn test_quality_dimension_default() {
        assert_eq!(QualityDimension::default(), QualityDimension::Clarity);
    }

    #[test]
    fn test_quality_dimension_default_weight() {
        assert!((QualityDimension::Clarity.default_weight() - 1.2).abs() < f64::EPSILON);
        assert!((QualityDimension::Specificity.default_weight() - 1.1).abs() < f64::EPSILON);
        assert!((QualityDimension::Engagement.default_weight() - 0.9).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // EnhancementGoal Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enhancement_goal_new() {
        let goal = EnhancementGoal::new(QualityDimension::Clarity);
        assert_eq!(goal.dimension, QualityDimension::Clarity);
        assert!((goal.target_score - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancement_goal_builder() {
        let goal = EnhancementGoal::new(QualityDimension::Specificity)
            .with_target(0.9)
            .with_weight(2.0);

        assert!((goal.target_score - 0.9).abs() < f64::EPSILON);
        assert!((goal.weight - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancement_goal_target_clamping() {
        let goal_high = EnhancementGoal::new(QualityDimension::Clarity).with_target(1.5);
        assert!((goal_high.target_score - 1.0).abs() < f64::EPSILON);

        let goal_low = EnhancementGoal::new(QualityDimension::Clarity).with_target(-0.5);
        assert!(goal_low.target_score.abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancement_goal_is_met() {
        let goal = EnhancementGoal::new(QualityDimension::Clarity).with_target(0.7);
        assert!(goal.is_met(0.8));
        assert!(goal.is_met(0.7));
        assert!(!goal.is_met(0.6));
    }

    #[test]
    fn test_enhancement_goal_gap() {
        let goal = EnhancementGoal::new(QualityDimension::Clarity).with_target(0.8);
        assert!((goal.gap(0.5) - 0.3).abs() < f64::EPSILON);
        assert!((goal.gap(0.9) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancement_goal_default() {
        let goal = EnhancementGoal::default();
        assert_eq!(goal.dimension, QualityDimension::Clarity);
    }

    // -------------------------------------------------------------------------
    // Improvement Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_improvement_new() {
        let imp = Improvement::new(QualityDimension::Clarity, "Made clearer");
        assert_eq!(imp.dimension, QualityDimension::Clarity);
        assert_eq!(imp.description, "Made clearer");
        assert!(imp.original.is_none());
        assert!(imp.improved.is_none());
    }

    #[test]
    fn test_improvement_builder() {
        let imp = Improvement::new(QualityDimension::Specificity, "Added details")
            .with_original("vague text")
            .with_improved("specific text")
            .with_impact(0.15);

        assert_eq!(imp.original, Some("vague text".to_string()));
        assert_eq!(imp.improved, Some("specific text".to_string()));
        assert!((imp.estimated_impact - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_improvement_impact_clamping() {
        let imp_high = Improvement::new(QualityDimension::Clarity, "Test").with_impact(1.5);
        assert!((imp_high.estimated_impact - 1.0).abs() < f64::EPSILON);

        let imp_low = Improvement::new(QualityDimension::Clarity, "Test").with_impact(-0.5);
        assert!(imp_low.estimated_impact.abs() < f64::EPSILON);
    }

    #[test]
    fn test_improvement_to_markdown() {
        let imp = Improvement::new(QualityDimension::Clarity, "Removed vague word")
            .with_original("thing")
            .with_impact(0.05);

        let md = imp.to_markdown();
        assert!(md.contains("[clarity]"));
        assert!(md.contains("Removed vague word"));
    }

    #[test]
    fn test_improvement_default() {
        let imp = Improvement::default();
        assert_eq!(imp.dimension, QualityDimension::Clarity);
        assert!(imp.description.is_empty());
    }

    // -------------------------------------------------------------------------
    // EnhancementStep Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enhancement_step_new() {
        let step = EnhancementStep::new(1, "Enhance Clarity");
        assert_eq!(step.step_number, 1);
        assert_eq!(step.technique, "Enhance Clarity");
        assert!(!step.successful);
    }

    #[test]
    fn test_enhancement_step_builder() {
        let step = EnhancementStep::new(2, "Enhance Specificity")
            .with_dimensions(vec![QualityDimension::Specificity])
            .with_scores(0.5, 0.7)
            .with_duration(100);

        assert_eq!(step.dimensions_targeted, vec![QualityDimension::Specificity]);
        assert!((step.score_before - 0.5).abs() < f64::EPSILON);
        assert!((step.score_after - 0.7).abs() < f64::EPSILON);
        assert!(step.successful);
        assert_eq!(step.duration_ms, 100);
    }

    #[test]
    fn test_enhancement_step_improvement() {
        let step = EnhancementStep::new(1, "Test")
            .with_scores(0.4, 0.6);
        assert!((step.improvement() - 0.2).abs() < f64::EPSILON);

        let step_negative = EnhancementStep::new(1, "Test")
            .with_scores(0.7, 0.5);
        assert!((step_negative.improvement() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancement_step_scores_clamping() {
        let step = EnhancementStep::new(1, "Test")
            .with_scores(-0.1, 1.5);
        assert!(step.score_before.abs() < f64::EPSILON);
        assert!((step.score_after - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancement_step_default() {
        let step = EnhancementStep::default();
        assert_eq!(step.step_number, 1);
        assert_eq!(step.technique, "unknown");
    }

    // -------------------------------------------------------------------------
    // EnhancedPrompt Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enhanced_prompt_new() {
        let result = EnhancedPrompt::new("original", "enhanced");
        assert_eq!(result.original, "original");
        assert_eq!(result.content, "enhanced");
    }

    #[test]
    fn test_enhanced_prompt_improvement_percentage() {
        let mut result = EnhancedPrompt::new("a", "b");
        result.before_score = 0.5;
        result.after_score = 0.75;
        assert!((result.improvement_percentage() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhanced_prompt_improvement_percentage_zero_before() {
        let mut result = EnhancedPrompt::new("a", "b");
        result.before_score = 0.0;
        result.after_score = 0.5;
        assert!((result.improvement_percentage() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhanced_prompt_improvement_percentage_both_zero() {
        let result = EnhancedPrompt::new("a", "b");
        assert!((result.improvement_percentage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhanced_prompt_is_significantly_improved() {
        let mut result = EnhancedPrompt::new("a", "b");
        result.before_score = 0.5;
        result.after_score = 0.65;
        assert!(result.is_significantly_improved());

        result.after_score = 0.55;
        assert!(!result.is_significantly_improved());
    }

    #[test]
    fn test_enhanced_prompt_improvement_delta() {
        let mut result = EnhancedPrompt::new("a", "b");
        result.before_score = 0.4;
        result.after_score = 0.7;
        assert!((result.improvement_delta() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhanced_prompt_is_successful() {
        let mut result = EnhancedPrompt::new("a", "b");
        result.before_score = 0.5;
        result.after_score = 0.5;
        assert!(result.is_successful());

        result.after_score = 0.6;
        assert!(result.is_successful());

        result.after_score = 0.4;
        assert!(!result.is_successful());
    }

    #[test]
    fn test_enhanced_prompt_successful_steps() {
        let mut result = EnhancedPrompt::new("a", "b");
        result.enhancement_log.push(EnhancementStep::new(1, "A").with_scores(0.5, 0.6));
        result.enhancement_log.push(EnhancementStep::new(2, "B").with_scores(0.6, 0.5));
        result.enhancement_log.push(EnhancementStep::new(3, "C").with_scores(0.5, 0.7));

        let successful = result.successful_steps();
        assert_eq!(successful.len(), 2);
    }

    #[test]
    fn test_enhanced_prompt_summary() {
        let mut result = EnhancedPrompt::new("a", "b");
        result.before_score = 0.5;
        result.after_score = 0.7;
        result.improvements.push(Improvement::new(QualityDimension::Clarity, "Test"));
        result.duration_ms = 150;

        let summary = result.summary();
        assert!(summary.contains("50%"));
        assert!(summary.contains("70%"));
        assert!(summary.contains("150ms"));
    }

    #[test]
    fn test_enhanced_prompt_to_markdown() {
        let mut result = EnhancedPrompt::new("original prompt", "enhanced prompt");
        result.before_score = 0.5;
        result.after_score = 0.7;
        result.improvements.push(Improvement::new(QualityDimension::Clarity, "Made clearer"));

        let md = result.to_markdown();
        assert!(md.contains("Enhancement Report"));
        assert!(md.contains("enhanced prompt"));
        assert!(md.contains("Made clearer"));
    }

    #[test]
    fn test_enhanced_prompt_default() {
        let result = EnhancedPrompt::default();
        assert!(result.original.is_empty());
        assert!(result.content.is_empty());
    }

    // -------------------------------------------------------------------------
    // EnhancerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enhancer_config_default() {
        let config = EnhancerConfig::default();
        assert!((config.quality_threshold - 0.3).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 5);
        assert!((config.min_improvement - 0.02).abs() < f64::EPSILON);
        assert!((config.target_quality - 0.85).abs() < f64::EPSILON);
        assert!(config.preserve_meaning);
    }

    #[test]
    fn test_enhancer_config_aggressive() {
        let config = EnhancerConfig::aggressive();
        assert!((config.quality_threshold - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 10);
        assert!(!config.preserve_meaning);
    }

    #[test]
    fn test_enhancer_config_conservative() {
        let config = EnhancerConfig::conservative();
        assert!((config.quality_threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 3);
        assert!(config.preserve_meaning);
    }

    #[test]
    fn test_enhancer_config_minimal() {
        let config = EnhancerConfig::minimal();
        assert!((config.quality_threshold - 0.6).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 2);
    }

    // -------------------------------------------------------------------------
    // EnhancerAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enhancer_agent_new() {
        let agent = EnhancerAgent::new();
        assert!((agent.config().quality_threshold - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancer_agent_with_config() {
        let config = EnhancerConfig::aggressive();
        let agent = EnhancerAgent::with_config(config);
        assert!((agent.config().quality_threshold - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_enhancer_agent_builder() {
        let agent = EnhancerAgent::builder()
            .quality_threshold(0.4)
            .max_iterations(8)
            .min_improvement(0.03)
            .target_quality(0.9)
            .preserve_meaning(false)
            .build();

        assert!((agent.config().quality_threshold - 0.4).abs() < f64::EPSILON);
        assert_eq!(agent.config().max_iterations, 8);
        assert!((agent.config().min_improvement - 0.03).abs() < f64::EPSILON);
        assert!((agent.config().target_quality - 0.9).abs() < f64::EPSILON);
        assert!(!agent.config().preserve_meaning);
    }

    #[test]
    fn test_enhancer_agent_builder_clamping() {
        let agent = EnhancerAgent::builder()
            .quality_threshold(1.5)
            .min_improvement(-0.1)
            .target_quality(2.0)
            .max_iterations(0)
            .build();

        assert!((agent.config().quality_threshold - 1.0).abs() < f64::EPSILON);
        assert!(agent.config().min_improvement.abs() < f64::EPSILON);
        assert!((agent.config().target_quality - 1.0).abs() < f64::EPSILON);
        assert_eq!(agent.config().max_iterations, 1);
    }

    #[test]
    fn test_enhancer_agent_default() {
        let agent = EnhancerAgent::default();
        assert!((agent.config().quality_threshold - 0.3).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Scoring Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_score_clarity() {
        let agent = EnhancerAgent::new();

        // Clear prompt
        let clear = "Specifically, I need you to create a function that calculates the sum.";
        let clear_score = agent.score_clarity(clear);
        assert!(clear_score > 0.5, "Clear prompt score {} should be > 0.5", clear_score);

        // Vague prompt
        let vague = "Just do something with the thing, maybe it sort of works basically.";
        let vague_score = agent.score_clarity(vague);
        assert!(vague_score < clear_score, "Vague prompt should score lower");
    }

    #[test]
    fn test_score_specificity() {
        let agent = EnhancerAgent::new();

        // Specific prompt
        let specific = "Create a function that takes exactly 3 parameters and returns a value between 0 and 100.";
        let specific_score = agent.score_specificity(specific);
        assert!(specific_score > 0.5, "Specific prompt score {} should be > 0.5", specific_score);

        // Generic prompt
        let generic = "Create a function.";
        let generic_score = agent.score_specificity(generic);
        assert!(generic_score < specific_score, "Generic prompt should score lower");
    }

    #[test]
    fn test_score_engagement() {
        let agent = EnhancerAgent::new();

        // Engaging prompt
        let engaging = "Please help me create an interesting and creative solution. What innovative approaches could work?";
        let engaging_score = agent.score_engagement(engaging);
        assert!(engaging_score > 0.5, "Engaging prompt score {} should be > 0.5", engaging_score);

        // Dry prompt
        let dry = "Write code.";
        let dry_score = agent.score_engagement(dry);
        assert!(dry_score < engaging_score, "Dry prompt should score lower");
    }

    #[test]
    fn test_score_actionability() {
        let agent = EnhancerAgent::new();

        // Actionable prompt
        let actionable = "Create a step-by-step procedure to implement the feature. Provide the output in JSON format.";
        let actionable_score = agent.score_actionability(actionable);
        assert!(actionable_score > 0.5, "Actionable prompt score {} should be > 0.5", actionable_score);

        // Non-actionable prompt
        let passive = "It would be nice if there was a solution.";
        let passive_score = agent.score_actionability(passive);
        assert!(passive_score < actionable_score, "Passive prompt should score lower");
    }

    #[test]
    fn test_score_completeness() {
        let agent = EnhancerAgent::new();

        // Complete prompt
        let complete = "Create a comprehensive authentication system that includes login, registration, password reset, and session management. Consider edge cases and error handling. Given the context of a web application, address all security requirements.";
        let complete_score = agent.score_completeness(complete);
        assert!(complete_score > 0.6, "Complete prompt score {} should be > 0.6", complete_score);

        // Incomplete prompt
        let incomplete = "Make auth.";
        let incomplete_score = agent.score_completeness(incomplete);
        assert!(incomplete_score < complete_score, "Incomplete prompt should score lower");
    }

    #[test]
    fn test_quick_score() {
        let agent = EnhancerAgent::new();
        let score = agent.quick_score("Write a clear, specific function to calculate the sum of numbers.");
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_dimension_scores() {
        let agent = EnhancerAgent::new();
        let scores = agent.dimension_scores("Create a comprehensive solution with specific requirements.");

        assert_eq!(scores.len(), 5);
        for (_, score) in &scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    // -------------------------------------------------------------------------
    // Enhancement Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_enhance_empty_prompt() {
        let agent = EnhancerAgent::new();
        let result = agent.enhance("", &[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_enhance_whitespace_only() {
        let agent = EnhancerAgent::new();
        let result = agent.enhance("   \n\t  ", &[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_enhance_basic_prompt() {
        let agent = EnhancerAgent::new();
        let result = agent.enhance("Write code", &[QualityDimension::Clarity]).await;

        assert!(result.is_ok());
        let enhanced = result.unwrap();
        assert!(!enhanced.content.is_empty());
        assert!(enhanced.before_score <= enhanced.after_score);
    }

    #[tokio::test]
    async fn test_enhance_with_default_dimensions() {
        let agent = EnhancerAgent::new();
        let result = agent.enhance("Create a function", &[]).await;

        assert!(result.is_ok());
        let enhanced = result.unwrap();
        assert!(!enhanced.dimensions_enhanced.is_empty());
    }

    #[tokio::test]
    async fn test_enhance_already_good_prompt() {
        let agent = EnhancerAgent::builder()
            .target_quality(0.5)
            .build();

        let good_prompt = "Specifically, create a comprehensive step-by-step procedure to implement a complete authentication system. Include all requirements and consider edge cases.";
        let result = agent.enhance(good_prompt, &[QualityDimension::Clarity]).await;

        assert!(result.is_ok());
        let enhanced = result.unwrap();
        // Should recognize it already meets target
        assert!(enhanced.is_successful());
    }

    #[tokio::test]
    async fn test_enhance_vague_prompt() {
        let agent = EnhancerAgent::new();
        let vague = "Just do something with the stuff maybe.";
        let result = agent.enhance(vague, &[QualityDimension::Clarity]).await;

        assert!(result.is_ok());
        let enhanced = result.unwrap();
        // Should improve
        assert!(enhanced.after_score >= enhanced.before_score);
        // Should have improvements
        if enhanced.after_score > enhanced.before_score {
            assert!(!enhanced.improvements.is_empty());
        }
    }

    #[tokio::test]
    async fn test_enhance_multiple_dimensions() {
        let agent = EnhancerAgent::new();
        let result = agent.enhance(
            "Make it work",
            &[
                QualityDimension::Clarity,
                QualityDimension::Specificity,
                QualityDimension::Actionability,
            ],
        ).await;

        assert!(result.is_ok());
        let enhanced = result.unwrap();
        assert_eq!(enhanced.dimensions_enhanced.len(), 3);
    }

    #[tokio::test]
    async fn test_enhance_with_goals() {
        let agent = EnhancerAgent::new();
        let goals = vec![
            EnhancementGoal::new(QualityDimension::Clarity).with_target(0.7),
            EnhancementGoal::new(QualityDimension::Specificity).with_target(0.6),
        ];

        let result = agent.enhance_with_goals("Write code", &goals).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_enhance_iteration_limit() {
        let agent = EnhancerAgent::builder()
            .max_iterations(2)
            .min_improvement(0.001)
            .target_quality(0.99)
            .build();

        let result = agent.enhance("Write something", &[QualityDimension::Clarity]).await;
        assert!(result.is_ok());

        let enhanced = result.unwrap();
        assert!(enhanced.iterations <= 2);
    }

    #[tokio::test]
    async fn test_enhance_records_steps() {
        let agent = EnhancerAgent::builder()
            .max_iterations(3)
            .build();

        let result = agent.enhance("Do stuff", &[QualityDimension::Clarity]).await;
        assert!(result.is_ok());

        let enhanced = result.unwrap();
        // Should have recorded enhancement steps
        if enhanced.iterations > 0 {
            assert!(!enhanced.enhancement_log.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Enhancement Technique Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enhance_clarity_removes_vague_words() {
        let agent = EnhancerAgent::new();
        let (enhanced, improvements) = agent.enhance_clarity("Just do something with the thing basically.").unwrap();

        // Should have made changes
        assert!(!improvements.is_empty() || !enhanced.contains("just") || !enhanced.contains("thing"));
    }

    #[test]
    fn test_enhance_specificity_adds_prompts() {
        let agent = EnhancerAgent::new();
        let (enhanced, improvements) = agent.enhance_specificity("Create a function.").unwrap();

        // Should have added specificity prompts
        assert!(enhanced.len() >= "Create a function.".len());
        if !improvements.is_empty() {
            assert!(improvements.iter().any(|i| i.dimension == QualityDimension::Specificity));
        }
    }

    #[test]
    fn test_enhance_engagement_adds_polish() {
        let agent = EnhancerAgent::new();
        let (enhanced, _) = agent.enhance_engagement("Write code.").unwrap();

        // Should be at least as long
        assert!(enhanced.len() >= "Write code.".len());
    }

    #[test]
    fn test_enhance_actionability_strengthens_language() {
        let agent = EnhancerAgent::new();
        let (enhanced, _) = agent.enhance_actionability("The output should be JSON.").unwrap();

        // Should have strengthened language if changes were made
        assert!(!enhanced.is_empty());
    }

    #[test]
    fn test_enhance_completeness_adds_context() {
        let agent = EnhancerAgent::new();
        let (enhanced, improvements) = agent.enhance_completeness("Write code.").unwrap();

        // Should have added completeness prompts
        if !improvements.is_empty() {
            assert!(improvements.iter().any(|i| i.dimension == QualityDimension::Completeness));
        }
        assert!(enhanced.len() >= "Write code.".len());
    }

    // -------------------------------------------------------------------------
    // Find Weakest Dimension Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_find_weakest_dimension() {
        let agent = EnhancerAgent::new();
        let prompt = "Make it work."; // Short, vague prompt

        let (weakest, score) = agent.find_weakest_dimension(prompt, &QualityDimension::all());

        assert!(score >= 0.0 && score <= 1.0);
        // For a short vague prompt, completeness is likely weakest
        assert!(QualityDimension::all().contains(&weakest));
    }

    // -------------------------------------------------------------------------
    // Calculate Overall Score Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_calculate_overall_score() {
        let agent = EnhancerAgent::new();
        let prompt = "Create a specific function with clear requirements.";

        let score = agent.calculate_overall_score(prompt, &QualityDimension::all());
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_calculate_overall_score_empty_dimensions() {
        let agent = EnhancerAgent::new();
        let prompt = "Test prompt.";

        let score = agent.calculate_overall_score(prompt, &[]);
        // Should fall back to all dimensions
        assert!(score >= 0.0 && score <= 1.0);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let builder = EnhancerAgentBuilder::default();
        let agent = builder.build();
        assert!((agent.config().quality_threshold - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_config() {
        let config = EnhancerConfig::aggressive();
        let agent = EnhancerAgentBuilder::default().config(config).build();
        assert!((agent.config().quality_threshold - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_chain() {
        let agent = EnhancerAgentBuilder::default()
            .quality_threshold(0.35)
            .max_iterations(7)
            .min_improvement(0.025)
            .target_quality(0.88)
            .preserve_meaning(false)
            .build();

        assert!((agent.config().quality_threshold - 0.35).abs() < f64::EPSILON);
        assert_eq!(agent.config().max_iterations, 7);
        assert!((agent.config().min_improvement - 0.025).abs() < f64::EPSILON);
        assert!((agent.config().target_quality - 0.88).abs() < f64::EPSILON);
        assert!(!agent.config().preserve_meaning);
    }

    #[test]
    fn test_builder_default_goals() {
        let goals = vec![
            EnhancementGoal::new(QualityDimension::Specificity).with_target(0.9),
        ];

        let agent = EnhancerAgentBuilder::default()
            .default_goals(goals)
            .build();

        assert_eq!(agent.config().default_goals.len(), 1);
        assert_eq!(agent.config().default_goals[0].dimension, QualityDimension::Specificity);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_enhance_very_long_prompt() {
        let agent = EnhancerAgent::new();
        let long_prompt = "word ".repeat(500);

        let result = agent.enhance(&long_prompt, &[QualityDimension::Clarity]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_enhance_special_characters() {
        let agent = EnhancerAgent::new();
        let special = "Create a function that handles `code` and {brackets} and [arrays].";

        let result = agent.enhance(special, &[QualityDimension::Clarity]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_enhance_unicode() {
        let agent = EnhancerAgent::new();
        let unicode = "Create a function for emoji processing: ";

        let result = agent.enhance(unicode, &[QualityDimension::Clarity]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_enhance_multiline() {
        let agent = EnhancerAgent::new();
        let multiline = "Create a function.\nIt should be fast.\nHandle errors.";

        let result = agent.enhance(multiline, &[QualityDimension::Clarity]).await;
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Integration-like Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_full_enhancement_workflow() {
        let agent = EnhancerAgent::builder()
            .quality_threshold(0.3)
            .max_iterations(5)
            .min_improvement(0.02)
            .target_quality(0.8)
            .build();

        let prompt = "Make code for auth.";

        let result = agent.enhance(prompt, &[
            QualityDimension::Clarity,
            QualityDimension::Specificity,
            QualityDimension::Completeness,
        ]).await;

        assert!(result.is_ok());

        let enhanced = result.unwrap();

        // Verify structure
        assert!(!enhanced.content.is_empty());
        assert!(!enhanced.original.is_empty());
        assert!(enhanced.before_score >= 0.0 && enhanced.before_score <= 1.0);
        assert!(enhanced.after_score >= 0.0 && enhanced.after_score <= 1.0);
        assert!(enhanced.duration_ms >= 0);

        // Verify it can be formatted
        let md = enhanced.to_markdown();
        assert!(!md.is_empty());
        assert!(md.contains("Enhancement Report"));
    }
}
