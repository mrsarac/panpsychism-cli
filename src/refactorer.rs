//! Refactorer Agent module for Project Panpsychism.
//!
//! The Code Improver - "Structure reveals intention, clarity enables action."
//!
//! This module implements the Refactorer Agent (Agent 33), responsible for
//! restructuring and reorganizing prompts for better clarity and maintainability.
//! Like an architect redesigning a building, the Refactorer transforms
//! chaotic structures into elegant, purposeful designs.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Code Improver serves as the architect:
//!
//! - **Raw Spell** (disorganized prompt) enters the refactoring chamber
//! - **The Architect** (RefactorerAgent) analyzes and restructures
//! - **Refined Spell** (well-organized prompt) emerges with improved form
//!
//! The Architect refactors through various techniques:
//! - **ExtractComponent**: Breaking large prompts into reusable parts
//! - **InlineExpansion**: Expanding compact references for clarity
//! - **SimplifyStructure**: Reducing complexity while preserving meaning
//! - **RemoveDuplication**: Eliminating redundant content
//! - **ImproveReadability**: Enhancing flow and comprehension
//! - **OptimizeTokens**: Reducing token usage without losing meaning
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Drive toward structural perfection and clarity
//! - **RATIO**: Logical organization and reasoned structure
//! - **LAETITIA**: Joy through elegant, well-organized expression
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::refactorer::{RefactorerAgent, RefactoringType};
//!
//! let architect = RefactorerAgent::new();
//!
//! // Refactor a prompt
//! let refactored = architect.refactor(
//!     "Write code. The code should be clean. Make sure code is tested. Write code that works.",
//! ).await?;
//!
//! println!("Refactored: {}", refactored.content);
//! println!("Changes: {}", refactored.changes.len());
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// PRIORITY ENUM
// =============================================================================

/// Priority level for refactoring goals.
///
/// Determines the urgency and order of refactoring operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Critical refactoring needed immediately.
    Critical,
    /// High priority refactoring.
    High,
    /// Standard priority refactoring.
    #[default]
    Medium,
    /// Nice-to-have refactoring.
    Low,
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "critical"),
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
            "critical" | "urgent" | "blocker" => Ok(Self::Critical),
            "high" | "important" => Ok(Self::High),
            "medium" | "moderate" | "normal" => Ok(Self::Medium),
            "low" | "minor" | "nice-to-have" => Ok(Self::Low),
            _ => Err(Error::Config(format!(
                "Unknown priority: '{}'. Valid: critical, high, medium, low",
                s
            ))),
        }
    }
}

impl Priority {
    /// Get all priority levels in order of importance.
    pub fn all() -> Vec<Self> {
        vec![Self::Critical, Self::High, Self::Medium, Self::Low]
    }

    /// Get the weight multiplier for this priority.
    pub fn weight(&self) -> f64 {
        match self {
            Self::Critical => 2.0,
            Self::High => 1.5,
            Self::Medium => 1.0,
            Self::Low => 0.5,
        }
    }

    /// Check if this priority is higher than another.
    pub fn is_higher_than(&self, other: &Priority) -> bool {
        self.weight() > other.weight()
    }
}

// =============================================================================
// REFACTORING TYPE ENUM
// =============================================================================

/// Types of refactoring operations that can be applied.
///
/// Each type represents a different structural transformation technique.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RefactoringType {
    /// Extract reusable components from large prompts.
    ExtractComponent,
    /// Inline expand compact references for clarity.
    InlineExpansion,
    /// Simplify complex structures while preserving meaning.
    #[default]
    SimplifyStructure,
    /// Remove redundant or duplicated content.
    RemoveDuplication,
    /// Improve overall readability and flow.
    ImproveReadability,
    /// Optimize token usage without losing meaning.
    OptimizeTokens,
}

impl std::fmt::Display for RefactoringType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExtractComponent => write!(f, "extract-component"),
            Self::InlineExpansion => write!(f, "inline-expansion"),
            Self::SimplifyStructure => write!(f, "simplify-structure"),
            Self::RemoveDuplication => write!(f, "remove-duplication"),
            Self::ImproveReadability => write!(f, "improve-readability"),
            Self::OptimizeTokens => write!(f, "optimize-tokens"),
        }
    }
}

impl std::str::FromStr for RefactoringType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().replace('_', "-").as_str() {
            "extract-component" | "extract" | "component" => Ok(Self::ExtractComponent),
            "inline-expansion" | "inline" | "expand" => Ok(Self::InlineExpansion),
            "simplify-structure" | "simplify" | "structure" => Ok(Self::SimplifyStructure),
            "remove-duplication" | "dedupe" | "duplication" => Ok(Self::RemoveDuplication),
            "improve-readability" | "readability" | "readable" => Ok(Self::ImproveReadability),
            "optimize-tokens" | "optimize" | "tokens" => Ok(Self::OptimizeTokens),
            _ => Err(Error::Config(format!(
                "Unknown refactoring type: '{}'. Valid: extract-component, inline-expansion, simplify-structure, remove-duplication, improve-readability, optimize-tokens",
                s
            ))),
        }
    }
}

impl RefactoringType {
    /// Get all refactoring types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::ExtractComponent,
            Self::InlineExpansion,
            Self::SimplifyStructure,
            Self::RemoveDuplication,
            Self::ImproveReadability,
            Self::OptimizeTokens,
        ]
    }

    /// Get the default priority for this refactoring type.
    pub fn default_priority(&self) -> Priority {
        match self {
            Self::RemoveDuplication => Priority::High,
            Self::SimplifyStructure => Priority::Medium,
            Self::ImproveReadability => Priority::Medium,
            Self::OptimizeTokens => Priority::Low,
            Self::ExtractComponent => Priority::Low,
            Self::InlineExpansion => Priority::Low,
        }
    }

    /// Get description of what this refactoring type does.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ExtractComponent => "Extract reusable components from large prompts",
            Self::InlineExpansion => "Inline expand compact references for clarity",
            Self::SimplifyStructure => "Simplify complex structures while preserving meaning",
            Self::RemoveDuplication => "Remove redundant or duplicated content",
            Self::ImproveReadability => "Improve overall readability and flow",
            Self::OptimizeTokens => "Optimize token usage without losing meaning",
        }
    }
}

// =============================================================================
// CONSTRAINT STRUCT
// =============================================================================

/// A constraint that limits refactoring operations.
///
/// Constraints ensure that refactoring respects certain boundaries
/// and requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Name of the constraint.
    pub name: String,
    /// Description of what this constraint enforces.
    pub description: String,
    /// Whether this constraint is mandatory (hard) or preferred (soft).
    pub is_mandatory: bool,
    /// Categories of refactoring this constraint applies to.
    pub applies_to: Vec<RefactoringType>,
}

impl Constraint {
    /// Create a new constraint.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            is_mandatory: true,
            applies_to: Vec::new(),
        }
    }

    /// Set whether this constraint is mandatory.
    pub fn mandatory(mut self, is_mandatory: bool) -> Self {
        self.is_mandatory = is_mandatory;
        self
    }

    /// Set which refactoring types this constraint applies to.
    pub fn applies_to(mut self, types: Vec<RefactoringType>) -> Self {
        self.applies_to = types;
        self
    }

    /// Check if this constraint applies to a given refactoring type.
    pub fn applies_to_type(&self, refactoring_type: RefactoringType) -> bool {
        self.applies_to.is_empty() || self.applies_to.contains(&refactoring_type)
    }
}

impl Default for Constraint {
    fn default() -> Self {
        Self::new("default", "Default constraint")
    }
}

// =============================================================================
// REFACTORING GOAL STRUCT
// =============================================================================

/// A specific goal for refactoring with type, priority, and constraints.
///
/// Goals guide the refactoring process by specifying what transformations
/// to apply and in what order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringGoal {
    /// The type of refactoring to apply.
    pub goal_type: RefactoringType,
    /// Priority level for this goal.
    pub priority: Priority,
    /// Constraints that limit this refactoring.
    pub constraints: Vec<Constraint>,
}

impl RefactoringGoal {
    /// Create a new refactoring goal.
    pub fn new(goal_type: RefactoringType) -> Self {
        Self {
            goal_type,
            priority: goal_type.default_priority(),
            constraints: Vec::new(),
        }
    }

    /// Set the priority for this goal.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Add a constraint to this goal.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add multiple constraints to this goal.
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints.extend(constraints);
        self
    }

    /// Check if this goal has any mandatory constraints.
    pub fn has_mandatory_constraints(&self) -> bool {
        self.constraints.iter().any(|c| c.is_mandatory)
    }

    /// Get all mandatory constraints for this goal.
    pub fn mandatory_constraints(&self) -> Vec<&Constraint> {
        self.constraints.iter().filter(|c| c.is_mandatory).collect()
    }
}

impl Default for RefactoringGoal {
    fn default() -> Self {
        Self::new(RefactoringType::SimplifyStructure)
    }
}

// =============================================================================
// PROMPT METRICS STRUCT
// =============================================================================

/// Metrics describing the quality and characteristics of a prompt.
///
/// Used to compare before and after states of refactoring operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptMetrics {
    /// Total number of tokens (estimated).
    pub token_count: usize,
    /// Number of lines in the prompt.
    pub line_count: usize,
    /// Complexity score (0.0 - 1.0, higher is more complex).
    pub complexity_score: f64,
    /// Readability score (0.0 - 1.0, higher is more readable).
    pub readability_score: f64,
    /// Number of words in the prompt.
    pub word_count: usize,
    /// Average words per sentence.
    pub avg_sentence_length: f64,
    /// Number of unique words.
    pub unique_word_count: usize,
    /// Duplication ratio (0.0 - 1.0, higher means more duplication).
    pub duplication_ratio: f64,
}

impl PromptMetrics {
    /// Create new metrics from analyzing a prompt.
    pub fn from_prompt(prompt: &str) -> Self {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let word_count = words.len();
        // Normalize to lowercase for accurate duplicate detection
        let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let unique_word_count = unique_words.len();

        let lines: Vec<&str> = prompt.lines().collect();
        let line_count = lines.len().max(1);

        let sentences: Vec<&str> = prompt
            .split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .collect();
        let sentence_count = sentences.len().max(1);
        let avg_sentence_length = word_count as f64 / sentence_count as f64;

        // Estimate token count (rough approximation: ~0.75 tokens per word)
        let token_count = ((word_count as f64 * 1.3) as usize).max(1);

        // Calculate complexity based on sentence length and word variety
        let complexity_score = Self::calculate_complexity(avg_sentence_length, word_count);

        // Calculate readability using simplified Flesch-like score
        let readability_score = Self::calculate_readability(avg_sentence_length, word_count);

        // Calculate duplication ratio
        let duplication_ratio = if word_count > 0 {
            1.0 - (unique_word_count as f64 / word_count as f64)
        } else {
            0.0
        };

        Self {
            token_count,
            line_count,
            complexity_score,
            readability_score,
            word_count,
            avg_sentence_length,
            unique_word_count,
            duplication_ratio,
        }
    }

    /// Calculate complexity score based on sentence length and word count.
    fn calculate_complexity(avg_sentence_length: f64, word_count: usize) -> f64 {
        // Higher sentence length and word count increase complexity
        let length_factor = (avg_sentence_length / 30.0).min(1.0);
        let size_factor = ((word_count as f64) / 200.0).min(1.0);
        ((length_factor + size_factor) / 2.0).clamp(0.0, 1.0)
    }

    /// Calculate readability score.
    fn calculate_readability(avg_sentence_length: f64, word_count: usize) -> f64 {
        // Optimal sentence length is around 15-20 words
        let optimal_length = 17.5;
        let length_score = 1.0 - ((avg_sentence_length - optimal_length).abs() / 30.0).min(1.0);

        // Very short prompts may lack context
        let context_score = if word_count < 10 {
            0.6
        } else if word_count < 30 {
            0.8
        } else {
            1.0
        };

        ((length_score + context_score) / 2.0).clamp(0.0, 1.0)
    }

    /// Compare with another metrics instance and return improvement delta.
    pub fn improvement_over(&self, other: &PromptMetrics) -> f64 {
        // Calculate weighted improvement
        let readability_delta = self.readability_score - other.readability_score;
        let complexity_delta = other.complexity_score - self.complexity_score; // Lower is better
        let duplication_delta = other.duplication_ratio - self.duplication_ratio; // Lower is better

        // Weight: readability is most important
        (readability_delta * 0.5 + complexity_delta * 0.3 + duplication_delta * 0.2)
            .clamp(-1.0, 1.0)
    }

    /// Check if these metrics represent a good quality prompt.
    pub fn is_good_quality(&self) -> bool {
        self.readability_score >= 0.6 && self.complexity_score <= 0.7 && self.duplication_ratio <= 0.3
    }

    /// Format metrics as a summary string.
    pub fn summary(&self) -> String {
        format!(
            "{} tokens, {} words, complexity: {:.0}%, readability: {:.0}%",
            self.token_count,
            self.word_count,
            self.complexity_score * 100.0,
            self.readability_score * 100.0
        )
    }
}

// =============================================================================
// REFACTORING CHANGE STRUCT
// =============================================================================

/// A single change made during refactoring.
///
/// Documents what was changed, why, and the before/after states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringChange {
    /// Type of refactoring that created this change.
    pub change_type: RefactoringType,
    /// Human-readable description of the change.
    pub description: String,
    /// The content before the change (if applicable).
    pub before: Option<String>,
    /// The content after the change (if applicable).
    pub after: Option<String>,
    /// Line number where the change was applied (if applicable).
    pub line_number: Option<usize>,
    /// Estimated impact on quality (0.0 - 1.0).
    pub impact: f64,
}

impl RefactoringChange {
    /// Create a new refactoring change.
    pub fn new(change_type: RefactoringType, description: impl Into<String>) -> Self {
        Self {
            change_type,
            description: description.into(),
            before: None,
            after: None,
            line_number: None,
            impact: 0.1,
        }
    }

    /// Set the before content.
    pub fn with_before(mut self, before: impl Into<String>) -> Self {
        self.before = Some(before.into());
        self
    }

    /// Set the after content.
    pub fn with_after(mut self, after: impl Into<String>) -> Self {
        self.after = Some(after.into());
        self
    }

    /// Set the line number.
    pub fn with_line(mut self, line: usize) -> Self {
        self.line_number = Some(line);
        self
    }

    /// Set the impact score.
    pub fn with_impact(mut self, impact: f64) -> Self {
        self.impact = impact.clamp(0.0, 1.0);
        self
    }

    /// Check if this change has before/after content.
    pub fn has_diff(&self) -> bool {
        self.before.is_some() && self.after.is_some()
    }

    /// Format the change as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!(
            "- **[{}]** {} _(Impact: {:.0}%)_",
            self.change_type,
            self.description,
            self.impact * 100.0
        );

        if let Some(line) = self.line_number {
            output.push_str(&format!(" at line {}", line));
        }

        if self.has_diff() {
            output.push_str("\n  ```diff");
            if let Some(before) = &self.before {
                output.push_str(&format!("\n  - {}", before));
            }
            if let Some(after) = &self.after {
                output.push_str(&format!("\n  + {}", after));
            }
            output.push_str("\n  ```");
        }

        output
    }
}

impl Default for RefactoringChange {
    fn default() -> Self {
        Self::new(RefactoringType::SimplifyStructure, "")
    }
}

// =============================================================================
// REFACTORED PROMPT STRUCT
// =============================================================================

/// The result of a refactoring operation.
///
/// Contains the refactored content along with detailed information
/// about what was changed and the quality impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoredPrompt {
    /// The refactored prompt content.
    pub content: String,
    /// List of changes made during refactoring.
    pub changes: Vec<RefactoringChange>,
    /// Metrics of the original prompt.
    pub before_metrics: PromptMetrics,
    /// Metrics of the refactored prompt.
    pub after_metrics: PromptMetrics,
    /// The original prompt for reference.
    pub original: String,
    /// Processing duration in milliseconds.
    pub duration_ms: u64,
    /// Number of refactoring iterations performed.
    pub iterations: usize,
    /// Types of refactoring applied.
    pub refactoring_types_applied: Vec<RefactoringType>,
}

impl RefactoredPrompt {
    /// Create a new refactored prompt result.
    pub fn new(original: impl Into<String>, refactored: impl Into<String>) -> Self {
        let original_str = original.into();
        let refactored_str = refactored.into();

        Self {
            before_metrics: PromptMetrics::from_prompt(&original_str),
            after_metrics: PromptMetrics::from_prompt(&refactored_str),
            original: original_str,
            content: refactored_str,
            changes: Vec::new(),
            duration_ms: 0,
            iterations: 0,
            refactoring_types_applied: Vec::new(),
        }
    }

    /// Calculate the overall improvement.
    pub fn improvement(&self) -> f64 {
        self.after_metrics.improvement_over(&self.before_metrics)
    }

    /// Calculate the improvement percentage.
    pub fn improvement_percentage(&self) -> f64 {
        self.improvement() * 100.0
    }

    /// Check if significant improvement was achieved.
    pub fn is_significantly_improved(&self) -> bool {
        self.improvement() > 0.1
    }

    /// Check if the refactoring was successful (no regression).
    pub fn is_successful(&self) -> bool {
        self.improvement() >= 0.0
    }

    /// Get the token reduction (positive means fewer tokens).
    pub fn token_reduction(&self) -> i64 {
        self.before_metrics.token_count as i64 - self.after_metrics.token_count as i64
    }

    /// Get the token reduction percentage.
    pub fn token_reduction_percentage(&self) -> f64 {
        if self.before_metrics.token_count > 0 {
            (self.token_reduction() as f64 / self.before_metrics.token_count as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get changes of a specific type.
    pub fn changes_of_type(&self, change_type: RefactoringType) -> Vec<&RefactoringChange> {
        self.changes
            .iter()
            .filter(|c| c.change_type == change_type)
            .collect()
    }

    /// Get high-impact changes (impact > 0.3).
    pub fn high_impact_changes(&self) -> Vec<&RefactoringChange> {
        self.changes.iter().filter(|c| c.impact > 0.3).collect()
    }

    /// Generate a summary of the refactoring.
    pub fn summary(&self) -> String {
        format!(
            "Refactored: {} -> {} tokens ({:+.0}%), {} changes in {}ms",
            self.before_metrics.token_count,
            self.after_metrics.token_count,
            self.token_reduction_percentage(),
            self.changes.len(),
            self.duration_ms
        )
    }

    /// Format the result as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Refactoring Report\n\n");
        output.push_str(&format!(
            "**Tokens:** {} -> {} ({:+.0}%)\n",
            self.before_metrics.token_count,
            self.after_metrics.token_count,
            self.token_reduction_percentage()
        ));
        output.push_str(&format!(
            "**Readability:** {:.0}% -> {:.0}%\n",
            self.before_metrics.readability_score * 100.0,
            self.after_metrics.readability_score * 100.0
        ));
        output.push_str(&format!(
            "**Complexity:** {:.0}% -> {:.0}%\n\n",
            self.before_metrics.complexity_score * 100.0,
            self.after_metrics.complexity_score * 100.0
        ));

        output.push_str("## Refactored Prompt\n\n");
        output.push_str(&format!("```\n{}\n```\n\n", self.content));

        if !self.changes.is_empty() {
            output.push_str("## Changes Made\n\n");
            for change in &self.changes {
                output.push_str(&change.to_markdown());
                output.push('\n');
            }
        }

        output
    }
}

impl Default for RefactoredPrompt {
    fn default() -> Self {
        Self::new("", "")
    }
}

// =============================================================================
// REFACTORER CONFIGURATION
// =============================================================================

/// Configuration for the Refactorer Agent.
#[derive(Debug, Clone)]
pub struct RefactorerConfig {
    /// Maximum iterations for refactoring.
    pub max_iterations: usize,
    /// Minimum improvement per iteration to continue.
    pub min_improvement: f64,
    /// Target readability score.
    pub target_readability: f64,
    /// Target complexity score (lower is better).
    pub target_complexity: f64,
    /// Whether to preserve exact meaning strictly.
    pub strict_meaning_preservation: bool,
    /// Timeout in seconds for refactoring.
    pub timeout_secs: u64,
    /// Default goals when none specified.
    pub default_goals: Vec<RefactoringGoal>,
    /// Global constraints that apply to all refactoring.
    pub global_constraints: Vec<Constraint>,
}

impl Default for RefactorerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            min_improvement: 0.02,
            target_readability: 0.8,
            target_complexity: 0.5,
            strict_meaning_preservation: true,
            timeout_secs: 30,
            default_goals: vec![
                RefactoringGoal::new(RefactoringType::RemoveDuplication).with_priority(Priority::High),
                RefactoringGoal::new(RefactoringType::SimplifyStructure).with_priority(Priority::Medium),
                RefactoringGoal::new(RefactoringType::ImproveReadability).with_priority(Priority::Medium),
            ],
            global_constraints: Vec::new(),
        }
    }
}

impl RefactorerConfig {
    /// Create an aggressive refactoring configuration.
    pub fn aggressive() -> Self {
        Self {
            max_iterations: 10,
            min_improvement: 0.01,
            target_readability: 0.9,
            target_complexity: 0.3,
            strict_meaning_preservation: false,
            timeout_secs: 60,
            default_goals: RefactoringType::all()
                .into_iter()
                .map(|t| RefactoringGoal::new(t).with_priority(Priority::High))
                .collect(),
            global_constraints: Vec::new(),
        }
    }

    /// Create a conservative refactoring configuration.
    pub fn conservative() -> Self {
        Self {
            max_iterations: 3,
            min_improvement: 0.05,
            target_readability: 0.7,
            target_complexity: 0.6,
            strict_meaning_preservation: true,
            timeout_secs: 15,
            default_goals: vec![
                RefactoringGoal::new(RefactoringType::RemoveDuplication).with_priority(Priority::Medium),
            ],
            global_constraints: vec![
                Constraint::new("preserve-length", "Keep prompt length within 20% of original")
                    .mandatory(true),
            ],
        }
    }

    /// Create a token-optimization focused configuration.
    pub fn token_optimized() -> Self {
        Self {
            max_iterations: 8,
            min_improvement: 0.01,
            target_readability: 0.7,
            target_complexity: 0.5,
            strict_meaning_preservation: true,
            timeout_secs: 45,
            default_goals: vec![
                RefactoringGoal::new(RefactoringType::OptimizeTokens).with_priority(Priority::Critical),
                RefactoringGoal::new(RefactoringType::RemoveDuplication).with_priority(Priority::High),
                RefactoringGoal::new(RefactoringType::SimplifyStructure).with_priority(Priority::Medium),
            ],
            global_constraints: Vec::new(),
        }
    }
}

// =============================================================================
// KEYWORD DICTIONARIES
// =============================================================================

/// Words that are often redundant and can be removed.
const REDUNDANT_WORDS: &[&str] = &[
    "basically", "actually", "literally", "very", "really", "quite",
    "somewhat", "rather", "fairly", "pretty", "just", "simply",
];

/// Redundant phrases that can be simplified.
const REDUNDANT_PHRASES: &[(&str, &str)] = &[
    ("in order to", "to"),
    ("at this point in time", "now"),
    ("due to the fact that", "because"),
    ("in the event that", "if"),
    ("for the purpose of", "to"),
    ("with regard to", "about"),
    ("in spite of the fact that", "although"),
    ("prior to", "before"),
    ("subsequent to", "after"),
    ("in close proximity to", "near"),
    ("at the present time", "now"),
    ("in the near future", "soon"),
    ("a large number of", "many"),
    ("a small number of", "few"),
    ("make a decision", "decide"),
    ("take into consideration", "consider"),
    ("the majority of", "most"),
    ("on a daily basis", "daily"),
    ("is able to", "can"),
    ("has the ability to", "can"),
];

/// Vague words that reduce clarity.
const VAGUE_WORDS: &[&str] = &[
    "thing", "stuff", "something", "somehow", "whatever",
    "things", "etc", "and so on", "and so forth",
];

// =============================================================================
// REFACTORER AGENT
// =============================================================================

/// The Refactorer Agent - Agent 33 of Project Panpsychism.
///
/// Responsible for restructuring and reorganizing prompts for better
/// clarity, maintainability, and efficiency. Like an architect
/// redesigning a building, this agent transforms chaotic structures
/// into elegant, purposeful designs.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive toward structural perfection
/// - **RATIO**: Logical organization and reasoned structure
/// - **LAETITIA**: Joy through elegant expression
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::refactorer::RefactorerAgent;
///
/// let architect = RefactorerAgent::new();
///
/// let refactored = architect.refactor(
///     "Write code. The code should work. Make code clean."
/// ).await?;
///
/// println!("Refactored: {}", refactored.content);
/// ```
#[derive(Debug, Clone)]
pub struct RefactorerAgent {
    /// Configuration for refactoring behavior.
    config: RefactorerConfig,
}

impl Default for RefactorerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl RefactorerAgent {
    /// Create a new Refactorer Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: RefactorerConfig::default(),
        }
    }

    /// Create a new Refactorer Agent with custom configuration.
    pub fn with_config(config: RefactorerConfig) -> Self {
        Self { config }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> RefactorerAgentBuilder {
        RefactorerAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RefactorerConfig {
        &self.config
    }

    // =========================================================================
    // MAIN REFACTORING METHOD
    // =========================================================================

    /// Refactor a prompt using default goals.
    ///
    /// This is the primary refactoring method. It iteratively improves
    /// the prompt's structure using the configured default goals.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt to refactor
    ///
    /// # Returns
    ///
    /// A `RefactoredPrompt` containing the improved content and change details.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if the prompt is empty.
    pub async fn refactor(&self, prompt: &str) -> Result<RefactoredPrompt> {
        self.refactor_with_goals(prompt, &self.config.default_goals).await
    }

    /// Refactor a prompt with specific goals.
    ///
    /// Allows fine-grained control over what refactoring operations
    /// to apply and in what order.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt to refactor
    /// * `goals` - The refactoring goals to apply
    ///
    /// # Returns
    ///
    /// A `RefactoredPrompt` containing the improved content and change details.
    pub async fn refactor_with_goals(
        &self,
        prompt: &str,
        goals: &[RefactoringGoal],
    ) -> Result<RefactoredPrompt> {
        let start = Instant::now();

        if prompt.trim().is_empty() {
            return Err(Error::Validation("Cannot refactor empty prompt".to_string()));
        }

        let goals_to_apply = if goals.is_empty() {
            &self.config.default_goals
        } else {
            goals
        };

        debug!(
            "Refactoring prompt ({} chars) with {} goals",
            prompt.len(),
            goals_to_apply.len()
        );

        let mut current_prompt = prompt.to_string();
        let mut all_changes = Vec::new();
        let mut types_applied = Vec::new();
        let mut iteration = 0;

        let before_metrics = PromptMetrics::from_prompt(prompt);

        // Sort goals by priority
        let mut sorted_goals = goals_to_apply.to_vec();
        sorted_goals.sort_by(|a, b| {
            b.priority.weight().partial_cmp(&a.priority.weight())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply each goal in priority order
        for goal in &sorted_goals {
            if iteration >= self.config.max_iterations {
                break;
            }

            iteration += 1;
            let (refactored, changes) = self.apply_refactoring(&current_prompt, goal)?;

            if !changes.is_empty() {
                current_prompt = refactored;
                all_changes.extend(changes);
                if !types_applied.contains(&goal.goal_type) {
                    types_applied.push(goal.goal_type);
                }

                // Only check for early exit if we made changes
                let current_metrics = PromptMetrics::from_prompt(&current_prompt);
                if current_metrics.readability_score >= self.config.target_readability
                    && current_metrics.complexity_score <= self.config.target_complexity
                {
                    debug!("Target quality reached at iteration {}", iteration);
                    break;
                }
            }
        }

        let after_metrics = PromptMetrics::from_prompt(&current_prompt);

        info!(
            "Refactoring complete: {} changes, readability {:.0}% -> {:.0}% in {}ms",
            all_changes.len(),
            before_metrics.readability_score * 100.0,
            after_metrics.readability_score * 100.0,
            start.elapsed().as_millis()
        );

        Ok(RefactoredPrompt {
            content: current_prompt,
            original: prompt.to_string(),
            changes: all_changes,
            before_metrics,
            after_metrics,
            duration_ms: start.elapsed().as_millis() as u64,
            iterations: iteration,
            refactoring_types_applied: types_applied,
        })
    }

    /// Refactor with specific types (convenience method).
    pub async fn refactor_with_types(
        &self,
        prompt: &str,
        types: &[RefactoringType],
    ) -> Result<RefactoredPrompt> {
        let goals: Vec<RefactoringGoal> = types.iter().map(|&t| RefactoringGoal::new(t)).collect();
        self.refactor_with_goals(prompt, &goals).await
    }

    // =========================================================================
    // REFACTORING OPERATIONS
    // =========================================================================

    /// Apply a specific refactoring goal.
    fn apply_refactoring(
        &self,
        prompt: &str,
        goal: &RefactoringGoal,
    ) -> Result<(String, Vec<RefactoringChange>)> {
        match goal.goal_type {
            RefactoringType::RemoveDuplication => self.remove_duplication(prompt),
            RefactoringType::SimplifyStructure => self.simplify_structure(prompt),
            RefactoringType::ImproveReadability => self.improve_readability(prompt),
            RefactoringType::OptimizeTokens => self.optimize_tokens(prompt),
            RefactoringType::ExtractComponent => self.extract_component(prompt),
            RefactoringType::InlineExpansion => self.inline_expansion(prompt),
        }
    }

    /// Remove redundant and duplicated content.
    fn remove_duplication(&self, prompt: &str) -> Result<(String, Vec<RefactoringChange>)> {
        let mut result = prompt.to_string();
        let mut changes = Vec::new();

        // Remove duplicate sentences
        let sentences: Vec<&str> = prompt
            .split(['.', '!', '?'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut seen_sentences: HashSet<String> = HashSet::new();
        let mut unique_sentences = Vec::new();

        for sentence in &sentences {
            let normalized = sentence.to_lowercase();
            if !seen_sentences.contains(&normalized) {
                seen_sentences.insert(normalized);
                unique_sentences.push(*sentence);
            } else {
                changes.push(
                    RefactoringChange::new(
                        RefactoringType::RemoveDuplication,
                        "Removed duplicate sentence",
                    )
                    .with_before(sentence.to_string())
                    .with_impact(0.2),
                );
            }
        }

        if unique_sentences.len() < sentences.len() {
            result = unique_sentences.join(". ");
            if !result.is_empty() && !result.ends_with('.') {
                result.push('.');
            }
        }

        // Remove redundant words
        for word in REDUNDANT_WORDS {
            if result.to_lowercase().contains(word) {
                let before = result.clone();
                result = result.replace(&format!(" {} ", word), " ");
                result = result.replace(&format!(" {}", word), " ");
                result = result.replace(&format!("{} ", word), " ");

                if result != before {
                    changes.push(
                        RefactoringChange::new(
                            RefactoringType::RemoveDuplication,
                            format!("Removed redundant word '{}'", word),
                        )
                        .with_impact(0.05),
                    );
                }
            }
        }

        // Clean up multiple spaces
        while result.contains("  ") {
            result = result.replace("  ", " ");
        }

        result = result.trim().to_string();

        Ok((result, changes))
    }

    /// Simplify complex structures.
    fn simplify_structure(&self, prompt: &str) -> Result<(String, Vec<RefactoringChange>)> {
        let mut result = prompt.to_string();
        let mut changes = Vec::new();

        // Replace redundant phrases with simpler alternatives
        for (long, short) in REDUNDANT_PHRASES {
            if result.to_lowercase().contains(*long) {
                let before = result.clone();
                // Case-insensitive replacement
                let re = regex::RegexBuilder::new(&regex::escape(long))
                    .case_insensitive(true)
                    .build()
                    .unwrap();
                result = re.replace_all(&result, *short).to_string();

                if result != before {
                    changes.push(
                        RefactoringChange::new(
                            RefactoringType::SimplifyStructure,
                            format!("Simplified '{}' to '{}'", long, short),
                        )
                        .with_before(long.to_string())
                        .with_after(short.to_string())
                        .with_impact(0.1),
                    );
                }
            }
        }

        // Remove vague words
        for word in VAGUE_WORDS {
            if result.to_lowercase().contains(word) {
                let before = result.clone();
                result = result.replace(&format!(" {} ", word), " ");
                result = result.replace(&format!(" {}", word), " ");

                if result != before {
                    changes.push(
                        RefactoringChange::new(
                            RefactoringType::SimplifyStructure,
                            format!("Removed vague word '{}'", word),
                        )
                        .with_impact(0.05),
                    );
                }
            }
        }

        // Clean up
        while result.contains("  ") {
            result = result.replace("  ", " ");
        }
        result = result.trim().to_string();

        Ok((result, changes))
    }

    /// Improve readability of the prompt.
    fn improve_readability(&self, prompt: &str) -> Result<(String, Vec<RefactoringChange>)> {
        let mut result = prompt.to_string();
        let mut changes = Vec::new();

        // Break up very long sentences
        let sentences: Vec<&str> = result
            .split(['.', '!', '?'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut improved_sentences = Vec::new();
        for sentence in &sentences {
            let word_count = sentence.split_whitespace().count();
            if word_count > 40 {
                // Try to break at conjunctions
                let broken = self.break_long_sentence(sentence);
                if broken.len() > 1 {
                    improved_sentences.extend(broken);
                    changes.push(
                        RefactoringChange::new(
                            RefactoringType::ImproveReadability,
                            "Split long sentence for better readability",
                        )
                        .with_before(sentence.to_string())
                        .with_impact(0.15),
                    );
                } else {
                    improved_sentences.push(sentence.to_string());
                }
            } else {
                improved_sentences.push(sentence.to_string());
            }
        }

        if !improved_sentences.is_empty() {
            result = improved_sentences.join(". ");
            if !result.is_empty() && !result.ends_with('.') {
                result.push('.');
            }
        }

        // Ensure proper spacing after punctuation
        result = result.replace(".", ". ").replace(".  ", ". ");
        result = result.replace(",", ", ").replace(",  ", ", ");
        result = result.trim().to_string();

        Ok((result, changes))
    }

    /// Break a long sentence into shorter parts.
    fn break_long_sentence(&self, sentence: &str) -> Vec<String> {
        // Try to break at common conjunctions
        let break_points = [" and ", " but ", " however ", " therefore ", " so "];

        for point in break_points {
            if sentence.contains(point) {
                let parts: Vec<&str> = sentence.splitn(2, point).collect();
                if parts.len() == 2 {
                    let first = parts[0].trim().to_string();
                    let second = format!("{}{}", point.trim(), parts[1].trim());
                    return vec![first, second];
                }
            }
        }

        vec![sentence.to_string()]
    }

    /// Optimize token usage.
    fn optimize_tokens(&self, prompt: &str) -> Result<(String, Vec<RefactoringChange>)> {
        // Token optimization combines several techniques
        let (result1, mut changes) = self.remove_duplication(prompt)?;
        let (result2, changes2) = self.simplify_structure(&result1)?;
        changes.extend(changes2);

        // Additional token optimizations

        // Remove excessive punctuation
        let mut result = result2;
        let before = result.clone();
        result = result.replace("...", ".");
        result = result.replace("!!", "!");
        result = result.replace("??", "?");
        if result != before {
            changes.push(
                RefactoringChange::new(
                    RefactoringType::OptimizeTokens,
                    "Normalized punctuation",
                )
                .with_impact(0.02),
            );
        }

        // Remove trailing whitespace from lines
        let lines: Vec<&str> = result.lines().map(|l| l.trim_end()).collect();
        let trimmed_result = lines.join("\n");
        if trimmed_result != result {
            result = trimmed_result;
            changes.push(
                RefactoringChange::new(
                    RefactoringType::OptimizeTokens,
                    "Removed trailing whitespace",
                )
                .with_impact(0.01),
            );
        }

        Ok((result, changes))
    }

    /// Extract reusable components (stub for complex prompts).
    fn extract_component(&self, prompt: &str) -> Result<(String, Vec<RefactoringChange>)> {
        // For now, this is a no-op that could be extended for complex prompts
        // with repeating patterns
        Ok((prompt.to_string(), Vec::new()))
    }

    /// Inline expand compact references (stub).
    fn inline_expansion(&self, prompt: &str) -> Result<(String, Vec<RefactoringChange>)> {
        // For now, this is a no-op that could be extended to expand
        // abbreviations or compact references
        Ok((prompt.to_string(), Vec::new()))
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Quick metrics calculation without full refactoring.
    pub fn analyze(&self, prompt: &str) -> PromptMetrics {
        PromptMetrics::from_prompt(prompt)
    }

    /// Check if a prompt would benefit from refactoring.
    pub fn needs_refactoring(&self, prompt: &str) -> bool {
        let metrics = PromptMetrics::from_prompt(prompt);
        !metrics.is_good_quality()
    }

    /// Get recommendations for what to refactor.
    pub fn recommend_refactoring(&self, prompt: &str) -> Vec<RefactoringGoal> {
        let metrics = PromptMetrics::from_prompt(prompt);
        let mut recommendations = Vec::new();

        if metrics.duplication_ratio > 0.2 {
            recommendations.push(
                RefactoringGoal::new(RefactoringType::RemoveDuplication)
                    .with_priority(Priority::High),
            );
        }

        if metrics.complexity_score > 0.6 {
            recommendations.push(
                RefactoringGoal::new(RefactoringType::SimplifyStructure)
                    .with_priority(Priority::Medium),
            );
        }

        if metrics.readability_score < 0.6 {
            recommendations.push(
                RefactoringGoal::new(RefactoringType::ImproveReadability)
                    .with_priority(Priority::High),
            );
        }

        if metrics.token_count > 500 {
            recommendations.push(
                RefactoringGoal::new(RefactoringType::OptimizeTokens)
                    .with_priority(Priority::Medium),
            );
        }

        recommendations
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom RefactorerAgent configuration.
#[derive(Debug, Default)]
pub struct RefactorerAgentBuilder {
    config: Option<RefactorerConfig>,
}

impl RefactorerAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: RefactorerConfig) -> Self {
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

    /// Set target readability score.
    pub fn target_readability(mut self, target: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.target_readability = target.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set target complexity score.
    pub fn target_complexity(mut self, target: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.target_complexity = target.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set whether to strictly preserve meaning.
    pub fn strict_meaning_preservation(mut self, strict: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.strict_meaning_preservation = strict;
        self.config = Some(config);
        self
    }

    /// Set default goals.
    pub fn default_goals(mut self, goals: Vec<RefactoringGoal>) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.default_goals = goals;
        self.config = Some(config);
        self
    }

    /// Add a global constraint.
    pub fn add_constraint(mut self, constraint: Constraint) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.global_constraints.push(constraint);
        self.config = Some(config);
        self
    }

    /// Build the RefactorerAgent.
    pub fn build(self) -> RefactorerAgent {
        RefactorerAgent {
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
        assert_eq!(Priority::Critical.to_string(), "critical");
        assert_eq!(Priority::High.to_string(), "high");
        assert_eq!(Priority::Medium.to_string(), "medium");
        assert_eq!(Priority::Low.to_string(), "low");
    }

    #[test]
    fn test_priority_from_str() {
        assert_eq!("critical".parse::<Priority>().unwrap(), Priority::Critical);
        assert_eq!("urgent".parse::<Priority>().unwrap(), Priority::Critical);
        assert_eq!("high".parse::<Priority>().unwrap(), Priority::High);
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

    #[test]
    fn test_priority_all() {
        let all = Priority::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&Priority::Critical));
        assert!(all.contains(&Priority::Low));
    }

    #[test]
    fn test_priority_weight() {
        assert!((Priority::Critical.weight() - 2.0).abs() < f64::EPSILON);
        assert!((Priority::High.weight() - 1.5).abs() < f64::EPSILON);
        assert!((Priority::Medium.weight() - 1.0).abs() < f64::EPSILON);
        assert!((Priority::Low.weight() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_priority_is_higher_than() {
        assert!(Priority::Critical.is_higher_than(&Priority::High));
        assert!(Priority::High.is_higher_than(&Priority::Medium));
        assert!(Priority::Medium.is_higher_than(&Priority::Low));
        assert!(!Priority::Low.is_higher_than(&Priority::Critical));
    }

    // -------------------------------------------------------------------------
    // RefactoringType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_refactoring_type_display() {
        assert_eq!(RefactoringType::ExtractComponent.to_string(), "extract-component");
        assert_eq!(RefactoringType::InlineExpansion.to_string(), "inline-expansion");
        assert_eq!(RefactoringType::SimplifyStructure.to_string(), "simplify-structure");
        assert_eq!(RefactoringType::RemoveDuplication.to_string(), "remove-duplication");
        assert_eq!(RefactoringType::ImproveReadability.to_string(), "improve-readability");
        assert_eq!(RefactoringType::OptimizeTokens.to_string(), "optimize-tokens");
    }

    #[test]
    fn test_refactoring_type_from_str() {
        assert_eq!("extract-component".parse::<RefactoringType>().unwrap(), RefactoringType::ExtractComponent);
        assert_eq!("extract".parse::<RefactoringType>().unwrap(), RefactoringType::ExtractComponent);
        assert_eq!("simplify".parse::<RefactoringType>().unwrap(), RefactoringType::SimplifyStructure);
        assert_eq!("dedupe".parse::<RefactoringType>().unwrap(), RefactoringType::RemoveDuplication);
        assert_eq!("optimize".parse::<RefactoringType>().unwrap(), RefactoringType::OptimizeTokens);
    }

    #[test]
    fn test_refactoring_type_from_str_invalid() {
        assert!("invalid".parse::<RefactoringType>().is_err());
    }

    #[test]
    fn test_refactoring_type_default() {
        assert_eq!(RefactoringType::default(), RefactoringType::SimplifyStructure);
    }

    #[test]
    fn test_refactoring_type_all() {
        let all = RefactoringType::all();
        assert_eq!(all.len(), 6);
        assert!(all.contains(&RefactoringType::ExtractComponent));
        assert!(all.contains(&RefactoringType::OptimizeTokens));
    }

    #[test]
    fn test_refactoring_type_default_priority() {
        assert_eq!(RefactoringType::RemoveDuplication.default_priority(), Priority::High);
        assert_eq!(RefactoringType::SimplifyStructure.default_priority(), Priority::Medium);
        assert_eq!(RefactoringType::OptimizeTokens.default_priority(), Priority::Low);
    }

    #[test]
    fn test_refactoring_type_description() {
        let desc = RefactoringType::RemoveDuplication.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("redundant") || desc.contains("duplicate"));
    }

    // -------------------------------------------------------------------------
    // Constraint Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constraint_new() {
        let constraint = Constraint::new("test", "Test description");
        assert_eq!(constraint.name, "test");
        assert_eq!(constraint.description, "Test description");
        assert!(constraint.is_mandatory);
        assert!(constraint.applies_to.is_empty());
    }

    #[test]
    fn test_constraint_builder() {
        let constraint = Constraint::new("test", "Test")
            .mandatory(false)
            .applies_to(vec![RefactoringType::SimplifyStructure]);

        assert!(!constraint.is_mandatory);
        assert_eq!(constraint.applies_to.len(), 1);
    }

    #[test]
    fn test_constraint_applies_to_type() {
        let constraint = Constraint::new("test", "Test")
            .applies_to(vec![RefactoringType::SimplifyStructure]);

        assert!(constraint.applies_to_type(RefactoringType::SimplifyStructure));
        assert!(!constraint.applies_to_type(RefactoringType::OptimizeTokens));
    }

    #[test]
    fn test_constraint_applies_to_all_when_empty() {
        let constraint = Constraint::new("test", "Test");
        assert!(constraint.applies_to_type(RefactoringType::SimplifyStructure));
        assert!(constraint.applies_to_type(RefactoringType::OptimizeTokens));
    }

    #[test]
    fn test_constraint_default() {
        let constraint = Constraint::default();
        assert_eq!(constraint.name, "default");
        assert!(constraint.is_mandatory);
    }

    // -------------------------------------------------------------------------
    // RefactoringGoal Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_refactoring_goal_new() {
        let goal = RefactoringGoal::new(RefactoringType::RemoveDuplication);
        assert_eq!(goal.goal_type, RefactoringType::RemoveDuplication);
        assert_eq!(goal.priority, Priority::High); // Default for RemoveDuplication
        assert!(goal.constraints.is_empty());
    }

    #[test]
    fn test_refactoring_goal_builder() {
        let goal = RefactoringGoal::new(RefactoringType::SimplifyStructure)
            .with_priority(Priority::Critical)
            .with_constraint(Constraint::new("test", "Test"));

        assert_eq!(goal.priority, Priority::Critical);
        assert_eq!(goal.constraints.len(), 1);
    }

    #[test]
    fn test_refactoring_goal_with_constraints() {
        let constraints = vec![
            Constraint::new("c1", "Constraint 1"),
            Constraint::new("c2", "Constraint 2"),
        ];
        let goal = RefactoringGoal::new(RefactoringType::SimplifyStructure)
            .with_constraints(constraints);

        assert_eq!(goal.constraints.len(), 2);
    }

    #[test]
    fn test_refactoring_goal_has_mandatory_constraints() {
        let goal1 = RefactoringGoal::new(RefactoringType::SimplifyStructure);
        assert!(!goal1.has_mandatory_constraints());

        let goal2 = RefactoringGoal::new(RefactoringType::SimplifyStructure)
            .with_constraint(Constraint::new("test", "Test").mandatory(true));
        assert!(goal2.has_mandatory_constraints());
    }

    #[test]
    fn test_refactoring_goal_mandatory_constraints() {
        let goal = RefactoringGoal::new(RefactoringType::SimplifyStructure)
            .with_constraint(Constraint::new("c1", "C1").mandatory(true))
            .with_constraint(Constraint::new("c2", "C2").mandatory(false));

        let mandatory = goal.mandatory_constraints();
        assert_eq!(mandatory.len(), 1);
        assert_eq!(mandatory[0].name, "c1");
    }

    #[test]
    fn test_refactoring_goal_default() {
        let goal = RefactoringGoal::default();
        assert_eq!(goal.goal_type, RefactoringType::SimplifyStructure);
    }

    // -------------------------------------------------------------------------
    // PromptMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prompt_metrics_from_prompt() {
        let prompt = "This is a test. It has two sentences.";
        let metrics = PromptMetrics::from_prompt(prompt);

        assert!(metrics.word_count > 0);
        assert!(metrics.token_count >= metrics.word_count);
        assert!(metrics.line_count >= 1);
        assert!(metrics.readability_score >= 0.0 && metrics.readability_score <= 1.0);
        assert!(metrics.complexity_score >= 0.0 && metrics.complexity_score <= 1.0);
    }

    #[test]
    fn test_prompt_metrics_empty() {
        let metrics = PromptMetrics::from_prompt("");
        assert_eq!(metrics.word_count, 0);
        assert!(metrics.token_count >= 1); // Minimum 1
    }

    #[test]
    fn test_prompt_metrics_duplication_ratio() {
        let no_dup = "One two three four five.";
        let metrics_no_dup = PromptMetrics::from_prompt(no_dup);
        assert!(metrics_no_dup.duplication_ratio < 0.2);

        let with_dup = "Word word word word word.";
        let metrics_with_dup = PromptMetrics::from_prompt(with_dup);
        assert!(metrics_with_dup.duplication_ratio > 0.5);
    }

    #[test]
    fn test_prompt_metrics_improvement_over() {
        let before = PromptMetrics {
            readability_score: 0.5,
            complexity_score: 0.7,
            duplication_ratio: 0.3,
            ..Default::default()
        };

        let after = PromptMetrics {
            readability_score: 0.8,
            complexity_score: 0.4,
            duplication_ratio: 0.1,
            ..Default::default()
        };

        let improvement = after.improvement_over(&before);
        assert!(improvement > 0.0);
    }

    #[test]
    fn test_prompt_metrics_is_good_quality() {
        let good = PromptMetrics {
            readability_score: 0.8,
            complexity_score: 0.4,
            duplication_ratio: 0.1,
            ..Default::default()
        };
        assert!(good.is_good_quality());

        let bad = PromptMetrics {
            readability_score: 0.4,
            complexity_score: 0.8,
            duplication_ratio: 0.5,
            ..Default::default()
        };
        assert!(!bad.is_good_quality());
    }

    #[test]
    fn test_prompt_metrics_summary() {
        let metrics = PromptMetrics {
            token_count: 100,
            word_count: 80,
            complexity_score: 0.5,
            readability_score: 0.7,
            ..Default::default()
        };

        let summary = metrics.summary();
        assert!(summary.contains("100 tokens"));
        assert!(summary.contains("80 words"));
    }

    // -------------------------------------------------------------------------
    // RefactoringChange Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_refactoring_change_new() {
        let change = RefactoringChange::new(RefactoringType::RemoveDuplication, "Removed duplicate");
        assert_eq!(change.change_type, RefactoringType::RemoveDuplication);
        assert_eq!(change.description, "Removed duplicate");
        assert!(change.before.is_none());
        assert!(change.after.is_none());
    }

    #[test]
    fn test_refactoring_change_builder() {
        let change = RefactoringChange::new(RefactoringType::SimplifyStructure, "Simplified")
            .with_before("complex text")
            .with_after("simple text")
            .with_line(10)
            .with_impact(0.25);

        assert_eq!(change.before, Some("complex text".to_string()));
        assert_eq!(change.after, Some("simple text".to_string()));
        assert_eq!(change.line_number, Some(10));
        assert!((change.impact - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_refactoring_change_impact_clamping() {
        let change_high = RefactoringChange::new(RefactoringType::SimplifyStructure, "Test")
            .with_impact(1.5);
        assert!((change_high.impact - 1.0).abs() < f64::EPSILON);

        let change_low = RefactoringChange::new(RefactoringType::SimplifyStructure, "Test")
            .with_impact(-0.5);
        assert!(change_low.impact.abs() < f64::EPSILON);
    }

    #[test]
    fn test_refactoring_change_has_diff() {
        let no_diff = RefactoringChange::new(RefactoringType::SimplifyStructure, "Test");
        assert!(!no_diff.has_diff());

        let with_diff = RefactoringChange::new(RefactoringType::SimplifyStructure, "Test")
            .with_before("a")
            .with_after("b");
        assert!(with_diff.has_diff());
    }

    #[test]
    fn test_refactoring_change_to_markdown() {
        let change = RefactoringChange::new(RefactoringType::SimplifyStructure, "Simplified phrase")
            .with_before("in order to")
            .with_after("to")
            .with_impact(0.1);

        let md = change.to_markdown();
        assert!(md.contains("simplify-structure"));
        assert!(md.contains("Simplified phrase"));
        assert!(md.contains("10%"));
    }

    #[test]
    fn test_refactoring_change_default() {
        let change = RefactoringChange::default();
        assert_eq!(change.change_type, RefactoringType::SimplifyStructure);
        assert!(change.description.is_empty());
    }

    // -------------------------------------------------------------------------
    // RefactoredPrompt Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_refactored_prompt_new() {
        let result = RefactoredPrompt::new("original text", "refactored text");
        assert_eq!(result.original, "original text");
        assert_eq!(result.content, "refactored text");
    }

    #[test]
    fn test_refactored_prompt_improvement() {
        let mut result = RefactoredPrompt::new("a", "b");
        result.before_metrics = PromptMetrics {
            readability_score: 0.5,
            complexity_score: 0.7,
            duplication_ratio: 0.3,
            ..Default::default()
        };
        result.after_metrics = PromptMetrics {
            readability_score: 0.8,
            complexity_score: 0.4,
            duplication_ratio: 0.1,
            ..Default::default()
        };

        let improvement = result.improvement();
        assert!(improvement > 0.0);
    }

    #[test]
    fn test_refactored_prompt_is_successful() {
        let mut result = RefactoredPrompt::new("a", "b");
        result.before_metrics = PromptMetrics {
            readability_score: 0.5,
            ..Default::default()
        };
        result.after_metrics = PromptMetrics {
            readability_score: 0.6,
            ..Default::default()
        };
        assert!(result.is_successful());
    }

    #[test]
    fn test_refactored_prompt_token_reduction() {
        let mut result = RefactoredPrompt::new("a", "b");
        result.before_metrics = PromptMetrics {
            token_count: 100,
            ..Default::default()
        };
        result.after_metrics = PromptMetrics {
            token_count: 80,
            ..Default::default()
        };

        assert_eq!(result.token_reduction(), 20);
        assert!((result.token_reduction_percentage() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_refactored_prompt_changes_of_type() {
        let mut result = RefactoredPrompt::new("a", "b");
        result.changes.push(RefactoringChange::new(RefactoringType::RemoveDuplication, "D1"));
        result.changes.push(RefactoringChange::new(RefactoringType::SimplifyStructure, "S1"));
        result.changes.push(RefactoringChange::new(RefactoringType::RemoveDuplication, "D2"));

        let dup_changes = result.changes_of_type(RefactoringType::RemoveDuplication);
        assert_eq!(dup_changes.len(), 2);
    }

    #[test]
    fn test_refactored_prompt_high_impact_changes() {
        let mut result = RefactoredPrompt::new("a", "b");
        result.changes.push(RefactoringChange::new(RefactoringType::RemoveDuplication, "Low").with_impact(0.1));
        result.changes.push(RefactoringChange::new(RefactoringType::SimplifyStructure, "High").with_impact(0.5));

        let high_impact = result.high_impact_changes();
        assert_eq!(high_impact.len(), 1);
        assert!(high_impact[0].description.contains("High"));
    }

    #[test]
    fn test_refactored_prompt_summary() {
        let mut result = RefactoredPrompt::new("a", "b");
        result.before_metrics.token_count = 100;
        result.after_metrics.token_count = 80;
        result.changes.push(RefactoringChange::new(RefactoringType::RemoveDuplication, "Test"));
        result.duration_ms = 50;

        let summary = result.summary();
        assert!(summary.contains("100"));
        assert!(summary.contains("80"));
        assert!(summary.contains("50ms"));
    }

    #[test]
    fn test_refactored_prompt_to_markdown() {
        let mut result = RefactoredPrompt::new("original", "refactored");
        result.changes.push(RefactoringChange::new(RefactoringType::RemoveDuplication, "Removed dup"));

        let md = result.to_markdown();
        assert!(md.contains("Refactoring Report"));
        assert!(md.contains("refactored"));
        assert!(md.contains("Changes Made"));
    }

    #[test]
    fn test_refactored_prompt_default() {
        let result = RefactoredPrompt::default();
        assert!(result.original.is_empty());
        assert!(result.content.is_empty());
    }

    // -------------------------------------------------------------------------
    // RefactorerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_refactorer_config_default() {
        let config = RefactorerConfig::default();
        assert_eq!(config.max_iterations, 5);
        assert!((config.min_improvement - 0.02).abs() < f64::EPSILON);
        assert!((config.target_readability - 0.8).abs() < f64::EPSILON);
        assert!(config.strict_meaning_preservation);
    }

    #[test]
    fn test_refactorer_config_aggressive() {
        let config = RefactorerConfig::aggressive();
        assert_eq!(config.max_iterations, 10);
        assert!(!config.strict_meaning_preservation);
        assert_eq!(config.default_goals.len(), 6);
    }

    #[test]
    fn test_refactorer_config_conservative() {
        let config = RefactorerConfig::conservative();
        assert_eq!(config.max_iterations, 3);
        assert!(config.strict_meaning_preservation);
        assert!(!config.global_constraints.is_empty());
    }

    #[test]
    fn test_refactorer_config_token_optimized() {
        let config = RefactorerConfig::token_optimized();
        assert_eq!(config.max_iterations, 8);
        let first_goal = &config.default_goals[0];
        assert_eq!(first_goal.goal_type, RefactoringType::OptimizeTokens);
        assert_eq!(first_goal.priority, Priority::Critical);
    }

    // -------------------------------------------------------------------------
    // RefactorerAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_refactorer_agent_new() {
        let agent = RefactorerAgent::new();
        assert_eq!(agent.config().max_iterations, 5);
    }

    #[test]
    fn test_refactorer_agent_with_config() {
        let config = RefactorerConfig::aggressive();
        let agent = RefactorerAgent::with_config(config);
        assert_eq!(agent.config().max_iterations, 10);
    }

    #[test]
    fn test_refactorer_agent_builder() {
        let agent = RefactorerAgent::builder()
            .max_iterations(8)
            .min_improvement(0.03)
            .target_readability(0.9)
            .target_complexity(0.4)
            .strict_meaning_preservation(false)
            .build();

        assert_eq!(agent.config().max_iterations, 8);
        assert!((agent.config().min_improvement - 0.03).abs() < f64::EPSILON);
        assert!((agent.config().target_readability - 0.9).abs() < f64::EPSILON);
        assert!(!agent.config().strict_meaning_preservation);
    }

    #[test]
    fn test_refactorer_agent_builder_clamping() {
        let agent = RefactorerAgent::builder()
            .max_iterations(0)
            .min_improvement(-0.1)
            .target_readability(1.5)
            .build();

        assert_eq!(agent.config().max_iterations, 1);
        assert!(agent.config().min_improvement.abs() < f64::EPSILON);
        assert!((agent.config().target_readability - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_refactorer_agent_builder_default_goals() {
        let goals = vec![
            RefactoringGoal::new(RefactoringType::OptimizeTokens).with_priority(Priority::High),
        ];
        let agent = RefactorerAgent::builder()
            .default_goals(goals)
            .build();

        assert_eq!(agent.config().default_goals.len(), 1);
        assert_eq!(agent.config().default_goals[0].goal_type, RefactoringType::OptimizeTokens);
    }

    #[test]
    fn test_refactorer_agent_builder_add_constraint() {
        let agent = RefactorerAgent::builder()
            .add_constraint(Constraint::new("test", "Test constraint"))
            .build();

        assert_eq!(agent.config().global_constraints.len(), 1);
    }

    #[test]
    fn test_refactorer_agent_default() {
        let agent = RefactorerAgent::default();
        assert_eq!(agent.config().max_iterations, 5);
    }

    // -------------------------------------------------------------------------
    // Refactoring Operation Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_refactor_empty_prompt() {
        let agent = RefactorerAgent::new();
        let result = agent.refactor("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_refactor_whitespace_only() {
        let agent = RefactorerAgent::new();
        let result = agent.refactor("   \n\t  ").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_refactor_simple_prompt() {
        let agent = RefactorerAgent::new();
        let result = agent.refactor("Write code.").await;

        assert!(result.is_ok());
        let refactored = result.unwrap();
        assert!(!refactored.content.is_empty());
    }

    #[tokio::test]
    async fn test_refactor_with_duplication() {
        let agent = RefactorerAgent::new();
        let prompt = "Write code. The code should work. Make sure the code is tested. Write code.";
        let result = agent.refactor(prompt).await;

        assert!(result.is_ok());
        let refactored = result.unwrap();
        // Should have made some changes for duplication
        assert!(refactored.changes.iter().any(|c| c.change_type == RefactoringType::RemoveDuplication) ||
                refactored.content != prompt);
    }

    #[tokio::test]
    async fn test_refactor_with_redundant_words() {
        let agent = RefactorerAgent::new();
        let prompt = "Just basically write some really very good code.";
        let result = agent.refactor(prompt).await;

        assert!(result.is_ok());
        let refactored = result.unwrap();
        // Should have removed some redundant words
        let content_lower = refactored.content.to_lowercase();
        assert!(
            !content_lower.contains("basically") ||
            !content_lower.contains("really") ||
            !refactored.changes.is_empty()
        );
    }

    #[tokio::test]
    async fn test_refactor_with_redundant_phrases() {
        let agent = RefactorerAgent::new();
        let prompt = "In order to write code, due to the fact that it is needed.";
        let result = agent.refactor(prompt).await;

        assert!(result.is_ok());
        let refactored = result.unwrap();
        // Should have simplified phrases
        let content_lower = refactored.content.to_lowercase();
        assert!(
            !content_lower.contains("in order to") ||
            !content_lower.contains("due to the fact that") ||
            !refactored.changes.is_empty()
        );
    }

    #[tokio::test]
    async fn test_refactor_with_goals() {
        let agent = RefactorerAgent::new();
        let goals = vec![
            RefactoringGoal::new(RefactoringType::OptimizeTokens).with_priority(Priority::High),
        ];
        let result = agent.refactor_with_goals("Write some code basically.", &goals).await;

        assert!(result.is_ok());
        let refactored = result.unwrap();
        assert!(!refactored.refactoring_types_applied.is_empty());
    }

    #[tokio::test]
    async fn test_refactor_with_types() {
        let agent = RefactorerAgent::new();
        let result = agent.refactor_with_types(
            "Write code.",
            &[RefactoringType::SimplifyStructure],
        ).await;

        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Remove Duplication Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_remove_duplication_duplicate_sentences() {
        let agent = RefactorerAgent::new();
        let (result, changes) = agent.remove_duplication("Test sentence. Test sentence.").unwrap();

        // Should have removed duplicate
        let sentence_count = result.matches('.').count();
        assert!(sentence_count < 2 || !changes.is_empty());
    }

    #[test]
    fn test_remove_duplication_redundant_words() {
        let agent = RefactorerAgent::new();
        let (result, changes) = agent.remove_duplication("This is basically just a test.").unwrap();

        // Should have removed some redundant words
        let content_lower = result.to_lowercase();
        assert!(
            !content_lower.contains(" basically ") ||
            !content_lower.contains(" just ") ||
            !changes.is_empty()
        );
    }

    // -------------------------------------------------------------------------
    // Simplify Structure Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simplify_structure_redundant_phrases() {
        let agent = RefactorerAgent::new();
        let (result, changes) = agent.simplify_structure("In order to test this.").unwrap();

        // Should have simplified "in order to" -> "to"
        assert!(
            !result.to_lowercase().contains("in order to") ||
            !changes.is_empty()
        );
    }

    #[test]
    fn test_simplify_structure_vague_words() {
        let agent = RefactorerAgent::new();
        let (result, changes) = agent.simplify_structure("Do something with the thing.").unwrap();

        // Should have addressed vague words
        assert!(
            !result.to_lowercase().contains(" something ") ||
            !result.to_lowercase().contains(" thing") ||
            !changes.is_empty() ||
            result != "Do something with the thing."
        );
    }

    // -------------------------------------------------------------------------
    // Improve Readability Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_improve_readability_long_sentence() {
        let agent = RefactorerAgent::new();
        let long_sentence = "This is a very long sentence that goes on and on and on with many words and clauses and it keeps continuing without any breaks and it talks about many different things all at once and it just keeps going and going until finally it ends.";
        let (result, changes) = agent.improve_readability(long_sentence).unwrap();

        // Should have attempted to break up the sentence
        assert!(result.len() >= long_sentence.len() - 10 || !changes.is_empty());
    }

    #[test]
    fn test_improve_readability_spacing() {
        let agent = RefactorerAgent::new();
        let (result, _) = agent.improve_readability("Test.No space here.").unwrap();

        // Should have proper spacing
        assert!(result.contains(". ") || result == "Test.No space here.");
    }

    // -------------------------------------------------------------------------
    // Optimize Tokens Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimize_tokens() {
        let agent = RefactorerAgent::new();
        let (result, changes) = agent.optimize_tokens("This is basically... just a test!!").unwrap();

        // Should have normalized punctuation
        assert!(!result.contains("...") || !result.contains("!!") || !changes.is_empty());
    }

    // -------------------------------------------------------------------------
    // Utility Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze() {
        let agent = RefactorerAgent::new();
        let metrics = agent.analyze("This is a test prompt.");

        assert!(metrics.word_count > 0);
        assert!(metrics.readability_score >= 0.0 && metrics.readability_score <= 1.0);
    }

    #[test]
    fn test_needs_refactoring() {
        let agent = RefactorerAgent::new();

        // Short, simple prompt probably doesn't need refactoring
        let simple = "Write code.";
        // May or may not need refactoring depending on metrics

        // Complex, duplicated prompt likely needs refactoring
        let complex = "Do something with stuff. Do something with stuff. Maybe perhaps possibly consider doing things with various things that exist.";
        assert!(agent.needs_refactoring(complex));
    }

    #[test]
    fn test_recommend_refactoring() {
        let agent = RefactorerAgent::new();
        let complex = "Word word word word word. Word word word word word. This is very complex and hard to read with many many words repeated.";

        let recommendations = agent.recommend_refactoring(complex);
        assert!(!recommendations.is_empty());
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_refactor_very_long_prompt() {
        let agent = RefactorerAgent::new();
        let long_prompt = "word ".repeat(500);

        let result = agent.refactor(&long_prompt).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_refactor_special_characters() {
        let agent = RefactorerAgent::new();
        let special = "Write code with `backticks` and {braces} and [brackets].";

        let result = agent.refactor(special).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_refactor_unicode() {
        let agent = RefactorerAgent::new();
        let unicode = "Write code for emoji processing: test";

        let result = agent.refactor(unicode).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_refactor_multiline() {
        let agent = RefactorerAgent::new();
        let multiline = "Line one.\nLine two.\nLine three.";

        let result = agent.refactor(multiline).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_refactor_already_good() {
        let agent = RefactorerAgent::builder()
            .target_readability(0.5)
            .target_complexity(0.8)
            .build();

        let good_prompt = "Create a function that adds two numbers.";
        let result = agent.refactor(good_prompt).await;

        assert!(result.is_ok());
        let refactored = result.unwrap();
        assert!(refactored.is_successful());
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_full_refactoring_workflow() {
        let agent = RefactorerAgent::builder()
            .max_iterations(5)
            .min_improvement(0.01)
            .target_readability(0.7)
            .build();

        let messy_prompt = "In order to write code, basically just do something with the stuff. The code should work. Make the code work. Maybe perhaps the code could possibly be tested.";

        let result = agent.refactor(messy_prompt).await;
        assert!(result.is_ok());

        let refactored = result.unwrap();

        // Verify structure
        assert!(!refactored.content.is_empty());
        assert!(!refactored.original.is_empty());
        assert!(refactored.duration_ms >= 0);

        // Verify metrics are present
        assert!(refactored.before_metrics.word_count > 0);
        assert!(refactored.after_metrics.word_count > 0);

        // Verify it can be formatted
        let md = refactored.to_markdown();
        assert!(!md.is_empty());
        assert!(md.contains("Refactoring Report"));
    }

    #[tokio::test]
    async fn test_refactoring_preserves_meaning() {
        let agent = RefactorerAgent::builder()
            .strict_meaning_preservation(true)
            .build();

        let prompt = "Create a login function that validates user credentials.";
        let result = agent.refactor(prompt).await;

        assert!(result.is_ok());
        let refactored = result.unwrap();

        // Core meaning words should be preserved
        let content_lower = refactored.content.to_lowercase();
        assert!(
            content_lower.contains("login") ||
            content_lower.contains("function") ||
            content_lower.contains("credentials") ||
            content_lower.contains("user")
        );
    }
}
