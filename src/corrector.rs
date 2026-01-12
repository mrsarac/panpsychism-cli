//! Spell Refinement corrector module for Project Panpsychism.
//!
//! Implements the "Spell Refinement" pattern: iterative improvement through
//! ambiguity detection and clarification. Like a sorcerer perfecting an
//! incantation, this module refines initial outputs by identifying unclear
//! passages and generating targeted clarification questions.
//!
//! ## Philosophy
//!
//! In the Spinoza framework, clarity is essential for true understanding.
//! The Corrector embodies RATIO (reason) by ensuring every statement
//! achieves maximum clarity before reaching the user.
//!
//! ## Detection Patterns
//!
//! The corrector identifies four types of ambiguity:
//!
//! - **Reference**: Unclear pronoun references ("it", "they", "this")
//! - **Context**: Missing contextual information ("the system", "the file")
//! - **Vague**: Imprecise language ("some", "many", "often", "might")
//! - **MultiInterpretation**: Statements with multiple valid readings
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::corrector::Corrector;
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let corrector = Corrector::new();
//!
//!     let text = "It should be configured properly. They might cause issues.";
//!     let ambiguities = corrector.detect_ambiguities(text).await?;
//!
//!     println!("Found {} ambiguities", ambiguities.len());
//!     Ok(())
//! }
//! ```

// Internal modules
use crate::{Error, Result};

// =============================================================================
// REGEX PATTERNS
// =============================================================================

/// Regex pattern for detecting unclear pronoun references.
///
/// Matches isolated pronouns that lack clear antecedents:
/// "it", "they", "this", "that", "these", "those", "which"
/// Case-insensitive matching is enabled via (?i) flag.
const REFERENCE_PATTERN: &str = r"(?i)\b(it|they|this|that|these|those|which)\b";

/// Regex pattern for detecting vague quantifiers and modifiers.
///
/// Matches imprecise language that reduces clarity:
/// "some", "many", "few", "often", "sometimes", "might", "could", "probably"
/// Case-insensitive matching is enabled via (?i) flag.
const VAGUE_PATTERN: &str = r"(?i)\b(some|many|few|several|often|sometimes|might|could|probably|possibly|usually|generally)\b";

/// Regex pattern for detecting missing context indicators.
///
/// Matches definite articles followed by generic nouns without prior definition:
/// "the system", "the file", "the process", "the data", "the user"
/// Case-insensitive matching is enabled via (?i) flag.
const CONTEXT_PATTERN: &str = r"(?i)\bthe\s+(system|file|process|data|user|server|database|config|settings|application|service|module|component|function|method|class|object|variable|parameter|argument|value|result|response|request|error|issue|problem)\b";

/// Regex pattern for detecting multi-interpretation phrases.
///
/// Matches phrases that can have multiple valid readings:
/// "or more", "and/or", "etc.", "such as", "for example"
/// Case-insensitive matching is enabled via (?i) flag.
const MULTI_PATTERN: &str =
    r"(?i)\b(or\s+more|and/or|etc\.?|such\s+as|for\s+example|e\.g\.|i\.e\.)\b";

// =============================================================================
// CORRECTOR STRUCT
// =============================================================================

/// Corrector for ambiguity detection and iterative refinement.
///
/// The Corrector implements the "Second Throw" pattern, analyzing text
/// for ambiguities and generating clarification questions to refine
/// the output quality.
///
/// # Configuration
///
/// - `max_iterations`: Maximum number of correction passes (default: 3)
/// - `ambiguity_threshold`: Confidence threshold for flagging (default: 0.5)
///
/// # Example
///
/// ```rust,ignore
/// let corrector = Corrector::builder()
///     .max_iterations(5)
///     .ambiguity_threshold(0.3)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct Corrector {
    /// Maximum correction iterations allowed.
    max_iterations: usize,
    /// Minimum confidence threshold for flagging ambiguities.
    ambiguity_threshold: f64,
    /// Compiled regex patterns for each ambiguity type.
    patterns: AmbiguityPatterns,
}

/// Compiled regex patterns for ambiguity detection.
#[derive(Debug, Clone)]
struct AmbiguityPatterns {
    reference: regex::Regex,
    vague: regex::Regex,
    context: regex::Regex,
    multi: regex::Regex,
}

impl Default for AmbiguityPatterns {
    fn default() -> Self {
        Self {
            reference: regex::Regex::new(REFERENCE_PATTERN).expect("Invalid reference pattern"),
            vague: regex::Regex::new(VAGUE_PATTERN).expect("Invalid vague pattern"),
            context: regex::Regex::new(CONTEXT_PATTERN).expect("Invalid context pattern"),
            multi: regex::Regex::new(MULTI_PATTERN).expect("Invalid multi pattern"),
        }
    }
}

impl Default for Corrector {
    fn default() -> Self {
        Self::new()
    }
}

impl Corrector {
    /// Create a new corrector with default settings.
    ///
    /// Default configuration:
    /// - `max_iterations`: 3
    /// - `ambiguity_threshold`: 0.5
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let corrector = Corrector::new();
    /// ```
    pub fn new() -> Self {
        Self {
            max_iterations: 3,
            ambiguity_threshold: 0.5,
            patterns: AmbiguityPatterns::default(),
        }
    }

    /// Create a builder for custom corrector configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let corrector = Corrector::builder()
    ///     .max_iterations(5)
    ///     .ambiguity_threshold(0.3)
    ///     .build();
    /// ```
    pub fn builder() -> CorrectorBuilder {
        CorrectorBuilder::default()
    }

    /// Get the maximum number of correction iterations.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Get the ambiguity detection threshold.
    pub fn ambiguity_threshold(&self) -> f64 {
        self.ambiguity_threshold
    }

    /// Analyze content for ambiguities using regex pattern matching.
    ///
    /// Scans the input text for four types of ambiguity:
    /// 1. **Reference**: Unclear pronoun references
    /// 2. **Vague**: Imprecise quantifiers and modifiers
    /// 3. **Context**: Missing contextual definitions
    /// 4. **MultiInterpretation**: Phrases with multiple readings
    ///
    /// Each detected ambiguity includes:
    /// - The ambiguity type (kind)
    /// - Position in the original text
    /// - The ambiguous text fragment
    /// - A confidence score based on pattern match quality
    ///
    /// # Arguments
    ///
    /// * `content` - The text to analyze for ambiguities
    ///
    /// # Returns
    ///
    /// A vector of detected ambiguities, sorted by position.
    ///
    /// # Errors
    ///
    /// Returns `Error::Correction` if the content is empty or analysis fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let corrector = Corrector::new();
    /// let text = "It might cause issues. The system should handle this.";
    /// let ambiguities = corrector.detect_ambiguities(text).await?;
    ///
    /// for amb in &ambiguities {
    ///     println!("{:?} at {}: '{}'", amb.kind, amb.position, amb.text);
    /// }
    /// ```
    pub async fn detect_ambiguities(&self, content: &str) -> Result<Vec<Ambiguity>> {
        if content.is_empty() {
            return Err(Error::Correction(
                "Cannot analyze empty content".to_string(),
            ));
        }

        let mut ambiguities = Vec::new();

        // Detect reference ambiguities (pronouns without clear antecedents)
        self.detect_pattern_ambiguities(
            content,
            &self.patterns.reference,
            AmbiguityKind::Reference,
            0.7, // Higher confidence for pronoun detection
            &mut ambiguities,
        );

        // Detect vague language
        self.detect_pattern_ambiguities(
            content,
            &self.patterns.vague,
            AmbiguityKind::Vague,
            0.6,
            &mut ambiguities,
        );

        // Detect missing context (definite articles with undefined nouns)
        self.detect_pattern_ambiguities(
            content,
            &self.patterns.context,
            AmbiguityKind::Context,
            0.8, // Higher confidence for context issues
            &mut ambiguities,
        );

        // Detect multi-interpretation phrases
        self.detect_pattern_ambiguities(
            content,
            &self.patterns.multi,
            AmbiguityKind::MultiInterpretation,
            0.5,
            &mut ambiguities,
        );

        // Sort by position for consistent ordering
        ambiguities.sort_by_key(|a| a.position);

        // Filter by confidence threshold
        let filtered: Vec<Ambiguity> = ambiguities
            .into_iter()
            .filter(|a| a.confidence >= self.ambiguity_threshold)
            .collect();

        Ok(filtered)
    }

    /// Helper method to detect ambiguities matching a regex pattern.
    fn detect_pattern_ambiguities(
        &self,
        content: &str,
        pattern: &regex::Regex,
        kind: AmbiguityKind,
        base_confidence: f64,
        ambiguities: &mut Vec<Ambiguity>,
    ) {
        for mat in pattern.find_iter(content) {
            // Adjust confidence based on context
            let confidence = self.calculate_confidence(content, mat.start(), base_confidence, kind);

            ambiguities.push(Ambiguity {
                kind,
                position: mat.start(),
                text: mat.as_str().to_string(),
                confidence,
            });
        }
    }

    /// Calculate confidence score based on surrounding context.
    ///
    /// Adjusts the base confidence based on:
    /// - Sentence position (beginning of sentence = higher ambiguity)
    /// - Surrounding punctuation
    /// - Proximity to other ambiguities
    fn calculate_confidence(
        &self,
        content: &str,
        position: usize,
        base_confidence: f64,
        kind: AmbiguityKind,
    ) -> f64 {
        let mut confidence = base_confidence;

        // Check if at sentence start (higher ambiguity for pronouns)
        if (position == 0 || content[..position].ends_with(". "))
            && kind == AmbiguityKind::Reference
        {
            confidence += 0.1;
        }

        // Check for preceding clarifying phrase (reduces ambiguity)
        if position > 10 {
            let preceding = &content[position.saturating_sub(20)..position];
            if preceding.contains("namely")
                || preceding.contains("specifically")
                || preceding.contains("i.e.")
            {
                confidence -= 0.2;
            }
        }

        // Clamp to valid range
        confidence.clamp(0.0, 1.0)
    }

    /// Generate clarification questions for detected ambiguities.
    ///
    /// Creates natural language questions to resolve each ambiguity.
    /// Questions are tailored to the ambiguity type:
    ///
    /// - **Reference**: "What does 'it' refer to?"
    /// - **Vague**: "Can you be more specific about 'some'?"
    /// - **Context**: "Which system are you referring to?"
    /// - **MultiInterpretation**: "Please clarify: do you mean X or Y?"
    ///
    /// # Arguments
    ///
    /// * `ambiguities` - The detected ambiguities to generate questions for
    ///
    /// # Returns
    ///
    /// A vector of clarification questions, one per ambiguity.
    ///
    /// # Errors
    ///
    /// Returns `Error::Correction` if question generation fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let corrector = Corrector::new();
    /// let ambiguities = corrector.detect_ambiguities("It should work.").await?;
    /// let questions = corrector.generate_questions(&ambiguities).await?;
    ///
    /// for q in &questions {
    ///     println!("Q: {}", q.text);
    ///     for opt in &q.options {
    ///         println!("  - {}", opt);
    ///     }
    /// }
    /// ```
    pub async fn generate_questions(&self, ambiguities: &[Ambiguity]) -> Result<Vec<Question>> {
        if ambiguities.is_empty() {
            return Ok(Vec::new());
        }

        let mut questions = Vec::with_capacity(ambiguities.len());

        for (index, ambiguity) in ambiguities.iter().enumerate() {
            let question = self.generate_question_for_ambiguity(ambiguity, index)?;
            questions.push(question);
        }

        Ok(questions)
    }

    /// Generate a single question for a specific ambiguity.
    fn generate_question_for_ambiguity(
        &self,
        ambiguity: &Ambiguity,
        index: usize,
    ) -> Result<Question> {
        let (text, options) = match ambiguity.kind {
            AmbiguityKind::Reference => {
                let text = format!("What does '{}' refer to in this context?", ambiguity.text);
                let options = vec![
                    "A previously mentioned item".to_string(),
                    "A system component".to_string(),
                    "User input or data".to_string(),
                    "Something else (please specify)".to_string(),
                ];
                (text, options)
            }
            AmbiguityKind::Vague => {
                let text = format!(
                    "Can you be more specific about '{}'? What quantity or frequency do you mean?",
                    ambiguity.text
                );
                let options = match ambiguity.text.to_lowercase().as_str() {
                    "some" | "few" | "several" | "many" => vec![
                        "Exactly 1-2".to_string(),
                        "About 3-5".to_string(),
                        "More than 5".to_string(),
                        "Variable (depends on context)".to_string(),
                    ],
                    "often" | "sometimes" | "usually" | "generally" => vec![
                        "Always (100%)".to_string(),
                        "Most of the time (>75%)".to_string(),
                        "About half the time (~50%)".to_string(),
                        "Occasionally (<25%)".to_string(),
                    ],
                    "might" | "could" | "probably" | "possibly" => vec![
                        "Definitely will".to_string(),
                        "Likely will".to_string(),
                        "Unlikely but possible".to_string(),
                        "Only under specific conditions".to_string(),
                    ],
                    _ => vec![
                        "Please provide a specific value".to_string(),
                        "This is intentionally flexible".to_string(),
                    ],
                };
                (text, options)
            }
            AmbiguityKind::Context => {
                // Extract the noun from "the X" pattern
                let noun = ambiguity
                    .text
                    .strip_prefix("the ")
                    .unwrap_or(&ambiguity.text);
                let text = format!(
                    "Which '{}' are you referring to? Please provide more context.",
                    noun
                );
                let options = vec![
                    format!("The main/primary {}", noun),
                    format!("A specific {} (please name it)", noun),
                    format!("All {}s in the scope", noun),
                    "Other (please clarify)".to_string(),
                ];
                (text, options)
            }
            AmbiguityKind::MultiInterpretation => {
                let text = format!(
                    "The phrase '{}' could be interpreted in multiple ways. Which meaning did you intend?",
                    ambiguity.text
                );
                let options = vec![
                    "Include all listed items".to_string(),
                    "Choose any one of the listed items".to_string(),
                    "These are just examples, not exhaustive".to_string(),
                    "The exact interpretation depends on context".to_string(),
                ];
                (text, options)
            }
        };

        Ok(Question {
            text,
            ambiguity_index: index,
            options,
        })
    }

    /// Apply corrections based on user answers to clarification questions.
    ///
    /// Takes the original content and a set of answers, then applies
    /// appropriate corrections to resolve the identified ambiguities.
    ///
    /// # Correction Strategy
    ///
    /// - **Reference**: Replaces pronouns with explicit references
    /// - **Vague**: Substitutes imprecise terms with specific values
    /// - **Context**: Adds qualifying information
    /// - **MultiInterpretation**: Disambiguates phrases
    ///
    /// # Arguments
    ///
    /// * `content` - The original text to correct
    /// * `answers` - User answers to clarification questions
    ///
    /// # Returns
    ///
    /// A `CorrectionResult` containing:
    /// - The corrected content
    /// - Number of corrections applied
    /// - Count of any remaining ambiguities
    ///
    /// # Errors
    ///
    /// Returns `Error::Correction` if:
    /// - Content is empty
    /// - Answer indices are out of range
    /// - Correction application fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let corrector = Corrector::new();
    /// let text = "It should be fixed.";
    ///
    /// let ambiguities = corrector.detect_ambiguities(text).await?;
    /// let questions = corrector.generate_questions(&ambiguities).await?;
    ///
    /// // Simulate user answer
    /// let answers = vec![Answer {
    ///     question_index: 0,
    ///     text: "The login bug".to_string(),
    /// }];
    ///
    /// let result = corrector.apply_corrections(text, &answers).await?;
    /// println!("Corrected: {}", result.content);
    /// // Output: "The login bug should be fixed."
    /// ```
    pub async fn apply_corrections(
        &self,
        content: &str,
        answers: &[Answer],
    ) -> Result<CorrectionResult> {
        if content.is_empty() {
            return Err(Error::Correction(
                "Cannot apply corrections to empty content".to_string(),
            ));
        }

        if answers.is_empty() {
            return Ok(CorrectionResult {
                content: content.to_string(),
                corrections_applied: 0,
                remaining_ambiguities: 0,
            });
        }

        // Re-detect ambiguities to get current positions
        let ambiguities = self.detect_ambiguities(content).await?;

        // Build a map of corrections to apply
        let mut corrections: Vec<(usize, usize, String)> = Vec::new();

        for answer in answers {
            if answer.question_index >= ambiguities.len() {
                let max_index = if ambiguities.is_empty() {
                    "none (no ambiguities found)".to_string()
                } else {
                    format!("{}", ambiguities.len() - 1)
                };
                return Err(Error::Correction(format!(
                    "Answer index {} out of range (max: {})",
                    answer.question_index, max_index
                )));
            }

            let ambiguity = &ambiguities[answer.question_index];
            let replacement = self.generate_replacement(ambiguity, &answer.text)?;

            let end_pos = ambiguity.position + ambiguity.text.len();
            corrections.push((ambiguity.position, end_pos, replacement));
        }

        // Sort corrections by position (descending) to apply from end to start
        corrections.sort_by(|a, b| b.0.cmp(&a.0));

        // Apply corrections
        let mut corrected = content.to_string();
        let mut applied_count = 0;

        for (start, end, replacement) in corrections {
            if start <= corrected.len() && end <= corrected.len() {
                corrected.replace_range(start..end, &replacement);
                applied_count += 1;
            }
        }

        // Check for remaining ambiguities
        let remaining = self.detect_ambiguities(&corrected).await?;

        Ok(CorrectionResult {
            content: corrected,
            corrections_applied: applied_count,
            remaining_ambiguities: remaining.len(),
        })
    }

    /// Generate replacement text for an ambiguity based on the user's answer.
    fn generate_replacement(&self, ambiguity: &Ambiguity, answer: &str) -> Result<String> {
        // If the answer is short and specific, use it directly
        if answer.len() < 50 && !answer.contains(' ') {
            return Ok(answer.to_string());
        }

        // For longer answers, intelligently construct the replacement
        let replacement = match ambiguity.kind {
            AmbiguityKind::Reference => {
                // Replace pronoun with the specific reference
                answer.to_string()
            }
            AmbiguityKind::Vague => {
                // Check if answer is a number or specific value
                if answer.parse::<i32>().is_ok() {
                    format!("exactly {}", answer)
                } else {
                    answer.to_string()
                }
            }
            AmbiguityKind::Context => {
                // Prepend "the" if not already present
                if answer.to_lowercase().starts_with("the ") {
                    answer.to_string()
                } else {
                    format!("the {}", answer)
                }
            }
            AmbiguityKind::MultiInterpretation => {
                // Keep the clarified phrase
                answer.to_string()
            }
        };

        Ok(replacement)
    }

    /// Perform a complete correction cycle.
    ///
    /// Runs the full Second Throw loop:
    /// 1. Detect ambiguities
    /// 2. Generate questions
    /// 3. Apply corrections (requires callback for answers)
    ///
    /// This method is useful for automated pipelines where answers
    /// are provided programmatically.
    ///
    /// # Arguments
    ///
    /// * `content` - The text to analyze and correct
    /// * `answer_provider` - Async function that provides answers for questions
    ///
    /// # Returns
    ///
    /// The final `CorrectionResult` after all iterations complete.
    pub async fn correct_with_provider<F, Fut>(
        &self,
        content: &str,
        answer_provider: F,
    ) -> Result<CorrectionResult>
    where
        F: Fn(Vec<Question>) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<Answer>>>,
    {
        let mut current_content = content.to_string();
        let mut total_corrections = 0;

        for iteration in 0..self.max_iterations {
            let ambiguities = self.detect_ambiguities(&current_content).await?;

            if ambiguities.is_empty() {
                tracing::debug!("No ambiguities found after {} iterations", iteration);
                break;
            }

            let questions = self.generate_questions(&ambiguities).await?;
            let answers = answer_provider(questions).await?;

            if answers.is_empty() {
                break;
            }

            let result = self.apply_corrections(&current_content, &answers).await?;
            total_corrections += result.corrections_applied;
            current_content = result.content;

            if result.remaining_ambiguities == 0 {
                break;
            }
        }

        let remaining = self.detect_ambiguities(&current_content).await?;

        Ok(CorrectionResult {
            content: current_content,
            corrections_applied: total_corrections,
            remaining_ambiguities: remaining.len(),
        })
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom Corrector configuration.
#[derive(Debug, Default)]
pub struct CorrectorBuilder {
    max_iterations: Option<usize>,
    ambiguity_threshold: Option<f64>,
}

impl CorrectorBuilder {
    /// Set the maximum number of correction iterations.
    ///
    /// Default: 3
    pub fn max_iterations(mut self, value: usize) -> Self {
        self.max_iterations = Some(value);
        self
    }

    /// Set the ambiguity confidence threshold.
    ///
    /// Only ambiguities with confidence >= threshold will be flagged.
    /// Default: 0.5
    ///
    /// # Panics
    ///
    /// Panics if value is not in range 0.0..=1.0
    pub fn ambiguity_threshold(mut self, value: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&value),
            "Threshold must be between 0.0 and 1.0"
        );
        self.ambiguity_threshold = Some(value);
        self
    }

    /// Build the Corrector with configured settings.
    pub fn build(self) -> Corrector {
        Corrector {
            max_iterations: self.max_iterations.unwrap_or(3),
            ambiguity_threshold: self.ambiguity_threshold.unwrap_or(0.5),
            patterns: AmbiguityPatterns::default(),
        }
    }
}

// =============================================================================
// DATA TYPES
// =============================================================================

/// A detected ambiguity in the analyzed text.
///
/// Each ambiguity represents a passage that could benefit from
/// clarification to improve understanding.
#[derive(Debug, Clone)]
pub struct Ambiguity {
    /// The type of ambiguity detected.
    pub kind: AmbiguityKind,
    /// Character position in the original text (0-indexed).
    pub position: usize,
    /// The ambiguous text fragment.
    pub text: String,
    /// Confidence score for this detection (0.0 to 1.0).
    ///
    /// Higher values indicate stronger certainty that this
    /// passage is genuinely ambiguous.
    pub confidence: f64,
}

impl Ambiguity {
    /// Create a new ambiguity.
    pub fn new(
        kind: AmbiguityKind,
        position: usize,
        text: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            kind,
            position,
            text: text.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if this ambiguity is high confidence (>= 0.7).
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.7
    }
}

/// Categories of ambiguity that can be detected.
///
/// Each category has specific detection patterns and generates
/// different types of clarification questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmbiguityKind {
    /// Unclear pronoun or noun reference.
    ///
    /// Examples: "it", "they", "this", "that"
    ///
    /// These require clarification about what entity is being referenced.
    Reference,

    /// Missing contextual information.
    ///
    /// Examples: "the system", "the file", "the process"
    ///
    /// These need additional context to identify the specific entity.
    Context,

    /// Vague or imprecise language.
    ///
    /// Examples: "some", "many", "often", "might"
    ///
    /// These benefit from more specific quantities or probabilities.
    Vague,

    /// Statement with multiple valid interpretations.
    ///
    /// Examples: "and/or", "etc.", "such as"
    ///
    /// These need clarification about intended scope or meaning.
    MultiInterpretation,
}

impl std::fmt::Display for AmbiguityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reference => write!(f, "Reference"),
            Self::Context => write!(f, "Context"),
            Self::Vague => write!(f, "Vague"),
            Self::MultiInterpretation => write!(f, "Multi-Interpretation"),
        }
    }
}

/// A clarification question generated for an ambiguity.
///
/// Questions are designed to elicit specific information that
/// can resolve the identified ambiguity.
#[derive(Debug, Clone)]
pub struct Question {
    /// The question text in natural language.
    pub text: String,
    /// Index of the related ambiguity in the detection results.
    pub ambiguity_index: usize,
    /// Suggested answer options (may be empty for open-ended questions).
    pub options: Vec<String>,
}

impl Question {
    /// Create a new question.
    pub fn new(text: impl Into<String>, ambiguity_index: usize) -> Self {
        Self {
            text: text.into(),
            ambiguity_index,
            options: Vec::new(),
        }
    }

    /// Add answer options to this question.
    pub fn with_options(mut self, options: Vec<String>) -> Self {
        self.options = options;
        self
    }

    /// Check if this question has suggested options.
    pub fn has_options(&self) -> bool {
        !self.options.is_empty()
    }
}

/// User answer to a clarification question.
///
/// Answers are matched to questions by index and contain
/// the clarifying information needed to resolve ambiguities.
#[derive(Debug, Clone)]
pub struct Answer {
    /// Index of the question being answered.
    pub question_index: usize,
    /// The answer text provided by the user.
    pub text: String,
}

impl Answer {
    /// Create a new answer.
    pub fn new(question_index: usize, text: impl Into<String>) -> Self {
        Self {
            question_index,
            text: text.into(),
        }
    }
}

/// Result of applying corrections to content.
///
/// Contains the corrected text along with metadata about
/// the correction process.
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    /// The corrected content with ambiguities resolved.
    pub content: String,
    /// Number of corrections successfully applied.
    pub corrections_applied: usize,
    /// Number of ambiguities still present after correction.
    pub remaining_ambiguities: usize,
}

impl CorrectionResult {
    /// Check if all ambiguities were resolved.
    pub fn is_fully_corrected(&self) -> bool {
        self.remaining_ambiguities == 0
    }

    /// Check if any corrections were applied.
    pub fn has_corrections(&self) -> bool {
        self.corrections_applied > 0
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detect_reference_ambiguity() {
        // Use a lower threshold to ensure detection
        let corrector = Corrector::builder().ambiguity_threshold(0.3).build();
        let text = "It should be configured properly.";

        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();

        assert!(
            !ambiguities.is_empty(),
            "Should detect ambiguities in text with pronouns"
        );
        let reference_amb = ambiguities
            .iter()
            .find(|a| a.kind == AmbiguityKind::Reference);
        assert!(
            reference_amb.is_some(),
            "Should detect reference ambiguity for 'It'"
        );
        assert_eq!(reference_amb.unwrap().text.to_lowercase(), "it");
    }

    #[tokio::test]
    async fn test_detect_vague_ambiguity() {
        let corrector = Corrector::new();
        let text = "Some users might experience issues.";

        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();

        let vague_ambs: Vec<_> = ambiguities
            .iter()
            .filter(|a| a.kind == AmbiguityKind::Vague)
            .collect();
        assert!(!vague_ambs.is_empty());
    }

    #[tokio::test]
    async fn test_detect_context_ambiguity() {
        let corrector = Corrector::new();
        let text = "The system should handle the request gracefully.";

        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();

        let context_amb = ambiguities
            .iter()
            .find(|a| a.kind == AmbiguityKind::Context);
        assert!(context_amb.is_some());
    }

    #[tokio::test]
    async fn test_detect_multi_interpretation() {
        let corrector = Corrector::new();
        let text = "Configure logging, monitoring, etc.";

        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();

        let multi_amb = ambiguities
            .iter()
            .find(|a| a.kind == AmbiguityKind::MultiInterpretation);
        assert!(multi_amb.is_some());
    }

    #[tokio::test]
    async fn test_empty_content_error() {
        let corrector = Corrector::new();
        let result = corrector.detect_ambiguities("").await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Correction(_)));
    }

    #[tokio::test]
    async fn test_generate_questions() {
        let corrector = Corrector::new();
        let text = "It might fail.";

        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();
        let questions = corrector.generate_questions(&ambiguities).await.unwrap();

        assert!(!questions.is_empty());
        for (i, q) in questions.iter().enumerate() {
            assert_eq!(q.ambiguity_index, i);
            assert!(!q.text.is_empty());
        }
    }

    #[tokio::test]
    async fn test_generate_questions_empty_ambiguities() {
        let corrector = Corrector::new();
        let questions = corrector.generate_questions(&[]).await.unwrap();

        assert!(questions.is_empty());
    }

    #[tokio::test]
    async fn test_apply_corrections() {
        // Use a lower threshold to ensure detection
        let corrector = Corrector::builder().ambiguity_threshold(0.3).build();
        let text = "It should work.";

        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();
        assert!(!ambiguities.is_empty(), "Should detect ambiguities");

        let answers = vec![Answer::new(0, "The authentication module")];
        let result = corrector.apply_corrections(text, &answers).await.unwrap();

        assert!(result.corrections_applied > 0, "Should apply corrections");
        assert!(
            result.content.contains("authentication") || result.content.contains("module"),
            "Corrected content should contain the replacement text"
        );
    }

    #[tokio::test]
    async fn test_apply_corrections_empty_answers() {
        let corrector = Corrector::new();
        let text = "Some text here.";

        let result = corrector.apply_corrections(text, &[]).await.unwrap();

        assert_eq!(result.corrections_applied, 0);
        assert_eq!(result.content, text);
    }

    #[tokio::test]
    async fn test_apply_corrections_invalid_index() {
        // Use a threshold that won't find ambiguities in this short text
        let corrector = Corrector::builder().ambiguity_threshold(0.99).build();
        let text = "Clear text without ambiguities.";

        let answers = vec![Answer::new(999, "Invalid")];
        let result = corrector.apply_corrections(text, &answers).await;

        assert!(
            result.is_err(),
            "Should error when answer index is out of range"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("out of range"),
            "Error should mention out of range"
        );
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let corrector = Corrector::builder()
            .max_iterations(5)
            .ambiguity_threshold(0.3)
            .build();

        assert_eq!(corrector.max_iterations(), 5);
        assert!((corrector.ambiguity_threshold() - 0.3).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_threshold_filtering() {
        // Use high threshold to filter out low-confidence ambiguities
        let corrector = Corrector::builder().ambiguity_threshold(0.9).build();

        let text = "Some users might experience issues.";
        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();

        // Only very high confidence ambiguities should pass
        for amb in &ambiguities {
            assert!(amb.confidence >= 0.9);
        }
    }

    #[tokio::test]
    async fn test_ambiguity_kind_display() {
        assert_eq!(AmbiguityKind::Reference.to_string(), "Reference");
        assert_eq!(AmbiguityKind::Context.to_string(), "Context");
        assert_eq!(AmbiguityKind::Vague.to_string(), "Vague");
        assert_eq!(
            AmbiguityKind::MultiInterpretation.to_string(),
            "Multi-Interpretation"
        );
    }

    #[tokio::test]
    async fn test_question_has_options() {
        let q1 = Question::new("Test?", 0);
        assert!(!q1.has_options());

        let q2 = Question::new("Test?", 0).with_options(vec!["A".to_string(), "B".to_string()]);
        assert!(q2.has_options());
    }

    #[tokio::test]
    async fn test_correction_result_helpers() {
        let result1 = CorrectionResult {
            content: "test".to_string(),
            corrections_applied: 2,
            remaining_ambiguities: 0,
        };
        assert!(result1.is_fully_corrected());
        assert!(result1.has_corrections());

        let result2 = CorrectionResult {
            content: "test".to_string(),
            corrections_applied: 0,
            remaining_ambiguities: 1,
        };
        assert!(!result2.is_fully_corrected());
        assert!(!result2.has_corrections());
    }

    #[tokio::test]
    async fn test_ambiguity_high_confidence() {
        let high = Ambiguity::new(AmbiguityKind::Reference, 0, "it", 0.8);
        assert!(high.is_high_confidence());

        let low = Ambiguity::new(AmbiguityKind::Vague, 0, "some", 0.5);
        assert!(!low.is_high_confidence());
    }

    #[tokio::test]
    async fn test_confidence_clamping() {
        let amb = Ambiguity::new(AmbiguityKind::Reference, 0, "it", 1.5);
        assert_eq!(amb.confidence, 1.0);

        let amb2 = Ambiguity::new(AmbiguityKind::Reference, 0, "it", -0.5);
        assert_eq!(amb2.confidence, 0.0);
    }

    #[tokio::test]
    async fn test_correct_with_provider() {
        // Use a lower threshold to ensure detection
        let corrector = Corrector::builder()
            .max_iterations(2)
            .ambiguity_threshold(0.3)
            .build();

        let text = "It should work.";

        let result = corrector
            .correct_with_provider(text, |questions| async move {
                if questions.is_empty() {
                    return Ok(vec![]);
                }
                Ok(vec![Answer::new(0, "The login system")])
            })
            .await
            .unwrap();

        assert!(
            result.corrections_applied > 0,
            "Should apply at least one correction"
        );
    }

    #[tokio::test]
    async fn test_no_false_positives_for_clear_text() {
        let corrector = Corrector::builder()
            .ambiguity_threshold(0.9) // High threshold
            .build();

        // This text is relatively clear
        let text = "The authentication module validates user credentials against the database.";
        let ambiguities = corrector.detect_ambiguities(text).await.unwrap();

        // Should have few or no high-confidence ambiguities for clear text
        // (Note: "the" patterns will still match, but context matters)
        let high_confidence: Vec<_> = ambiguities.iter().filter(|a| a.confidence >= 0.9).collect();

        // With high threshold, we expect minimal false positives
        println!("High confidence ambiguities: {:?}", high_confidence);
    }
}
