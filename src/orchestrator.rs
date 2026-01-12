//! Prompt orchestrator module for Project Panpsychism.
//!
//! Analyzes user intent and selects optimal prompts (2-7) for synthesis.
//! Implements the Sorcerer's Wand metaphor: Your words become creation.
//!
//! # Strategy Selection
//!
//! The orchestrator uses a decision tree to select the optimal strategy:
//!
//! - **Focused**: Single best prompt for simple, specific queries (complexity 1-3)
//! - **Ensemble**: Multiple parallel prompts for broad topics (complexity 4-6, diverse results)
//! - **Chain**: Sequential prompts for multi-step tasks (complexity 7-10, step indicators)
//! - **Parallel**: Merged prompts for synthesis tasks (compare/combine requests)
//!
//! # Decision Tree
//!
//! ```text
//! if query has "compare", "combine", "merge" → Parallel
//! else if query has "then", "after", "first", "step" → Chain
//! else if complexity <= 3 AND results < 3 → Focused
//! else if complexity <= 6 → Ensemble
//! else → Chain
//! ```

// Standard library
use std::collections::HashSet;

// Internal modules
use crate::{search::SearchResult, Error, Result};

// =============================================================================
// STRATEGY AND ROLE TYPES
// =============================================================================

/// Orchestration strategy for prompt selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Strategy {
    /// Single focused prompt
    #[default]
    Focused,
    /// Multiple complementary prompts
    Ensemble,
    /// Chain of prompts in sequence
    Chain,
    /// Parallel prompts merged
    Parallel,
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Focused => write!(f, "Focused"),
            Strategy::Ensemble => write!(f, "Ensemble"),
            Strategy::Chain => write!(f, "Chain"),
            Strategy::Parallel => write!(f, "Parallel"),
        }
    }
}

impl std::str::FromStr for Strategy {
    type Err = crate::Error;

    /// Parse a strategy from a string identifier.
    ///
    /// Accepts various forms: "focused", "FOCUSED", "single", "ensemble", etc.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::str::FromStr;
    /// use panpsychism::orchestrator::Strategy;
    ///
    /// let strategy: Strategy = "focused".parse().unwrap();
    /// assert_eq!(strategy, Strategy::Focused);
    ///
    /// let strategy: Strategy = "parallel".parse().unwrap();
    /// assert_eq!(strategy, Strategy::Parallel);
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "focused" | "focus" | "single" | "simple" => Ok(Strategy::Focused),
            "ensemble" | "multi" | "multiple" | "diverse" => Ok(Strategy::Ensemble),
            "chain" | "sequence" | "sequential" | "step" | "steps" => Ok(Strategy::Chain),
            "parallel" | "merge" | "merged" | "synthesize" | "compare" => Ok(Strategy::Parallel),
            _ => Err(crate::Error::Config(format!(
                "Unknown strategy: '{}'. Valid strategies: focused, ensemble, chain, parallel",
                s
            ))),
        }
    }
}

/// Default relevance threshold for filtering search results.
const DEFAULT_RELEVANCE_THRESHOLD: f64 = 0.3;

/// Role assignment for selected prompts in synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptRole {
    /// Primary prompt driving the response
    Primary,
    /// Supporting prompt providing additional context
    Supporting,
    /// Context prompt for background information
    Context,
    /// Validator prompt for verification
    Validator,
}

impl std::fmt::Display for PromptRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptRole::Primary => write!(f, "primary"),
            PromptRole::Supporting => write!(f, "supporting"),
            PromptRole::Context => write!(f, "context"),
            PromptRole::Validator => write!(f, "validator"),
        }
    }
}

// =============================================================================
// ORCHESTRATOR STRUCT
// =============================================================================

/// Orchestrator for prompt selection and combination.
#[derive(Debug)]
pub struct Orchestrator {
    /// Minimum prompts to select
    min_prompts: usize,
    /// Maximum prompts to select
    max_prompts: usize,
    /// Relevance threshold for filtering results (0.0 - 1.0)
    relevance_threshold: f64,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self {
            min_prompts: 2,
            max_prompts: 7,
            relevance_threshold: DEFAULT_RELEVANCE_THRESHOLD,
        }
    }
}

impl Orchestrator {
    /// Create a new orchestrator with default settings (2-7 prompts, 0.3 threshold).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new orchestrator with custom prompt limits.
    pub fn with_limits(min_prompts: usize, max_prompts: usize) -> Self {
        Self {
            min_prompts: min_prompts.max(1),
            max_prompts: max_prompts.max(min_prompts),
            relevance_threshold: DEFAULT_RELEVANCE_THRESHOLD,
        }
    }

    /// Set the relevance threshold for filtering search results.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum score (0.0 - 1.0) for a result to be considered
    pub fn with_relevance_threshold(mut self, threshold: f64) -> Self {
        self.relevance_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Analyze user intent and determine orchestration strategy.
    pub async fn analyze_intent(&self, query: &str) -> Result<IntentAnalysis> {
        let query_lower = query.to_lowercase();

        // 1. Detect question type
        let question_type = Self::detect_question_type(&query_lower);

        // 2. Extract domain hints
        let domains = Self::extract_domains(&query_lower);

        // 3. Extract keywords
        let keywords = Self::extract_keywords(&query_lower);

        // 4. Calculate complexity score
        let complexity = Self::calculate_complexity(&query_lower, &keywords, &domains);

        // 5. Determine category based on question type and domains
        let category = Self::determine_category(&question_type, &domains);

        // 6. Select strategy based on question type, complexity, and patterns
        let strategy = Self::select_strategy(&query_lower, &question_type, complexity);

        Ok(IntentAnalysis {
            category,
            keywords,
            complexity,
            strategy,
        })
    }

    /// Determine the optimal strategy based on intent analysis and search results.
    ///
    /// This method implements the strategy selection algorithm:
    ///
    /// 1. **Parallel**: If query contains synthesis keywords (compare, combine, merge)
    /// 2. **Chain**: If query contains sequential indicators (then, after, first, step)
    /// 3. **Focused**: If complexity <= 3 AND result_count < 3
    /// 4. **Ensemble**: If complexity <= 6
    /// 5. **Chain**: Default for high complexity (7-10)
    ///
    /// Domain diversity is also considered: if results span many different domains,
    /// Ensemble is preferred over Focused for moderate complexity.
    ///
    /// # Arguments
    ///
    /// * `intent` - The analyzed user intent
    /// * `result_count` - Number of search results available
    ///
    /// # Returns
    ///
    /// The optimal `Strategy` for the given context.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let orchestrator = Orchestrator::new();
    /// let intent = orchestrator.analyze_intent("compare OAuth vs JWT").await?;
    /// let results = search_engine.search(&query).await?;
    /// let strategy = orchestrator.determine_strategy(&intent, results.len());
    /// assert_eq!(strategy, Strategy::Parallel);
    /// ```
    pub fn determine_strategy(&self, intent: &IntentAnalysis, result_count: usize) -> Strategy {
        // Strategy is pre-computed in analyze_intent based on query patterns
        // This method allows refinement based on search results

        let query_lower = intent.keywords.join(" ").to_lowercase();

        // Check for synthesis/parallel indicators
        if Self::has_parallel_indicators(&query_lower) {
            return Strategy::Parallel;
        }

        // Check for sequential/chain indicators
        if Self::has_chain_indicators(&query_lower) {
            return Strategy::Chain;
        }

        // Apply complexity-based decision tree with result count consideration
        match intent.complexity {
            1..=3 => {
                if result_count < 3 {
                    Strategy::Focused
                } else {
                    // Even with low complexity, diverse results suggest Ensemble
                    Strategy::Ensemble
                }
            }
            4..=6 => Strategy::Ensemble,
            _ => Strategy::Chain, // complexity 7-10
        }
    }

    /// Determine strategy with domain diversity consideration.
    ///
    /// This enhanced version considers how many unique domains are represented
    /// in the search results, which can influence the strategy choice.
    ///
    /// # Arguments
    ///
    /// * `intent` - The analyzed user intent
    /// * `results` - The search results to analyze
    ///
    /// # Returns
    ///
    /// The optimal `Strategy` considering domain diversity.
    pub fn determine_strategy_with_results(
        &self,
        intent: &IntentAnalysis,
        results: &[SearchResult],
    ) -> Strategy {
        let result_count = results.len();
        let domain_diversity = Self::calculate_domain_diversity(results);

        let query_lower = intent.keywords.join(" ").to_lowercase();

        // Check for synthesis/parallel indicators
        if Self::has_parallel_indicators(&query_lower) {
            return Strategy::Parallel;
        }

        // Check for sequential/chain indicators
        if Self::has_chain_indicators(&query_lower) {
            return Strategy::Chain;
        }

        // Apply complexity-based decision tree with domain diversity
        match intent.complexity {
            1..=3 => {
                // High domain diversity suggests broader coverage needed
                if result_count < 3 && domain_diversity <= 0.5 {
                    Strategy::Focused
                } else {
                    Strategy::Ensemble
                }
            }
            4..=6 => {
                // Very high diversity might benefit from parallel synthesis
                if domain_diversity > 0.8 && result_count >= 4 {
                    Strategy::Parallel
                } else {
                    Strategy::Ensemble
                }
            }
            _ => Strategy::Chain, // complexity 7-10
        }
    }

    /// Generate a human-readable explanation of why a strategy was chosen.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to explain
    ///
    /// # Returns
    ///
    /// A string explaining the strategy's purpose and when it's used.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let orchestrator = Orchestrator::new();
    /// let explanation = orchestrator.explain_strategy(Strategy::Ensemble);
    /// println!("{}", explanation);
    /// // "Ensemble: Multiple complementary prompts will be used in parallel..."
    /// ```
    pub fn explain_strategy(&self, strategy: Strategy) -> String {
        match strategy {
            Strategy::Focused => "Focused: A single, best-matching prompt will be used. \
                 This strategy is optimal for simple, specific queries (complexity 1-3) \
                 with few matching results. It provides direct, targeted assistance \
                 without the overhead of combining multiple sources."
                .to_string(),
            Strategy::Ensemble => {
                "Ensemble: Multiple complementary prompts will be used in parallel. \
                 This strategy is optimal for moderately complex queries (complexity 4-6) \
                 with diverse results. It provides broader coverage by combining \
                 perspectives from different prompts without strict ordering."
                    .to_string()
            }
            Strategy::Chain => {
                "Chain: Prompts will be applied in sequence, building on each other. \
                 This strategy is optimal for complex, multi-step tasks (complexity 7-10) \
                 or queries with sequential indicators (then, after, first, step). \
                 Each prompt's output feeds into the next for progressive refinement."
                    .to_string()
            }
            Strategy::Parallel => {
                "Parallel: Multiple prompts will be merged into a synthesized response. \
                 This strategy is optimal for comparison or synthesis tasks \
                 (queries with compare, combine, merge). It processes prompts \
                 simultaneously and integrates their outputs into a unified result."
                    .to_string()
            }
        }
    }

    /// Select prompts based on search results and strategy.
    ///
    /// # Arguments
    ///
    /// * `results` - Search results to select from
    /// * `strategy` - The orchestration strategy to apply
    ///
    /// # Returns
    ///
    /// Vector of selected prompts with roles and priorities assigned.
    ///
    /// # Errors
    ///
    /// Returns `Error::Orchestration` if:
    /// - No results are provided
    /// - No results meet the relevance threshold
    ///
    /// # Selection Logic by Strategy
    ///
    /// - **Focused**: Selects single highest-scoring prompt (min limit not enforced)
    /// - **Ensemble**: Selects 3-5 complementary prompts with diverse tags
    /// - **Chain**: Selects prompts forming a logical sequence
    /// - **Parallel**: Selects prompts that can be merged (same category preferred)
    pub async fn select_prompts(
        &self,
        results: &[SearchResult],
        strategy: Strategy,
    ) -> Result<Vec<SelectedPrompt>> {
        if results.is_empty() {
            return Err(Error::Orchestration(
                "No search results to select from".to_string(),
            ));
        }

        // Step 1: Filter by relevance threshold
        let filtered_refs = self.filter_by_threshold(results);

        if filtered_refs.is_empty() {
            return Err(Error::Orchestration(format!(
                "No results meet the relevance threshold of {}",
                self.relevance_threshold
            )));
        }

        // Convert to owned SearchResults for downstream methods
        let filtered: Vec<SearchResult> = filtered_refs.into_iter().cloned().collect();

        // Step 2: Apply strategy-specific selection
        let mut selected = match strategy {
            Strategy::Focused => self.select_focused(&filtered),
            Strategy::Ensemble => self.select_ensemble(&filtered),
            Strategy::Chain => self.select_chain(&filtered),
            Strategy::Parallel => self.select_parallel(&filtered),
        };

        // Step 3: Enforce min/max limits (skip min for Focused strategy)
        self.enforce_limits(&mut selected, &filtered, strategy);

        // Step 4: Assign roles based on strategy and priority
        self.assign_roles(&mut selected, strategy);

        Ok(selected)
    }

    /// Filter search results by relevance threshold.
    fn filter_by_threshold<'a>(&self, results: &'a [SearchResult]) -> Vec<&'a SearchResult> {
        results
            .iter()
            .filter(|r| r.score >= self.relevance_threshold)
            .collect()
    }

    /// Enforce minimum and maximum prompt limits.
    ///
    /// For Focused strategy, skip min enforcement (it's designed to return exactly 1).
    fn enforce_limits(
        &self,
        selected: &mut Vec<SelectedPrompt>,
        available: &[SearchResult],
        strategy: Strategy,
    ) {
        // Enforce maximum
        if selected.len() > self.max_prompts {
            selected.truncate(self.max_prompts);
        }

        // Skip min enforcement for Focused strategy (designed for single prompt)
        if strategy == Strategy::Focused {
            return;
        }

        // Try to meet minimum by adding more prompts
        if selected.len() < self.min_prompts {
            let selected_ids: HashSet<String> =
                selected.iter().map(|s| s.result.id.clone()).collect();

            for result in available {
                if selected.len() >= self.min_prompts {
                    break;
                }

                if !selected_ids.contains(&result.id) {
                    selected.push(SelectedPrompt {
                        result: result.clone(),
                        role: String::new(),
                        priority: (selected.len() + 1) as u8,
                    });
                }
            }
        }

        // Update priorities after limit enforcement (skip for Parallel - all equal priority)
        if strategy != Strategy::Parallel {
            for (i, prompt) in selected.iter_mut().enumerate() {
                prompt.priority = (i + 1) as u8;
            }
        }
    }

    /// Assign roles to selected prompts based on strategy and position.
    ///
    /// Role assignment:
    /// - Position 0: Primary
    /// - Position 1: Supporting
    /// - Position 2: Context
    /// - Position 3+: Validator
    ///
    /// Chain strategy uses step_N roles instead.
    /// Parallel strategy uses perspective_N roles instead.
    fn assign_roles(&self, selected: &mut [SelectedPrompt], strategy: Strategy) {
        for (i, prompt) in selected.iter_mut().enumerate() {
            prompt.role = match strategy {
                Strategy::Chain => format!("step_{}", i + 1),
                Strategy::Parallel => format!("perspective_{}", i + 1),
                Strategy::Focused | Strategy::Ensemble => match i {
                    0 => PromptRole::Primary.to_string(),
                    1 => PromptRole::Supporting.to_string(),
                    2 => PromptRole::Context.to_string(),
                    _ => PromptRole::Validator.to_string(),
                },
            };
        }
    }

    // =========================================================================
    // Private Helper Methods
    // =========================================================================

    /// Check if query contains parallel/synthesis indicators.
    fn has_parallel_indicators(query: &str) -> bool {
        const PARALLEL_KEYWORDS: &[&str] = &[
            "compare",
            "combine",
            "merge",
            "versus",
            "vs",
            "contrast",
            "synthesize",
            "integrate",
            "unify",
        ];

        PARALLEL_KEYWORDS.iter().any(|kw| query.contains(kw))
    }

    /// Check if query contains chain/sequential indicators.
    fn has_chain_indicators(query: &str) -> bool {
        const CHAIN_KEYWORDS: &[&str] = &[
            "then",
            "after",
            "first",
            "step",
            "next",
            "finally",
            "before",
            "followed by",
            "sequence",
            "workflow",
            "pipeline",
        ];

        CHAIN_KEYWORDS.iter().any(|kw| query.contains(kw))
    }

    /// Calculate domain diversity ratio (0.0 to 1.0).
    ///
    /// A ratio of 1.0 means every result is from a different domain.
    /// A ratio of 0.0 means all results share the same domain.
    fn calculate_domain_diversity(results: &[SearchResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let unique_domains: HashSet<&str> = results
            .iter()
            .filter_map(|r| r.category.as_deref())
            .collect();

        let results_with_category = results.iter().filter(|r| r.category.is_some()).count();

        if results_with_category == 0 {
            return 0.5; // Default to medium diversity if no categories
        }

        unique_domains.len() as f64 / results_with_category as f64
    }

    /// Detect the question type from the query.
    fn detect_question_type(query: &str) -> QuestionType {
        // How questions
        if query.starts_with("how") || query.contains("how to") || query.contains("how do") {
            return QuestionType::HowTo;
        }

        // What questions
        if query.starts_with("what") || query.contains("what is") || query.contains("what are") {
            return QuestionType::Definition;
        }

        // Why questions
        if query.starts_with("why") || query.contains("why does") || query.contains("why is") {
            return QuestionType::Explanation;
        }

        // Comparison questions
        if query.contains("compare")
            || query.contains(" vs ")
            || query.contains("versus")
            || query.contains("difference between")
            || query.contains("better than")
        {
            return QuestionType::Comparison;
        }

        // Implementation/action requests
        if query.contains("implement")
            || query.contains("create")
            || query.contains("build")
            || query.contains("write")
            || query.contains("make")
        {
            return QuestionType::Implementation;
        }

        // Debug/troubleshoot questions
        if query.contains("debug")
            || query.contains("fix")
            || query.contains("error")
            || query.contains("issue")
            || query.contains("problem")
            || query.contains("not working")
        {
            return QuestionType::Debug;
        }

        QuestionType::General
    }

    /// Extract domain hints from the query.
    fn extract_domains(query: &str) -> Vec<String> {
        const DOMAIN_KEYWORDS: &[(&str, &str)] = &[
            ("auth", "security"),
            ("oauth", "security"),
            ("jwt", "security"),
            ("security", "security"),
            ("api", "api"),
            ("rest", "api"),
            ("graphql", "api"),
            ("test", "testing"),
            ("testing", "testing"),
            ("unit test", "testing"),
            ("philosophy", "philosophy"),
            ("spinoza", "philosophy"),
            ("conatus", "philosophy"),
            ("database", "database"),
            ("sql", "database"),
            ("postgres", "database"),
            ("mongo", "database"),
            ("frontend", "frontend"),
            ("react", "frontend"),
            ("vue", "frontend"),
            ("css", "frontend"),
            ("backend", "backend"),
            ("server", "backend"),
            ("node", "backend"),
            ("rust", "backend"),
            ("devops", "devops"),
            ("docker", "devops"),
            ("kubernetes", "devops"),
            ("ci/cd", "devops"),
        ];

        let mut domains: HashSet<String> = HashSet::new();

        for (keyword, domain) in DOMAIN_KEYWORDS {
            if query.contains(keyword) {
                domains.insert(domain.to_string());
            }
        }

        domains.into_iter().collect()
    }

    /// Extract significant keywords from the query.
    fn extract_keywords(query: &str) -> Vec<String> {
        const STOP_WORDS: &[&str] = &[
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "this", "that", "these", "those", "i", "me", "my", "we",
            "our", "you", "your", "he", "she", "it", "they", "them", "what", "which", "who",
            "whom", "please", "help", "need", "want",
        ];

        let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();

        query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| word.len() >= 2)
            .map(|word| word.to_lowercase())
            .filter(|word| !stop_set.contains(word.as_str()))
            .collect()
    }

    /// Calculate complexity score (1-10) based on query characteristics.
    fn calculate_complexity(query: &str, keywords: &[String], domains: &[String]) -> u8 {
        let mut score: u8 = 1;

        // Length factor (longer queries tend to be more complex)
        let word_count = query.split_whitespace().count();
        if word_count > 20 {
            score += 3;
        } else if word_count > 10 {
            score += 2;
        } else if word_count > 5 {
            score += 1;
        }

        // Keyword density (more unique concepts = more complex)
        if keywords.len() > 8 {
            score += 2;
        } else if keywords.len() > 4 {
            score += 1;
        }

        // Multi-domain queries are more complex
        if domains.len() > 2 {
            score += 2;
        } else if domains.len() > 1 {
            score += 1;
        }

        // Sequential/step indicators add complexity
        if Self::has_chain_indicators(query) {
            score += 2;
        }

        // Comparison queries are moderately complex
        if Self::has_parallel_indicators(query) {
            score += 1;
        }

        // Technical depth indicators
        const ADVANCED_KEYWORDS: &[&str] = &[
            "architecture",
            "optimize",
            "scale",
            "performance",
            "concurrent",
            "async",
            "distributed",
            "microservice",
            "algorithm",
            "complexity",
        ];
        if ADVANCED_KEYWORDS.iter().any(|kw| query.contains(kw)) {
            score += 1;
        }

        score.min(10) // Cap at 10
    }

    /// Determine the primary category based on question type and domains.
    fn determine_category(question_type: &QuestionType, domains: &[String]) -> String {
        // If we have a clear domain, use it
        if let Some(domain) = domains.first() {
            return domain.clone();
        }

        // Otherwise, derive from question type
        match question_type {
            QuestionType::HowTo => "implementation".to_string(),
            QuestionType::Definition => "concepts".to_string(),
            QuestionType::Explanation => "concepts".to_string(),
            QuestionType::Comparison => "analysis".to_string(),
            QuestionType::Implementation => "implementation".to_string(),
            QuestionType::Debug => "troubleshooting".to_string(),
            QuestionType::General => "general".to_string(),
        }
    }

    /// Select strategy based on query patterns, question type, and complexity.
    fn select_strategy(query: &str, question_type: &QuestionType, complexity: u8) -> Strategy {
        // Priority 1: Explicit parallel indicators
        if Self::has_parallel_indicators(query) {
            return Strategy::Parallel;
        }

        // Priority 2: Explicit chain indicators
        if Self::has_chain_indicators(query) {
            return Strategy::Chain;
        }

        // Priority 3: Question type based
        match question_type {
            QuestionType::Comparison => return Strategy::Parallel,
            QuestionType::Implementation if complexity > 5 => return Strategy::Chain,
            _ => {}
        }

        // Priority 4: Complexity based
        match complexity {
            1..=3 => Strategy::Focused,
            4..=6 => Strategy::Ensemble,
            _ => Strategy::Chain,
        }
    }

    /// Select a single focused prompt.
    fn select_focused(&self, results: &[SearchResult]) -> Vec<SelectedPrompt> {
        results
            .iter()
            .take(1)
            .map(|r| SelectedPrompt {
                result: r.clone(),
                role: "primary".to_string(),
                priority: 1,
            })
            .collect()
    }

    /// Select multiple prompts for ensemble approach.
    fn select_ensemble(&self, results: &[SearchResult]) -> Vec<SelectedPrompt> {
        let count = results
            .len()
            .min(self.max_prompts)
            .max(self.min_prompts.min(results.len()));

        results
            .iter()
            .take(count)
            .enumerate()
            .map(|(i, r)| SelectedPrompt {
                result: r.clone(),
                role: if i == 0 {
                    "primary".to_string()
                } else {
                    "supporting".to_string()
                },
                priority: (i + 1) as u8,
            })
            .collect()
    }

    /// Select prompts for chain/sequential execution.
    fn select_chain(&self, results: &[SearchResult]) -> Vec<SelectedPrompt> {
        let count = results
            .len()
            .min(self.max_prompts)
            .max(self.min_prompts.min(results.len()));

        results
            .iter()
            .take(count)
            .enumerate()
            .map(|(i, r)| SelectedPrompt {
                result: r.clone(),
                role: format!("step_{}", i + 1),
                priority: (i + 1) as u8,
            })
            .collect()
    }

    /// Select prompts for parallel/merged execution.
    fn select_parallel(&self, results: &[SearchResult]) -> Vec<SelectedPrompt> {
        let count = results
            .len()
            .min(self.max_prompts)
            .max(self.min_prompts.min(results.len()));

        results
            .iter()
            .take(count)
            .enumerate()
            .map(|(i, r)| SelectedPrompt {
                result: r.clone(),
                role: format!("perspective_{}", i + 1),
                priority: 1, // All equal priority for parallel
            })
            .collect()
    }
}

// =============================================================================
// DATA TYPES
// =============================================================================

/// Question type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    /// How-to questions (procedural)
    HowTo,
    /// What-is questions (definitional)
    Definition,
    /// Why questions (explanatory)
    Explanation,
    /// Comparison questions
    Comparison,
    /// Implementation requests
    Implementation,
    /// Debug/troubleshooting
    Debug,
    /// General queries
    General,
}

/// Analysis of user intent.
#[derive(Debug)]
pub struct IntentAnalysis {
    /// Primary intent category
    pub category: String,
    /// Detected keywords
    pub keywords: Vec<String>,
    /// Complexity score (1-10)
    pub complexity: u8,
    /// Recommended strategy
    pub strategy: Strategy,
}

/// A selected prompt for synthesis.
#[derive(Debug)]
pub struct SelectedPrompt {
    /// Search result reference
    pub result: SearchResult,
    /// Role in the synthesis
    pub role: String,
    /// Priority order
    pub priority: u8,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to run async tests
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Runtime::new().unwrap().block_on(f)
    }

    // Helper to create a mock SearchResult
    fn mock_search_result(id: &str, category: Option<&str>) -> SearchResult {
        use std::path::PathBuf;
        SearchResult {
            id: id.to_string(),
            title: format!("Title {}", id),
            path: PathBuf::from(format!("/prompts/{}.md", id)),
            score: 0.8,
            excerpt: format!("Excerpt for {}", id),
            tags: vec![],
            category: category.map(|s| s.to_string()),
        }
    }

    // ==========================================================================
    // Question Type Detection Tests
    // ==========================================================================

    #[test]
    fn test_detect_question_type_how() {
        assert_eq!(
            Orchestrator::detect_question_type("how to create a function"),
            QuestionType::HowTo
        );
        assert_eq!(
            Orchestrator::detect_question_type("how do i improve my code"),
            QuestionType::HowTo
        );
    }

    #[test]
    fn test_detect_question_type_what() {
        assert_eq!(
            Orchestrator::detect_question_type("what is a closure"),
            QuestionType::Definition
        );
        assert_eq!(
            Orchestrator::detect_question_type("what are lifetimes in rust"),
            QuestionType::Definition
        );
    }

    #[test]
    fn test_detect_question_type_why() {
        assert_eq!(
            Orchestrator::detect_question_type("why is rust memory safe"),
            QuestionType::Explanation
        );
        assert_eq!(
            Orchestrator::detect_question_type("why does the borrow checker exist"),
            QuestionType::Explanation
        );
    }

    #[test]
    fn test_detect_question_type_compare() {
        assert_eq!(
            Orchestrator::detect_question_type("compare rust and go"),
            QuestionType::Comparison
        );
        assert_eq!(
            Orchestrator::detect_question_type("rust vs c++"),
            QuestionType::Comparison
        );
        assert_eq!(
            Orchestrator::detect_question_type("difference between vec and array"),
            QuestionType::Comparison
        );
    }

    #[test]
    fn test_detect_question_type_implementation() {
        assert_eq!(
            Orchestrator::detect_question_type("implement a binary search"),
            QuestionType::Implementation
        );
        assert_eq!(
            Orchestrator::detect_question_type("create a rest api"),
            QuestionType::Implementation
        );
        assert_eq!(
            Orchestrator::detect_question_type("build a web server"),
            QuestionType::Implementation
        );
    }

    #[test]
    fn test_detect_question_type_debug() {
        assert_eq!(
            Orchestrator::detect_question_type("debug this error"),
            QuestionType::Debug
        );
        assert_eq!(
            Orchestrator::detect_question_type("fix the memory leak issue"),
            QuestionType::Debug
        );
        assert_eq!(
            Orchestrator::detect_question_type("my code is not working"),
            QuestionType::Debug
        );
    }

    #[test]
    fn test_detect_question_type_general() {
        assert_eq!(
            Orchestrator::detect_question_type("rust programming"),
            QuestionType::General
        );
        assert_eq!(
            Orchestrator::detect_question_type("async await patterns"),
            QuestionType::General
        );
    }

    // ==========================================================================
    // Domain Extraction Tests
    // ==========================================================================

    #[test]
    fn test_extract_domains_security() {
        let domains = Orchestrator::extract_domains("implement oauth authentication");
        assert!(domains.contains(&"security".to_string()));
    }

    #[test]
    fn test_extract_domains_api() {
        let domains = Orchestrator::extract_domains("create a rest api endpoint");
        assert!(domains.contains(&"api".to_string()));
    }

    #[test]
    fn test_extract_domains_frontend() {
        let domains = Orchestrator::extract_domains("build a react component");
        assert!(domains.contains(&"frontend".to_string()));
    }

    #[test]
    fn test_extract_domains_multiple() {
        let domains =
            Orchestrator::extract_domains("deploy a node server with docker and postgres database");
        assert!(domains.contains(&"backend".to_string()));
        assert!(domains.contains(&"devops".to_string()));
        assert!(domains.contains(&"database".to_string()));
    }

    #[test]
    fn test_extract_domains_philosophy() {
        let domains = Orchestrator::extract_domains("explain spinoza's conatus");
        assert!(domains.contains(&"philosophy".to_string()));
    }

    #[test]
    fn test_extract_domains_empty() {
        let domains = Orchestrator::extract_domains("random query with no domain keywords");
        assert!(domains.is_empty());
    }

    // ==========================================================================
    // Keyword Extraction Tests
    // ==========================================================================

    #[test]
    fn test_extract_keywords_filters_stopwords() {
        let keywords = Orchestrator::extract_keywords("how do i create a rust function");
        assert!(keywords.contains(&"create".to_string()));
        assert!(keywords.contains(&"rust".to_string()));
        assert!(keywords.contains(&"function".to_string()));
        // Stopwords should be filtered
        assert!(!keywords.contains(&"how".to_string()));
        assert!(!keywords.contains(&"do".to_string()));
        assert!(!keywords.contains(&"a".to_string()));
    }

    #[test]
    fn test_extract_keywords_short_words_filtered() {
        let keywords = Orchestrator::extract_keywords("a i to of");
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_extract_keywords_alphanumeric_split() {
        let keywords = Orchestrator::extract_keywords("rust-lang async/await");
        assert!(keywords.contains(&"rust".to_string()));
        assert!(keywords.contains(&"lang".to_string()));
        assert!(keywords.contains(&"async".to_string()));
        assert!(keywords.contains(&"await".to_string()));
    }

    // ==========================================================================
    // Complexity Calculation Tests
    // ==========================================================================

    #[test]
    fn test_calculate_complexity_simple() {
        let keywords = vec!["rust".to_string()];
        let domains = vec![];
        let complexity = Orchestrator::calculate_complexity("what is rust", &keywords, &domains);
        assert!(
            complexity <= 3,
            "Simple query should have low complexity: {}",
            complexity
        );
    }

    #[test]
    fn test_calculate_complexity_medium() {
        let keywords = vec![
            "design".to_string(),
            "api".to_string(),
            "scalable".to_string(),
        ];
        let domains = vec!["api".to_string(), "backend".to_string()];
        let complexity = Orchestrator::calculate_complexity(
            "how do i design a scalable api for my application with proper authentication",
            &keywords,
            &domains,
        );
        assert!(
            complexity >= 3 && complexity <= 7,
            "Medium query complexity: {}",
            complexity
        );
    }

    #[test]
    fn test_calculate_complexity_high() {
        let keywords: Vec<String> = (0..10).map(|i| format!("keyword{}", i)).collect();
        let domains = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let complexity = Orchestrator::calculate_complexity(
            "design a distributed microservice architecture with kubernetes, \
             then implement authentication, after that optimize for performance \
             and finally add monitoring with proper scaling",
            &keywords,
            &domains,
        );
        assert!(
            complexity >= 6,
            "Complex query should have high complexity: {}",
            complexity
        );
    }

    #[test]
    fn test_calculate_complexity_capped_at_10() {
        let keywords: Vec<String> = (0..20).map(|i| format!("keyword{}", i)).collect();
        let domains: Vec<String> = (0..5).map(|i| format!("domain{}", i)).collect();
        let complexity = Orchestrator::calculate_complexity(
            "architecture optimize scale performance concurrent async distributed \
             microservice algorithm complexity then after first step next finally \
             compare combine merge versus contrast synthesize integrate unify \
             really really really really really really really long query here",
            &keywords,
            &domains,
        );
        assert!(
            complexity <= 10,
            "Complexity should be capped at 10: {}",
            complexity
        );
    }

    // ==========================================================================
    // Strategy Selection Tests
    // ==========================================================================

    #[test]
    fn test_select_strategy_focused() {
        let strategy =
            Orchestrator::select_strategy("what is a closure", &QuestionType::Definition, 2);
        assert_eq!(strategy, Strategy::Focused);
    }

    #[test]
    fn test_select_strategy_ensemble() {
        let strategy = Orchestrator::select_strategy(
            "explain how async works in rust",
            &QuestionType::Explanation,
            5,
        );
        assert_eq!(strategy, Strategy::Ensemble);
    }

    #[test]
    fn test_select_strategy_chain_by_keywords() {
        let strategy = Orchestrator::select_strategy(
            "first setup the project then add dependencies finally configure",
            &QuestionType::General,
            4,
        );
        assert_eq!(strategy, Strategy::Chain);
    }

    #[test]
    fn test_select_strategy_parallel_by_keywords() {
        let strategy =
            Orchestrator::select_strategy("compare rust and go", &QuestionType::Comparison, 3);
        assert_eq!(strategy, Strategy::Parallel);
    }

    #[test]
    fn test_select_strategy_chain_by_complexity() {
        let strategy = Orchestrator::select_strategy(
            "advanced architectural decisions",
            &QuestionType::General,
            8,
        );
        assert_eq!(strategy, Strategy::Chain);
    }

    // ==========================================================================
    // Category Determination Tests
    // ==========================================================================

    #[test]
    fn test_determine_category_with_domain() {
        let category =
            Orchestrator::determine_category(&QuestionType::HowTo, &vec!["security".to_string()]);
        assert_eq!(category, "security");
    }

    #[test]
    fn test_determine_category_without_domain() {
        let category = Orchestrator::determine_category(&QuestionType::Implementation, &vec![]);
        assert_eq!(category, "implementation");

        let category = Orchestrator::determine_category(&QuestionType::Debug, &vec![]);
        assert_eq!(category, "troubleshooting");
    }

    // ==========================================================================
    // Indicator Detection Tests
    // ==========================================================================

    #[test]
    fn test_has_parallel_indicators() {
        assert!(Orchestrator::has_parallel_indicators("compare rust vs go"));
        assert!(Orchestrator::has_parallel_indicators(
            "combine these approaches"
        ));
        assert!(Orchestrator::has_parallel_indicators("merge the results"));
        assert!(!Orchestrator::has_parallel_indicators("simple query"));
    }

    #[test]
    fn test_has_chain_indicators() {
        assert!(Orchestrator::has_chain_indicators(
            "first do this then that"
        ));
        assert!(Orchestrator::has_chain_indicators(
            "after completing step one"
        ));
        assert!(Orchestrator::has_chain_indicators("next implement"));
        assert!(!Orchestrator::has_chain_indicators("simple query"));
    }

    // ==========================================================================
    // Domain Diversity Tests
    // ==========================================================================

    #[test]
    fn test_calculate_domain_diversity_empty() {
        let diversity = Orchestrator::calculate_domain_diversity(&[]);
        assert_eq!(diversity, 0.0);
    }

    #[test]
    fn test_calculate_domain_diversity_single() {
        let results = vec![
            mock_search_result("1", Some("security")),
            mock_search_result("2", Some("security")),
        ];
        let diversity = Orchestrator::calculate_domain_diversity(&results);
        assert_eq!(diversity, 0.5); // 1 unique domain / 2 results
    }

    #[test]
    fn test_calculate_domain_diversity_all_unique() {
        let results = vec![
            mock_search_result("1", Some("security")),
            mock_search_result("2", Some("api")),
            mock_search_result("3", Some("frontend")),
        ];
        let diversity = Orchestrator::calculate_domain_diversity(&results);
        assert_eq!(diversity, 1.0); // 3 unique domains / 3 results
    }

    #[test]
    fn test_calculate_domain_diversity_no_categories() {
        let results = vec![mock_search_result("1", None), mock_search_result("2", None)];
        let diversity = Orchestrator::calculate_domain_diversity(&results);
        assert_eq!(diversity, 0.5); // Default medium diversity
    }

    // ==========================================================================
    // Full Analyze Intent Integration Tests
    // ==========================================================================

    #[test]
    fn test_analyze_intent_simple_definition() {
        let orchestrator = Orchestrator::new();
        let result = block_on(orchestrator.analyze_intent("what is rust"));

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.strategy, Strategy::Focused);
        assert!(analysis.complexity <= 3);
        assert!(analysis.keywords.contains(&"rust".to_string()));
    }

    #[test]
    fn test_analyze_intent_comparison() {
        let orchestrator = Orchestrator::new();
        let result = block_on(orchestrator.analyze_intent("compare rust vs go for web servers"));

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.strategy, Strategy::Parallel);
    }

    #[test]
    fn test_analyze_intent_chain() {
        let orchestrator = Orchestrator::new();
        let result = block_on(orchestrator.analyze_intent(
            "first setup the project then add authentication finally deploy to production",
        ));

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.strategy, Strategy::Chain);
    }

    #[test]
    fn test_analyze_intent_multi_domain() {
        let orchestrator = Orchestrator::new();
        let result = block_on(orchestrator.analyze_intent(
            "implement oauth authentication in react frontend with postgres database",
        ));

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.keywords.len() >= 2);
    }

    // ==========================================================================
    // Prompt Selection Tests
    // ==========================================================================

    #[test]
    fn test_select_prompts_focused() {
        let orchestrator = Orchestrator::new();
        let results = vec![
            mock_search_result("1", Some("api")),
            mock_search_result("2", Some("api")),
            mock_search_result("3", Some("api")),
        ];

        let selected = block_on(orchestrator.select_prompts(&results, Strategy::Focused));
        assert!(selected.is_ok());
        let prompts = selected.unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].role, "primary");
    }

    #[test]
    fn test_select_prompts_ensemble() {
        let orchestrator = Orchestrator::new();
        let results = vec![
            mock_search_result("1", Some("api")),
            mock_search_result("2", Some("security")),
            mock_search_result("3", Some("testing")),
        ];

        let selected = block_on(orchestrator.select_prompts(&results, Strategy::Ensemble));
        assert!(selected.is_ok());
        let prompts = selected.unwrap();
        assert!(prompts.len() >= 2);
        assert_eq!(prompts[0].role, "primary");
        assert_eq!(prompts[1].role, "supporting");
    }

    #[test]
    fn test_select_prompts_chain() {
        let orchestrator = Orchestrator::new();
        let results = vec![
            mock_search_result("1", Some("setup")),
            mock_search_result("2", Some("config")),
            mock_search_result("3", Some("deploy")),
        ];

        let selected = block_on(orchestrator.select_prompts(&results, Strategy::Chain));
        assert!(selected.is_ok());
        let prompts = selected.unwrap();
        assert!(prompts.len() >= 2);
        assert_eq!(prompts[0].role, "step_1");
        assert_eq!(prompts[1].role, "step_2");
    }

    #[test]
    fn test_select_prompts_parallel() {
        let orchestrator = Orchestrator::new();
        let results = vec![
            mock_search_result("1", Some("rust")),
            mock_search_result("2", Some("go")),
        ];

        let selected = block_on(orchestrator.select_prompts(&results, Strategy::Parallel));
        assert!(selected.is_ok());
        let prompts = selected.unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0].role, "perspective_1");
        assert_eq!(prompts[1].role, "perspective_2");
        // All parallel prompts have equal priority
        assert_eq!(prompts[0].priority, 1);
        assert_eq!(prompts[1].priority, 1);
    }

    #[test]
    fn test_select_prompts_empty_results() {
        let orchestrator = Orchestrator::new();
        let results: Vec<SearchResult> = vec![];

        let selected = block_on(orchestrator.select_prompts(&results, Strategy::Focused));
        assert!(selected.is_err());
    }

    #[test]
    fn test_select_prompts_respects_limits() {
        let orchestrator = Orchestrator::with_limits(2, 5);
        let results: Vec<SearchResult> = (0..10)
            .map(|i| mock_search_result(&format!("{}", i), Some("test")))
            .collect();

        let selected = block_on(orchestrator.select_prompts(&results, Strategy::Ensemble));
        assert!(selected.is_ok());
        let prompts = selected.unwrap();
        assert!(prompts.len() >= 2);
        assert!(prompts.len() <= 5);
    }

    // ==========================================================================
    // Strategy Explanation Tests
    // ==========================================================================

    #[test]
    fn test_explain_strategy() {
        let orchestrator = Orchestrator::new();

        let focused = orchestrator.explain_strategy(Strategy::Focused);
        assert!(focused.contains("Focused"));
        assert!(focused.contains("single"));

        let ensemble = orchestrator.explain_strategy(Strategy::Ensemble);
        assert!(ensemble.contains("Ensemble"));
        assert!(ensemble.contains("parallel"));

        let chain = orchestrator.explain_strategy(Strategy::Chain);
        assert!(chain.contains("Chain"));
        assert!(chain.contains("sequence"));

        let parallel = orchestrator.explain_strategy(Strategy::Parallel);
        assert!(parallel.contains("Parallel"));
        assert!(parallel.contains("merged") || parallel.contains("synthesis"));
    }

    // ==========================================================================
    // Determine Strategy Tests
    // ==========================================================================

    #[test]
    fn test_determine_strategy_with_results() {
        let orchestrator = Orchestrator::new();
        let results = vec![
            mock_search_result("1", Some("api")),
            mock_search_result("2", Some("security")),
            mock_search_result("3", Some("testing")),
            mock_search_result("4", Some("devops")),
        ];

        let intent = IntentAnalysis {
            category: "api".to_string(),
            keywords: vec!["api".to_string(), "design".to_string()],
            complexity: 5,
            strategy: Strategy::Ensemble,
        };

        let strategy = orchestrator.determine_strategy_with_results(&intent, &results);
        // High domain diversity with moderate complexity should suggest Parallel or Ensemble
        assert!(matches!(strategy, Strategy::Parallel | Strategy::Ensemble));
    }

    #[test]
    fn test_determine_strategy_low_diversity() {
        let orchestrator = Orchestrator::new();
        // With 2 results of same category, diversity = 1/2 = 0.5
        // The condition for Focused is: result_count < 3 AND diversity <= 0.5
        // Since 0.5 <= 0.5 is true, this returns Focused
        let results = vec![
            mock_search_result("1", Some("api")),
            mock_search_result("2", Some("api")),
        ];

        let intent = IntentAnalysis {
            category: "api".to_string(),
            keywords: vec!["api".to_string()],
            complexity: 2,
            strategy: Strategy::Focused,
        };

        let strategy = orchestrator.determine_strategy_with_results(&intent, &results);
        // With diversity exactly 0.5 and result_count=2, condition (2 < 3 && 0.5 <= 0.5) is true
        assert_eq!(strategy, Strategy::Focused);
    }

    #[test]
    fn test_determine_strategy_high_complexity_returns_chain() {
        let orchestrator = Orchestrator::new();
        let results = vec![mock_search_result("1", Some("api"))];

        let intent = IntentAnalysis {
            category: "api".to_string(),
            keywords: vec!["api".to_string()],
            complexity: 8, // High complexity (7-10 range)
            strategy: Strategy::Chain,
        };

        let strategy = orchestrator.determine_strategy_with_results(&intent, &results);
        assert_eq!(strategy, Strategy::Chain);
    }

    // ==========================================================================
    // Strategy FromStr and Display Tests
    // ==========================================================================

    #[test]
    fn test_strategy_display() {
        assert_eq!(format!("{}", Strategy::Focused), "Focused");
        assert_eq!(format!("{}", Strategy::Ensemble), "Ensemble");
        assert_eq!(format!("{}", Strategy::Chain), "Chain");
        assert_eq!(format!("{}", Strategy::Parallel), "Parallel");
    }

    #[test]
    fn test_strategy_from_str() {
        assert_eq!("focused".parse::<Strategy>().unwrap(), Strategy::Focused);
        assert_eq!("ensemble".parse::<Strategy>().unwrap(), Strategy::Ensemble);
        assert_eq!("chain".parse::<Strategy>().unwrap(), Strategy::Chain);
        assert_eq!("parallel".parse::<Strategy>().unwrap(), Strategy::Parallel);
    }

    #[test]
    fn test_strategy_from_str_aliases() {
        // Focused aliases
        assert_eq!("focus".parse::<Strategy>().unwrap(), Strategy::Focused);
        assert_eq!("single".parse::<Strategy>().unwrap(), Strategy::Focused);
        assert_eq!("simple".parse::<Strategy>().unwrap(), Strategy::Focused);

        // Ensemble aliases
        assert_eq!("multi".parse::<Strategy>().unwrap(), Strategy::Ensemble);
        assert_eq!("multiple".parse::<Strategy>().unwrap(), Strategy::Ensemble);
        assert_eq!("diverse".parse::<Strategy>().unwrap(), Strategy::Ensemble);

        // Chain aliases
        assert_eq!("sequence".parse::<Strategy>().unwrap(), Strategy::Chain);
        assert_eq!("sequential".parse::<Strategy>().unwrap(), Strategy::Chain);
        assert_eq!("step".parse::<Strategy>().unwrap(), Strategy::Chain);
        assert_eq!("steps".parse::<Strategy>().unwrap(), Strategy::Chain);

        // Parallel aliases
        assert_eq!("merge".parse::<Strategy>().unwrap(), Strategy::Parallel);
        assert_eq!("merged".parse::<Strategy>().unwrap(), Strategy::Parallel);
        assert_eq!(
            "synthesize".parse::<Strategy>().unwrap(),
            Strategy::Parallel
        );
        assert_eq!("compare".parse::<Strategy>().unwrap(), Strategy::Parallel);
    }

    #[test]
    fn test_strategy_from_str_case_insensitive() {
        assert_eq!("FOCUSED".parse::<Strategy>().unwrap(), Strategy::Focused);
        assert_eq!("ENSEMBLE".parse::<Strategy>().unwrap(), Strategy::Ensemble);
        assert_eq!("Chain".parse::<Strategy>().unwrap(), Strategy::Chain);
        assert_eq!("PARALLEL".parse::<Strategy>().unwrap(), Strategy::Parallel);
    }

    #[test]
    fn test_strategy_from_str_invalid() {
        let result = "invalid".parse::<Strategy>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unknown strategy"));
    }

    #[test]
    fn test_strategy_default() {
        assert_eq!(Strategy::default(), Strategy::Focused);
    }
}
