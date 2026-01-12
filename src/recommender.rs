//! Recommender Agent module for Project Panpsychism.
//!
//! Implements "The Path Advisor" — an agent that suggests relevant prompts
//! and optimal paths to users. Like a wise oracle who perceives connections
//! between knowledge and intent, the Recommender Agent guides seekers toward
//! the most relevant magical incantations for their needs.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In the arcane arts of prompt orchestration, the Recommender Agent serves as
//! the **Path Advisor** — a specialist in the magical art of guidance:
//!
//! - **Recommendations** are divined paths leading to relevant spells
//! - **Relevance Scores** measure the alignment between seeker and spell
//! - **Explanations** illuminate why a particular path is suggested
//! - **Diversity** ensures the seeker explores varied magical approaches
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Drive to help users find the most effective prompts
//! - **RATIO**: Logical matching of user intent to prompt capabilities
//! - **LAETITIA**: Joy through successful discovery and guidance
//! - **NATURA**: Natural flow from user need to appropriate solution
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::recommender::{RecommenderAgent, RecommendationStrategy};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let agent = RecommenderAgent::new();
//!
//!     let query = "How do I implement authentication?";
//!     let prompts = vec!["auth-oauth2", "auth-jwt", "api-security"];
//!
//!     let recommendations = agent.recommend(query, &prompts).await?;
//!
//!     for rec in recommendations.items() {
//!         println!("{}: {:.2} - {}", rec.prompt_id, rec.relevance_score, rec.explanation);
//!     }
//!     Ok(())
//! }
//! ```

use crate::{Error, PromptId, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// RECOMMENDATION STRATEGY
// =============================================================================

/// Strategies for generating recommendations.
///
/// Each strategy represents a different approach to divining the most
/// relevant prompts for a user's needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum RecommendationStrategy {
    /// Content-based filtering using text similarity.
    ///
    /// Analyzes the content of prompts and matches them to the user's
    /// query based on textual similarity and keyword overlap.
    #[default]
    ContentBased,

    /// Collaborative filtering based on usage patterns.
    ///
    /// Recommends prompts that similar users found useful,
    /// leveraging collective wisdom from past interactions.
    Collaborative,

    /// Knowledge-based recommendations using domain expertise.
    ///
    /// Uses structured knowledge about prompt relationships,
    /// categories, and prerequisites to make informed suggestions.
    KnowledgeBased,

    /// Hybrid approach combining multiple strategies.
    ///
    /// Blends content-based, collaborative, and knowledge-based
    /// signals for more robust recommendations.
    Hybrid,
}

impl std::fmt::Display for RecommendationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContentBased => write!(f, "content-based"),
            Self::Collaborative => write!(f, "collaborative"),
            Self::KnowledgeBased => write!(f, "knowledge-based"),
            Self::Hybrid => write!(f, "hybrid"),
        }
    }
}

impl std::str::FromStr for RecommendationStrategy {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "content" | "content-based" | "contentbased" => Ok(Self::ContentBased),
            "collaborative" | "collab" => Ok(Self::Collaborative),
            "knowledge" | "knowledge-based" | "knowledgebased" => Ok(Self::KnowledgeBased),
            "hybrid" | "mixed" | "combined" => Ok(Self::Hybrid),
            _ => Err(Error::Config(format!(
                "Unknown recommendation strategy: '{}'. Valid strategies: content-based, \
                 collaborative, knowledge-based, hybrid",
                s
            ))),
        }
    }
}

impl RecommendationStrategy {
    /// Get all available strategies.
    pub fn all() -> Vec<Self> {
        vec![
            Self::ContentBased,
            Self::Collaborative,
            Self::KnowledgeBased,
            Self::Hybrid,
        ]
    }

    /// Get a description of this strategy.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ContentBased => "Matches prompts based on textual similarity and keyword overlap",
            Self::Collaborative => "Recommends prompts that similar users found useful",
            Self::KnowledgeBased => "Uses structured knowledge about prompt relationships",
            Self::Hybrid => "Combines multiple strategies for robust recommendations",
        }
    }
}

// =============================================================================
// RECOMMENDATION CONFIG
// =============================================================================

/// Configuration for the Recommender Agent.
#[derive(Debug, Clone)]
pub struct RecommendationConfig {
    /// Maximum number of recommendations to return.
    pub max_recommendations: usize,

    /// Diversity factor (0.0 = no diversity, 1.0 = maximum diversity).
    ///
    /// Higher values promote variety in recommendations, while lower
    /// values prioritize pure relevance scores.
    pub diversity_factor: f64,

    /// Weight for recency in scoring (0.0 = ignore recency).
    ///
    /// Higher values boost prompts that have been recently updated
    /// or used.
    pub recency_weight: f64,

    /// Weight for popularity in scoring (0.0 = ignore popularity).
    ///
    /// Higher values boost prompts that are frequently used.
    pub popularity_weight: f64,

    /// Minimum relevance score to include in results (0.0 - 1.0).
    pub min_relevance_threshold: f64,

    /// The recommendation strategy to use.
    pub strategy: RecommendationStrategy,

    /// Whether to include explanations for each recommendation.
    pub include_explanations: bool,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            max_recommendations: 10,
            diversity_factor: 0.3,
            recency_weight: 0.1,
            popularity_weight: 0.2,
            min_relevance_threshold: 0.1,
            strategy: RecommendationStrategy::default(),
            include_explanations: true,
        }
    }
}

impl RecommendationConfig {
    /// Create a fast configuration with fewer recommendations.
    pub fn fast() -> Self {
        Self {
            max_recommendations: 5,
            diversity_factor: 0.2,
            recency_weight: 0.0,
            popularity_weight: 0.0,
            min_relevance_threshold: 0.2,
            strategy: RecommendationStrategy::ContentBased,
            include_explanations: false,
        }
    }

    /// Create a thorough configuration with more recommendations.
    pub fn thorough() -> Self {
        Self {
            max_recommendations: 20,
            diversity_factor: 0.5,
            recency_weight: 0.2,
            popularity_weight: 0.3,
            min_relevance_threshold: 0.05,
            strategy: RecommendationStrategy::Hybrid,
            include_explanations: true,
        }
    }

    /// Create a diversity-focused configuration.
    pub fn diverse() -> Self {
        Self {
            max_recommendations: 15,
            diversity_factor: 0.7,
            recency_weight: 0.1,
            popularity_weight: 0.1,
            min_relevance_threshold: 0.1,
            strategy: RecommendationStrategy::Hybrid,
            include_explanations: true,
        }
    }
}

// =============================================================================
// RECOMMENDATION
// =============================================================================

/// A single recommendation from the Path Advisor.
///
/// Contains the recommended prompt along with metadata explaining
/// why it was selected and how relevant it is to the user's query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// The unique identifier of the recommended prompt.
    pub prompt_id: PromptId,

    /// Relevance score (0.0 - 1.0) indicating how well this prompt
    /// matches the user's needs.
    pub relevance_score: f64,

    /// Human-readable explanation of why this prompt was recommended.
    pub explanation: String,

    /// Tags associated with this prompt.
    pub tags: Vec<String>,

    /// Category of the prompt (if available).
    pub category: Option<String>,

    /// Confidence in the recommendation (0.0 - 1.0).
    pub confidence: f64,
}

impl Recommendation {
    /// Create a new recommendation.
    pub fn new(prompt_id: impl Into<String>, relevance_score: f64) -> Self {
        Self {
            prompt_id: prompt_id.into(),
            relevance_score: normalize_score(relevance_score),
            explanation: String::new(),
            tags: Vec::new(),
            category: None,
            confidence: normalize_score(relevance_score),
        }
    }

    /// Set the explanation.
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    /// Set the tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = normalize_score(confidence);
        self
    }

    /// Check if this recommendation has a high relevance score.
    pub fn is_highly_relevant(&self) -> bool {
        self.relevance_score >= 0.7
    }

    /// Check if this recommendation has a moderate relevance score.
    pub fn is_moderately_relevant(&self) -> bool {
        self.relevance_score >= 0.4 && self.relevance_score < 0.7
    }
}

// =============================================================================
// RECOMMENDATION SET
// =============================================================================

/// A set of recommendations returned by the Path Advisor.
///
/// Contains the recommendations along with metadata about the
/// recommendation process.
#[derive(Debug, Clone)]
pub struct RecommendationSet {
    /// The recommendations, ordered by relevance.
    recommendations: Vec<Recommendation>,

    /// The original query that triggered these recommendations.
    pub query: String,

    /// The strategy used to generate recommendations.
    pub strategy: RecommendationStrategy,

    /// Processing duration in milliseconds.
    pub duration_ms: u64,

    /// Total number of prompts considered.
    pub total_considered: usize,

    /// Number of prompts that passed the relevance threshold.
    pub passed_threshold: usize,

    /// Average relevance score across all recommendations.
    pub average_relevance: f64,

    /// Diversity score of the recommendation set (0.0 - 1.0).
    pub diversity_score: f64,
}

impl RecommendationSet {
    /// Create a new recommendation set.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            recommendations: Vec::new(),
            query: query.into(),
            strategy: RecommendationStrategy::default(),
            duration_ms: 0,
            total_considered: 0,
            passed_threshold: 0,
            average_relevance: 0.0,
            diversity_score: 0.0,
        }
    }

    /// Get the recommendations.
    pub fn items(&self) -> &[Recommendation] {
        &self.recommendations
    }

    /// Get the number of recommendations.
    pub fn len(&self) -> usize {
        self.recommendations.len()
    }

    /// Check if there are no recommendations.
    pub fn is_empty(&self) -> bool {
        self.recommendations.is_empty()
    }

    /// Get the top recommendation, if any.
    pub fn top(&self) -> Option<&Recommendation> {
        self.recommendations.first()
    }

    /// Get the top N recommendations.
    pub fn top_n(&self, n: usize) -> &[Recommendation] {
        let end = n.min(self.recommendations.len());
        &self.recommendations[..end]
    }

    /// Get recommendations with relevance above a threshold.
    pub fn above_threshold(&self, threshold: f64) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.relevance_score >= threshold)
            .collect()
    }

    /// Get recommendations by category.
    pub fn by_category(&self, category: &str) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.category.as_deref() == Some(category))
            .collect()
    }

    /// Get recommendations by tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Get a summary of the recommendation set.
    pub fn summary(&self) -> String {
        if self.recommendations.is_empty() {
            return "No recommendations found".to_string();
        }

        format!(
            "Found {} recommendations (avg relevance: {:.2}, diversity: {:.2}) in {}ms",
            self.recommendations.len(),
            self.average_relevance,
            self.diversity_score,
            self.duration_ms
        )
    }

    /// Convert to a vector of prompt IDs.
    pub fn prompt_ids(&self) -> Vec<&str> {
        self.recommendations.iter().map(|r| r.prompt_id.as_str()).collect()
    }

    /// Add a recommendation to the set.
    fn push(&mut self, recommendation: Recommendation) {
        self.recommendations.push(recommendation);
    }

    /// Sort recommendations by relevance score (descending).
    fn sort_by_relevance(&mut self) {
        self.recommendations
            .sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Calculate and update the average relevance score.
    fn update_average_relevance(&mut self) {
        if self.recommendations.is_empty() {
            self.average_relevance = 0.0;
        } else {
            let sum: f64 = self.recommendations.iter().map(|r| r.relevance_score).sum();
            self.average_relevance = sum / self.recommendations.len() as f64;
        }
    }

    /// Truncate to maximum number of recommendations.
    fn truncate(&mut self, max: usize) {
        self.recommendations.truncate(max);
    }
}

// =============================================================================
// PROMPT METADATA
// =============================================================================

/// Metadata about a prompt used for recommendations.
#[derive(Debug, Clone, Default)]
pub struct PromptMetadata {
    /// The prompt ID.
    pub id: PromptId,

    /// Title of the prompt.
    pub title: String,

    /// Description of the prompt.
    pub description: String,

    /// Tags associated with the prompt.
    pub tags: Vec<String>,

    /// Category of the prompt.
    pub category: Option<String>,

    /// Usage count (for popularity scoring).
    pub usage_count: u64,

    /// Last used timestamp (for recency scoring).
    pub last_used: Option<u64>,

    /// Related prompt IDs.
    pub related_prompts: Vec<PromptId>,
}

impl PromptMetadata {
    /// Create new prompt metadata.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Set the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Get the text content for similarity matching.
    pub fn text_content(&self) -> String {
        format!(
            "{} {} {}",
            self.title,
            self.description,
            self.tags.join(" ")
        )
    }
}

// =============================================================================
// RECOMMENDER AGENT (THE PATH ADVISOR)
// =============================================================================

/// The Recommender Agent — The Path Advisor of the Sorcerer's Tower.
///
/// Responsible for suggesting relevant prompts and optimal paths to users.
/// The Path Advisor perceives connections between user intent and available
/// spells, guiding seekers toward the most appropriate magical incantations.
///
/// ## Capabilities
///
/// - **Content-Based Matching**: Analyzes text similarity between query and prompts
/// - **Collaborative Filtering**: Leverages usage patterns from similar users
/// - **Knowledge-Based Reasoning**: Uses structured prompt relationships
/// - **Hybrid Recommendations**: Combines multiple approaches for robustness
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::recommender::{RecommenderAgent, RecommendationStrategy};
///
/// let agent = RecommenderAgent::builder()
///     .max_recommendations(5)
///     .strategy(RecommendationStrategy::Hybrid)
///     .build();
///
/// let recommendations = agent.recommend("authentication", &prompts).await?;
/// ```
#[derive(Debug, Clone)]
pub struct RecommenderAgent {
    /// Configuration for recommendation behavior.
    config: RecommendationConfig,

    /// Cached prompt metadata for efficient lookups.
    prompt_cache: HashMap<PromptId, PromptMetadata>,

    /// Usage history for collaborative filtering.
    usage_history: HashMap<PromptId, u64>,
}

impl Default for RecommenderAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl RecommenderAgent {
    /// Create a new Recommender Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: RecommendationConfig::default(),
            prompt_cache: HashMap::new(),
            usage_history: HashMap::new(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> RecommenderAgentBuilder {
        RecommenderAgentBuilder::default()
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: RecommendationConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RecommendationConfig {
        &self.config
    }

    /// Register prompt metadata for better recommendations.
    pub fn register_prompt(&mut self, metadata: PromptMetadata) {
        self.prompt_cache.insert(metadata.id.clone(), metadata);
    }

    /// Record usage of a prompt for collaborative filtering.
    pub fn record_usage(&mut self, prompt_id: &str) {
        *self.usage_history.entry(prompt_id.to_string()).or_insert(0) += 1;
    }

    // =========================================================================
    // MAIN RECOMMENDATION METHOD
    // =========================================================================

    /// Generate recommendations for the given query.
    ///
    /// This is the primary method of the Path Advisor, divining the most
    /// relevant prompts for the user's needs.
    ///
    /// # Arguments
    ///
    /// * `query` - The user's search query or intent
    /// * `available_prompts` - List of prompt IDs to consider
    ///
    /// # Returns
    ///
    /// A `RecommendationSet` containing ranked recommendations.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if the query is empty.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = RecommenderAgent::new();
    /// let recommendations = agent.recommend(
    ///     "How do I implement OAuth2?",
    ///     &["auth-oauth2", "auth-jwt", "api-security"]
    /// ).await?;
    /// ```
    pub async fn recommend(&self, query: &str, available_prompts: &[&str]) -> Result<RecommendationSet> {
        let start = Instant::now();

        if query.trim().is_empty() {
            return Err(Error::Validation("Cannot recommend for empty query".to_string()));
        }

        if available_prompts.is_empty() {
            let mut set = RecommendationSet::new(query);
            set.duration_ms = start.elapsed().as_millis() as u64;
            return Ok(set);
        }

        debug!(
            "Generating recommendations for query '{}' using {} strategy",
            query, self.config.strategy
        );

        // Generate recommendations based on strategy
        let mut recommendations = match self.config.strategy {
            RecommendationStrategy::ContentBased => {
                self.recommend_content_based(query, available_prompts)
            }
            RecommendationStrategy::Collaborative => {
                self.recommend_collaborative(query, available_prompts)
            }
            RecommendationStrategy::KnowledgeBased => {
                self.recommend_knowledge_based(query, available_prompts)
            }
            RecommendationStrategy::Hybrid => {
                self.recommend_hybrid(query, available_prompts)
            }
        };

        // Apply diversity if configured
        if self.config.diversity_factor > 0.0 {
            recommendations = self.apply_diversity(recommendations);
        }

        // Filter by threshold
        let passed_threshold = recommendations
            .iter()
            .filter(|r| r.relevance_score >= self.config.min_relevance_threshold)
            .count();

        recommendations.retain(|r| r.relevance_score >= self.config.min_relevance_threshold);

        // Sort and truncate
        recommendations.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        recommendations.truncate(self.config.max_recommendations);

        // Build result set
        let mut result = RecommendationSet::new(query);
        result.strategy = self.config.strategy;
        result.total_considered = available_prompts.len();
        result.passed_threshold = passed_threshold;

        for rec in recommendations {
            result.push(rec);
        }

        result.update_average_relevance();
        result.diversity_score = self.calculate_diversity_score(&result);
        result.duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "Generated {} recommendations in {}ms (strategy: {})",
            result.len(),
            result.duration_ms,
            self.config.strategy
        );

        Ok(result)
    }

    // =========================================================================
    // STRATEGY IMPLEMENTATIONS
    // =========================================================================

    /// Content-based recommendation using text similarity.
    fn recommend_content_based(&self, query: &str, prompts: &[&str]) -> Vec<Recommendation> {
        let query_terms = extract_terms(query);

        prompts
            .iter()
            .map(|&prompt_id| {
                let score = if let Some(metadata) = self.prompt_cache.get(prompt_id) {
                    // Use cached metadata for better matching
                    let prompt_terms = extract_terms(&metadata.text_content());
                    calculate_text_similarity(&query_terms, &prompt_terms)
                } else {
                    // Fallback to ID-based matching
                    let prompt_terms = extract_terms(prompt_id);
                    calculate_text_similarity(&query_terms, &prompt_terms)
                };

                let explanation = if self.config.include_explanations {
                    generate_content_explanation(&query_terms, prompt_id, score)
                } else {
                    String::new()
                };

                let mut rec = Recommendation::new(prompt_id, score)
                    .with_explanation(explanation);

                if let Some(metadata) = self.prompt_cache.get(prompt_id) {
                    rec = rec.with_tags(metadata.tags.clone());
                    if let Some(ref cat) = metadata.category {
                        rec = rec.with_category(cat.clone());
                    }
                }

                rec
            })
            .collect()
    }

    /// Collaborative filtering based on usage patterns.
    fn recommend_collaborative(&self, query: &str, prompts: &[&str]) -> Vec<Recommendation> {
        let query_terms = extract_terms(query);

        prompts
            .iter()
            .map(|&prompt_id| {
                // Base score from content similarity
                let content_score = if let Some(metadata) = self.prompt_cache.get(prompt_id) {
                    let prompt_terms = extract_terms(&metadata.text_content());
                    calculate_text_similarity(&query_terms, &prompt_terms)
                } else {
                    let prompt_terms = extract_terms(prompt_id);
                    calculate_text_similarity(&query_terms, &prompt_terms)
                };

                // Boost from usage popularity
                let usage_count = self.usage_history.get(prompt_id).copied().unwrap_or(0);
                let popularity_boost = calculate_popularity_boost(usage_count);

                let score = content_score * (1.0 - self.config.popularity_weight)
                    + popularity_boost * self.config.popularity_weight;

                let explanation = if self.config.include_explanations {
                    if usage_count > 0 {
                        format!(
                            "Matches your query and has been used {} times by others",
                            usage_count
                        )
                    } else {
                        format!("Matches your query based on content similarity")
                    }
                } else {
                    String::new()
                };

                Recommendation::new(prompt_id, score).with_explanation(explanation)
            })
            .collect()
    }

    /// Knowledge-based recommendations using domain expertise.
    fn recommend_knowledge_based(&self, query: &str, prompts: &[&str]) -> Vec<Recommendation> {
        let query_terms = extract_terms(query);
        let query_categories = infer_categories(&query_terms);

        prompts
            .iter()
            .map(|&prompt_id| {
                let mut score = 0.0;
                let mut reasons = Vec::new();

                if let Some(metadata) = self.prompt_cache.get(prompt_id) {
                    // Category matching
                    if let Some(ref category) = metadata.category {
                        if query_categories.contains(category) {
                            score += 0.4;
                            reasons.push(format!("Category '{}' matches your intent", category));
                        }
                    }

                    // Tag matching
                    let matching_tags: Vec<_> = metadata
                        .tags
                        .iter()
                        .filter(|tag| query_terms.iter().any(|t| tag.contains(t) || t.contains(tag.as_str())))
                        .collect();
                    if !matching_tags.is_empty() {
                        score += 0.3 * (matching_tags.len() as f64 / metadata.tags.len().max(1) as f64);
                        reasons.push(format!("Tags match: {}", matching_tags.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
                    }

                    // Related prompts boost
                    if !metadata.related_prompts.is_empty() {
                        score += 0.1;
                    }

                    // Base content similarity
                    let prompt_terms = extract_terms(&metadata.text_content());
                    score += 0.2 * calculate_text_similarity(&query_terms, &prompt_terms);
                } else {
                    // Fallback to content-based
                    let prompt_terms = extract_terms(prompt_id);
                    score = calculate_text_similarity(&query_terms, &prompt_terms);
                }

                let explanation = if self.config.include_explanations && !reasons.is_empty() {
                    reasons.join("; ")
                } else if self.config.include_explanations {
                    "Potential match based on domain knowledge".to_string()
                } else {
                    String::new()
                };

                Recommendation::new(prompt_id, score).with_explanation(explanation)
            })
            .collect()
    }

    /// Hybrid recommendations combining multiple strategies.
    fn recommend_hybrid(&self, query: &str, prompts: &[&str]) -> Vec<Recommendation> {
        let content_recs = self.recommend_content_based(query, prompts);
        let collab_recs = self.recommend_collaborative(query, prompts);
        let knowledge_recs = self.recommend_knowledge_based(query, prompts);

        // Create score maps
        let mut combined_scores: HashMap<&str, (f64, Vec<String>)> = HashMap::new();

        // Weights for each strategy
        let content_weight = 0.4;
        let collab_weight = 0.3;
        let knowledge_weight = 0.3;

        for rec in &content_recs {
            let entry = combined_scores.entry(rec.prompt_id.as_str()).or_insert((0.0, Vec::new()));
            entry.0 += rec.relevance_score * content_weight;
            if !rec.explanation.is_empty() {
                entry.1.push(format!("Content: {}", rec.explanation));
            }
        }

        for rec in &collab_recs {
            let entry = combined_scores.entry(rec.prompt_id.as_str()).or_insert((0.0, Vec::new()));
            entry.0 += rec.relevance_score * collab_weight;
            if !rec.explanation.is_empty() && !entry.1.iter().any(|s| s.contains(&rec.explanation)) {
                entry.1.push(format!("Usage: {}", rec.explanation));
            }
        }

        for rec in &knowledge_recs {
            let entry = combined_scores.entry(rec.prompt_id.as_str()).or_insert((0.0, Vec::new()));
            entry.0 += rec.relevance_score * knowledge_weight;
            if !rec.explanation.is_empty() && !entry.1.iter().any(|s| s.contains(&rec.explanation)) {
                entry.1.push(format!("Knowledge: {}", rec.explanation));
            }
        }

        combined_scores
            .into_iter()
            .map(|(id, (score, reasons))| {
                let explanation = if self.config.include_explanations {
                    if reasons.is_empty() {
                        "Combined analysis from multiple strategies".to_string()
                    } else {
                        reasons.join("; ")
                    }
                } else {
                    String::new()
                };

                let mut rec = Recommendation::new(id, score).with_explanation(explanation);

                if let Some(metadata) = self.prompt_cache.get(id) {
                    rec = rec.with_tags(metadata.tags.clone());
                    if let Some(ref cat) = metadata.category {
                        rec = rec.with_category(cat.clone());
                    }
                }

                rec
            })
            .collect()
    }

    // =========================================================================
    // DIVERSITY AND SCORING
    // =========================================================================

    /// Apply diversity to the recommendations.
    fn apply_diversity(&self, mut recommendations: Vec<Recommendation>) -> Vec<Recommendation> {
        if recommendations.len() <= 1 {
            return recommendations;
        }

        // Sort by relevance first
        recommendations.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut diversified = Vec::new();
        let mut seen_categories: HashSet<String> = HashSet::new();
        let mut seen_tags: HashSet<String> = HashSet::new();

        for rec in recommendations {
            let category_penalty = if let Some(ref cat) = rec.category {
                if seen_categories.contains(cat) {
                    self.config.diversity_factor * 0.5
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let tag_overlap: usize = rec
                .tags
                .iter()
                .filter(|t| seen_tags.contains(*t))
                .count();
            let tag_penalty = if !rec.tags.is_empty() {
                self.config.diversity_factor * 0.3 * (tag_overlap as f64 / rec.tags.len() as f64)
            } else {
                0.0
            };

            let diversified_score = rec.relevance_score * (1.0 - category_penalty - tag_penalty);

            // Track seen categories and tags
            if let Some(ref cat) = rec.category {
                seen_categories.insert(cat.clone());
            }
            for tag in &rec.tags {
                seen_tags.insert(tag.clone());
            }

            diversified.push(Recommendation {
                relevance_score: diversified_score,
                ..rec
            });
        }

        diversified
    }

    /// Calculate the diversity score of a recommendation set.
    fn calculate_diversity_score(&self, set: &RecommendationSet) -> f64 {
        if set.is_empty() {
            return 0.0;
        }

        let unique_categories: HashSet<_> = set
            .items()
            .iter()
            .filter_map(|r| r.category.as_deref())
            .collect();

        let all_tags: Vec<_> = set.items().iter().flat_map(|r| &r.tags).collect();
        let unique_tags: HashSet<_> = all_tags.iter().collect();

        let category_diversity = if !unique_categories.is_empty() {
            unique_categories.len() as f64 / set.len() as f64
        } else {
            0.0
        };

        let tag_diversity = if !all_tags.is_empty() {
            unique_tags.len() as f64 / all_tags.len() as f64
        } else {
            0.0
        };

        (category_diversity + tag_diversity) / 2.0
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom RecommenderAgent configuration.
#[derive(Debug, Default)]
pub struct RecommenderAgentBuilder {
    config: Option<RecommendationConfig>,
    prompts: Vec<PromptMetadata>,
    usage_history: HashMap<PromptId, u64>,
}

impl RecommenderAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: RecommendationConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the maximum number of recommendations.
    pub fn max_recommendations(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_recommendations = max;
        self.config = Some(config);
        self
    }

    /// Set the diversity factor.
    pub fn diversity_factor(mut self, factor: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.diversity_factor = factor.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the recency weight.
    pub fn recency_weight(mut self, weight: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.recency_weight = weight.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the popularity weight.
    pub fn popularity_weight(mut self, weight: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.popularity_weight = weight.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the minimum relevance threshold.
    pub fn min_relevance_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.min_relevance_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the recommendation strategy.
    pub fn strategy(mut self, strategy: RecommendationStrategy) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.strategy = strategy;
        self.config = Some(config);
        self
    }

    /// Set whether to include explanations.
    pub fn include_explanations(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_explanations = include;
        self.config = Some(config);
        self
    }

    /// Add prompt metadata.
    pub fn add_prompt(mut self, metadata: PromptMetadata) -> Self {
        self.prompts.push(metadata);
        self
    }

    /// Add usage history.
    pub fn add_usage(mut self, prompt_id: impl Into<String>, count: u64) -> Self {
        self.usage_history.insert(prompt_id.into(), count);
        self
    }

    /// Build the RecommenderAgent.
    pub fn build(self) -> RecommenderAgent {
        let mut agent = RecommenderAgent {
            config: self.config.unwrap_or_default(),
            prompt_cache: HashMap::new(),
            usage_history: self.usage_history,
        };

        for metadata in self.prompts {
            agent.prompt_cache.insert(metadata.id.clone(), metadata);
        }

        agent
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Normalize a score to the 0.0 - 1.0 range.
fn normalize_score(score: f64) -> f64 {
    score.clamp(0.0, 1.0)
}

/// Extract terms from text for matching.
fn extract_terms(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter(|s| s.len() >= 2)
        .map(String::from)
        .collect()
}

/// Calculate text similarity between two term sets.
fn calculate_text_similarity(query_terms: &HashSet<String>, prompt_terms: &HashSet<String>) -> f64 {
    if query_terms.is_empty() || prompt_terms.is_empty() {
        return 0.0;
    }

    let intersection: HashSet<_> = query_terms.intersection(prompt_terms).collect();
    let union_size = query_terms.len() + prompt_terms.len() - intersection.len();

    if union_size == 0 {
        return 0.0;
    }

    // Jaccard similarity
    let jaccard = intersection.len() as f64 / union_size as f64;

    // Also consider partial matches
    let mut partial_score = 0.0;
    for query_term in query_terms {
        for prompt_term in prompt_terms {
            if query_term.contains(prompt_term) || prompt_term.contains(query_term) {
                partial_score += 0.5;
            }
        }
    }
    partial_score /= (query_terms.len() * prompt_terms.len()).max(1) as f64;

    // Combined score
    (jaccard + partial_score).min(1.0)
}

/// Calculate popularity boost from usage count.
fn calculate_popularity_boost(usage_count: u64) -> f64 {
    // Logarithmic scaling to prevent runaway scores
    if usage_count == 0 {
        return 0.0;
    }
    (1.0 + (usage_count as f64).ln()) / 10.0
}

/// Infer categories from query terms.
fn infer_categories(terms: &HashSet<String>) -> HashSet<String> {
    let mut categories = HashSet::new();

    // Map common terms to categories
    let category_keywords: HashMap<&str, &str> = [
        ("auth", "security"),
        ("authentication", "security"),
        ("oauth", "security"),
        ("jwt", "security"),
        ("login", "security"),
        ("api", "development"),
        ("rest", "development"),
        ("graphql", "development"),
        ("database", "data"),
        ("sql", "data"),
        ("postgres", "data"),
        ("mongo", "data"),
        ("test", "testing"),
        ("testing", "testing"),
        ("unit", "testing"),
        ("deploy", "devops"),
        ("docker", "devops"),
        ("kubernetes", "devops"),
        ("ci", "devops"),
        ("cd", "devops"),
    ]
    .into_iter()
    .collect();

    for term in terms {
        if let Some(&category) = category_keywords.get(term.as_str()) {
            categories.insert(category.to_string());
        }
    }

    categories
}

/// Generate an explanation for content-based matching.
fn generate_content_explanation(query_terms: &HashSet<String>, prompt_id: &str, score: f64) -> String {
    let prompt_terms = extract_terms(prompt_id);
    let matching: Vec<_> = query_terms
        .intersection(&prompt_terms)
        .take(3)
        .collect();

    if matching.is_empty() {
        if score > 0.0 {
            format!("Partial match with relevance score {:.2}", score)
        } else {
            "Low relevance to your query".to_string()
        }
    } else {
        format!(
            "Matches on: {}",
            matching.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        )
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // RecommendationStrategy Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_strategy_display() {
        assert_eq!(RecommendationStrategy::ContentBased.to_string(), "content-based");
        assert_eq!(RecommendationStrategy::Collaborative.to_string(), "collaborative");
        assert_eq!(RecommendationStrategy::KnowledgeBased.to_string(), "knowledge-based");
        assert_eq!(RecommendationStrategy::Hybrid.to_string(), "hybrid");
    }

    #[test]
    fn test_strategy_from_str() {
        assert_eq!(
            "content-based".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::ContentBased
        );
        assert_eq!(
            "collaborative".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::Collaborative
        );
        assert_eq!(
            "knowledge-based".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::KnowledgeBased
        );
        assert_eq!(
            "hybrid".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::Hybrid
        );
    }

    #[test]
    fn test_strategy_from_str_aliases() {
        assert_eq!(
            "content".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::ContentBased
        );
        assert_eq!(
            "collab".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::Collaborative
        );
        assert_eq!(
            "knowledge".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::KnowledgeBased
        );
        assert_eq!(
            "mixed".parse::<RecommendationStrategy>().unwrap(),
            RecommendationStrategy::Hybrid
        );
    }

    #[test]
    fn test_strategy_from_str_invalid() {
        assert!("invalid".parse::<RecommendationStrategy>().is_err());
    }

    #[test]
    fn test_strategy_all() {
        let all = RecommendationStrategy::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&RecommendationStrategy::ContentBased));
        assert!(all.contains(&RecommendationStrategy::Collaborative));
        assert!(all.contains(&RecommendationStrategy::KnowledgeBased));
        assert!(all.contains(&RecommendationStrategy::Hybrid));
    }

    #[test]
    fn test_strategy_description() {
        assert!(!RecommendationStrategy::ContentBased.description().is_empty());
        assert!(!RecommendationStrategy::Collaborative.description().is_empty());
        assert!(!RecommendationStrategy::KnowledgeBased.description().is_empty());
        assert!(!RecommendationStrategy::Hybrid.description().is_empty());
    }

    // -------------------------------------------------------------------------
    // RecommendationConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = RecommendationConfig::default();
        assert_eq!(config.max_recommendations, 10);
        assert!((config.diversity_factor - 0.3).abs() < f64::EPSILON);
        assert_eq!(config.strategy, RecommendationStrategy::ContentBased);
    }

    #[test]
    fn test_config_fast() {
        let config = RecommendationConfig::fast();
        assert_eq!(config.max_recommendations, 5);
        assert!(!config.include_explanations);
    }

    #[test]
    fn test_config_thorough() {
        let config = RecommendationConfig::thorough();
        assert_eq!(config.max_recommendations, 20);
        assert_eq!(config.strategy, RecommendationStrategy::Hybrid);
    }

    #[test]
    fn test_config_diverse() {
        let config = RecommendationConfig::diverse();
        assert!((config.diversity_factor - 0.7).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Recommendation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_recommendation_new() {
        let rec = Recommendation::new("auth-oauth2", 0.85);
        assert_eq!(rec.prompt_id, "auth-oauth2");
        assert!((rec.relevance_score - 0.85).abs() < f64::EPSILON);
        assert!(rec.explanation.is_empty());
        assert!(rec.tags.is_empty());
    }

    #[test]
    fn test_recommendation_builder() {
        let rec = Recommendation::new("api-design", 0.75)
            .with_explanation("Good match for your query")
            .with_tags(vec!["api".to_string(), "rest".to_string()])
            .with_category("development")
            .with_confidence(0.9);

        assert_eq!(rec.explanation, "Good match for your query");
        assert_eq!(rec.tags.len(), 2);
        assert_eq!(rec.category, Some("development".to_string()));
        assert!((rec.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recommendation_relevance_levels() {
        let high = Recommendation::new("test", 0.8);
        assert!(high.is_highly_relevant());
        assert!(!high.is_moderately_relevant());

        let moderate = Recommendation::new("test", 0.5);
        assert!(!moderate.is_highly_relevant());
        assert!(moderate.is_moderately_relevant());

        let low = Recommendation::new("test", 0.2);
        assert!(!low.is_highly_relevant());
        assert!(!low.is_moderately_relevant());
    }

    #[test]
    fn test_recommendation_score_normalization() {
        let rec = Recommendation::new("test", 1.5);
        assert!((rec.relevance_score - 1.0).abs() < f64::EPSILON);

        let rec2 = Recommendation::new("test", -0.5);
        assert!((rec2.relevance_score - 0.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // RecommendationSet Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_recommendation_set_new() {
        let set = RecommendationSet::new("test query");
        assert_eq!(set.query, "test query");
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_recommendation_set_operations() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("prompt1", 0.9));
        set.push(Recommendation::new("prompt2", 0.7));
        set.push(Recommendation::new("prompt3", 0.5));

        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
    }

    #[test]
    fn test_recommendation_set_top() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("prompt1", 0.9));
        set.push(Recommendation::new("prompt2", 0.7));
        set.sort_by_relevance();

        assert_eq!(set.top().unwrap().prompt_id, "prompt1");
    }

    #[test]
    fn test_recommendation_set_top_n() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("prompt1", 0.9));
        set.push(Recommendation::new("prompt2", 0.7));
        set.push(Recommendation::new("prompt3", 0.5));
        set.sort_by_relevance();

        let top2 = set.top_n(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].prompt_id, "prompt1");
        assert_eq!(top2[1].prompt_id, "prompt2");
    }

    #[test]
    fn test_recommendation_set_above_threshold() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("high", 0.9));
        set.push(Recommendation::new("medium", 0.6));
        set.push(Recommendation::new("low", 0.3));

        let above_half = set.above_threshold(0.5);
        assert_eq!(above_half.len(), 2);
    }

    #[test]
    fn test_recommendation_set_by_category() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("auth1", 0.9).with_category("security"));
        set.push(Recommendation::new("api1", 0.8).with_category("development"));
        set.push(Recommendation::new("auth2", 0.7).with_category("security"));

        let security = set.by_category("security");
        assert_eq!(security.len(), 2);
    }

    #[test]
    fn test_recommendation_set_by_tag() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("prompt1", 0.9).with_tags(vec!["api".to_string()]));
        set.push(Recommendation::new("prompt2", 0.8).with_tags(vec!["database".to_string()]));
        set.push(Recommendation::new("prompt3", 0.7).with_tags(vec!["api".to_string(), "rest".to_string()]));

        let api_prompts = set.by_tag("api");
        assert_eq!(api_prompts.len(), 2);
    }

    #[test]
    fn test_recommendation_set_summary() {
        let mut set = RecommendationSet::new("test");
        assert!(set.summary().contains("No recommendations"));

        set.push(Recommendation::new("prompt1", 0.8));
        set.update_average_relevance();
        set.duration_ms = 50;
        assert!(set.summary().contains("Found 1 recommendations"));
    }

    #[test]
    fn test_recommendation_set_prompt_ids() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("prompt1", 0.9));
        set.push(Recommendation::new("prompt2", 0.8));

        let ids = set.prompt_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"prompt1"));
        assert!(ids.contains(&"prompt2"));
    }

    #[test]
    fn test_recommendation_set_truncate() {
        let mut set = RecommendationSet::new("test");
        for i in 0..10 {
            set.push(Recommendation::new(format!("prompt{}", i), 0.5));
        }
        set.truncate(5);
        assert_eq!(set.len(), 5);
    }

    // -------------------------------------------------------------------------
    // PromptMetadata Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prompt_metadata_new() {
        let metadata = PromptMetadata::new("auth-oauth2");
        assert_eq!(metadata.id, "auth-oauth2");
        assert!(metadata.title.is_empty());
        assert!(metadata.tags.is_empty());
    }

    #[test]
    fn test_prompt_metadata_builder() {
        let metadata = PromptMetadata::new("api-design")
            .with_title("API Design Best Practices")
            .with_description("Guide to designing RESTful APIs")
            .with_tags(vec!["api".to_string(), "rest".to_string()])
            .with_category("development");

        assert_eq!(metadata.title, "API Design Best Practices");
        assert_eq!(metadata.description, "Guide to designing RESTful APIs");
        assert_eq!(metadata.tags.len(), 2);
        assert_eq!(metadata.category, Some("development".to_string()));
    }

    #[test]
    fn test_prompt_metadata_text_content() {
        let metadata = PromptMetadata::new("test")
            .with_title("Test Title")
            .with_description("Test Description")
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()]);

        let content = metadata.text_content();
        assert!(content.contains("Test Title"));
        assert!(content.contains("Test Description"));
        assert!(content.contains("tag1"));
    }

    // -------------------------------------------------------------------------
    // RecommenderAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_recommender_agent_new() {
        let agent = RecommenderAgent::new();
        assert_eq!(agent.config.max_recommendations, 10);
        assert!(agent.prompt_cache.is_empty());
    }

    #[test]
    fn test_recommender_agent_builder() {
        let agent = RecommenderAgent::builder()
            .max_recommendations(5)
            .diversity_factor(0.5)
            .strategy(RecommendationStrategy::Hybrid)
            .include_explanations(false)
            .build();

        assert_eq!(agent.config.max_recommendations, 5);
        assert!((agent.config.diversity_factor - 0.5).abs() < f64::EPSILON);
        assert_eq!(agent.config.strategy, RecommendationStrategy::Hybrid);
        assert!(!agent.config.include_explanations);
    }

    #[test]
    fn test_recommender_agent_register_prompt() {
        let mut agent = RecommenderAgent::new();
        agent.register_prompt(PromptMetadata::new("test-prompt").with_title("Test"));

        assert!(agent.prompt_cache.contains_key("test-prompt"));
    }

    #[test]
    fn test_recommender_agent_record_usage() {
        let mut agent = RecommenderAgent::new();
        agent.record_usage("prompt1");
        agent.record_usage("prompt1");
        agent.record_usage("prompt2");

        assert_eq!(*agent.usage_history.get("prompt1").unwrap(), 2);
        assert_eq!(*agent.usage_history.get("prompt2").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_recommend_empty_query() {
        let agent = RecommenderAgent::new();
        let result = agent.recommend("", &["prompt1", "prompt2"]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_recommend_empty_prompts() {
        let agent = RecommenderAgent::new();
        let result = agent.recommend("test query", &[]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_recommend_content_based() {
        let agent = RecommenderAgent::builder()
            .strategy(RecommendationStrategy::ContentBased)
            .build();

        let prompts = vec!["auth-oauth2", "auth-jwt", "api-design"];
        let result = agent.recommend("authentication oauth", &prompts).await;

        assert!(result.is_ok());
        let set = result.unwrap();
        assert!(!set.is_empty());
        assert_eq!(set.strategy, RecommendationStrategy::ContentBased);
    }

    #[tokio::test]
    async fn test_recommend_collaborative() {
        let mut agent = RecommenderAgent::builder()
            .strategy(RecommendationStrategy::Collaborative)
            .min_relevance_threshold(0.0) // Allow all results for testing
            .build();

        agent.record_usage("auth-oauth2");
        agent.record_usage("auth-oauth2");
        agent.record_usage("auth-oauth2");

        let prompts = vec!["auth-oauth2", "auth-jwt", "api-design"];
        let result = agent.recommend("auth", &prompts).await;

        assert!(result.is_ok());
        let set = result.unwrap();
        assert!(!set.is_empty());
        // auth-oauth2 should rank higher due to usage boost
    }

    #[tokio::test]
    async fn test_recommend_knowledge_based() {
        let agent = RecommenderAgent::builder()
            .strategy(RecommendationStrategy::KnowledgeBased)
            .add_prompt(
                PromptMetadata::new("auth-oauth2")
                    .with_title("OAuth2 Authentication")
                    .with_category("security")
                    .with_tags(vec!["auth".to_string(), "oauth".to_string()]),
            )
            .build();

        let prompts = vec!["auth-oauth2", "api-design"];
        let result = agent.recommend("security authentication", &prompts).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recommend_hybrid() {
        let agent = RecommenderAgent::builder()
            .strategy(RecommendationStrategy::Hybrid)
            .build();

        let prompts = vec!["auth-oauth2", "auth-jwt", "api-design"];
        let result = agent.recommend("api authentication", &prompts).await;

        assert!(result.is_ok());
        let set = result.unwrap();
        assert_eq!(set.strategy, RecommendationStrategy::Hybrid);
    }

    #[tokio::test]
    async fn test_recommend_with_metadata() {
        let agent = RecommenderAgent::builder()
            .add_prompt(
                PromptMetadata::new("auth-oauth2")
                    .with_title("OAuth2 Guide")
                    .with_description("Complete guide to OAuth2 authentication")
                    .with_tags(vec!["auth".to_string(), "oauth".to_string(), "security".to_string()])
                    .with_category("security"),
            )
            .add_prompt(
                PromptMetadata::new("api-rest")
                    .with_title("REST API Design")
                    .with_description("Best practices for RESTful API design")
                    .with_tags(vec!["api".to_string(), "rest".to_string()])
                    .with_category("development"),
            )
            .build();

        let prompts = vec!["auth-oauth2", "api-rest"];
        let result = agent.recommend("oauth authentication security", &prompts).await;

        assert!(result.is_ok());
        let set = result.unwrap();
        assert!(!set.is_empty());

        // The oauth prompt should rank higher
        let top = set.top().unwrap();
        assert_eq!(top.prompt_id, "auth-oauth2");
    }

    #[tokio::test]
    async fn test_recommend_respects_threshold() {
        let agent = RecommenderAgent::builder()
            .min_relevance_threshold(0.8)
            .build();

        let prompts = vec!["unrelated-topic", "random-stuff"];
        let result = agent.recommend("authentication oauth", &prompts).await;

        assert!(result.is_ok());
        // Low similarity prompts should be filtered out
        let set = result.unwrap();
        for rec in set.items() {
            assert!(rec.relevance_score >= 0.8);
        }
    }

    #[tokio::test]
    async fn test_recommend_respects_max_count() {
        let agent = RecommenderAgent::builder()
            .max_recommendations(2)
            .min_relevance_threshold(0.0)
            .build();

        let prompts = vec!["auth-1", "auth-2", "auth-3", "auth-4", "auth-5"];
        let result = agent.recommend("auth", &prompts).await;

        assert!(result.is_ok());
        let set = result.unwrap();
        assert!(set.len() <= 2);
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalize_score() {
        assert!((normalize_score(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((normalize_score(1.5) - 1.0).abs() < f64::EPSILON);
        assert!((normalize_score(-0.5) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_terms() {
        let terms = extract_terms("Hello World! How are you?");
        assert!(terms.contains("hello"));
        assert!(terms.contains("world"));
        assert!(terms.contains("how"));
        assert!(terms.contains("are"));
        assert!(terms.contains("you"));
    }

    #[test]
    fn test_extract_terms_with_special_chars() {
        let terms = extract_terms("auth-oauth2 api_key");
        assert!(terms.contains("auth-oauth2"));
        assert!(terms.contains("api_key"));
    }

    #[test]
    fn test_extract_terms_filters_short() {
        let terms = extract_terms("a b cd ef");
        assert!(!terms.contains("a"));
        assert!(!terms.contains("b"));
        assert!(terms.contains("cd"));
        assert!(terms.contains("ef"));
    }

    #[test]
    fn test_calculate_text_similarity_identical() {
        let terms1 = extract_terms("hello world");
        let terms2 = extract_terms("hello world");
        let sim = calculate_text_similarity(&terms1, &terms2);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_text_similarity_partial() {
        let terms1 = extract_terms("hello world");
        let terms2 = extract_terms("hello everyone");
        let sim = calculate_text_similarity(&terms1, &terms2);
        assert!(sim > 0.0);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_calculate_text_similarity_no_overlap() {
        let terms1 = extract_terms("hello world");
        let terms2 = extract_terms("foo bar");
        let sim = calculate_text_similarity(&terms1, &terms2);
        assert!((sim - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_text_similarity_empty() {
        let terms1: HashSet<String> = HashSet::new();
        let terms2 = extract_terms("hello");
        assert!((calculate_text_similarity(&terms1, &terms2) - 0.0).abs() < f64::EPSILON);
        assert!((calculate_text_similarity(&terms2, &terms1) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_popularity_boost() {
        assert!((calculate_popularity_boost(0) - 0.0).abs() < f64::EPSILON);
        assert!(calculate_popularity_boost(1) > 0.0);
        assert!(calculate_popularity_boost(10) > calculate_popularity_boost(1));
        assert!(calculate_popularity_boost(100) > calculate_popularity_boost(10));
    }

    #[test]
    fn test_infer_categories() {
        let terms = extract_terms("oauth authentication api");
        let categories = infer_categories(&terms);
        assert!(categories.contains("security"));
        assert!(categories.contains("development"));
    }

    #[test]
    fn test_infer_categories_devops() {
        let terms = extract_terms("docker kubernetes deploy");
        let categories = infer_categories(&terms);
        assert!(categories.contains("devops"));
    }

    #[test]
    fn test_infer_categories_empty() {
        let terms = extract_terms("random words here");
        let categories = infer_categories(&terms);
        assert!(categories.is_empty());
    }

    #[test]
    fn test_generate_content_explanation() {
        let query_terms = extract_terms("oauth authentication");
        let explanation = generate_content_explanation(&query_terms, "oauth-guide", 0.8);
        assert!(explanation.contains("oauth") || explanation.contains("relevance"));
    }

    // -------------------------------------------------------------------------
    // Diversity Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_diversity_applied() {
        let agent = RecommenderAgent::builder()
            .diversity_factor(0.8)
            .min_relevance_threshold(0.0)
            .add_prompt(
                PromptMetadata::new("auth-1")
                    .with_category("security")
                    .with_tags(vec!["auth".to_string()]),
            )
            .add_prompt(
                PromptMetadata::new("auth-2")
                    .with_category("security")
                    .with_tags(vec!["auth".to_string()]),
            )
            .add_prompt(
                PromptMetadata::new("api-1")
                    .with_category("development")
                    .with_tags(vec!["api".to_string()]),
            )
            .build();

        let prompts = vec!["auth-1", "auth-2", "api-1"];
        let result = agent.recommend("programming", &prompts).await;

        assert!(result.is_ok());
        let set = result.unwrap();
        assert!(set.diversity_score > 0.0);
    }

    #[test]
    fn test_calculate_diversity_score_empty() {
        let agent = RecommenderAgent::new();
        let set = RecommendationSet::new("test");
        let score = agent.calculate_diversity_score(&set);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_diversity_score_varied() {
        let agent = RecommenderAgent::new();
        let mut set = RecommendationSet::new("test");
        set.push(
            Recommendation::new("p1", 0.9)
                .with_category("cat1")
                .with_tags(vec!["tag1".to_string()]),
        );
        set.push(
            Recommendation::new("p2", 0.8)
                .with_category("cat2")
                .with_tags(vec!["tag2".to_string()]),
        );

        let score = agent.calculate_diversity_score(&set);
        assert!(score > 0.5); // High diversity with different categories and tags
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_with_config() {
        let config = RecommendationConfig::thorough();
        let agent = RecommenderAgent::builder().config(config.clone()).build();

        assert_eq!(agent.config.max_recommendations, config.max_recommendations);
    }

    #[test]
    fn test_builder_add_usage() {
        let agent = RecommenderAgent::builder()
            .add_usage("prompt1", 5)
            .add_usage("prompt2", 10)
            .build();

        assert_eq!(*agent.usage_history.get("prompt1").unwrap(), 5);
        assert_eq!(*agent.usage_history.get("prompt2").unwrap(), 10);
    }

    #[test]
    fn test_builder_clamps_values() {
        let agent = RecommenderAgent::builder()
            .diversity_factor(2.0) // Should be clamped to 1.0
            .recency_weight(-1.0) // Should be clamped to 0.0
            .popularity_weight(1.5) // Should be clamped to 1.0
            .build();

        assert!((agent.config.diversity_factor - 1.0).abs() < f64::EPSILON);
        assert!((agent.config.recency_weight - 0.0).abs() < f64::EPSILON);
        assert!((agent.config.popularity_weight - 1.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_whitespace_query() {
        let agent = RecommenderAgent::new();
        let result = agent.recommend("   ", &["prompt1"]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_single_prompt() {
        let agent = RecommenderAgent::new();
        let result = agent.recommend("test", &["single-prompt"]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().len() <= 1);
    }

    #[tokio::test]
    async fn test_duplicate_prompts() {
        let agent = RecommenderAgent::new();
        let prompts = vec!["prompt1", "prompt1", "prompt2"];
        let result = agent.recommend("test", &prompts).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_recommendation_set_average_relevance() {
        let mut set = RecommendationSet::new("test");
        set.push(Recommendation::new("p1", 0.8));
        set.push(Recommendation::new("p2", 0.6));
        set.push(Recommendation::new("p3", 0.4));
        set.update_average_relevance();

        assert!((set.average_relevance - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recommendation_set_average_relevance_empty() {
        let mut set = RecommendationSet::new("test");
        set.update_average_relevance();
        assert!((set.average_relevance - 0.0).abs() < f64::EPSILON);
    }
}
