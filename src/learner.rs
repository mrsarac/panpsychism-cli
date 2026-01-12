//! Learner Agent module for Project Panpsychism.
//!
//! The Knowledge Absorber - learns from interactions, discovers patterns, and improves the system.
//! Like a wise sage who observes and understands the patterns of the universe, this agent
//! absorbs knowledge from every interaction and transforms it into actionable insights.
//!
//! # The Sorcerer's Wand Metaphor
//!
//! In the realm of Project Panpsychism, the Learner Agent acts as the
//! **Knowledge Absorber** - a powerful entity that observes, learns, and grows.
//! Just as a student of the arcane arts studies patterns in magical phenomena,
//! this agent discovers patterns in system behavior and user interactions.
//!
//! ## Learning Domains
//!
//! The absorption process covers multiple domains:
//!
//! - **Query Patterns**: Recurring search and intent patterns
//! - **Error Patterns**: Common failure modes and their causes
//! - **Success Patterns**: Configurations and approaches that work well
//! - **User Behavior**: How users interact with the system
//! - **Performance Trends**: System performance over time
//!
//! # Philosophical Foundation
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Drive toward self-improvement and knowledge accumulation
//! - **RATIO**: Logical pattern extraction and reasoning
//! - **LAETITIA**: Joy through discovery and understanding
//! - **NATURA**: Natural growth of knowledge through observation
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::learner::{LearnerAgent, LearnerConfig};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let learner = LearnerAgent::new();
//!
//!     // Learn from an interaction
//!     learner.learn(&interaction_data).await?;
//!
//!     // Get accumulated insights
//!     let insights = learner.get_insights().await?;
//!
//!     println!("Discovered {} patterns", insights.patterns_discovered.len());
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// PATTERN TYPES
// =============================================================================

/// Types of patterns the Knowledge Absorber can discover.
///
/// Each pattern type represents a different domain of learning:
/// - Query patterns reveal how users seek information
/// - Error patterns expose system weaknesses
/// - Success patterns highlight best practices
/// - User behavior patterns show interaction preferences
/// - Performance anomalies indicate optimization opportunities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Recurring patterns in user queries and search terms.
    QueryPattern,

    /// Common error modes and failure patterns.
    ErrorPattern,

    /// Configurations and approaches that consistently succeed.
    SuccessPattern,

    /// Patterns in how users interact with the system.
    UserBehavior,

    /// Deviations from normal performance metrics.
    PerformanceAnomaly,
}

impl std::fmt::Display for PatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueryPattern => write!(f, "Query Pattern"),
            Self::ErrorPattern => write!(f, "Error Pattern"),
            Self::SuccessPattern => write!(f, "Success Pattern"),
            Self::UserBehavior => write!(f, "User Behavior"),
            Self::PerformanceAnomaly => write!(f, "Performance Anomaly"),
        }
    }
}

impl Default for PatternType {
    fn default() -> Self {
        Self::QueryPattern
    }
}

impl std::str::FromStr for PatternType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "query" | "querypattern" | "query_pattern" => Ok(Self::QueryPattern),
            "error" | "errorpattern" | "error_pattern" => Ok(Self::ErrorPattern),
            "success" | "successpattern" | "success_pattern" => Ok(Self::SuccessPattern),
            "user" | "userbehavior" | "user_behavior" | "behavior" => Ok(Self::UserBehavior),
            "performance" | "performanceanomaly" | "performance_anomaly" | "anomaly" => {
                Ok(Self::PerformanceAnomaly)
            }
            _ => Err(Error::Config(format!(
                "Unknown pattern type: '{}'. Valid types: query, error, success, behavior, anomaly",
                s
            ))),
        }
    }
}

impl PatternType {
    /// Get all pattern types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::QueryPattern,
            Self::ErrorPattern,
            Self::SuccessPattern,
            Self::UserBehavior,
            Self::PerformanceAnomaly,
        ]
    }
}

// =============================================================================
// TREND DIRECTION
// =============================================================================

/// Direction of a performance trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Metrics are improving over time.
    Improving,

    /// Metrics are stable, neither improving nor declining.
    #[default]
    Stable,

    /// Metrics are declining over time.
    Declining,
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Improving => write!(f, "Improving"),
            Self::Stable => write!(f, "Stable"),
            Self::Declining => write!(f, "Declining"),
        }
    }
}

// =============================================================================
// PRIORITY
// =============================================================================

/// Priority level for recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Default, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority - nice to have improvements.
    Low,

    /// Medium priority - should be addressed.
    #[default]
    Medium,

    /// High priority - important improvements.
    High,

    /// Critical priority - must address immediately.
    Critical,
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

// =============================================================================
// PATTERN
// =============================================================================

/// A discovered pattern from system interactions.
///
/// Patterns are like constellations in the night sky - once recognized,
/// they provide guidance and understanding. Each pattern captures a
/// recurring theme that emerges from system observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// The type of pattern discovered.
    pub pattern_type: PatternType,

    /// Confidence score for this pattern (0.0 to 1.0).
    pub confidence: f64,

    /// Number of times this pattern has been observed.
    pub occurrences: usize,

    /// Human-readable description of the pattern.
    pub description: String,

    /// ISO 8601 timestamp of first observation (e.g., "2024-01-15T10:30:00Z").
    pub first_seen: Option<String>,

    /// ISO 8601 timestamp of most recent observation.
    pub last_seen: Option<String>,
}

impl Pattern {
    /// Create a new pattern with basic information.
    pub fn new(pattern_type: PatternType, description: impl Into<String>) -> Self {
        Self {
            pattern_type,
            confidence: 0.5,
            occurrences: 1,
            description: description.into(),
            first_seen: None,
            last_seen: None,
        }
    }

    /// Set the confidence score.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the occurrence count.
    pub fn with_occurrences(mut self, occurrences: usize) -> Self {
        self.occurrences = occurrences;
        self
    }

    /// Set the first seen timestamp.
    pub fn with_first_seen(mut self, timestamp: impl Into<String>) -> Self {
        self.first_seen = Some(timestamp.into());
        self
    }

    /// Set the last seen timestamp.
    pub fn with_last_seen(mut self, timestamp: impl Into<String>) -> Self {
        self.last_seen = Some(timestamp.into());
        self
    }

    /// Check if this is a high-confidence pattern (>= 0.7).
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.7
    }

    /// Check if this is a recurring pattern (>= 3 occurrences).
    pub fn is_recurring(&self) -> bool {
        self.occurrences >= 3
    }

    /// Increment the occurrence count and update last_seen.
    pub fn record_occurrence(&mut self, timestamp: impl Into<String>) {
        self.occurrences += 1;
        self.last_seen = Some(timestamp.into());
        // Increase confidence slightly with each occurrence, capped at 0.95
        self.confidence = (self.confidence + 0.05).min(0.95);
    }
}

impl Default for Pattern {
    fn default() -> Self {
        Self {
            pattern_type: PatternType::default(),
            confidence: 0.5,
            occurrences: 1,
            description: String::new(),
            first_seen: None,
            last_seen: None,
        }
    }
}

// =============================================================================
// KNOWLEDGE UPDATE
// =============================================================================

/// A knowledge update derived from learning.
///
/// Knowledge updates are nuggets of wisdom extracted from observations.
/// They represent actionable insights that can improve system behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeUpdate {
    /// Category of the knowledge (e.g., "Performance", "UX", "Error Handling").
    pub category: String,

    /// The insight or learning.
    pub insight: String,

    /// Number of observations supporting this insight.
    pub evidence_count: usize,

    /// ISO 8601 timestamp when this update was created.
    pub timestamp: String,
}

impl KnowledgeUpdate {
    /// Create a new knowledge update.
    pub fn new(
        category: impl Into<String>,
        insight: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Self {
        Self {
            category: category.into(),
            insight: insight.into(),
            evidence_count: 1,
            timestamp: timestamp.into(),
        }
    }

    /// Increment evidence count.
    pub fn add_evidence(&mut self) {
        self.evidence_count += 1;
    }

    /// Check if this insight is well-supported (>= 5 evidence points).
    pub fn is_well_supported(&self) -> bool {
        self.evidence_count >= 5
    }
}

impl Default for KnowledgeUpdate {
    fn default() -> Self {
        Self {
            category: String::new(),
            insight: String::new(),
            evidence_count: 0,
            timestamp: String::new(),
        }
    }
}

// =============================================================================
// PERFORMANCE TRENDS
// =============================================================================

/// Performance trends observed over time.
///
/// Like a barometer measuring the health of the system,
/// performance trends reveal whether things are getting better or worse.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Average response time in milliseconds.
    pub average_response_time_ms: Option<f64>,

    /// Success rate (0.0 to 1.0).
    pub success_rate: Option<f64>,

    /// Error rate (0.0 to 1.0).
    pub error_rate: Option<f64>,

    /// Overall trend direction.
    pub trend_direction: TrendDirection,
}

impl PerformanceTrends {
    /// Create new performance trends.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set average response time.
    pub fn with_response_time(mut self, ms: f64) -> Self {
        self.average_response_time_ms = Some(ms);
        self
    }

    /// Set success rate.
    pub fn with_success_rate(mut self, rate: f64) -> Self {
        self.success_rate = Some(rate.clamp(0.0, 1.0));
        self
    }

    /// Set error rate.
    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate.clamp(0.0, 1.0));
        self
    }

    /// Set trend direction.
    pub fn with_trend(mut self, direction: TrendDirection) -> Self {
        self.trend_direction = direction;
        self
    }

    /// Check if performance is healthy (success rate >= 0.95, error rate <= 0.05).
    pub fn is_healthy(&self) -> bool {
        let success_ok = self.success_rate.map_or(true, |r| r >= 0.95);
        let error_ok = self.error_rate.map_or(true, |r| r <= 0.05);
        success_ok && error_ok
    }

    /// Calculate overall health score (0.0 to 1.0).
    pub fn health_score(&self) -> f64 {
        let success_score = self.success_rate.unwrap_or(1.0);
        let error_score = 1.0 - self.error_rate.unwrap_or(0.0);
        let trend_score = match self.trend_direction {
            TrendDirection::Improving => 1.0,
            TrendDirection::Stable => 0.8,
            TrendDirection::Declining => 0.5,
        };
        (success_score + error_score + trend_score) / 3.0
    }
}

// =============================================================================
// SYSTEM RECOMMENDATION
// =============================================================================

/// A recommendation for system improvement.
///
/// Recommendations are the practical output of the learning process -
/// actionable suggestions that can improve system performance and user experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemRecommendation {
    /// The recommended action or change.
    pub recommendation: String,

    /// Priority level of this recommendation.
    pub priority: Priority,

    /// Expected benefit of implementing this recommendation.
    pub expected_benefit: String,
}

impl SystemRecommendation {
    /// Create a new system recommendation.
    pub fn new(
        recommendation: impl Into<String>,
        priority: Priority,
        expected_benefit: impl Into<String>,
    ) -> Self {
        Self {
            recommendation: recommendation.into(),
            priority,
            expected_benefit: expected_benefit.into(),
        }
    }

    /// Check if this is a high-priority recommendation.
    pub fn is_high_priority(&self) -> bool {
        matches!(self.priority, Priority::High | Priority::Critical)
    }
}

impl Default for SystemRecommendation {
    fn default() -> Self {
        Self {
            recommendation: String::new(),
            priority: Priority::default(),
            expected_benefit: String::new(),
        }
    }
}

// =============================================================================
// LEARNING INSIGHTS
// =============================================================================

/// Accumulated insights from the learning process.
///
/// Like a treasure chest of wisdom, this structure holds all the
/// valuable insights discovered through observation and analysis.
#[derive(Debug, Clone, Default)]
pub struct LearningInsights {
    /// Patterns discovered in system behavior.
    pub patterns_discovered: Vec<Pattern>,

    /// Knowledge updates derived from learning.
    pub knowledge_updates: Vec<KnowledgeUpdate>,

    /// Performance trends observed.
    pub performance_trends: PerformanceTrends,

    /// Recommendations for system improvement.
    pub recommendations: Vec<SystemRecommendation>,
}

impl LearningInsights {
    /// Create new empty learning insights.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the total number of insights.
    pub fn total_insights(&self) -> usize {
        self.patterns_discovered.len()
            + self.knowledge_updates.len()
            + self.recommendations.len()
    }

    /// Check if there are any high-priority recommendations.
    pub fn has_high_priority_recommendations(&self) -> bool {
        self.recommendations.iter().any(|r| r.is_high_priority())
    }

    /// Get high-confidence patterns only.
    pub fn high_confidence_patterns(&self) -> Vec<&Pattern> {
        self.patterns_discovered
            .iter()
            .filter(|p| p.is_high_confidence())
            .collect()
    }

    /// Get well-supported knowledge updates only.
    pub fn well_supported_updates(&self) -> Vec<&KnowledgeUpdate> {
        self.knowledge_updates
            .iter()
            .filter(|u| u.is_well_supported())
            .collect()
    }

    /// Generate a summary of the insights.
    pub fn summary(&self) -> String {
        format!(
            "Insights: {} patterns, {} knowledge updates, {} recommendations (health: {:.1}%)",
            self.patterns_discovered.len(),
            self.knowledge_updates.len(),
            self.recommendations.len(),
            self.performance_trends.health_score() * 100.0
        )
    }
}

// =============================================================================
// INTERACTION DATA
// =============================================================================

/// Data from a single interaction for learning.
#[derive(Debug, Clone, Default)]
pub struct InteractionData {
    /// Type of interaction (e.g., "search", "synthesis", "validation").
    pub interaction_type: String,

    /// The query or input.
    pub query: String,

    /// Whether the interaction was successful.
    pub success: bool,

    /// Response time in milliseconds.
    pub response_time_ms: u64,

    /// Error message if failed.
    pub error: Option<String>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,

    /// ISO 8601 timestamp of the interaction.
    pub timestamp: String,
}

impl InteractionData {
    /// Create new interaction data.
    pub fn new(interaction_type: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            interaction_type: interaction_type.into(),
            query: query.into(),
            success: true,
            response_time_ms: 0,
            error: None,
            metadata: HashMap::new(),
            timestamp: String::new(),
        }
    }

    /// Mark as successful with response time.
    pub fn with_success(mut self, response_time_ms: u64) -> Self {
        self.success = true;
        self.response_time_ms = response_time_ms;
        self
    }

    /// Mark as failed with error.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.success = false;
        self.error = Some(error.into());
        self
    }

    /// Set the timestamp.
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = timestamp.into();
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// =============================================================================
// LEARNER CONFIGURATION
// =============================================================================

/// Configuration for the Learner Agent.
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Minimum occurrences before considering something a pattern.
    pub min_pattern_occurrences: usize,

    /// Learning rate for confidence adjustments (0.0 to 1.0).
    pub learning_rate: f64,

    /// Maximum patterns to store in memory.
    pub max_patterns: usize,

    /// Maximum knowledge updates to store.
    pub max_knowledge_updates: usize,

    /// Whether to automatically generate recommendations.
    pub auto_recommend: bool,

    /// Minimum confidence threshold for patterns.
    pub confidence_threshold: f64,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            min_pattern_occurrences: 3,
            learning_rate: 0.1,
            max_patterns: 1000,
            max_knowledge_updates: 500,
            auto_recommend: true,
            confidence_threshold: 0.5,
        }
    }
}

impl LearnerConfig {
    /// Create a fast learning configuration.
    pub fn fast() -> Self {
        Self {
            min_pattern_occurrences: 2,
            learning_rate: 0.2,
            max_patterns: 500,
            max_knowledge_updates: 250,
            auto_recommend: true,
            confidence_threshold: 0.4,
        }
    }

    /// Create a conservative learning configuration.
    pub fn conservative() -> Self {
        Self {
            min_pattern_occurrences: 5,
            learning_rate: 0.05,
            max_patterns: 2000,
            max_knowledge_updates: 1000,
            auto_recommend: true,
            confidence_threshold: 0.7,
        }
    }
}

// =============================================================================
// LEARNER STORAGE
// =============================================================================

/// Internal storage for the learner agent.
#[derive(Debug, Default)]
struct LearnerStorage {
    /// Patterns indexed by a key (pattern_type + description hash).
    patterns: HashMap<String, Pattern>,

    /// Knowledge updates indexed by category.
    knowledge: HashMap<String, Vec<KnowledgeUpdate>>,

    /// Recent response times for trend calculation.
    response_times: Vec<u64>,

    /// Success/failure counts.
    success_count: usize,
    failure_count: usize,

    /// Query frequency map for pattern detection.
    query_frequency: HashMap<String, usize>,

    /// Error frequency map.
    error_frequency: HashMap<String, usize>,
}

impl LearnerStorage {
    fn new() -> Self {
        Self::default()
    }
}

// =============================================================================
// LEARNER AGENT
// =============================================================================

/// The Knowledge Absorber - Agent 20 of Project Panpsychism.
///
/// This agent learns from interactions, discovers patterns, and generates
/// insights to improve system behavior. Like a wise sage who observes and
/// understands, it transforms raw observations into actionable knowledge.
///
/// # Architecture
///
/// The LearnerAgent uses thread-safe storage for pattern and knowledge
/// accumulation, allowing concurrent learning from multiple sources.
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::learner::{LearnerAgent, InteractionData};
///
/// let learner = LearnerAgent::new();
///
/// // Learn from interactions
/// let data = InteractionData::new("search", "OAuth2 authentication")
///     .with_success(150)
///     .with_timestamp("2024-01-15T10:30:00Z");
///
/// learner.learn(&data).await?;
///
/// // Get insights
/// let insights = learner.get_insights().await?;
/// println!("{}", insights.summary());
/// ```
#[derive(Debug, Clone)]
pub struct LearnerAgent {
    /// Configuration for learning behavior.
    config: LearnerConfig,

    /// Thread-safe storage for patterns and knowledge.
    storage: Arc<RwLock<LearnerStorage>>,
}

impl Default for LearnerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl LearnerAgent {
    /// Create a new Learner Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: LearnerConfig::default(),
            storage: Arc::new(RwLock::new(LearnerStorage::new())),
        }
    }

    /// Create a new Learner Agent with custom configuration.
    pub fn with_config(config: LearnerConfig) -> Self {
        Self {
            config,
            storage: Arc::new(RwLock::new(LearnerStorage::new())),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> LearnerAgentBuilder {
        LearnerAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &LearnerConfig {
        &self.config
    }

    // =========================================================================
    // LEARNING METHODS
    // =========================================================================

    /// Learn from an interaction.
    ///
    /// This is the primary learning method. It processes interaction data
    /// and updates internal patterns and knowledge accordingly.
    ///
    /// # Arguments
    ///
    /// * `data` - The interaction data to learn from
    ///
    /// # Returns
    ///
    /// Ok(()) if learning was successful, or an error if processing failed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let data = InteractionData::new("search", "implement rate limiting")
    ///     .with_success(200)
    ///     .with_timestamp("2024-01-15T10:30:00Z");
    ///
    /// learner.learn(&data).await?;
    /// ```
    pub async fn learn(&self, data: &InteractionData) -> Result<()> {
        let start = Instant::now();

        if data.query.trim().is_empty() && data.interaction_type.trim().is_empty() {
            return Err(Error::Validation(
                "Cannot learn from empty interaction data".to_string(),
            ));
        }

        // Update storage with new observation
        {
            let mut storage = self.storage.write().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to acquire write lock: {}", e),
                }
            })?;

            // Track response times
            if data.response_time_ms > 0 {
                storage.response_times.push(data.response_time_ms);
                // Keep only last 1000 response times
                if storage.response_times.len() > 1000 {
                    storage.response_times.remove(0);
                }
            }

            // Track success/failure
            if data.success {
                storage.success_count += 1;
            } else {
                storage.failure_count += 1;
            }

            // Track query frequency
            if !data.query.is_empty() {
                let normalized_query = self.normalize_query(&data.query);
                *storage.query_frequency.entry(normalized_query).or_insert(0) += 1;
            }

            // Track error frequency
            if let Some(ref error) = data.error {
                let error_key = self.extract_error_key(error);
                *storage.error_frequency.entry(error_key).or_insert(0) += 1;
            }

            // Detect and record patterns
            self.detect_patterns(&mut storage, data)?;
        }

        let duration = start.elapsed().as_millis() as u64;
        debug!("Learned from interaction in {}ms", duration);

        Ok(())
    }

    /// Get accumulated learning insights.
    ///
    /// Returns all patterns, knowledge updates, performance trends,
    /// and recommendations derived from the learning process.
    ///
    /// # Returns
    ///
    /// A `LearningInsights` structure containing all accumulated insights.
    pub async fn get_insights(&self) -> Result<LearningInsights> {
        let storage = self.storage.read().map_err(|e| {
            Error::Internal {
                message: format!("Failed to acquire read lock: {}", e),
            }
        })?;

        let patterns_discovered: Vec<Pattern> = storage
            .patterns
            .values()
            .filter(|p| p.confidence >= self.config.confidence_threshold)
            .cloned()
            .collect();

        let knowledge_updates: Vec<KnowledgeUpdate> = storage
            .knowledge
            .values()
            .flatten()
            .cloned()
            .collect();

        let performance_trends = self.calculate_trends(&storage);

        let recommendations = if self.config.auto_recommend {
            self.generate_recommendations(&storage, &patterns_discovered, &performance_trends)
        } else {
            Vec::new()
        };

        Ok(LearningInsights {
            patterns_discovered,
            knowledge_updates,
            performance_trends,
            recommendations,
        })
    }

    /// Reset learning state (clear all patterns and knowledge).
    pub fn reset(&self) -> Result<()> {
        let mut storage = self.storage.write().map_err(|e| {
            Error::Internal {
                message: format!("Failed to acquire write lock: {}", e),
            }
        })?;

        storage.patterns.clear();
        storage.knowledge.clear();
        storage.response_times.clear();
        storage.success_count = 0;
        storage.failure_count = 0;
        storage.query_frequency.clear();
        storage.error_frequency.clear();

        info!("Learner state reset");
        Ok(())
    }

    /// Get the number of patterns currently stored.
    pub fn pattern_count(&self) -> usize {
        self.storage
            .read()
            .map(|s| s.patterns.len())
            .unwrap_or(0)
    }

    /// Get the number of interactions processed.
    pub fn interaction_count(&self) -> usize {
        self.storage
            .read()
            .map(|s| s.success_count + s.failure_count)
            .unwrap_or(0)
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Normalize a query for pattern matching.
    fn normalize_query(&self, query: &str) -> String {
        query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract a key from an error message for pattern matching.
    fn extract_error_key(&self, error: &str) -> String {
        // Extract error code if present, otherwise use first 50 chars
        if let Some(code_start) = error.find('[') {
            if let Some(code_end) = error.find(']') {
                return error[code_start..=code_end].to_string();
            }
        }
        error.chars().take(50).collect()
    }

    /// Detect patterns from interaction data.
    fn detect_patterns(
        &self,
        storage: &mut LearnerStorage,
        data: &InteractionData,
    ) -> Result<()> {
        let timestamp = if data.timestamp.is_empty() {
            "unknown".to_string()
        } else {
            data.timestamp.clone()
        };

        // Detect query patterns
        if !data.query.is_empty() {
            let normalized = self.normalize_query(&data.query);
            if let Some(&count) = storage.query_frequency.get(&normalized) {
                if count >= self.config.min_pattern_occurrences {
                    let key = format!("query:{}", normalized);
                    storage
                        .patterns
                        .entry(key)
                        .and_modify(|p| p.record_occurrence(&timestamp))
                        .or_insert_with(|| {
                            Pattern::new(PatternType::QueryPattern, format!("Recurring query: {}", normalized))
                                .with_first_seen(&timestamp)
                                .with_last_seen(&timestamp)
                                .with_occurrences(count)
                        });
                }
            }
        }

        // Detect error patterns
        if let Some(ref error) = data.error {
            let error_key = self.extract_error_key(error);
            if let Some(&count) = storage.error_frequency.get(&error_key) {
                if count >= self.config.min_pattern_occurrences {
                    let key = format!("error:{}", error_key);
                    storage
                        .patterns
                        .entry(key)
                        .and_modify(|p| p.record_occurrence(&timestamp))
                        .or_insert_with(|| {
                            Pattern::new(PatternType::ErrorPattern, format!("Recurring error: {}", error_key))
                                .with_first_seen(&timestamp)
                                .with_last_seen(&timestamp)
                                .with_occurrences(count)
                        });
                }
            }
        }

        // Detect success patterns
        if data.success && data.response_time_ms < 100 {
            let key = format!("success:fast:{}", data.interaction_type);
            storage
                .patterns
                .entry(key)
                .and_modify(|p| p.record_occurrence(&timestamp))
                .or_insert_with(|| {
                    Pattern::new(
                        PatternType::SuccessPattern,
                        format!("Fast successful {}", data.interaction_type),
                    )
                    .with_first_seen(&timestamp)
                    .with_last_seen(&timestamp)
                });
        }

        // Detect performance anomalies (very slow responses)
        if data.response_time_ms > 5000 {
            let key = format!("anomaly:slow:{}", data.interaction_type);
            storage
                .patterns
                .entry(key)
                .and_modify(|p| p.record_occurrence(&timestamp))
                .or_insert_with(|| {
                    Pattern::new(
                        PatternType::PerformanceAnomaly,
                        format!("Slow {} (>5s)", data.interaction_type),
                    )
                    .with_first_seen(&timestamp)
                    .with_last_seen(&timestamp)
                });
        }

        // Enforce max patterns limit
        if storage.patterns.len() > self.config.max_patterns {
            // Remove lowest confidence patterns
            let mut patterns: Vec<_> = storage.patterns.drain().collect();
            patterns.sort_by(|a, b| {
                b.1.confidence
                    .partial_cmp(&a.1.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            patterns.truncate(self.config.max_patterns);
            storage.patterns = patterns.into_iter().collect();
        }

        Ok(())
    }

    /// Calculate performance trends from stored data.
    fn calculate_trends(&self, storage: &LearnerStorage) -> PerformanceTrends {
        let mut trends = PerformanceTrends::new();

        // Calculate average response time
        if !storage.response_times.is_empty() {
            let sum: u64 = storage.response_times.iter().sum();
            let avg = sum as f64 / storage.response_times.len() as f64;
            trends = trends.with_response_time(avg);
        }

        // Calculate success/error rates
        let total = storage.success_count + storage.failure_count;
        if total > 0 {
            let success_rate = storage.success_count as f64 / total as f64;
            let error_rate = storage.failure_count as f64 / total as f64;
            trends = trends.with_success_rate(success_rate).with_error_rate(error_rate);
        }

        // Determine trend direction based on recent vs older response times
        if storage.response_times.len() >= 20 {
            let mid = storage.response_times.len() / 2;
            let older_avg: f64 = storage.response_times[..mid].iter().sum::<u64>() as f64 / mid as f64;
            let recent_avg: f64 = storage.response_times[mid..].iter().sum::<u64>() as f64
                / (storage.response_times.len() - mid) as f64;

            let diff_ratio = (recent_avg - older_avg) / older_avg.max(1.0);
            trends.trend_direction = if diff_ratio < -0.1 {
                TrendDirection::Improving
            } else if diff_ratio > 0.1 {
                TrendDirection::Declining
            } else {
                TrendDirection::Stable
            };
        }

        trends
    }

    /// Generate recommendations based on patterns and trends.
    fn generate_recommendations(
        &self,
        storage: &LearnerStorage,
        patterns: &[Pattern],
        trends: &PerformanceTrends,
    ) -> Vec<SystemRecommendation> {
        let mut recommendations = Vec::new();

        // Recommend based on error patterns
        let error_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::ErrorPattern && p.is_recurring())
            .collect();

        if !error_patterns.is_empty() {
            recommendations.push(SystemRecommendation::new(
                format!("Address {} recurring error patterns", error_patterns.len()),
                Priority::High,
                "Reduce error rate and improve reliability",
            ));
        }

        // Recommend based on performance anomalies
        let anomalies: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::PerformanceAnomaly && p.is_recurring())
            .collect();

        if !anomalies.is_empty() {
            recommendations.push(SystemRecommendation::new(
                format!("Investigate {} performance anomalies", anomalies.len()),
                Priority::Medium,
                "Improve response times and user experience",
            ));
        }

        // Recommend based on declining trends
        if trends.trend_direction == TrendDirection::Declining {
            recommendations.push(SystemRecommendation::new(
                "Performance is declining - investigate root causes",
                Priority::Critical,
                "Prevent further degradation",
            ));
        }

        // Recommend based on error rate
        if let Some(error_rate) = trends.error_rate {
            if error_rate > 0.1 {
                recommendations.push(SystemRecommendation::new(
                    format!("Error rate ({:.1}%) is above 10% threshold", error_rate * 100.0),
                    Priority::High,
                    "Reduce errors to improve reliability",
                ));
            }
        }

        // Recommend caching for frequently repeated queries
        let frequent_queries: Vec<_> = storage
            .query_frequency
            .iter()
            .filter(|(_, &count)| count >= 10)
            .collect();

        if !frequent_queries.is_empty() {
            recommendations.push(SystemRecommendation::new(
                format!("Consider caching {} frequently repeated queries", frequent_queries.len()),
                Priority::Low,
                "Improve response times for common queries",
            ));
        }

        recommendations
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom LearnerAgent configuration.
#[derive(Debug, Default)]
pub struct LearnerAgentBuilder {
    config: Option<LearnerConfig>,
}

impl LearnerAgentBuilder {
    /// Set the configuration.
    pub fn config(mut self, config: LearnerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set minimum pattern occurrences.
    pub fn min_pattern_occurrences(mut self, min: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.min_pattern_occurrences = min;
        self.config = Some(config);
        self
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, rate: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.learning_rate = rate.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set maximum patterns.
    pub fn max_patterns(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_patterns = max;
        self.config = Some(config);
        self
    }

    /// Set confidence threshold.
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Enable or disable auto recommendations.
    pub fn auto_recommend(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.auto_recommend = enabled;
        self.config = Some(config);
        self
    }

    /// Build the LearnerAgent.
    pub fn build(self) -> LearnerAgent {
        LearnerAgent {
            config: self.config.unwrap_or_default(),
            storage: Arc::new(RwLock::new(LearnerStorage::new())),
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
    // PatternType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pattern_type_display() {
        assert_eq!(PatternType::QueryPattern.to_string(), "Query Pattern");
        assert_eq!(PatternType::ErrorPattern.to_string(), "Error Pattern");
        assert_eq!(PatternType::SuccessPattern.to_string(), "Success Pattern");
        assert_eq!(PatternType::UserBehavior.to_string(), "User Behavior");
        assert_eq!(PatternType::PerformanceAnomaly.to_string(), "Performance Anomaly");
    }

    #[test]
    fn test_pattern_type_from_str() {
        assert_eq!("query".parse::<PatternType>().unwrap(), PatternType::QueryPattern);
        assert_eq!("error".parse::<PatternType>().unwrap(), PatternType::ErrorPattern);
        assert_eq!("success".parse::<PatternType>().unwrap(), PatternType::SuccessPattern);
        assert_eq!("behavior".parse::<PatternType>().unwrap(), PatternType::UserBehavior);
        assert_eq!("anomaly".parse::<PatternType>().unwrap(), PatternType::PerformanceAnomaly);
    }

    #[test]
    fn test_pattern_type_from_str_aliases() {
        assert_eq!("query_pattern".parse::<PatternType>().unwrap(), PatternType::QueryPattern);
        assert_eq!("error_pattern".parse::<PatternType>().unwrap(), PatternType::ErrorPattern);
        assert_eq!("user_behavior".parse::<PatternType>().unwrap(), PatternType::UserBehavior);
        assert_eq!("performance_anomaly".parse::<PatternType>().unwrap(), PatternType::PerformanceAnomaly);
    }

    #[test]
    fn test_pattern_type_from_str_invalid() {
        let result = "invalid".parse::<PatternType>();
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_type_all() {
        let all = PatternType::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&PatternType::QueryPattern));
        assert!(all.contains(&PatternType::ErrorPattern));
        assert!(all.contains(&PatternType::SuccessPattern));
        assert!(all.contains(&PatternType::UserBehavior));
        assert!(all.contains(&PatternType::PerformanceAnomaly));
    }

    #[test]
    fn test_pattern_type_default() {
        assert_eq!(PatternType::default(), PatternType::QueryPattern);
    }

    // -------------------------------------------------------------------------
    // TrendDirection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trend_direction_display() {
        assert_eq!(TrendDirection::Improving.to_string(), "Improving");
        assert_eq!(TrendDirection::Stable.to_string(), "Stable");
        assert_eq!(TrendDirection::Declining.to_string(), "Declining");
    }

    #[test]
    fn test_trend_direction_default() {
        assert_eq!(TrendDirection::default(), TrendDirection::Stable);
    }

    // -------------------------------------------------------------------------
    // Priority Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_priority_display() {
        assert_eq!(Priority::Low.to_string(), "Low");
        assert_eq!(Priority::Medium.to_string(), "Medium");
        assert_eq!(Priority::High.to_string(), "High");
        assert_eq!(Priority::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low < Priority::Medium);
        assert!(Priority::Medium < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Medium);
    }

    // -------------------------------------------------------------------------
    // Pattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pattern_new() {
        let pattern = Pattern::new(PatternType::QueryPattern, "Test pattern");
        assert_eq!(pattern.pattern_type, PatternType::QueryPattern);
        assert_eq!(pattern.description, "Test pattern");
        assert_eq!(pattern.occurrences, 1);
        assert!((pattern.confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pattern_builder() {
        let pattern = Pattern::new(PatternType::ErrorPattern, "Error")
            .with_confidence(0.8)
            .with_occurrences(5)
            .with_first_seen("2024-01-01T00:00:00Z")
            .with_last_seen("2024-01-15T00:00:00Z");

        assert!((pattern.confidence - 0.8).abs() < f64::EPSILON);
        assert_eq!(pattern.occurrences, 5);
        assert_eq!(pattern.first_seen, Some("2024-01-01T00:00:00Z".to_string()));
        assert_eq!(pattern.last_seen, Some("2024-01-15T00:00:00Z".to_string()));
    }

    #[test]
    fn test_pattern_confidence_clamping() {
        let pattern_high = Pattern::new(PatternType::QueryPattern, "Test").with_confidence(1.5);
        assert!((pattern_high.confidence - 1.0).abs() < f64::EPSILON);

        let pattern_low = Pattern::new(PatternType::QueryPattern, "Test").with_confidence(-0.5);
        assert!(pattern_low.confidence.abs() < f64::EPSILON);
    }

    #[test]
    fn test_pattern_is_high_confidence() {
        let high = Pattern::new(PatternType::QueryPattern, "Test").with_confidence(0.8);
        assert!(high.is_high_confidence());

        let low = Pattern::new(PatternType::QueryPattern, "Test").with_confidence(0.5);
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_pattern_is_recurring() {
        let recurring = Pattern::new(PatternType::QueryPattern, "Test").with_occurrences(5);
        assert!(recurring.is_recurring());

        let single = Pattern::new(PatternType::QueryPattern, "Test").with_occurrences(1);
        assert!(!single.is_recurring());
    }

    #[test]
    fn test_pattern_record_occurrence() {
        let mut pattern = Pattern::new(PatternType::QueryPattern, "Test");
        let initial_confidence = pattern.confidence;
        let initial_occurrences = pattern.occurrences;

        pattern.record_occurrence("2024-01-15T10:00:00Z");

        assert_eq!(pattern.occurrences, initial_occurrences + 1);
        assert!(pattern.confidence > initial_confidence);
        assert_eq!(pattern.last_seen, Some("2024-01-15T10:00:00Z".to_string()));
    }

    #[test]
    fn test_pattern_default() {
        let pattern = Pattern::default();
        assert_eq!(pattern.pattern_type, PatternType::QueryPattern);
        assert!(pattern.description.is_empty());
    }

    // -------------------------------------------------------------------------
    // KnowledgeUpdate Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_knowledge_update_new() {
        let update = KnowledgeUpdate::new("Performance", "Caching improves speed", "2024-01-15T00:00:00Z");
        assert_eq!(update.category, "Performance");
        assert_eq!(update.insight, "Caching improves speed");
        assert_eq!(update.evidence_count, 1);
    }

    #[test]
    fn test_knowledge_update_add_evidence() {
        let mut update = KnowledgeUpdate::new("UX", "Users prefer dark mode", "2024-01-15T00:00:00Z");
        update.add_evidence();
        update.add_evidence();
        assert_eq!(update.evidence_count, 3);
    }

    #[test]
    fn test_knowledge_update_is_well_supported() {
        let mut update = KnowledgeUpdate::new("Test", "Insight", "2024-01-15T00:00:00Z");
        assert!(!update.is_well_supported());

        for _ in 0..4 {
            update.add_evidence();
        }
        assert!(update.is_well_supported());
    }

    #[test]
    fn test_knowledge_update_default() {
        let update = KnowledgeUpdate::default();
        assert!(update.category.is_empty());
        assert!(update.insight.is_empty());
        assert_eq!(update.evidence_count, 0);
    }

    // -------------------------------------------------------------------------
    // PerformanceTrends Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_performance_trends_new() {
        let trends = PerformanceTrends::new();
        assert!(trends.average_response_time_ms.is_none());
        assert!(trends.success_rate.is_none());
        assert!(trends.error_rate.is_none());
        assert_eq!(trends.trend_direction, TrendDirection::Stable);
    }

    #[test]
    fn test_performance_trends_builder() {
        let trends = PerformanceTrends::new()
            .with_response_time(150.0)
            .with_success_rate(0.95)
            .with_error_rate(0.05)
            .with_trend(TrendDirection::Improving);

        assert!((trends.average_response_time_ms.unwrap() - 150.0).abs() < f64::EPSILON);
        assert!((trends.success_rate.unwrap() - 0.95).abs() < f64::EPSILON);
        assert!((trends.error_rate.unwrap() - 0.05).abs() < f64::EPSILON);
        assert_eq!(trends.trend_direction, TrendDirection::Improving);
    }

    #[test]
    fn test_performance_trends_rate_clamping() {
        let trends = PerformanceTrends::new()
            .with_success_rate(1.5)
            .with_error_rate(-0.1);

        assert!((trends.success_rate.unwrap() - 1.0).abs() < f64::EPSILON);
        assert!(trends.error_rate.unwrap().abs() < f64::EPSILON);
    }

    #[test]
    fn test_performance_trends_is_healthy() {
        let healthy = PerformanceTrends::new()
            .with_success_rate(0.98)
            .with_error_rate(0.02);
        assert!(healthy.is_healthy());

        let unhealthy = PerformanceTrends::new()
            .with_success_rate(0.85)
            .with_error_rate(0.15);
        assert!(!unhealthy.is_healthy());
    }

    #[test]
    fn test_performance_trends_health_score() {
        let perfect = PerformanceTrends::new()
            .with_success_rate(1.0)
            .with_error_rate(0.0)
            .with_trend(TrendDirection::Improving);
        assert!((perfect.health_score() - 1.0).abs() < f64::EPSILON);

        let mixed = PerformanceTrends::new()
            .with_success_rate(0.8)
            .with_error_rate(0.2)
            .with_trend(TrendDirection::Stable);
        assert!(mixed.health_score() < 1.0);
        assert!(mixed.health_score() > 0.5);
    }

    // -------------------------------------------------------------------------
    // SystemRecommendation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_system_recommendation_new() {
        let rec = SystemRecommendation::new("Add caching", Priority::High, "Faster responses");
        assert_eq!(rec.recommendation, "Add caching");
        assert_eq!(rec.priority, Priority::High);
        assert_eq!(rec.expected_benefit, "Faster responses");
    }

    #[test]
    fn test_system_recommendation_is_high_priority() {
        let high = SystemRecommendation::new("Test", Priority::High, "Benefit");
        assert!(high.is_high_priority());

        let critical = SystemRecommendation::new("Test", Priority::Critical, "Benefit");
        assert!(critical.is_high_priority());

        let medium = SystemRecommendation::new("Test", Priority::Medium, "Benefit");
        assert!(!medium.is_high_priority());

        let low = SystemRecommendation::new("Test", Priority::Low, "Benefit");
        assert!(!low.is_high_priority());
    }

    #[test]
    fn test_system_recommendation_default() {
        let rec = SystemRecommendation::default();
        assert!(rec.recommendation.is_empty());
        assert_eq!(rec.priority, Priority::Medium);
    }

    // -------------------------------------------------------------------------
    // LearningInsights Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learning_insights_new() {
        let insights = LearningInsights::new();
        assert!(insights.patterns_discovered.is_empty());
        assert!(insights.knowledge_updates.is_empty());
        assert!(insights.recommendations.is_empty());
    }

    #[test]
    fn test_learning_insights_total_insights() {
        let mut insights = LearningInsights::new();
        assert_eq!(insights.total_insights(), 0);

        insights.patterns_discovered.push(Pattern::new(PatternType::QueryPattern, "Test"));
        insights.knowledge_updates.push(KnowledgeUpdate::new("Cat", "Insight", "ts"));
        insights.recommendations.push(SystemRecommendation::new("Rec", Priority::Low, "Ben"));

        assert_eq!(insights.total_insights(), 3);
    }

    #[test]
    fn test_learning_insights_has_high_priority_recommendations() {
        let mut insights = LearningInsights::new();
        assert!(!insights.has_high_priority_recommendations());

        insights.recommendations.push(SystemRecommendation::new("Test", Priority::Low, "Benefit"));
        assert!(!insights.has_high_priority_recommendations());

        insights.recommendations.push(SystemRecommendation::new("Test", Priority::High, "Benefit"));
        assert!(insights.has_high_priority_recommendations());
    }

    #[test]
    fn test_learning_insights_high_confidence_patterns() {
        let mut insights = LearningInsights::new();
        insights.patterns_discovered.push(
            Pattern::new(PatternType::QueryPattern, "High").with_confidence(0.8),
        );
        insights.patterns_discovered.push(
            Pattern::new(PatternType::ErrorPattern, "Low").with_confidence(0.3),
        );

        let high_conf = insights.high_confidence_patterns();
        assert_eq!(high_conf.len(), 1);
        assert_eq!(high_conf[0].description, "High");
    }

    #[test]
    fn test_learning_insights_well_supported_updates() {
        let mut insights = LearningInsights::new();

        let mut supported = KnowledgeUpdate::new("Cat", "Insight", "ts");
        for _ in 0..5 {
            supported.add_evidence();
        }
        insights.knowledge_updates.push(supported);

        insights.knowledge_updates.push(KnowledgeUpdate::new("Cat2", "Insight2", "ts2"));

        let well_supported = insights.well_supported_updates();
        assert_eq!(well_supported.len(), 1);
    }

    #[test]
    fn test_learning_insights_summary() {
        let insights = LearningInsights::new();
        let summary = insights.summary();
        assert!(summary.contains("0 patterns"));
        assert!(summary.contains("0 knowledge updates"));
        assert!(summary.contains("0 recommendations"));
    }

    // -------------------------------------------------------------------------
    // InteractionData Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_interaction_data_new() {
        let data = InteractionData::new("search", "OAuth2 authentication");
        assert_eq!(data.interaction_type, "search");
        assert_eq!(data.query, "OAuth2 authentication");
        assert!(data.success);
    }

    #[test]
    fn test_interaction_data_with_success() {
        let data = InteractionData::new("search", "query").with_success(150);
        assert!(data.success);
        assert_eq!(data.response_time_ms, 150);
    }

    #[test]
    fn test_interaction_data_with_error() {
        let data = InteractionData::new("search", "query").with_error("Connection failed");
        assert!(!data.success);
        assert_eq!(data.error, Some("Connection failed".to_string()));
    }

    #[test]
    fn test_interaction_data_with_timestamp() {
        let data = InteractionData::new("search", "query")
            .with_timestamp("2024-01-15T10:30:00Z");
        assert_eq!(data.timestamp, "2024-01-15T10:30:00Z");
    }

    #[test]
    fn test_interaction_data_with_metadata() {
        let data = InteractionData::new("search", "query")
            .with_metadata("user_id", "123")
            .with_metadata("session_id", "abc");
        assert_eq!(data.metadata.get("user_id"), Some(&"123".to_string()));
        assert_eq!(data.metadata.get("session_id"), Some(&"abc".to_string()));
    }

    // -------------------------------------------------------------------------
    // LearnerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_config_default() {
        let config = LearnerConfig::default();
        assert_eq!(config.min_pattern_occurrences, 3);
        assert!((config.learning_rate - 0.1).abs() < f64::EPSILON);
        assert_eq!(config.max_patterns, 1000);
        assert_eq!(config.max_knowledge_updates, 500);
        assert!(config.auto_recommend);
        assert!((config.confidence_threshold - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_learner_config_fast() {
        let config = LearnerConfig::fast();
        assert_eq!(config.min_pattern_occurrences, 2);
        assert!((config.learning_rate - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.max_patterns, 500);
    }

    #[test]
    fn test_learner_config_conservative() {
        let config = LearnerConfig::conservative();
        assert_eq!(config.min_pattern_occurrences, 5);
        assert!((config.learning_rate - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.max_patterns, 2000);
    }

    // -------------------------------------------------------------------------
    // LearnerAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_agent_new() {
        let agent = LearnerAgent::new();
        assert_eq!(agent.config().min_pattern_occurrences, 3);
        assert_eq!(agent.pattern_count(), 0);
        assert_eq!(agent.interaction_count(), 0);
    }

    #[test]
    fn test_learner_agent_with_config() {
        let config = LearnerConfig {
            min_pattern_occurrences: 5,
            ..Default::default()
        };
        let agent = LearnerAgent::with_config(config);
        assert_eq!(agent.config().min_pattern_occurrences, 5);
    }

    #[test]
    fn test_learner_agent_builder() {
        let agent = LearnerAgent::builder()
            .min_pattern_occurrences(2)
            .learning_rate(0.15)
            .max_patterns(500)
            .confidence_threshold(0.6)
            .auto_recommend(false)
            .build();

        assert_eq!(agent.config().min_pattern_occurrences, 2);
        assert!((agent.config().learning_rate - 0.15).abs() < f64::EPSILON);
        assert_eq!(agent.config().max_patterns, 500);
        assert!((agent.config().confidence_threshold - 0.6).abs() < f64::EPSILON);
        assert!(!agent.config().auto_recommend);
    }

    #[test]
    fn test_learner_agent_builder_rate_clamping() {
        let agent = LearnerAgent::builder()
            .learning_rate(1.5)
            .confidence_threshold(-0.1)
            .build();

        assert!((agent.config().learning_rate - 1.0).abs() < f64::EPSILON);
        assert!(agent.config().confidence_threshold.abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_learner_agent_learn_empty_data() {
        let agent = LearnerAgent::new();
        let data = InteractionData::default();
        let result = agent.learn(&data).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_learner_agent_learn_success() {
        let agent = LearnerAgent::new();
        let data = InteractionData::new("search", "OAuth2 authentication")
            .with_success(150)
            .with_timestamp("2024-01-15T10:30:00Z");

        let result = agent.learn(&data).await;
        assert!(result.is_ok());
        assert_eq!(agent.interaction_count(), 1);
    }

    #[tokio::test]
    async fn test_learner_agent_learn_with_error() {
        let agent = LearnerAgent::new();
        let data = InteractionData::new("search", "test query")
            .with_error("[E030] Connection failed")
            .with_timestamp("2024-01-15T10:30:00Z");

        let result = agent.learn(&data).await;
        assert!(result.is_ok());
        assert_eq!(agent.interaction_count(), 1);
    }

    #[tokio::test]
    async fn test_learner_agent_get_insights() {
        let agent = LearnerAgent::new();

        // Learn from a few interactions
        for i in 0..5 {
            let data = InteractionData::new("search", "repeated query")
                .with_success(100 + i * 10)
                .with_timestamp(format!("2024-01-15T10:{:02}:00Z", i));
            agent.learn(&data).await.unwrap();
        }

        let insights = agent.get_insights().await.unwrap();
        assert!(insights.performance_trends.average_response_time_ms.is_some());
        assert!(insights.performance_trends.success_rate.is_some());
    }

    #[tokio::test]
    async fn test_learner_agent_pattern_detection() {
        let agent = LearnerAgent::builder()
            .min_pattern_occurrences(3)
            .build();

        // Send the same query multiple times to trigger pattern detection
        for i in 0..5 {
            let data = InteractionData::new("search", "OAuth2 authentication")
                .with_success(100)
                .with_timestamp(format!("2024-01-15T10:{:02}:00Z", i));
            agent.learn(&data).await.unwrap();
        }

        // Patterns should be detected after enough occurrences
        let insights = agent.get_insights().await.unwrap();
        assert!(insights.patterns_discovered.len() > 0);
    }

    #[tokio::test]
    async fn test_learner_agent_error_pattern_detection() {
        let agent = LearnerAgent::builder()
            .min_pattern_occurrences(3)
            .build();

        // Send the same error multiple times
        for i in 0..5 {
            let data = InteractionData::new("api_call", "test")
                .with_error("[E030] Connection timeout")
                .with_timestamp(format!("2024-01-15T10:{:02}:00Z", i));
            agent.learn(&data).await.unwrap();
        }

        let insights = agent.get_insights().await.unwrap();
        let error_patterns: Vec<_> = insights
            .patterns_discovered
            .iter()
            .filter(|p| p.pattern_type == PatternType::ErrorPattern)
            .collect();
        assert!(!error_patterns.is_empty());
    }

    #[tokio::test]
    async fn test_learner_agent_performance_anomaly_detection() {
        let agent = LearnerAgent::new();

        // Send a very slow request
        let data = InteractionData::new("search", "slow query")
            .with_success(6000) // 6 seconds
            .with_timestamp("2024-01-15T10:00:00Z");
        agent.learn(&data).await.unwrap();

        let insights = agent.get_insights().await.unwrap();
        let anomalies: Vec<_> = insights
            .patterns_discovered
            .iter()
            .filter(|p| p.pattern_type == PatternType::PerformanceAnomaly)
            .collect();
        assert!(!anomalies.is_empty());
    }

    #[tokio::test]
    async fn test_learner_agent_recommendations_generation() {
        let agent = LearnerAgent::builder()
            .min_pattern_occurrences(2)
            .auto_recommend(true)
            .build();

        // Send multiple errors to trigger recommendations
        for i in 0..5 {
            let data = InteractionData::new("api_call", "test")
                .with_error("[E030] Connection timeout")
                .with_timestamp(format!("2024-01-15T10:{:02}:00Z", i));
            agent.learn(&data).await.unwrap();
        }

        let insights = agent.get_insights().await.unwrap();
        // Should have recommendations about error rate
        assert!(!insights.recommendations.is_empty());
    }

    #[test]
    fn test_learner_agent_reset() {
        let agent = LearnerAgent::new();

        // Add some data
        {
            let mut storage = agent.storage.write().unwrap();
            storage.success_count = 10;
            storage.failure_count = 5;
            storage.query_frequency.insert("test".to_string(), 5);
        }

        assert!(agent.interaction_count() > 0);

        // Reset
        agent.reset().unwrap();

        assert_eq!(agent.interaction_count(), 0);
        assert_eq!(agent.pattern_count(), 0);
    }

    #[tokio::test]
    async fn test_learner_agent_trend_calculation() {
        let agent = LearnerAgent::new();

        // Add response times that show improvement
        for i in 0..30 {
            let response_time = if i < 15 { 500 } else { 100 }; // Faster recently
            let data = InteractionData::new("search", "query")
                .with_success(response_time)
                .with_timestamp(format!("2024-01-15T10:{:02}:00Z", i));
            agent.learn(&data).await.unwrap();
        }

        let insights = agent.get_insights().await.unwrap();
        assert_eq!(insights.performance_trends.trend_direction, TrendDirection::Improving);
    }

    #[test]
    fn test_learner_agent_normalize_query() {
        let agent = LearnerAgent::new();
        assert_eq!(agent.normalize_query("  Hello   World  "), "hello world");
        assert_eq!(agent.normalize_query("OAuth2 Auth"), "oauth2 auth");
    }

    #[test]
    fn test_learner_agent_extract_error_key() {
        let agent = LearnerAgent::new();
        assert_eq!(agent.extract_error_key("[E030] Connection failed"), "[E030]");
        assert_eq!(
            agent.extract_error_key("Some long error without code that exceeds fifty characters total"),
            "Some long error without code that exceeds fifty ch"
        );
    }

    #[tokio::test]
    async fn test_learner_agent_max_patterns_limit() {
        let agent = LearnerAgent::builder()
            .max_patterns(5)
            .min_pattern_occurrences(1)
            .build();

        // Add more patterns than the limit
        for i in 0..10 {
            let data = InteractionData::new("search", format!("unique query {}", i))
                .with_success(100)
                .with_timestamp("2024-01-15T10:00:00Z");
            agent.learn(&data).await.unwrap();
        }

        // Pattern count should be limited
        assert!(agent.pattern_count() <= 5);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let builder = LearnerAgentBuilder::default();
        let agent = builder.build();
        assert_eq!(agent.config().min_pattern_occurrences, 3);
    }

    #[test]
    fn test_builder_config() {
        let config = LearnerConfig::fast();
        let agent = LearnerAgentBuilder::default().config(config).build();
        assert_eq!(agent.config().min_pattern_occurrences, 2);
    }

    #[test]
    fn test_builder_chain() {
        let agent = LearnerAgentBuilder::default()
            .min_pattern_occurrences(10)
            .learning_rate(0.5)
            .max_patterns(100)
            .confidence_threshold(0.8)
            .auto_recommend(false)
            .build();

        assert_eq!(agent.config().min_pattern_occurrences, 10);
        assert!((agent.config().learning_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(agent.config().max_patterns, 100);
        assert!((agent.config().confidence_threshold - 0.8).abs() < f64::EPSILON);
        assert!(!agent.config().auto_recommend);
    }
}
