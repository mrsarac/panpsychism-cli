//! Context and Memory Management for Project Panpsychism.
//!
//! 🧵 **The Memory Weaver** — Threads of past incantations strengthen future spells.
//!
//! This module implements Agent 12: the Contextualizer, which manages session memory
//! and context injection for the Sorcerer's Wand system. Like a master weaver who
//! remembers every thread ever spun, the Contextualizer maintains a rich tapestry
//! of past interactions to enhance future spell-casting.
//!
//! ## Philosophy
//!
//! In the Spinoza framework, memory is essential for CONATUS (persistence) and
//! RATIO (reason). The Contextualizer embodies both:
//!
//! - **CONATUS**: Preserves session state and interaction history
//! - **RATIO**: Applies logical context injection based on topic relevance
//! - **NATURA**: Natural decay of less relevant memories (LRU eviction)
//! - **LAETITIA**: Joy through contextual understanding and personalization
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |   Interaction    | --> |  SessionMemory   | --> |  ContextWindow   |
//! |   (Raw Input)    |     |   (LRU Cache)    |     | (Token Budget)   |
//! +------------------+     +------------------+     +------------------+
//!                                  |
//!                                  v
//!                         +------------------+
//!                         |  Topic Recall    |
//!                         | (Keyword Match)  |
//!                         +------------------+
//!                                  |
//!                                  v
//!                         +------------------+
//!                         | ContextualizedQuery |
//!                         |  (Enriched Output)  |
//!                         +------------------+
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::contextualizer::{ContextualizerAgent, ContextConfig};
//!
//! let mut agent = ContextualizerAgent::new(ContextConfig::default());
//!
//! // Remember an interaction
//! let interaction = Interaction::new("What is OAuth2?", "OAuth2 is an authorization framework...");
//! agent.remember(interaction);
//!
//! // Later, inject context for a related query
//! let query = agent.inject_context("How do I implement refresh tokens?");
//! println!("Context injected: {} memories used", query.memories_used);
//! ```

use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::Result;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors specific to the contextualizer module.
#[derive(Error, Debug)]
pub enum ContextError {
    /// Memory capacity exceeded and could not evict.
    #[error("Memory capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// Token budget exceeded for context window.
    #[error("Token budget exceeded: used {used}, limit {limit}")]
    TokenBudgetExceeded { used: usize, limit: usize },

    /// Invalid topic for recall.
    #[error("Invalid topic for recall: {0}")]
    InvalidTopic(String),

    /// Session memory is empty.
    #[error("Session memory is empty - no memories to recall")]
    EmptyMemory,

    /// Memory lock acquisition failed.
    #[error("Failed to acquire memory lock")]
    LockError,
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the Contextualizer Agent.
///
/// # Default Values
///
/// | Setting | Default | Description |
/// |---------|---------|-------------|
/// | `memory_capacity` | 100 | Max interactions to remember |
/// | `token_budget` | 4096 | Max tokens for context window |
/// | `relevance_threshold` | 0.3 | Min relevance score for recall |
/// | `max_memories_per_query` | 5 | Max memories to inject per query |
/// | `enable_topic_extraction` | true | Auto-extract topics from interactions |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum number of interactions to store in session memory.
    #[serde(default = "default_memory_capacity")]
    pub memory_capacity: usize,

    /// Maximum token budget for the context window.
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,

    /// Minimum relevance score for memory recall (0.0 - 1.0).
    #[serde(default = "default_relevance_threshold")]
    pub relevance_threshold: f64,

    /// Maximum number of memories to inject per query.
    #[serde(default = "default_max_memories")]
    pub max_memories_per_query: usize,

    /// Whether to automatically extract topics from interactions.
    #[serde(default = "default_enable_topic_extraction")]
    pub enable_topic_extraction: bool,

    /// Memory decay factor (older memories get lower scores).
    #[serde(default = "default_decay_factor")]
    pub decay_factor: f64,
}

fn default_memory_capacity() -> usize {
    100
}

fn default_token_budget() -> usize {
    4096
}

fn default_relevance_threshold() -> f64 {
    0.3
}

fn default_max_memories() -> usize {
    5
}

fn default_enable_topic_extraction() -> bool {
    true
}

fn default_decay_factor() -> f64 {
    0.95
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            memory_capacity: default_memory_capacity(),
            token_budget: default_token_budget(),
            relevance_threshold: default_relevance_threshold(),
            max_memories_per_query: default_max_memories(),
            enable_topic_extraction: default_enable_topic_extraction(),
            decay_factor: default_decay_factor(),
        }
    }
}

impl ContextConfig {
    /// Create a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            memory_capacity: 10,
            token_budget: 1024,
            relevance_threshold: 0.1,
            max_memories_per_query: 3,
            enable_topic_extraction: false,
            decay_factor: 1.0,
        }
    }

    /// Create a configuration optimized for long sessions.
    pub fn long_session() -> Self {
        Self {
            memory_capacity: 500,
            token_budget: 8192,
            relevance_threshold: 0.4,
            max_memories_per_query: 10,
            enable_topic_extraction: true,
            decay_factor: 0.9,
        }
    }
}

// =============================================================================
// MEMORY TYPES
// =============================================================================

/// A single memory unit stored in session memory.
///
/// Memories are the threads in the weaver's tapestry, each representing
/// a past interaction that may inform future spell-casting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier for this memory.
    pub id: String,

    /// The original query or prompt.
    pub query: String,

    /// The response or outcome.
    pub response: String,

    /// Extracted topics/keywords for recall.
    pub topics: HashSet<String>,

    /// When this memory was created.
    #[serde(with = "instant_serde")]
    pub created_at: Instant,

    /// Estimated token count for this memory.
    pub token_count: usize,

    /// Number of times this memory has been recalled.
    pub recall_count: u32,

    /// Relevance score (updated on recall).
    pub relevance_score: f64,
}

/// Serde support for Instant (using duration from now).
mod instant_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Instant;

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        instant.elapsed().as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let _secs = u64::deserialize(deserializer)?;
        Ok(Instant::now())
    }
}

impl Memory {
    /// Create a new memory from query and response.
    ///
    /// Automatically generates an ID and extracts topics if the content
    /// contains recognizable keywords.
    ///
    /// # Arguments
    ///
    /// * `query` - The original user query
    /// * `response` - The system response
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let memory = Memory::new("What is OAuth2?", "OAuth2 is an authorization framework...");
    /// assert!(!memory.id.is_empty());
    /// ```
    pub fn new(query: impl Into<String>, response: impl Into<String>) -> Self {
        let query = query.into();
        let response = response.into();
        let topics = Self::extract_topics(&query, &response);
        let token_count = Self::estimate_tokens(&query) + Self::estimate_tokens(&response);

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            query,
            response,
            topics,
            created_at: Instant::now(),
            token_count,
            recall_count: 0,
            relevance_score: 1.0,
        }
    }

    /// Create a memory with explicit topics.
    pub fn with_topics(
        query: impl Into<String>,
        response: impl Into<String>,
        topics: HashSet<String>,
    ) -> Self {
        let query = query.into();
        let response = response.into();
        let token_count = Self::estimate_tokens(&query) + Self::estimate_tokens(&response);

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            query,
            response,
            topics,
            created_at: Instant::now(),
            token_count,
            recall_count: 0,
            relevance_score: 1.0,
        }
    }

    /// Extract topics from query and response using keyword matching.
    fn extract_topics(query: &str, response: &str) -> HashSet<String> {
        let mut topics = HashSet::new();
        let combined = format!("{} {}", query, response).to_lowercase();

        // Common technical topics
        let topic_keywords = [
            "authentication",
            "authorization",
            "oauth",
            "jwt",
            "api",
            "database",
            "cache",
            "error",
            "performance",
            "security",
            "testing",
            "deployment",
            "docker",
            "kubernetes",
            "rust",
            "typescript",
            "python",
            "react",
            "config",
            "validation",
        ];

        for keyword in topic_keywords {
            if combined.contains(keyword) {
                topics.insert(keyword.to_string());
            }
        }

        // Extract words that appear multiple times (likely important)
        let words: Vec<&str> = combined.split_whitespace().collect();
        let mut word_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();

        for word in words {
            if word.len() > 4 {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        for (word, count) in word_counts {
            if count >= 2 {
                topics.insert(word.to_string());
            }
        }

        topics
    }

    /// Estimate token count for text (~4 chars per token).
    fn estimate_tokens(text: &str) -> usize {
        (text.len() + 3) / 4
    }

    /// Get the age of this memory.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check if this memory matches a given topic.
    pub fn matches_topic(&self, topic: &str) -> bool {
        let topic_lower = topic.to_lowercase();
        self.topics.iter().any(|t| t.contains(&topic_lower))
            || self.query.to_lowercase().contains(&topic_lower)
            || self.response.to_lowercase().contains(&topic_lower)
    }

    /// Calculate relevance score based on topic match and age.
    pub fn calculate_relevance(&self, query: &str, decay_factor: f64) -> f64 {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<&str> = query_lower.split_whitespace().collect();

        // Topic overlap score
        let topic_matches = self
            .topics
            .iter()
            .filter(|t| query_words.iter().any(|w| t.contains(w)))
            .count();

        let topic_score = if self.topics.is_empty() {
            0.0
        } else {
            topic_matches as f64 / self.topics.len() as f64
        };

        // Word overlap in query/response
        let memory_words: HashSet<String> = format!("{} {}", self.query, self.response)
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(String::from)
            .collect();

        let word_matches = query_words
            .iter()
            .filter(|w| memory_words.contains(&w.to_string()))
            .count();

        let word_score = if query_words.is_empty() {
            0.0
        } else {
            word_matches as f64 / query_words.len() as f64
        };

        // Age decay
        let age_secs = self.age().as_secs() as f64;
        let age_decay = decay_factor.powf(age_secs / 3600.0); // Decay per hour

        // Recall boost (frequently recalled memories are more relevant)
        let recall_boost = 1.0 + (self.recall_count as f64 * 0.1).min(0.5);

        // Combined score
        ((topic_score * 0.4) + (word_score * 0.4) + (self.relevance_score * 0.2))
            * age_decay
            * recall_boost
    }
}

/// An interaction to be remembered.
///
/// This is the input type for the `remember` method, representing
/// a complete query-response cycle.
#[derive(Debug, Clone)]
pub struct Interaction {
    /// The user's query.
    pub query: String,
    /// The system's response.
    pub response: String,
    /// Optional explicit topics.
    pub topics: Option<HashSet<String>>,
    /// Optional metadata.
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

impl Interaction {
    /// Create a new interaction.
    pub fn new(query: impl Into<String>, response: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            response: response.into(),
            topics: None,
            metadata: None,
        }
    }

    /// Add explicit topics to the interaction.
    pub fn with_topics(mut self, topics: HashSet<String>) -> Self {
        self.topics = Some(topics);
        self
    }

    /// Add metadata to the interaction.
    pub fn with_metadata(
        mut self,
        metadata: std::collections::HashMap<String, String>,
    ) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Convert to a Memory.
    fn into_memory(self) -> Memory {
        match self.topics {
            Some(topics) => Memory::with_topics(self.query, self.response, topics),
            None => Memory::new(self.query, self.response),
        }
    }
}

// =============================================================================
// CONTEXT WINDOW
// =============================================================================

/// Manages the token budget for context injection.
///
/// The context window is like the sorcerer's focus — it can only hold
/// so much magical energy (tokens) at once. The ContextWindow ensures
/// we never exceed our capacity while maximizing relevance.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    /// Maximum token budget.
    capacity: usize,

    /// Currently used tokens.
    used: usize,

    /// Memories included in the window.
    included_memories: Vec<String>,
}

impl ContextWindow {
    /// Create a new context window with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: 0,
            included_memories: Vec::new(),
        }
    }

    /// Get the remaining token budget.
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }

    /// Get the current usage.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Get the total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if a memory can fit in the window.
    pub fn can_fit(&self, memory: &Memory) -> bool {
        memory.token_count <= self.remaining()
    }

    /// Try to add a memory to the window.
    ///
    /// Returns `true` if the memory was added, `false` if it would exceed capacity.
    pub fn try_add(&mut self, memory: &Memory) -> bool {
        if self.can_fit(memory) {
            self.used += memory.token_count;
            self.included_memories.push(memory.id.clone());
            true
        } else {
            false
        }
    }

    /// Get the IDs of memories included in the window.
    pub fn included_ids(&self) -> &[String] {
        &self.included_memories
    }

    /// Get the utilization percentage.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.used as f64 / self.capacity as f64) * 100.0
        }
    }

    /// Reset the window.
    pub fn reset(&mut self) {
        self.used = 0;
        self.included_memories.clear();
    }
}

// =============================================================================
// SESSION MEMORY
// =============================================================================

/// LRU-based session memory storage.
///
/// The SessionMemory is the weaver's loom, holding threads (memories)
/// in order of recency. Older threads naturally fall away as new ones
/// are added, maintaining a manageable tapestry size.
#[derive(Debug)]
pub struct SessionMemory {
    /// LRU cache of memories.
    cache: RwLock<LruCache<String, Memory>>,

    /// Total memories ever stored.
    total_stored: std::sync::atomic::AtomicU64,

    /// Total memories evicted.
    total_evicted: std::sync::atomic::AtomicU64,
}

impl SessionMemory {
    /// Create a new session memory with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("capacity must be > 0"),
            )),
            total_stored: std::sync::atomic::AtomicU64::new(0),
            total_evicted: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Store a memory in the session.
    ///
    /// If capacity is exceeded, the least recently used memory is evicted.
    pub fn store(&self, memory: Memory) -> std::result::Result<(), ContextError> {
        let mut cache = self.cache.write().map_err(|_| ContextError::LockError)?;

        // Check if we'll evict
        if cache.len() == cache.cap().get() && !cache.contains(&memory.id) {
            self.total_evicted
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        cache.put(memory.id.clone(), memory);
        self.total_stored
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Retrieve a memory by ID.
    pub fn get(&self, id: &str) -> Option<Memory> {
        self.cache
            .write()
            .ok()?
            .get(id)
            .cloned()
    }

    /// Get all memories matching a topic.
    pub fn recall_by_topic(&self, topic: &str) -> Vec<Memory> {
        let cache = match self.cache.read() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        cache
            .iter()
            .filter(|(_, memory)| memory.matches_topic(topic))
            .map(|(_, memory)| memory.clone())
            .collect()
    }

    /// Get all memories, sorted by relevance to a query.
    pub fn recall_relevant(&self, query: &str, decay_factor: f64, limit: usize) -> Vec<Memory> {
        let cache = match self.cache.read() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut memories: Vec<(Memory, f64)> = cache
            .iter()
            .map(|(_, memory)| {
                let score = memory.calculate_relevance(query, decay_factor);
                (memory.clone(), score)
            })
            .collect();

        // Sort by relevance (descending)
        memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        memories
            .into_iter()
            .take(limit)
            .map(|(memory, _)| memory)
            .collect()
    }

    /// Get the number of stored memories.
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Check if the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all memories.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let (len, cap) = self
            .cache
            .read()
            .map(|c| (c.len(), c.cap().get()))
            .unwrap_or((0, 0));

        MemoryStats {
            current_size: len,
            capacity: cap,
            total_stored: self.total_stored.load(std::sync::atomic::Ordering::Relaxed),
            total_evicted: self.total_evicted.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Increment recall count for a memory.
    pub fn mark_recalled(&self, id: &str) {
        if let Ok(mut cache) = self.cache.write() {
            if let Some(memory) = cache.get_mut(id) {
                memory.recall_count += 1;
            }
        }
    }
}

/// Statistics about session memory.
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Current number of memories stored.
    pub current_size: usize,
    /// Maximum capacity.
    pub capacity: usize,
    /// Total memories ever stored.
    pub total_stored: u64,
    /// Total memories evicted.
    pub total_evicted: u64,
}

impl MemoryStats {
    /// Get the eviction rate as a percentage.
    pub fn eviction_rate(&self) -> f64 {
        if self.total_stored == 0 {
            0.0
        } else {
            (self.total_evicted as f64 / self.total_stored as f64) * 100.0
        }
    }

    /// Get the utilization percentage.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.current_size as f64 / self.capacity as f64) * 100.0
        }
    }
}

// =============================================================================
// CONTEXTUALIZED QUERY
// =============================================================================

/// A query enriched with context from session memory.
///
/// This is the output of the context injection process — the original
/// incantation strengthened with threads from past spells.
#[derive(Debug, Clone)]
pub struct ContextualizedQuery {
    /// The original query.
    pub original_query: String,

    /// The query with injected context.
    pub enriched_query: String,

    /// Memories used for context injection.
    pub memories_used: Vec<Memory>,

    /// Total tokens used for context.
    pub context_tokens: usize,

    /// Relevance scores for each memory.
    pub relevance_scores: Vec<f64>,
}

impl ContextualizedQuery {
    /// Check if context was injected.
    pub fn has_context(&self) -> bool {
        !self.memories_used.is_empty()
    }

    /// Get the number of memories used.
    pub fn memory_count(&self) -> usize {
        self.memories_used.len()
    }

    /// Get the average relevance score.
    pub fn average_relevance(&self) -> f64 {
        if self.relevance_scores.is_empty() {
            0.0
        } else {
            self.relevance_scores.iter().sum::<f64>() / self.relevance_scores.len() as f64
        }
    }
}

// =============================================================================
// CONTEXTUALIZER AGENT
// =============================================================================

/// The Memory Weaver — Agent 12 of the Sorcerer's Wand system.
///
/// The Contextualizer manages session memory and injects relevant context
/// into queries. Like a master weaver, it remembers every thread and knows
/// which ones to pull for each new creation.
///
/// # Responsibilities
///
/// 1. **Remember**: Store interactions in session memory
/// 2. **Recall**: Retrieve relevant memories by topic
/// 3. **Inject**: Enrich queries with contextual memory
/// 4. **Forget**: Clear session memory when needed
///
/// # Example
///
/// ```rust,ignore
/// let mut agent = ContextualizerAgent::new(ContextConfig::default());
///
/// // Build up memory
/// agent.remember(Interaction::new("What is OAuth?", "OAuth is..."));
/// agent.remember(Interaction::new("JWT tokens?", "JWT stands for..."));
///
/// // Query with context
/// let query = agent.inject_context("How do refresh tokens work?");
/// // Query now includes relevant OAuth/JWT context
/// ```
#[derive(Debug)]
pub struct ContextualizerAgent {
    /// Configuration.
    config: ContextConfig,

    /// Session memory storage.
    memory: Arc<SessionMemory>,

    /// Current context window.
    window: ContextWindow,
}

impl ContextualizerAgent {
    /// Create a new Contextualizer Agent with the given configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = ContextualizerAgent::new(ContextConfig::default());
    /// ```
    pub fn new(config: ContextConfig) -> Self {
        let memory = Arc::new(SessionMemory::new(config.memory_capacity));
        let window = ContextWindow::new(config.token_budget);

        Self {
            config,
            memory,
            window,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ContextConfig::default())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory.stats()
    }

    // =========================================================================
    // CORE METHODS
    // =========================================================================

    /// Inject context from session memory into a query.
    ///
    /// Retrieves relevant memories and constructs an enriched query
    /// that includes contextual information from past interactions.
    ///
    /// # Arguments
    ///
    /// * `query` - The original user query
    ///
    /// # Returns
    ///
    /// A `ContextualizedQuery` containing the enriched query and metadata.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let query = agent.inject_context("How do I implement refresh tokens?");
    /// println!("Enriched: {}", query.enriched_query);
    /// println!("Used {} memories", query.memories_used.len());
    /// ```
    pub fn inject_context(&mut self, query: &str) -> ContextualizedQuery {
        // Reset the context window for this query
        self.window.reset();

        // Recall relevant memories
        let memories = self.memory.recall_relevant(
            query,
            self.config.decay_factor,
            self.config.max_memories_per_query * 2, // Fetch extra in case some don't fit
        );

        let mut used_memories = Vec::new();
        let mut relevance_scores = Vec::new();

        // Build context from relevant memories that fit
        let mut context_parts = Vec::new();

        for memory in memories {
            let score = memory.calculate_relevance(query, self.config.decay_factor);

            if score < self.config.relevance_threshold {
                continue;
            }

            if used_memories.len() >= self.config.max_memories_per_query {
                break;
            }

            if self.window.try_add(&memory) {
                self.memory.mark_recalled(&memory.id);

                context_parts.push(format!(
                    "Previous context:\nQ: {}\nA: {}\n",
                    memory.query, memory.response
                ));

                relevance_scores.push(score);
                used_memories.push(memory);
            }
        }

        // Build enriched query
        let enriched_query = if context_parts.is_empty() {
            query.to_string()
        } else {
            format!(
                "=== CONTEXT FROM PREVIOUS INTERACTIONS ===\n\n{}\n=== CURRENT QUERY ===\n\n{}",
                context_parts.join("\n---\n"),
                query
            )
        };

        ContextualizedQuery {
            original_query: query.to_string(),
            enriched_query,
            memories_used: used_memories,
            context_tokens: self.window.used(),
            relevance_scores,
        }
    }

    /// Remember an interaction in session memory.
    ///
    /// Stores the interaction for future context injection. The memory
    /// will be automatically evicted when capacity is exceeded (LRU).
    ///
    /// # Arguments
    ///
    /// * `interaction` - The interaction to remember
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let interaction = Interaction::new(
    ///     "What is OAuth2?",
    ///     "OAuth2 is an authorization framework..."
    /// );
    /// agent.remember(interaction);
    /// ```
    pub fn remember(&mut self, interaction: Interaction) {
        let memory = interaction.into_memory();
        let _ = self.memory.store(memory);
    }

    /// Recall memories matching a topic.
    ///
    /// Searches session memory for interactions related to the given topic.
    ///
    /// # Arguments
    ///
    /// * `topic` - The topic to search for
    ///
    /// # Returns
    ///
    /// A vector of matching memories.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let memories = agent.recall("authentication");
    /// for memory in memories {
    ///     println!("Found: {}", memory.query);
    /// }
    /// ```
    pub fn recall(&self, topic: &str) -> Vec<Memory> {
        self.memory.recall_by_topic(topic)
    }

    /// Clear all session memory.
    ///
    /// Removes all stored memories, effectively "forgetting" the session.
    /// Use this when starting a new session or when memory cleanup is needed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// agent.forget();
    /// assert!(agent.memory_stats().current_size == 0);
    /// ```
    pub fn forget(&mut self) {
        self.memory.clear();
        self.window.reset();
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Get the current context window utilization.
    pub fn window_utilization(&self) -> f64 {
        self.window.utilization()
    }

    /// Get the number of memories currently stored.
    pub fn memory_count(&self) -> usize {
        self.memory.len()
    }

    /// Check if the session has any memories.
    pub fn has_memories(&self) -> bool {
        !self.memory.is_empty()
    }

    /// Get a specific memory by ID.
    pub fn get_memory(&self, id: &str) -> Option<Memory> {
        self.memory.get(id)
    }

    /// Get all memories (for debugging/inspection).
    pub fn all_memories(&self) -> Vec<Memory> {
        self.memory.recall_relevant("", 1.0, self.config.memory_capacity)
    }
}

impl Default for ContextualizerAgent {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Clone for ContextualizerAgent {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            memory: Arc::clone(&self.memory),
            window: self.window.clone(),
        }
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Memory Tests
    // =========================================================================

    #[test]
    fn test_memory_creation() {
        let memory = Memory::new("What is OAuth?", "OAuth is an authorization framework.");

        assert!(!memory.id.is_empty());
        assert_eq!(memory.query, "What is OAuth?");
        assert!(!memory.topics.is_empty());
        assert!(memory.token_count > 0);
        assert_eq!(memory.recall_count, 0);
    }

    #[test]
    fn test_memory_topic_extraction() {
        let memory = Memory::new(
            "How do I implement JWT authentication?",
            "JWT tokens are used for authentication. You can use the jsonwebtoken crate.",
        );

        assert!(memory.topics.contains("authentication"));
        assert!(memory.topics.contains("jwt"));
    }

    #[test]
    fn test_memory_matches_topic() {
        let memory = Memory::new("OAuth2 setup", "Configure OAuth2 with Google.");

        assert!(memory.matches_topic("oauth"));
        assert!(memory.matches_topic("OAuth2"));
        assert!(memory.matches_topic("google"));
        assert!(!memory.matches_topic("kubernetes"));
    }

    #[test]
    fn test_memory_relevance_calculation() {
        let memory = Memory::new("How to use JWT?", "JWT is a token format for auth.");

        let score1 = memory.calculate_relevance("JWT authentication", 1.0);
        let score2 = memory.calculate_relevance("kubernetes deployment", 1.0);

        assert!(score1 > score2, "JWT query should be more relevant than kubernetes");
    }

    #[test]
    fn test_memory_with_explicit_topics() {
        let topics: HashSet<String> = ["custom", "topic"].iter().map(|s| s.to_string()).collect();
        let memory = Memory::with_topics("query", "response", topics);

        assert!(memory.topics.contains("custom"));
        assert!(memory.topics.contains("topic"));
    }

    // =========================================================================
    // Interaction Tests
    // =========================================================================

    #[test]
    fn test_interaction_creation() {
        let interaction = Interaction::new("query", "response");

        assert_eq!(interaction.query, "query");
        assert_eq!(interaction.response, "response");
        assert!(interaction.topics.is_none());
    }

    #[test]
    fn test_interaction_with_topics() {
        let topics: HashSet<String> = ["test"].iter().map(|s| s.to_string()).collect();
        let interaction = Interaction::new("q", "r").with_topics(topics);

        assert!(interaction.topics.is_some());
        assert!(interaction.topics.as_ref().unwrap().contains("test"));
    }

    #[test]
    fn test_interaction_into_memory() {
        let interaction = Interaction::new("query", "response");
        let memory = interaction.into_memory();

        assert_eq!(memory.query, "query");
        assert_eq!(memory.response, "response");
    }

    // =========================================================================
    // Context Window Tests
    // =========================================================================

    #[test]
    fn test_context_window_creation() {
        let window = ContextWindow::new(1000);

        assert_eq!(window.capacity(), 1000);
        assert_eq!(window.used(), 0);
        assert_eq!(window.remaining(), 1000);
    }

    #[test]
    fn test_context_window_add_memory() {
        let mut window = ContextWindow::new(1000);
        let memory = Memory::new("short query", "short response");

        assert!(window.can_fit(&memory));
        assert!(window.try_add(&memory));
        assert!(window.used() > 0);
        assert_eq!(window.included_ids().len(), 1);
    }

    #[test]
    fn test_context_window_capacity_limit() {
        let mut window = ContextWindow::new(10); // Very small capacity
        let memory = Memory::new(
            "This is a very long query that will exceed the capacity",
            "And this is an even longer response that definitely won't fit",
        );

        assert!(!window.can_fit(&memory));
        assert!(!window.try_add(&memory));
    }

    #[test]
    fn test_context_window_reset() {
        let mut window = ContextWindow::new(1000);
        let memory = Memory::new("query", "response");

        window.try_add(&memory);
        assert!(window.used() > 0);

        window.reset();
        assert_eq!(window.used(), 0);
        assert!(window.included_ids().is_empty());
    }

    #[test]
    fn test_context_window_utilization() {
        let mut window = ContextWindow::new(100);
        assert_eq!(window.utilization(), 0.0);

        // Manually set used for testing
        window.used = 50;
        assert_eq!(window.utilization(), 50.0);
    }

    // =========================================================================
    // Session Memory Tests
    // =========================================================================

    #[test]
    fn test_session_memory_store_and_get() {
        let memory_store = SessionMemory::new(10);
        let memory = Memory::new("query", "response");
        let id = memory.id.clone();

        memory_store.store(memory).unwrap();
        let retrieved = memory_store.get(&id);

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().query, "query");
    }

    #[test]
    fn test_session_memory_lru_eviction() {
        let memory_store = SessionMemory::new(2);

        let m1 = Memory::new("q1", "r1");
        let m2 = Memory::new("q2", "r2");
        let m3 = Memory::new("q3", "r3");

        let id1 = m1.id.clone();

        memory_store.store(m1).unwrap();
        memory_store.store(m2).unwrap();
        memory_store.store(m3).unwrap(); // Should evict m1

        assert!(memory_store.get(&id1).is_none(), "m1 should be evicted");
        assert_eq!(memory_store.len(), 2);
    }

    #[test]
    fn test_session_memory_recall_by_topic() {
        let memory_store = SessionMemory::new(10);

        memory_store
            .store(Memory::new("OAuth setup", "Configure OAuth"))
            .unwrap();
        memory_store
            .store(Memory::new("Docker setup", "Configure Docker"))
            .unwrap();

        let oauth_memories = memory_store.recall_by_topic("oauth");
        let docker_memories = memory_store.recall_by_topic("docker");

        assert!(!oauth_memories.is_empty());
        assert!(!docker_memories.is_empty());
    }

    #[test]
    fn test_session_memory_recall_relevant() {
        let memory_store = SessionMemory::new(10);

        memory_store
            .store(Memory::new("JWT tokens", "JWT is for auth"))
            .unwrap();
        memory_store
            .store(Memory::new("Docker containers", "Docker runs containers"))
            .unwrap();
        memory_store
            .store(Memory::new("OAuth authentication", "OAuth is for auth"))
            .unwrap();

        let relevant = memory_store.recall_relevant("authentication tokens", 1.0, 2);

        assert!(!relevant.is_empty());
        // Auth-related memories should be first
    }

    #[test]
    fn test_session_memory_stats() {
        let memory_store = SessionMemory::new(5);

        memory_store.store(Memory::new("q1", "r1")).unwrap();
        memory_store.store(Memory::new("q2", "r2")).unwrap();

        let stats = memory_store.stats();
        assert_eq!(stats.current_size, 2);
        assert_eq!(stats.capacity, 5);
        assert_eq!(stats.total_stored, 2);
    }

    #[test]
    fn test_session_memory_clear() {
        let memory_store = SessionMemory::new(10);

        memory_store.store(Memory::new("q1", "r1")).unwrap();
        memory_store.store(Memory::new("q2", "r2")).unwrap();

        assert_eq!(memory_store.len(), 2);

        memory_store.clear();
        assert!(memory_store.is_empty());
    }

    // =========================================================================
    // Contextualizer Agent Tests
    // =========================================================================

    #[test]
    fn test_agent_creation() {
        let agent = ContextualizerAgent::new(ContextConfig::default());

        assert_eq!(agent.memory_count(), 0);
        assert!(!agent.has_memories());
    }

    #[test]
    fn test_agent_remember() {
        let mut agent = ContextualizerAgent::new(ContextConfig::minimal());

        agent.remember(Interaction::new("What is OAuth?", "OAuth is..."));

        assert_eq!(agent.memory_count(), 1);
        assert!(agent.has_memories());
    }

    #[test]
    fn test_agent_forget() {
        let mut agent = ContextualizerAgent::new(ContextConfig::minimal());

        agent.remember(Interaction::new("q1", "r1"));
        agent.remember(Interaction::new("q2", "r2"));

        assert_eq!(agent.memory_count(), 2);

        agent.forget();
        assert_eq!(agent.memory_count(), 0);
    }

    #[test]
    fn test_agent_recall() {
        let mut agent = ContextualizerAgent::new(ContextConfig::minimal());

        agent.remember(Interaction::new("OAuth setup", "Configure OAuth2"));
        agent.remember(Interaction::new("Docker tutorial", "Use docker-compose"));

        let oauth_memories = agent.recall("oauth");
        let docker_memories = agent.recall("docker");

        assert!(!oauth_memories.is_empty());
        assert!(!docker_memories.is_empty());
    }

    #[test]
    fn test_agent_inject_context_empty() {
        let mut agent = ContextualizerAgent::new(ContextConfig::minimal());

        let result = agent.inject_context("How do I use OAuth?");

        assert!(!result.has_context());
        assert_eq!(result.enriched_query, result.original_query);
    }

    #[test]
    fn test_agent_inject_context_with_memories() {
        let mut agent = ContextualizerAgent::new(ContextConfig::minimal());

        agent.remember(Interaction::new(
            "What is OAuth?",
            "OAuth is an authorization framework used for secure access.",
        ));

        let result = agent.inject_context("How do I implement OAuth refresh tokens?");

        assert!(result.has_context());
        assert!(result.enriched_query.contains("CONTEXT FROM PREVIOUS"));
        assert!(result.enriched_query.contains("OAuth"));
    }

    #[test]
    fn test_agent_inject_context_relevance_filter() {
        let mut agent = ContextualizerAgent::new(ContextConfig {
            relevance_threshold: 0.8, // High threshold
            ..ContextConfig::minimal()
        });

        agent.remember(Interaction::new(
            "Docker containers",
            "Docker is a containerization platform.",
        ));

        // Query about OAuth should not include Docker context
        let result = agent.inject_context("How do I set up OAuth?");

        // May or may not have context depending on relevance calculation
        // The key is that irrelevant memories should be filtered out
        if result.has_context() {
            for memory in &result.memories_used {
                assert!(
                    memory.matches_topic("oauth")
                        || memory.calculate_relevance("oauth", 1.0) >= 0.8
                );
            }
        }
    }

    #[test]
    fn test_agent_inject_context_max_memories() {
        let mut agent = ContextualizerAgent::new(ContextConfig {
            max_memories_per_query: 2,
            relevance_threshold: 0.0, // Accept all
            ..ContextConfig::minimal()
        });

        agent.remember(Interaction::new("OAuth1", "Response 1"));
        agent.remember(Interaction::new("OAuth2", "Response 2"));
        agent.remember(Interaction::new("OAuth3", "Response 3"));
        agent.remember(Interaction::new("OAuth4", "Response 4"));

        let result = agent.inject_context("OAuth query");

        assert!(result.memories_used.len() <= 2);
    }

    #[test]
    fn test_agent_clone_shares_memory() {
        let mut agent1 = ContextualizerAgent::new(ContextConfig::minimal());
        agent1.remember(Interaction::new("shared", "memory"));

        let agent2 = agent1.clone();

        assert_eq!(agent1.memory_count(), agent2.memory_count());
    }

    #[test]
    fn test_contextualized_query_properties() {
        let query = ContextualizedQuery {
            original_query: "test".to_string(),
            enriched_query: "enriched test".to_string(),
            memories_used: vec![Memory::new("q", "r")],
            context_tokens: 100,
            relevance_scores: vec![0.8, 0.6],
        };

        assert!(query.has_context());
        assert_eq!(query.memory_count(), 1);
        assert_eq!(query.average_relevance(), 0.7);
    }

    #[test]
    fn test_config_presets() {
        let minimal = ContextConfig::minimal();
        let long_session = ContextConfig::long_session();

        assert!(minimal.memory_capacity < long_session.memory_capacity);
        assert!(minimal.token_budget < long_session.token_budget);
    }

    #[test]
    fn test_memory_stats_calculations() {
        let stats = MemoryStats {
            current_size: 50,
            capacity: 100,
            total_stored: 200,
            total_evicted: 50,
        };

        assert_eq!(stats.utilization(), 50.0);
        assert_eq!(stats.eviction_rate(), 25.0);
    }
}
