//! # Persistent Memory Layer (PML)
//!
//! Provides short-term cache and long-term persistent storage for agent states
//! and learned knowledge in Project Panpsychism v4.0.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------+
//! |    MemoryLayer        |
//! +-----------+-----------+
//!             |
//!     +-------+-------+
//!     |               |
//! +---v---+       +---v---+
//! | Short |       | Long  |
//! | Term  |       | Term  |
//! | Cache |       | Store |
//! +-------+       +-------+
//!     |               |
//!     |          +----v----+
//!     |          |  JSON   |
//!     |          |  File   |
//!     |          +---------+
//! (in-memory)    (persistent)
//! ```
//!
//! ## Features
//!
//! - **Short-term memory**: Fast in-memory LRU-style cache
//! - **Long-term memory**: File-backed JSON persistence
//! - **TTL support**: Automatic expiration of memories
//! - **Scoped storage**: Session, Persistent, Global, and Agent-specific scopes
//! - **Tag-based search**: Find memories by tags or patterns
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use panpsychism::memory::{MemoryLayer, MemoryScope, Memorable};
//! use serde_json::json;
//!
//! let mut memory = MemoryLayer::new();
//!
//! // Remember something
//! memory.remember("user_preference", json!({"theme": "dark"}), MemoryScope::Persistent)?;
//!
//! // Recall it later
//! if let Some(mem) = memory.recall("user_preference") {
//!     println!("Theme: {}", mem.value["theme"]);
//! }
//!
//! // Save to file for persistence
//! memory.save_to_file("memory.json")?;
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

// =============================================================================
// CORE TYPES
// =============================================================================

/// A single memory entry with metadata.
///
/// Memories track access patterns, expiration, and can be tagged for
/// easy retrieval and organization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier for this memory.
    pub key: String,

    /// The stored value (any JSON-serializable data).
    pub value: serde_json::Value,

    /// When this memory was first created.
    pub created_at: DateTime<Utc>,

    /// When this memory was last modified.
    pub updated_at: DateTime<Utc>,

    /// Optional expiration time. After this time, the memory is considered stale.
    pub expires_at: Option<DateTime<Utc>>,

    /// Number of times this memory has been accessed (read).
    pub access_count: u64,

    /// Tags for categorization and search.
    pub tags: Vec<String>,
}

impl Memory {
    /// Create a new memory with the given key and value.
    pub fn new(key: impl Into<String>, value: serde_json::Value) -> Self {
        let now = Utc::now();
        Self {
            key: key.into(),
            value,
            created_at: now,
            updated_at: now,
            expires_at: None,
            access_count: 0,
            tags: Vec::new(),
        }
    }

    /// Create a new memory with TTL (time-to-live) in seconds.
    pub fn with_ttl(key: impl Into<String>, value: serde_json::Value, ttl_seconds: u64) -> Self {
        let now = Utc::now();
        let expires_at = now + chrono::Duration::seconds(ttl_seconds as i64);
        Self {
            key: key.into(),
            value,
            created_at: now,
            updated_at: now,
            expires_at: Some(expires_at),
            access_count: 0,
            tags: Vec::new(),
        }
    }

    /// Create a new memory with tags.
    pub fn with_tags(
        key: impl Into<String>,
        value: serde_json::Value,
        tags: Vec<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            key: key.into(),
            value,
            created_at: now,
            updated_at: now,
            expires_at: None,
            access_count: 0,
            tags,
        }
    }

    /// Check if this memory has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Utc::now() > exp)
            .unwrap_or(false)
    }

    /// Increment the access counter and return the new count.
    pub fn touch(&mut self) -> u64 {
        self.access_count += 1;
        self.access_count
    }

    /// Update the value and refresh the updated_at timestamp.
    pub fn update(&mut self, value: serde_json::Value) {
        self.value = value;
        self.updated_at = Utc::now();
    }

    /// Add a tag to this memory.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag from this memory.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if this memory has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Get the age of this memory in seconds.
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }

    /// Get remaining TTL in seconds, if applicable.
    pub fn remaining_ttl_seconds(&self) -> Option<i64> {
        self.expires_at.map(|exp| {
            let remaining = (exp - Utc::now()).num_seconds();
            remaining.max(0)
        })
    }
}

/// Scope determines where and how long a memory is stored.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryScope {
    /// Cleared on restart. Fast, in-memory only.
    Session,

    /// Survives restart. File-backed JSON persistence.
    Persistent,

    /// Shared across all agents. Global namespace.
    Global,

    /// Scoped to a specific agent by name.
    Agent(String),
}

impl Default for MemoryScope {
    fn default() -> Self {
        Self::Session
    }
}

impl std::fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session => write!(f, "session"),
            Self::Persistent => write!(f, "persistent"),
            Self::Global => write!(f, "global"),
            Self::Agent(name) => write!(f, "agent:{}", name),
        }
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the memory layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum number of entries in short-term memory.
    pub max_short_term_entries: usize,

    /// Maximum number of entries in long-term memory.
    pub max_long_term_entries: usize,

    /// Default TTL in seconds for new memories (None = no expiration).
    pub default_ttl_seconds: Option<u64>,

    /// Path for persistent storage (None = no file persistence).
    pub persistence_path: Option<PathBuf>,

    /// Interval for auto-saving to file (in seconds).
    pub auto_save_interval_seconds: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_short_term_entries: 1000,
            max_long_term_entries: 10000,
            default_ttl_seconds: None,
            persistence_path: None,
            auto_save_interval_seconds: 300, // 5 minutes
        }
    }
}

impl MemoryConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum short-term entries.
    pub fn with_max_short_term(mut self, max: usize) -> Self {
        self.max_short_term_entries = max;
        self
    }

    /// Set the maximum long-term entries.
    pub fn with_max_long_term(mut self, max: usize) -> Self {
        self.max_long_term_entries = max;
        self
    }

    /// Set the default TTL.
    pub fn with_default_ttl(mut self, ttl_seconds: u64) -> Self {
        self.default_ttl_seconds = Some(ttl_seconds);
        self
    }

    /// Set the persistence path.
    pub fn with_persistence_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.persistence_path = Some(path.into());
        self
    }

    /// Set the auto-save interval.
    pub fn with_auto_save_interval(mut self, interval_seconds: u64) -> Self {
        self.auto_save_interval_seconds = interval_seconds;
        self
    }
}

// =============================================================================
// MEMORY STATS
// =============================================================================

/// Statistics about the memory layer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStats {
    /// Total number of entries across all stores.
    pub total_entries: usize,

    /// Number of entries in short-term memory.
    pub short_term_count: usize,

    /// Number of entries in long-term memory.
    pub long_term_count: usize,

    /// Total access count across all memories.
    pub total_access_count: u64,

    /// Timestamp of the oldest entry.
    pub oldest_entry: Option<DateTime<Utc>>,

    /// Timestamp of the newest entry.
    pub newest_entry: Option<DateTime<Utc>>,

    /// Number of expired entries (pending cleanup).
    pub expired_count: usize,

    /// Total size estimate in bytes (approximate).
    pub estimated_size_bytes: usize,
}

// =============================================================================
// MEMORABLE TRAIT
// =============================================================================

/// Trait for types that can store and recall memories.
pub trait Memorable {
    /// Store a memory with the given key, value, and scope.
    fn remember(
        &mut self,
        key: &str,
        value: serde_json::Value,
        scope: MemoryScope,
    ) -> Result<()>;

    /// Recall a memory by key. Returns None if not found or expired.
    fn recall(&self, key: &str) -> Option<&Memory>;

    /// Forget (delete) a memory by key. Returns true if it existed.
    fn forget(&mut self, key: &str) -> bool;

    /// Search for memories matching a pattern (simple substring match).
    fn search(&self, pattern: &str) -> Vec<&Memory>;
}

// =============================================================================
// MEMORY LAYER
// =============================================================================

/// The main memory layer providing short-term and long-term storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLayer {
    /// Short-term in-memory cache (cleared on restart).
    short_term: HashMap<String, Memory>,

    /// Long-term file-backed storage (persisted).
    long_term: HashMap<String, Memory>,

    /// Configuration for the memory layer.
    config: MemoryConfig,

    /// Dirty flag indicating unsaved changes.
    #[serde(skip)]
    dirty: bool,
}

impl Default for MemoryLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryLayer {
    /// Create a new memory layer with default configuration.
    pub fn new() -> Self {
        Self {
            short_term: HashMap::new(),
            long_term: HashMap::new(),
            config: MemoryConfig::default(),
            dirty: false,
        }
    }

    /// Create a new memory layer with custom configuration.
    pub fn with_config(config: MemoryConfig) -> Self {
        Self {
            short_term: HashMap::new(),
            long_term: HashMap::new(),
            config,
            dirty: false,
        }
    }

    /// Get a builder for constructing a memory layer.
    pub fn builder() -> MemoryLayerBuilder {
        MemoryLayerBuilder::new()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Remember a value with a specific TTL (time-to-live).
    pub fn remember_with_ttl(
        &mut self,
        key: &str,
        value: serde_json::Value,
        scope: MemoryScope,
        ttl_seconds: u64,
    ) -> Result<()> {
        let memory = Memory::with_ttl(key, value, ttl_seconds);
        self.store_memory(memory, scope)
    }

    /// Remember a value with specific tags.
    pub fn remember_with_tags(
        &mut self,
        key: &str,
        value: serde_json::Value,
        scope: MemoryScope,
        tags: Vec<String>,
    ) -> Result<()> {
        let memory = Memory::with_tags(key, value, tags);
        self.store_memory(memory, scope)
    }

    /// Store a memory in the appropriate store based on scope.
    fn store_memory(&mut self, memory: Memory, scope: MemoryScope) -> Result<()> {
        let key = memory.key.clone();

        match scope {
            MemoryScope::Session => {
                // Check capacity
                if self.short_term.len() >= self.config.max_short_term_entries
                    && !self.short_term.contains_key(&key)
                {
                    self.evict_oldest_short_term();
                }
                self.short_term.insert(key, memory);
            }
            MemoryScope::Persistent | MemoryScope::Global | MemoryScope::Agent(_) => {
                // Check capacity
                if self.long_term.len() >= self.config.max_long_term_entries
                    && !self.long_term.contains_key(&key)
                {
                    self.evict_oldest_long_term();
                }
                self.long_term.insert(key, memory);
                self.dirty = true;
            }
        }

        Ok(())
    }

    /// Evict the oldest entry from short-term memory.
    fn evict_oldest_short_term(&mut self) {
        if let Some(oldest_key) = self
            .short_term
            .iter()
            .min_by_key(|(_, m)| m.created_at)
            .map(|(k, _)| k.clone())
        {
            self.short_term.remove(&oldest_key);
        }
    }

    /// Evict the oldest entry from long-term memory.
    fn evict_oldest_long_term(&mut self) {
        if let Some(oldest_key) = self
            .long_term
            .iter()
            .min_by_key(|(_, m)| m.created_at)
            .map(|(k, _)| k.clone())
        {
            self.long_term.remove(&oldest_key);
            self.dirty = true;
        }
    }

    /// Recall a memory mutably (to update access count).
    pub fn recall_mut(&mut self, key: &str) -> Option<&mut Memory> {
        // Check short-term first
        if let Some(memory) = self.short_term.get_mut(key) {
            if !memory.is_expired() {
                memory.touch();
                return Some(memory);
            }
        }

        // Then check long-term
        if let Some(memory) = self.long_term.get_mut(key) {
            if !memory.is_expired() {
                memory.touch();
                self.dirty = true;
                return Some(memory);
            }
        }

        None
    }

    /// Search for memories by tag.
    pub fn search_by_tag(&self, tag: &str) -> Vec<&Memory> {
        let mut results: Vec<&Memory> = self
            .short_term
            .values()
            .chain(self.long_term.values())
            .filter(|m| !m.is_expired() && m.has_tag(tag))
            .collect();

        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        results
    }

    /// Search for memories by pattern (matches key or value).
    pub fn search_by_pattern(&self, pattern: &str) -> Vec<&Memory> {
        let pattern_lower = pattern.to_lowercase();

        let mut results: Vec<&Memory> = self
            .short_term
            .values()
            .chain(self.long_term.values())
            .filter(|m| {
                if m.is_expired() {
                    return false;
                }

                // Check key
                if m.key.to_lowercase().contains(&pattern_lower) {
                    return true;
                }

                // Check value (if string)
                if let Some(s) = m.value.as_str() {
                    if s.to_lowercase().contains(&pattern_lower) {
                        return true;
                    }
                }

                // Check tags
                m.tags.iter().any(|t| t.to_lowercase().contains(&pattern_lower))
            })
            .collect();

        results.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        results
    }

    /// Get statistics about the memory layer.
    pub fn get_stats(&self) -> MemoryStats {
        let all_memories: Vec<&Memory> = self
            .short_term
            .values()
            .chain(self.long_term.values())
            .collect();

        let total_access_count: u64 = all_memories.iter().map(|m| m.access_count).sum();

        let oldest_entry = all_memories
            .iter()
            .map(|m| m.created_at)
            .min();

        let newest_entry = all_memories
            .iter()
            .map(|m| m.created_at)
            .max();

        let expired_count = all_memories.iter().filter(|m| m.is_expired()).count();

        // Rough size estimate
        let estimated_size_bytes = all_memories
            .iter()
            .map(|m| {
                m.key.len()
                    + m.value.to_string().len()
                    + m.tags.iter().map(|t| t.len()).sum::<usize>()
                    + 64 // overhead for timestamps and counts
            })
            .sum();

        MemoryStats {
            total_entries: all_memories.len(),
            short_term_count: self.short_term.len(),
            long_term_count: self.long_term.len(),
            total_access_count,
            oldest_entry,
            newest_entry,
            expired_count,
            estimated_size_bytes,
        }
    }

    /// Save long-term memory to a file.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    Error::DirectoryCreateError {
                        path: parent.display().to_string(),
                        source: e,
                    }
                })?;
            }
        }

        let json = serde_json::to_string_pretty(&self.long_term)
            .map_err(|e| Error::Json(e))?;

        fs::write(path, json).map_err(|e| Error::FileWriteError {
            path: path.display().to_string(),
            source: e,
        })?;

        Ok(())
    }

    /// Load long-term memory from a file.
    pub fn load_from_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            // Not an error - just start with empty memory
            return Ok(());
        }

        let content = fs::read_to_string(path).map_err(|e| Error::FileReadError {
            path: path.display().to_string(),
            source: e,
        })?;

        let loaded: HashMap<String, Memory> =
            serde_json::from_str(&content).map_err(|e| Error::Json(e))?;

        self.long_term = loaded;
        self.dirty = false;

        Ok(())
    }

    /// Clean up expired entries from both stores.
    pub fn cleanup_expired(&mut self) -> usize {
        let short_term_expired: Vec<String> = self
            .short_term
            .iter()
            .filter(|(_, m)| m.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        let long_term_expired: Vec<String> = self
            .long_term
            .iter()
            .filter(|(_, m)| m.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        let count = short_term_expired.len() + long_term_expired.len();

        for key in short_term_expired {
            self.short_term.remove(&key);
        }

        for key in long_term_expired {
            self.long_term.remove(&key);
            self.dirty = true;
        }

        count
    }

    /// Clear all memories in a specific scope.
    pub fn clear_scope(&mut self, scope: MemoryScope) {
        match scope {
            MemoryScope::Session => {
                self.short_term.clear();
            }
            MemoryScope::Persistent | MemoryScope::Global => {
                self.long_term.clear();
                self.dirty = true;
            }
            MemoryScope::Agent(ref agent_name) => {
                // For agent-scoped, we clear by key prefix convention
                let prefix = format!("agent:{}:", agent_name);
                let keys_to_remove: Vec<String> = self
                    .long_term
                    .keys()
                    .filter(|k| k.starts_with(&prefix))
                    .cloned()
                    .collect();

                for key in keys_to_remove {
                    self.long_term.remove(&key);
                }
                self.dirty = true;
            }
        }
    }

    /// Check if there are unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark as saved (clear dirty flag).
    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    /// Get all keys in short-term memory.
    pub fn short_term_keys(&self) -> impl Iterator<Item = &String> {
        self.short_term.keys()
    }

    /// Get all keys in long-term memory.
    pub fn long_term_keys(&self) -> impl Iterator<Item = &String> {
        self.long_term.keys()
    }

    /// Check if a key exists (in either store).
    pub fn contains_key(&self, key: &str) -> bool {
        self.short_term.contains_key(key) || self.long_term.contains_key(key)
    }

    /// Get the total number of entries.
    pub fn len(&self) -> usize {
        self.short_term.len() + self.long_term.len()
    }

    /// Check if both stores are empty.
    pub fn is_empty(&self) -> bool {
        self.short_term.is_empty() && self.long_term.is_empty()
    }

    /// Update a memory's value if it exists.
    pub fn update(&mut self, key: &str, value: serde_json::Value) -> bool {
        if let Some(memory) = self.short_term.get_mut(key) {
            memory.update(value);
            return true;
        }

        if let Some(memory) = self.long_term.get_mut(key) {
            memory.update(value);
            self.dirty = true;
            return true;
        }

        false
    }

    /// Add a tag to a memory if it exists.
    pub fn add_tag(&mut self, key: &str, tag: impl Into<String>) -> bool {
        let tag = tag.into();

        if let Some(memory) = self.short_term.get_mut(key) {
            memory.add_tag(tag);
            return true;
        }

        if let Some(memory) = self.long_term.get_mut(key) {
            memory.add_tag(tag);
            self.dirty = true;
            return true;
        }

        false
    }

    /// Remove a tag from a memory if it exists.
    pub fn remove_tag(&mut self, key: &str, tag: &str) -> bool {
        if let Some(memory) = self.short_term.get_mut(key) {
            return memory.remove_tag(tag);
        }

        if let Some(memory) = self.long_term.get_mut(key) {
            let removed = memory.remove_tag(tag);
            if removed {
                self.dirty = true;
            }
            return removed;
        }

        false
    }
}

impl Memorable for MemoryLayer {
    fn remember(
        &mut self,
        key: &str,
        value: serde_json::Value,
        scope: MemoryScope,
    ) -> Result<()> {
        let mut memory = Memory::new(key, value);

        // Apply default TTL if configured
        if let Some(ttl) = self.config.default_ttl_seconds {
            memory.expires_at = Some(Utc::now() + chrono::Duration::seconds(ttl as i64));
        }

        self.store_memory(memory, scope)
    }

    fn recall(&self, key: &str) -> Option<&Memory> {
        // Check short-term first (faster)
        if let Some(memory) = self.short_term.get(key) {
            if !memory.is_expired() {
                return Some(memory);
            }
        }

        // Then check long-term
        if let Some(memory) = self.long_term.get(key) {
            if !memory.is_expired() {
                return Some(memory);
            }
        }

        None
    }

    fn forget(&mut self, key: &str) -> bool {
        let removed_short = self.short_term.remove(key).is_some();
        let removed_long = self.long_term.remove(key).is_some();

        if removed_long {
            self.dirty = true;
        }

        removed_short || removed_long
    }

    fn search(&self, pattern: &str) -> Vec<&Memory> {
        self.search_by_pattern(pattern)
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for constructing a MemoryLayer with custom configuration.
#[derive(Debug, Clone, Default)]
pub struct MemoryLayerBuilder {
    config: MemoryConfig,
    load_from_file: Option<PathBuf>,
}

impl MemoryLayerBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
            load_from_file: None,
        }
    }

    /// Set the maximum short-term entries.
    pub fn max_short_term_entries(mut self, max: usize) -> Self {
        self.config.max_short_term_entries = max;
        self
    }

    /// Set the maximum long-term entries.
    pub fn max_long_term_entries(mut self, max: usize) -> Self {
        self.config.max_long_term_entries = max;
        self
    }

    /// Set the default TTL in seconds.
    pub fn default_ttl_seconds(mut self, ttl: u64) -> Self {
        self.config.default_ttl_seconds = Some(ttl);
        self
    }

    /// Set the persistence path.
    pub fn persistence_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.persistence_path = Some(path.into());
        self
    }

    /// Set the auto-save interval in seconds.
    pub fn auto_save_interval_seconds(mut self, interval: u64) -> Self {
        self.config.auto_save_interval_seconds = interval;
        self
    }

    /// Load existing data from a file on build.
    pub fn load_from(mut self, path: impl Into<PathBuf>) -> Self {
        self.load_from_file = Some(path.into());
        self
    }

    /// Build the MemoryLayer.
    pub fn build(self) -> Result<MemoryLayer> {
        let mut layer = MemoryLayer::with_config(self.config);

        if let Some(path) = self.load_from_file {
            layer.load_from_file(&path)?;
        }

        Ok(layer)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    // -------------------------------------------------------------------------
    // Memory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_new() {
        let memory = Memory::new("test_key", json!({"value": 42}));

        assert_eq!(memory.key, "test_key");
        assert_eq!(memory.value["value"], 42);
        assert_eq!(memory.access_count, 0);
        assert!(memory.tags.is_empty());
        assert!(memory.expires_at.is_none());
    }

    #[test]
    fn test_memory_with_ttl() {
        let memory = Memory::with_ttl("key", json!("value"), 3600);

        assert!(memory.expires_at.is_some());
        assert!(!memory.is_expired());

        // TTL should be approximately 3600 seconds from now
        let remaining = memory.remaining_ttl_seconds().unwrap();
        assert!(remaining > 3590 && remaining <= 3600);
    }

    #[test]
    fn test_memory_with_tags() {
        let memory = Memory::with_tags(
            "key",
            json!("value"),
            vec!["tag1".to_string(), "tag2".to_string()],
        );

        assert!(memory.has_tag("tag1"));
        assert!(memory.has_tag("tag2"));
        assert!(!memory.has_tag("tag3"));
    }

    #[test]
    fn test_memory_touch() {
        let mut memory = Memory::new("key", json!("value"));

        assert_eq!(memory.access_count, 0);
        assert_eq!(memory.touch(), 1);
        assert_eq!(memory.touch(), 2);
        assert_eq!(memory.access_count, 2);
    }

    #[test]
    fn test_memory_update() {
        let mut memory = Memory::new("key", json!(1));
        let original_created = memory.created_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        memory.update(json!(2));

        assert_eq!(memory.value, json!(2));
        assert_eq!(memory.created_at, original_created);
        assert!(memory.updated_at > original_created);
    }

    #[test]
    fn test_memory_add_remove_tag() {
        let mut memory = Memory::new("key", json!("value"));

        memory.add_tag("tag1");
        assert!(memory.has_tag("tag1"));

        // Adding duplicate should not create duplicates
        memory.add_tag("tag1");
        assert_eq!(memory.tags.len(), 1);

        assert!(memory.remove_tag("tag1"));
        assert!(!memory.has_tag("tag1"));
        assert!(!memory.remove_tag("nonexistent"));
    }

    #[test]
    fn test_memory_is_expired() {
        let mut memory = Memory::new("key", json!("value"));
        assert!(!memory.is_expired());

        // Set expiration to the past
        memory.expires_at = Some(Utc::now() - chrono::Duration::seconds(10));
        assert!(memory.is_expired());
    }

    #[test]
    fn test_memory_age_seconds() {
        let memory = Memory::new("key", json!("value"));

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(memory.age_seconds() >= 0);
    }

    // -------------------------------------------------------------------------
    // MemoryScope Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_scope_display() {
        assert_eq!(MemoryScope::Session.to_string(), "session");
        assert_eq!(MemoryScope::Persistent.to_string(), "persistent");
        assert_eq!(MemoryScope::Global.to_string(), "global");
        assert_eq!(MemoryScope::Agent("agent1".to_string()).to_string(), "agent:agent1");
    }

    #[test]
    fn test_memory_scope_default() {
        assert_eq!(MemoryScope::default(), MemoryScope::Session);
    }

    #[test]
    fn test_memory_scope_equality() {
        assert_eq!(MemoryScope::Session, MemoryScope::Session);
        assert_ne!(MemoryScope::Session, MemoryScope::Persistent);
        assert_eq!(
            MemoryScope::Agent("a".to_string()),
            MemoryScope::Agent("a".to_string())
        );
        assert_ne!(
            MemoryScope::Agent("a".to_string()),
            MemoryScope::Agent("b".to_string())
        );
    }

    // -------------------------------------------------------------------------
    // MemoryConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();

        assert_eq!(config.max_short_term_entries, 1000);
        assert_eq!(config.max_long_term_entries, 10000);
        assert!(config.default_ttl_seconds.is_none());
        assert!(config.persistence_path.is_none());
        assert_eq!(config.auto_save_interval_seconds, 300);
    }

    #[test]
    fn test_memory_config_builder_style() {
        let config = MemoryConfig::new()
            .with_max_short_term(500)
            .with_max_long_term(5000)
            .with_default_ttl(3600)
            .with_persistence_path("/tmp/memory.json")
            .with_auto_save_interval(60);

        assert_eq!(config.max_short_term_entries, 500);
        assert_eq!(config.max_long_term_entries, 5000);
        assert_eq!(config.default_ttl_seconds, Some(3600));
        assert_eq!(config.persistence_path, Some(PathBuf::from("/tmp/memory.json")));
        assert_eq!(config.auto_save_interval_seconds, 60);
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_new() {
        let layer = MemoryLayer::new();

        assert!(layer.is_empty());
        assert_eq!(layer.len(), 0);
        assert!(!layer.is_dirty());
    }

    #[test]
    fn test_memory_layer_with_config() {
        let config = MemoryConfig::new().with_max_short_term(100);
        let layer = MemoryLayer::with_config(config);

        assert_eq!(layer.config().max_short_term_entries, 100);
    }

    #[test]
    fn test_memory_layer_remember_session() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!({"value": 1}), MemoryScope::Session).unwrap();

        assert_eq!(layer.len(), 1);
        assert!(layer.contains_key("key1"));
        assert!(!layer.is_dirty()); // Session scope doesn't mark dirty
    }

    #[test]
    fn test_memory_layer_remember_persistent() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!({"value": 1}), MemoryScope::Persistent).unwrap();

        assert_eq!(layer.len(), 1);
        assert!(layer.contains_key("key1"));
        assert!(layer.is_dirty()); // Persistent scope marks dirty
    }

    #[test]
    fn test_memory_layer_recall() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(42), MemoryScope::Session).unwrap();

        let memory = layer.recall("key1");
        assert!(memory.is_some());
        assert_eq!(memory.unwrap().value, json!(42));

        assert!(layer.recall("nonexistent").is_none());
    }

    #[test]
    fn test_memory_layer_recall_mut() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(42), MemoryScope::Session).unwrap();

        {
            let memory = layer.recall_mut("key1");
            assert!(memory.is_some());
            assert_eq!(memory.unwrap().access_count, 1);
        }

        {
            let memory = layer.recall_mut("key1");
            assert_eq!(memory.unwrap().access_count, 2);
        }
    }

    #[test]
    fn test_memory_layer_forget() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(42), MemoryScope::Session).unwrap();
        assert!(layer.contains_key("key1"));

        assert!(layer.forget("key1"));
        assert!(!layer.contains_key("key1"));

        assert!(!layer.forget("nonexistent"));
    }

    #[test]
    fn test_memory_layer_update() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(1), MemoryScope::Session).unwrap();

        assert!(layer.update("key1", json!(2)));
        assert_eq!(layer.recall("key1").unwrap().value, json!(2));

        assert!(!layer.update("nonexistent", json!(3)));
    }

    // -------------------------------------------------------------------------
    // MemoryLayer TTL Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_remember_with_ttl() {
        let mut layer = MemoryLayer::new();

        layer.remember_with_ttl("key1", json!(42), MemoryScope::Session, 3600).unwrap();

        let memory = layer.recall("key1").unwrap();
        assert!(memory.expires_at.is_some());
    }

    #[test]
    fn test_memory_layer_expired_recall() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(42), MemoryScope::Session).unwrap();

        // Manually expire the memory
        if let Some(memory) = layer.short_term.get_mut("key1") {
            memory.expires_at = Some(Utc::now() - chrono::Duration::seconds(10));
        }

        // Recall should return None for expired memory
        assert!(layer.recall("key1").is_none());
    }

    #[test]
    fn test_memory_layer_cleanup_expired() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(1), MemoryScope::Session).unwrap();
        layer.remember("key2", json!(2), MemoryScope::Session).unwrap();

        // Expire one memory
        if let Some(memory) = layer.short_term.get_mut("key1") {
            memory.expires_at = Some(Utc::now() - chrono::Duration::seconds(10));
        }

        let cleaned = layer.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert!(!layer.contains_key("key1"));
        assert!(layer.contains_key("key2"));
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Tag Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_remember_with_tags() {
        let mut layer = MemoryLayer::new();

        layer.remember_with_tags(
            "key1",
            json!(42),
            MemoryScope::Session,
            vec!["important".to_string(), "config".to_string()],
        ).unwrap();

        let memory = layer.recall("key1").unwrap();
        assert!(memory.has_tag("important"));
        assert!(memory.has_tag("config"));
    }

    #[test]
    fn test_memory_layer_search_by_tag() {
        let mut layer = MemoryLayer::new();

        layer.remember_with_tags("key1", json!(1), MemoryScope::Session, vec!["tag1".to_string()]).unwrap();
        layer.remember_with_tags("key2", json!(2), MemoryScope::Session, vec!["tag1".to_string(), "tag2".to_string()]).unwrap();
        layer.remember_with_tags("key3", json!(3), MemoryScope::Session, vec!["tag2".to_string()]).unwrap();

        let results = layer.search_by_tag("tag1");
        assert_eq!(results.len(), 2);

        let results = layer.search_by_tag("tag2");
        assert_eq!(results.len(), 2);

        let results = layer.search_by_tag("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_layer_add_remove_tag() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(42), MemoryScope::Session).unwrap();

        assert!(layer.add_tag("key1", "newtag"));
        assert!(layer.recall("key1").unwrap().has_tag("newtag"));

        assert!(layer.remove_tag("key1", "newtag"));
        assert!(!layer.recall("key1").unwrap().has_tag("newtag"));

        assert!(!layer.add_tag("nonexistent", "tag"));
        assert!(!layer.remove_tag("nonexistent", "tag"));
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Search Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_search_by_pattern() {
        let mut layer = MemoryLayer::new();

        layer.remember("user_config", json!(1), MemoryScope::Session).unwrap();
        layer.remember("user_preferences", json!(2), MemoryScope::Session).unwrap();
        layer.remember("system_settings", json!(3), MemoryScope::Session).unwrap();

        let results = layer.search_by_pattern("user");
        assert_eq!(results.len(), 2);

        let results = layer.search_by_pattern("settings");
        assert_eq!(results.len(), 1);

        let results = layer.search_by_pattern("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_layer_search_case_insensitive() {
        let mut layer = MemoryLayer::new();

        layer.remember("UserConfig", json!(1), MemoryScope::Session).unwrap();

        let results = layer.search_by_pattern("userconfig");
        assert_eq!(results.len(), 1);

        let results = layer.search_by_pattern("USERCONFIG");
        assert_eq!(results.len(), 1);
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Capacity Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_eviction_short_term() {
        let config = MemoryConfig::new().with_max_short_term(3);
        let mut layer = MemoryLayer::with_config(config);

        layer.remember("key1", json!(1), MemoryScope::Session).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        layer.remember("key2", json!(2), MemoryScope::Session).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        layer.remember("key3", json!(3), MemoryScope::Session).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        layer.remember("key4", json!(4), MemoryScope::Session).unwrap();

        // key1 should have been evicted (oldest)
        assert!(!layer.contains_key("key1"));
        assert!(layer.contains_key("key2"));
        assert!(layer.contains_key("key3"));
        assert!(layer.contains_key("key4"));
        assert_eq!(layer.len(), 3);
    }

    #[test]
    fn test_memory_layer_eviction_long_term() {
        let config = MemoryConfig::new().with_max_long_term(2);
        let mut layer = MemoryLayer::with_config(config);

        layer.remember("key1", json!(1), MemoryScope::Persistent).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        layer.remember("key2", json!(2), MemoryScope::Persistent).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        layer.remember("key3", json!(3), MemoryScope::Persistent).unwrap();

        // key1 should have been evicted
        assert!(!layer.contains_key("key1"));
        assert!(layer.contains_key("key2"));
        assert!(layer.contains_key("key3"));
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Scope Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_clear_scope_session() {
        let mut layer = MemoryLayer::new();

        layer.remember("session1", json!(1), MemoryScope::Session).unwrap();
        layer.remember("session2", json!(2), MemoryScope::Session).unwrap();
        layer.remember("persistent1", json!(3), MemoryScope::Persistent).unwrap();

        layer.clear_scope(MemoryScope::Session);

        assert!(!layer.contains_key("session1"));
        assert!(!layer.contains_key("session2"));
        assert!(layer.contains_key("persistent1"));
    }

    #[test]
    fn test_memory_layer_clear_scope_persistent() {
        let mut layer = MemoryLayer::new();

        layer.remember("session1", json!(1), MemoryScope::Session).unwrap();
        layer.remember("persistent1", json!(2), MemoryScope::Persistent).unwrap();
        layer.remember("persistent2", json!(3), MemoryScope::Persistent).unwrap();

        layer.clear_scope(MemoryScope::Persistent);

        assert!(layer.contains_key("session1"));
        assert!(!layer.contains_key("persistent1"));
        assert!(!layer.contains_key("persistent2"));
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Stats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_get_stats() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(1), MemoryScope::Session).unwrap();
        layer.remember("key2", json!(2), MemoryScope::Persistent).unwrap();
        layer.remember("key3", json!(3), MemoryScope::Persistent).unwrap();

        // Access some memories
        layer.recall_mut("key1");
        layer.recall_mut("key2");
        layer.recall_mut("key2");

        let stats = layer.get_stats();

        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.short_term_count, 1);
        assert_eq!(stats.long_term_count, 2);
        assert_eq!(stats.total_access_count, 3);
        assert!(stats.oldest_entry.is_some());
        assert!(stats.newest_entry.is_some());
        assert_eq!(stats.expired_count, 0);
        assert!(stats.estimated_size_bytes > 0);
    }

    #[test]
    fn test_memory_layer_stats_expired_count() {
        let mut layer = MemoryLayer::new();

        layer.remember("key1", json!(1), MemoryScope::Session).unwrap();
        layer.remember("key2", json!(2), MemoryScope::Session).unwrap();

        // Expire one memory
        if let Some(memory) = layer.short_term.get_mut("key1") {
            memory.expires_at = Some(Utc::now() - chrono::Duration::seconds(10));
        }

        let stats = layer.get_stats();
        assert_eq!(stats.expired_count, 1);
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Persistence Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_save_and_load() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("memory.json");

        // Create and save
        {
            let mut layer = MemoryLayer::new();
            layer.remember("key1", json!({"name": "test"}), MemoryScope::Persistent).unwrap();
            layer.remember("key2", json!(42), MemoryScope::Persistent).unwrap();
            layer.save_to_file(&file_path).unwrap();
        }

        // Load and verify
        {
            let mut layer = MemoryLayer::new();
            layer.load_from_file(&file_path).unwrap();

            assert_eq!(layer.len(), 2);
            assert_eq!(layer.recall("key1").unwrap().value["name"], "test");
            assert_eq!(layer.recall("key2").unwrap().value, json!(42));
        }
    }

    #[test]
    fn test_memory_layer_load_nonexistent_file() {
        let mut layer = MemoryLayer::new();

        // Should not error, just remain empty
        let result = layer.load_from_file("/nonexistent/path/memory.json");
        assert!(result.is_ok());
        assert!(layer.is_empty());
    }

    #[test]
    fn test_memory_layer_save_creates_directories() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nested").join("deep").join("memory.json");

        let mut layer = MemoryLayer::new();
        layer.remember("key1", json!(1), MemoryScope::Persistent).unwrap();

        layer.save_to_file(&file_path).unwrap();
        assert!(file_path.exists());
    }

    #[test]
    fn test_memory_layer_dirty_flag() {
        let mut layer = MemoryLayer::new();
        assert!(!layer.is_dirty());

        // Session scope doesn't mark dirty
        layer.remember("key1", json!(1), MemoryScope::Session).unwrap();
        assert!(!layer.is_dirty());

        // Persistent scope marks dirty
        layer.remember("key2", json!(2), MemoryScope::Persistent).unwrap();
        assert!(layer.is_dirty());

        layer.mark_saved();
        assert!(!layer.is_dirty());

        // Forgetting from long-term marks dirty
        layer.forget("key2");
        assert!(layer.is_dirty());
    }

    // -------------------------------------------------------------------------
    // MemoryLayerBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_builder() {
        let layer = MemoryLayerBuilder::new()
            .max_short_term_entries(500)
            .max_long_term_entries(5000)
            .default_ttl_seconds(3600)
            .auto_save_interval_seconds(60)
            .build()
            .unwrap();

        assert_eq!(layer.config().max_short_term_entries, 500);
        assert_eq!(layer.config().max_long_term_entries, 5000);
        assert_eq!(layer.config().default_ttl_seconds, Some(3600));
        assert_eq!(layer.config().auto_save_interval_seconds, 60);
    }

    #[test]
    fn test_memory_layer_builder_with_load() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("memory.json");

        // Create initial data
        {
            let mut layer = MemoryLayer::new();
            layer.remember("existing", json!(42), MemoryScope::Persistent).unwrap();
            layer.save_to_file(&file_path).unwrap();
        }

        // Build with load
        let layer = MemoryLayerBuilder::new()
            .load_from(file_path)
            .build()
            .unwrap();

        assert!(layer.contains_key("existing"));
        assert_eq!(layer.recall("existing").unwrap().value, json!(42));
    }

    #[test]
    fn test_memory_layer_builder_default() {
        let builder = MemoryLayerBuilder::default();
        let layer = builder.build().unwrap();

        assert!(layer.is_empty());
    }

    // -------------------------------------------------------------------------
    // MemoryLayer Keys and Iteration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_keys() {
        let mut layer = MemoryLayer::new();

        layer.remember("session1", json!(1), MemoryScope::Session).unwrap();
        layer.remember("session2", json!(2), MemoryScope::Session).unwrap();
        layer.remember("persistent1", json!(3), MemoryScope::Persistent).unwrap();

        let short_keys: Vec<_> = layer.short_term_keys().collect();
        let long_keys: Vec<_> = layer.long_term_keys().collect();

        assert_eq!(short_keys.len(), 2);
        assert_eq!(long_keys.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Memorable Trait Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memorable_trait_remember() {
        let mut layer = MemoryLayer::new();

        Memorable::remember(&mut layer, "key", json!(42), MemoryScope::Session).unwrap();
        assert!(layer.contains_key("key"));
    }

    #[test]
    fn test_memorable_trait_recall() {
        let mut layer = MemoryLayer::new();
        layer.remember("key", json!(42), MemoryScope::Session).unwrap();

        let memory = Memorable::recall(&layer, "key");
        assert!(memory.is_some());
        assert_eq!(memory.unwrap().value, json!(42));
    }

    #[test]
    fn test_memorable_trait_forget() {
        let mut layer = MemoryLayer::new();
        layer.remember("key", json!(42), MemoryScope::Session).unwrap();

        assert!(Memorable::forget(&mut layer, "key"));
        assert!(!layer.contains_key("key"));
    }

    #[test]
    fn test_memorable_trait_search() {
        let mut layer = MemoryLayer::new();
        layer.remember("user_config", json!(1), MemoryScope::Session).unwrap();
        layer.remember("user_prefs", json!(2), MemoryScope::Session).unwrap();

        let results = Memorable::search(&layer, "user");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_memorable_with_default_ttl() {
        let config = MemoryConfig::new().with_default_ttl(3600);
        let mut layer = MemoryLayer::with_config(config);

        Memorable::remember(&mut layer, "key", json!(42), MemoryScope::Session).unwrap();

        let memory = layer.recall("key").unwrap();
        assert!(memory.expires_at.is_some());
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Error Handling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_layer_overwrite_existing() {
        let mut layer = MemoryLayer::new();

        layer.remember("key", json!(1), MemoryScope::Session).unwrap();
        layer.remember("key", json!(2), MemoryScope::Session).unwrap();

        assert_eq!(layer.len(), 1);
        assert_eq!(layer.recall("key").unwrap().value, json!(2));
    }

    #[test]
    fn test_memory_layer_empty_operations() {
        let layer = MemoryLayer::new();

        assert!(layer.is_empty());
        assert!(layer.recall("anything").is_none());
        assert!(layer.search("anything").is_empty());
    }

    #[test]
    fn test_memory_serialization() {
        let memory = Memory::with_tags(
            "test",
            json!({"nested": {"value": 42}}),
            vec!["tag1".to_string()],
        );

        let serialized = serde_json::to_string(&memory).unwrap();
        let deserialized: Memory = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.key, "test");
        assert_eq!(deserialized.value["nested"]["value"], 42);
        assert!(deserialized.has_tag("tag1"));
    }

    #[test]
    fn test_memory_layer_serialization() {
        let mut layer = MemoryLayer::new();
        layer.remember("key1", json!(1), MemoryScope::Session).unwrap();
        layer.remember("key2", json!(2), MemoryScope::Persistent).unwrap();

        let serialized = serde_json::to_string(&layer).unwrap();
        let deserialized: MemoryLayer = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.len(), 2);
        assert!(deserialized.contains_key("key1"));
        assert!(deserialized.contains_key("key2"));
    }

    #[test]
    fn test_memory_config_serialization() {
        let config = MemoryConfig::new()
            .with_max_short_term(500)
            .with_default_ttl(3600);

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: MemoryConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.max_short_term_entries, 500);
        assert_eq!(deserialized.default_ttl_seconds, Some(3600));
    }

    #[test]
    fn test_memory_scope_serialization() {
        let scopes = vec![
            MemoryScope::Session,
            MemoryScope::Persistent,
            MemoryScope::Global,
            MemoryScope::Agent("agent1".to_string()),
        ];

        for scope in scopes {
            let serialized = serde_json::to_string(&scope).unwrap();
            let deserialized: MemoryScope = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, scope);
        }
    }
}
