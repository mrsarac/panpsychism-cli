//! Caching module for Project Panpsychism.
//!
//! This module provides caching mechanisms to reduce redundant operations:
//!
//! - **LRU Cache**: For search results with configurable capacity
//! - **TTL Cache**: For Gemini API responses with time-based expiration
//! - **Prompt Cache**: For parsed prompt content to avoid re-reading files
//!
//! # Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |   Search Cache   |     |   Gemini Cache   |     |  Prompt Cache    |
//! |   (LRU-based)    |     |   (TTL-based)    |     |  (Content Hash)  |
//! +------------------+     +------------------+     +------------------+
//!          |                       |                       |
//!          +-----------------------+-----------------------+
//!                                  |
//!                         +------------------+
//!                         |  CacheManager    |
//!                         |  (Unified API)   |
//!                         +------------------+
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::cache::{CacheManager, CacheConfig};
//!
//! let config = CacheConfig::default();
//! let cache_manager = CacheManager::new(config);
//!
//! // Search caching
//! let results = cache_manager.search_cache().get_or_insert("query", || {
//!     // Expensive search operation
//!     engine.search(&query).await
//! }).await?;
//!
//! // Get cache statistics
//! let stats = cache_manager.stats();
//! println!("Search cache: {} hits, {} misses", stats.search.hits, stats.search.misses);
//! ```

use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, trace};

use crate::indexer::PromptMetadata;
use crate::search::SearchResult;

// ============================================================================
// Cache Configuration
// ============================================================================

/// Configuration for all cache subsystems.
///
/// # Default Values
///
/// | Setting | Default | Description |
/// |---------|---------|-------------|
/// | `search_capacity` | 1000 | Max cached search queries |
/// | `gemini_capacity` | 100 | Max cached Gemini responses |
/// | `gemini_ttl_secs` | 300 | Gemini cache TTL (5 minutes) |
/// | `prompt_capacity` | 500 | Max cached prompt files |
/// | `enabled` | true | Global cache enable/disable |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of search queries to cache (LRU eviction).
    #[serde(default = "default_search_capacity")]
    pub search_capacity: usize,

    /// Maximum number of Gemini responses to cache.
    #[serde(default = "default_gemini_capacity")]
    pub gemini_capacity: usize,

    /// Time-to-live for Gemini cache entries in seconds.
    #[serde(default = "default_gemini_ttl_secs")]
    pub gemini_ttl_secs: u64,

    /// Maximum number of prompt files to cache.
    #[serde(default = "default_prompt_capacity")]
    pub prompt_capacity: usize,

    /// Global cache enable/disable flag.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_search_capacity() -> usize {
    1000
}

fn default_gemini_capacity() -> usize {
    100
}

fn default_gemini_ttl_secs() -> u64 {
    300 // 5 minutes
}

fn default_prompt_capacity() -> usize {
    500
}

fn default_enabled() -> bool {
    true
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            search_capacity: default_search_capacity(),
            gemini_capacity: default_gemini_capacity(),
            gemini_ttl_secs: default_gemini_ttl_secs(),
            prompt_capacity: default_prompt_capacity(),
            enabled: default_enabled(),
        }
    }
}

// ============================================================================
// Cache Statistics
// ============================================================================

/// Statistics for a single cache subsystem.
#[derive(Debug, Default, Clone, Copy)]
pub struct CacheStats {
    /// Number of cache hits (value found in cache).
    pub hits: u64,
    /// Number of cache misses (value not in cache).
    pub misses: u64,
    /// Number of entries evicted due to capacity limits.
    pub evictions: u64,
    /// Number of entries expired (TTL caches only).
    pub expirations: u64,
    /// Current number of entries in the cache.
    pub size: u64,
}

impl CacheStats {
    /// Calculate the hit rate as a percentage (0.0 - 100.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Get the total number of requests (hits + misses).
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }
}

/// Atomic cache statistics tracker.
///
/// Thread-safe statistics collection for concurrent cache access.
#[derive(Debug, Default)]
pub struct AtomicCacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
}

impl AtomicCacheStats {
    /// Create new statistics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cache hit.
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss.
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an eviction.
    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an expiration.
    pub fn record_expiration(&self) {
        self.expirations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current statistics.
    pub fn snapshot(&self, size: u64) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
            size,
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.expirations.store(0, Ordering::Relaxed);
    }
}

/// Aggregated statistics for all cache subsystems.
#[derive(Debug, Default, Clone)]
pub struct AllCacheStats {
    /// Search cache statistics.
    pub search: CacheStats,
    /// Gemini response cache statistics.
    pub gemini: CacheStats,
    /// Prompt content cache statistics.
    pub prompt: CacheStats,
}

impl AllCacheStats {
    /// Calculate overall hit rate across all caches.
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.search.hits + self.gemini.hits + self.prompt.hits;
        let total_misses = self.search.misses + self.gemini.misses + self.prompt.misses;
        let total = total_hits + total_misses;
        if total == 0 {
            0.0
        } else {
            (total_hits as f64 / total as f64) * 100.0
        }
    }
}

// ============================================================================
// TTL Cache Entry
// ============================================================================

/// A cache entry with time-to-live expiration.
#[derive(Debug, Clone)]
struct TtlEntry<V> {
    /// The cached value.
    value: V,
    /// When this entry expires.
    expires_at: Instant,
}

impl<V> TtlEntry<V> {
    /// Create a new TTL entry.
    fn new(value: V, ttl: Duration) -> Self {
        Self {
            value,
            expires_at: Instant::now() + ttl,
        }
    }

    /// Check if this entry has expired.
    fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }
}

// ============================================================================
// TTL Cache
// ============================================================================

/// A cache with time-to-live expiration for entries.
///
/// Combines LRU eviction with TTL-based expiration. Entries are evicted
/// when they expire or when capacity is reached.
///
/// # Thread Safety
///
/// Uses `Mutex` for thread-safe access. Not suitable for high-contention
/// scenarios - consider sharding for better concurrent performance.
#[derive(Debug)]
pub struct TtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Internal LRU cache storing TTL entries.
    cache: Mutex<LruCache<K, TtlEntry<V>>>,
    /// Default TTL for new entries.
    ttl: Duration,
    /// Statistics tracker.
    stats: AtomicCacheStats,
}

impl<K, V> TtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new TTL cache with specified capacity and TTL.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries
    /// * `ttl` - Time-to-live for entries
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("capacity must be > 0"),
            )),
            ttl,
            stats: AtomicCacheStats::new(),
        }
    }

    /// Get a value from the cache if present and not expired.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().unwrap();

        match cache.get(key) {
            Some(entry) if !entry.is_expired() => {
                self.stats.record_hit();
                trace!("TTL cache hit");
                Some(entry.value.clone())
            }
            Some(_) => {
                // Entry expired - remove it
                cache.pop(key);
                self.stats.record_miss();
                self.stats.record_expiration();
                trace!("TTL cache miss (expired)");
                None
            }
            None => {
                self.stats.record_miss();
                trace!("TTL cache miss");
                None
            }
        }
    }

    /// Insert a value into the cache with default TTL.
    pub fn insert(&self, key: K, value: V) {
        self.insert_with_ttl(key, value, self.ttl);
    }

    /// Insert a value with a custom TTL.
    pub fn insert_with_ttl(&self, key: K, value: V, ttl: Duration) {
        let mut cache = self.cache.lock().unwrap();

        // Check if we're at capacity and will evict
        if cache.len() == cache.cap().get() && !cache.contains(&key) {
            self.stats.record_eviction();
        }

        cache.put(key, TtlEntry::new(value, ttl));
    }

    /// Remove an entry from the cache.
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().unwrap();
        cache.pop(key).map(|entry| entry.value)
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Get the current size of the cache.
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all expired entries.
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired(&self) -> usize {
        let mut cache = self.cache.lock().unwrap();
        let mut expired_keys = Vec::new();

        // Collect expired keys
        for (key, entry) in cache.iter() {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        // Remove expired entries
        for key in &expired_keys {
            cache.pop(key);
            self.stats.record_expiration();
        }

        expired_keys.len()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot(self.len() as u64)
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }
}

// ============================================================================
// Search Cache
// ============================================================================

/// LRU cache for search results.
///
/// Caches search query results to avoid redundant search operations.
/// Keys are normalized (lowercased, trimmed) for consistent caching.
#[derive(Debug)]
pub struct SearchCache {
    /// Internal LRU cache.
    cache: Mutex<LruCache<String, Vec<SearchResult>>>,
    /// Statistics tracker.
    stats: AtomicCacheStats,
    /// Whether caching is enabled.
    enabled: bool,
}

impl SearchCache {
    /// Create a new search cache with specified capacity.
    pub fn new(capacity: usize, enabled: bool) -> Self {
        Self {
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap())),
            stats: AtomicCacheStats::new(),
            enabled,
        }
    }

    /// Normalize a search query for cache key generation.
    fn normalize_key(query: &str) -> String {
        query.to_lowercase().trim().to_string()
    }

    /// Get cached search results for a query.
    pub fn get(&self, query: &str) -> Option<Vec<SearchResult>> {
        if !self.enabled {
            return None;
        }

        let key = Self::normalize_key(query);
        let mut cache = self.cache.lock().unwrap();

        match cache.get(&key) {
            Some(results) => {
                self.stats.record_hit();
                debug!("Search cache hit for query: {}", query);
                Some(results.clone())
            }
            None => {
                self.stats.record_miss();
                None
            }
        }
    }

    /// Insert search results into the cache.
    pub fn insert(&self, query: &str, results: Vec<SearchResult>) {
        if !self.enabled {
            return;
        }

        let key = Self::normalize_key(query);
        let mut cache = self.cache.lock().unwrap();

        // Track eviction
        if cache.len() == cache.cap().get() && !cache.contains(&key) {
            self.stats.record_eviction();
        }

        cache.put(key, results);
    }

    /// Get or compute search results.
    ///
    /// Returns cached results if available, otherwise computes using
    /// the provided function and caches the result.
    pub fn get_or_insert<F>(&self, query: &str, compute: F) -> Vec<SearchResult>
    where
        F: FnOnce() -> Vec<SearchResult>,
    {
        if let Some(cached) = self.get(query) {
            return cached;
        }

        let results = compute();
        self.insert(query, results.clone());
        results
    }

    /// Invalidate (remove) a cached query.
    pub fn invalidate(&self, query: &str) {
        let key = Self::normalize_key(query);
        self.cache.lock().unwrap().pop(&key);
    }

    /// Clear all cached search results.
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Get the current cache size.
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot(self.len() as u64)
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }
}

// ============================================================================
// Gemini Response Cache
// ============================================================================

/// TTL cache for Gemini API responses.
///
/// Caches LLM responses to avoid redundant API calls. Responses are
/// expired after a configurable TTL to ensure freshness for prompts
/// that may have different optimal responses over time.
#[derive(Debug)]
pub struct GeminiCache {
    /// Internal TTL cache.
    cache: TtlCache<String, String>,
    /// Whether caching is enabled.
    enabled: bool,
}

impl GeminiCache {
    /// Create a new Gemini response cache.
    pub fn new(capacity: usize, ttl_secs: u64, enabled: bool) -> Self {
        Self {
            cache: TtlCache::new(capacity.max(1), Duration::from_secs(ttl_secs)),
            enabled,
        }
    }

    /// Generate a cache key from prompt and model.
    fn make_key(prompt: &str, model: &str) -> String {
        // Use a hash for the prompt to keep keys short
        let hash = Self::hash_prompt(prompt);
        format!("{}:{}", model, hash)
    }

    /// Simple hash function for prompts.
    fn hash_prompt(prompt: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        hasher.finish()
    }

    /// Get a cached response for a prompt.
    pub fn get(&self, prompt: &str, model: &str) -> Option<String> {
        if !self.enabled {
            return None;
        }

        let key = Self::make_key(prompt, model);
        self.cache.get(&key)
    }

    /// Insert a response into the cache.
    pub fn insert(&self, prompt: &str, model: &str, response: String) {
        if !self.enabled {
            return;
        }

        let key = Self::make_key(prompt, model);
        self.cache.insert(key, response);
    }

    /// Insert with custom TTL.
    pub fn insert_with_ttl(&self, prompt: &str, model: &str, response: String, ttl: Duration) {
        if !self.enabled {
            return;
        }

        let key = Self::make_key(prompt, model);
        self.cache.insert_with_ttl(key, response, ttl);
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Evict expired entries.
    pub fn evict_expired(&self) -> usize {
        self.cache.evict_expired()
    }

    /// Get cache size.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.cache.reset_stats();
    }
}

// ============================================================================
// Prompt Content Cache
// ============================================================================

/// Cache for parsed prompt file content.
///
/// Caches parsed `PromptMetadata` along with file content to avoid
/// re-reading and re-parsing prompt files from disk.
#[derive(Debug, Clone)]
pub struct CachedPrompt {
    /// Parsed metadata from frontmatter.
    pub metadata: PromptMetadata,
    /// Full prompt content (after frontmatter).
    pub content: String,
    /// File modification time when cached.
    pub cached_at: Instant,
}

/// LRU cache for prompt file content.
#[derive(Debug)]
pub struct PromptCache {
    /// Internal LRU cache: path -> cached prompt.
    cache: RwLock<LruCache<PathBuf, CachedPrompt>>,
    /// Statistics tracker.
    stats: AtomicCacheStats,
    /// Whether caching is enabled.
    enabled: bool,
}

impl PromptCache {
    /// Create a new prompt cache.
    pub fn new(capacity: usize, enabled: bool) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap())),
            stats: AtomicCacheStats::new(),
            enabled,
        }
    }

    /// Get cached prompt content.
    pub fn get(&self, path: &PathBuf) -> Option<CachedPrompt> {
        if !self.enabled {
            return None;
        }

        let mut cache = self.cache.write().unwrap();
        match cache.get(path) {
            Some(cached) => {
                self.stats.record_hit();
                trace!("Prompt cache hit: {:?}", path);
                Some(cached.clone())
            }
            None => {
                self.stats.record_miss();
                None
            }
        }
    }

    /// Insert a prompt into the cache.
    pub fn insert(&self, path: PathBuf, metadata: PromptMetadata, content: String) {
        if !self.enabled {
            return;
        }

        let mut cache = self.cache.write().unwrap();

        // Track eviction
        if cache.len() == cache.cap().get() && !cache.contains(&path) {
            self.stats.record_eviction();
        }

        cache.put(
            path,
            CachedPrompt {
                metadata,
                content,
                cached_at: Instant::now(),
            },
        );
    }

    /// Invalidate a cached prompt (e.g., when file changes).
    pub fn invalidate(&self, path: &PathBuf) {
        self.cache.write().unwrap().pop(path);
    }

    /// Invalidate all prompts matching a pattern.
    pub fn invalidate_prefix(&self, prefix: &PathBuf) {
        let mut cache = self.cache.write().unwrap();
        let keys_to_remove: Vec<_> = cache
            .iter()
            .filter(|(path, _)| path.starts_with(prefix))
            .map(|(path, _)| path.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
        }
    }

    /// Clear all cached prompts.
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
    }

    /// Get cache size.
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot(self.len() as u64)
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }
}

// ============================================================================
// Cache Manager
// ============================================================================

/// Unified cache manager for all subsystems.
///
/// Provides a single point of access for all caching functionality
/// with aggregated statistics and configuration.
#[derive(Debug)]
pub struct CacheManager {
    /// Configuration.
    config: CacheConfig,
    /// Search result cache.
    search: Arc<SearchCache>,
    /// Gemini response cache.
    gemini: Arc<GeminiCache>,
    /// Prompt content cache.
    prompt: Arc<PromptCache>,
}

impl CacheManager {
    /// Create a new cache manager with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            search: Arc::new(SearchCache::new(config.search_capacity, config.enabled)),
            gemini: Arc::new(GeminiCache::new(
                config.gemini_capacity,
                config.gemini_ttl_secs,
                config.enabled,
            )),
            prompt: Arc::new(PromptCache::new(config.prompt_capacity, config.enabled)),
            config,
        }
    }

    /// Create a cache manager with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get access to the search cache.
    pub fn search(&self) -> &SearchCache {
        &self.search
    }

    /// Get access to the Gemini cache.
    pub fn gemini(&self) -> &GeminiCache {
        &self.gemini
    }

    /// Get access to the prompt cache.
    pub fn prompt(&self) -> &PromptCache {
        &self.prompt
    }

    /// Get the current configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Get aggregated statistics for all caches.
    pub fn stats(&self) -> AllCacheStats {
        AllCacheStats {
            search: self.search.stats(),
            gemini: self.gemini.stats(),
            prompt: self.prompt.stats(),
        }
    }

    /// Reset all statistics.
    pub fn reset_stats(&self) {
        self.search.reset_stats();
        self.gemini.reset_stats();
        self.prompt.reset_stats();
    }

    /// Clear all caches.
    pub fn clear_all(&self) {
        self.search.clear();
        self.gemini.clear();
        self.prompt.clear();
    }

    /// Evict expired entries from TTL-based caches.
    pub fn evict_expired(&self) -> usize {
        self.gemini.evict_expired()
    }

    /// Check if caching is globally enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Clone for CacheManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            search: Arc::clone(&self.search),
            gemini: Arc::clone(&self.gemini),
            prompt: Arc::clone(&self.prompt),
        }
    }
}

// ============================================================================
// Cached Search Engine Wrapper
// ============================================================================

use crate::search::{SearchEngine, SearchQuery};
use crate::Result;

/// A search engine wrapper that caches results.
///
/// Wraps a `SearchEngine` and automatically caches search results
/// using an LRU cache.
#[derive(Debug)]
pub struct CachedSearchEngine {
    /// The underlying search engine.
    engine: SearchEngine,
    /// Search result cache.
    cache: SearchCache,
}

impl CachedSearchEngine {
    /// Create a new cached search engine.
    pub fn new(engine: SearchEngine, capacity: usize) -> Self {
        Self {
            engine,
            cache: SearchCache::new(capacity, true),
        }
    }

    /// Search with caching.
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        // Check cache first
        if let Some(cached) = self.cache.get(&query.query) {
            return Ok(cached);
        }

        // Cache miss - perform search
        let results = self.engine.search(query).await?;

        // Cache the results
        self.cache.insert(&query.query, results.clone());

        Ok(results)
    }

    /// Get the underlying engine.
    pub fn engine(&self) -> &SearchEngine {
        &self.engine
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get the index size from the underlying engine.
    pub fn index_size(&self) -> usize {
        self.engine.index_size()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // TTL Cache Tests

    #[test]
    fn test_ttl_cache_basic() {
        let cache: TtlCache<String, i32> = TtlCache::new(10, Duration::from_secs(60));

        cache.insert("key1".to_string(), 42);
        assert_eq!(cache.get(&"key1".to_string()), Some(42));
        assert_eq!(cache.get(&"key2".to_string()), None);
    }

    #[test]
    fn test_ttl_cache_expiration() {
        let cache: TtlCache<String, i32> = TtlCache::new(10, Duration::from_millis(50));

        cache.insert("key".to_string(), 42);
        assert_eq!(cache.get(&"key".to_string()), Some(42));

        // Wait for expiration
        thread::sleep(Duration::from_millis(100));

        assert_eq!(cache.get(&"key".to_string()), None);
    }

    #[test]
    fn test_ttl_cache_evict_expired() {
        let cache: TtlCache<String, i32> = TtlCache::new(10, Duration::from_millis(50));

        cache.insert("key1".to_string(), 1);
        cache.insert("key2".to_string(), 2);

        thread::sleep(Duration::from_millis(100));

        let evicted = cache.evict_expired();
        assert_eq!(evicted, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_ttl_cache_lru_eviction() {
        let cache: TtlCache<String, i32> = TtlCache::new(2, Duration::from_secs(60));

        cache.insert("key1".to_string(), 1);
        cache.insert("key2".to_string(), 2);
        cache.insert("key3".to_string(), 3); // Should evict key1

        assert_eq!(cache.get(&"key1".to_string()), None);
        assert_eq!(cache.get(&"key2".to_string()), Some(2));
        assert_eq!(cache.get(&"key3".to_string()), Some(3));
    }

    #[test]
    fn test_ttl_cache_stats() {
        let cache: TtlCache<String, i32> = TtlCache::new(10, Duration::from_secs(60));

        cache.get(&"miss".to_string()); // miss
        cache.insert("hit".to_string(), 42);
        cache.get(&"hit".to_string()); // hit

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 50.0);
    }

    // Search Cache Tests

    #[test]
    fn test_search_cache_normalize_key() {
        let cache = SearchCache::new(10, true);

        // Same query with different casing should use same cache entry
        cache.insert("Hello World", vec![]);
        assert!(cache.get("HELLO WORLD").is_some());
        assert!(cache.get("  hello world  ").is_some());
    }

    #[test]
    fn test_search_cache_disabled() {
        let cache = SearchCache::new(10, false);

        cache.insert("query", vec![]);
        assert!(cache.get("query").is_none());
    }

    #[test]
    fn test_search_cache_get_or_insert() {
        let cache = SearchCache::new(10, true);
        let mut compute_count = 0;

        // First call - computes
        let result1 = cache.get_or_insert("query", || {
            compute_count += 1;
            vec![]
        });

        // Second call - from cache
        let result2 = cache.get_or_insert("query", || {
            compute_count += 1;
            vec![]
        });

        assert_eq!(compute_count, 1);
        assert_eq!(result1.len(), result2.len());
    }

    // Gemini Cache Tests

    #[test]
    fn test_gemini_cache_key_generation() {
        let cache = GeminiCache::new(10, 60, true);

        cache.insert("prompt1", "model-a", "response1".to_string());
        cache.insert("prompt1", "model-b", "response2".to_string());

        // Different models should have different cache entries
        assert_eq!(
            cache.get("prompt1", "model-a"),
            Some("response1".to_string())
        );
        assert_eq!(
            cache.get("prompt1", "model-b"),
            Some("response2".to_string())
        );
    }

    #[test]
    fn test_gemini_cache_disabled() {
        let cache = GeminiCache::new(10, 60, false);

        cache.insert("prompt", "model", "response".to_string());
        assert!(cache.get("prompt", "model").is_none());
    }

    // Prompt Cache Tests

    #[test]
    fn test_prompt_cache_basic() {
        let cache = PromptCache::new(10, true);

        let metadata = PromptMetadata {
            title: "Test".to_string(),
            description: "Test desc".to_string(),
            tags: vec!["tag1".to_string()],
            category: "test".to_string(),
            privacy_tier: None,
            author: None,
            version: None,
        };

        let path = PathBuf::from("test.md");
        cache.insert(path.clone(), metadata.clone(), "content".to_string());

        let cached = cache.get(&path).unwrap();
        assert_eq!(cached.metadata.title, "Test");
        assert_eq!(cached.content, "content");
    }

    #[test]
    fn test_prompt_cache_invalidate_prefix() {
        let cache = PromptCache::new(10, true);

        let metadata = PromptMetadata {
            title: "Test".to_string(),
            description: "".to_string(),
            tags: vec![],
            category: "".to_string(),
            privacy_tier: None,
            author: None,
            version: None,
        };

        cache.insert(
            PathBuf::from("category/a.md"),
            metadata.clone(),
            "a".to_string(),
        );
        cache.insert(
            PathBuf::from("category/b.md"),
            metadata.clone(),
            "b".to_string(),
        );
        cache.insert(
            PathBuf::from("other/c.md"),
            metadata.clone(),
            "c".to_string(),
        );

        assert_eq!(cache.len(), 3);

        cache.invalidate_prefix(&PathBuf::from("category"));
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&PathBuf::from("other/c.md")).is_some());
    }

    // Cache Manager Tests

    #[test]
    fn test_cache_manager_defaults() {
        let manager = CacheManager::with_defaults();

        assert!(manager.is_enabled());
        assert!(manager.search().is_empty());
        assert!(manager.gemini().is_empty());
        assert!(manager.prompt().is_empty());
    }

    #[test]
    fn test_cache_manager_aggregated_stats() {
        let manager = CacheManager::with_defaults();

        // Generate some stats
        manager.search().get("miss1");
        manager.search().get("miss2");
        manager.gemini().get("miss", "model");

        let stats = manager.stats();
        assert_eq!(stats.search.misses, 2);
        assert_eq!(stats.gemini.misses, 1);
        assert_eq!(stats.overall_hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_manager_clear_all() {
        let manager = CacheManager::with_defaults();

        manager.search().insert("query", vec![]);
        manager
            .gemini()
            .insert("prompt", "model", "response".to_string());

        assert!(!manager.search().is_empty());
        assert!(!manager.gemini().is_empty());

        manager.clear_all();

        assert!(manager.search().is_empty());
        assert!(manager.gemini().is_empty());
    }

    #[test]
    fn test_cache_manager_clone_shares_caches() {
        let manager1 = CacheManager::with_defaults();
        let manager2 = manager1.clone();

        manager1.search().insert("query", vec![]);

        // Both managers should see the same cached data
        assert!(manager2.search().get("query").is_some());
    }

    // CacheConfig Tests

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();

        assert_eq!(config.search_capacity, 1000);
        assert_eq!(config.gemini_capacity, 100);
        assert_eq!(config.gemini_ttl_secs, 300);
        assert_eq!(config.prompt_capacity, 500);
        assert!(config.enabled);
    }

    #[test]
    fn test_cache_config_serialization() {
        let config = CacheConfig {
            search_capacity: 500,
            gemini_capacity: 50,
            gemini_ttl_secs: 600,
            prompt_capacity: 250,
            enabled: false,
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: CacheConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(deserialized.search_capacity, 500);
        assert_eq!(deserialized.gemini_ttl_secs, 600);
        assert!(!deserialized.enabled);
    }

    // CacheStats Tests

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 75,
            misses: 25,
            evictions: 0,
            expirations: 0,
            size: 50,
        };

        assert_eq!(stats.hit_rate(), 75.0);
        assert_eq!(stats.total_requests(), 100);
    }

    #[test]
    fn test_cache_stats_empty() {
        let stats = CacheStats::default();

        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.total_requests(), 0);
    }

    // AtomicCacheStats Tests

    #[test]
    fn test_atomic_cache_stats_thread_safety() {
        let stats = Arc::new(AtomicCacheStats::new());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let stats = Arc::clone(&stats);
                thread::spawn(move || {
                    for _ in 0..100 {
                        stats.record_hit();
                        stats.record_miss();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = stats.snapshot(0);
        assert_eq!(snapshot.hits, 1000);
        assert_eq!(snapshot.misses, 1000);
    }
}
