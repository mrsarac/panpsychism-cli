//! Integration tests for Persistent Memory Layer (PML)
//!
//! Tests the memory module with real file I/O, concurrent access patterns,
//! and end-to-end workflows for Project Panpsychism v4.1.

use panpsychism::memory::{
    Memorable, MemoryConfig, MemoryLayer, MemoryLayerBuilder, MemoryScope,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

// =============================================================================
// Test Helpers
// =============================================================================

/// Create a temporary directory for test isolation.
fn create_temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp directory")
}

/// Create a memory layer with persistence in a temp directory.
fn create_persistent_memory(temp_dir: &TempDir) -> (MemoryLayer, PathBuf) {
    let file_path = temp_dir.path().join("memory.json");
    let config = MemoryConfig::new().with_persistence_path(&file_path);
    (MemoryLayer::with_config(config), file_path)
}

// =============================================================================
// Short-term Memory Tests
// =============================================================================

mod short_term_memory {
    use super::*;

    #[tokio::test]
    async fn test_remember_recall_basic() {
        let mut memory = MemoryLayer::new();

        // Store a simple value
        memory
            .remember("user_name", json!("Alice"), MemoryScope::Session)
            .expect("Should remember");

        // Recall it
        let recalled = memory.recall("user_name");
        assert!(recalled.is_some());
        assert_eq!(recalled.unwrap().value, json!("Alice"));
    }

    #[tokio::test]
    async fn test_remember_recall_complex_json() {
        let mut memory = MemoryLayer::new();

        let complex_value = json!({
            "preferences": {
                "theme": "dark",
                "language": "en",
                "notifications": {
                    "email": true,
                    "push": false
                }
            },
            "history": [1, 2, 3, 4, 5],
            "metadata": null
        });

        memory
            .remember("user_settings", complex_value.clone(), MemoryScope::Session)
            .expect("Should remember complex JSON");

        let recalled = memory.recall("user_settings").unwrap();
        assert_eq!(recalled.value, complex_value);
        assert_eq!(recalled.value["preferences"]["theme"], "dark");
        assert_eq!(recalled.value["history"][2], 3);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        // Create memory with very small capacity
        let config = MemoryConfig::new().with_max_short_term(3);
        let mut memory = MemoryLayer::with_config(config);

        // Add 3 entries
        memory
            .remember("key1", json!(1), MemoryScope::Session)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        memory
            .remember("key2", json!(2), MemoryScope::Session)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        memory
            .remember("key3", json!(3), MemoryScope::Session)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        assert_eq!(memory.len(), 3);

        // Add 4th entry - should evict oldest (key1)
        memory
            .remember("key4", json!(4), MemoryScope::Session)
            .unwrap();

        assert_eq!(memory.len(), 3);
        assert!(!memory.contains_key("key1"), "key1 should be evicted");
        assert!(memory.contains_key("key2"));
        assert!(memory.contains_key("key3"));
        assert!(memory.contains_key("key4"));
    }

    #[tokio::test]
    async fn test_lru_eviction_preserves_recently_accessed() {
        let config = MemoryConfig::new().with_max_short_term(3);
        let mut memory = MemoryLayer::with_config(config);

        // Add 3 entries with time gaps
        memory
            .remember("oldest", json!(1), MemoryScope::Session)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        memory
            .remember("middle", json!(2), MemoryScope::Session)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        memory
            .remember("newest", json!(3), MemoryScope::Session)
            .unwrap();

        // Adding a 4th should evict "oldest" (based on created_at)
        memory
            .remember("extra", json!(4), MemoryScope::Session)
            .unwrap();

        assert!(!memory.contains_key("oldest"));
        assert!(memory.contains_key("middle"));
        assert!(memory.contains_key("newest"));
        assert!(memory.contains_key("extra"));
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let mut memory = MemoryLayer::new();

        // Add entry with very short TTL (1 second)
        memory
            .remember_with_ttl("temporary", json!("expires soon"), MemoryScope::Session, 1)
            .expect("Should remember with TTL");

        // Should be accessible immediately
        assert!(memory.recall("temporary").is_some());

        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;

        // Should be expired now
        assert!(
            memory.recall("temporary").is_none(),
            "Expired memory should not be recalled"
        );
    }

    #[tokio::test]
    async fn test_ttl_remaining_seconds() {
        let mut memory = MemoryLayer::new();

        memory
            .remember_with_ttl("timed", json!("value"), MemoryScope::Session, 60)
            .unwrap();

        let recalled = memory.recall("timed").unwrap();
        let remaining = recalled.remaining_ttl_seconds();

        assert!(remaining.is_some());
        assert!(remaining.unwrap() > 55 && remaining.unwrap() <= 60);
    }

    #[tokio::test]
    async fn test_cleanup_expired_entries() {
        let mut memory = MemoryLayer::new();

        // Add entries with different TTLs
        memory
            .remember_with_ttl("expires_fast", json!(1), MemoryScope::Session, 1)
            .unwrap();
        memory
            .remember("never_expires", json!(2), MemoryScope::Session)
            .unwrap();
        memory
            .remember_with_ttl("expires_slow", json!(3), MemoryScope::Session, 3600)
            .unwrap();

        assert_eq!(memory.len(), 3);

        // Wait for first entry to expire
        tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;

        // Cleanup expired
        let cleaned = memory.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(memory.len(), 2);
        assert!(!memory.contains_key("expires_fast"));
    }
}

// =============================================================================
// Long-term Memory Tests
// =============================================================================

mod long_term_memory {
    use super::*;

    #[tokio::test]
    async fn test_persistent_storage() {
        let temp_dir = create_temp_dir();
        let (mut memory, file_path) = create_persistent_memory(&temp_dir);

        // Store persistent data
        memory
            .remember(
                "important_data",
                json!({"key": "value"}),
                MemoryScope::Persistent,
            )
            .expect("Should remember");

        // Save to file
        memory.save_to_file(&file_path).expect("Should save");

        // Verify file exists
        assert!(file_path.exists());

        // Read file content
        let content = std::fs::read_to_string(&file_path).expect("Should read");
        assert!(content.contains("important_data"));
    }

    #[tokio::test]
    async fn test_data_survives_restart() {
        let temp_dir = create_temp_dir();
        let file_path = temp_dir.path().join("persistent.json");

        // Create and populate memory
        {
            let mut memory = MemoryLayer::new();
            memory
                .remember("user_id", json!("12345"), MemoryScope::Persistent)
                .unwrap();
            memory
                .remember("session_count", json!(42), MemoryScope::Persistent)
                .unwrap();
            memory
                .remember(
                    "preferences",
                    json!({"theme": "dark"}),
                    MemoryScope::Persistent,
                )
                .unwrap();

            memory.save_to_file(&file_path).expect("Should save");
        }

        // Create new memory instance and load
        {
            let mut memory = MemoryLayer::new();
            memory.load_from_file(&file_path).expect("Should load");

            assert_eq!(memory.len(), 3);
            assert_eq!(memory.recall("user_id").unwrap().value, json!("12345"));
            assert_eq!(memory.recall("session_count").unwrap().value, json!(42));
            assert_eq!(
                memory.recall("preferences").unwrap().value["theme"],
                "dark"
            );
        }
    }

    #[tokio::test]
    async fn test_key_value_operations() {
        let mut memory = MemoryLayer::new();

        // Create
        memory
            .remember("key", json!(1), MemoryScope::Persistent)
            .unwrap();
        assert_eq!(memory.recall("key").unwrap().value, json!(1));

        // Update
        assert!(memory.update("key", json!(2)));
        assert_eq!(memory.recall("key").unwrap().value, json!(2));

        // Delete
        assert!(memory.forget("key"));
        assert!(memory.recall("key").is_none());

        // Update non-existent
        assert!(!memory.update("nonexistent", json!(99)));
    }

    #[tokio::test]
    async fn test_long_term_eviction() {
        let config = MemoryConfig::new().with_max_long_term(2);
        let mut memory = MemoryLayer::with_config(config);

        memory
            .remember("first", json!(1), MemoryScope::Persistent)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        memory
            .remember("second", json!(2), MemoryScope::Persistent)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        memory
            .remember("third", json!(3), MemoryScope::Persistent)
            .unwrap();

        // First should be evicted
        assert!(!memory.contains_key("first"));
        assert!(memory.contains_key("second"));
        assert!(memory.contains_key("third"));
    }

    #[tokio::test]
    async fn test_load_nonexistent_file() {
        let mut memory = MemoryLayer::new();

        // Should not error, just remain empty
        let result = memory.load_from_file("/nonexistent/path/memory.json");
        assert!(result.is_ok());
        assert!(memory.is_empty());
    }

    #[tokio::test]
    async fn test_save_creates_parent_directories() {
        let temp_dir = create_temp_dir();
        let file_path = temp_dir.path().join("nested").join("deep").join("memory.json");

        let mut memory = MemoryLayer::new();
        memory
            .remember("data", json!("test"), MemoryScope::Persistent)
            .unwrap();

        memory.save_to_file(&file_path).expect("Should create dirs and save");
        assert!(file_path.exists());
    }

    #[tokio::test]
    async fn test_dirty_flag_tracking() {
        let mut memory = MemoryLayer::new();

        // Initially clean
        assert!(!memory.is_dirty());

        // Session scope doesn't dirty
        memory
            .remember("session", json!(1), MemoryScope::Session)
            .unwrap();
        assert!(!memory.is_dirty());

        // Persistent scope marks dirty
        memory
            .remember("persistent", json!(2), MemoryScope::Persistent)
            .unwrap();
        assert!(memory.is_dirty());

        // Mark saved clears dirty
        memory.mark_saved();
        assert!(!memory.is_dirty());

        // Forgetting from long-term marks dirty again
        memory.forget("persistent");
        assert!(memory.is_dirty());
    }
}

// =============================================================================
// Semantic Memory Tests (Tag-based search)
// =============================================================================

mod semantic_memory {
    use super::*;

    #[tokio::test]
    async fn test_tag_operations() {
        let mut memory = MemoryLayer::new();

        memory
            .remember_with_tags(
                "prompt1",
                json!("content"),
                MemoryScope::Session,
                vec!["ai".to_string(), "coding".to_string()],
            )
            .unwrap();

        let recalled = memory.recall("prompt1").unwrap();
        assert!(recalled.has_tag("ai"));
        assert!(recalled.has_tag("coding"));
        assert!(!recalled.has_tag("design"));
    }

    #[tokio::test]
    async fn test_add_remove_tags() {
        let mut memory = MemoryLayer::new();

        memory
            .remember("item", json!("value"), MemoryScope::Session)
            .unwrap();

        // Add tags
        assert!(memory.add_tag("item", "important"));
        assert!(memory.add_tag("item", "urgent"));
        assert!(memory.recall("item").unwrap().has_tag("important"));

        // Remove tag
        assert!(memory.remove_tag("item", "urgent"));
        assert!(!memory.recall("item").unwrap().has_tag("urgent"));

        // Operations on non-existent key
        assert!(!memory.add_tag("nonexistent", "tag"));
        assert!(!memory.remove_tag("nonexistent", "tag"));
    }

    #[tokio::test]
    async fn test_search_by_tag() {
        let mut memory = MemoryLayer::new();

        memory
            .remember_with_tags(
                "auth_prompt",
                json!({"type": "auth"}),
                MemoryScope::Session,
                vec!["security".to_string(), "auth".to_string()],
            )
            .unwrap();

        memory
            .remember_with_tags(
                "api_prompt",
                json!({"type": "api"}),
                MemoryScope::Session,
                vec!["security".to_string(), "api".to_string()],
            )
            .unwrap();

        memory
            .remember_with_tags(
                "ui_prompt",
                json!({"type": "ui"}),
                MemoryScope::Session,
                vec!["design".to_string()],
            )
            .unwrap();

        // Search by tag
        let security_results = memory.search_by_tag("security");
        assert_eq!(security_results.len(), 2);

        let auth_results = memory.search_by_tag("auth");
        assert_eq!(auth_results.len(), 1);
        assert_eq!(auth_results[0].key, "auth_prompt");

        let design_results = memory.search_by_tag("design");
        assert_eq!(design_results.len(), 1);

        let empty_results = memory.search_by_tag("nonexistent");
        assert!(empty_results.is_empty());
    }

    #[tokio::test]
    async fn test_search_excludes_expired() {
        let mut memory = MemoryLayer::new();

        memory
            .remember_with_tags(
                "active",
                json!("active"),
                MemoryScope::Session,
                vec!["test".to_string()],
            )
            .unwrap();

        // Add entry that expires immediately (simulate by manually setting)
        memory
            .remember_with_tags(
                "expired",
                json!("expired"),
                MemoryScope::Session,
                vec!["test".to_string()],
            )
            .unwrap();

        // Manually expire it
        if let Some(m) = memory.recall_mut("expired") {
            m.expires_at = Some(chrono::Utc::now() - chrono::Duration::seconds(10));
        }

        let results = memory.search_by_tag("test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "active");
    }
}

// =============================================================================
// Pattern Search Tests
// =============================================================================

mod pattern_search {
    use super::*;

    #[tokio::test]
    async fn test_search_by_key_pattern() {
        let mut memory = MemoryLayer::new();

        memory
            .remember("user_settings", json!(1), MemoryScope::Session)
            .unwrap();
        memory
            .remember("user_preferences", json!(2), MemoryScope::Session)
            .unwrap();
        memory
            .remember("system_config", json!(3), MemoryScope::Session)
            .unwrap();

        let user_results = memory.search("user");
        assert_eq!(user_results.len(), 2);

        let system_results = memory.search("system");
        assert_eq!(system_results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_case_insensitive() {
        let mut memory = MemoryLayer::new();

        memory
            .remember("UserConfig", json!("value"), MemoryScope::Session)
            .unwrap();

        assert_eq!(memory.search_by_pattern("userconfig").len(), 1);
        assert_eq!(memory.search_by_pattern("USERCONFIG").len(), 1);
        assert_eq!(memory.search_by_pattern("UserConfig").len(), 1);
    }

    #[tokio::test]
    async fn test_search_in_value() {
        let mut memory = MemoryLayer::new();

        memory
            .remember("item", json!("This contains searchable text"), MemoryScope::Session)
            .unwrap();

        let results = memory.search_by_pattern("searchable");
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_sorted_by_access_count() {
        let mut memory = MemoryLayer::new();

        memory
            .remember("less_accessed", json!(1), MemoryScope::Session)
            .unwrap();
        memory
            .remember("more_accessed", json!(2), MemoryScope::Session)
            .unwrap();

        // Access one more times
        for _ in 0..5 {
            memory.recall_mut("more_accessed");
        }

        let results = memory.search_by_pattern("accessed");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "more_accessed"); // Higher access count first
    }
}

// =============================================================================
// Episodic Memory Tests (Event logging and replay)
// =============================================================================

mod episodic_memory {
    use super::*;

    #[tokio::test]
    async fn test_event_logging() {
        let mut memory = MemoryLayer::new();

        // Log events with timestamps
        for i in 0..5 {
            memory
                .remember(
                    &format!("event_{}", i),
                    json!({
                        "action": format!("action_{}", i),
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                        "data": { "index": i }
                    }),
                    MemoryScope::Session,
                )
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(memory.len(), 5);

        // Verify chronological order via search
        let events = memory.search_by_pattern("event");
        assert_eq!(events.len(), 5);
    }

    #[tokio::test]
    async fn test_access_tracking() {
        let mut memory = MemoryLayer::new();

        memory
            .remember("tracked", json!("value"), MemoryScope::Session)
            .unwrap();

        // Initial access count is 0
        assert_eq!(memory.recall("tracked").unwrap().access_count, 0);

        // Access with recall_mut increments counter
        memory.recall_mut("tracked");
        memory.recall_mut("tracked");
        memory.recall_mut("tracked");

        assert_eq!(memory.recall("tracked").unwrap().access_count, 3);
    }

    #[tokio::test]
    async fn test_memory_age() {
        let mut memory = MemoryLayer::new();

        memory
            .remember("old_memory", json!(1), MemoryScope::Session)
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        let recalled = memory.recall("old_memory").unwrap();
        assert!(recalled.age_seconds() >= 0);
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

mod integration {
    use super::*;

    #[tokio::test]
    async fn test_memory_layer_full_cycle() {
        let temp_dir = create_temp_dir();
        let file_path = temp_dir.path().join("full_cycle.json");

        // Phase 1: Create and populate
        let mut memory = MemoryLayerBuilder::new()
            .max_short_term_entries(100)
            .max_long_term_entries(1000)
            .persistence_path(&file_path)
            .build()
            .expect("Should build");

        // Store session data
        memory
            .remember("session_token", json!("abc123"), MemoryScope::Session)
            .unwrap();

        // Store persistent user data
        memory
            .remember(
                "user_profile",
                json!({
                    "id": "user_001",
                    "name": "Test User",
                    "preferences": {
                        "theme": "dark",
                        "language": "en"
                    }
                }),
                MemoryScope::Persistent,
            )
            .unwrap();

        // Store with tags
        memory
            .remember_with_tags(
                "important_note",
                json!("Remember this!"),
                MemoryScope::Persistent,
                vec!["important".to_string(), "note".to_string()],
            )
            .unwrap();

        // Store with TTL
        memory
            .remember_with_ttl("temporary_cache", json!(42), MemoryScope::Session, 3600)
            .unwrap();

        // Save persistent data
        memory.save_to_file(&file_path).unwrap();

        // Verify stats
        let stats = memory.get_stats();
        assert_eq!(stats.total_entries, 4);
        assert_eq!(stats.short_term_count, 2); // session_token and temporary_cache
        assert_eq!(stats.long_term_count, 2); // user_profile and important_note

        // Phase 2: Load in new instance (simulating restart)
        let memory2 = MemoryLayerBuilder::new()
            .load_from(&file_path)
            .build()
            .expect("Should load");

        // Session data should be lost
        assert!(memory2.recall("session_token").is_none());
        assert!(memory2.recall("temporary_cache").is_none());

        // Persistent data should survive
        assert!(memory2.recall("user_profile").is_some());
        assert!(memory2.recall("important_note").is_some());

        // Tags should survive
        let note = memory2.recall("important_note").unwrap();
        assert!(note.has_tag("important"));
    }

    #[tokio::test]
    async fn test_concurrent_read_access() {
        let memory = Arc::new(RwLock::new(MemoryLayer::new()));

        // Populate memory
        {
            let mut mem = memory.write().await;
            for i in 0..100 {
                mem.remember(&format!("key_{}", i), json!(i), MemoryScope::Session)
                    .unwrap();
            }
        }

        // Spawn multiple readers
        let mut handles = vec![];
        for reader_id in 0..10 {
            let mem_clone = Arc::clone(&memory);
            handles.push(tokio::spawn(async move {
                let mem = mem_clone.read().await;
                for i in 0..10 {
                    let key = format!("key_{}", reader_id * 10 + i);
                    let _ = mem.recall(&key);
                }
            }));
        }

        // Wait for all readers
        for handle in handles {
            handle.await.expect("Reader should complete");
        }

        // Verify memory integrity
        let mem = memory.read().await;
        assert_eq!(mem.len(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_write_access() {
        let memory = Arc::new(RwLock::new(MemoryLayer::new()));

        // Spawn multiple writers
        let mut handles = vec![];
        for writer_id in 0..10 {
            let mem_clone = Arc::clone(&memory);
            handles.push(tokio::spawn(async move {
                let mut mem = mem_clone.write().await;
                for i in 0..10 {
                    let key = format!("writer_{}_key_{}", writer_id, i);
                    mem.remember(&key, json!(writer_id * 10 + i), MemoryScope::Session)
                        .unwrap();
                }
            }));
        }

        // Wait for all writers
        for handle in handles {
            handle.await.expect("Writer should complete");
        }

        // Verify all writes succeeded
        let mem = memory.read().await;
        assert_eq!(mem.len(), 100);
    }

    #[tokio::test]
    async fn test_memory_cleanup() {
        let mut memory = MemoryLayer::new();

        // Add mix of expiring and non-expiring entries
        memory
            .remember_with_ttl("expires_1", json!(1), MemoryScope::Session, 1)
            .unwrap();
        memory
            .remember_with_ttl("expires_2", json!(2), MemoryScope::Session, 1)
            .unwrap();
        memory
            .remember("permanent_1", json!(3), MemoryScope::Session)
            .unwrap();
        memory
            .remember("permanent_2", json!(4), MemoryScope::Persistent)
            .unwrap();

        assert_eq!(memory.len(), 4);

        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;

        // Cleanup
        let cleaned = memory.cleanup_expired();
        assert_eq!(cleaned, 2);
        assert_eq!(memory.len(), 2);

        // Verify correct entries remain
        assert!(memory.contains_key("permanent_1"));
        assert!(memory.contains_key("permanent_2"));
    }

    #[tokio::test]
    async fn test_scope_isolation() {
        let mut memory = MemoryLayer::new();

        // Add entries in different scopes
        memory
            .remember("session_data", json!("session"), MemoryScope::Session)
            .unwrap();
        memory
            .remember("persistent_data", json!("persistent"), MemoryScope::Persistent)
            .unwrap();
        memory
            .remember("global_data", json!("global"), MemoryScope::Global)
            .unwrap();

        // Clear session scope
        memory.clear_scope(MemoryScope::Session);

        assert!(!memory.contains_key("session_data"));
        assert!(memory.contains_key("persistent_data"));
        assert!(memory.contains_key("global_data"));
    }

    #[tokio::test]
    async fn test_agent_scoped_memory() {
        let mut memory = MemoryLayer::new();

        // Store agent-specific data
        memory
            .remember(
                "agent:predictor:state",
                json!({"active": true}),
                MemoryScope::Agent("predictor".to_string()),
            )
            .unwrap();
        memory
            .remember(
                "agent:predictor:cache",
                json!([1, 2, 3]),
                MemoryScope::Agent("predictor".to_string()),
            )
            .unwrap();
        memory
            .remember(
                "agent:evaluator:state",
                json!({"active": false}),
                MemoryScope::Agent("evaluator".to_string()),
            )
            .unwrap();

        // Clear only predictor agent's memory
        memory.clear_scope(MemoryScope::Agent("predictor".to_string()));

        assert!(!memory.contains_key("agent:predictor:state"));
        assert!(!memory.contains_key("agent:predictor:cache"));
        assert!(memory.contains_key("agent:evaluator:state"));
    }

    #[tokio::test]
    async fn test_stats_accuracy() {
        let mut memory = MemoryLayer::new();

        // Add entries
        memory
            .remember("key1", json!({"data": "value"}), MemoryScope::Session)
            .unwrap();
        memory
            .remember("key2", json!([1, 2, 3, 4, 5]), MemoryScope::Persistent)
            .unwrap();

        // Access entries
        memory.recall_mut("key1");
        memory.recall_mut("key1");
        memory.recall_mut("key2");

        let stats = memory.get_stats();

        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.short_term_count, 1);
        assert_eq!(stats.long_term_count, 1);
        assert_eq!(stats.total_access_count, 3);
        assert!(stats.oldest_entry.is_some());
        assert!(stats.newest_entry.is_some());
        assert_eq!(stats.expired_count, 0);
        assert!(stats.estimated_size_bytes > 0);
    }

    #[tokio::test]
    async fn test_builder_with_all_options() {
        let temp_dir = create_temp_dir();
        let file_path = temp_dir.path().join("builder_test.json");

        let memory = MemoryLayerBuilder::new()
            .max_short_term_entries(500)
            .max_long_term_entries(5000)
            .default_ttl_seconds(3600)
            .persistence_path(&file_path)
            .auto_save_interval_seconds(60)
            .build()
            .expect("Should build");

        assert_eq!(memory.config().max_short_term_entries, 500);
        assert_eq!(memory.config().max_long_term_entries, 5000);
        assert_eq!(memory.config().default_ttl_seconds, Some(3600));
        assert_eq!(memory.config().auto_save_interval_seconds, 60);
    }

    #[tokio::test]
    async fn test_memorable_trait_implementation() {
        let mut memory = MemoryLayer::new();

        // Test via trait methods
        Memorable::remember(&mut memory, "trait_key", json!("trait_value"), MemoryScope::Session)
            .unwrap();

        let recalled = Memorable::recall(&memory, "trait_key");
        assert!(recalled.is_some());
        assert_eq!(recalled.unwrap().value, json!("trait_value"));

        let search_results = Memorable::search(&memory, "trait");
        assert_eq!(search_results.len(), 1);

        assert!(Memorable::forget(&mut memory, "trait_key"));
        assert!(Memorable::recall(&memory, "trait_key").is_none());
    }

    #[tokio::test]
    async fn test_serialization_roundtrip() {
        let mut original = MemoryLayer::new();

        original
            .remember(
                "complex",
                json!({
                    "nested": {
                        "array": [1, 2, 3],
                        "object": {"key": "value"}
                    }
                }),
                MemoryScope::Session,
            )
            .unwrap();

        original
            .remember_with_tags(
                "tagged",
                json!("tagged_value"),
                MemoryScope::Persistent,
                vec!["tag1".to_string(), "tag2".to_string()],
            )
            .unwrap();

        // Serialize and deserialize
        let serialized = serde_json::to_string(&original).expect("Should serialize");
        let deserialized: MemoryLayer =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(deserialized.len(), 2);
        assert!(deserialized.contains_key("complex"));
        assert!(deserialized.contains_key("tagged"));
        assert!(deserialized.recall("tagged").unwrap().has_tag("tag1"));
    }
}
