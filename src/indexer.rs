//! Prompt indexer module for Project Panpsychism.
//!
//! This module provides functionality to scan a prompts directory, extract YAML
//! frontmatter metadata from Markdown files, and build an in-memory searchable index.
//!
//! # Architecture
//!
//! The indexer follows a two-phase approach:
//! 1. **Scanning Phase**: Recursively finds all `.md` files in the prompts directory
//! 2. **Indexing Phase**: Parses YAML frontmatter and builds a HashMap-based index
//!
//! # Mock Memvid Integration
//!
//! Currently uses an in-memory HashMap as a mock for the Memvid `.mv2` format.
//! Future versions will integrate with the actual Memvid library for:
//! - Semantic search via embeddings
//! - Single-file portable index (`.mv2` format)
//! - Sub-5ms retrieval times
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::indexer::Indexer;
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let indexer = Indexer::new("./prompts", "./index.mv2");
//!     let stats = indexer.index().await?;
//!     println!("Indexed {} prompts in {}ms", stats.prompts_indexed, stats.duration_ms);
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::fs;
use tracing::{debug, info, instrument, trace, warn};

/// Metadata extracted from prompt file YAML frontmatter.
///
/// Each prompt file should have a YAML frontmatter block at the top:
///
/// ```yaml
/// ---
/// title: Authentication Guide
/// description: How to implement secure authentication
/// tags: [auth, security, backend]
/// category: development
/// ---
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptMetadata {
    /// Human-readable title of the prompt
    pub title: String,

    /// Brief description of the prompt's purpose
    #[serde(default)]
    pub description: String,

    /// Searchable tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Primary category (e.g., development, writing, analysis)
    #[serde(default)]
    pub category: String,

    /// Privacy tier for access control (optional)
    #[serde(default)]
    pub privacy_tier: Option<String>,

    /// Author of the prompt (optional)
    #[serde(default)]
    pub author: Option<String>,

    /// Version string (optional)
    #[serde(default)]
    pub version: Option<String>,
}

/// A single indexed prompt entry.
///
/// Combines the file path with extracted metadata for quick lookup.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Relative path to the prompt file
    pub path: PathBuf,

    /// Extracted metadata from frontmatter
    pub metadata: PromptMetadata,

    /// Full content of the prompt (excluding frontmatter)
    pub content: String,
}

/// In-memory index structure (mock Memvid `.mv2` format).
///
/// This structure simulates what a Memvid index would provide:
/// - Fast lookup by various keys
/// - Searchable metadata
/// - Content storage for retrieval
#[derive(Debug, Default)]
pub struct MemvidIndex {
    /// Primary index: path -> entry
    entries: HashMap<PathBuf, IndexEntry>,

    /// Secondary index: tag -> paths
    tag_index: HashMap<String, Vec<PathBuf>>,

    /// Secondary index: category -> paths
    category_index: HashMap<String, Vec<PathBuf>>,
}

impl MemvidIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the index.
    #[instrument(skip(self, entry), fields(path = %entry.path.display()))]
    pub fn insert(&mut self, entry: IndexEntry) {
        let path = entry.path.clone();

        trace!(
            tags = entry.metadata.tags.len(),
            category = %entry.metadata.category,
            "Inserting entry into index"
        );

        // Update tag index
        for tag in &entry.metadata.tags {
            self.tag_index
                .entry(tag.to_lowercase())
                .or_default()
                .push(path.clone());
        }

        // Update category index
        if !entry.metadata.category.is_empty() {
            self.category_index
                .entry(entry.metadata.category.to_lowercase())
                .or_default()
                .push(path.clone());
        }

        // Insert into primary index
        self.entries.insert(path, entry);
    }

    /// Get an entry by path.
    pub fn get(&self, path: &Path) -> Option<&IndexEntry> {
        self.entries.get(path)
    }

    /// Get all entries with a specific tag.
    #[instrument(skip(self), fields(tag = %tag))]
    pub fn get_by_tag(&self, tag: &str) -> Vec<&IndexEntry> {
        let results: Vec<_> = self
            .tag_index
            .get(&tag.to_lowercase())
            .map(|paths| paths.iter().filter_map(|p| self.entries.get(p)).collect())
            .unwrap_or_default();
        
        trace!(count = results.len(), "Retrieved entries by tag");
        results
    }

    /// Get all entries in a category.
    #[instrument(skip(self), fields(category = %category))]
    pub fn get_by_category(&self, category: &str) -> Vec<&IndexEntry> {
        let results: Vec<_> = self
            .category_index
            .get(&category.to_lowercase())
            .map(|paths| paths.iter().filter_map(|p| self.entries.get(p)).collect())
            .unwrap_or_default();
        
        trace!(count = results.len(), "Retrieved entries by category");
        results
    }

    /// Get total number of indexed entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all unique tags in the index.
    pub fn all_tags(&self) -> Vec<&String> {
        self.tag_index.keys().collect()
    }

    /// Get all unique categories in the index.
    pub fn all_categories(&self) -> Vec<&String> {
        self.category_index.keys().collect()
    }

    /// Get all entries as an iterator.
    pub fn iter(&self) -> impl Iterator<Item = &IndexEntry> {
        self.entries.values()
    }
}

/// Statistics from an indexing operation.
///
/// Provides metrics about the indexing process including counts
/// of successfully indexed prompts, skipped files, and errors.
#[derive(Debug, Default, Clone)]
pub struct IndexStats {
    /// Number of prompts successfully indexed
    pub prompts_indexed: usize,

    /// Number of prompts skipped (e.g., no frontmatter)
    pub prompts_skipped: usize,

    /// Number of errors encountered during indexing
    pub errors: usize,

    /// Total time taken in milliseconds
    pub duration_ms: u64,

    /// Unique tags found across all prompts
    pub unique_tags: usize,

    /// Unique categories found
    pub unique_categories: usize,
}

/// Indexer for prompt files.
///
/// Scans a directory for Markdown files with YAML frontmatter and builds
/// a searchable in-memory index. The index can be used for fast prompt
/// lookup by title, tags, category, or content.
///
/// # Example
///
/// ```rust,ignore
/// let indexer = Indexer::new("./prompts", "./index.mv2");
///
/// // Build the index
/// let stats = indexer.index().await?;
///
/// // Access the index
/// let index = indexer.get_index();
/// let auth_prompts = index.get_by_tag("authentication");
/// ```
#[derive(Debug)]
pub struct Indexer {
    /// Path to the prompts directory
    prompts_dir: PathBuf,

    /// Path to the index file (mock .mv2 format)
    index_path: PathBuf,

    /// In-memory index (mock Memvid)
    index: MemvidIndex,
}

impl Default for Indexer {
    fn default() -> Self {
        Self {
            prompts_dir: PathBuf::from("./prompts"),
            index_path: PathBuf::from("./index.mv2"),
            index: MemvidIndex::new(),
        }
    }
}

impl Indexer {
    /// Create a new indexer with specified paths.
    ///
    /// # Arguments
    ///
    /// * `prompts_dir` - Directory containing prompt Markdown files
    /// * `index_path` - Path where the index file will be stored
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let indexer = Indexer::new("./prompts", "./cache/index.mv2");
    /// ```
    pub fn new(prompts_dir: impl AsRef<Path>, index_path: impl AsRef<Path>) -> Self {
        Self {
            prompts_dir: prompts_dir.as_ref().to_path_buf(),
            index_path: index_path.as_ref().to_path_buf(),
            index: MemvidIndex::new(),
        }
    }

    /// Index all prompts in the prompts directory.
    ///
    /// Recursively scans the prompts directory for `.md` files, extracts
    /// YAML frontmatter, and builds an in-memory searchable index.
    ///
    /// # Returns
    ///
    /// Returns `IndexStats` with metrics about the indexing operation.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The prompts directory doesn't exist
    /// - There are permission issues reading files
    ///
    /// Note: Individual file parsing errors are counted but don't fail
    /// the entire indexing operation.
    #[instrument(skip(self), fields(prompts_dir = %self.prompts_dir.display()))]
    pub async fn index(&mut self) -> Result<IndexStats> {
        let start = Instant::now();
        let mut stats = IndexStats::default();

        // Verify prompts directory exists
        if !self.prompts_dir.exists() {
            warn!(path = %self.prompts_dir.display(), "Prompts directory not found");
            return Err(Error::Index(format!(
                "Prompts directory not found: {:?}",
                self.prompts_dir
            )));
        }

        info!(path = %self.prompts_dir.display(), "Starting indexing operation");

        // Clear existing index
        self.index = MemvidIndex::new();

        // Scan for markdown files
        let md_files = self.scan_directory(&self.prompts_dir.clone()).await?;
        debug!(count = md_files.len(), "Found markdown files");

        // Process each file
        for file_path in md_files {
            match self.process_file(&file_path).await {
                Ok(Some(entry)) => {
                    trace!(
                        path = %file_path.display(),
                        title = %entry.metadata.title,
                        "Successfully indexed prompt"
                    );
                    self.index.insert(entry);
                    stats.prompts_indexed += 1;
                }
                Ok(None) => {
                    debug!(path = %file_path.display(), "Skipped file (no frontmatter)");
                    stats.prompts_skipped += 1;
                }
                Err(e) => {
                    warn!(path = %file_path.display(), error = %e, "Error processing file");
                    stats.errors += 1;
                }
            }
        }

        // Calculate final stats
        stats.duration_ms = start.elapsed().as_millis() as u64;
        stats.unique_tags = self.index.all_tags().len();
        stats.unique_categories = self.index.all_categories().len();

        info!(
            prompts_indexed = stats.prompts_indexed,
            prompts_skipped = stats.prompts_skipped,
            errors = stats.errors,
            duration_ms = stats.duration_ms,
            unique_tags = stats.unique_tags,
            unique_categories = stats.unique_categories,
            "Indexing complete"
        );

        // TODO: Persist index to .mv2 file when Memvid is integrated
        // For now, we only maintain the in-memory index

        Ok(stats)
    }

    /// Check if the index exists and is valid.
    ///
    /// Currently checks if the index file exists. Future versions will
    /// also validate the index against the source files for staleness.
    pub fn is_index_valid(&self) -> bool {
        // For in-memory index, check if we have any entries
        if !self.index.is_empty() {
            return true;
        }

        // Also check if persisted index exists
        self.index_path.exists()
    }

    /// Get a reference to the in-memory index.
    pub fn get_index(&self) -> &MemvidIndex {
        &self.index
    }

    /// Get a mutable reference to the in-memory index.
    pub fn get_index_mut(&mut self) -> &mut MemvidIndex {
        &mut self.index
    }

    /// Recursively scan a directory for `.md` files.
    #[instrument(skip(self), fields(dir = %dir.display()))]
    async fn scan_directory(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        let mut read_dir = fs::read_dir(dir).await?;

        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                // Recursively scan subdirectories
                trace!(subdir = %path.display(), "Scanning subdirectory");
                let sub_files = Box::pin(self.scan_directory(&path)).await?;
                files.extend(sub_files);
            } else if let Some(ext) = path.extension() {
                if ext == "md" {
                    trace!(file = %path.display(), "Found markdown file");
                    files.push(path);
                }
            }
        }

        debug!(count = files.len(), dir = %dir.display(), "Directory scan complete");
        Ok(files)
    }

    /// Process a single markdown file.
    ///
    /// Extracts YAML frontmatter and content, returning an IndexEntry
    /// if the file has valid frontmatter.
    #[instrument(skip(self), fields(path = %path.display()))]
    async fn process_file(&self, path: &Path) -> Result<Option<IndexEntry>> {
        trace!("Reading file content");
        let content = fs::read_to_string(path).await?;

        // Check for YAML frontmatter
        if !content.starts_with("---") {
            trace!("No frontmatter delimiter found");
            return Ok(None);
        }

        // Find the closing frontmatter delimiter
        let rest = &content[3..];
        let end_idx = rest.find("\n---").or_else(|| rest.find("\r\n---"));

        let Some(end_idx) = end_idx else {
            trace!("No closing frontmatter delimiter found");
            return Ok(None);
        };

        // Extract frontmatter YAML
        let frontmatter = &rest[..end_idx].trim();

        // Extract content after frontmatter
        let content_start = end_idx + 4; // Skip "\n---"
        let body = if content_start < rest.len() {
            rest[content_start..].trim().to_string()
        } else {
            String::new()
        };

        // Parse YAML frontmatter
        let metadata: PromptMetadata = serde_yaml::from_str(frontmatter).map_err(|e| {
            debug!(error = %e, "Failed to parse YAML frontmatter");
            Error::Index(format!("Failed to parse frontmatter in {:?}: {}", path, e))
        })?;

        // Create relative path for storage
        let relative_path = path
            .strip_prefix(&self.prompts_dir)
            .unwrap_or(path)
            .to_path_buf();

        trace!(
            title = %metadata.title,
            tags = metadata.tags.len(),
            content_len = body.len(),
            "Successfully parsed prompt"
        );

        Ok(Some(IndexEntry {
            path: relative_path,
            metadata,
            content: body,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    /// Create a test prompt file with frontmatter.
    async fn create_test_prompt(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).await.unwrap();
        file.write_all(content.as_bytes()).await.unwrap();
        path
    }

    #[tokio::test]
    async fn test_indexer_new() {
        let indexer = Indexer::new("./prompts", "./index.mv2");
        assert_eq!(indexer.prompts_dir, PathBuf::from("./prompts"));
        assert_eq!(indexer.index_path, PathBuf::from("./index.mv2"));
    }

    #[tokio::test]
    async fn test_index_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let mut indexer = Indexer::new(temp_dir.path(), temp_dir.path().join("index.mv2"));

        let stats = indexer.index().await.unwrap();
        assert_eq!(stats.prompts_indexed, 0);
        assert_eq!(stats.prompts_skipped, 0);
        assert_eq!(stats.errors, 0);
    }

    #[tokio::test]
    async fn test_index_single_prompt() {
        let temp_dir = TempDir::new().unwrap();

        let prompt_content = r#"---
title: Test Prompt
description: A test prompt for unit testing
tags: [test, unit-test]
category: testing
---

This is the prompt content.
"#;

        create_test_prompt(temp_dir.path(), "test.md", prompt_content).await;

        let mut indexer = Indexer::new(temp_dir.path(), temp_dir.path().join("index.mv2"));
        let stats = indexer.index().await.unwrap();

        assert_eq!(stats.prompts_indexed, 1);
        assert_eq!(stats.prompts_skipped, 0);
        assert_eq!(stats.unique_tags, 2);
        assert_eq!(stats.unique_categories, 1);

        // Verify index content
        let index = indexer.get_index();
        assert_eq!(index.len(), 1);

        let entry = index.get(&PathBuf::from("test.md")).unwrap();
        assert_eq!(entry.metadata.title, "Test Prompt");
        assert_eq!(entry.metadata.tags, vec!["test", "unit-test"]);
    }

    #[tokio::test]
    async fn test_index_skips_no_frontmatter() {
        let temp_dir = TempDir::new().unwrap();

        // File without frontmatter
        create_test_prompt(temp_dir.path(), "no_frontmatter.md", "Just plain markdown").await;

        let mut indexer = Indexer::new(temp_dir.path(), temp_dir.path().join("index.mv2"));
        let stats = indexer.index().await.unwrap();

        assert_eq!(stats.prompts_indexed, 0);
        assert_eq!(stats.prompts_skipped, 1);
    }

    #[tokio::test]
    async fn test_index_invalid_yaml() {
        let temp_dir = TempDir::new().unwrap();

        let invalid_content = r#"---
title: [Invalid YAML
description: missing bracket
---

Content here.
"#;

        create_test_prompt(temp_dir.path(), "invalid.md", invalid_content).await;

        let mut indexer = Indexer::new(temp_dir.path(), temp_dir.path().join("index.mv2"));
        let stats = indexer.index().await.unwrap();

        assert_eq!(stats.prompts_indexed, 0);
        assert_eq!(stats.errors, 1);
    }

    #[tokio::test]
    async fn test_index_subdirectory() {
        let temp_dir = TempDir::new().unwrap();
        let sub_dir = temp_dir.path().join("category");
        fs::create_dir(&sub_dir).await.unwrap();

        let prompt_content = r#"---
title: Nested Prompt
description: Prompt in subdirectory
tags: [nested]
category: sub
---

Nested content.
"#;

        create_test_prompt(&sub_dir, "nested.md", prompt_content).await;

        let mut indexer = Indexer::new(temp_dir.path(), temp_dir.path().join("index.mv2"));
        let stats = indexer.index().await.unwrap();

        assert_eq!(stats.prompts_indexed, 1);

        let index = indexer.get_index();
        let nested_prompts = index.get_by_tag("nested");
        assert_eq!(nested_prompts.len(), 1);
    }

    #[tokio::test]
    async fn test_index_nonexistent_directory() {
        let mut indexer = Indexer::new("/nonexistent/path", "./index.mv2");
        let result = indexer.index().await;

        assert!(result.is_err());
        assert!(matches!(result, Err(Error::Index(_))));
    }

    #[tokio::test]
    async fn test_memvid_index_operations() {
        let mut index = MemvidIndex::new();

        let entry = IndexEntry {
            path: PathBuf::from("test.md"),
            metadata: PromptMetadata {
                title: "Test".to_string(),
                description: "Description".to_string(),
                tags: vec!["rust".to_string(), "testing".to_string()],
                category: "development".to_string(),
                privacy_tier: None,
                author: None,
                version: None,
            },
            content: "Test content".to_string(),
        };

        index.insert(entry);

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());

        // Test tag lookup
        let rust_prompts = index.get_by_tag("rust");
        assert_eq!(rust_prompts.len(), 1);
        assert_eq!(rust_prompts[0].metadata.title, "Test");

        // Test category lookup
        let dev_prompts = index.get_by_category("development");
        assert_eq!(dev_prompts.len(), 1);

        // Test case insensitivity
        let rust_upper = index.get_by_tag("RUST");
        assert_eq!(rust_upper.len(), 1);
    }

    #[tokio::test]
    async fn test_is_index_valid() {
        let temp_dir = TempDir::new().unwrap();
        let mut indexer = Indexer::new(temp_dir.path(), temp_dir.path().join("index.mv2"));

        // Empty index is invalid
        assert!(!indexer.is_index_valid());

        // Add content and index
        let prompt_content = r#"---
title: Validity Test
description: Testing validity
tags: []
category: test
---

Content.
"#;
        create_test_prompt(temp_dir.path(), "valid.md", prompt_content).await;
        indexer.index().await.unwrap();

        // Now index should be valid
        assert!(indexer.is_index_valid());
    }

    #[test]
    fn test_prompt_metadata_deserialize() {
        let yaml = r#"
title: API Integration
description: How to integrate external APIs
tags: [api, integration, http]
category: backend
privacy_tier: internal
author: mustafa
version: "1.0"
"#;

        let metadata: PromptMetadata = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(metadata.title, "API Integration");
        assert_eq!(metadata.tags.len(), 3);
        assert_eq!(metadata.privacy_tier, Some("internal".to_string()));
        assert_eq!(metadata.author, Some("mustafa".to_string()));
    }

    #[test]
    fn test_prompt_metadata_minimal() {
        let yaml = r#"
title: Minimal Prompt
"#;

        let metadata: PromptMetadata = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(metadata.title, "Minimal Prompt");
        assert!(metadata.description.is_empty());
        assert!(metadata.tags.is_empty());
        assert!(metadata.category.is_empty());
    }

    #[tokio::test]
    async fn test_index_multiple_prompts() {
        let temp_dir = TempDir::new().unwrap();

        let prompts = vec![
            (
                "auth.md",
                r#"---
title: Authentication
description: Auth patterns
tags: [auth, security]
category: backend
---
Auth content."#,
            ),
            (
                "database.md",
                r#"---
title: Database
description: DB patterns
tags: [database, sql]
category: backend
---
DB content."#,
            ),
            (
                "frontend.md",
                r#"---
title: Frontend
description: UI patterns
tags: [react, ui]
category: frontend
---
Frontend content."#,
            ),
        ];

        for (name, content) in prompts {
            create_test_prompt(temp_dir.path(), name, content).await;
        }

        let mut indexer = Indexer::new(temp_dir.path(), temp_dir.path().join("index.mv2"));
        let stats = indexer.index().await.unwrap();

        assert_eq!(stats.prompts_indexed, 3);
        assert_eq!(stats.unique_categories, 2); // backend, frontend

        let index = indexer.get_index();
        let backend_prompts = index.get_by_category("backend");
        assert_eq!(backend_prompts.len(), 2);
    }
}
