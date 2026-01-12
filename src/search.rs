//! Semantic search module for Project Panpsychism.
//!
//! This module provides keyword-based search functionality for prompts
//! with mock semantic search capabilities for MVP. The search engine
//! scores results by keyword matches and supports filtering by tags
//! and categories.
//!
//! # Architecture
//!
//! The search system operates in the "Sorcerer's Wand" metaphor:
//! - **Query** is the incantation spoken by the sorcerer
//! - **Prompts** are the spells in the grimoire
//! - **Score** represents how well the spell matches the intent
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::search::{SearchEngine, SearchQuery, PromptMetadata};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let prompts = vec![
//!         PromptMetadata {
//!             id: "auth-01".to_string(),
//!             title: "Authentication Flow".to_string(),
//!             content: "Implement OAuth2 authentication...".to_string(),
//!             tags: vec!["auth".to_string(), "security".to_string()],
//!             category: Some("security".to_string()),
//!             path: "prompts/auth-flow.md".into(),
//!         },
//!     ];
//!
//!     let engine = SearchEngine::new(prompts);
//!     let query = SearchQuery::new("OAuth authentication")
//!         .with_top_k(5)
//!         .with_tags(vec!["auth".to_string()]);
//!
//!     let results = engine.search(&query).await?;
//!     println!("Found {} results", results.len());
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use async_stream::stream;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::pin::Pin;

/// Metadata for a prompt in the library.
///
/// Contains all searchable fields extracted from prompt files
/// including YAML frontmatter and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMetadata {
    /// Unique identifier for the prompt (e.g., "auth-01", "spinoza-conatus")
    pub id: String,
    /// Human-readable title of the prompt
    pub title: String,
    /// Full content of the prompt (for keyword matching)
    pub content: String,
    /// Tags for categorization and filtering
    pub tags: Vec<String>,
    /// Optional category (e.g., "security", "philosophy", "coding")
    pub category: Option<String>,
    /// Path to the source prompt file
    pub path: PathBuf,
}

impl PromptMetadata {
    /// Create a new PromptMetadata instance.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the prompt
    /// * `title` - Human-readable title
    /// * `content` - Full prompt content
    /// * `path` - Path to the source file
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            content: content.into(),
            tags: Vec::new(),
            category: None,
            path: path.into(),
        }
    }

    /// Add tags to the prompt metadata.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the category for the prompt.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
}

/// Search query parameters.
///
/// Encapsulates all parameters for a search operation including
/// the query string, filters, and result limits.
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// The search query string
    pub query: String,
    /// Maximum number of results to return (default: 10)
    pub top_k: usize,
    /// Optional tags to filter by (AND logic)
    pub tags: Option<Vec<String>>,
    /// Optional category to filter by
    pub category: Option<String>,
    /// Minimum score threshold (0.0 - 1.0)
    pub min_score: f64,
}

impl SearchQuery {
    /// Create a new search query.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let query = SearchQuery::new("implement authentication");
    /// ```
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            top_k: 10,
            tags: None,
            category: None,
            min_score: 0.0,
        }
    }

    /// Set the maximum number of results to return.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Filter results by tags (AND logic - all tags must match).
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Filter results by category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set minimum score threshold.
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score.clamp(0.0, 1.0);
        self
    }
}

/// A search result containing a matched prompt.
///
/// Includes the prompt metadata along with relevance scoring
/// and an excerpt showing the matching context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Unique identifier of the prompt
    pub id: String,
    /// Title of the prompt
    pub title: String,
    /// Path to the prompt file
    pub path: PathBuf,
    /// Relevance score (0.0 - 1.0)
    pub score: f64,
    /// Excerpt from the prompt content showing match context
    pub excerpt: String,
    /// Tags associated with the prompt
    pub tags: Vec<String>,
    /// Category of the prompt
    pub category: Option<String>,
}

impl SearchResult {
    /// Create a new search result from prompt metadata.
    fn from_metadata(metadata: &PromptMetadata, score: f64, excerpt: String) -> Self {
        Self {
            id: metadata.id.clone(),
            title: metadata.title.clone(),
            path: metadata.path.clone(),
            score,
            excerpt,
            tags: metadata.tags.clone(),
            category: metadata.category.clone(),
        }
    }
}

/// Search engine for prompt library.
///
/// Provides keyword-based search with mock semantic capabilities.
/// Uses TF-IDF-like scoring for relevance ranking.
///
/// # Sorcerer's Wand Metaphor
///
/// The search engine is like consulting the grimoire:
/// - Query keywords are the incantation's power words
/// - Prompt matches are spells that resonate
/// - Higher scores mean stronger spell alignment
#[derive(Debug)]
pub struct SearchEngine {
    /// Index of prompt metadata
    index: Vec<PromptMetadata>,
    /// Path to the index file (for future Memvid integration)
    #[allow(dead_code)]
    index_path: Option<PathBuf>,
}

impl SearchEngine {
    /// Create a new search engine with an in-memory index.
    ///
    /// # Arguments
    ///
    /// * `prompts` - Vector of prompt metadata to index
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let engine = SearchEngine::new(prompts);
    /// ```
    pub fn new(prompts: Vec<PromptMetadata>) -> Self {
        Self {
            index: prompts,
            index_path: None,
        }
    }

    /// Create a search engine with a path for future index persistence.
    ///
    /// # Arguments
    ///
    /// * `prompts` - Vector of prompt metadata to index
    /// * `index_path` - Path where the index file should be stored
    pub fn with_index_path(prompts: Vec<PromptMetadata>, index_path: impl Into<PathBuf>) -> Self {
        Self {
            index: prompts,
            index_path: Some(index_path.into()),
        }
    }

    /// Search for prompts matching the query.
    ///
    /// Performs keyword-based search with TF-IDF-like scoring.
    /// Results are filtered by tags/category if specified and
    /// sorted by relevance score in descending order.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query parameters
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by relevance score.
    ///
    /// # Errors
    ///
    /// Returns `Error::Search` if the query is empty.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let query = SearchQuery::new("authentication oauth");
    /// let results = engine.search(&query).await?;
    /// ```
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        if query.query.trim().is_empty() {
            return Err(Error::Search("Query cannot be empty".to_string()));
        }

        let query_terms = self.tokenize(&query.query);
        if query_terms.is_empty() {
            return Err(Error::Search(
                "Query contains no searchable terms".to_string(),
            ));
        }

        let mut results: Vec<SearchResult> = self
            .index
            .iter()
            .filter(|prompt| self.matches_filters(prompt, query))
            .filter_map(|prompt| {
                let score = self.calculate_score(prompt, &query_terms);
                if score >= query.min_score {
                    let excerpt = self.generate_excerpt(prompt, &query_terms);
                    Some(SearchResult::from_metadata(prompt, score, excerpt))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top_k
        results.truncate(query.top_k);

        Ok(results)
    }

    /// Search for prompts by tags only.
    ///
    /// Returns all prompts that match ALL specified tags,
    /// sorted alphabetically by title.
    ///
    /// # Arguments
    ///
    /// * `tags` - Tags to filter by (AND logic)
    ///
    /// # Returns
    ///
    /// Vector of search results matching all tags.
    ///
    /// # Errors
    ///
    /// Returns `Error::Search` if no tags are specified.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = engine.search_by_tags(&["auth", "security"]).await?;
    /// ```
    pub async fn search_by_tags(&self, tags: &[&str]) -> Result<Vec<SearchResult>> {
        if tags.is_empty() {
            return Err(Error::Search(
                "At least one tag must be specified".to_string(),
            ));
        }

        let tag_set: HashSet<&str> = tags.iter().copied().collect();

        let mut results: Vec<SearchResult> = self
            .index
            .iter()
            .filter(|prompt| {
                let prompt_tags: HashSet<&str> = prompt.tags.iter().map(|s| s.as_str()).collect();
                tag_set.iter().all(|tag| prompt_tags.contains(tag))
            })
            .map(|prompt| {
                let excerpt = self.first_n_chars(&prompt.content, 150);
                SearchResult::from_metadata(prompt, 1.0, excerpt)
            })
            .collect();

        // Sort alphabetically by title
        results.sort_by(|a, b| a.title.cmp(&b.title));

        Ok(results)
    }

    /// Search by category.
    ///
    /// Returns all prompts in the specified category,
    /// sorted alphabetically by title.
    ///
    /// # Arguments
    ///
    /// * `category` - Category to filter by
    ///
    /// # Returns
    ///
    /// Vector of search results in the category.
    pub async fn search_by_category(&self, category: &str) -> Result<Vec<SearchResult>> {
        if category.trim().is_empty() {
            return Err(Error::Search("Category cannot be empty".to_string()));
        }

        let category_lower = category.to_lowercase();

        let mut results: Vec<SearchResult> = self
            .index
            .iter()
            .filter(|prompt| {
                prompt
                    .category
                    .as_ref()
                    .map(|c| c.to_lowercase() == category_lower)
                    .unwrap_or(false)
            })
            .map(|prompt| {
                let excerpt = self.first_n_chars(&prompt.content, 150);
                SearchResult::from_metadata(prompt, 1.0, excerpt)
            })
            .collect();

        results.sort_by(|a, b| a.title.cmp(&b.title));

        Ok(results)
    }

    /// Search across multiple categories concurrently using futures::join_all.
    ///
    /// This method runs searches for each category in parallel, significantly
    /// reducing total search time when querying multiple categories.
    ///
    /// # Arguments
    ///
    /// * `categories` - Slice of category names to search in
    ///
    /// # Returns
    ///
    /// A HashMap mapping category names to their search results.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = engine.search_categories_concurrent(&["security", "api", "testing"]).await?;
    /// for (category, category_results) in results {
    ///     println!("{}: {} results", category, category_results.len());
    /// }
    /// ```
    pub async fn search_categories_concurrent(
        &self,
        categories: &[&str],
    ) -> Result<std::collections::HashMap<String, Vec<SearchResult>>> {
        use futures::future::join_all;

        // Create futures for each category search
        let futures: Vec<_> = categories
            .iter()
            .map(|&category| {
                let cat = category.to_string();
                async move {
                    let results = self.search_by_category(&cat).await;
                    (cat, results)
                }
            })
            .collect();

        // Run all searches concurrently
        let results = join_all(futures).await;

        // Collect results into a HashMap
        let mut map = std::collections::HashMap::new();
        for (category, result) in results {
            match result {
                Ok(search_results) => {
                    map.insert(category, search_results);
                }
                Err(_) => {
                    // Insert empty results for failed searches
                    map.insert(category, Vec::new());
                }
            }
        }

        Ok(map)
    }

    /// Search across multiple tags concurrently using futures::join_all.
    ///
    /// # Arguments
    ///
    /// * `tag_groups` - Slice of tag arrays to search for
    ///
    /// # Returns
    ///
    /// A vector of search result vectors, one for each tag group.
    pub async fn search_tags_concurrent(
        &self,
        tag_groups: &[&[&str]],
    ) -> Vec<Result<Vec<SearchResult>>> {
        use futures::future::join_all;

        let futures: Vec<_> = tag_groups
            .iter()
            .map(|tags| self.search_by_tags(tags))
            .collect();

        join_all(futures).await
    }

    /// Stream search results as they are found using async_stream.
    ///
    /// Instead of waiting for all results to be computed, this method
    /// yields results one at a time as they are discovered. This is useful
    /// for large indexes where you want to show partial results early.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query parameters
    ///
    /// # Returns
    ///
    /// A stream of SearchResult items.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let query = SearchQuery::new("authentication");
    /// let mut stream = engine.search_stream(&query);
    ///
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(search_result) => println!("Found: {}", search_result.title),
    ///         Err(e) => eprintln!("Error: {}", e),
    ///     }
    /// }
    /// ```
    pub fn search_stream<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> Pin<Box<dyn Stream<Item = Result<SearchResult>> + Send + 'a>> {
        let query_terms = self.tokenize(&query.query);

        Box::pin(stream! {
            if query.query.trim().is_empty() {
                yield Err(Error::Search("Query cannot be empty".to_string()));
                return;
            }

            if query_terms.is_empty() {
                yield Err(Error::Search("Query contains no searchable terms".to_string()));
                return;
            }

            // Collect and sort results
            let mut scored_results: Vec<(SearchResult, f64)> = Vec::new();

            for prompt in &self.index {
                if !self.matches_filters(prompt, query) {
                    continue;
                }

                let score = self.calculate_score(prompt, &query_terms);
                if score >= query.min_score {
                    let excerpt = self.generate_excerpt(prompt, &query_terms);
                    let result = SearchResult::from_metadata(prompt, score, excerpt);
                    scored_results.push((result, score));
                }
            }

            // Sort by score descending
            scored_results.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Yield results up to top_k
            for (result, _) in scored_results.into_iter().take(query.top_k) {
                yield Ok(result);
            }
        })
    }

    /// Stream search results from multiple queries sequentially.
    ///
    /// Processes each query and yields results as they come.
    /// For true concurrent execution, use `search_multi_concurrent` instead.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of search queries to execute
    ///
    /// # Returns
    ///
    /// A stream of (query_index, SearchResult) tuples.
    pub fn search_multi_stream<'a>(
        &'a self,
        queries: &'a [SearchQuery],
    ) -> Pin<Box<dyn Stream<Item = (usize, Result<SearchResult>)> + Send + 'a>> {
        Box::pin(stream! {
            use futures::StreamExt;

            // Process queries sequentially, yielding results as streams
            for (idx, query) in queries.iter().enumerate() {
                let mut s = self.search_stream(query);
                while let Some(result) = s.next().await {
                    yield (idx, result);
                }
            }
        })
    }

    /// Search multiple queries concurrently and collect all results.
    ///
    /// This is more efficient than search_multi_stream when you need
    /// all results at once, as it uses join_all for true parallelism.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of search queries to execute concurrently
    ///
    /// # Returns
    ///
    /// A vector of (query_index, results) tuples.
    pub async fn search_multi_concurrent(
        &self,
        queries: &[SearchQuery],
    ) -> Vec<(usize, Vec<SearchResult>)> {
        use futures::future::join_all;

        let futures: Vec<_> = queries
            .iter()
            .enumerate()
            .map(|(idx, query)| async move {
                let results = self.search(query).await.unwrap_or_default();
                (idx, results)
            })
            .collect();

        join_all(futures).await
    }

    /// Get the number of prompts in the index.
    pub fn index_size(&self) -> usize {
        self.index.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Tokenize a string into searchable terms.
    ///
    /// Converts to lowercase, removes punctuation, and splits on whitespace.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect::<String>()
            .split_whitespace()
            .filter(|s| s.len() >= 2) // Filter out single characters
            .map(String::from)
            .collect()
    }

    /// Check if a prompt matches the query filters.
    fn matches_filters(&self, prompt: &PromptMetadata, query: &SearchQuery) -> bool {
        // Check category filter
        if let Some(ref category) = query.category {
            let matches_category = prompt
                .category
                .as_ref()
                .map(|c| c.to_lowercase() == category.to_lowercase())
                .unwrap_or(false);
            if !matches_category {
                return false;
            }
        }

        // Check tags filter (AND logic)
        if let Some(ref tags) = query.tags {
            let prompt_tags: HashSet<String> =
                prompt.tags.iter().map(|s| s.to_lowercase()).collect();
            let all_tags_match = tags
                .iter()
                .all(|tag| prompt_tags.contains(&tag.to_lowercase()));
            if !all_tags_match {
                return false;
            }
        }

        true
    }

    /// Calculate relevance score for a prompt based on query terms.
    ///
    /// Uses a simplified TF-IDF-like scoring:
    /// - Title matches are weighted 3x
    /// - Tag matches are weighted 2x
    /// - Content matches are weighted 1x
    /// - ID exact match bonus
    fn calculate_score(&self, prompt: &PromptMetadata, query_terms: &[String]) -> f64 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let title_tokens = self.tokenize(&prompt.title);
        let content_tokens = self.tokenize(&prompt.content);
        let tag_tokens: Vec<String> = prompt.tags.iter().flat_map(|t| self.tokenize(t)).collect();
        let id_lower = prompt.id.to_lowercase();

        let mut score = 0.0;
        let mut matched_terms = 0;

        for term in query_terms {
            let mut term_score = 0.0;

            // Title match (weight: 3.0)
            if title_tokens.contains(term) {
                term_score += 3.0;
            }

            // Tag match (weight: 2.0)
            if tag_tokens.contains(term) {
                term_score += 2.0;
            }

            // Content match (weight: 1.0)
            let content_matches = content_tokens.iter().filter(|t| *t == term).count();
            term_score += (content_matches as f64).min(5.0) * 0.2; // Cap at 5 matches

            // ID exact match bonus
            if id_lower.contains(term) {
                term_score += 1.5;
            }

            if term_score > 0.0 {
                matched_terms += 1;
            }
            score += term_score;
        }

        // Normalize by number of query terms
        let raw_score = score / query_terms.len() as f64;

        // Boost score if more terms matched (coverage bonus)
        let coverage = matched_terms as f64 / query_terms.len() as f64;
        let final_score = raw_score * (0.5 + 0.5 * coverage);

        // Normalize to 0.0 - 1.0 range
        (final_score / 5.0).min(1.0)
    }

    /// Generate an excerpt showing matching context.
    ///
    /// Finds the first occurrence of any query term and extracts
    /// surrounding context.
    fn generate_excerpt(&self, prompt: &PromptMetadata, query_terms: &[String]) -> String {
        let content_lower = prompt.content.to_lowercase();
        let excerpt_len = 150;

        // Find first matching term position
        let first_match_pos = query_terms
            .iter()
            .filter_map(|term| content_lower.find(term))
            .min();

        match first_match_pos {
            Some(pos) => {
                // Calculate excerpt boundaries
                let start = pos.saturating_sub(20);
                let end = (start + excerpt_len).min(prompt.content.len());

                let mut excerpt = String::new();
                if start > 0 {
                    excerpt.push_str("...");
                }
                excerpt.push_str(&prompt.content[start..end].replace('\n', " "));
                if end < prompt.content.len() {
                    excerpt.push_str("...");
                }
                excerpt
            }
            None => self.first_n_chars(&prompt.content, excerpt_len),
        }
    }

    /// Get the first N characters of a string.
    fn first_n_chars(&self, s: &str, n: usize) -> String {
        let cleaned = s.replace('\n', " ");
        if cleaned.len() <= n {
            cleaned
        } else {
            format!("{}...", &cleaned[..n])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prompts() -> Vec<PromptMetadata> {
        vec![
            PromptMetadata::new(
                "auth-01",
                "OAuth2 Authentication Flow",
                "Implement secure OAuth2 authentication with refresh tokens and proper error handling.",
                "prompts/auth/oauth2.md",
            )
            .with_tags(vec!["auth".to_string(), "security".to_string(), "oauth".to_string()])
            .with_category("security"),
            PromptMetadata::new(
                "spinoza-01",
                "Conatus Self-Preservation",
                "Apply Spinoza's concept of conatus to system design for self-preserving architectures.",
                "prompts/philosophy/conatus.md",
            )
            .with_tags(vec!["philosophy".to_string(), "spinoza".to_string()])
            .with_category("philosophy"),
            PromptMetadata::new(
                "api-design",
                "RESTful API Design Patterns",
                "Design REST APIs with proper authentication, versioning, and error responses.",
                "prompts/api/rest.md",
            )
            .with_tags(vec!["api".to_string(), "rest".to_string(), "auth".to_string()])
            .with_category("api"),
            PromptMetadata::new(
                "testing-01",
                "Unit Testing Best Practices",
                "Write comprehensive unit tests with proper mocking and assertions.",
                "prompts/testing/unit.md",
            )
            .with_tags(vec!["testing".to_string(), "quality".to_string()])
            .with_category("testing"),
        ]
    }

    #[tokio::test]
    async fn test_search_basic() {
        let prompts = create_test_prompts();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("authentication oauth");
        let results = engine.search(&query).await.unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "auth-01");
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_search_empty_query() {
        let engine = SearchEngine::new(create_test_prompts());

        let query = SearchQuery::new("");
        let result = engine.search(&query).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Search(_)));
    }

    #[tokio::test]
    async fn test_search_with_category_filter() {
        let engine = SearchEngine::new(create_test_prompts());

        let query = SearchQuery::new("design").with_category("api");
        let results = engine.search(&query).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].category, Some("api".to_string()));
    }

    #[tokio::test]
    async fn test_search_with_tags_filter() {
        let engine = SearchEngine::new(create_test_prompts());

        let query = SearchQuery::new("authentication").with_tags(vec!["security".to_string()]);
        let results = engine.search(&query).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "auth-01");
    }

    #[tokio::test]
    async fn test_search_top_k() {
        let engine = SearchEngine::new(create_test_prompts());

        let query = SearchQuery::new("design authentication api").with_top_k(2);
        let results = engine.search(&query).await.unwrap();

        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_search_min_score() {
        let engine = SearchEngine::new(create_test_prompts());

        let query = SearchQuery::new("authentication").with_min_score(0.5);
        let results = engine.search(&query).await.unwrap();

        for result in &results {
            assert!(result.score >= 0.5);
        }
    }

    #[tokio::test]
    async fn test_search_by_tags() {
        let engine = SearchEngine::new(create_test_prompts());

        let results = engine.search_by_tags(&["auth"]).await.unwrap();

        assert_eq!(results.len(), 2); // auth-01 and api-design both have "auth" tag
        for result in &results {
            assert!(result.tags.iter().any(|t| t == "auth"));
        }
    }

    #[tokio::test]
    async fn test_search_by_tags_multiple() {
        let engine = SearchEngine::new(create_test_prompts());

        let results = engine.search_by_tags(&["auth", "security"]).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "auth-01");
    }

    #[tokio::test]
    async fn test_search_by_tags_empty() {
        let engine = SearchEngine::new(create_test_prompts());

        let result = engine.search_by_tags(&[]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_by_category() {
        let engine = SearchEngine::new(create_test_prompts());

        let results = engine.search_by_category("philosophy").await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "spinoza-01");
    }

    #[tokio::test]
    async fn test_search_by_category_case_insensitive() {
        let engine = SearchEngine::new(create_test_prompts());

        let results = engine.search_by_category("PHILOSOPHY").await.unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_results_sorted_by_score() {
        let engine = SearchEngine::new(create_test_prompts());

        let query = SearchQuery::new("authentication design api");
        let results = engine.search(&query).await.unwrap();

        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[tokio::test]
    async fn test_search_engine_empty() {
        let engine = SearchEngine::new(vec![]);

        assert!(engine.is_empty());
        assert_eq!(engine.index_size(), 0);

        let query = SearchQuery::new("test");
        let results = engine.search(&query).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_excerpt_generation() {
        let prompts = vec![PromptMetadata::new(
            "test-01",
            "Test Prompt",
            "This is a long content that contains the keyword authentication somewhere in the middle of the text for testing excerpt generation functionality.",
            "prompts/test.md",
        )];
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("authentication");
        let results = engine.search(&query).await.unwrap();

        assert!(!results.is_empty());
        assert!(results[0].excerpt.contains("authentication"));
    }

    #[tokio::test]
    async fn test_tokenization() {
        let engine = SearchEngine::new(vec![]);

        let tokens = engine.tokenize("Hello, World! This is a TEST-123.");

        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"123".to_string()));
        // Single character 'a' should be filtered out
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[tokio::test]
    async fn test_search_no_matches() {
        let engine = SearchEngine::new(create_test_prompts());

        let query = SearchQuery::new("xyznonexistent qqq zzz");
        let results = engine.search(&query).await.unwrap();

        // Results should be empty or have very low scores
        for result in &results {
            assert!(
                result.score < 0.1,
                "Unexpected high score for non-matching query"
            );
        }
    }

    #[tokio::test]
    async fn test_prompt_metadata_builder() {
        let prompt = PromptMetadata::new("id", "title", "content", "path.md")
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()])
            .with_category("category");

        assert_eq!(prompt.id, "id");
        assert_eq!(prompt.tags.len(), 2);
        assert_eq!(prompt.category, Some("category".to_string()));
    }

    #[tokio::test]
    async fn test_search_query_builder() {
        let query = SearchQuery::new("test query")
            .with_top_k(5)
            .with_min_score(0.3)
            .with_category("test")
            .with_tags(vec!["tag1".to_string()]);

        assert_eq!(query.query, "test query");
        assert_eq!(query.top_k, 5);
        assert_eq!(query.min_score, 0.3);
        assert_eq!(query.category, Some("test".to_string()));
        assert!(query.tags.is_some());
    }

    #[tokio::test]
    async fn test_search_categories_concurrent() {
        let engine = SearchEngine::new(create_test_prompts());

        let results = engine
            .search_categories_concurrent(&["security", "philosophy", "api"])
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(results.contains_key("security"));
        assert!(results.contains_key("philosophy"));
        assert!(results.contains_key("api"));

        // Security category should have auth-01
        assert_eq!(results.get("security").unwrap().len(), 1);
        assert_eq!(results.get("security").unwrap()[0].id, "auth-01");

        // Philosophy category should have spinoza-01
        assert_eq!(results.get("philosophy").unwrap().len(), 1);
        assert_eq!(results.get("philosophy").unwrap()[0].id, "spinoza-01");
    }

    #[tokio::test]
    async fn test_search_tags_concurrent() {
        let engine = SearchEngine::new(create_test_prompts());

        let results = engine
            .search_tags_concurrent(&[&["auth"], &["philosophy"], &["testing"]])
            .await;

        assert_eq!(results.len(), 3);

        // Auth tag matches auth-01 and api-design
        assert!(results[0].is_ok());
        assert_eq!(results[0].as_ref().unwrap().len(), 2);

        // Philosophy tag matches spinoza-01
        assert!(results[1].is_ok());
        assert_eq!(results[1].as_ref().unwrap().len(), 1);

        // Testing tag matches testing-01
        assert!(results[2].is_ok());
        assert_eq!(results[2].as_ref().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_search_stream() {
        use futures::StreamExt;

        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("authentication oauth").with_top_k(3);

        let mut stream = engine.search_stream(&query);
        let mut results = Vec::new();

        while let Some(result) = stream.next().await {
            match result {
                Ok(search_result) => results.push(search_result),
                Err(e) => panic!("Stream error: {}", e),
            }
        }

        // Should have found results
        assert!(!results.is_empty());
        assert!(results.len() <= 3); // Respects top_k

        // First result should be auth-01 (highest score)
        assert_eq!(results[0].id, "auth-01");
    }

    #[tokio::test]
    async fn test_search_stream_empty_query() {
        use futures::StreamExt;

        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("");

        let mut stream = engine.search_stream(&query);
        let result = stream.next().await;

        // Should yield an error for empty query
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[tokio::test]
    async fn test_search_multi_stream() {
        use futures::StreamExt;

        let engine = SearchEngine::new(create_test_prompts());
        let queries = vec![
            SearchQuery::new("authentication").with_top_k(2),
            SearchQuery::new("spinoza philosophy").with_top_k(2),
        ];

        let mut stream = engine.search_multi_stream(&queries);
        let mut results: Vec<(usize, String)> = Vec::new();

        while let Some((idx, result)) = stream.next().await {
            if let Ok(search_result) = result {
                results.push((idx, search_result.id));
            }
        }

        // Should have results from both queries
        assert!(!results.is_empty());

        // Should have results tagged with query index 0 and 1
        let query0_results: Vec<_> = results.iter().filter(|(idx, _)| *idx == 0).collect();
        let query1_results: Vec<_> = results.iter().filter(|(idx, _)| *idx == 1).collect();

        assert!(!query0_results.is_empty());
        assert!(!query1_results.is_empty());
    }
}
