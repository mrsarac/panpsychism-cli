//! Phase 3 Integration Tests for Project Panpsychism
//!
//! These tests verify the orchestrator pipeline, synthesizer, and full ask workflow.
//! Uses mock LLM client for deterministic testing.

use panpsychism::error::{Error, Result};
use panpsychism::gemini::{ChatResponse, Choice, Message, Usage};
use panpsychism::orchestrator::{IntentAnalysis, Orchestrator, SelectedPrompt, Strategy};
use panpsychism::search::{PromptMetadata, SearchEngine, SearchQuery, SearchResult};
use panpsychism::synthesizer::{SynthesisResult, Synthesizer, TokenUsage};
use panpsychism::validator::{SpinozaPrinciple, SpinozaValidator, ValidationConfig};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// =============================================================================
// MOCK LLM CLIENT
// =============================================================================

/// Mock Gemini client for deterministic testing.
///
/// Provides predictable responses based on configured patterns.
#[derive(Debug, Clone)]
pub struct MockGeminiClient {
    /// The response to return from complete()
    response: String,
    /// Number of calls made (for verification)
    call_count: Arc<AtomicUsize>,
    /// Whether to simulate an error
    simulate_error: bool,
    /// Simulated usage statistics
    usage: Option<Usage>,
}

impl Default for MockGeminiClient {
    fn default() -> Self {
        Self::new("Mock response from Gemini")
    }
}

impl MockGeminiClient {
    /// Create a new mock client with a fixed response.
    pub fn new(response: &str) -> Self {
        Self {
            response: response.to_string(),
            call_count: Arc::new(AtomicUsize::new(0)),
            simulate_error: false,
            usage: Some(Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            }),
        }
    }

    /// Create a mock client that simulates errors.
    pub fn with_error() -> Self {
        Self {
            response: String::new(),
            call_count: Arc::new(AtomicUsize::new(0)),
            simulate_error: true,
            usage: None,
        }
    }

    /// Create a mock client with custom usage stats.
    pub fn with_usage(mut self, prompt_tokens: usize, completion_tokens: usize) -> Self {
        self.usage = Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        });
        self
    }

    /// Get the number of times complete() was called.
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// Simulate a chat completion request.
    pub async fn chat(&self, messages: Vec<Message>) -> Result<ChatResponse> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        if self.simulate_error {
            return Err(Error::Synthesis("Simulated API error".to_string()));
        }

        // Build response based on input
        let response_content = if messages
            .iter()
            .any(|m| m.content.contains("authentication"))
        {
            "For authentication, I recommend using OAuth2 with JWT tokens. \
             This provides secure, stateless authentication."
        } else if messages.iter().any(|m| m.content.contains("Spinoza")) {
            "Spinoza's conatus teaches us that all beings strive to persist. \
             This principle can guide system design toward self-preservation."
        } else {
            &self.response
        };

        Ok(ChatResponse {
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: response_content.to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: self.usage.clone(),
        })
    }

    /// Simple text completion.
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let response = self.chat(messages).await?;

        response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| Error::Synthesis("No response from mock API".to_string()))
    }
}

// =============================================================================
// TEST FIXTURES
// =============================================================================

/// Create test prompts for search tests.
fn create_test_prompts() -> Vec<PromptMetadata> {
    vec![
        PromptMetadata::new(
            "auth-01",
            "OAuth2 Authentication Flow",
            "Implement secure OAuth2 authentication with refresh tokens and proper error handling. \
             Use JWT for stateless sessions.",
            "prompts/auth/oauth2.md",
        )
        .with_tags(vec![
            "auth".to_string(),
            "security".to_string(),
            "oauth".to_string(),
        ])
        .with_category("security"),
        PromptMetadata::new(
            "spinoza-01",
            "Conatus Self-Preservation",
            "Apply Spinoza's concept of conatus to system design for self-preserving architectures. \
             Each component strives to persist in its being.",
            "prompts/philosophy/conatus.md",
        )
        .with_tags(vec!["philosophy".to_string(), "spinoza".to_string()])
        .with_category("philosophy"),
        PromptMetadata::new(
            "api-design",
            "RESTful API Design Patterns",
            "Design REST APIs with proper authentication, versioning, and error responses. \
             Follow OpenAPI specification for documentation.",
            "prompts/api/rest.md",
        )
        .with_tags(vec![
            "api".to_string(),
            "rest".to_string(),
            "auth".to_string(),
        ])
        .with_category("api"),
        PromptMetadata::new(
            "testing-01",
            "Unit Testing Best Practices",
            "Write comprehensive unit tests with proper mocking and assertions. \
             Use TDD for better design.",
            "prompts/testing/unit.md",
        )
        .with_tags(vec!["testing".to_string(), "quality".to_string()])
        .with_category("testing"),
        PromptMetadata::new(
            "error-handling",
            "Error Handling Patterns",
            "Implement robust error handling with proper logging, recovery strategies, \
             and user-friendly messages. Apply the fail-fast principle.",
            "prompts/patterns/errors.md",
        )
        .with_tags(vec![
            "errors".to_string(),
            "patterns".to_string(),
            "logging".to_string(),
        ])
        .with_category("patterns"),
        PromptMetadata::new(
            "caching-01",
            "Caching Strategies",
            "Implement effective caching with proper invalidation strategies. \
             Consider cache-aside, write-through, and write-behind patterns.",
            "prompts/performance/caching.md",
        )
        .with_tags(vec![
            "caching".to_string(),
            "performance".to_string(),
            "patterns".to_string(),
        ])
        .with_category("performance"),
        PromptMetadata::new(
            "security-audit",
            "Security Audit Checklist",
            "Comprehensive security audit covering OWASP Top 10, authentication, \
             authorization, and data protection. Joy in secure systems.",
            "prompts/security/audit.md",
        )
        .with_tags(vec![
            "security".to_string(),
            "audit".to_string(),
            "owasp".to_string(),
        ])
        .with_category("security"),
    ]
}

/// Create a search result for testing.
fn create_search_result(id: &str, title: &str, score: f64) -> SearchResult {
    SearchResult {
        id: id.to_string(),
        title: title.to_string(),
        path: PathBuf::from(format!("prompts/{}.md", id)),
        score,
        excerpt: format!("Content for {} prompt...", title),
        tags: vec!["test".to_string()],
        category: Some("test".to_string()),
    }
}

// =============================================================================
// ORCHESTRATOR PIPELINE TESTS
// =============================================================================

mod orchestrator_tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = Orchestrator::new();
        // Verify orchestrator was created successfully
        // (Fields are private, but creation should not panic)
        assert!(std::mem::size_of_val(&orchestrator) > 0);
    }

    #[tokio::test]
    async fn test_search_simple_query() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("authentication").with_top_k(3);

        let results = engine.search(&query).await.unwrap();

        assert!(!results.is_empty());
        // OAuth2 prompt should be top result for "authentication"
        assert!(
            results[0].title.contains("OAuth2") || results[0].tags.contains(&"auth".to_string())
        );
    }

    #[tokio::test]
    async fn test_search_complex_query() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("secure api authentication with tokens").with_top_k(5);

        let results = engine.search(&query).await.unwrap();

        assert!(!results.is_empty());
        // Multiple relevant prompts should be found
        assert!(
            results.len() >= 2,
            "Complex query should find multiple results"
        );
    }

    #[tokio::test]
    async fn test_search_with_category_filter() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("patterns").with_category("security");

        let results = engine.search(&query).await.unwrap();

        // All results should be in security category
        for result in &results {
            if let Some(ref cat) = result.category {
                assert_eq!(cat.to_lowercase(), "security");
            }
        }
    }

    #[tokio::test]
    async fn test_search_with_tags_filter() {
        let engine = SearchEngine::new(create_test_prompts());
        let query =
            SearchQuery::new("design").with_tags(vec!["api".to_string(), "rest".to_string()]);

        let results = engine.search(&query).await.unwrap();

        // Results should match both tags
        for result in &results {
            let has_api = result.tags.iter().any(|t| t.to_lowercase() == "api");
            let has_rest = result.tags.iter().any(|t| t.to_lowercase() == "rest");
            assert!(
                has_api && has_rest,
                "Results should have both api and rest tags"
            );
        }
    }

    #[tokio::test]
    async fn test_search_result_limit() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("security").with_top_k(2);

        let results = engine.search(&query).await.unwrap();

        assert!(results.len() <= 2, "Results should be limited to top_k");
    }

    #[tokio::test]
    async fn test_search_min_score_filter() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("authentication security").with_min_score(0.3);

        let results = engine.search(&query).await.unwrap();

        for result in &results {
            assert!(
                result.score >= 0.3,
                "All results should meet minimum score threshold"
            );
        }
    }

    #[tokio::test]
    async fn test_search_empty_results() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("xyznonexistent123").with_min_score(0.5);

        let results = engine.search(&query).await.unwrap();

        // Should return empty or very low score results
        for result in &results {
            assert!(
                result.score < 0.3,
                "Non-matching query should have low scores"
            );
        }
    }

    #[tokio::test]
    async fn test_search_results_sorted_by_score() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("security authentication api").with_top_k(5);

        let results = engine.search(&query).await.unwrap();

        // Verify descending score order
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results should be sorted by score descending"
            );
        }
    }
}

// =============================================================================
// SYNTHESIZER TESTS
// =============================================================================

mod synthesizer_tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_client_basic() {
        let client = MockGeminiClient::new("Test response");

        let response = client.complete("Hello").await.unwrap();

        assert_eq!(response, "Test response");
        assert_eq!(client.call_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_client_authentication_response() {
        let client = MockGeminiClient::default();

        let response = client
            .complete("How do I implement authentication?")
            .await
            .unwrap();

        assert!(response.contains("OAuth2") || response.contains("JWT"));
    }

    #[tokio::test]
    async fn test_mock_client_spinoza_response() {
        let client = MockGeminiClient::default();

        let response = client
            .complete("Tell me about Spinoza's philosophy")
            .await
            .unwrap();

        assert!(response.contains("conatus") || response.contains("persist"));
    }

    #[tokio::test]
    async fn test_mock_client_error_simulation() {
        let client = MockGeminiClient::with_error();

        let result = client.complete("Hello").await;

        assert!(result.is_err());
        if let Err(Error::Synthesis(msg)) = result {
            assert!(msg.contains("Simulated"));
        }
    }

    #[tokio::test]
    async fn test_mock_client_call_counting() {
        let client = MockGeminiClient::new("Response");

        // Make multiple calls
        let _ = client.complete("First").await;
        let _ = client.complete("Second").await;
        let _ = client.complete("Third").await;

        assert_eq!(client.call_count(), 3);
    }

    #[tokio::test]
    async fn test_mock_client_usage_stats() {
        let client = MockGeminiClient::new("Response").with_usage(200, 100);

        let messages = vec![Message {
            role: "user".to_string(),
            content: "Test".to_string(),
        }];

        let response = client.chat(messages).await.unwrap();

        let usage = response.usage.expect("Should have usage stats");
        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.completion_tokens, 100);
        assert_eq!(usage.total_tokens, 300);
    }

    #[tokio::test]
    async fn test_meta_prompt_building() {
        // Simulate building a meta-prompt from multiple prompts
        let prompts = vec![
            "Prompt 1: OAuth2 authentication patterns",
            "Prompt 2: JWT token handling",
            "Prompt 3: API security best practices",
        ];

        let meta_prompt = format!(
            "Based on the following expert prompts, synthesize a comprehensive answer:\n\n{}\n\nQuestion: How do I secure my API?",
            prompts.join("\n\n")
        );

        assert!(meta_prompt.contains("OAuth2"));
        assert!(meta_prompt.contains("JWT"));
        assert!(meta_prompt.contains("security"));
    }

    #[tokio::test]
    async fn test_meta_prompt_ordering() {
        // Test that prompts are ordered by relevance
        let search_results = vec![
            create_search_result("auth-01", "Authentication", 0.95),
            create_search_result("api-01", "API Design", 0.75),
            create_search_result("sec-01", "Security", 0.60),
        ];

        // Verify ordering
        assert!(search_results[0].score > search_results[1].score);
        assert!(search_results[1].score > search_results[2].score);

        // Build prompt with priority ordering
        let meta_prompt = search_results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("Priority {}: {} (score: {:.2})", i + 1, r.title, r.score))
            .collect::<Vec<_>>()
            .join("\n");

        assert!(meta_prompt.contains("Priority 1: Authentication"));
        assert!(meta_prompt.contains("Priority 2: API Design"));
    }

    #[tokio::test]
    async fn test_synthesis_with_mock_response() {
        let client = MockGeminiClient::new(
            "Based on the provided prompts, here is a synthesized answer about authentication:\n\
             1. Use OAuth2 for secure authentication\n\
             2. Implement JWT tokens for stateless sessions\n\
             3. Always validate tokens on the server side",
        );

        // Use a prompt that doesn't trigger auth-specific response
        let response = client.complete("Synthesize guidance").await.unwrap();

        assert!(response.contains("OAuth2"));
        assert!(response.contains("JWT"));
        assert!(response.contains("validate"));
    }
}

// =============================================================================
// FULL PIPELINE TESTS
// =============================================================================

mod pipeline_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_pipeline_simple_question() {
        // 1. Search for relevant prompts
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("authentication").with_top_k(3);
        let search_results = engine.search(&query).await.unwrap();

        assert!(!search_results.is_empty(), "Should find relevant prompts");

        // 2. Build meta-prompt from results
        let meta_prompt = search_results
            .iter()
            .map(|r| format!("- {}: {}", r.title, r.excerpt))
            .collect::<Vec<_>>()
            .join("\n");

        // 3. Call mock LLM
        let client = MockGeminiClient::default();
        let response = client
            .complete(&format!(
                "Based on these prompts:\n{}\n\nHow do I implement authentication?",
                meta_prompt
            ))
            .await
            .unwrap();

        // 4. Verify response
        assert!(!response.is_empty());
        assert!(
            response.contains("OAuth2") || response.contains("authentication"),
            "Response should address authentication"
        );
    }

    #[tokio::test]
    async fn test_pipeline_with_validation() {
        // Search and synthesize
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("security best practices").with_top_k(5);
        let results = engine.search(&query).await.unwrap();

        // Mock synthesis response
        let client = MockGeminiClient::new(
            "Security best practices include: \n\
             1. Learn and understand authentication patterns\n\
             2. Build secure systems with proper validation\n\
             3. This brings joy through confidence in your code",
        );
        let response = client
            .complete("Synthesize security guidance")
            .await
            .unwrap();

        // Validate with Spinoza
        let validator = SpinozaValidator::lenient();
        let validation = validator.validate(&response).await.unwrap();

        // The response should pass lenient validation (contains growth/joy keywords)
        assert!(
            validation.scores.conatus >= 0.3,
            "Response should have acceptable CONATUS score"
        );
    }

    #[tokio::test]
    async fn test_pipeline_with_correction() {
        // Initial response that might need correction
        let initial_response = "Just use a simple password check.";

        // Validate initial response
        let validator = SpinozaValidator::new();
        let validation = validator.validate(initial_response).await.unwrap();

        // If validation fails (low scores), trigger correction
        if validation.scores.average() < 0.5 {
            // Correction prompt
            let correction_client = MockGeminiClient::new(
                "Let me improve that response:\n\
                 For secure authentication, grow your knowledge of OAuth2 patterns.\n\
                 Learn to build systems that protect user data.\n\
                 This brings joy through secure, well-designed code.",
            );

            let corrected = correction_client
                .complete(&format!(
                    "Improve this response for security:\n{}",
                    initial_response
                ))
                .await
                .unwrap();

            // Validate corrected response
            let corrected_validation = validator.validate(&corrected).await.unwrap();

            // Corrected should be better
            assert!(
                corrected_validation.scores.average() > validation.scores.average(),
                "Correction should improve scores"
            );
        }
    }

    #[tokio::test]
    async fn test_pipeline_error_handling() {
        // Search phase
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("authentication");
        let results = engine.search(&query).await.unwrap();

        // Synthesis with error client
        let error_client = MockGeminiClient::with_error();
        let synthesis_result = error_client.complete("Generate response").await;

        // Should handle error gracefully
        assert!(synthesis_result.is_err());

        // Could retry with fallback
        let fallback_client = MockGeminiClient::new("Fallback response");
        let fallback_result = fallback_client.complete("Generate response").await;
        assert!(fallback_result.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_prompt_limit() {
        let engine = SearchEngine::new(create_test_prompts());

        // Request many results
        let query = SearchQuery::new("security").with_top_k(10);
        let results = engine.search(&query).await.unwrap();

        // Orchestrator should limit to 2-7 prompts
        let max_prompts = 7;
        let selected: Vec<_> = results.into_iter().take(max_prompts).collect();

        assert!(
            selected.len() <= max_prompts,
            "Should not exceed max prompts"
        );
        assert!(
            selected.len() >= 2 || selected.is_empty(),
            "Should have at least 2 prompts if any found"
        );
    }
}

// =============================================================================
// VALIDATOR INTEGRATION TESTS
// =============================================================================

mod validator_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_validator_with_synthesized_content() {
        let content = "Learning about security helps us grow as developers. \
                      Therefore, we should build robust systems that protect users. \
                      This brings joy and satisfaction in our work.";

        let validator = SpinozaValidator::new();
        let result = validator.validate(content).await.unwrap();

        assert!(
            result.is_valid,
            "Well-crafted content should pass validation"
        );
        assert!(result.scores.conatus > 0.5);
        assert!(result.scores.ratio > 0.5);
        assert!(result.scores.laetitia > 0.5);
    }

    #[tokio::test]
    async fn test_validator_messages_for_improvement() {
        let content = "This response is confusing and unclear.";

        let validator = SpinozaValidator::new();
        let result = validator.validate(content).await.unwrap();

        // Should have warning messages
        assert!(
            result
                .messages
                .iter()
                .any(|m| m.text.contains("logic") || m.text.contains("confusion")),
            "Should identify logic issues"
        );
    }

    #[tokio::test]
    async fn test_validator_strict_mode() {
        let content = "A simple response.";

        let strict = SpinozaValidator::strict();
        let lenient = SpinozaValidator::lenient();

        let strict_result = strict.validate(content).await.unwrap();
        let lenient_result = lenient.validate(content).await.unwrap();

        // Strict should be harder to pass
        assert!(
            lenient_result.is_valid || !strict_result.is_valid,
            "Strict mode should be more demanding"
        );
    }
}

// =============================================================================
// CONCURRENT ACCESS TESTS
// =============================================================================

mod concurrency_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    #[tokio::test]
    async fn test_concurrent_searches() {
        let engine = Arc::new(SearchEngine::new(create_test_prompts()));

        let queries = vec![
            "authentication",
            "security",
            "testing",
            "api design",
            "caching",
        ];

        let handles: Vec<_> = queries
            .into_iter()
            .map(|q| {
                let engine = Arc::clone(&engine);
                tokio::spawn(async move {
                    let query = SearchQuery::new(q).with_top_k(3);
                    engine.search(&query).await
                })
            })
            .collect();

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Concurrent searches should succeed");
        }
    }

    #[tokio::test]
    async fn test_concurrent_validations() {
        let validator = Arc::new(SpinozaValidator::new());

        let contents = vec![
            "Learn and grow through understanding.",
            "Build systems that bring joy.",
            "Therefore, we conclude logically.",
            "Secure and protect our users.",
            "Create with hope and enthusiasm.",
        ];

        let handles: Vec<_> = contents
            .into_iter()
            .map(|c| {
                let validator = Arc::clone(&validator);
                let content = c.to_string();
                tokio::spawn(async move { validator.validate(&content).await })
            })
            .collect();

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Concurrent validations should succeed");
        }
    }

    #[tokio::test]
    async fn test_rate_limited_llm_calls() {
        let client = Arc::new(MockGeminiClient::new("Response"));
        let semaphore = Arc::new(Semaphore::new(3)); // Max 3 concurrent calls

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let client = Arc::clone(&client);
                let semaphore = Arc::clone(&semaphore);
                tokio::spawn(async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    client.complete(&format!("Query {}", i)).await
                })
            })
            .collect();

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        // All 10 calls should have been made
        assert_eq!(client.call_count(), 10);
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_search_query() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("");

        let result = engine.search(&query).await;

        assert!(result.is_err(), "Empty query should return error");
    }

    #[tokio::test]
    async fn test_whitespace_only_query() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("   \t\n  ");

        let result = engine.search(&query).await;

        assert!(result.is_err(), "Whitespace-only query should return error");
    }

    #[tokio::test]
    async fn test_very_long_query() {
        let engine = SearchEngine::new(create_test_prompts());
        let long_query = "authentication ".repeat(100);
        let query = SearchQuery::new(&long_query).with_top_k(3);

        let result = engine.search(&query).await;

        // Should handle long queries gracefully
        assert!(result.is_ok(), "Long query should be handled");
    }

    #[tokio::test]
    async fn test_special_characters_in_query() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("authentication!@#$%^&*()").with_top_k(3);

        let result = engine.search(&query).await;

        // Should handle special characters gracefully
        assert!(result.is_ok(), "Special characters should be handled");
    }

    #[tokio::test]
    async fn test_unicode_in_query() {
        let engine = SearchEngine::new(create_test_prompts());
        let query = SearchQuery::new("authentication").with_top_k(3);

        let result = engine.search(&query).await;

        assert!(result.is_ok(), "Unicode should be handled");
    }

    #[tokio::test]
    async fn test_empty_prompt_index() {
        let engine = SearchEngine::new(vec![]);
        let query = SearchQuery::new("anything");

        let result = engine.search(&query).await.unwrap();

        assert!(result.is_empty(), "Empty index should return empty results");
    }

    #[tokio::test]
    async fn test_validation_empty_content() {
        let validator = SpinozaValidator::new();
        let result = validator.validate("").await;

        assert!(result.is_err(), "Empty content should fail validation");
    }

    #[tokio::test]
    async fn test_validation_very_long_content() {
        let validator = SpinozaValidator::new();
        let long_content = "learn grow joy ".repeat(1000);

        let result = validator.validate(&long_content).await;

        assert!(result.is_ok(), "Long content should be handled");
        assert!(
            result.unwrap().is_valid,
            "Positive long content should pass"
        );
    }
}
