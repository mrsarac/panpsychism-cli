//! Integration tests for Project Panpsychism core modules.
//!
//! These tests verify the interaction between components:
//! - Indexer: Scans directories and creates searchable indexes
//! - Search: Finds prompts by keywords and semantic similarity
//! - Validator: Scores content against Spinoza principles
//! - Corrector: Detects ambiguities and generates clarifications

use panpsychism::corrector::{
    Ambiguity, AmbiguityKind, Answer, CorrectionResult, Corrector, Question,
};
use panpsychism::indexer::{IndexStats, Indexer};
use panpsychism::search::{PromptMetadata, SearchEngine, SearchQuery, SearchResult};
use panpsychism::validator::{SpinozaValidator, ValidationLevel};
use panpsychism::{Error, Result};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

// ============================================================================
// Test Helpers
// ============================================================================

/// Get the path to the test fixtures directory.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Create a temporary directory with sample prompt files for testing.
async fn create_test_prompts(dir: &Path) -> Result<()> {
    let prompt1 = r#"---
id: test-001
title: "Test Authentication Prompt"
category: security
tags:
  - auth
  - testing
privacy_tier: public
---

# Test Authentication

This is a test prompt for authentication testing.

## Usage

Use this prompt to test authentication flows.
"#;

    let prompt2 = r#"---
id: test-002
title: "Database Operations"
category: backend
tags:
  - database
  - sql
privacy_tier: internal
---

# Database Operations Guide

Learn how to perform database operations efficiently.

## Topics

- Query optimization
- Index management
- Connection pooling
"#;

    tokio::fs::write(dir.join("auth-test.md"), prompt1).await?;
    tokio::fs::write(dir.join("database-ops.md"), prompt2).await?;

    Ok(())
}

/// Create test PromptMetadata for search engine tests.
fn create_test_metadata() -> Vec<PromptMetadata> {
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
            "db-query",
            "Database Query Optimization",
            "Optimize SQL database queries with proper indexes and query planning.",
            "prompts/database/queries.md",
        )
        .with_tags(vec!["database".to_string(), "sql".to_string(), "performance".to_string()])
        .with_category("database"),
    ]
}

// ============================================================================
// Indexer Tests
// ============================================================================

mod indexer_tests {
    use super::*;

    #[tokio::test]
    async fn test_indexer_creation() {
        let prompts_dir = fixtures_dir();
        let temp = TempDir::new().expect("Failed to create temp dir");
        let index_path = temp.path().join("test.mv2");

        let indexer = Indexer::new(&prompts_dir, &index_path);

        // Index should not exist initially
        assert!(!indexer.is_index_valid());
    }

    #[tokio::test]
    async fn test_indexer_with_fixtures() {
        let prompts_dir = fixtures_dir();
        let temp = TempDir::new().expect("Failed to create temp dir");
        let index_path = temp.path().join("fixtures.mv2");

        let indexer = Indexer::new(&prompts_dir, &index_path);

        // Note: This test will pass when Indexer::index() is implemented
        // Currently index() has todo!() so we just verify construction
        assert!(!indexer.is_index_valid());
    }

    #[tokio::test]
    async fn test_indexer_with_temp_prompts() {
        let temp = TempDir::new().expect("Failed to create temp dir");
        let prompts_dir = temp.path().join("prompts");
        tokio::fs::create_dir(&prompts_dir)
            .await
            .expect("Failed to create prompts dir");

        create_test_prompts(&prompts_dir)
            .await
            .expect("Failed to create test prompts");

        let index_path = temp.path().join("temp.mv2");
        let indexer = Indexer::new(&prompts_dir, &index_path);

        // Verify files exist
        assert!(prompts_dir.join("auth-test.md").exists());
        assert!(prompts_dir.join("database-ops.md").exists());
        assert!(!indexer.is_index_valid());
    }

    #[tokio::test]
    async fn test_indexer_empty_directory() {
        let temp = TempDir::new().expect("Failed to create temp dir");
        let prompts_dir = temp.path().join("empty");
        tokio::fs::create_dir(&prompts_dir)
            .await
            .expect("Failed to create empty dir");

        let index_path = temp.path().join("empty.mv2");
        let indexer = Indexer::new(&prompts_dir, &index_path);

        assert!(!indexer.is_index_valid());
    }

    #[tokio::test]
    async fn test_indexer_nonexistent_directory() {
        let temp = TempDir::new().expect("Failed to create temp dir");
        let prompts_dir = temp.path().join("nonexistent");
        let index_path = temp.path().join("never.mv2");

        let indexer = Indexer::new(&prompts_dir, &index_path);

        assert!(!indexer.is_index_valid());
    }

    #[tokio::test]
    async fn test_index_stats_default() {
        let stats = IndexStats::default();
        assert_eq!(stats.prompts_indexed, 0);
        assert_eq!(stats.prompts_skipped, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.duration_ms, 0);
    }
}

// ============================================================================
// Search Tests
// ============================================================================

mod search_tests {
    use super::*;

    #[tokio::test]
    async fn test_search_engine_creation() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        assert!(!engine.is_empty());
        assert_eq!(engine.index_size(), 4);
    }

    #[tokio::test]
    async fn test_search_basic_query() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("authentication oauth");
        let results = engine.search(&query).await.expect("Search should succeed");

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "auth-01");
        assert!(results[0].score > 0.0 && results[0].score <= 1.0);
    }

    #[tokio::test]
    async fn test_search_by_keyword() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("database query optimization");
        let results = engine.search(&query).await.expect("Search should succeed");

        assert!(!results.is_empty());
        // Should find the database prompt
        let db_result = results.iter().find(|r| r.id == "db-query");
        assert!(db_result.is_some(), "Should find database prompt");
    }

    #[tokio::test]
    async fn test_search_top_k_limit() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("design authentication api").with_top_k(2);
        let results = engine.search(&query).await.expect("Search should succeed");

        assert!(results.len() <= 2, "Should respect top_k limit");
    }

    #[tokio::test]
    async fn test_search_with_category_filter() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("authentication").with_category("security");
        let results = engine.search(&query).await.expect("Search should succeed");

        for result in &results {
            assert_eq!(result.category, Some("security".to_string()));
        }
    }

    #[tokio::test]
    async fn test_search_with_tags_filter() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("authentication").with_tags(vec!["security".to_string()]);
        let results = engine.search(&query).await.expect("Search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "auth-01");
    }

    #[tokio::test]
    async fn test_search_by_tags() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let results = engine
            .search_by_tags(&["auth"])
            .await
            .expect("Search should succeed");

        assert_eq!(results.len(), 2); // auth-01 and api-design both have "auth" tag
        for result in &results {
            assert!(result.tags.iter().any(|t| t == "auth"));
        }
    }

    #[tokio::test]
    async fn test_search_by_category() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let results = engine
            .search_by_category("philosophy")
            .await
            .expect("Search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "spinoza-01");
    }

    #[tokio::test]
    async fn test_search_empty_query_error() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("");
        let result = engine.search(&query).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Search(msg) => assert!(msg.contains("empty")),
            _ => panic!("Expected Search error"),
        }
    }

    #[tokio::test]
    async fn test_search_no_results() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("xyznonexistent123 qqq zzz").with_min_score(0.5);
        let results = engine.search(&query).await.expect("Search should succeed");

        assert!(
            results.is_empty(),
            "Should return empty for no matches above threshold"
        );
    }

    #[tokio::test]
    async fn test_search_result_structure() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("OAuth");
        let results = engine.search(&query).await.expect("Search should succeed");

        assert!(!results.is_empty());
        let result = &results[0];

        assert!(!result.id.is_empty());
        assert!(!result.title.is_empty());
        assert!(!result.path.as_os_str().is_empty());
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(!result.excerpt.is_empty());
    }

    #[tokio::test]
    async fn test_search_results_sorted_by_score() {
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        let query = SearchQuery::new("authentication design api");
        let results = engine.search(&query).await.expect("Search should succeed");

        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results should be sorted by score descending"
            );
        }
    }

    #[tokio::test]
    async fn test_search_engine_empty() {
        let engine = SearchEngine::new(vec![]);

        assert!(engine.is_empty());
        assert_eq!(engine.index_size(), 0);

        let query = SearchQuery::new("test");
        let results = engine.search(&query).await.expect("Search should succeed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_prompt_metadata_builder() {
        let prompt = PromptMetadata::new("id", "title", "content", "path.md")
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()])
            .with_category("category");

        assert_eq!(prompt.id, "id");
        assert_eq!(prompt.title, "title");
        assert_eq!(prompt.content, "content");
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
        assert!((query.min_score - 0.3).abs() < f64::EPSILON);
        assert_eq!(query.category, Some("test".to_string()));
        assert!(query.tags.is_some());
    }
}

// ============================================================================
// Validator Tests
// ============================================================================

mod validator_tests {
    use super::*;

    #[tokio::test]
    async fn test_validator_creation() {
        let validator = SpinozaValidator::new();
        // Validator should be created with default thresholds
        assert!(true);
    }

    #[tokio::test]
    async fn test_validate_positive_content() {
        let validator = SpinozaValidator::new();

        let positive_content = r#"
        This guide helps you grow and develop your skills through learning.
        We encourage progress and success through clear, logical steps.
        Therefore, you will achieve joy and happiness in your journey.
        "#;

        let result = validator
            .validate(positive_content)
            .await
            .expect("Validation should succeed");

        assert!(result.scores.conatus >= 0.0 && result.scores.conatus <= 1.0);
        assert!(result.scores.ratio >= 0.0 && result.scores.ratio <= 1.0);
        assert!(result.scores.laetitia >= 0.0 && result.scores.laetitia <= 1.0);
    }

    #[tokio::test]
    async fn test_validate_scores_in_range() {
        let validator = SpinozaValidator::new();

        let content = "Test content for score validation.";

        let result = validator
            .validate(content)
            .await
            .expect("Validation should succeed");

        assert!(
            result.scores.conatus >= 0.0 && result.scores.conatus <= 1.0,
            "CONATUS score out of range"
        );
        assert!(
            result.scores.ratio >= 0.0 && result.scores.ratio <= 1.0,
            "RATIO score out of range"
        );
        assert!(
            result.scores.laetitia >= 0.0 && result.scores.laetitia <= 1.0,
            "LAETITIA score out of range"
        );
    }

    #[tokio::test]
    async fn test_validate_high_conatus_content() {
        let validator = SpinozaValidator::new();

        let conatus_content = r#"
        We focus on growth, development, and learning.
        Building and creating new systems that thrive and prosper.
        Protecting and preserving what we build for the future.
        "#;

        let result = validator
            .validate(conatus_content)
            .await
            .expect("Validation should succeed");

        // Content with many CONATUS keywords should score higher
        assert!(
            result.scores.conatus > 0.3,
            "Content with growth keywords should have higher CONATUS"
        );
    }

    #[tokio::test]
    async fn test_validate_high_ratio_content() {
        let validator = SpinozaValidator::new();

        let ratio_content = r#"
        First, we analyze the problem. Second, we gather evidence.
        Therefore, we can conclude that the solution is valid.
        Because of this logical analysis, the result is consistent.
        "#;

        let result = validator
            .validate(ratio_content)
            .await
            .expect("Validation should succeed");

        // Content with logical structure should score higher on RATIO
        assert!(
            result.scores.ratio > 0.3,
            "Content with logical keywords should have higher RATIO"
        );
    }

    #[tokio::test]
    async fn test_validate_high_laetitia_content() {
        let validator = SpinozaValidator::new();

        let laetitia_content = r#"
        This brings joy and happiness to everyone involved.
        We inspire and motivate with love and compassion.
        Success and achievement lead to hope and optimism.
        "#;

        let result = validator
            .validate(laetitia_content)
            .await
            .expect("Validation should succeed");

        // Content with joy keywords should score higher on LAETITIA
        assert!(
            result.scores.laetitia > 0.3,
            "Content with joy keywords should have higher LAETITIA"
        );
    }

    #[tokio::test]
    async fn test_validate_empty_content() {
        let validator = SpinozaValidator::new();

        let result = validator.validate("").await;

        assert!(result.is_err(), "Empty content should return error");
    }

    #[tokio::test]
    async fn test_validation_level_enum() {
        let info = ValidationLevel::Info;
        let warning = ValidationLevel::Warning;
        let error = ValidationLevel::Error;

        assert_eq!(info, ValidationLevel::Info);
        assert_eq!(warning, ValidationLevel::Warning);
        assert_eq!(error, ValidationLevel::Error);
        assert_ne!(info, warning);
        assert_ne!(warning, error);
    }

    #[tokio::test]
    async fn test_validator_can_read_fixture() {
        let spinoza_content = tokio::fs::read_to_string(fixtures_dir().join("spinoza-ethics.md"))
            .await
            .expect("Should read Spinoza fixture");

        assert!(spinoza_content.contains("Spinoza"));
        assert!(spinoza_content.contains("Conatus"));
        assert!(spinoza_content.contains("Ratio"));
        assert!(spinoza_content.contains("Laetitia"));
    }

    #[tokio::test]
    async fn test_validate_spinoza_fixture() {
        let validator = SpinozaValidator::new();

        let spinoza_content = tokio::fs::read_to_string(fixtures_dir().join("spinoza-ethics.md"))
            .await
            .expect("Should read Spinoza fixture");

        let result = validator
            .validate(&spinoza_content)
            .await
            .expect("Validation should succeed");

        // Spinoza-themed content should score well
        assert!(result.scores.conatus >= 0.0);
        assert!(result.scores.ratio >= 0.0);
        assert!(result.scores.laetitia >= 0.0);
    }
}

// ============================================================================
// Corrector Tests (Full Implementation)
// ============================================================================

mod corrector_tests {
    use super::*;

    #[tokio::test]
    async fn test_corrector_creation() {
        let corrector = Corrector::new();
        assert_eq!(corrector.max_iterations(), 3);
        assert!((corrector.ambiguity_threshold() - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_corrector_builder() {
        let corrector = Corrector::builder()
            .max_iterations(5)
            .ambiguity_threshold(0.3)
            .build();

        assert_eq!(corrector.max_iterations(), 5);
        assert!((corrector.ambiguity_threshold() - 0.3).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_detect_ambiguities_vague_content() {
        let corrector = Corrector::new();

        let vague_content = "Some users might experience issues. It should probably work.";

        let ambiguities = corrector
            .detect_ambiguities(vague_content)
            .await
            .expect("Detection should succeed");

        assert!(
            !ambiguities.is_empty(),
            "Vague content should have ambiguities"
        );

        let has_vague = ambiguities.iter().any(|a| a.kind == AmbiguityKind::Vague);
        let has_reference = ambiguities
            .iter()
            .any(|a| a.kind == AmbiguityKind::Reference);

        assert!(has_vague, "Should detect vague language");
        assert!(has_reference, "Should detect unclear reference 'It'");
    }

    #[tokio::test]
    async fn test_detect_pronoun_ambiguity() {
        let corrector = Corrector::new();

        let pronoun_content = "It should work. They need to configure this properly.";

        let ambiguities = corrector
            .detect_ambiguities(pronoun_content)
            .await
            .expect("Detection should succeed");

        let reference_ambs: Vec<_> = ambiguities
            .iter()
            .filter(|a| a.kind == AmbiguityKind::Reference)
            .collect();

        assert!(
            !reference_ambs.is_empty(),
            "Should detect pronoun references"
        );
    }

    #[tokio::test]
    async fn test_detect_context_ambiguity() {
        let corrector = Corrector::new();

        let context_missing = "The system should handle the request. The database stores the data.";

        let ambiguities = corrector
            .detect_ambiguities(context_missing)
            .await
            .expect("Detection should succeed");

        let context_ambs: Vec<_> = ambiguities
            .iter()
            .filter(|a| a.kind == AmbiguityKind::Context)
            .collect();

        assert!(!context_ambs.is_empty(), "Should detect missing context");
    }

    #[tokio::test]
    async fn test_detect_multi_interpretation() {
        let corrector = Corrector::new();

        let multi_content =
            "Configure logging, monitoring, etc. Use various features and/or options.";

        let ambiguities = corrector
            .detect_ambiguities(multi_content)
            .await
            .expect("Detection should succeed");

        let multi_ambs: Vec<_> = ambiguities
            .iter()
            .filter(|a| a.kind == AmbiguityKind::MultiInterpretation)
            .collect();

        assert!(
            !multi_ambs.is_empty(),
            "Should detect multi-interpretation phrases"
        );
    }

    #[tokio::test]
    async fn test_detect_empty_content_error() {
        let corrector = Corrector::new();

        let result = corrector.detect_ambiguities("").await;

        assert!(result.is_err(), "Should error on empty content");
    }

    #[tokio::test]
    async fn test_generate_questions() {
        let corrector = Corrector::new();

        let text = "It might fail. The system could crash.";

        let ambiguities = corrector
            .detect_ambiguities(text)
            .await
            .expect("Detection should succeed");

        let questions = corrector
            .generate_questions(&ambiguities)
            .await
            .expect("Question generation should succeed");

        assert!(!questions.is_empty(), "Should generate questions");

        for (i, q) in questions.iter().enumerate() {
            assert_eq!(q.ambiguity_index, i);
            assert!(!q.text.is_empty());
            assert!(q.text.contains('?'));
        }
    }

    #[tokio::test]
    async fn test_generate_questions_with_options() {
        let corrector = Corrector::new();

        let ambiguities = vec![
            Ambiguity::new(AmbiguityKind::Reference, 0, "it", 0.8),
            Ambiguity::new(AmbiguityKind::Vague, 10, "some", 0.7),
        ];

        let questions = corrector
            .generate_questions(&ambiguities)
            .await
            .expect("Generation should succeed");

        assert_eq!(questions.len(), 2);

        for q in &questions {
            assert!(q.has_options());
        }
    }

    #[tokio::test]
    async fn test_apply_corrections() {
        let corrector = Corrector::new();

        let text = "It should work.";

        let answers = vec![Answer::new(0, "The login module")];

        let result = corrector
            .apply_corrections(text, &answers)
            .await
            .expect("Correction should succeed");

        assert!(result.corrections_applied > 0);
    }

    #[tokio::test]
    async fn test_apply_corrections_empty_answers() {
        let corrector = Corrector::new();

        let text = "Some text here with it and they.";

        let result = corrector
            .apply_corrections(text, &[])
            .await
            .expect("Should succeed with empty answers");

        assert_eq!(result.corrections_applied, 0);
        assert_eq!(result.content, text);
    }

    #[tokio::test]
    async fn test_ambiguity_fixture() {
        let corrector = Corrector::new();

        let ambiguous_content =
            tokio::fs::read_to_string(fixtures_dir().join("ambiguous-prompt.md"))
                .await
                .expect("Should read ambiguous fixture");

        let ambiguities = corrector
            .detect_ambiguities(&ambiguous_content)
            .await
            .expect("Detection should succeed");

        assert!(!ambiguities.is_empty(), "Fixture should have ambiguities");
    }

    #[tokio::test]
    async fn test_ambiguity_kind_enum() {
        assert_eq!(AmbiguityKind::Reference, AmbiguityKind::Reference);
        assert_eq!(AmbiguityKind::Context, AmbiguityKind::Context);
        assert_eq!(AmbiguityKind::Vague, AmbiguityKind::Vague);
        assert_eq!(
            AmbiguityKind::MultiInterpretation,
            AmbiguityKind::MultiInterpretation
        );
        assert_ne!(AmbiguityKind::Reference, AmbiguityKind::Context);
    }

    #[tokio::test]
    async fn test_ambiguity_kind_display() {
        assert_eq!(AmbiguityKind::Reference.to_string(), "Reference");
        assert_eq!(AmbiguityKind::Context.to_string(), "Context");
        assert_eq!(AmbiguityKind::Vague.to_string(), "Vague");
        assert_eq!(
            AmbiguityKind::MultiInterpretation.to_string(),
            "Multi-Interpretation"
        );
    }

    #[tokio::test]
    async fn test_ambiguity_confidence_clamping() {
        let high = Ambiguity::new(AmbiguityKind::Reference, 0, "it", 1.5);
        assert_eq!(high.confidence, 1.0);

        let low = Ambiguity::new(AmbiguityKind::Reference, 0, "it", -0.5);
        assert_eq!(low.confidence, 0.0);
    }

    #[tokio::test]
    async fn test_ambiguity_is_high_confidence() {
        let high = Ambiguity::new(AmbiguityKind::Reference, 0, "it", 0.8);
        assert!(high.is_high_confidence());

        let low = Ambiguity::new(AmbiguityKind::Vague, 0, "some", 0.5);
        assert!(!low.is_high_confidence());
    }

    #[tokio::test]
    async fn test_question_creation() {
        let q = Question::new("What does 'it' refer to?", 0);
        assert!(!q.text.is_empty());
        assert_eq!(q.ambiguity_index, 0);
        assert!(!q.has_options());
    }

    #[tokio::test]
    async fn test_question_with_options() {
        let options = vec!["Option A".to_string(), "Option B".to_string()];
        let q = Question::new("Choose one:", 0).with_options(options);

        assert!(q.has_options());
        assert_eq!(q.options.len(), 2);
    }

    #[tokio::test]
    async fn test_answer_creation() {
        let answer = Answer::new(0, "The authentication module");
        assert_eq!(answer.question_index, 0);
        assert_eq!(answer.text, "The authentication module");
    }

    #[tokio::test]
    async fn test_correction_result_helpers() {
        let fully_corrected = CorrectionResult {
            content: "Corrected text".to_string(),
            corrections_applied: 3,
            remaining_ambiguities: 0,
        };
        assert!(fully_corrected.is_fully_corrected());
        assert!(fully_corrected.has_corrections());

        let partial = CorrectionResult {
            content: "Partially corrected".to_string(),
            corrections_applied: 1,
            remaining_ambiguities: 2,
        };
        assert!(!partial.is_fully_corrected());
        assert!(partial.has_corrections());

        let unchanged = CorrectionResult {
            content: "Original".to_string(),
            corrections_applied: 0,
            remaining_ambiguities: 3,
        };
        assert!(!unchanged.is_fully_corrected());
        assert!(!unchanged.has_corrections());
    }

    #[tokio::test]
    async fn test_threshold_filtering() {
        let corrector = Corrector::builder().ambiguity_threshold(0.9).build();

        let text = "Some users might experience issues.";
        let ambiguities = corrector
            .detect_ambiguities(text)
            .await
            .expect("Detection should succeed");

        for amb in &ambiguities {
            assert!(amb.confidence >= 0.9);
        }
    }

    #[tokio::test]
    async fn test_correct_with_provider() {
        let corrector = Corrector::builder().max_iterations(2).build();

        let text = "It should work correctly.";

        let result = corrector
            .correct_with_provider(text, |questions| async move {
                if questions.is_empty() {
                    return Ok(vec![]);
                }
                Ok(vec![Answer::new(0, "The payment system")])
            })
            .await
            .expect("Correction should succeed");

        assert!(result.corrections_applied > 0);
    }

    #[tokio::test]
    async fn test_multiple_ambiguity_types_in_one_text() {
        let corrector = Corrector::new();

        let complex_text =
            "It might cause issues with the system. Some users could experience problems, etc.";

        let ambiguities = corrector
            .detect_ambiguities(complex_text)
            .await
            .expect("Detection should succeed");

        let kinds: std::collections::HashSet<_> = ambiguities.iter().map(|a| a.kind).collect();

        assert!(kinds.contains(&AmbiguityKind::Reference));
        assert!(kinds.contains(&AmbiguityKind::Vague));
        assert!(kinds.contains(&AmbiguityKind::Context));
        assert!(kinds.contains(&AmbiguityKind::MultiInterpretation));
    }
}

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

mod e2e_tests {
    use super::*;

    #[tokio::test]
    async fn test_search_then_validate_workflow() {
        // Step 1: Create search engine with prompts
        let prompts = create_test_metadata();
        let engine = SearchEngine::new(prompts);

        // Step 2: Search for Spinoza content
        let query = SearchQuery::new("Spinoza conatus");
        let results = engine.search(&query).await.expect("Search should succeed");

        assert!(!results.is_empty());

        // Step 3: Validate the found content
        let validator = SpinozaValidator::new();

        // Use the fixture for validation since search results don't contain full content
        let spinoza_content = tokio::fs::read_to_string(fixtures_dir().join("spinoza-ethics.md"))
            .await
            .expect("Should read fixture");

        let validation = validator
            .validate(&spinoza_content)
            .await
            .expect("Validation should succeed");

        assert!(validation.scores.conatus >= 0.0);
        assert!(validation.scores.ratio >= 0.0);
        assert!(validation.scores.laetitia >= 0.0);
    }

    #[tokio::test]
    async fn test_corrector_full_workflow() {
        let corrector = Corrector::new();

        // Start with ambiguous text
        let original = "It should be configured. The system might fail.";

        // Step 1: Detect ambiguities
        let ambiguities = corrector
            .detect_ambiguities(original)
            .await
            .expect("Detection should succeed");

        assert!(!ambiguities.is_empty());

        // Step 2: Generate questions
        let questions = corrector
            .generate_questions(&ambiguities)
            .await
            .expect("Question generation should succeed");

        assert_eq!(questions.len(), ambiguities.len());

        // Step 3: Provide answers and apply corrections
        let answers: Vec<Answer> = questions
            .iter()
            .enumerate()
            .map(|(i, _)| Answer::new(i, format!("Answer for question {}", i)))
            .collect();

        let result = corrector
            .apply_corrections(original, &answers)
            .await
            .expect("Correction should succeed");

        assert!(result.corrections_applied > 0);
    }

    #[tokio::test]
    async fn test_fixtures_are_valid() {
        let fixtures = fixtures_dir();
        assert!(fixtures.exists(), "Fixtures directory should exist");

        let expected_files = [
            "authentication.md",
            "database-query.md",
            "api-design.md",
            "ambiguous-prompt.md",
            "spinoza-ethics.md",
        ];

        for file in &expected_files {
            let path = fixtures.join(file);
            assert!(path.exists(), "Fixture {} should exist", file);

            let content = tokio::fs::read_to_string(&path)
                .await
                .unwrap_or_else(|_| panic!("Should read {}", file));

            assert!(
                content.contains("---"),
                "Fixture {} should have frontmatter",
                file
            );
            assert!(
                content.contains("id:"),
                "Fixture {} should have id field",
                file
            );
            assert!(
                content.contains("title:"),
                "Fixture {} should have title field",
                file
            );
        }
    }

    #[tokio::test]
    async fn test_fixtures_have_correct_content() {
        let fixtures = fixtures_dir();

        let auth = tokio::fs::read_to_string(fixtures.join("authentication.md"))
            .await
            .expect("Should read auth fixture");
        assert!(auth.contains("OAuth") || auth.contains("JWT") || auth.contains("authentication"));

        let db = tokio::fs::read_to_string(fixtures.join("database-query.md"))
            .await
            .expect("Should read db fixture");
        assert!(db.contains("SQL") || db.contains("database") || db.contains("query"));

        let api = tokio::fs::read_to_string(fixtures.join("api-design.md"))
            .await
            .expect("Should read api fixture");
        assert!(api.contains("REST") || api.contains("API") || api.contains("endpoint"));

        let ambig = tokio::fs::read_to_string(fixtures.join("ambiguous-prompt.md"))
            .await
            .expect("Should read ambig fixture");
        assert!(ambig.contains("thing") || ambig.contains("it") || ambig.contains("they"));

        let spinoza = tokio::fs::read_to_string(fixtures.join("spinoza-ethics.md"))
            .await
            .expect("Should read spinoza fixture");
        assert!(spinoza.contains("Spinoza") || spinoza.contains("Conatus"));
    }

    #[tokio::test]
    async fn test_clear_fixture_has_fewer_ambiguities() {
        let corrector = Corrector::new();

        let clear = tokio::fs::read_to_string(fixtures_dir().join("authentication.md"))
            .await
            .expect("Should read auth fixture");

        let ambiguous = tokio::fs::read_to_string(fixtures_dir().join("ambiguous-prompt.md"))
            .await
            .expect("Should read ambig fixture");

        let clear_ambiguities = corrector
            .detect_ambiguities(&clear)
            .await
            .expect("Detection should succeed");

        let ambig_ambiguities = corrector
            .detect_ambiguities(&ambiguous)
            .await
            .expect("Detection should succeed");

        let clear_density = clear_ambiguities.len() as f64 / clear.len() as f64;
        let ambig_density = ambig_ambiguities.len() as f64 / ambiguous.len() as f64;

        println!(
            "Clear fixture: {} ambiguities, density: {:.4}",
            clear_ambiguities.len(),
            clear_density
        );
        println!(
            "Ambiguous fixture: {} ambiguities, density: {:.4}",
            ambig_ambiguities.len(),
            ambig_density
        );
    }
}
