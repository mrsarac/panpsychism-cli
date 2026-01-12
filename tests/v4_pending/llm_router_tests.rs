//! Integration tests for LLM Router
//!
//! These tests verify the LLM Router functionality including:
//! - Provider registration and management
//! - Routing strategies (Primary, LoadBalance, CostOptimized, etc.)
//! - Fallback mechanisms
//! - Circuit breaker behavior
//! - Cost tracking
//! - Concurrent request handling
//!
//! Uses wiremock to simulate LLM API responses without real API calls.

use panpsychism::llm::router::{
    CircuitBreaker, CircuitState, CostTracker, LLMProvider, LLMRouter, ProviderConfig,
    ProviderHealthStatus, RouterStats, RoutingStrategy,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a mock OpenAI-compatible response
fn mock_openai_response() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I am a mock response."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    })
}

/// Create a mock Anthropic-compatible response
fn mock_anthropic_response() -> serde_json::Value {
    serde_json::json!({
        "id": "msg-test123",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "Hello from Claude!"
        }],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        }
    })
}

/// Create a mock Ollama-compatible response
fn mock_ollama_response() -> serde_json::Value {
    serde_json::json!({
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "response": "Hello from Ollama!",
        "done": true,
        "context": [],
        "total_duration": 500000000,
        "load_duration": 100000000,
        "prompt_eval_count": 10,
        "eval_count": 20
    })
}

// ============================================================================
// Provider Registration Tests
// ============================================================================

mod provider_registration {
    use super::*;

    #[tokio::test]
    async fn test_register_openai_provider() {
        let mock_server = MockServer::start().await;

        // Setup mock endpoint
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]
            })))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::new();
        let config = ProviderConfig::new(LLMProvider::OpenAI)
            .with_endpoint(mock_server.uri())
            .with_api_key("sk-test-key")
            .with_priority(0);

        router.register_provider(config);

        assert!(router.has_provider(LLMProvider::OpenAI));
        assert_eq!(router.provider_count(), 1);

        let provider = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert_eq!(provider.provider, LLMProvider::OpenAI);
        assert!(provider.endpoint.contains(&mock_server.uri()));
    }

    #[tokio::test]
    async fn test_register_anthropic_provider() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_anthropic_response()))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::new();
        let config = ProviderConfig::new(LLMProvider::Anthropic)
            .with_endpoint(mock_server.uri())
            .with_api_key("sk-ant-test")
            .with_priority(1);

        router.register_provider(config);

        assert!(router.has_provider(LLMProvider::Anthropic));
        let provider = router.get_provider(LLMProvider::Anthropic).unwrap();
        assert_eq!(provider.priority, 1);
    }

    #[tokio::test]
    async fn test_register_ollama_provider() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{"name": "llama3.2"}, {"name": "mistral"}]
            })))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::new();
        let config = ProviderConfig::new(LLMProvider::Ollama)
            .with_endpoint(mock_server.uri())
            .with_default_model("llama3.2");

        router.register_provider(config);

        assert!(router.has_provider(LLMProvider::Ollama));
        let provider = router.get_provider(LLMProvider::Ollama).unwrap();
        assert_eq!(provider.default_model, "llama3.2");
        // Ollama has zero cost (local)
        assert_eq!(provider.input_cost_per_1k, 0.0);
    }

    #[tokio::test]
    async fn test_register_custom_provider() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "response": "Custom response",
                "done": true
            })))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::new();
        let config = ProviderConfig::new(LLMProvider::Custom)
            .with_endpoint(mock_server.uri())
            .with_default_model("custom-model")
            .with_costs(0.001, 0.002)
            .with_quality_score(0.8);

        router.register_provider(config);

        assert!(router.has_provider(LLMProvider::Custom));
        let provider = router.get_provider(LLMProvider::Custom).unwrap();
        assert_eq!(provider.input_cost_per_1k, 0.001);
        assert_eq!(provider.output_cost_per_1k, 0.002);
        assert_eq!(provider.quality_score, 0.8);
    }

    #[tokio::test]
    async fn test_unregister_provider() {
        let router = LLMRouter::new();
        router.register_provider(ProviderConfig::new(LLMProvider::OpenAI));
        router.register_provider(ProviderConfig::new(LLMProvider::Anthropic));

        assert_eq!(router.provider_count(), 2);

        assert!(router.unregister_provider(LLMProvider::OpenAI));
        assert_eq!(router.provider_count(), 1);
        assert!(!router.has_provider(LLMProvider::OpenAI));
        assert!(router.has_provider(LLMProvider::Anthropic));

        // Unregister non-existent
        assert!(!router.unregister_provider(LLMProvider::OpenAI));
    }
}

// ============================================================================
// Routing Strategy Tests
// ============================================================================

mod routing_strategy {
    use super::*;

    #[tokio::test]
    async fn test_primary_routing() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_priority(1))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_priority(0))
            .provider(ProviderConfig::new(LLMProvider::Google).with_priority(2))
            .strategy(RoutingStrategy::Primary)
            .build();

        // Primary routing selects lowest priority
        let selected = router.select_provider().unwrap();
        assert_eq!(selected, LLMProvider::Anthropic);

        // Multiple calls should always return same primary
        for _ in 0..10 {
            assert_eq!(router.select_provider().unwrap(), LLMProvider::Anthropic);
        }
    }

    #[tokio::test]
    async fn test_load_balance_routing() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .provider(ProviderConfig::new(LLMProvider::Google))
            .strategy(RoutingStrategy::LoadBalance)
            .build();

        let mut counts: HashMap<LLMProvider, u32> = HashMap::new();

        // Make many selections
        for _ in 0..30 {
            let selected = router.select_provider().unwrap();
            *counts.entry(selected).or_insert(0) += 1;
        }

        // Each provider should be selected roughly equally (round-robin)
        assert_eq!(counts.get(&LLMProvider::OpenAI), Some(&10));
        assert_eq!(counts.get(&LLMProvider::Anthropic), Some(&10));
        assert_eq!(counts.get(&LLMProvider::Google), Some(&10));
    }

    #[tokio::test]
    async fn test_cost_optimized_routing() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_costs(0.01, 0.03))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_costs(0.003, 0.015))
            .provider(ProviderConfig::new(LLMProvider::Google).with_costs(0.00025, 0.0005))
            .provider(ProviderConfig::new(LLMProvider::Ollama).with_costs(0.0, 0.0))
            .strategy(RoutingStrategy::CostOptimized)
            .build();

        let selected = router.select_provider().unwrap();
        // Ollama has zero cost
        assert_eq!(selected, LLMProvider::Ollama);

        // Disable Ollama and try again
        router.disable_provider(LLMProvider::Ollama);
        let selected = router.select_provider().unwrap();
        // Google is cheapest after Ollama
        assert_eq!(selected, LLMProvider::Google);
    }

    #[tokio::test]
    async fn test_quality_optimized_routing() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_quality_score(0.9))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_quality_score(0.95))
            .provider(ProviderConfig::new(LLMProvider::Google).with_quality_score(0.85))
            .provider(ProviderConfig::new(LLMProvider::Ollama).with_quality_score(0.75))
            .strategy(RoutingStrategy::QualityOptimized)
            .build();

        let selected = router.select_provider().unwrap();
        // Anthropic has highest quality
        assert_eq!(selected, LLMProvider::Anthropic);
    }

    #[tokio::test]
    async fn test_latency_optimized_routing() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_latency(500))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_latency(600))
            .provider(ProviderConfig::new(LLMProvider::Ollama).with_latency(100))
            .strategy(RoutingStrategy::LatencyOptimized)
            .build();

        let selected = router.select_provider().unwrap();
        // Ollama has lowest latency
        assert_eq!(selected, LLMProvider::Ollama);
    }

    #[tokio::test]
    async fn test_hybrid_routing() {
        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::OpenAI)
                    .with_costs(0.01, 0.03)
                    .with_quality_score(0.9)
                    .with_latency(500),
            )
            .provider(
                ProviderConfig::new(LLMProvider::Anthropic)
                    .with_costs(0.003, 0.015)
                    .with_quality_score(0.92)
                    .with_latency(600),
            )
            .provider(
                ProviderConfig::new(LLMProvider::Google)
                    .with_costs(0.00025, 0.0005)
                    .with_quality_score(0.85)
                    .with_latency(400),
            )
            .strategy(RoutingStrategy::hybrid(0.5, 0.3, 0.2)) // Cost-focused hybrid
            .build();

        let selected = router.select_provider().unwrap();
        // With heavy cost weight, Google should win (cheapest)
        assert_eq!(selected, LLMProvider::Google);
    }

    #[tokio::test]
    async fn test_hybrid_balanced_routing() {
        let strategy = RoutingStrategy::hybrid_balanced();
        assert!(strategy.is_hybrid());

        if let RoutingStrategy::Hybrid {
            cost_weight,
            quality_weight,
            latency_weight,
        } = strategy
        {
            let total = cost_weight + quality_weight + latency_weight;
            assert!((total - 1.0).abs() < 0.01, "Weights should sum to ~1.0");
        }
    }

    #[tokio::test]
    async fn test_strategy_change_at_runtime() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_priority(0))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_priority(1))
            .strategy(RoutingStrategy::Primary)
            .build();

        assert_eq!(router.select_provider().unwrap(), LLMProvider::OpenAI);

        // Change strategy at runtime
        router.set_strategy(RoutingStrategy::LoadBalance);

        // Should now alternate
        let first = router.select_provider().unwrap();
        let second = router.select_provider().unwrap();
        assert_ne!(first, second);
    }
}

// ============================================================================
// Fallback Tests
// ============================================================================

mod fallback_tests {
    use super::*;

    #[tokio::test]
    async fn test_fallback_on_failure() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .provider(ProviderConfig::new(LLMProvider::Google))
            .fallback_chain(vec![
                LLMProvider::OpenAI,
                LLMProvider::Anthropic,
                LLMProvider::Google,
            ])
            .build();

        // Simulate OpenAI failure
        let fallback = router.select_fallback(LLMProvider::OpenAI);
        assert_eq!(fallback, Some(LLMProvider::Anthropic));

        // Simulate Anthropic failure
        let fallback = router.select_fallback(LLMProvider::Anthropic);
        assert_eq!(fallback, Some(LLMProvider::Google));
    }

    #[tokio::test]
    async fn test_fallback_chain_exhaustion() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .fallback_chain(vec![LLMProvider::OpenAI, LLMProvider::Anthropic])
            .build();

        // After last provider in chain fails, no more fallback
        let fallback = router.select_fallback(LLMProvider::Anthropic);
        assert_eq!(fallback, None);
    }

    #[tokio::test]
    async fn test_fallback_skips_disabled_providers() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .provider(ProviderConfig::new(LLMProvider::Google))
            .fallback_chain(vec![
                LLMProvider::OpenAI,
                LLMProvider::Anthropic,
                LLMProvider::Google,
            ])
            .build();

        // Disable Anthropic
        router.disable_provider(LLMProvider::Anthropic);

        // Fallback from OpenAI should skip Anthropic
        let fallback = router.select_fallback(LLMProvider::OpenAI);
        assert_eq!(fallback, Some(LLMProvider::Google));
    }

    #[tokio::test]
    async fn test_fallback_counter() {
        let router = LLMRouter::new();
        router.register_provider(ProviderConfig::new(LLMProvider::OpenAI));

        router.record_fallback();
        router.record_fallback();
        router.record_fallback();

        let stats = router.stats();
        assert_eq!(stats.fallback_requests, 3);
    }
}

// ============================================================================
// Circuit Breaker Tests
// ============================================================================

mod circuit_breaker_tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_activation() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .circuit_breaker(3, 2, Duration::from_secs(30))
            .build();

        // Record failures until circuit opens
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::OpenAI);

        let cb = router.circuit_breaker(LLMProvider::OpenAI).unwrap();
        assert_eq!(cb.state(), CircuitState::Closed);

        router.record_failure(LLMProvider::OpenAI);
        assert_eq!(cb.state(), CircuitState::Open);

        // Anthropic should still be healthy
        let cb_anthropic = router.circuit_breaker(LLMProvider::Anthropic).unwrap();
        assert_eq!(cb_anthropic.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let cb = CircuitBreaker::new(2, 2, Duration::from_millis(100));

        // Trip the circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for recovery timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should transition to half-open
        assert!(cb.should_allow());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Record successes to close
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_failure() {
        let cb = CircuitBreaker::new(2, 2, Duration::from_millis(50));

        // Trip and recover to half-open
        cb.record_failure();
        cb.record_failure();
        tokio::time::sleep(Duration::from_millis(60)).await;
        cb.should_allow(); // Transitions to half-open

        // Failure in half-open opens circuit again
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(2, 2, Duration::from_secs(30));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert!(cb.should_allow());
    }

    #[tokio::test]
    async fn test_circuit_breaker_blocks_requests_when_open() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .circuit_breaker(2, 2, Duration::from_secs(60))
            .build();

        // Trip the circuit
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::OpenAI);

        // Provider should be unavailable
        let result = router.select_provider();
        assert!(result.is_err());
    }
}

// ============================================================================
// Cost Tracking Tests
// ============================================================================

mod cost_tracking_tests {
    use super::*;

    #[tokio::test]
    async fn test_cost_calculation() {
        let config = ProviderConfig::new(LLMProvider::OpenAI).with_costs(0.01, 0.03);

        // 1000 input tokens, 500 output tokens
        let cost = config.calculate_cost(1000, 500);
        // Expected: 1.0 * 0.01 + 0.5 * 0.03 = 0.01 + 0.015 = 0.025
        assert!((cost - 0.025).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_token_counting() {
        let tracker = CostTracker::new();

        tracker.record("openai", 1000, 500, 0.01, 0.03);
        tracker.record("anthropic", 2000, 1000, 0.003, 0.015);

        assert_eq!(tracker.total_input_tokens(), 3000);
        assert_eq!(tracker.total_output_tokens(), 1500);
        assert_eq!(tracker.total_tokens(), 4500);
        assert_eq!(tracker.request_count(), 2);
    }

    #[tokio::test]
    async fn test_per_provider_costs() {
        let tracker = CostTracker::new();

        tracker.record("openai", 1000, 500, 0.01, 0.03);
        tracker.record("anthropic", 1000, 500, 0.003, 0.015);

        let openai_cost = tracker.provider_cost_usd("openai");
        let anthropic_cost = tracker.provider_cost_usd("anthropic");

        // OpenAI: 0.01 + 0.015 = 0.025
        assert!((openai_cost - 0.025).abs() < 0.0001);
        // Anthropic: 0.003 + 0.0075 = 0.0105
        assert!((anthropic_cost - 0.0105).abs() < 0.0001);

        // Unknown provider returns 0
        assert_eq!(tracker.provider_cost_usd("unknown"), 0.0);
    }

    #[tokio::test]
    async fn test_budget_limit() {
        let tracker = CostTracker::new();
        let budget_limit = 0.10; // $0.10 budget

        // Record usage until budget exceeded
        tracker.record("openai", 1000, 500, 0.01, 0.03); // $0.025
        assert!(tracker.total_cost_usd() < budget_limit);

        tracker.record("openai", 1000, 500, 0.01, 0.03); // $0.050
        assert!(tracker.total_cost_usd() < budget_limit);

        tracker.record("openai", 1000, 500, 0.01, 0.03); // $0.075
        assert!(tracker.total_cost_usd() < budget_limit);

        tracker.record("openai", 1000, 500, 0.01, 0.03); // $0.100
        assert!(tracker.total_cost_usd() >= budget_limit);
    }

    #[tokio::test]
    async fn test_cost_snapshot() {
        let tracker = CostTracker::new();
        tracker.record("openai", 1000, 500, 0.01, 0.03);

        let snapshot = tracker.snapshot();

        assert_eq!(snapshot.total_input_tokens, 1000);
        assert_eq!(snapshot.total_output_tokens, 500);
        assert_eq!(snapshot.request_count, 1);
        assert!(snapshot.provider_costs.contains_key("openai"));
    }

    #[tokio::test]
    async fn test_cost_tracker_reset() {
        let tracker = CostTracker::new();
        tracker.record("openai", 1000, 500, 0.01, 0.03);
        tracker.record("anthropic", 500, 250, 0.003, 0.015);

        tracker.reset();

        assert_eq!(tracker.total_tokens(), 0);
        assert_eq!(tracker.total_cost_usd(), 0.0);
        assert_eq!(tracker.request_count(), 0);
        assert!(tracker.all_provider_costs_usd().is_empty());
    }
}

// ============================================================================
// Integration Tests with Mock Server
// ============================================================================

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_request_cycle() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response()))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::OpenAI)
                    .with_endpoint(mock_server.uri())
                    .with_api_key("test-key"),
            )
            .strategy(RoutingStrategy::Primary)
            .build();

        // Select provider
        let provider = router.select_provider().unwrap();
        assert_eq!(provider, LLMProvider::OpenAI);

        // Simulate successful request
        router.record_success(LLMProvider::OpenAI, 150);

        let stats = router.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.avg_latency_ms, 150.0);
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        let router = Arc::new(
            LLMRouter::builder()
                .provider(ProviderConfig::new(LLMProvider::OpenAI))
                .provider(ProviderConfig::new(LLMProvider::Anthropic))
                .provider(ProviderConfig::new(LLMProvider::Google))
                .strategy(RoutingStrategy::LoadBalance)
                .build(),
        );

        let mut handles = Vec::new();

        // Spawn 30 concurrent requests
        for _ in 0..30 {
            let router_clone = Arc::clone(&router);
            let handle = tokio::spawn(async move {
                let provider = router_clone.select_provider().unwrap();
                router_clone.record_success(provider, 100);
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let stats = router.stats();
        assert_eq!(stats.total_requests, 30);
        assert_eq!(stats.successful_requests, 30);

        // Each provider should have ~10 requests (load balanced)
        let openai = stats.provider_requests.get("openai").unwrap_or(&0);
        let anthropic = stats.provider_requests.get("anthropic").unwrap_or(&0);
        let google = stats.provider_requests.get("google").unwrap_or(&0);
        assert_eq!(openai + anthropic + google, 30);
    }

    #[tokio::test]
    async fn test_provider_health_check_mock() {
        let mock_server = MockServer::start().await;

        // Setup health endpoint
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "status": "healthy"
            })))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::OpenAI).with_endpoint(mock_server.uri()),
            )
            .build();

        assert!(router.has_provider(LLMProvider::OpenAI));

        let config = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_mock_openai_error_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429).set_body_json(serde_json::json!({
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                })),
            )
            .mount(&mock_server)
            .await;

        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::OpenAI)
                    .with_endpoint(mock_server.uri())
                    .with_api_key("test-key"),
            )
            .circuit_breaker(3, 2, Duration::from_secs(30))
            .build();

        // Simulate rate limit failures
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::OpenAI);

        let cb = router.circuit_breaker(LLMProvider::OpenAI).unwrap();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_mock_anthropic_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_anthropic_response()))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::Anthropic)
                    .with_endpoint(mock_server.uri())
                    .with_api_key("sk-ant-test"),
            )
            .build();

        let provider = router.select_provider().unwrap();
        assert_eq!(provider, LLMProvider::Anthropic);
    }

    #[tokio::test]
    async fn test_mock_ollama_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_ollama_response()))
            .mount(&mock_server)
            .await;

        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::Ollama)
                    .with_endpoint(mock_server.uri())
                    .with_default_model("llama3.2"),
            )
            .strategy(RoutingStrategy::Primary)
            .build();

        let provider = router.select_provider().unwrap();
        assert_eq!(provider, LLMProvider::Ollama);
    }

    #[tokio::test]
    async fn test_failover_with_mock() {
        let mock_openai = MockServer::start().await;
        let mock_anthropic = MockServer::start().await;

        // OpenAI returns 500
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_openai)
            .await;

        // Anthropic returns success
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_anthropic_response()))
            .mount(&mock_anthropic)
            .await;

        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::OpenAI)
                    .with_endpoint(mock_openai.uri())
                    .with_priority(0),
            )
            .provider(
                ProviderConfig::new(LLMProvider::Anthropic)
                    .with_endpoint(mock_anthropic.uri())
                    .with_priority(1),
            )
            .fallback_chain(vec![LLMProvider::OpenAI, LLMProvider::Anthropic])
            .circuit_breaker(1, 2, Duration::from_secs(30))
            .build();

        // First selection is OpenAI (priority 0)
        let first = router.select_provider().unwrap();
        assert_eq!(first, LLMProvider::OpenAI);

        // Record failure
        router.record_failure(LLMProvider::OpenAI);

        // Circuit should be open, fallback to Anthropic
        let fallback = router.select_fallback(LLMProvider::OpenAI);
        assert_eq!(fallback, Some(LLMProvider::Anthropic));

        // Select provider now should return Anthropic (OpenAI circuit is open)
        let selected = router.select_provider().unwrap();
        assert_eq!(selected, LLMProvider::Anthropic);
    }
}

// ============================================================================
// Router Statistics Tests
// ============================================================================

mod router_stats_tests {
    use super::*;

    #[tokio::test]
    async fn test_router_stats_initial() {
        let router = LLMRouter::new();
        let stats = router.stats();

        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.fallback_requests, 0);
        assert_eq!(stats.avg_latency_ms, 0.0);
        assert!(stats.provider_requests.is_empty());
    }

    #[tokio::test]
    async fn test_router_stats_aggregation() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .build();

        router.record_success(LLMProvider::OpenAI, 100);
        router.record_success(LLMProvider::OpenAI, 200);
        router.record_failure(LLMProvider::Anthropic);
        router.record_fallback();

        let stats = router.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.fallback_requests, 1);
        assert_eq!(stats.avg_latency_ms, 150.0); // (100 + 200) / 2
    }

    #[tokio::test]
    async fn test_router_stats_reset() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .build();

        router.record_success(LLMProvider::OpenAI, 100);
        router.record_failure(LLMProvider::OpenAI);
        router.record_fallback();

        router.reset_stats();

        let stats = router.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.fallback_requests, 0);
    }

    #[tokio::test]
    async fn test_provider_health_status_enum() {
        assert_eq!(
            ProviderHealthStatus::default(),
            ProviderHealthStatus::Healthy
        );
        assert_eq!(
            format!("{}", ProviderHealthStatus::Healthy),
            "healthy"
        );
        assert_eq!(
            format!("{}", ProviderHealthStatus::Degraded),
            "degraded"
        );
        assert_eq!(
            format!("{}", ProviderHealthStatus::Unhealthy),
            "unhealthy"
        );
        assert_eq!(
            format!("{}", ProviderHealthStatus::Unknown),
            "unknown"
        );
    }

    #[tokio::test]
    async fn test_router_stats_serde() {
        let stats = RouterStats {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            fallback_requests: 3,
            avg_latency_ms: 250.5,
            provider_requests: {
                let mut map = HashMap::new();
                map.insert("openai".to_string(), 60);
                map.insert("anthropic".to_string(), 40);
                map
            },
        };

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: RouterStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_requests, 100);
        assert_eq!(parsed.successful_requests, 95);
        assert!((parsed.avg_latency_ms - 250.5).abs() < 0.01);
    }
}

// ============================================================================
// Builder Pattern Tests
// ============================================================================

mod builder_tests {
    use super::*;

    #[tokio::test]
    async fn test_router_builder_empty() {
        let router = LLMRouter::builder().build();
        assert_eq!(router.provider_count(), 0);
        assert_eq!(router.strategy(), RoutingStrategy::Primary);
    }

    #[tokio::test]
    async fn test_router_builder_full_config() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_priority(0))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_priority(1))
            .provider(ProviderConfig::new(LLMProvider::Google).with_priority(2))
            .strategy(RoutingStrategy::CostOptimized)
            .fallback_chain(vec![
                LLMProvider::OpenAI,
                LLMProvider::Anthropic,
                LLMProvider::Google,
            ])
            .circuit_breaker(5, 3, Duration::from_secs(60))
            .build();

        assert_eq!(router.provider_count(), 3);
        assert_eq!(router.strategy(), RoutingStrategy::CostOptimized);
        assert_eq!(router.fallback_chain().len(), 3);
    }

    #[tokio::test]
    async fn test_provider_config_builder_chain() {
        let config = ProviderConfig::new(LLMProvider::OpenAI)
            .with_endpoint("https://api.custom.com")
            .with_api_key("sk-test")
            .with_default_model("gpt-4-turbo")
            .with_timeout(120)
            .with_max_retries(5)
            .with_priority(0)
            .with_enabled(true)
            .with_costs(0.01, 0.03)
            .with_quality_score(0.95)
            .with_latency(400);

        assert_eq!(config.endpoint, "https://api.custom.com");
        assert_eq!(config.api_key, Some("sk-test".to_string()));
        assert_eq!(config.default_model, "gpt-4-turbo");
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.priority, 0);
        assert!(config.enabled);
        assert_eq!(config.input_cost_per_1k, 0.01);
        assert_eq!(config.output_cost_per_1k, 0.03);
        assert_eq!(config.quality_score, 0.95);
        assert_eq!(config.avg_latency_ms, 400);
    }

    #[tokio::test]
    async fn test_provider_config_defaults_by_provider() {
        let openai = ProviderConfig::new(LLMProvider::OpenAI);
        assert!(openai.endpoint.contains("openai.com"));
        assert_eq!(openai.default_model, "gpt-4o");

        let anthropic = ProviderConfig::new(LLMProvider::Anthropic);
        assert!(anthropic.endpoint.contains("anthropic.com"));
        assert!(anthropic.default_model.contains("claude"));

        let google = ProviderConfig::new(LLMProvider::Google);
        assert!(google.endpoint.contains("googleapis.com"));
        assert!(google.default_model.contains("gemini"));

        let ollama = ProviderConfig::new(LLMProvider::Ollama);
        assert!(ollama.endpoint.contains("localhost:11434"));
        assert_eq!(ollama.input_cost_per_1k, 0.0);

        let custom = ProviderConfig::new(LLMProvider::Custom);
        assert!(custom.endpoint.contains("localhost:8080"));
    }
}

// ============================================================================
// Provider Management Tests
// ============================================================================

mod provider_management_tests {
    use super::*;

    #[tokio::test]
    async fn test_enable_disable_provider() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .build();

        // Disable OpenAI
        assert!(router.disable_provider(LLMProvider::OpenAI));
        let config = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert!(!config.enabled);

        // Re-enable
        assert!(router.enable_provider(LLMProvider::OpenAI));
        let config = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert!(config.enabled);

        // Non-existent provider
        assert!(!router.enable_provider(LLMProvider::Custom));
        assert!(!router.disable_provider(LLMProvider::Custom));
    }

    #[tokio::test]
    async fn test_update_provider() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .build();

        let updated = ProviderConfig::new(LLMProvider::OpenAI)
            .with_default_model("gpt-4-turbo")
            .with_priority(5)
            .with_costs(0.02, 0.06);

        assert!(router.update_provider(updated));

        let config = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert_eq!(config.default_model, "gpt-4-turbo");
        assert_eq!(config.priority, 5);
        assert_eq!(config.input_cost_per_1k, 0.02);

        // Non-existent provider
        let non_existent = ProviderConfig::new(LLMProvider::Custom);
        assert!(!router.update_provider(non_existent));
    }

    #[tokio::test]
    async fn test_list_providers() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .provider(ProviderConfig::new(LLMProvider::Google))
            .build();

        let providers = router.providers();
        assert_eq!(providers.len(), 3);

        let types = router.provider_types();
        assert_eq!(types.len(), 3);
        assert!(types.contains(&LLMProvider::OpenAI));
        assert!(types.contains(&LLMProvider::Anthropic));
        assert!(types.contains(&LLMProvider::Google));
    }

    #[tokio::test]
    async fn test_reset_circuit_breakers() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .circuit_breaker(2, 2, Duration::from_secs(30))
            .build();

        // Trip both circuits
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::Anthropic);
        router.record_failure(LLMProvider::Anthropic);

        // Both should be open
        assert_eq!(
            router.circuit_breaker(LLMProvider::OpenAI).unwrap().state(),
            CircuitState::Open
        );
        assert_eq!(
            router.circuit_breaker(LLMProvider::Anthropic).unwrap().state(),
            CircuitState::Open
        );

        // Reset all
        router.reset_circuit_breakers();

        assert_eq!(
            router.circuit_breaker(LLMProvider::OpenAI).unwrap().state(),
            CircuitState::Closed
        );
        assert_eq!(
            router.circuit_breaker(LLMProvider::Anthropic).unwrap().state(),
            CircuitState::Closed
        );
    }
}
