//! LLM Router for intelligent request routing across providers.
//!
//! This module provides a sophisticated routing layer for LLM requests with:
//!
//! - **Multi-Provider Support**: Route requests to OpenAI, Anthropic, Google, Ollama, or custom providers
//! - **Routing Strategies**: Primary, LoadBalance, CostOptimized, QualityOptimized, LatencyOptimized, Hybrid
//! - **Circuit Breaker**: Automatic failure detection and recovery
//! - **Cost Tracking**: Monitor and optimize spending across providers
//! - **Fallback Chains**: Automatic failover to backup providers
//!
//! # Architecture
//!
//! ```text
//! +------------------+
//! |    LLMRouter     |  <-- Selects appropriate provider
//! +------------------+
//!          |
//!    +-----+-----+-----+-----+-----+
//!    |     |     |     |     |     |
//!    v     v     v     v     v     v
//! +------+------+------+------+------+------+
//! |OpenAI|Claude|Gemini|Ollama|Custom|...   |
//! +------+------+------+------+------+------+
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::llm::router::{LLMRouter, LLMProvider, RoutingStrategy, ProviderConfig};
//!
//! let router = LLMRouter::builder()
//!     .provider(ProviderConfig::new(LLMProvider::OpenAI)
//!         .with_api_key("sk-...")
//!         .with_priority(0))
//!     .provider(ProviderConfig::new(LLMProvider::Anthropic)
//!         .with_api_key("sk-ant-...")
//!         .with_priority(1))
//!     .strategy(RoutingStrategy::CostOptimized)
//!     .fallback_chain(vec![LLMProvider::OpenAI, LLMProvider::Anthropic])
//!     .build();
//!
//! let provider = router.select_provider()?;
//! println!("Selected: {}", provider);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::error::{Error, Result};

// ============================================================================
// LLM PROVIDER ENUM
// ============================================================================

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LLMProvider {
    /// OpenAI GPT models.
    #[default]
    OpenAI,
    /// Anthropic Claude models.
    Anthropic,
    /// Google Gemini models.
    Google,
    /// Local Ollama instance.
    Ollama,
    /// Custom provider (user-defined).
    Custom,
}

impl LLMProvider {
    /// Returns the string representation of the provider.
    pub fn as_str(&self) -> &'static str {
        match self {
            LLMProvider::OpenAI => "openai",
            LLMProvider::Anthropic => "anthropic",
            LLMProvider::Google => "google",
            LLMProvider::Ollama => "ollama",
            LLMProvider::Custom => "custom",
        }
    }

    /// Get all available providers.
    pub fn all() -> &'static [LLMProvider] {
        &[
            LLMProvider::OpenAI,
            LLMProvider::Anthropic,
            LLMProvider::Google,
            LLMProvider::Ollama,
            LLMProvider::Custom,
        ]
    }
}

impl std::fmt::Display for LLMProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for LLMProvider {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" | "gpt" => Ok(LLMProvider::OpenAI),
            "anthropic" | "claude" => Ok(LLMProvider::Anthropic),
            "google" | "gemini" => Ok(LLMProvider::Google),
            "ollama" | "local" => Ok(LLMProvider::Ollama),
            "custom" => Ok(LLMProvider::Custom),
            _ => Err(Error::Config(format!("Unknown LLM provider: {}", s))),
        }
    }
}

// ============================================================================
// ROUTING STRATEGY
// ============================================================================

/// Strategy for routing requests to LLM providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Use a single primary provider.
    #[default]
    Primary,
    /// Round-robin load balancing across providers.
    LoadBalance,
    /// Route to minimize cost.
    CostOptimized,
    /// Route for best quality (largest/best models).
    QualityOptimized,
    /// Route for lowest latency.
    LatencyOptimized,
    /// Hybrid strategy with weighted distribution.
    Hybrid {
        /// Weight for cost optimization (0.0 - 1.0).
        cost_weight: f64,
        /// Weight for quality optimization (0.0 - 1.0).
        quality_weight: f64,
        /// Weight for latency optimization (0.0 - 1.0).
        latency_weight: f64,
    },
}

impl RoutingStrategy {
    /// Create a hybrid strategy with equal weights.
    pub fn hybrid_balanced() -> Self {
        Self::Hybrid {
            cost_weight: 0.33,
            quality_weight: 0.33,
            latency_weight: 0.34,
        }
    }

    /// Create a hybrid strategy with custom weights.
    pub fn hybrid(cost_weight: f64, quality_weight: f64, latency_weight: f64) -> Self {
        Self::Hybrid {
            cost_weight,
            quality_weight,
            latency_weight,
        }
    }

    /// Check if this is a hybrid strategy.
    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid { .. })
    }
}

// ============================================================================
// PROVIDER CONFIGURATION
// ============================================================================

/// Configuration for an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider identifier.
    pub provider: LLMProvider,
    /// API endpoint URL.
    pub endpoint: String,
    /// API key (if required).
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    /// Default model for this provider.
    pub default_model: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum retries for failed requests.
    pub max_retries: u32,
    /// Priority (lower = higher priority).
    pub priority: u32,
    /// Whether this provider is enabled.
    pub enabled: bool,
    /// Cost per 1K input tokens (USD).
    pub input_cost_per_1k: f64,
    /// Cost per 1K output tokens (USD).
    pub output_cost_per_1k: f64,
    /// Quality score (0.0 - 1.0).
    pub quality_score: f64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: u64,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::OpenAI,
            endpoint: "https://api.openai.com/v1".to_string(),
            api_key: None,
            default_model: "gpt-4o".to_string(),
            timeout_secs: 60,
            max_retries: 3,
            priority: 0,
            enabled: true,
            input_cost_per_1k: 0.005,
            output_cost_per_1k: 0.015,
            quality_score: 0.9,
            avg_latency_ms: 500,
        }
    }
}

impl ProviderConfig {
    /// Create a new provider configuration.
    pub fn new(provider: LLMProvider) -> Self {
        let (endpoint, default_model, input_cost, output_cost, quality_score, avg_latency) =
            match provider {
                LLMProvider::OpenAI => (
                    "https://api.openai.com/v1",
                    "gpt-4o",
                    0.005,
                    0.015,
                    0.9,
                    500,
                ),
                LLMProvider::Anthropic => (
                    "https://api.anthropic.com/v1",
                    "claude-sonnet-4-20250514",
                    0.003,
                    0.015,
                    0.92,
                    600,
                ),
                LLMProvider::Google => (
                    "https://generativelanguage.googleapis.com/v1beta",
                    "gemini-pro",
                    0.00025,
                    0.0005,
                    0.85,
                    400,
                ),
                LLMProvider::Ollama => ("http://localhost:11434", "llama3.2", 0.0, 0.0, 0.75, 200),
                LLMProvider::Custom => ("http://localhost:8080", "custom", 0.001, 0.002, 0.7, 300),
            };

        Self {
            provider,
            endpoint: endpoint.to_string(),
            default_model: default_model.to_string(),
            input_cost_per_1k: input_cost,
            output_cost_per_1k: output_cost,
            quality_score,
            avg_latency_ms: avg_latency,
            ..Default::default()
        }
    }

    /// Set the API endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Set the API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the max retries.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable or disable the provider.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set cost per 1K tokens.
    pub fn with_costs(mut self, input_cost: f64, output_cost: f64) -> Self {
        self.input_cost_per_1k = input_cost;
        self.output_cost_per_1k = output_cost;
        self
    }

    /// Set quality score.
    pub fn with_quality_score(mut self, score: f64) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set average latency.
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.avg_latency_ms = latency_ms;
        self
    }

    /// Calculate total cost for given tokens.
    pub fn calculate_cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = (input_tokens as f64 / 1000.0) * self.input_cost_per_1k;
        let output_cost = (output_tokens as f64 / 1000.0) * self.output_cost_per_1k;
        input_cost + output_cost
    }
}

// ============================================================================
// CIRCUIT BREAKER
// ============================================================================

/// State of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    /// Circuit is closed (normal operation).
    #[default]
    Closed,
    /// Circuit is open (blocking requests).
    Open,
    /// Circuit is half-open (testing recovery).
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "closed"),
            CircuitState::Open => write!(f, "open"),
            CircuitState::HalfOpen => write!(f, "half-open"),
        }
    }
}

/// Circuit breaker for provider failure handling.
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state.
    state: RwLock<CircuitState>,
    /// Failure count in current window.
    failure_count: AtomicU64,
    /// Success count since last failure.
    success_count: AtomicU64,
    /// Failure threshold to open circuit.
    failure_threshold: u64,
    /// Success threshold to close circuit from half-open.
    success_threshold: u64,
    /// Time when circuit was opened.
    opened_at: RwLock<Option<DateTime<Utc>>>,
    /// Duration to wait before half-opening.
    recovery_timeout: Duration,
    /// Total requests through this breaker.
    total_requests: AtomicU64,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, 2, Duration::from_secs(30))
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(failure_threshold: u64, success_threshold: u64, recovery_timeout: Duration) -> Self {
        Self {
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            failure_threshold,
            success_threshold,
            opened_at: RwLock::new(None),
            recovery_timeout,
            total_requests: AtomicU64::new(0),
        }
    }

    /// Get the current state.
    pub fn state(&self) -> CircuitState {
        *self.state.read().unwrap()
    }

    /// Check if requests should be allowed.
    pub fn should_allow(&self) -> bool {
        let mut state = self.state.write().unwrap();

        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has passed
                let opened_at = self.opened_at.read().unwrap();
                if let Some(opened) = *opened_at {
                    let elapsed = Utc::now().signed_duration_since(opened);
                    if elapsed.num_milliseconds() as u64 >= self.recovery_timeout.as_millis() as u64
                    {
                        *state = CircuitState::HalfOpen;
                        self.success_count.store(0, Ordering::SeqCst);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        let mut state = self.state.write().unwrap();

        match *state {
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count >= self.success_threshold {
                    *state = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::SeqCst);
                    *self.opened_at.write().unwrap() = None;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        let mut state = self.state.write().unwrap();

        match *state {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count >= self.failure_threshold {
                    *state = CircuitState::Open;
                    *self.opened_at.write().unwrap() = Some(Utc::now());
                }
            }
            CircuitState::HalfOpen => {
                *state = CircuitState::Open;
                *self.opened_at.write().unwrap() = Some(Utc::now());
                self.success_count.store(0, Ordering::SeqCst);
            }
            CircuitState::Open => {}
        }
    }

    /// Reset the circuit breaker.
    pub fn reset(&self) {
        *self.state.write().unwrap() = CircuitState::Closed;
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
        *self.opened_at.write().unwrap() = None;
    }

    /// Get failure count.
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::SeqCst)
    }

    /// Get success count (in half-open state).
    pub fn success_count(&self) -> u64 {
        self.success_count.load(Ordering::SeqCst)
    }

    /// Get total requests.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::SeqCst)
    }

    /// Get failure threshold.
    pub fn failure_threshold(&self) -> u64 {
        self.failure_threshold
    }

    /// Get success threshold.
    pub fn success_threshold(&self) -> u64 {
        self.success_threshold
    }

    /// Get recovery timeout.
    pub fn recovery_timeout(&self) -> Duration {
        self.recovery_timeout
    }
}

// ============================================================================
// COST TRACKER
// ============================================================================

/// Tracks costs across LLM providers.
#[derive(Debug, Default)]
pub struct CostTracker {
    /// Total input tokens across all providers.
    total_input_tokens: AtomicU64,
    /// Total output tokens across all providers.
    total_output_tokens: AtomicU64,
    /// Total cost in USD (stored as micro-dollars for precision).
    total_cost_micros: AtomicU64,
    /// Per-provider costs (provider name -> micro-dollars).
    provider_costs: RwLock<HashMap<String, u64>>,
    /// Per-provider token counts.
    provider_tokens: RwLock<HashMap<String, (u64, u64)>>,
    /// Request count.
    request_count: AtomicU64,
}

impl CostTracker {
    /// Create a new cost tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record token usage and cost.
    pub fn record(
        &self,
        provider: &str,
        input_tokens: u64,
        output_tokens: u64,
        input_cost_per_1k: f64,
        output_cost_per_1k: f64,
    ) {
        self.total_input_tokens
            .fetch_add(input_tokens, Ordering::SeqCst);
        self.total_output_tokens
            .fetch_add(output_tokens, Ordering::SeqCst);
        self.request_count.fetch_add(1, Ordering::SeqCst);

        // Calculate cost in micro-dollars (1 million = 1 USD)
        let input_cost = (input_tokens as f64 / 1000.0) * input_cost_per_1k;
        let output_cost = (output_tokens as f64 / 1000.0) * output_cost_per_1k;
        let total_cost = input_cost + output_cost;
        let cost_micros = (total_cost * 1_000_000.0) as u64;

        self.total_cost_micros
            .fetch_add(cost_micros, Ordering::SeqCst);

        // Update per-provider costs
        let mut costs = self.provider_costs.write().unwrap();
        *costs.entry(provider.to_string()).or_insert(0) += cost_micros;

        // Update per-provider tokens
        let mut tokens = self.provider_tokens.write().unwrap();
        let entry = tokens.entry(provider.to_string()).or_insert((0, 0));
        entry.0 += input_tokens;
        entry.1 += output_tokens;
    }

    /// Get total input tokens.
    pub fn total_input_tokens(&self) -> u64 {
        self.total_input_tokens.load(Ordering::SeqCst)
    }

    /// Get total output tokens.
    pub fn total_output_tokens(&self) -> u64 {
        self.total_output_tokens.load(Ordering::SeqCst)
    }

    /// Get total tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.total_input_tokens() + self.total_output_tokens()
    }

    /// Get total cost in USD.
    pub fn total_cost_usd(&self) -> f64 {
        self.total_cost_micros.load(Ordering::SeqCst) as f64 / 1_000_000.0
    }

    /// Get cost for a specific provider in USD.
    pub fn provider_cost_usd(&self, provider: &str) -> f64 {
        let costs = self.provider_costs.read().unwrap();
        costs.get(provider).copied().unwrap_or(0) as f64 / 1_000_000.0
    }

    /// Get tokens for a specific provider.
    pub fn provider_tokens(&self, provider: &str) -> (u64, u64) {
        let tokens = self.provider_tokens.read().unwrap();
        tokens.get(provider).copied().unwrap_or((0, 0))
    }

    /// Get request count.
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::SeqCst)
    }

    /// Get all provider costs.
    pub fn all_provider_costs_usd(&self) -> HashMap<String, f64> {
        let costs = self.provider_costs.read().unwrap();
        costs
            .iter()
            .map(|(k, v)| (k.clone(), *v as f64 / 1_000_000.0))
            .collect()
    }

    /// Reset all tracked costs.
    pub fn reset(&self) {
        self.total_input_tokens.store(0, Ordering::SeqCst);
        self.total_output_tokens.store(0, Ordering::SeqCst);
        self.total_cost_micros.store(0, Ordering::SeqCst);
        self.request_count.store(0, Ordering::SeqCst);
        self.provider_costs.write().unwrap().clear();
        self.provider_tokens.write().unwrap().clear();
    }

    /// Get a snapshot of current costs.
    pub fn snapshot(&self) -> CostSnapshot {
        CostSnapshot {
            total_input_tokens: self.total_input_tokens(),
            total_output_tokens: self.total_output_tokens(),
            total_cost_usd: self.total_cost_usd(),
            request_count: self.request_count(),
            provider_costs: self.all_provider_costs_usd(),
        }
    }
}

/// Snapshot of cost tracking data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSnapshot {
    /// Total input tokens.
    pub total_input_tokens: u64,
    /// Total output tokens.
    pub total_output_tokens: u64,
    /// Total cost in USD.
    pub total_cost_usd: f64,
    /// Total request count.
    pub request_count: u64,
    /// Per-provider costs in USD.
    pub provider_costs: HashMap<String, f64>,
}

// ============================================================================
// HEALTH STATUS
// ============================================================================

/// Health status of an LLM provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderHealthStatus {
    /// Provider is healthy and accepting requests.
    #[default]
    Healthy,
    /// Provider is degraded (slow or partial failures).
    Degraded,
    /// Provider is unhealthy (not accepting requests).
    Unhealthy,
    /// Health status is unknown (not yet checked).
    Unknown,
}

impl std::fmt::Display for ProviderHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderHealthStatus::Healthy => write!(f, "healthy"),
            ProviderHealthStatus::Degraded => write!(f, "degraded"),
            ProviderHealthStatus::Unhealthy => write!(f, "unhealthy"),
            ProviderHealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

// ============================================================================
// ROUTER STATISTICS
// ============================================================================

/// Router statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouterStats {
    /// Total requests routed.
    pub total_requests: u64,
    /// Successful requests.
    pub successful_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Fallback requests (primary failed, used fallback).
    pub fallback_requests: u64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Per-provider request counts.
    pub provider_requests: HashMap<String, u64>,
}

// ============================================================================
// LLM ROUTER
// ============================================================================

/// LLM Router for intelligent request routing across providers.
pub struct LLMRouter {
    /// Registered providers.
    providers: RwLock<HashMap<LLMProvider, ProviderConfig>>,
    /// Routing strategy.
    strategy: RwLock<RoutingStrategy>,
    /// Fallback chain (ordered by priority).
    fallback_chain: RwLock<Vec<LLMProvider>>,
    /// Circuit breakers per provider.
    circuit_breakers: RwLock<HashMap<LLMProvider, Arc<CircuitBreaker>>>,
    /// Cost tracker.
    cost_tracker: Arc<CostTracker>,
    /// Request counter.
    request_count: AtomicU64,
    /// Success counter.
    success_count: AtomicU64,
    /// Failure counter.
    failure_count: AtomicU64,
    /// Fallback counter.
    fallback_count: AtomicU64,
    /// Round-robin index for load balancing.
    round_robin_index: AtomicUsize,
    /// Total latency in milliseconds (for averaging).
    total_latency_ms: AtomicU64,
    /// Per-provider request counts.
    provider_request_counts: RwLock<HashMap<LLMProvider, u64>>,
}

impl Default for LLMRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMRouter {
    /// Create a new LLM router.
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            strategy: RwLock::new(RoutingStrategy::Primary),
            fallback_chain: RwLock::new(Vec::new()),
            circuit_breakers: RwLock::new(HashMap::new()),
            cost_tracker: Arc::new(CostTracker::new()),
            request_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
            fallback_count: AtomicU64::new(0),
            round_robin_index: AtomicUsize::new(0),
            total_latency_ms: AtomicU64::new(0),
            provider_request_counts: RwLock::new(HashMap::new()),
        }
    }

    /// Create a builder for LLMRouter.
    pub fn builder() -> LLMRouterBuilder {
        LLMRouterBuilder::new()
    }

    /// Register a provider.
    pub fn register_provider(&self, config: ProviderConfig) {
        let provider = config.provider;
        self.providers.write().unwrap().insert(provider, config);

        // Create circuit breaker for this provider
        self.circuit_breakers
            .write()
            .unwrap()
            .entry(provider)
            .or_insert_with(|| Arc::new(CircuitBreaker::default()));

        // Initialize request count
        self.provider_request_counts
            .write()
            .unwrap()
            .entry(provider)
            .or_insert(0);
    }

    /// Unregister a provider.
    pub fn unregister_provider(&self, provider: LLMProvider) -> bool {
        let removed = self.providers.write().unwrap().remove(&provider).is_some();
        self.circuit_breakers.write().unwrap().remove(&provider);
        self.provider_request_counts
            .write()
            .unwrap()
            .remove(&provider);
        removed
    }

    /// Check if a provider is registered.
    pub fn has_provider(&self, provider: LLMProvider) -> bool {
        self.providers.read().unwrap().contains_key(&provider)
    }

    /// Get provider count.
    pub fn provider_count(&self) -> usize {
        self.providers.read().unwrap().len()
    }

    /// Set the routing strategy.
    pub fn set_strategy(&self, strategy: RoutingStrategy) {
        *self.strategy.write().unwrap() = strategy;
    }

    /// Get the current routing strategy.
    pub fn strategy(&self) -> RoutingStrategy {
        self.strategy.read().unwrap().clone()
    }

    /// Set the fallback chain.
    pub fn set_fallback_chain(&self, chain: Vec<LLMProvider>) {
        *self.fallback_chain.write().unwrap() = chain;
    }

    /// Get the fallback chain.
    pub fn fallback_chain(&self) -> Vec<LLMProvider> {
        self.fallback_chain.read().unwrap().clone()
    }

    /// Get the cost tracker.
    pub fn cost_tracker(&self) -> Arc<CostTracker> {
        Arc::clone(&self.cost_tracker)
    }

    /// Get circuit breaker for a provider.
    pub fn circuit_breaker(&self, provider: LLMProvider) -> Option<Arc<CircuitBreaker>> {
        self.circuit_breakers
            .read()
            .unwrap()
            .get(&provider)
            .cloned()
    }

    /// Select the next provider based on routing strategy.
    pub fn select_provider(&self) -> Result<LLMProvider> {
        let providers = self.providers.read().unwrap();
        let strategy = self.strategy.read().unwrap();
        let breakers = self.circuit_breakers.read().unwrap();

        // Filter enabled providers with healthy circuit breakers
        let available: Vec<_> = providers
            .iter()
            .filter(|(_, cfg)| cfg.enabled)
            .filter(|(p, _)| breakers.get(p).map(|cb| cb.should_allow()).unwrap_or(true))
            .collect();

        if available.is_empty() {
            return Err(Error::Orchestration(
                "No available LLM providers".to_string(),
            ));
        }

        match &*strategy {
            RoutingStrategy::Primary => {
                // Select by priority (lowest first)
                available
                    .iter()
                    .min_by_key(|(_, cfg)| cfg.priority)
                    .map(|(p, _)| **p)
                    .ok_or_else(|| Error::Orchestration("No primary provider".to_string()))
            }
            RoutingStrategy::LoadBalance => {
                // Round-robin selection
                let index = self.round_robin_index.fetch_add(1, Ordering::SeqCst) % available.len();
                Ok(*available[index].0)
            }
            RoutingStrategy::CostOptimized => {
                // Select cheapest provider
                available
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        let cost_a = a.input_cost_per_1k + a.output_cost_per_1k;
                        let cost_b = b.input_cost_per_1k + b.output_cost_per_1k;
                        cost_a
                            .partial_cmp(&cost_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(p, _)| **p)
                    .ok_or_else(|| Error::Orchestration("No cost-optimized provider".to_string()))
            }
            RoutingStrategy::QualityOptimized => {
                // Select highest quality provider
                available
                    .iter()
                    .max_by(|(_, a), (_, b)| {
                        a.quality_score
                            .partial_cmp(&b.quality_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(p, _)| **p)
                    .ok_or_else(|| {
                        Error::Orchestration("No quality-optimized provider".to_string())
                    })
            }
            RoutingStrategy::LatencyOptimized => {
                // Select lowest latency provider
                available
                    .iter()
                    .min_by_key(|(_, cfg)| cfg.avg_latency_ms)
                    .map(|(p, _)| **p)
                    .ok_or_else(|| {
                        Error::Orchestration("No latency-optimized provider".to_string())
                    })
            }
            RoutingStrategy::Hybrid {
                cost_weight,
                quality_weight,
                latency_weight,
            } => {
                // Calculate weighted scores
                let scored: Vec<_> = available
                    .iter()
                    .map(|(p, cfg)| {
                        // Normalize scores (lower is better for cost/latency, higher for quality)
                        let cost_score =
                            1.0 - (cfg.input_cost_per_1k + cfg.output_cost_per_1k).min(1.0);
                        let quality_score = cfg.quality_score;
                        let latency_score = 1.0 - (cfg.avg_latency_ms as f64 / 1000.0).min(1.0);

                        let total_score = cost_weight * cost_score
                            + quality_weight * quality_score
                            + latency_weight * latency_score;

                        (**p, total_score)
                    })
                    .collect();

                scored
                    .iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(p, _)| *p)
                    .ok_or_else(|| Error::Orchestration("No hybrid-optimized provider".to_string()))
            }
        }
    }

    /// Select next provider from fallback chain.
    pub fn select_fallback(&self, failed_provider: LLMProvider) -> Option<LLMProvider> {
        let chain = self.fallback_chain.read().unwrap();
        let breakers = self.circuit_breakers.read().unwrap();
        let providers = self.providers.read().unwrap();

        // Find the next provider in the chain after the failed one
        let mut found_failed = false;
        for provider in chain.iter() {
            if *provider == failed_provider {
                found_failed = true;
                continue;
            }

            if found_failed {
                // Check if this provider is available
                if let Some(config) = providers.get(provider) {
                    if config.enabled {
                        if let Some(cb) = breakers.get(provider) {
                            if cb.should_allow() {
                                return Some(*provider);
                            }
                        } else {
                            return Some(*provider);
                        }
                    }
                }
            }
        }

        None
    }

    /// Get a provider configuration.
    pub fn get_provider(&self, provider: LLMProvider) -> Option<ProviderConfig> {
        self.providers.read().unwrap().get(&provider).cloned()
    }

    /// Get all registered providers.
    pub fn providers(&self) -> Vec<ProviderConfig> {
        self.providers.read().unwrap().values().cloned().collect()
    }

    /// Get all registered provider types.
    pub fn provider_types(&self) -> Vec<LLMProvider> {
        self.providers.read().unwrap().keys().copied().collect()
    }

    /// Record a successful request.
    pub fn record_success(&self, provider: LLMProvider, latency_ms: u64) {
        self.request_count.fetch_add(1, Ordering::SeqCst);
        self.success_count.fetch_add(1, Ordering::SeqCst);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::SeqCst);

        // Update provider request count
        let mut counts = self.provider_request_counts.write().unwrap();
        *counts.entry(provider).or_insert(0) += 1;

        if let Some(cb) = self.circuit_breakers.read().unwrap().get(&provider) {
            cb.record_success();
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self, provider: LLMProvider) {
        self.request_count.fetch_add(1, Ordering::SeqCst);
        self.failure_count.fetch_add(1, Ordering::SeqCst);

        // Update provider request count
        let mut counts = self.provider_request_counts.write().unwrap();
        *counts.entry(provider).or_insert(0) += 1;

        if let Some(cb) = self.circuit_breakers.read().unwrap().get(&provider) {
            cb.record_failure();
        }
    }

    /// Record a fallback request.
    pub fn record_fallback(&self) {
        self.fallback_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Get router statistics.
    pub fn stats(&self) -> RouterStats {
        let total = self.request_count.load(Ordering::SeqCst);
        let total_latency = self.total_latency_ms.load(Ordering::SeqCst);
        let success = self.success_count.load(Ordering::SeqCst);

        let avg_latency = if success > 0 {
            total_latency as f64 / success as f64
        } else {
            0.0
        };

        let provider_requests: HashMap<String, u64> = self
            .provider_request_counts
            .read()
            .unwrap()
            .iter()
            .map(|(p, count)| (p.to_string(), *count))
            .collect();

        RouterStats {
            total_requests: total,
            successful_requests: success,
            failed_requests: self.failure_count.load(Ordering::SeqCst),
            fallback_requests: self.fallback_count.load(Ordering::SeqCst),
            avg_latency_ms: avg_latency,
            provider_requests,
        }
    }

    /// Reset all statistics.
    pub fn reset_stats(&self) {
        self.request_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
        self.failure_count.store(0, Ordering::SeqCst);
        self.fallback_count.store(0, Ordering::SeqCst);
        self.total_latency_ms.store(0, Ordering::SeqCst);
        self.round_robin_index.store(0, Ordering::SeqCst);
        self.provider_request_counts.write().unwrap().clear();
        self.cost_tracker.reset();
    }

    /// Reset circuit breakers for all providers.
    pub fn reset_circuit_breakers(&self) {
        for cb in self.circuit_breakers.read().unwrap().values() {
            cb.reset();
        }
    }

    /// Update provider configuration.
    pub fn update_provider(&self, config: ProviderConfig) -> bool {
        let provider = config.provider;
        let mut providers = self.providers.write().unwrap();
        if providers.contains_key(&provider) {
            providers.insert(provider, config);
            true
        } else {
            false
        }
    }

    /// Enable a provider.
    pub fn enable_provider(&self, provider: LLMProvider) -> bool {
        let mut providers = self.providers.write().unwrap();
        if let Some(config) = providers.get_mut(&provider) {
            config.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a provider.
    pub fn disable_provider(&self, provider: LLMProvider) -> bool {
        let mut providers = self.providers.write().unwrap();
        if let Some(config) = providers.get_mut(&provider) {
            config.enabled = false;
            true
        } else {
            false
        }
    }
}

// ============================================================================
// LLM ROUTER BUILDER
// ============================================================================

/// Builder for LLMRouter.
#[derive(Default)]
pub struct LLMRouterBuilder {
    providers: Vec<ProviderConfig>,
    strategy: RoutingStrategy,
    fallback_chain: Vec<LLMProvider>,
    circuit_breaker_config: Option<(u64, u64, Duration)>,
}

impl LLMRouterBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a provider configuration.
    pub fn provider(mut self, config: ProviderConfig) -> Self {
        self.providers.push(config);
        self
    }

    /// Add multiple provider configurations.
    pub fn providers(mut self, configs: Vec<ProviderConfig>) -> Self {
        self.providers.extend(configs);
        self
    }

    /// Set the routing strategy.
    pub fn strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the fallback chain.
    pub fn fallback_chain(mut self, chain: Vec<LLMProvider>) -> Self {
        self.fallback_chain = chain;
        self
    }

    /// Configure circuit breakers.
    pub fn circuit_breaker(
        mut self,
        failure_threshold: u64,
        success_threshold: u64,
        recovery_timeout: Duration,
    ) -> Self {
        self.circuit_breaker_config =
            Some((failure_threshold, success_threshold, recovery_timeout));
        self
    }

    /// Build the router.
    pub fn build(self) -> LLMRouter {
        let router = LLMRouter::new();

        // Set strategy
        router.set_strategy(self.strategy);

        // Set fallback chain
        if !self.fallback_chain.is_empty() {
            router.set_fallback_chain(self.fallback_chain);
        }

        // Register providers
        for config in self.providers {
            let provider = config.provider;
            router.register_provider(config);

            // Configure circuit breaker if specified
            if let Some((failure_threshold, success_threshold, recovery_timeout)) =
                self.circuit_breaker_config
            {
                let cb =
                    CircuitBreaker::new(failure_threshold, success_threshold, recovery_timeout);
                router
                    .circuit_breakers
                    .write()
                    .unwrap()
                    .insert(provider, Arc::new(cb));
            }
        }

        router
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // LLMProvider Tests
    // ========================================================================

    #[test]
    fn test_llm_provider_as_str() {
        assert_eq!(LLMProvider::OpenAI.as_str(), "openai");
        assert_eq!(LLMProvider::Anthropic.as_str(), "anthropic");
        assert_eq!(LLMProvider::Google.as_str(), "google");
        assert_eq!(LLMProvider::Ollama.as_str(), "ollama");
        assert_eq!(LLMProvider::Custom.as_str(), "custom");
    }

    #[test]
    fn test_llm_provider_display() {
        assert_eq!(format!("{}", LLMProvider::OpenAI), "openai");
        assert_eq!(format!("{}", LLMProvider::Anthropic), "anthropic");
    }

    #[test]
    fn test_llm_provider_from_str() {
        assert_eq!(
            "openai".parse::<LLMProvider>().unwrap(),
            LLMProvider::OpenAI
        );
        assert_eq!("gpt".parse::<LLMProvider>().unwrap(), LLMProvider::OpenAI);
        assert_eq!(
            "anthropic".parse::<LLMProvider>().unwrap(),
            LLMProvider::Anthropic
        );
        assert_eq!(
            "claude".parse::<LLMProvider>().unwrap(),
            LLMProvider::Anthropic
        );
        assert_eq!(
            "google".parse::<LLMProvider>().unwrap(),
            LLMProvider::Google
        );
        assert_eq!(
            "gemini".parse::<LLMProvider>().unwrap(),
            LLMProvider::Google
        );
        assert_eq!(
            "ollama".parse::<LLMProvider>().unwrap(),
            LLMProvider::Ollama
        );
        assert_eq!("local".parse::<LLMProvider>().unwrap(), LLMProvider::Ollama);
        assert!("unknown".parse::<LLMProvider>().is_err());
    }

    #[test]
    fn test_llm_provider_all() {
        let all = LLMProvider::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&LLMProvider::OpenAI));
        assert!(all.contains(&LLMProvider::Anthropic));
        assert!(all.contains(&LLMProvider::Google));
        assert!(all.contains(&LLMProvider::Ollama));
        assert!(all.contains(&LLMProvider::Custom));
    }

    #[test]
    fn test_llm_provider_default() {
        assert_eq!(LLMProvider::default(), LLMProvider::OpenAI);
    }

    #[test]
    fn test_llm_provider_serde() {
        let provider = LLMProvider::Anthropic;
        let json = serde_json::to_string(&provider).unwrap();
        assert_eq!(json, "\"anthropic\"");

        let parsed: LLMProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, LLMProvider::Anthropic);
    }

    // ========================================================================
    // RoutingStrategy Tests
    // ========================================================================

    #[test]
    fn test_routing_strategy_default() {
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::Primary);
    }

    #[test]
    fn test_routing_strategy_hybrid_balanced() {
        let strategy = RoutingStrategy::hybrid_balanced();
        if let RoutingStrategy::Hybrid {
            cost_weight,
            quality_weight,
            latency_weight,
        } = strategy
        {
            assert!((cost_weight - 0.33).abs() < 0.01);
            assert!((quality_weight - 0.33).abs() < 0.01);
            assert!((latency_weight - 0.34).abs() < 0.01);
        } else {
            panic!("Expected Hybrid strategy");
        }
    }

    #[test]
    fn test_routing_strategy_hybrid_custom() {
        let strategy = RoutingStrategy::hybrid(0.5, 0.3, 0.2);
        if let RoutingStrategy::Hybrid {
            cost_weight,
            quality_weight,
            latency_weight,
        } = strategy
        {
            assert_eq!(cost_weight, 0.5);
            assert_eq!(quality_weight, 0.3);
            assert_eq!(latency_weight, 0.2);
        } else {
            panic!("Expected Hybrid strategy");
        }
    }

    #[test]
    fn test_routing_strategy_is_hybrid() {
        assert!(!RoutingStrategy::Primary.is_hybrid());
        assert!(!RoutingStrategy::LoadBalance.is_hybrid());
        assert!(RoutingStrategy::hybrid_balanced().is_hybrid());
    }

    #[test]
    fn test_routing_strategy_serde() {
        let strategy = RoutingStrategy::CostOptimized;
        let json = serde_json::to_string(&strategy).unwrap();
        assert_eq!(json, "\"cost_optimized\"");

        let parsed: RoutingStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, RoutingStrategy::CostOptimized);
    }

    // ========================================================================
    // ProviderConfig Tests
    // ========================================================================

    #[test]
    fn test_provider_config_new() {
        let config = ProviderConfig::new(LLMProvider::OpenAI);
        assert_eq!(config.provider, LLMProvider::OpenAI);
        assert_eq!(config.endpoint, "https://api.openai.com/v1");
        assert_eq!(config.default_model, "gpt-4o");
        assert!(config.enabled);
    }

    #[test]
    fn test_provider_config_anthropic() {
        let config = ProviderConfig::new(LLMProvider::Anthropic);
        assert_eq!(config.provider, LLMProvider::Anthropic);
        assert!(config.endpoint.contains("anthropic"));
        assert!(config.default_model.contains("claude"));
    }

    #[test]
    fn test_provider_config_ollama() {
        let config = ProviderConfig::new(LLMProvider::Ollama);
        assert_eq!(config.provider, LLMProvider::Ollama);
        assert!(config.endpoint.contains("localhost:11434"));
        assert_eq!(config.input_cost_per_1k, 0.0);
        assert_eq!(config.output_cost_per_1k, 0.0);
    }

    #[test]
    fn test_provider_config_builder_pattern() {
        let config = ProviderConfig::new(LLMProvider::OpenAI)
            .with_api_key("sk-test")
            .with_endpoint("https://custom.api.com")
            .with_default_model("gpt-4-turbo")
            .with_timeout(120)
            .with_max_retries(5)
            .with_priority(1)
            .with_enabled(false)
            .with_costs(0.01, 0.03)
            .with_quality_score(0.95)
            .with_latency(300);

        assert_eq!(config.api_key, Some("sk-test".to_string()));
        assert_eq!(config.endpoint, "https://custom.api.com");
        assert_eq!(config.default_model, "gpt-4-turbo");
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.priority, 1);
        assert!(!config.enabled);
        assert_eq!(config.input_cost_per_1k, 0.01);
        assert_eq!(config.output_cost_per_1k, 0.03);
        assert_eq!(config.quality_score, 0.95);
        assert_eq!(config.avg_latency_ms, 300);
    }

    #[test]
    fn test_provider_config_quality_score_clamp() {
        let config = ProviderConfig::new(LLMProvider::OpenAI).with_quality_score(1.5);
        assert_eq!(config.quality_score, 1.0);

        let config = ProviderConfig::new(LLMProvider::OpenAI).with_quality_score(-0.5);
        assert_eq!(config.quality_score, 0.0);
    }

    #[test]
    fn test_provider_config_calculate_cost() {
        let config = ProviderConfig::new(LLMProvider::OpenAI).with_costs(0.01, 0.03); // $0.01/1K input, $0.03/1K output

        let cost = config.calculate_cost(1000, 500);
        assert!((cost - 0.025).abs() < 0.0001); // 1K * 0.01 + 0.5K * 0.03 = 0.01 + 0.015 = 0.025
    }

    // ========================================================================
    // CircuitBreaker Tests
    // ========================================================================

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 0);
        assert!(cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new(3, 2, Duration::from_secs(30));

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_success_resets_failure_count() {
        let cb = CircuitBreaker::new(3, 2, Duration::from_secs(30));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);

        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(3, 2, Duration::from_secs(30));

        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert!(cb.should_allow());
    }

    #[test]
    fn test_circuit_state_display() {
        assert_eq!(format!("{}", CircuitState::Closed), "closed");
        assert_eq!(format!("{}", CircuitState::Open), "open");
        assert_eq!(format!("{}", CircuitState::HalfOpen), "half-open");
    }

    #[test]
    fn test_circuit_breaker_getters() {
        let cb = CircuitBreaker::new(5, 3, Duration::from_secs(60));
        assert_eq!(cb.failure_threshold(), 5);
        assert_eq!(cb.success_threshold(), 3);
        assert_eq!(cb.recovery_timeout(), Duration::from_secs(60));
    }

    // ========================================================================
    // CostTracker Tests
    // ========================================================================

    #[test]
    fn test_cost_tracker_initial() {
        let tracker = CostTracker::new();
        assert_eq!(tracker.total_input_tokens(), 0);
        assert_eq!(tracker.total_output_tokens(), 0);
        assert_eq!(tracker.total_tokens(), 0);
        assert_eq!(tracker.total_cost_usd(), 0.0);
        assert_eq!(tracker.request_count(), 0);
    }

    #[test]
    fn test_cost_tracker_record() {
        let tracker = CostTracker::new();

        // Record: 1000 input tokens, 500 output tokens
        // Cost: $0.01/1K input, $0.03/1K output
        tracker.record("openai", 1000, 500, 0.01, 0.03);

        assert_eq!(tracker.total_input_tokens(), 1000);
        assert_eq!(tracker.total_output_tokens(), 500);
        assert_eq!(tracker.total_tokens(), 1500);
        assert_eq!(tracker.request_count(), 1);

        // Expected cost: 1.0 * 0.01 + 0.5 * 0.03 = 0.01 + 0.015 = 0.025
        assert!((tracker.total_cost_usd() - 0.025).abs() < 0.0001);
    }

    #[test]
    fn test_cost_tracker_multiple_providers() {
        let tracker = CostTracker::new();

        tracker.record("openai", 1000, 500, 0.01, 0.03);
        tracker.record("anthropic", 2000, 1000, 0.003, 0.015);

        assert_eq!(tracker.request_count(), 2);
        assert_eq!(tracker.total_input_tokens(), 3000);
        assert_eq!(tracker.total_output_tokens(), 1500);

        // OpenAI cost: 0.01 + 0.015 = 0.025
        // Anthropic cost: 0.006 + 0.015 = 0.021
        let openai_cost = tracker.provider_cost_usd("openai");
        let anthropic_cost = tracker.provider_cost_usd("anthropic");
        assert!((openai_cost - 0.025).abs() < 0.0001);
        assert!((anthropic_cost - 0.021).abs() < 0.0001);
    }

    #[test]
    fn test_cost_tracker_provider_tokens() {
        let tracker = CostTracker::new();
        tracker.record("openai", 1000, 500, 0.01, 0.03);

        let (input, output) = tracker.provider_tokens("openai");
        assert_eq!(input, 1000);
        assert_eq!(output, 500);

        let (input, output) = tracker.provider_tokens("unknown");
        assert_eq!(input, 0);
        assert_eq!(output, 0);
    }

    #[test]
    fn test_cost_tracker_reset() {
        let tracker = CostTracker::new();
        tracker.record("openai", 1000, 500, 0.01, 0.03);

        tracker.reset();

        assert_eq!(tracker.total_input_tokens(), 0);
        assert_eq!(tracker.total_output_tokens(), 0);
        assert_eq!(tracker.total_cost_usd(), 0.0);
        assert_eq!(tracker.request_count(), 0);
        assert!(tracker.all_provider_costs_usd().is_empty());
    }

    #[test]
    fn test_cost_tracker_snapshot() {
        let tracker = CostTracker::new();
        tracker.record("openai", 1000, 500, 0.01, 0.03);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.total_input_tokens, 1000);
        assert_eq!(snapshot.total_output_tokens, 500);
        assert_eq!(snapshot.request_count, 1);
        assert!(snapshot.provider_costs.contains_key("openai"));
    }

    // ========================================================================
    // ProviderHealthStatus Tests
    // ========================================================================

    #[test]
    fn test_provider_health_status_display() {
        assert_eq!(format!("{}", ProviderHealthStatus::Healthy), "healthy");
        assert_eq!(format!("{}", ProviderHealthStatus::Degraded), "degraded");
        assert_eq!(format!("{}", ProviderHealthStatus::Unhealthy), "unhealthy");
        assert_eq!(format!("{}", ProviderHealthStatus::Unknown), "unknown");
    }

    #[test]
    fn test_provider_health_status_default() {
        assert_eq!(
            ProviderHealthStatus::default(),
            ProviderHealthStatus::Healthy
        );
    }

    // ========================================================================
    // LLMRouter Tests
    // ========================================================================

    #[test]
    fn test_router_new() {
        let router = LLMRouter::new();
        assert_eq!(router.provider_count(), 0);
        assert_eq!(router.strategy(), RoutingStrategy::Primary);
        assert!(router.fallback_chain().is_empty());
    }

    #[test]
    fn test_router_register_provider() {
        let router = LLMRouter::new();
        let config = ProviderConfig::new(LLMProvider::OpenAI);

        router.register_provider(config);

        assert_eq!(router.provider_count(), 1);
        assert!(router.has_provider(LLMProvider::OpenAI));
        assert!(!router.has_provider(LLMProvider::Anthropic));
    }

    #[test]
    fn test_router_unregister_provider() {
        let router = LLMRouter::new();
        router.register_provider(ProviderConfig::new(LLMProvider::OpenAI));

        assert!(router.unregister_provider(LLMProvider::OpenAI));
        assert!(!router.has_provider(LLMProvider::OpenAI));
        assert!(!router.unregister_provider(LLMProvider::OpenAI)); // Already removed
    }

    #[test]
    fn test_router_select_primary() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_priority(1))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_priority(0))
            .strategy(RoutingStrategy::Primary)
            .build();

        let selected = router.select_provider().unwrap();
        assert_eq!(selected, LLMProvider::Anthropic); // Lower priority = selected
    }

    #[test]
    fn test_router_select_load_balance() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .strategy(RoutingStrategy::LoadBalance)
            .build();

        let first = router.select_provider().unwrap();
        let second = router.select_provider().unwrap();

        // Should alternate between providers
        assert_ne!(first, second);
    }

    #[test]
    fn test_router_select_cost_optimized() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_costs(0.01, 0.03))
            .provider(ProviderConfig::new(LLMProvider::Google).with_costs(0.001, 0.002))
            .strategy(RoutingStrategy::CostOptimized)
            .build();

        let selected = router.select_provider().unwrap();
        assert_eq!(selected, LLMProvider::Google); // Cheapest
    }

    #[test]
    fn test_router_select_quality_optimized() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_quality_score(0.9))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_quality_score(0.95))
            .strategy(RoutingStrategy::QualityOptimized)
            .build();

        let selected = router.select_provider().unwrap();
        assert_eq!(selected, LLMProvider::Anthropic); // Highest quality
    }

    #[test]
    fn test_router_select_latency_optimized() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_latency(500))
            .provider(ProviderConfig::new(LLMProvider::Ollama).with_latency(100))
            .strategy(RoutingStrategy::LatencyOptimized)
            .build();

        let selected = router.select_provider().unwrap();
        assert_eq!(selected, LLMProvider::Ollama); // Lowest latency
    }

    #[test]
    fn test_router_select_no_providers() {
        let router = LLMRouter::new();
        let result = router.select_provider();
        assert!(result.is_err());
    }

    #[test]
    fn test_router_select_all_disabled() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_enabled(false))
            .build();

        let result = router.select_provider();
        assert!(result.is_err());
    }

    #[test]
    fn test_router_fallback_chain() {
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

        // Simulate Google failure (end of chain)
        let fallback = router.select_fallback(LLMProvider::Google);
        assert_eq!(fallback, None);
    }

    #[test]
    fn test_router_record_success() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .build();

        router.record_success(LLMProvider::OpenAI, 100);
        router.record_success(LLMProvider::OpenAI, 200);

        let stats = router.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.avg_latency_ms, 150.0);
    }

    #[test]
    fn test_router_record_failure() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .build();

        router.record_failure(LLMProvider::OpenAI);

        let stats = router.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 1);
    }

    #[test]
    fn test_router_record_fallback() {
        let router = LLMRouter::new();
        router.record_fallback();
        router.record_fallback();

        let stats = router.stats();
        assert_eq!(stats.fallback_requests, 2);
    }

    #[test]
    fn test_router_reset_stats() {
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

    #[test]
    fn test_router_circuit_breaker_integration() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .circuit_breaker(3, 2, Duration::from_secs(30))
            .build();

        // Record failures to trip circuit breaker
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::OpenAI);
        router.record_failure(LLMProvider::OpenAI);

        // OpenAI circuit should be open
        let cb = router.circuit_breaker(LLMProvider::OpenAI).unwrap();
        assert_eq!(cb.state(), CircuitState::Open);

        // Anthropic should still be available
        let cb = router.circuit_breaker(LLMProvider::Anthropic).unwrap();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_router_enable_disable_provider() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .build();

        assert!(router.disable_provider(LLMProvider::OpenAI));
        let config = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert!(!config.enabled);

        assert!(router.enable_provider(LLMProvider::OpenAI));
        let config = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert!(config.enabled);

        // Non-existent provider
        assert!(!router.enable_provider(LLMProvider::Anthropic));
    }

    #[test]
    fn test_router_update_provider() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .build();

        let updated = ProviderConfig::new(LLMProvider::OpenAI)
            .with_default_model("gpt-4-turbo")
            .with_priority(5);

        assert!(router.update_provider(updated));

        let config = router.get_provider(LLMProvider::OpenAI).unwrap();
        assert_eq!(config.default_model, "gpt-4-turbo");
        assert_eq!(config.priority, 5);

        // Non-existent provider
        let new_config = ProviderConfig::new(LLMProvider::Anthropic);
        assert!(!router.update_provider(new_config));
    }

    #[test]
    fn test_router_providers_list() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .build();

        let providers = router.providers();
        assert_eq!(providers.len(), 2);

        let types = router.provider_types();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&LLMProvider::OpenAI));
        assert!(types.contains(&LLMProvider::Anthropic));
    }

    #[test]
    fn test_router_cost_tracker() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .build();

        let tracker = router.cost_tracker();
        tracker.record("openai", 1000, 500, 0.01, 0.03);

        assert_eq!(tracker.request_count(), 1);
        assert!(tracker.total_cost_usd() > 0.0);
    }

    // ========================================================================
    // LLMRouterBuilder Tests
    // ========================================================================

    #[test]
    fn test_router_builder_new() {
        let builder = LLMRouterBuilder::new();
        let router = builder.build();
        assert_eq!(router.provider_count(), 0);
    }

    #[test]
    fn test_router_builder_providers() {
        let router = LLMRouterBuilder::new()
            .providers(vec![
                ProviderConfig::new(LLMProvider::OpenAI),
                ProviderConfig::new(LLMProvider::Anthropic),
            ])
            .build();

        assert_eq!(router.provider_count(), 2);
    }

    #[test]
    fn test_router_builder_full() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI).with_priority(0))
            .provider(ProviderConfig::new(LLMProvider::Anthropic).with_priority(1))
            .strategy(RoutingStrategy::LoadBalance)
            .fallback_chain(vec![LLMProvider::OpenAI, LLMProvider::Anthropic])
            .circuit_breaker(5, 3, Duration::from_secs(60))
            .build();

        assert_eq!(router.provider_count(), 2);
        assert_eq!(router.strategy(), RoutingStrategy::LoadBalance);
        assert_eq!(router.fallback_chain().len(), 2);
    }

    // ========================================================================
    // Hybrid Strategy Tests
    // ========================================================================

    #[test]
    fn test_router_hybrid_strategy() {
        let router = LLMRouter::builder()
            .provider(
                ProviderConfig::new(LLMProvider::OpenAI)
                    .with_costs(0.01, 0.03)
                    .with_quality_score(0.9)
                    .with_latency(500),
            )
            .provider(
                ProviderConfig::new(LLMProvider::Google)
                    .with_costs(0.001, 0.002)
                    .with_quality_score(0.8)
                    .with_latency(400),
            )
            .strategy(RoutingStrategy::hybrid(0.7, 0.2, 0.1)) // Heavily cost-weighted
            .build();

        let selected = router.select_provider().unwrap();
        assert_eq!(selected, LLMProvider::Google); // Should favor cheaper provider
    }

    // ========================================================================
    // Provider Request Count Tests
    // ========================================================================

    #[test]
    fn test_router_provider_request_counts() {
        let router = LLMRouter::builder()
            .provider(ProviderConfig::new(LLMProvider::OpenAI))
            .provider(ProviderConfig::new(LLMProvider::Anthropic))
            .build();

        router.record_success(LLMProvider::OpenAI, 100);
        router.record_success(LLMProvider::OpenAI, 100);
        router.record_success(LLMProvider::Anthropic, 100);

        let stats = router.stats();
        assert_eq!(stats.provider_requests.get("openai"), Some(&2));
        assert_eq!(stats.provider_requests.get("anthropic"), Some(&1));
    }

    // ========================================================================
    // RouterStats Tests
    // ========================================================================

    #[test]
    fn test_router_stats_default() {
        let stats = RouterStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.fallback_requests, 0);
        assert_eq!(stats.avg_latency_ms, 0.0);
        assert!(stats.provider_requests.is_empty());
    }

    #[test]
    fn test_router_stats_serde() {
        let stats = RouterStats {
            total_requests: 100,
            successful_requests: 90,
            failed_requests: 10,
            fallback_requests: 5,
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
        assert_eq!(parsed.successful_requests, 90);
        assert!((parsed.avg_latency_ms - 250.5).abs() < 0.01);
    }
}
