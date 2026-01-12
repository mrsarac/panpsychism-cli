//! OpenAI/GPT API client module for Project Panpsychism.
//!
//! Provides integration with OpenAI's GPT models for prompt synthesis and completion.
//!
//! # Features
//!
//! - **Multiple Models**: Support for GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4, GPT-3.5-turbo
//! - **Retry Logic**: Exponential backoff with configurable max retries
//! - **Rate Limiting**: Configurable TPM (tokens per minute) and RPM (requests per minute)
//! - **Streaming**: Token-by-token response streaming via SSE
//! - **Statistics**: Usage tracking for requests and tokens
//! - **Token Estimation**: Approximate token counting for cost estimation
//! - **Error Categorization**: Detailed error types for proper handling
//!
//! # Environment Variables
//!
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: Override API endpoint (default: https://api.openai.com/v1)
//! - `OPENAI_ORGANIZATION`: Optional organization ID for API requests
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::llm::openai::{OpenAIClient, OpenAIModel};
//!
//! let client = OpenAIClient::builder()
//!     .api_key("sk-...")
//!     .model(OpenAIModel::Gpt4o)
//!     .build()?;
//!
//! let response = client.complete_with_retry("Explain quantum computing").await?;
//! println!("Response: {}", response);
//! ```

use crate::{Error, Result};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::env;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Default OpenAI API endpoint.
pub const DEFAULT_OPENAI_ENDPOINT: &str = "https://api.openai.com/v1";

/// Default request timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Default maximum retry attempts.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default requests per minute limit.
pub const DEFAULT_REQUESTS_PER_MINUTE: u32 = 60;

/// Default tokens per minute limit (for GPT-4o).
pub const DEFAULT_TOKENS_PER_MINUTE: u32 = 30_000;

/// Default connection pool size per host.
pub const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 10;

/// Default idle connection timeout in seconds.
pub const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 30;

// =============================================================================
// MODEL DEFINITIONS
// =============================================================================

/// Available OpenAI model variants with their specifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAIModel {
    /// GPT-4o - Latest flagship model (128k context)
    #[default]
    Gpt4o,
    /// GPT-4o-mini - Smaller, faster, cheaper (128k context)
    Gpt4oMini,
    /// GPT-4-turbo - Previous flagship with vision (128k context)
    Gpt4Turbo,
    /// GPT-4 - Original GPT-4 (8k context)
    Gpt4,
    /// GPT-3.5-turbo - Fast and economical (16k context)
    Gpt35Turbo,
}

impl OpenAIModel {
    /// Get the model identifier string for API requests.
    pub fn as_str(&self) -> &'static str {
        match self {
            OpenAIModel::Gpt4o => "gpt-4o",
            OpenAIModel::Gpt4oMini => "gpt-4o-mini",
            OpenAIModel::Gpt4Turbo => "gpt-4-turbo",
            OpenAIModel::Gpt4 => "gpt-4",
            OpenAIModel::Gpt35Turbo => "gpt-3.5-turbo",
        }
    }

    /// Get the context window size (max tokens) for this model.
    pub fn context_window(&self) -> usize {
        match self {
            OpenAIModel::Gpt4o => 128_000,
            OpenAIModel::Gpt4oMini => 128_000,
            OpenAIModel::Gpt4Turbo => 128_000,
            OpenAIModel::Gpt4 => 8_192,
            OpenAIModel::Gpt35Turbo => 16_385,
        }
    }

    /// Get the maximum output tokens for this model.
    pub fn max_output_tokens(&self) -> usize {
        match self {
            OpenAIModel::Gpt4o => 16_384,
            OpenAIModel::Gpt4oMini => 16_384,
            OpenAIModel::Gpt4Turbo => 4_096,
            OpenAIModel::Gpt4 => 8_192,
            OpenAIModel::Gpt35Turbo => 4_096,
        }
    }

    /// Get the input cost per 1K tokens in USD.
    pub fn input_cost_per_1k(&self) -> f64 {
        match self {
            OpenAIModel::Gpt4o => 0.005,
            OpenAIModel::Gpt4oMini => 0.00015,
            OpenAIModel::Gpt4Turbo => 0.01,
            OpenAIModel::Gpt4 => 0.03,
            OpenAIModel::Gpt35Turbo => 0.0005,
        }
    }

    /// Get the output cost per 1K tokens in USD.
    pub fn output_cost_per_1k(&self) -> f64 {
        match self {
            OpenAIModel::Gpt4o => 0.015,
            OpenAIModel::Gpt4oMini => 0.0006,
            OpenAIModel::Gpt4Turbo => 0.03,
            OpenAIModel::Gpt4 => 0.06,
            OpenAIModel::Gpt35Turbo => 0.0015,
        }
    }

    /// Calculate the estimated cost for a given number of tokens.
    pub fn estimate_cost(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        let input_cost = (input_tokens as f64 / 1000.0) * self.input_cost_per_1k();
        let output_cost = (output_tokens as f64 / 1000.0) * self.output_cost_per_1k();
        input_cost + output_cost
    }

    /// Get tokens per minute limit for this model (tier 1 defaults).
    pub fn tokens_per_minute(&self) -> u32 {
        match self {
            OpenAIModel::Gpt4o => 30_000,
            OpenAIModel::Gpt4oMini => 200_000,
            OpenAIModel::Gpt4Turbo => 30_000,
            OpenAIModel::Gpt4 => 10_000,
            OpenAIModel::Gpt35Turbo => 60_000,
        }
    }

    /// Get requests per minute limit for this model (tier 1 defaults).
    pub fn requests_per_minute(&self) -> u32 {
        match self {
            OpenAIModel::Gpt4o => 500,
            OpenAIModel::Gpt4oMini => 500,
            OpenAIModel::Gpt4Turbo => 500,
            OpenAIModel::Gpt4 => 500,
            OpenAIModel::Gpt35Turbo => 3_500,
        }
    }

    /// Get all available models.
    pub fn all() -> &'static [OpenAIModel] {
        &[
            OpenAIModel::Gpt4o,
            OpenAIModel::Gpt4oMini,
            OpenAIModel::Gpt4Turbo,
            OpenAIModel::Gpt4,
            OpenAIModel::Gpt35Turbo,
        ]
    }
}

impl std::fmt::Display for OpenAIModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for OpenAIModel {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gpt-4o" | "gpt4o" | "4o" => Ok(OpenAIModel::Gpt4o),
            "gpt-4o-mini" | "gpt4o-mini" | "4o-mini" => Ok(OpenAIModel::Gpt4oMini),
            "gpt-4-turbo" | "gpt4-turbo" | "4-turbo" => Ok(OpenAIModel::Gpt4Turbo),
            "gpt-4" | "gpt4" | "4" => Ok(OpenAIModel::Gpt4),
            "gpt-3.5-turbo" | "gpt35-turbo" | "3.5-turbo" | "gpt-3.5" => {
                Ok(OpenAIModel::Gpt35Turbo)
            }
            _ => Err(Error::Config(format!(
                "Unknown OpenAI model: '{}'. Valid models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo",
                s
            ))),
        }
    }
}

impl OpenAIModel {
    /// Parse a model from a string identifier (infallible version).
    ///
    /// Returns the matching model variant or defaults to Gpt4o if unknown.
    pub fn from_str_lossy(s: &str) -> Self {
        s.parse().unwrap_or(OpenAIModel::Gpt4o)
    }
}

// =============================================================================
// TOKEN ESTIMATION
// =============================================================================

/// Estimate the number of tokens in a text string.
///
/// Uses a simple heuristic: ~4 characters per token on average.
/// For more accurate counting, use tiktoken or the actual API.
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::llm::openai::estimate_tokens;
///
/// let text = "Hello, world!";
/// let tokens = estimate_tokens(text);
/// assert!(tokens > 0);
/// ```
pub fn estimate_tokens(text: &str) -> usize {
    // Average of ~4 characters per token for English text
    // This is a rough approximation; actual tokenization varies by model
    (text.len() as f64 / 4.0).ceil() as usize
}

/// Estimate tokens for a list of messages.
pub fn estimate_messages_tokens(messages: &[OpenAIMessage]) -> usize {
    // Each message has ~4 tokens of overhead (role, formatting)
    let overhead_per_message = 4;
    let base_overhead = 3; // System message overhead

    let content_tokens: usize = messages.iter().map(|m| estimate_tokens(&m.content)).sum();
    let message_overhead = messages.len() * overhead_per_message;

    base_overhead + content_tokens + message_overhead
}

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// A chat message for OpenAI API requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIMessage {
    /// The role of the message author: "system", "user", or "assistant"
    pub role: String,
    /// The content of the message
    pub content: String,
    /// Optional name for the author (useful for multi-participant chats)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl OpenAIMessage {
    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            name: None,
        }
    }

    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            name: None,
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            name: None,
        }
    }

    /// Set an optional name for the message author.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Chat completion request to OpenAI API.
#[derive(Debug, Serialize)]
pub struct OpenAIRequest {
    /// Model identifier
    pub model: String,
    /// List of messages in the conversation
    pub messages: Vec<OpenAIMessage>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 - 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-p sampling (nucleus sampling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// User identifier for abuse detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl OpenAIRequest {
    /// Create a new request with default settings.
    pub fn new(model: &str, messages: Vec<OpenAIMessage>) -> Self {
        Self {
            model: model.to_string(),
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
        }
    }

    /// Set maximum tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Enable streaming.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
}

/// Chat completion response from OpenAI API.
#[derive(Debug, Deserialize)]
pub struct OpenAIResponse {
    /// Unique identifier for the completion
    pub id: String,
    /// Object type (always "chat.completion")
    pub object: String,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// List of completion choices
    pub choices: Vec<OpenAIChoice>,
    /// Token usage statistics
    pub usage: Option<OpenAIUsage>,
    /// System fingerprint
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

/// A completion choice from the response.
#[derive(Debug, Deserialize)]
pub struct OpenAIChoice {
    /// Index of this choice
    pub index: usize,
    /// The generated message
    pub message: OpenAIMessage,
    /// Reason for stopping generation
    pub finish_reason: Option<String>,
    /// Log probabilities (if requested)
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

/// Token usage statistics from the API response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIUsage {
    /// Tokens used in the prompt
    pub prompt_tokens: usize,
    /// Tokens used in the completion
    pub completion_tokens: usize,
    /// Total tokens used
    pub total_tokens: usize,
}

/// Streaming response chunk from OpenAI API.
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    /// Unique identifier for the completion
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for the completion
    pub model: String,
    /// List of delta choices
    pub choices: Vec<OpenAIStreamChoice>,
}

/// A streaming choice delta.
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChoice {
    /// Index of this choice
    pub index: usize,
    /// Incremental content delta
    pub delta: OpenAIStreamDelta,
    /// Reason for stopping (only on final chunk)
    pub finish_reason: Option<String>,
}

/// Incremental content in a streaming response.
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamDelta {
    /// Role (only in first chunk)
    #[serde(default)]
    pub role: Option<String>,
    /// Incremental content
    #[serde(default)]
    pub content: Option<String>,
}

// =============================================================================
// ERROR CATEGORIES
// =============================================================================

/// Categories of OpenAI API errors for proper handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAIErrorCategory {
    /// Authentication error (invalid API key)
    Authentication,
    /// Rate limit exceeded
    RateLimit,
    /// Invalid request (bad parameters)
    InvalidRequest,
    /// Server error (5xx)
    ServerError,
    /// Network/connection error
    NetworkError,
    /// Context length exceeded
    ContextLengthExceeded,
    /// Content moderation filter triggered
    ContentFiltered,
    /// Unknown error
    Unknown,
}

impl OpenAIErrorCategory {
    /// Check if this error category is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            OpenAIErrorCategory::RateLimit
                | OpenAIErrorCategory::ServerError
                | OpenAIErrorCategory::NetworkError
        )
    }

    /// Get the recommended retry delay for this error type.
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            OpenAIErrorCategory::RateLimit => Some(Duration::from_secs(60)),
            OpenAIErrorCategory::ServerError => Some(Duration::from_secs(5)),
            OpenAIErrorCategory::NetworkError => Some(Duration::from_secs(2)),
            _ => None,
        }
    }
}

/// Categorize an error based on status code and message.
pub fn categorize_error(status: u16, message: &str) -> OpenAIErrorCategory {
    match status {
        401 => OpenAIErrorCategory::Authentication,
        429 => OpenAIErrorCategory::RateLimit,
        400 => {
            if message.contains("context_length_exceeded") || message.contains("maximum context") {
                OpenAIErrorCategory::ContextLengthExceeded
            } else {
                OpenAIErrorCategory::InvalidRequest
            }
        }
        403 => {
            if message.contains("content") || message.contains("policy") {
                OpenAIErrorCategory::ContentFiltered
            } else {
                OpenAIErrorCategory::Authentication
            }
        }
        500..=599 => OpenAIErrorCategory::ServerError,
        _ => OpenAIErrorCategory::Unknown,
    }
}

// =============================================================================
// RATE LIMITER
// =============================================================================

/// Rate limiter for API request and token throttling.
#[derive(Debug)]
pub struct OpenAIRateLimiter {
    /// Maximum requests allowed per minute.
    requests_per_minute: u32,
    /// Maximum tokens allowed per minute.
    tokens_per_minute: u32,
    /// Timestamp of the current rate limit window start.
    window_start: Mutex<Instant>,
    /// Request count in current window.
    request_count: Mutex<u32>,
    /// Token count in current window.
    token_count: Mutex<u32>,
}

impl OpenAIRateLimiter {
    /// Create a new rate limiter with the specified limits.
    pub fn new(requests_per_minute: u32, tokens_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            tokens_per_minute,
            window_start: Mutex::new(Instant::now()),
            request_count: Mutex::new(0),
            token_count: Mutex::new(0),
        }
    }

    /// Create a rate limiter with model-specific defaults.
    pub fn for_model(model: OpenAIModel) -> Self {
        Self::new(model.requests_per_minute(), model.tokens_per_minute())
    }

    /// Reset the window if a minute has passed.
    async fn maybe_reset_window(&self) {
        let mut window_start = self.window_start.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(*window_start);

        if elapsed >= Duration::from_secs(60) {
            *window_start = now;
            *self.request_count.lock().await = 0;
            *self.token_count.lock().await = 0;
        }
    }

    /// Acquire permission for a request with estimated token count.
    /// Returns the time waited, if any.
    pub async fn acquire(&self, estimated_tokens: u32) -> Duration {
        self.maybe_reset_window().await;

        let mut window_start = self.window_start.lock().await;
        let mut request_count = self.request_count.lock().await;
        let mut token_count = self.token_count.lock().await;

        let now = Instant::now();
        let elapsed = now.duration_since(*window_start);

        // Check if we need to wait for request limit
        let request_wait = if *request_count >= self.requests_per_minute {
            Some(Duration::from_secs(60) - elapsed)
        } else {
            None
        };

        // Check if we need to wait for token limit
        let token_wait = if *token_count + estimated_tokens > self.tokens_per_minute {
            Some(Duration::from_secs(60) - elapsed)
        } else {
            None
        };

        // Wait for the longer of the two limits
        let wait_time = match (request_wait, token_wait) {
            (Some(r), Some(t)) => Some(r.max(t)),
            (Some(r), None) => Some(r),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        };

        if let Some(wait) = wait_time {
            if !wait.is_zero() {
                debug!("Rate limit reached, waiting {:?}", wait);
                drop(window_start);
                drop(request_count);
                drop(token_count);
                tokio::time::sleep(wait).await;

                // Reset after waiting
                let mut window_start = self.window_start.lock().await;
                let mut request_count = self.request_count.lock().await;
                let mut token_count = self.token_count.lock().await;
                *window_start = Instant::now();
                *request_count = 1;
                *token_count = estimated_tokens;
                return wait;
            }
        }

        *request_count += 1;
        *token_count += estimated_tokens;
        Duration::ZERO
    }

    /// Record actual token usage after a request completes.
    pub async fn record_tokens(&self, actual_tokens: u32) {
        let mut token_count = self.token_count.lock().await;
        // Adjust token count if actual usage differs from estimate
        *token_count = token_count.saturating_add(actual_tokens);
    }

    /// Get current request count in the window.
    pub async fn current_request_count(&self) -> u32 {
        self.maybe_reset_window().await;
        *self.request_count.lock().await
    }

    /// Get current token count in the window.
    pub async fn current_token_count(&self) -> u32 {
        self.maybe_reset_window().await;
        *self.token_count.lock().await
    }
}

// =============================================================================
// USAGE STATISTICS
// =============================================================================

/// Usage statistics for tracking OpenAI API consumption.
#[derive(Debug, Default)]
pub struct OpenAIUsageStats {
    /// Total number of API requests made.
    pub total_requests: AtomicU64,
    /// Number of successful requests.
    pub successful_requests: AtomicU64,
    /// Number of failed requests.
    pub failed_requests: AtomicU64,
    /// Total prompt tokens consumed.
    pub total_prompt_tokens: AtomicU64,
    /// Total completion tokens generated.
    pub total_completion_tokens: AtomicU64,
    /// Total estimated cost in USD (multiplied by 1_000_000 for precision).
    pub total_cost_micros: AtomicU64,
}

impl OpenAIUsageStats {
    /// Create a new empty stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request with token usage.
    pub fn record_success(
        &self,
        prompt_tokens: u64,
        completion_tokens: u64,
        model: OpenAIModel,
    ) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.total_completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);

        // Calculate and store cost in micros (USD * 1_000_000)
        let cost =
            model.estimate_cost(prompt_tokens as usize, completion_tokens as usize);
        let cost_micros = (cost * 1_000_000.0) as u64;
        self.total_cost_micros.fetch_add(cost_micros, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current statistics.
    pub fn snapshot(&self) -> OpenAIUsageStatsSnapshot {
        OpenAIUsageStatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            total_prompt_tokens: self.total_prompt_tokens.load(Ordering::Relaxed),
            total_completion_tokens: self.total_completion_tokens.load(Ordering::Relaxed),
            total_cost_usd: self.total_cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.total_prompt_tokens.store(0, Ordering::Relaxed);
        self.total_completion_tokens.store(0, Ordering::Relaxed);
        self.total_cost_micros.store(0, Ordering::Relaxed);
    }
}

/// A point-in-time snapshot of usage statistics.
#[derive(Debug, Clone, Default)]
pub struct OpenAIUsageStatsSnapshot {
    /// Total number of API requests made.
    pub total_requests: u64,
    /// Number of successful requests.
    pub successful_requests: u64,
    /// Number of failed requests.
    pub failed_requests: u64,
    /// Total prompt tokens consumed.
    pub total_prompt_tokens: u64,
    /// Total completion tokens generated.
    pub total_completion_tokens: u64,
    /// Total estimated cost in USD.
    pub total_cost_usd: f64,
}

impl OpenAIUsageStatsSnapshot {
    /// Calculate the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            100.0
        } else {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        }
    }

    /// Calculate total tokens (prompt + completion).
    pub fn total_tokens(&self) -> u64 {
        self.total_prompt_tokens + self.total_completion_tokens
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for OpenAI API client.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Base URL for the API (default: https://api.openai.com/v1).
    pub base_url: String,
    /// Optional organization ID.
    pub organization: Option<String>,
    /// Default model to use.
    pub default_model: OpenAIModel,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum retry attempts.
    pub max_retries: u32,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_OPENAI_ENDPOINT.to_string()),
            organization: env::var("OPENAI_ORGANIZATION").ok(),
            default_model: OpenAIModel::Gpt4o,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}

impl OpenAIConfig {
    /// Create a new configuration with custom API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Create a new configuration with custom base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Create a new configuration with organization.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Create a new configuration with custom model.
    pub fn with_model(mut self, model: OpenAIModel) -> Self {
        self.default_model = model;
        self
    }

    /// Create a new configuration with custom timeout.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Create a new configuration with custom max retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(Error::Config(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable.".to_string(),
            ));
        }
        if self.base_url.is_empty() {
            return Err(Error::Config("OpenAI base URL cannot be empty.".to_string()));
        }
        Ok(())
    }
}

// =============================================================================
// CLIENT BUILDER
// =============================================================================

/// Builder for creating an OpenAI client with custom configuration.
#[derive(Debug, Default)]
pub struct OpenAIClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    default_model: Option<OpenAIModel>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    requests_per_minute: Option<u32>,
    tokens_per_minute: Option<u32>,
}

impl OpenAIClientBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the API key.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the organization ID.
    pub fn organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the default model.
    pub fn model(mut self, model: OpenAIModel) -> Self {
        self.default_model = Some(model);
        self
    }

    /// Set the request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the request timeout in seconds.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout = Some(Duration::from_secs(secs));
        self
    }

    /// Set the maximum retry attempts.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set the requests per minute limit.
    pub fn requests_per_minute(mut self, rpm: u32) -> Self {
        self.requests_per_minute = Some(rpm);
        self
    }

    /// Set the tokens per minute limit.
    pub fn tokens_per_minute(mut self, tpm: u32) -> Self {
        self.tokens_per_minute = Some(tpm);
        self
    }

    /// Build the OpenAI client.
    pub fn build(self) -> Result<OpenAIClient> {
        let api_key = self
            .api_key
            .or_else(|| env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                Error::Config(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable or use .api_key()".to_string()
                )
            })?;

        let base_url = self
            .base_url
            .or_else(|| env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| DEFAULT_OPENAI_ENDPOINT.to_string());

        let organization = self.organization.or_else(|| env::var("OPENAI_ORGANIZATION").ok());

        let model = self.default_model.unwrap_or(OpenAIModel::Gpt4o);
        let timeout = self.timeout.unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS));
        let max_retries = self.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);

        let rpm = self.requests_per_minute.unwrap_or(model.requests_per_minute());
        let tpm = self.tokens_per_minute.unwrap_or(model.tokens_per_minute());

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(DEFAULT_POOL_MAX_IDLE_PER_HOST)
            .pool_idle_timeout(Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS))
            .build()
            .map_err(|e| Error::Config(format!("Failed to create HTTP client: {}", e)))?;

        Ok(OpenAIClient {
            http_client: client,
            api_key,
            base_url,
            organization,
            default_model: model,
            timeout,
            max_retries,
            rate_limiter: Arc::new(OpenAIRateLimiter::new(rpm, tpm)),
            stats: Arc::new(OpenAIUsageStats::new()),
        })
    }
}

// =============================================================================
// OPENAI CLIENT
// =============================================================================

/// OpenAI API client with automatic retry, rate limiting, and usage tracking.
///
/// # Features
///
/// - Automatic retry with exponential backoff
/// - Rate limiting (requests and tokens per minute)
/// - Multiple model support
/// - Streaming responses
/// - Usage statistics and cost tracking
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::llm::openai::{OpenAIClient, OpenAIModel};
///
/// let client = OpenAIClient::builder()
///     .api_key("sk-...")
///     .model(OpenAIModel::Gpt4o)
///     .build()?;
///
/// let response = client.complete_with_retry("Hello!").await?;
/// println!("{}", response);
/// ```
#[derive(Debug)]
pub struct OpenAIClient {
    /// HTTP client for making requests.
    http_client: reqwest::Client,
    /// API key for authentication.
    api_key: String,
    /// Base URL for the API.
    base_url: String,
    /// Optional organization ID.
    organization: Option<String>,
    /// Default model to use.
    default_model: OpenAIModel,
    /// Request timeout.
    timeout: Duration,
    /// Maximum retry attempts.
    max_retries: u32,
    /// Rate limiter.
    rate_limiter: Arc<OpenAIRateLimiter>,
    /// Usage statistics.
    stats: Arc<OpenAIUsageStats>,
}

impl Clone for OpenAIClient {
    fn clone(&self) -> Self {
        Self {
            http_client: self.http_client.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            default_model: self.default_model,
            timeout: self.timeout,
            max_retries: self.max_retries,
            rate_limiter: Arc::clone(&self.rate_limiter),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl OpenAIClient {
    /// Create a new builder for configuring the client.
    pub fn builder() -> OpenAIClientBuilder {
        OpenAIClientBuilder::new()
    }

    /// Create a new client with default settings.
    ///
    /// Requires `OPENAI_API_KEY` environment variable to be set.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    /// Create a new client from configuration.
    pub fn from_config(config: OpenAIConfig) -> Result<Self> {
        config.validate()?;

        let timeout = Duration::from_secs(config.timeout_secs);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(DEFAULT_POOL_MAX_IDLE_PER_HOST)
            .pool_idle_timeout(Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS))
            .build()
            .map_err(|e| Error::Config(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            http_client: client,
            api_key: config.api_key,
            base_url: config.base_url,
            organization: config.organization,
            default_model: config.default_model,
            timeout,
            max_retries: config.max_retries,
            rate_limiter: Arc::new(OpenAIRateLimiter::for_model(config.default_model)),
            stats: Arc::new(OpenAIUsageStats::new()),
        })
    }

    /// Get the current model.
    pub fn model(&self) -> OpenAIModel {
        self.default_model
    }

    /// Set a different model for this client.
    pub fn with_model(mut self, model: OpenAIModel) -> Self {
        self.default_model = model;
        self.rate_limiter = Arc::new(OpenAIRateLimiter::for_model(model));
        self
    }

    /// Set custom endpoint.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.http_client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(DEFAULT_POOL_MAX_IDLE_PER_HOST)
            .pool_idle_timeout(Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS))
            .build()
            .expect("Failed to create HTTP client");
        self
    }

    /// Set maximum retry attempts.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Get a reference to the usage statistics.
    pub fn stats(&self) -> &Arc<OpenAIUsageStats> {
        &self.stats
    }

    /// Get a snapshot of current usage statistics.
    pub fn get_stats(&self) -> OpenAIUsageStatsSnapshot {
        self.stats.snapshot()
    }

    /// Reset usage statistics.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// Check if the API endpoint is healthy and responding.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", self.base_url);

        let mut request = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key));

        if let Some(ref org) = self.organization {
            request = request.header("OpenAI-Organization", org);
        }

        let response = request.timeout(Duration::from_secs(10)).send().await;

        match response {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(e) => {
                warn!("Health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Calculate exponential backoff delay for retry attempts.
    fn backoff_delay(attempt: u32) -> Duration {
        // 1s, 2s, 4s, 8s, ... capped at 32s
        let secs = (1u64 << attempt.min(5)).min(32);
        Duration::from_secs(secs)
    }

    /// Build request headers.
    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.api_key)
                .parse()
                .expect("Invalid auth header"),
        );
        headers.insert(
            "Content-Type",
            "application/json".parse().expect("Invalid content type"),
        );
        if let Some(ref org) = self.organization {
            headers.insert(
                "OpenAI-Organization",
                org.parse().expect("Invalid organization header"),
            );
        }
        headers
    }

    /// Send a chat completion request with automatic retry.
    pub async fn chat_with_retry(
        &self,
        messages: Vec<OpenAIMessage>,
    ) -> Result<OpenAIResponse> {
        self.chat_with_retry_model(messages, self.default_model).await
    }

    /// Send a chat completion request with automatic retry and specific model.
    pub async fn chat_with_retry_model(
        &self,
        messages: Vec<OpenAIMessage>,
        model: OpenAIModel,
    ) -> Result<OpenAIResponse> {
        let mut last_error = None;
        let estimated_tokens = estimate_messages_tokens(&messages) as u32;

        for attempt in 0..=self.max_retries {
            // Apply rate limiting
            let waited = self.rate_limiter.acquire(estimated_tokens).await;
            if !waited.is_zero() {
                info!("Rate limited, waited {:?}", waited);
            }

            // Apply backoff on retry attempts
            if attempt > 0 {
                let delay = Self::backoff_delay(attempt - 1);
                debug!("Retry attempt {}, waiting {:?}", attempt, delay);
                tokio::time::sleep(delay).await;
            }

            match self.chat_internal(&messages, model).await {
                Ok(response) => {
                    // Record success
                    if let Some(usage) = &response.usage {
                        self.stats.record_success(
                            usage.prompt_tokens as u64,
                            usage.completion_tokens as u64,
                            model,
                        );
                    } else {
                        self.stats.record_success(0, 0, model);
                    }
                    return Ok(response);
                }
                Err(e) => {
                    warn!("Request failed (attempt {}): {}", attempt + 1, e);

                    // Check if error is retryable
                    let should_retry = match &e {
                        Error::ApiResponse { status, .. } => {
                            let category = categorize_error(*status, "");
                            category.is_retryable()
                        }
                        Error::Http(_) => true,
                        Error::ApiTimeout { .. } => true,
                        Error::ApiRateLimited { .. } => true,
                        _ => false,
                    };

                    if !should_retry {
                        self.stats.record_failure();
                        return Err(e);
                    }

                    last_error = Some(e);
                }
            }
        }

        // All retries exhausted
        self.stats.record_failure();
        Err(last_error.unwrap_or_else(|| {
            Error::Synthesis("Max retries exceeded".to_string())
        }))
    }

    /// Internal chat implementation without retry logic.
    async fn chat_internal(
        &self,
        messages: &[OpenAIMessage],
        model: OpenAIModel,
    ) -> Result<OpenAIResponse> {
        let request = OpenAIRequest::new(model.as_str(), messages.to_vec());

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let status_code = status.as_u16();

            return match status_code {
                401 => Err(Error::ApiAuthFailed),
                429 => Err(Error::ApiRateLimited {
                    retry_after_secs: None,
                }),
                _ => Err(Error::ApiResponse {
                    status: status_code,
                    message: text,
                }),
            };
        }

        Ok(response.json().await?)
    }

    /// Send a chat completion request (no retry).
    pub async fn chat(&self, messages: Vec<OpenAIMessage>) -> Result<OpenAIResponse> {
        let estimated_tokens = estimate_messages_tokens(&messages) as u32;
        self.rate_limiter.acquire(estimated_tokens).await;

        let result = self.chat_internal(&messages, self.default_model).await;

        match &result {
            Ok(response) => {
                if let Some(usage) = &response.usage {
                    self.stats.record_success(
                        usage.prompt_tokens as u64,
                        usage.completion_tokens as u64,
                        self.default_model,
                    );
                } else {
                    self.stats.record_success(0, 0, self.default_model);
                }
            }
            Err(_) => {
                self.stats.record_failure();
            }
        }

        result
    }

    /// Simple text completion with retry logic.
    pub async fn complete_with_retry(&self, prompt: &str) -> Result<String> {
        let messages = vec![OpenAIMessage::user(prompt)];
        let response = self.chat_with_retry(messages).await?;

        response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| Error::Synthesis("No response from API".to_string()))
    }

    /// Simple text completion (no retry).
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let messages = vec![OpenAIMessage::user(prompt)];
        let response = self.chat(messages).await?;

        response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| Error::Synthesis("No response from API".to_string()))
    }

    /// Stream a chat completion response, yielding tokens as they arrive.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let client = OpenAIClient::new()?;
    /// let messages = vec![OpenAIMessage::user("Hello!")];
    ///
    /// let mut stream = client.chat_streaming(messages).await?;
    /// while let Some(chunk) = stream.next().await {
    ///     match chunk {
    ///         Ok(text) => print!("{}", text),
    ///         Err(e) => eprintln!("Error: {}", e),
    ///     }
    /// }
    /// ```
    pub async fn chat_streaming(
        &self,
        messages: Vec<OpenAIMessage>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        use futures::StreamExt;

        let estimated_tokens = estimate_messages_tokens(&messages) as u32;
        self.rate_limiter.acquire(estimated_tokens).await;

        let request = OpenAIRequest::new(self.default_model.as_str(), messages).with_stream(true);

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            self.stats.record_failure();
            return Err(Error::ApiResponse {
                status: status.as_u16(),
                message: text,
            });
        }

        let stats = Arc::clone(&self.stats);
        let model = self.default_model;
        let byte_stream = response.bytes_stream();

        let stream = async_stream::stream! {
            let mut buffer = String::new();
            let mut total_tokens = 0u64;

            futures::pin_mut!(byte_stream);

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process complete SSE lines
                        while let Some(line_end) = buffer.find('\n') {
                            let line = buffer[..line_end].trim().to_string();
                            buffer = buffer[line_end + 1..].to_string();

                            if line.is_empty() || line == "data: [DONE]" {
                                continue;
                            }

                            if let Some(json_str) = line.strip_prefix("data: ") {
                                match serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                                    Ok(chunk) => {
                                        if let Some(choice) = chunk.choices.first() {
                                            if let Some(content) = &choice.delta.content {
                                                total_tokens += 1;
                                                yield Ok(content.clone());
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        debug!("Failed to parse stream chunk: {}", e);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        stats.record_failure();
                        yield Err(Error::Http(e));
                        return;
                    }
                }
            }

            // Record approximate token usage for streaming
            stats.record_success(0, total_tokens, model);
        };

        Ok(Box::pin(stream))
    }

    /// Stream a simple text completion.
    pub async fn complete_streaming(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let messages = vec![OpenAIMessage::user(prompt)];
        self.chat_streaming(messages).await
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MODEL TESTS
    // =========================================================================

    #[test]
    fn test_model_as_str() {
        assert_eq!(OpenAIModel::Gpt4o.as_str(), "gpt-4o");
        assert_eq!(OpenAIModel::Gpt4oMini.as_str(), "gpt-4o-mini");
        assert_eq!(OpenAIModel::Gpt4Turbo.as_str(), "gpt-4-turbo");
        assert_eq!(OpenAIModel::Gpt4.as_str(), "gpt-4");
        assert_eq!(OpenAIModel::Gpt35Turbo.as_str(), "gpt-3.5-turbo");
    }

    #[test]
    fn test_model_display() {
        assert_eq!(format!("{}", OpenAIModel::Gpt4o), "gpt-4o");
        assert_eq!(format!("{}", OpenAIModel::Gpt35Turbo), "gpt-3.5-turbo");
    }

    #[test]
    fn test_model_from_str() {
        assert_eq!("gpt-4o".parse::<OpenAIModel>().unwrap(), OpenAIModel::Gpt4o);
        assert_eq!(
            "gpt-4o-mini".parse::<OpenAIModel>().unwrap(),
            OpenAIModel::Gpt4oMini
        );
        assert_eq!(
            "gpt-4-turbo".parse::<OpenAIModel>().unwrap(),
            OpenAIModel::Gpt4Turbo
        );
        assert_eq!("gpt-4".parse::<OpenAIModel>().unwrap(), OpenAIModel::Gpt4);
        assert_eq!(
            "gpt-3.5-turbo".parse::<OpenAIModel>().unwrap(),
            OpenAIModel::Gpt35Turbo
        );
    }

    #[test]
    fn test_model_from_str_short_names() {
        assert_eq!("4o".parse::<OpenAIModel>().unwrap(), OpenAIModel::Gpt4o);
        assert_eq!(
            "4o-mini".parse::<OpenAIModel>().unwrap(),
            OpenAIModel::Gpt4oMini
        );
        assert_eq!(
            "4-turbo".parse::<OpenAIModel>().unwrap(),
            OpenAIModel::Gpt4Turbo
        );
        assert_eq!("4".parse::<OpenAIModel>().unwrap(), OpenAIModel::Gpt4);
        assert_eq!(
            "3.5-turbo".parse::<OpenAIModel>().unwrap(),
            OpenAIModel::Gpt35Turbo
        );
    }

    #[test]
    fn test_model_from_str_case_insensitive() {
        assert_eq!("GPT-4O".parse::<OpenAIModel>().unwrap(), OpenAIModel::Gpt4o);
        assert_eq!(
            "Gpt-4o-Mini".parse::<OpenAIModel>().unwrap(),
            OpenAIModel::Gpt4oMini
        );
    }

    #[test]
    fn test_model_from_str_invalid() {
        let result = "invalid-model".parse::<OpenAIModel>();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown OpenAI model"));
    }

    #[test]
    fn test_model_from_str_lossy() {
        assert_eq!(OpenAIModel::from_str_lossy("gpt-4o"), OpenAIModel::Gpt4o);
        assert_eq!(OpenAIModel::from_str_lossy("invalid"), OpenAIModel::Gpt4o);
    }

    #[test]
    fn test_model_context_window() {
        assert_eq!(OpenAIModel::Gpt4o.context_window(), 128_000);
        assert_eq!(OpenAIModel::Gpt4oMini.context_window(), 128_000);
        assert_eq!(OpenAIModel::Gpt4Turbo.context_window(), 128_000);
        assert_eq!(OpenAIModel::Gpt4.context_window(), 8_192);
        assert_eq!(OpenAIModel::Gpt35Turbo.context_window(), 16_385);
    }

    #[test]
    fn test_model_max_output_tokens() {
        assert_eq!(OpenAIModel::Gpt4o.max_output_tokens(), 16_384);
        assert_eq!(OpenAIModel::Gpt4.max_output_tokens(), 8_192);
        assert_eq!(OpenAIModel::Gpt35Turbo.max_output_tokens(), 4_096);
    }

    #[test]
    fn test_model_costs() {
        // GPT-4o costs
        assert_eq!(OpenAIModel::Gpt4o.input_cost_per_1k(), 0.005);
        assert_eq!(OpenAIModel::Gpt4o.output_cost_per_1k(), 0.015);

        // GPT-4o-mini costs (should be much cheaper)
        assert!(OpenAIModel::Gpt4oMini.input_cost_per_1k() < OpenAIModel::Gpt4o.input_cost_per_1k());
        assert!(
            OpenAIModel::Gpt4oMini.output_cost_per_1k() < OpenAIModel::Gpt4o.output_cost_per_1k()
        );
    }

    #[test]
    fn test_model_estimate_cost() {
        let cost = OpenAIModel::Gpt4o.estimate_cost(1000, 500);
        let expected = (1.0 * 0.005) + (0.5 * 0.015);
        assert!((cost - expected).abs() < 0.0001);
    }

    #[test]
    fn test_model_all() {
        let models = OpenAIModel::all();
        assert_eq!(models.len(), 5);
        assert!(models.contains(&OpenAIModel::Gpt4o));
        assert!(models.contains(&OpenAIModel::Gpt35Turbo));
    }

    #[test]
    fn test_model_rate_limits() {
        assert!(OpenAIModel::Gpt4o.requests_per_minute() > 0);
        assert!(OpenAIModel::Gpt4o.tokens_per_minute() > 0);
        // GPT-3.5 should have higher limits than GPT-4
        assert!(
            OpenAIModel::Gpt35Turbo.requests_per_minute()
                >= OpenAIModel::Gpt4.requests_per_minute()
        );
    }

    // =========================================================================
    // TOKEN ESTIMATION TESTS
    // =========================================================================

    #[test]
    fn test_estimate_tokens() {
        let text = "Hello, world!";
        let tokens = estimate_tokens(text);
        // ~13 chars / 4 = ~4 tokens
        assert!(tokens >= 3 && tokens <= 5);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_long_text() {
        let text = "a".repeat(1000);
        let tokens = estimate_tokens(&text);
        // 1000 chars / 4 = 250 tokens
        assert!(tokens >= 240 && tokens <= 260);
    }

    #[test]
    fn test_estimate_messages_tokens() {
        let messages = vec![
            OpenAIMessage::system("You are a helpful assistant."),
            OpenAIMessage::user("Hello!"),
        ];
        let tokens = estimate_messages_tokens(&messages);
        // Should include content tokens plus overhead
        assert!(tokens > 0);
        assert!(tokens > estimate_tokens("You are a helpful assistant.Hello!"));
    }

    // =========================================================================
    // MESSAGE TESTS
    // =========================================================================

    #[test]
    fn test_message_system() {
        let msg = OpenAIMessage::system("You are helpful.");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are helpful.");
        assert!(msg.name.is_none());
    }

    #[test]
    fn test_message_user() {
        let msg = OpenAIMessage::user("Hello!");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn test_message_assistant() {
        let msg = OpenAIMessage::assistant("Hi there!");
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "Hi there!");
    }

    #[test]
    fn test_message_with_name() {
        let msg = OpenAIMessage::user("Hello!").with_name("Alice");
        assert_eq!(msg.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_message_serialization() {
        let msg = OpenAIMessage::user("Hello!");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello!\""));
        // name should not be serialized when None
        assert!(!json.contains("name"));
    }

    #[test]
    fn test_message_with_name_serialization() {
        let msg = OpenAIMessage::user("Hello!").with_name("Bob");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"name\":\"Bob\""));
    }

    #[test]
    fn test_message_deserialization() {
        let json = r#"{"role":"user","content":"Hello!"}"#;
        let msg: OpenAIMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
        assert!(msg.name.is_none());
    }

    // =========================================================================
    // REQUEST TESTS
    // =========================================================================

    #[test]
    fn test_request_new() {
        let messages = vec![OpenAIMessage::user("Hello")];
        let request = OpenAIRequest::new("gpt-4o", messages);
        assert_eq!(request.model, "gpt-4o");
        assert_eq!(request.messages.len(), 1);
        assert!(request.max_tokens.is_none());
        assert!(request.stream.is_none());
    }

    #[test]
    fn test_request_with_max_tokens() {
        let messages = vec![OpenAIMessage::user("Hello")];
        let request = OpenAIRequest::new("gpt-4o", messages).with_max_tokens(100);
        assert_eq!(request.max_tokens, Some(100));
    }

    #[test]
    fn test_request_with_temperature() {
        let messages = vec![OpenAIMessage::user("Hello")];
        let request = OpenAIRequest::new("gpt-4o", messages).with_temperature(0.7);
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_request_with_stream() {
        let messages = vec![OpenAIMessage::user("Hello")];
        let request = OpenAIRequest::new("gpt-4o", messages).with_stream(true);
        assert_eq!(request.stream, Some(true));
    }

    #[test]
    fn test_request_serialization_minimal() {
        let messages = vec![OpenAIMessage::user("Hello")];
        let request = OpenAIRequest::new("gpt-4o", messages);
        let json = serde_json::to_string(&request).unwrap();
        // Optional fields should not be present
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("temperature"));
        assert!(!json.contains("stream"));
    }

    // =========================================================================
    // RESPONSE TESTS
    // =========================================================================

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: OpenAIResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, "Hello!");
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));

        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_response_without_usage() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }]
        }"#;

        let response: OpenAIResponse = serde_json::from_str(json).unwrap();
        assert!(response.usage.is_none());
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: OpenAIStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id, "chatcmpl-123");
        assert_eq!(
            chunk.choices[0].delta.content,
            Some("Hello".to_string())
        );
    }

    // =========================================================================
    // ERROR CATEGORY TESTS
    // =========================================================================

    #[test]
    fn test_categorize_error_auth() {
        assert_eq!(
            categorize_error(401, "Unauthorized"),
            OpenAIErrorCategory::Authentication
        );
    }

    #[test]
    fn test_categorize_error_rate_limit() {
        assert_eq!(
            categorize_error(429, "Rate limit exceeded"),
            OpenAIErrorCategory::RateLimit
        );
    }

    #[test]
    fn test_categorize_error_context_length() {
        assert_eq!(
            categorize_error(400, "context_length_exceeded"),
            OpenAIErrorCategory::ContextLengthExceeded
        );
    }

    #[test]
    fn test_categorize_error_content_filtered() {
        assert_eq!(
            categorize_error(403, "content policy violation"),
            OpenAIErrorCategory::ContentFiltered
        );
    }

    #[test]
    fn test_categorize_error_server() {
        assert_eq!(
            categorize_error(500, "Internal server error"),
            OpenAIErrorCategory::ServerError
        );
        assert_eq!(
            categorize_error(503, "Service unavailable"),
            OpenAIErrorCategory::ServerError
        );
    }

    #[test]
    fn test_error_category_is_retryable() {
        assert!(OpenAIErrorCategory::RateLimit.is_retryable());
        assert!(OpenAIErrorCategory::ServerError.is_retryable());
        assert!(OpenAIErrorCategory::NetworkError.is_retryable());
        assert!(!OpenAIErrorCategory::Authentication.is_retryable());
        assert!(!OpenAIErrorCategory::InvalidRequest.is_retryable());
    }

    #[test]
    fn test_error_category_retry_delay() {
        assert!(OpenAIErrorCategory::RateLimit.retry_delay().is_some());
        assert!(OpenAIErrorCategory::ServerError.retry_delay().is_some());
        assert!(OpenAIErrorCategory::Authentication.retry_delay().is_none());
    }

    // =========================================================================
    // USAGE STATS TESTS
    // =========================================================================

    #[test]
    fn test_usage_stats_new() {
        let stats = OpenAIUsageStats::new();
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.total_tokens(), 0);
    }

    #[test]
    fn test_usage_stats_record_success() {
        let stats = OpenAIUsageStats::new();
        stats.record_success(100, 50, OpenAIModel::Gpt4o);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.successful_requests, 1);
        assert_eq!(snapshot.failed_requests, 0);
        assert_eq!(snapshot.total_prompt_tokens, 100);
        assert_eq!(snapshot.total_completion_tokens, 50);
        assert!(snapshot.total_cost_usd > 0.0);
    }

    #[test]
    fn test_usage_stats_record_failure() {
        let stats = OpenAIUsageStats::new();
        stats.record_failure();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.failed_requests, 1);
    }

    #[test]
    fn test_usage_stats_multiple() {
        let stats = OpenAIUsageStats::new();
        stats.record_success(100, 50, OpenAIModel::Gpt4o);
        stats.record_success(200, 100, OpenAIModel::Gpt4o);
        stats.record_failure();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 3);
        assert_eq!(snapshot.successful_requests, 2);
        assert_eq!(snapshot.failed_requests, 1);
        assert_eq!(snapshot.total_prompt_tokens, 300);
        assert_eq!(snapshot.total_completion_tokens, 150);
    }

    #[test]
    fn test_usage_stats_reset() {
        let stats = OpenAIUsageStats::new();
        stats.record_success(100, 50, OpenAIModel::Gpt4o);
        stats.reset();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.total_tokens(), 0);
        assert_eq!(snapshot.total_cost_usd, 0.0);
    }

    #[test]
    fn test_usage_stats_snapshot_success_rate() {
        let stats = OpenAIUsageStats::new();
        stats.record_success(100, 50, OpenAIModel::Gpt4o);
        stats.record_success(100, 50, OpenAIModel::Gpt4o);
        stats.record_failure();

        let snapshot = stats.snapshot();
        let rate = snapshot.success_rate();
        assert!((rate - 66.666).abs() < 0.01);
    }

    #[test]
    fn test_usage_stats_snapshot_success_rate_empty() {
        let snapshot = OpenAIUsageStatsSnapshot::default();
        assert_eq!(snapshot.success_rate(), 100.0);
    }

    #[test]
    fn test_usage_stats_snapshot_total_tokens() {
        let mut snapshot = OpenAIUsageStatsSnapshot::default();
        snapshot.total_prompt_tokens = 100;
        snapshot.total_completion_tokens = 50;
        assert_eq!(snapshot.total_tokens(), 150);
    }

    // =========================================================================
    // CONFIG TESTS
    // =========================================================================

    #[test]
    fn test_config_default() {
        // Clear env vars for predictable test
        let saved_key = env::var("OPENAI_API_KEY").ok();
        let saved_url = env::var("OPENAI_BASE_URL").ok();
        env::remove_var("OPENAI_API_KEY");
        env::remove_var("OPENAI_BASE_URL");

        let config = OpenAIConfig::default();
        assert!(config.api_key.is_empty());
        assert_eq!(config.base_url, DEFAULT_OPENAI_ENDPOINT);
        assert_eq!(config.default_model, OpenAIModel::Gpt4o);
        assert_eq!(config.timeout_secs, DEFAULT_TIMEOUT_SECS);

        // Restore
        if let Some(key) = saved_key {
            env::set_var("OPENAI_API_KEY", key);
        }
        if let Some(url) = saved_url {
            env::set_var("OPENAI_BASE_URL", url);
        }
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = OpenAIConfig::default()
            .with_api_key("test-key")
            .with_base_url("https://custom.api.com")
            .with_organization("org-123")
            .with_model(OpenAIModel::Gpt4Turbo)
            .with_timeout(120)
            .with_max_retries(5);

        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://custom.api.com");
        assert_eq!(config.organization, Some("org-123".to_string()));
        assert_eq!(config.default_model, OpenAIModel::Gpt4Turbo);
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_config_validate_missing_key() {
        let config = OpenAIConfig::default();
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validate_success() {
        let config = OpenAIConfig::default().with_api_key("sk-test");
        let result = config.validate();
        assert!(result.is_ok());
    }

    // =========================================================================
    // BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_builder_new() {
        let builder = OpenAIClientBuilder::new();
        // Should be default/empty
        assert!(builder.api_key.is_none());
        assert!(builder.base_url.is_none());
    }

    #[test]
    fn test_builder_api_key() {
        let builder = OpenAIClientBuilder::new().api_key("sk-test");
        assert_eq!(builder.api_key, Some("sk-test".to_string()));
    }

    #[test]
    fn test_builder_base_url() {
        let builder = OpenAIClientBuilder::new().base_url("https://custom.com");
        assert_eq!(builder.base_url, Some("https://custom.com".to_string()));
    }

    #[test]
    fn test_builder_organization() {
        let builder = OpenAIClientBuilder::new().organization("org-123");
        assert_eq!(builder.organization, Some("org-123".to_string()));
    }

    #[test]
    fn test_builder_model() {
        let builder = OpenAIClientBuilder::new().model(OpenAIModel::Gpt35Turbo);
        assert_eq!(builder.default_model, Some(OpenAIModel::Gpt35Turbo));
    }

    #[test]
    fn test_builder_timeout() {
        let builder = OpenAIClientBuilder::new().timeout(Duration::from_secs(120));
        assert_eq!(builder.timeout, Some(Duration::from_secs(120)));
    }

    #[test]
    fn test_builder_timeout_secs() {
        let builder = OpenAIClientBuilder::new().timeout_secs(90);
        assert_eq!(builder.timeout, Some(Duration::from_secs(90)));
    }

    #[test]
    fn test_builder_max_retries() {
        let builder = OpenAIClientBuilder::new().max_retries(5);
        assert_eq!(builder.max_retries, Some(5));
    }

    #[test]
    fn test_builder_rate_limits() {
        let builder = OpenAIClientBuilder::new()
            .requests_per_minute(100)
            .tokens_per_minute(50000);
        assert_eq!(builder.requests_per_minute, Some(100));
        assert_eq!(builder.tokens_per_minute, Some(50000));
    }

    #[test]
    fn test_builder_build_success() {
        let result = OpenAIClientBuilder::new().api_key("sk-test").build();
        assert!(result.is_ok());
        let client = result.unwrap();
        assert_eq!(client.default_model, OpenAIModel::Gpt4o);
    }

    #[test]
    fn test_builder_build_missing_key() {
        // Clear env var
        let saved = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");

        let result = OpenAIClientBuilder::new().build();
        assert!(result.is_err());

        // Restore
        if let Some(key) = saved {
            env::set_var("OPENAI_API_KEY", key);
        }
    }

    #[test]
    fn test_builder_chain() {
        let result = OpenAIClientBuilder::new()
            .api_key("sk-test")
            .base_url("https://custom.com")
            .model(OpenAIModel::Gpt4Turbo)
            .timeout_secs(90)
            .max_retries(5)
            .requests_per_minute(100)
            .tokens_per_minute(50000)
            .build();

        assert!(result.is_ok());
        let client = result.unwrap();
        assert_eq!(client.base_url, "https://custom.com");
        assert_eq!(client.default_model, OpenAIModel::Gpt4Turbo);
        assert_eq!(client.timeout, Duration::from_secs(90));
        assert_eq!(client.max_retries, 5);
    }

    // =========================================================================
    // CLIENT TESTS
    // =========================================================================

    #[test]
    fn test_client_builder_method() {
        let builder = OpenAIClient::builder();
        // Should return a builder
        assert!(builder.api_key.is_none());
    }

    #[test]
    fn test_client_from_config() {
        let config = OpenAIConfig::default()
            .with_api_key("sk-test")
            .with_model(OpenAIModel::Gpt35Turbo);

        let result = OpenAIClient::from_config(config);
        assert!(result.is_ok());
        let client = result.unwrap();
        assert_eq!(client.model(), OpenAIModel::Gpt35Turbo);
    }

    #[test]
    fn test_client_with_model() {
        let client = OpenAIClientBuilder::new()
            .api_key("sk-test")
            .build()
            .unwrap()
            .with_model(OpenAIModel::Gpt4Turbo);

        assert_eq!(client.model(), OpenAIModel::Gpt4Turbo);
    }

    #[test]
    fn test_client_with_base_url() {
        let client = OpenAIClientBuilder::new()
            .api_key("sk-test")
            .build()
            .unwrap()
            .with_base_url("https://custom.com");

        assert_eq!(client.base_url, "https://custom.com");
    }

    #[test]
    fn test_client_with_max_retries() {
        let client = OpenAIClientBuilder::new()
            .api_key("sk-test")
            .build()
            .unwrap()
            .with_max_retries(10);

        assert_eq!(client.max_retries, 10);
    }

    #[test]
    fn test_client_clone_shares_stats() {
        let client1 = OpenAIClientBuilder::new()
            .api_key("sk-test")
            .build()
            .unwrap();

        let client2 = client1.clone();

        client1.stats.record_success(100, 50, OpenAIModel::Gpt4o);

        // Both clients should share the same stats
        let snapshot1 = client1.get_stats();
        let snapshot2 = client2.get_stats();

        assert_eq!(snapshot1.total_requests, snapshot2.total_requests);
        assert_eq!(snapshot1.successful_requests, snapshot2.successful_requests);
    }

    #[test]
    fn test_client_reset_stats() {
        let client = OpenAIClientBuilder::new()
            .api_key("sk-test")
            .build()
            .unwrap();

        client.stats.record_success(100, 50, OpenAIModel::Gpt4o);
        client.reset_stats();

        let snapshot = client.get_stats();
        assert_eq!(snapshot.total_requests, 0);
    }

    #[test]
    fn test_backoff_delay() {
        assert_eq!(OpenAIClient::backoff_delay(0), Duration::from_secs(1));
        assert_eq!(OpenAIClient::backoff_delay(1), Duration::from_secs(2));
        assert_eq!(OpenAIClient::backoff_delay(2), Duration::from_secs(4));
        assert_eq!(OpenAIClient::backoff_delay(3), Duration::from_secs(8));
        assert_eq!(OpenAIClient::backoff_delay(4), Duration::from_secs(16));
        assert_eq!(OpenAIClient::backoff_delay(5), Duration::from_secs(32));
        // Capped at 32 seconds
        assert_eq!(OpenAIClient::backoff_delay(6), Duration::from_secs(32));
        assert_eq!(OpenAIClient::backoff_delay(100), Duration::from_secs(32));
    }

    // =========================================================================
    // RATE LIMITER TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_rate_limiter_new() {
        let limiter = OpenAIRateLimiter::new(60, 30000);
        assert_eq!(limiter.current_request_count().await, 0);
        assert_eq!(limiter.current_token_count().await, 0);
    }

    #[tokio::test]
    async fn test_rate_limiter_for_model() {
        let limiter = OpenAIRateLimiter::for_model(OpenAIModel::Gpt4o);
        assert_eq!(limiter.requests_per_minute, OpenAIModel::Gpt4o.requests_per_minute());
        assert_eq!(limiter.tokens_per_minute, OpenAIModel::Gpt4o.tokens_per_minute());
    }

    #[tokio::test]
    async fn test_rate_limiter_acquire() {
        let limiter = OpenAIRateLimiter::new(100, 100000);

        // First request should not wait
        let waited = limiter.acquire(100).await;
        assert_eq!(waited, Duration::ZERO);

        assert_eq!(limiter.current_request_count().await, 1);
        assert_eq!(limiter.current_token_count().await, 100);
    }

    #[tokio::test]
    async fn test_rate_limiter_multiple_requests() {
        let limiter = OpenAIRateLimiter::new(100, 100000);

        for _ in 0..10 {
            let waited = limiter.acquire(50).await;
            assert_eq!(waited, Duration::ZERO);
        }

        assert_eq!(limiter.current_request_count().await, 10);
        assert_eq!(limiter.current_token_count().await, 500);
    }

    #[tokio::test]
    async fn test_rate_limiter_record_tokens() {
        let limiter = OpenAIRateLimiter::new(100, 100000);
        limiter.acquire(50).await;
        limiter.record_tokens(100).await;

        // Token count should include both estimated and recorded
        let count = limiter.current_token_count().await;
        assert!(count >= 100);
    }

    // =========================================================================
    // INTEGRATION PATTERN TESTS (mock-based, no real API calls)
    // =========================================================================

    #[test]
    fn test_full_workflow_types() {
        // Test that all types work together correctly
        let messages = vec![
            OpenAIMessage::system("You are a helpful assistant."),
            OpenAIMessage::user("Hello!"),
        ];

        let request = OpenAIRequest::new("gpt-4o", messages.clone())
            .with_max_tokens(100)
            .with_temperature(0.7);

        assert_eq!(request.model, "gpt-4o");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));

        // Serialize and verify
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"gpt-4o\""));
        assert!(json.contains("\"max_tokens\":100"));
    }

    #[test]
    fn test_cost_calculation_consistency() {
        // Verify cost calculations are consistent across methods
        let model = OpenAIModel::Gpt4o;
        let prompt_tokens = 1000;
        let completion_tokens = 500;

        let cost1 = model.estimate_cost(prompt_tokens, completion_tokens);

        let input_cost = (prompt_tokens as f64 / 1000.0) * model.input_cost_per_1k();
        let output_cost = (completion_tokens as f64 / 1000.0) * model.output_cost_per_1k();
        let cost2 = input_cost + output_cost;

        assert!((cost1 - cost2).abs() < 0.0001);
    }

    #[test]
    fn test_model_info_consistency() {
        // All models should have valid info
        for model in OpenAIModel::all() {
            assert!(!model.as_str().is_empty());
            assert!(model.context_window() > 0);
            assert!(model.max_output_tokens() > 0);
            assert!(model.input_cost_per_1k() > 0.0);
            assert!(model.output_cost_per_1k() > 0.0);
            assert!(model.tokens_per_minute() > 0);
            assert!(model.requests_per_minute() > 0);
        }
    }

    #[test]
    fn test_default_model() {
        // Default model should be Gpt4o
        assert_eq!(OpenAIModel::default(), OpenAIModel::Gpt4o);
    }

    #[test]
    fn test_message_equality() {
        let msg1 = OpenAIMessage::user("Hello");
        let msg2 = OpenAIMessage::user("Hello");
        let msg3 = OpenAIMessage::user("World");

        assert_eq!(msg1, msg2);
        assert_ne!(msg1, msg3);
    }

    #[test]
    fn test_usage_clone() {
        let usage = OpenAIUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        let cloned = usage.clone();
        assert_eq!(usage.prompt_tokens, cloned.prompt_tokens);
        assert_eq!(usage.completion_tokens, cloned.completion_tokens);
        assert_eq!(usage.total_tokens, cloned.total_tokens);
    }
}
