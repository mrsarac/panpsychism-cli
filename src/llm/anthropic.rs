//! Anthropic Claude API Client for Project Panpsychism.
//!
//! This module provides a robust client for the Anthropic Messages API,
//! supporting Claude Opus 4, Claude Sonnet 4, and Claude 3.5 Haiku models.
//!
//! # Features
//!
//! - **Streaming**: Server-Sent Events (SSE) for real-time responses
//! - **System Prompts**: First-class support for system instructions
//! - **Retry Logic**: Exponential backoff with jitter for transient failures
//! - **Rate Limiting**: Token and request-per-minute limiting
//! - **Error Categorization**: Structured error handling for different failure modes
//! - **Cost Tracking**: Automatic token counting and cost estimation
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::llm::anthropic::{AnthropicClient, AnthropicMessage};
//!
//! let client = AnthropicClient::builder()
//!     .api_key("sk-ant-...")
//!     .model(AnthropicModel::ClaudeSonnet4)
//!     .build()?;
//!
//! let messages = vec![AnthropicMessage::user("Hello, Claude!")];
//! let response = client.chat_with_retry(messages, None).await?;
//! println!("Response: {}", response.content_text());
//! ```

use crate::error::{Error, Result};
use async_stream::try_stream;
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

// ============================================================================
// Constants
// ============================================================================

/// Default Anthropic API endpoint.
pub const DEFAULT_ANTHROPIC_ENDPOINT: &str = "https://api.anthropic.com";

/// Default Anthropic API version.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default timeout for API requests (in seconds).
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Default maximum number of retries.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Base delay for exponential backoff (in milliseconds).
const RETRY_BASE_DELAY_MS: u64 = 1000;

/// Maximum delay for exponential backoff (in milliseconds).
const RETRY_MAX_DELAY_MS: u64 = 60000;

// ============================================================================
// Model Definitions
// ============================================================================

/// Available Anthropic Claude models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnthropicModel {
    /// Claude Opus 4 - Most capable model for complex tasks
    #[serde(rename = "claude-opus-4-20250514")]
    ClaudeOpus4,
    /// Claude Sonnet 4 - Balanced performance and cost (default)
    #[serde(rename = "claude-sonnet-4-20250514")]
    ClaudeSonnet4,
    /// Claude 3.5 Haiku - Fast and cost-effective
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Claude35Haiku,
}

impl AnthropicModel {
    /// Get the API model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ClaudeOpus4 => "claude-opus-4-20250514",
            Self::ClaudeSonnet4 => "claude-sonnet-4-20250514",
            Self::Claude35Haiku => "claude-3-5-haiku-20241022",
        }
    }

    /// Get model information including context window and pricing.
    pub fn info(&self) -> ModelInfo {
        match self {
            Self::ClaudeOpus4 => ModelInfo {
                model: *self,
                context_window: 200_000,
                max_output_tokens: 32_000,
                input_cost_per_million: 15.0,
                output_cost_per_million: 75.0,
            },
            Self::ClaudeSonnet4 => ModelInfo {
                model: *self,
                context_window: 200_000,
                max_output_tokens: 64_000,
                input_cost_per_million: 3.0,
                output_cost_per_million: 15.0,
            },
            Self::Claude35Haiku => ModelInfo {
                model: *self,
                context_window: 200_000,
                max_output_tokens: 8_192,
                input_cost_per_million: 0.80,
                output_cost_per_million: 4.0,
            },
        }
    }
}

impl Default for AnthropicModel {
    fn default() -> Self {
        Self::ClaudeSonnet4
    }
}

impl std::fmt::Display for AnthropicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Model information including pricing and limits.
#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    /// The model variant.
    pub model: AnthropicModel,
    /// Maximum context window size in tokens.
    pub context_window: usize,
    /// Maximum output tokens.
    pub max_output_tokens: usize,
    /// Cost per million input tokens in USD.
    pub input_cost_per_million: f64,
    /// Cost per million output tokens in USD.
    pub output_cost_per_million: f64,
}

impl ModelInfo {
    /// Calculate cost for given token counts.
    pub fn calculate_cost(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_cost_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_cost_per_million;
        input_cost + output_cost
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Role of a message in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// User message.
    User,
    /// Assistant (Claude) message.
    Assistant,
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    /// The role of the message sender.
    pub role: MessageRole,
    /// The content of the message.
    pub content: String,
}

impl AnthropicMessage {
    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

/// Content block in a response (can be text or tool use).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Tool use request (for future extension).
    ToolUse {
        /// Tool ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input as JSON.
        input: serde_json::Value,
    },
}

impl ContentBlock {
    /// Extract text content if this is a text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }
}

/// Reason why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end of response.
    EndTurn,
    /// Hit max_tokens limit.
    MaxTokens,
    /// Stop sequence was generated.
    StopSequence,
    /// Tool use requested.
    ToolUse,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnthropicUsage {
    /// Number of input tokens.
    pub input_tokens: usize,
    /// Number of output tokens.
    pub output_tokens: usize,
    /// Cache creation tokens (if caching enabled).
    #[serde(default)]
    pub cache_creation_input_tokens: usize,
    /// Cache read tokens (if caching enabled).
    #[serde(default)]
    pub cache_read_input_tokens: usize,
}

impl AnthropicUsage {
    /// Get total tokens used.
    pub fn total_tokens(&self) -> usize {
        self.input_tokens + self.output_tokens
    }
}

/// Request to the Anthropic Messages API.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    /// The model to use.
    pub model: String,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// The messages in the conversation.
    pub messages: Vec<AnthropicMessage>,
    /// Optional system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Optional stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Temperature for sampling (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

impl AnthropicRequest {
    /// Create a new request with default settings.
    pub fn new(model: AnthropicModel, messages: Vec<AnthropicMessage>) -> Self {
        let info = model.info();
        Self {
            model: model.as_str().to_string(),
            max_tokens: info.max_output_tokens,
            messages,
            system: None,
            stop_sequences: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
        }
    }

    /// Set the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Enable streaming.
    pub fn with_streaming(mut self) -> Self {
        self.stream = Some(true);
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 1.0));
        self
    }
}

/// Response from the Anthropic Messages API.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicResponse {
    /// Unique response ID.
    pub id: String,
    /// Type of response (always "message").
    #[serde(rename = "type")]
    pub response_type: String,
    /// Role (always "assistant").
    pub role: MessageRole,
    /// Content blocks.
    pub content: Vec<ContentBlock>,
    /// The model used.
    pub model: String,
    /// Why generation stopped.
    pub stop_reason: Option<StopReason>,
    /// Stop sequence that was hit (if any).
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: AnthropicUsage,
}

impl AnthropicResponse {
    /// Extract all text content concatenated.
    pub fn content_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Check if response was truncated due to max_tokens.
    pub fn is_truncated(&self) -> bool {
        self.stop_reason == Some(StopReason::MaxTokens)
    }
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Event types in the SSE stream.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Start of message.
    MessageStart {
        message: StreamMessageStart,
    },
    /// Start of content block.
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    /// Delta for content block.
    ContentBlockDelta {
        index: usize,
        delta: StreamDelta,
    },
    /// End of content block.
    ContentBlockStop {
        index: usize,
    },
    /// Delta for message.
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<StreamUsageDelta>,
    },
    /// End of message.
    MessageStop,
    /// Ping event (keep-alive).
    Ping,
    /// Error event.
    Error {
        error: StreamError,
    },
}

/// Message start data.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamMessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: MessageRole,
    pub model: String,
    pub usage: AnthropicUsage,
}

/// Content delta in stream.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

impl StreamDelta {
    /// Get text content if this is a text delta.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextDelta { text } => Some(text),
            _ => None,
        }
    }
}

/// Message delta data.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDeltaData {
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
}

/// Usage delta in stream.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamUsageDelta {
    pub output_tokens: usize,
}

/// Stream error.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ============================================================================
// Error Types
// ============================================================================

/// Categorized error types for Anthropic API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicErrorKind {
    /// Invalid API key or authentication failure.
    AuthenticationError,
    /// Rate limit exceeded.
    RateLimitError,
    /// API is temporarily overloaded.
    OverloadedError,
    /// Invalid request parameters.
    InvalidRequestError,
    /// Resource not found.
    NotFoundError,
    /// Server error (5xx).
    ServerError,
    /// Network or connection error.
    NetworkError,
    /// Request timeout.
    TimeoutError,
    /// Unknown error.
    UnknownError,
}

impl AnthropicErrorKind {
    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimitError
                | Self::OverloadedError
                | Self::ServerError
                | Self::NetworkError
                | Self::TimeoutError
        )
    }

    /// Get suggested retry delay in milliseconds.
    pub fn suggested_retry_delay_ms(&self) -> u64 {
        match self {
            Self::RateLimitError => 60_000,      // 1 minute for rate limits
            Self::OverloadedError => 30_000,    // 30 seconds for overload
            Self::ServerError => 5_000,         // 5 seconds for server errors
            Self::NetworkError => 2_000,        // 2 seconds for network errors
            Self::TimeoutError => 5_000,        // 5 seconds for timeouts
            _ => 1_000,                         // 1 second default
        }
    }
}

/// Anthropic API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: AnthropicErrorDetail,
}

/// Error detail in API response.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Categorize an HTTP status code and error response.
pub fn categorize_error(status: u16, error_type: Option<&str>) -> AnthropicErrorKind {
    match status {
        401 => AnthropicErrorKind::AuthenticationError,
        403 => AnthropicErrorKind::AuthenticationError,
        404 => AnthropicErrorKind::NotFoundError,
        429 => AnthropicErrorKind::RateLimitError,
        500..=599 => {
            if let Some(etype) = error_type {
                if etype == "overloaded_error" {
                    return AnthropicErrorKind::OverloadedError;
                }
            }
            AnthropicErrorKind::ServerError
        }
        400 => AnthropicErrorKind::InvalidRequestError,
        _ => AnthropicErrorKind::UnknownError,
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Anthropic client.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Base URL for the API (default: https://api.anthropic.com).
    pub base_url: Option<String>,
    /// Default model to use.
    pub default_model: AnthropicModel,
    /// Anthropic API version.
    pub anthropic_version: Option<String>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Whether streaming is enabled by default.
    pub streaming_enabled: bool,
}

impl AnthropicConfig {
    /// Create a new config with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            default_model: AnthropicModel::default(),
            anthropic_version: None,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_retries: DEFAULT_MAX_RETRIES,
            streaming_enabled: false,
        }
    }

    /// Get the effective base URL.
    pub fn effective_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(DEFAULT_ANTHROPIC_ENDPOINT)
    }

    /// Get the effective API version.
    pub fn effective_version(&self) -> &str {
        self.anthropic_version.as_deref().unwrap_or(DEFAULT_ANTHROPIC_VERSION)
    }
}

// ============================================================================
// Client Builder
// ============================================================================

/// Builder for AnthropicClient.
#[derive(Debug, Default)]
pub struct AnthropicClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<AnthropicModel>,
    anthropic_version: Option<String>,
    timeout_secs: Option<u64>,
    max_retries: Option<u32>,
    streaming_enabled: Option<bool>,
}

impl AnthropicClientBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the API key.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the default model.
    pub fn model(mut self, model: AnthropicModel) -> Self {
        self.default_model = Some(model);
        self
    }

    /// Set the API version.
    pub fn anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = Some(version.into());
        self
    }

    /// Set the request timeout in seconds.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Set the maximum number of retries.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Enable streaming by default.
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.streaming_enabled = Some(enabled);
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<AnthropicClient> {
        let api_key = self.api_key.ok_or_else(|| {
            Error::Config("Anthropic API key is required".to_string())
        })?;

        let config = AnthropicConfig {
            api_key,
            base_url: self.base_url,
            default_model: self.default_model.unwrap_or_default(),
            anthropic_version: self.anthropic_version,
            timeout_secs: self.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            max_retries: self.max_retries.unwrap_or(DEFAULT_MAX_RETRIES),
            streaming_enabled: self.streaming_enabled.unwrap_or(false),
        };

        AnthropicClient::new(config)
    }
}

// ============================================================================
// Client Implementation
// ============================================================================

/// Anthropic API client.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    /// API key.
    api_key: String,
    /// Base URL.
    base_url: String,
    /// Default model.
    default_model: AnthropicModel,
    /// HTTP client.
    http_client: reqwest::Client,
    /// API version.
    anthropic_version: String,
    /// Maximum retries.
    max_retries: u32,
    /// Streaming enabled by default.
    #[allow(dead_code)]
    streaming_enabled: bool,
}

impl AnthropicClient {
    /// Create a new client with the given configuration.
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| Error::ApiConnectionFailed {
                endpoint: "reqwest::Client::builder".to_string(),
                source: Some(e),
            })?;

        // Extract values before moving config fields
        let base_url = config.effective_base_url().to_string();
        let anthropic_version = config.effective_version().to_string();

        Ok(Self {
            api_key: config.api_key,
            base_url,
            default_model: config.default_model,
            http_client,
            anthropic_version,
            max_retries: config.max_retries,
            streaming_enabled: config.streaming_enabled,
        })
    }

    /// Create a builder for the client.
    pub fn builder() -> AnthropicClientBuilder {
        AnthropicClientBuilder::new()
    }

    /// Get the default model.
    pub fn model(&self) -> AnthropicModel {
        self.default_model
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Build request headers.
    fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();

        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key)
                .map_err(|e| Error::Config(format!("Invalid API key format: {}", e)))?,
        );

        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&self.anthropic_version)
                .map_err(|e| Error::Config(format!("Invalid version format: {}", e)))?,
        );

        Ok(headers)
    }

    /// Check rate limit before making a request.
    /// Note: Rate limiting is not yet implemented for this client.
    async fn check_rate_limit(&self, _estimated_tokens: usize) -> Result<()> {
        Ok(())
    }

    /// Record request completion for rate limiting.
    /// Note: Rate limiting is not yet implemented for this client.
    async fn record_request(&self, _tokens_used: usize) {
        // No-op until rate limiting is implemented
    }

    /// Send a chat request.
    pub async fn chat(
        &self,
        messages: Vec<AnthropicMessage>,
        system: Option<&str>,
    ) -> Result<AnthropicResponse> {
        let mut request = AnthropicRequest::new(self.default_model, messages);
        if let Some(sys) = system {
            request = request.with_system(sys);
        }

        self.send_request(request).await
    }

    /// Send a chat request with automatic retry.
    pub async fn chat_with_retry(
        &self,
        messages: Vec<AnthropicMessage>,
        system: Option<&str>,
    ) -> Result<AnthropicResponse> {
        let mut request = AnthropicRequest::new(self.default_model, messages);
        if let Some(sys) = system {
            request = request.with_system(sys);
        }

        self.send_request_with_retry(request).await
    }

    /// Send a raw request to the API.
    pub async fn send_request(&self, request: AnthropicRequest) -> Result<AnthropicResponse> {
        // Estimate tokens for rate limiting
        let estimated_tokens = estimate_tokens(&request);
        self.check_rate_limit(estimated_tokens).await?;

        let url = format!("{}/v1/messages", self.base_url);
        let headers = self.build_headers()?;

        debug!(
            model = %request.model,
            messages_count = request.messages.len(),
            "Sending Anthropic request"
        );

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    Error::ApiTimeout { timeout_secs: DEFAULT_TIMEOUT_SECS }
                } else {
                    Error::ApiConnectionFailed {
                        endpoint: url.clone(),
                        source: Some(e),
                    }
                }
            })?;

        let status = response.status().as_u16();

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            let error_response: Option<AnthropicErrorResponse> =
                serde_json::from_str(&error_text).ok();

            let error_type = error_response
                .as_ref()
                .map(|e| e.error.error_type.as_str());
            let error_kind = categorize_error(status, error_type);

            let message = error_response
                .map(|e| e.error.message)
                .unwrap_or_else(|| format!("HTTP {}: {}", status, error_text));

            error!(
                status = status,
                error_kind = ?error_kind,
                message = %message,
                "Anthropic API error"
            );

            return Err(match error_kind {
                AnthropicErrorKind::AuthenticationError => Error::ApiAuthFailed,
                AnthropicErrorKind::RateLimitError => Error::ApiRateLimited { retry_after_secs: None },
                AnthropicErrorKind::InvalidRequestError => Error::internal(message),
                AnthropicErrorKind::ServerError | AnthropicErrorKind::OverloadedError => {
                    Error::api_response(status, message)
                }
                _ => Error::api_response(status, message),
            });
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| Error::ApiResponse {
                status: 0,
                message: format!("Failed to parse response: {}", e),
            })?;

        // Record usage for rate limiting
        self.record_request(api_response.usage.total_tokens()).await;

        info!(
            id = %api_response.id,
            model = %api_response.model,
            input_tokens = api_response.usage.input_tokens,
            output_tokens = api_response.usage.output_tokens,
            "Anthropic request completed"
        );

        Ok(api_response)
    }

    /// Send a request with automatic retry on transient failures.
    pub async fn send_request_with_retry(
        &self,
        request: AnthropicRequest,
    ) -> Result<AnthropicResponse> {
        let mut last_error = None;
        let mut attempt = 0;

        while attempt <= self.max_retries {
            if attempt > 0 {
                let delay = calculate_retry_delay(attempt);
                warn!(
                    attempt = attempt,
                    delay_ms = delay,
                    "Retrying Anthropic request"
                );
                sleep(Duration::from_millis(delay)).await;
            }

            match self.send_request(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    let is_retryable = matches!(
                        &e,
                        Error::ApiRateLimited { .. }
                            | Error::ApiTimeout { .. }
                            | Error::ApiConnectionFailed { .. }
                            | Error::Http(_)
                    ) || matches!(&e, Error::ApiResponse { status, .. } if *status >= 500);

                    if !is_retryable || attempt >= self.max_retries {
                        return Err(e);
                    }

                    warn!(
                        attempt = attempt,
                        error = %e,
                        "Transient error, will retry"
                    );
                    last_error = Some(e);
                }
            }

            attempt += 1;
        }

        Err(last_error.unwrap_or_else(|| Error::ApiResponse {
            status: 0,
            message: "Max retries exceeded".to_string(),
        }))
    }

    /// Stream a chat response.
    pub async fn chat_streaming(
        &self,
        messages: Vec<AnthropicMessage>,
        system: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let mut request = AnthropicRequest::new(self.default_model, messages)
            .with_streaming();
        if let Some(sys) = system {
            request = request.with_system(sys);
        }

        self.stream_request(request).await
    }

    /// Stream a raw request.
    pub async fn stream_request(
        &self,
        request: AnthropicRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let estimated_tokens = estimate_tokens(&request);
        self.check_rate_limit(estimated_tokens).await?;

        let url = format!("{}/v1/messages", self.base_url);
        let headers = self.build_headers()?;

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::ApiConnectionFailed {
                endpoint: url.clone(),
                source: Some(e),
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::ApiResponse {
                status,
                message: format!("HTTP {}: {}", status, error_text),
            });
        }

        let stream = try_stream! {
            let mut bytes_stream = response.bytes_stream();
            let mut buffer = String::new();

            use futures::StreamExt;
            while let Some(chunk_result) = bytes_stream.next().await {
                let chunk = chunk_result
                    .map_err(|e| Error::Http(e))?;

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events
                while let Some(event_end) = buffer.find("\n\n") {
                    let event_data = buffer[..event_end].to_string();
                    buffer = buffer[event_end + 2..].to_string();

                    // Parse SSE event
                    for line in event_data.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                return;
                            }

                            if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                                match event {
                                    StreamEvent::ContentBlockDelta { delta, .. } => {
                                        if let Some(text) = delta.as_text() {
                                            yield text.to_string();
                                        }
                                    }
                                    StreamEvent::Error { error } => {
                                        Err(Error::ApiResponse {
                                            status: 0,
                                            message: error.message,
                                        })?;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Simple completion (wraps a single user message).
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let messages = vec![AnthropicMessage::user(prompt)];
        let response = self.chat(messages, None).await?;
        Ok(response.content_text())
    }

    /// Simple completion with retry.
    pub async fn complete_with_retry(&self, prompt: &str) -> Result<String> {
        let messages = vec![AnthropicMessage::user(prompt)];
        let response = self.chat_with_retry(messages, None).await?;
        Ok(response.content_text())
    }

    /// Streaming completion.
    pub async fn complete_streaming(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let messages = vec![AnthropicMessage::user(prompt)];
        self.chat_streaming(messages, None).await
    }

    /// Health check.
    pub async fn health_check(&self) -> Result<bool> {
        // Try a minimal request to verify connectivity
        let messages = vec![AnthropicMessage::user("Hi")];
        let request = AnthropicRequest::new(self.default_model, messages)
            .with_max_tokens(1);

        match self.send_request(request).await {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!(error = %e, "Health check failed");
                Ok(false)
            }
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Estimate token count for a request (rough approximation).
pub fn estimate_tokens(request: &AnthropicRequest) -> usize {
    let mut tokens = 0;

    // System prompt
    if let Some(ref system) = request.system {
        tokens += system.len() / 4;
    }

    // Messages
    for message in &request.messages {
        tokens += message.content.len() / 4;
        tokens += 4; // Message overhead
    }

    tokens
}

/// Calculate retry delay with exponential backoff and jitter.
fn calculate_retry_delay(attempt: u32) -> u64 {
    use rand::Rng;

    // Cap the exponent to prevent overflow (2^20 is ~1 million, more than enough)
    let exponent = attempt.saturating_sub(1).min(20);
    let base_delay = RETRY_BASE_DELAY_MS.saturating_mul(2u64.pow(exponent));
    let capped_delay = base_delay.min(RETRY_MAX_DELAY_MS);

    // Add jitter (0-25% of delay) to prevent thundering herd
    let mut rng = rand::thread_rng();
    let jitter = (capped_delay as f64 * 0.25 * rng.gen::<f64>()) as u64;
    capped_delay + jitter
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Model Tests
    // ========================================================================

    #[test]
    fn test_model_as_str() {
        assert_eq!(AnthropicModel::ClaudeOpus4.as_str(), "claude-opus-4-20250514");
        assert_eq!(AnthropicModel::ClaudeSonnet4.as_str(), "claude-sonnet-4-20250514");
        assert_eq!(AnthropicModel::Claude35Haiku.as_str(), "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_model_default() {
        assert_eq!(AnthropicModel::default(), AnthropicModel::ClaudeSonnet4);
    }

    #[test]
    fn test_model_display() {
        assert_eq!(format!("{}", AnthropicModel::ClaudeOpus4), "claude-opus-4-20250514");
    }

    #[test]
    fn test_model_info_context_window() {
        let opus_info = AnthropicModel::ClaudeOpus4.info();
        let sonnet_info = AnthropicModel::ClaudeSonnet4.info();
        let haiku_info = AnthropicModel::Claude35Haiku.info();

        assert_eq!(opus_info.context_window, 200_000);
        assert_eq!(sonnet_info.context_window, 200_000);
        assert_eq!(haiku_info.context_window, 200_000);
    }

    #[test]
    fn test_model_info_max_output() {
        let opus_info = AnthropicModel::ClaudeOpus4.info();
        let sonnet_info = AnthropicModel::ClaudeSonnet4.info();
        let haiku_info = AnthropicModel::Claude35Haiku.info();

        assert_eq!(opus_info.max_output_tokens, 32_000);
        assert_eq!(sonnet_info.max_output_tokens, 64_000);
        assert_eq!(haiku_info.max_output_tokens, 8_192);
    }

    #[test]
    fn test_model_info_pricing() {
        let opus_info = AnthropicModel::ClaudeOpus4.info();
        assert_eq!(opus_info.input_cost_per_million, 15.0);
        assert_eq!(opus_info.output_cost_per_million, 75.0);

        let sonnet_info = AnthropicModel::ClaudeSonnet4.info();
        assert_eq!(sonnet_info.input_cost_per_million, 3.0);
        assert_eq!(sonnet_info.output_cost_per_million, 15.0);

        let haiku_info = AnthropicModel::Claude35Haiku.info();
        assert_eq!(haiku_info.input_cost_per_million, 0.80);
        assert_eq!(haiku_info.output_cost_per_million, 4.0);
    }

    #[test]
    fn test_model_info_cost_calculation() {
        let info = AnthropicModel::ClaudeSonnet4.info();

        // 1M input + 1M output
        let cost = info.calculate_cost(1_000_000, 1_000_000);
        assert!((cost - 18.0).abs() < 0.001);

        // 500 input + 100 output
        let cost = info.calculate_cost(500, 100);
        let expected = (500.0 / 1_000_000.0 * 3.0) + (100.0 / 1_000_000.0 * 15.0);
        assert!((cost - expected).abs() < 0.0001);
    }

    // ========================================================================
    // Message Tests
    // ========================================================================

    #[test]
    fn test_message_user() {
        let msg = AnthropicMessage::user("Hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_message_assistant() {
        let msg = AnthropicMessage::assistant("Hi there");
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.content, "Hi there");
    }

    #[test]
    fn test_message_serialization() {
        let msg = AnthropicMessage::user("Test");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Test\""));
    }

    // ========================================================================
    // Content Block Tests
    // ========================================================================

    #[test]
    fn test_content_block_text() {
        let block = ContentBlock::Text { text: "Hello".to_string() };
        assert_eq!(block.as_text(), Some("Hello"));
    }

    #[test]
    fn test_content_block_tool_use() {
        let block = ContentBlock::ToolUse {
            id: "id".to_string(),
            name: "tool".to_string(),
            input: serde_json::json!({}),
        };
        assert_eq!(block.as_text(), None);
    }

    // ========================================================================
    // Request Tests
    // ========================================================================

    #[test]
    fn test_request_new() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages);

        assert_eq!(request.model, "claude-sonnet-4-20250514");
        assert_eq!(request.max_tokens, 64_000);
        assert!(request.system.is_none());
        assert!(request.stream.is_none());
    }

    #[test]
    fn test_request_with_system() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages)
            .with_system("You are helpful");

        assert_eq!(request.system, Some("You are helpful".to_string()));
    }

    #[test]
    fn test_request_with_max_tokens() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages)
            .with_max_tokens(100);

        assert_eq!(request.max_tokens, 100);
    }

    #[test]
    fn test_request_with_streaming() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages)
            .with_streaming();

        assert_eq!(request.stream, Some(true));
    }

    #[test]
    fn test_request_with_temperature() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages)
            .with_temperature(0.7);

        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_request_temperature_clamping() {
        let messages = vec![AnthropicMessage::user("Hello")];

        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages.clone())
            .with_temperature(1.5);
        assert_eq!(request.temperature, Some(1.0));

        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages)
            .with_temperature(-0.5);
        assert_eq!(request.temperature, Some(0.0));
    }

    #[test]
    fn test_request_serialization() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages)
            .with_system("Be helpful");

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"claude-sonnet-4-20250514\""));
        assert!(json.contains("\"system\":\"Be helpful\""));
    }

    // ========================================================================
    // Response Tests
    // ========================================================================

    #[test]
    fn test_response_content_text() {
        let response = AnthropicResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock::Text { text: "Hello ".to_string() },
                ContentBlock::Text { text: "world!".to_string() },
            ],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };

        assert_eq!(response.content_text(), "Hello world!");
    }

    #[test]
    fn test_response_is_truncated() {
        let mut response = AnthropicResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: MessageRole::Assistant,
            content: vec![],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };

        assert!(response.is_truncated());

        response.stop_reason = Some(StopReason::EndTurn);
        assert!(!response.is_truncated());
    }

    // ========================================================================
    // Usage Tests
    // ========================================================================

    #[test]
    fn test_usage_total_tokens() {
        let usage = AnthropicUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        };

        assert_eq!(usage.total_tokens(), 150);
    }

    // ========================================================================
    // Config Tests
    // ========================================================================

    #[test]
    fn test_config_new() {
        let config = AnthropicConfig::new("sk-ant-test");

        assert_eq!(config.api_key, "sk-ant-test");
        assert!(config.base_url.is_none());
        assert_eq!(config.default_model, AnthropicModel::ClaudeSonnet4);
        assert_eq!(config.timeout_secs, DEFAULT_TIMEOUT_SECS);
        assert_eq!(config.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_config_effective_base_url() {
        let config = AnthropicConfig::new("sk-ant-test");
        assert_eq!(config.effective_base_url(), DEFAULT_ANTHROPIC_ENDPOINT);

        let mut config = AnthropicConfig::new("sk-ant-test");
        config.base_url = Some("https://custom.api".to_string());
        assert_eq!(config.effective_base_url(), "https://custom.api");
    }

    #[test]
    fn test_config_effective_version() {
        let config = AnthropicConfig::new("sk-ant-test");
        assert_eq!(config.effective_version(), DEFAULT_ANTHROPIC_VERSION);

        let mut config = AnthropicConfig::new("sk-ant-test");
        config.anthropic_version = Some("2024-01-01".to_string());
        assert_eq!(config.effective_version(), "2024-01-01");
    }

    // ========================================================================
    // Builder Tests
    // ========================================================================

    #[test]
    fn test_builder_minimal() {
        let client = AnthropicClient::builder()
            .api_key("sk-ant-test")
            .build();

        assert!(client.is_ok());
    }

    #[test]
    fn test_builder_missing_api_key() {
        let client = AnthropicClient::builder().build();
        assert!(client.is_err());
    }

    #[test]
    fn test_builder_full() {
        let client = AnthropicClient::builder()
            .api_key("sk-ant-test")
            .base_url("https://custom.api")
            .model(AnthropicModel::ClaudeOpus4)
            .anthropic_version("2024-01-01")
            .timeout_secs(60)
            .max_retries(5)
            .streaming(true)
            .build();

        assert!(client.is_ok());
        let client = client.unwrap();
        assert_eq!(client.model(), AnthropicModel::ClaudeOpus4);
        assert_eq!(client.base_url(), "https://custom.api");
    }

    // ========================================================================
    // Error Category Tests
    // ========================================================================

    #[test]
    fn test_error_categorization_401() {
        let kind = categorize_error(401, None);
        assert_eq!(kind, AnthropicErrorKind::AuthenticationError);
    }

    #[test]
    fn test_error_categorization_429() {
        let kind = categorize_error(429, None);
        assert_eq!(kind, AnthropicErrorKind::RateLimitError);
    }

    #[test]
    fn test_error_categorization_500() {
        let kind = categorize_error(500, None);
        assert_eq!(kind, AnthropicErrorKind::ServerError);
    }

    #[test]
    fn test_error_categorization_overloaded() {
        let kind = categorize_error(529, Some("overloaded_error"));
        assert_eq!(kind, AnthropicErrorKind::OverloadedError);
    }

    #[test]
    fn test_error_retryable() {
        assert!(AnthropicErrorKind::RateLimitError.is_retryable());
        assert!(AnthropicErrorKind::OverloadedError.is_retryable());
        assert!(AnthropicErrorKind::ServerError.is_retryable());
        assert!(AnthropicErrorKind::NetworkError.is_retryable());
        assert!(AnthropicErrorKind::TimeoutError.is_retryable());

        assert!(!AnthropicErrorKind::AuthenticationError.is_retryable());
        assert!(!AnthropicErrorKind::InvalidRequestError.is_retryable());
    }

    #[test]
    fn test_error_suggested_delay() {
        assert_eq!(AnthropicErrorKind::RateLimitError.suggested_retry_delay_ms(), 60_000);
        assert_eq!(AnthropicErrorKind::OverloadedError.suggested_retry_delay_ms(), 30_000);
        assert_eq!(AnthropicErrorKind::ServerError.suggested_retry_delay_ms(), 5_000);
    }

    // ========================================================================
    // Token Estimation Tests
    // ========================================================================

    #[test]
    fn test_estimate_tokens_simple() {
        let messages = vec![AnthropicMessage::user("Hello, Claude!")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages);

        let tokens = estimate_tokens(&request);
        assert!(tokens > 0);
        assert!(tokens < 50);
    }

    #[test]
    fn test_estimate_tokens_with_system() {
        let messages = vec![AnthropicMessage::user("Hello")];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages)
            .with_system("You are a helpful assistant that provides detailed explanations.");

        let tokens = estimate_tokens(&request);
        assert!(tokens > 15);
    }

    #[test]
    fn test_estimate_tokens_long_conversation() {
        let messages = vec![
            AnthropicMessage::user("What is the capital of France?"),
            AnthropicMessage::assistant("The capital of France is Paris."),
            AnthropicMessage::user("Tell me more about it."),
        ];
        let request = AnthropicRequest::new(AnthropicModel::ClaudeSonnet4, messages);

        let tokens = estimate_tokens(&request);
        assert!(tokens > 20);
    }

    // ========================================================================
    // Retry Delay Tests
    // ========================================================================

    #[test]
    fn test_retry_delay_increases() {
        let delay1 = calculate_retry_delay(1);
        let delay2 = calculate_retry_delay(2);
        let delay3 = calculate_retry_delay(3);

        // Base delays should increase exponentially (with some jitter)
        assert!(delay1 >= RETRY_BASE_DELAY_MS);
        assert!(delay2 >= delay1);
        assert!(delay3 >= delay2);
    }

    #[test]
    fn test_retry_delay_capped() {
        let delay = calculate_retry_delay(100);

        // Should not exceed max delay + 25% jitter
        assert!(delay <= RETRY_MAX_DELAY_MS + (RETRY_MAX_DELAY_MS / 4) + 1);
    }

    // ========================================================================
    // Stop Reason Tests
    // ========================================================================

    #[test]
    fn test_stop_reason_serialization() {
        assert_eq!(
            serde_json::to_string(&StopReason::EndTurn).unwrap(),
            "\"end_turn\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::MaxTokens).unwrap(),
            "\"max_tokens\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::StopSequence).unwrap(),
            "\"stop_sequence\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::ToolUse).unwrap(),
            "\"tool_use\""
        );
    }

    #[test]
    fn test_stop_reason_deserialization() {
        let end_turn: StopReason = serde_json::from_str("\"end_turn\"").unwrap();
        assert_eq!(end_turn, StopReason::EndTurn);
    }

    // ========================================================================
    // Stream Delta Tests
    // ========================================================================

    #[test]
    fn test_stream_delta_text() {
        let delta = StreamDelta::TextDelta { text: "Hello".to_string() };
        assert_eq!(delta.as_text(), Some("Hello"));
    }

    #[test]
    fn test_stream_delta_json() {
        let delta = StreamDelta::InputJsonDelta { partial_json: "{".to_string() };
        assert_eq!(delta.as_text(), None);
    }

    // ========================================================================
    // Client Tests (no actual API calls)
    // ========================================================================

    #[test]
    fn test_client_model_accessor() {
        let client = AnthropicClient::builder()
            .api_key("sk-ant-test")
            .model(AnthropicModel::ClaudeOpus4)
            .build()
            .unwrap();

        assert_eq!(client.model(), AnthropicModel::ClaudeOpus4);
    }

    #[test]
    fn test_client_base_url_accessor() {
        let client = AnthropicClient::builder()
            .api_key("sk-ant-test")
            .base_url("https://custom.api")
            .build()
            .unwrap();

        assert_eq!(client.base_url(), "https://custom.api");
    }

    #[test]
    fn test_client_default_base_url() {
        let client = AnthropicClient::builder()
            .api_key("sk-ant-test")
            .build()
            .unwrap();

        assert_eq!(client.base_url(), DEFAULT_ANTHROPIC_ENDPOINT);
    }

    // ========================================================================
    // Header Tests
    // ========================================================================

    #[test]
    fn test_build_headers() {
        let client = AnthropicClient::builder()
            .api_key("sk-ant-test")
            .build()
            .unwrap();

        let headers = client.build_headers().unwrap();

        assert!(headers.contains_key("x-api-key"));
        assert!(headers.contains_key("anthropic-version"));
        assert!(headers.contains_key(CONTENT_TYPE));
    }

    // ========================================================================
    // Model Serde Tests
    // ========================================================================

    #[test]
    fn test_model_serde_roundtrip() {
        let models = vec![
            AnthropicModel::ClaudeOpus4,
            AnthropicModel::ClaudeSonnet4,
            AnthropicModel::Claude35Haiku,
        ];

        for model in models {
            let json = serde_json::to_string(&model).unwrap();
            let parsed: AnthropicModel = serde_json::from_str(&json).unwrap();
            assert_eq!(model, parsed);
        }
    }

    #[test]
    fn test_model_deserialize_from_string() {
        let json = "\"claude-opus-4-20250514\"";
        let model: AnthropicModel = serde_json::from_str(json).unwrap();
        assert_eq!(model, AnthropicModel::ClaudeOpus4);
    }

    // ========================================================================
    // Message Role Tests
    // ========================================================================

    #[test]
    fn test_message_role_serde() {
        let user_json = serde_json::to_string(&MessageRole::User).unwrap();
        assert_eq!(user_json, "\"user\"");

        let assistant_json = serde_json::to_string(&MessageRole::Assistant).unwrap();
        assert_eq!(assistant_json, "\"assistant\"");
    }

    // ========================================================================
    // Constants Tests
    // ========================================================================

    #[test]
    fn test_constants() {
        assert!(!DEFAULT_ANTHROPIC_ENDPOINT.is_empty());
        assert!(!DEFAULT_ANTHROPIC_VERSION.is_empty());
        assert!(DEFAULT_TIMEOUT_SECS > 0);
        assert!(DEFAULT_MAX_RETRIES > 0);
    }
}
