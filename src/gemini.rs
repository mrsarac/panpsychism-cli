//! Gemini API client module for Project Panpsychism.
//!
//! Provides integration with Antigravity proxy for Gemini API access.
//! Default endpoint: http://127.0.0.1:8045 (OpenAI-compatible)
//!
//! # Features
//!
//! - **Retry Logic**: Exponential backoff with configurable max retries
//! - **Rate Limiting**: Configurable requests per minute
//! - **Model Selection**: Flash, Pro, and Thinking variants
//! - **Streaming**: Token-by-token response streaming
//! - **Statistics**: Usage tracking for requests and tokens
//! - **Health Check**: Endpoint availability verification
//! - **Antigravity Fallback**: Automatic fallback to direct Gemini API
//!
//! # Environment Variables
//!
//! - `ANTIGRAVITY_ENDPOINT`: Override Antigravity proxy endpoint (default: http://127.0.0.1:8045)
//! - `ANTIGRAVITY_API_KEY`: Override Antigravity API key (default: sk-antigravity)
//! - `ANTIGRAVITY_ENABLED`: Enable/disable Antigravity proxy (default: true)
//! - `GEMINI_API_KEY`: Direct Gemini API key for fallback

use crate::{Error, Result};

// Re-export constants for backward compatibility (source of truth is crate::constants)
pub use crate::constants::{
    DEFAULT_API_KEY, DEFAULT_ENDPOINT, DEFAULT_MAX_RETRIES, DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_TIMEOUT_SECS,
};
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
// ANTIGRAVITY CONFIGURATION
// =============================================================================

/// Configuration for Antigravity proxy connection.
///
/// Antigravity is a local proxy that provides free Google API quota for Gemini.
/// It runs at http://127.0.0.1:8045 and exposes an OpenAI-compatible API.
///
/// # Environment Variables
///
/// - `ANTIGRAVITY_ENDPOINT`: Override the default endpoint
/// - `ANTIGRAVITY_API_KEY`: Override the default API key
/// - `ANTIGRAVITY_ENABLED`: Set to "false" to disable Antigravity
///
/// # Example
///
/// ```rust
/// use panpsychism::gemini::AntigravityConfig;
///
/// let config = AntigravityConfig::default();
/// assert_eq!(config.endpoint, "http://127.0.0.1:8045");
/// assert_eq!(config.api_key, "sk-antigravity");
/// ```
#[derive(Debug, Clone)]
pub struct AntigravityConfig {
    /// Antigravity proxy endpoint (default: http://127.0.0.1:8045)
    pub endpoint: String,
    /// API key for authentication (default: sk-antigravity)
    pub api_key: String,
    /// Model to use (default: gemini-3-flash)
    pub model: String,
    /// Request timeout in seconds (default: 30)
    pub timeout_secs: u64,
}

impl Default for AntigravityConfig {
    fn default() -> Self {
        Self {
            endpoint: env::var("ANTIGRAVITY_ENDPOINT")
                .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string()),
            api_key: env::var("ANTIGRAVITY_API_KEY")
                .unwrap_or_else(|_| DEFAULT_API_KEY.to_string()),
            model: crate::constants::DEFAULT_MODEL.to_string(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }
}

impl AntigravityConfig {
    /// Create a new configuration with custom endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Create a new configuration with custom API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Create a new configuration with custom model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Create a new configuration with custom timeout.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
}

/// Check if Antigravity proxy is enabled via environment variable.
///
/// Returns `true` if `ANTIGRAVITY_ENABLED` is not set or is set to "true"/"1".
/// Returns `false` if set to "false"/"0".
///
/// # Example
///
/// ```rust
/// use panpsychism::gemini::is_antigravity_enabled;
///
/// // Default is enabled
/// // std::env::remove_var("ANTIGRAVITY_ENABLED");
/// // assert!(is_antigravity_enabled());
/// ```
pub fn is_antigravity_enabled() -> bool {
    match env::var("ANTIGRAVITY_ENABLED") {
        Ok(val) => !matches!(val.to_lowercase().as_str(), "false" | "0" | "no" | "off"),
        Err(_) => true, // Default: enabled
    }
}

/// Check if Antigravity proxy is available and responding.
///
/// Makes a GET request to the `/v1/models` endpoint to verify the proxy is running.
/// Returns `true` if the proxy responds with HTTP 200.
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::gemini::is_antigravity_available;
///
/// if is_antigravity_available().await {
///     println!("Antigravity proxy is running");
/// } else {
///     println!("Falling back to direct Gemini API");
/// }
/// ```
pub async fn is_antigravity_available() -> bool {
    let config = AntigravityConfig::default();
    is_antigravity_available_with_config(&config).await
}

/// Check if Antigravity proxy is available with custom configuration.
///
/// Makes a GET request to the `/v1/models` endpoint with the specified config.
pub async fn is_antigravity_available_with_config(config: &AntigravityConfig) -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    let url = format!("{}/v1/models", config.endpoint);

    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .send()
        .await;

    match response {
        Ok(resp) => {
            let is_ok = resp.status().is_success();
            if is_ok {
                debug!("Antigravity proxy available at {}", config.endpoint);
            } else {
                debug!(
                    "Antigravity proxy returned status {} at {}",
                    resp.status(),
                    config.endpoint
                );
            }
            is_ok
        }
        Err(e) => {
            debug!("Antigravity proxy unavailable: {}", e);
            false
        }
    }
}

/// Direct Gemini API endpoint for fallback.
pub const GEMINI_DIRECT_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Get the direct Gemini API key from environment.
///
/// Reads from `GEMINI_API_KEY` environment variable.
pub fn get_gemini_api_key() -> Option<String> {
    env::var("GEMINI_API_KEY").ok()
}

/// Available Gemini model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeminiModel {
    /// Fast model for quick responses (gemini-2.0-flash)
    #[default]
    Flash,
    /// Advanced model for complex tasks (gemini-pro)
    Pro,
    /// Thinking model with chain-of-thought (gemini-2.0-flash-thinking)
    ProThinking,
    /// Gemini 3 Flash (gemini-3-flash) - default via Antigravity
    Flash3,
    /// Gemini 3 Pro High quality (gemini-3-pro-high)
    Pro3High,
}

impl GeminiModel {
    /// Get the model identifier string for API requests.
    pub fn as_str(&self) -> &'static str {
        match self {
            GeminiModel::Flash => "gemini-2.0-flash",
            GeminiModel::Pro => "gemini-pro",
            GeminiModel::ProThinking => "gemini-2.0-flash-thinking",
            GeminiModel::Flash3 => "gemini-3-flash",
            GeminiModel::Pro3High => "gemini-3-pro-high",
        }
    }
}

impl std::fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for GeminiModel {
    type Err = crate::Error;

    /// Parse a model from a string identifier.
    ///
    /// Accepts both short names (e.g., "flash", "pro") and full identifiers
    /// (e.g., "gemini-3-flash", "gemini-pro").
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::str::FromStr;
    /// use panpsychism::gemini::GeminiModel;
    ///
    /// let model: GeminiModel = "flash".parse().unwrap();
    /// assert_eq!(model, GeminiModel::Flash);
    ///
    /// let model: GeminiModel = "gemini-3-pro-high".parse().unwrap();
    /// assert_eq!(model, GeminiModel::Pro3High);
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            // Short names
            "flash" | "flash2" | "2.0-flash" => Ok(GeminiModel::Flash),
            "pro" => Ok(GeminiModel::Pro),
            "thinking" | "pro-thinking" | "flash-thinking" => Ok(GeminiModel::ProThinking),
            "flash3" | "3-flash" => Ok(GeminiModel::Flash3),
            "pro3" | "pro3-high" | "3-pro-high" => Ok(GeminiModel::Pro3High),
            // Full identifiers
            "gemini-2.0-flash" => Ok(GeminiModel::Flash),
            "gemini-pro" => Ok(GeminiModel::Pro),
            "gemini-2.0-flash-thinking" => Ok(GeminiModel::ProThinking),
            "gemini-3-flash" => Ok(GeminiModel::Flash3),
            "gemini-3-pro-high" => Ok(GeminiModel::Pro3High),
            _ => Err(crate::Error::Config(format!(
                "Unknown Gemini model: '{}'. Valid models: flash, pro, thinking, flash3, pro3-high",
                s
            ))),
        }
    }
}

impl GeminiModel {
    /// Parse a model from a string identifier (infallible version).
    ///
    /// Returns the matching model variant or defaults to Flash3 if unknown.
    /// For fallible parsing with error messages, use the `FromStr` trait.
    pub fn from_str_lossy(s: &str) -> Self {
        s.parse().unwrap_or(GeminiModel::Flash3)
    }
}

/// Rate limiter for API request throttling.
#[derive(Debug)]
pub struct RateLimiter {
    /// Maximum requests allowed per minute.
    requests_per_minute: u32,
    /// Timestamp of the current rate limit window start.
    window_start: Mutex<Instant>,
    /// Request count in current window.
    request_count: Mutex<u32>,
}

impl RateLimiter {
    /// Create a new rate limiter with the specified requests per minute.
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            window_start: Mutex::new(Instant::now()),
            request_count: Mutex::new(0),
        }
    }

    /// Wait if necessary to respect rate limits.
    /// Returns the time waited, if any.
    pub async fn acquire(&self) -> Duration {
        let mut window_start = self.window_start.lock().await;
        let mut request_count = self.request_count.lock().await;

        let now = Instant::now();
        let elapsed = now.duration_since(*window_start);

        // Reset window if a minute has passed
        if elapsed >= Duration::from_secs(60) {
            *window_start = now;
            *request_count = 0;
        }

        // Check if we need to wait
        if *request_count >= self.requests_per_minute {
            let wait_time = Duration::from_secs(60) - elapsed;
            if !wait_time.is_zero() {
                debug!("Rate limit reached, waiting {:?}", wait_time);
                drop(window_start);
                drop(request_count);
                tokio::time::sleep(wait_time).await;

                // Reset after waiting
                let mut window_start = self.window_start.lock().await;
                let mut request_count = self.request_count.lock().await;
                *window_start = Instant::now();
                *request_count = 1;
                return wait_time;
            }
        }

        *request_count += 1;
        Duration::ZERO
    }

    /// Get current request count in the window.
    pub async fn current_count(&self) -> u32 {
        *self.request_count.lock().await
    }
}

/// Usage statistics for tracking API consumption.
#[derive(Debug, Default)]
pub struct UsageStats {
    /// Total number of API requests made.
    pub total_requests: AtomicU64,
    /// Number of successful requests.
    pub successful_requests: AtomicU64,
    /// Number of failed requests.
    pub failed_requests: AtomicU64,
    /// Total input tokens consumed.
    pub total_input_tokens: AtomicU64,
    /// Total output tokens generated.
    pub total_output_tokens: AtomicU64,
}

impl UsageStats {
    /// Create a new empty stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request with token usage.
    pub fn record_success(&self, input_tokens: u64, output_tokens: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_input_tokens
            .fetch_add(input_tokens, Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(output_tokens, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current statistics.
    pub fn snapshot(&self) -> UsageStatsSnapshot {
        UsageStatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            total_input_tokens: self.total_input_tokens.load(Ordering::Relaxed),
            total_output_tokens: self.total_output_tokens.load(Ordering::Relaxed),
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.total_input_tokens.store(0, Ordering::Relaxed);
        self.total_output_tokens.store(0, Ordering::Relaxed);
    }
}

/// A point-in-time snapshot of usage statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct UsageStatsSnapshot {
    /// Total number of API requests made.
    pub total_requests: u64,
    /// Number of successful requests.
    pub successful_requests: u64,
    /// Number of failed requests.
    pub failed_requests: u64,
    /// Total input tokens consumed.
    pub total_input_tokens: u64,
    /// Total output tokens generated.
    pub total_output_tokens: u64,
}

impl UsageStatsSnapshot {
    /// Calculate the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            100.0
        } else {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        }
    }

    /// Calculate total tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.total_input_tokens + self.total_output_tokens
    }
}

/// Gemini API client via Antigravity proxy.
///
/// # Features
///
/// - Automatic retry with exponential backoff
/// - Rate limiting to prevent API throttling
/// - Multiple model support
/// - Streaming responses
/// - Usage statistics tracking
/// - Antigravity fallback support
///
/// # Fallback Behavior
///
/// When `fallback_enabled` is true (default), the client will:
/// 1. First try Antigravity proxy if enabled via `ANTIGRAVITY_ENABLED`
/// 2. If Antigravity fails, fall back to direct Gemini API (requires `GEMINI_API_KEY`)
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::gemini::{GeminiClient, GeminiModel};
/// use std::time::Duration;
///
/// let client = GeminiClient::new()
///     .with_model(GeminiModel::Flash3)
///     .with_timeout(Duration::from_secs(60))
///     .with_max_retries(5);
///
/// let response = client.complete_with_retry("Hello, world!").await?;
/// println!("Response: {}", response);
/// ```
#[derive(Debug)]
pub struct GeminiClient {
    /// HTTP client
    client: reqwest::Client,
    /// API endpoint
    endpoint: String,
    /// API key
    api_key: String,
    /// Model to use
    model: GeminiModel,
    /// Request timeout
    timeout: Duration,
    /// Maximum retry attempts
    max_retries: u32,
    /// Rate limiter
    rate_limiter: Arc<RateLimiter>,
    /// Usage statistics
    stats: Arc<UsageStats>,
    /// Fallback endpoint (direct Gemini API)
    fallback_endpoint: Option<String>,
    /// Fallback API key (direct Gemini API)
    fallback_api_key: Option<String>,
    /// Whether fallback is enabled
    fallback_enabled: bool,
}

impl Default for GeminiClient {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GeminiClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            endpoint: self.endpoint.clone(),
            api_key: self.api_key.clone(),
            model: self.model,
            timeout: self.timeout,
            max_retries: self.max_retries,
            rate_limiter: Arc::clone(&self.rate_limiter),
            stats: Arc::clone(&self.stats),
            fallback_endpoint: self.fallback_endpoint.clone(),
            fallback_api_key: self.fallback_api_key.clone(),
            fallback_enabled: self.fallback_enabled,
        }
    }
}

/// Default connection pool size per host.
pub const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 10;

/// Default idle connection timeout in seconds.
pub const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 30;

impl GeminiClient {
    /// Create a new client with default Antigravity settings and connection pooling.
    ///
    /// Connection pooling is enabled by default with:
    /// - `pool_max_idle_per_host`: 10 connections
    /// - `pool_idle_timeout`: 30 seconds
    ///
    /// This improves performance for multiple sequential or parallel API calls
    /// by reusing existing TCP connections instead of establishing new ones.
    ///
    /// # Fallback Support
    ///
    /// By default, fallback to direct Gemini API is enabled. If Antigravity
    /// proxy is unavailable and `GEMINI_API_KEY` is set, the client will
    /// automatically switch to direct API access.
    pub fn new() -> Self {
        let config = AntigravityConfig::default();
        let timeout = Duration::from_secs(config.timeout_secs);
        Self {
            client: reqwest::Client::builder()
                .timeout(timeout)
                .pool_max_idle_per_host(DEFAULT_POOL_MAX_IDLE_PER_HOST)
                .pool_idle_timeout(Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS))
                .build()
                .expect("Failed to create HTTP client"),
            endpoint: config.endpoint,
            api_key: config.api_key,
            model: GeminiModel::Flash3,
            timeout,
            max_retries: DEFAULT_MAX_RETRIES,
            rate_limiter: Arc::new(RateLimiter::new(DEFAULT_REQUESTS_PER_MINUTE)),
            stats: Arc::new(UsageStats::new()),
            fallback_endpoint: Some(GEMINI_DIRECT_ENDPOINT.to_string()),
            fallback_api_key: get_gemini_api_key(),
            fallback_enabled: true,
        }
    }

    /// Create a new client from AntigravityConfig.
    pub fn from_config(config: AntigravityConfig) -> Self {
        let timeout = Duration::from_secs(config.timeout_secs);
        Self {
            client: reqwest::Client::builder()
                .timeout(timeout)
                .pool_max_idle_per_host(DEFAULT_POOL_MAX_IDLE_PER_HOST)
                .pool_idle_timeout(Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS))
                .build()
                .expect("Failed to create HTTP client"),
            endpoint: config.endpoint,
            api_key: config.api_key,
            model: config.model.parse().unwrap_or(GeminiModel::Flash3),
            timeout,
            max_retries: DEFAULT_MAX_RETRIES,
            rate_limiter: Arc::new(RateLimiter::new(DEFAULT_REQUESTS_PER_MINUTE)),
            stats: Arc::new(UsageStats::new()),
            fallback_endpoint: Some(GEMINI_DIRECT_ENDPOINT.to_string()),
            fallback_api_key: get_gemini_api_key(),
            fallback_enabled: true,
        }
    }

    /// Set custom endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Enable or disable fallback to direct Gemini API.
    ///
    /// When enabled (default), if Antigravity proxy fails, the client will
    /// attempt to use direct Gemini API with `GEMINI_API_KEY`.
    pub fn with_fallback(mut self, enabled: bool) -> Self {
        self.fallback_enabled = enabled;
        self
    }

    /// Set custom fallback endpoint.
    pub fn with_fallback_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.fallback_endpoint = Some(endpoint.into());
        self
    }

    /// Set custom fallback API key.
    pub fn with_fallback_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.fallback_api_key = Some(api_key.into());
        self
    }

    /// Check if fallback is available (has API key configured).
    pub fn has_fallback(&self) -> bool {
        self.fallback_enabled && self.fallback_api_key.is_some()
    }

    /// Set custom API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    /// Set the model to use.
    pub fn with_model(mut self, model: GeminiModel) -> Self {
        self.model = model;
        self
    }

    /// Set request timeout while preserving connection pooling settings.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = reqwest::Client::builder()
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

    /// Set rate limit (requests per minute).
    pub fn with_rate_limit(mut self, requests_per_minute: u32) -> Self {
        self.rate_limiter = Arc::new(RateLimiter::new(requests_per_minute));
        self
    }

    /// Get a reference to the usage statistics.
    pub fn stats(&self) -> &Arc<UsageStats> {
        &self.stats
    }

    /// Get a snapshot of current usage statistics.
    pub fn get_stats(&self) -> UsageStatsSnapshot {
        self.stats.snapshot()
    }

    /// Reset usage statistics.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// Check if the API endpoint is healthy and responding.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/v1/models", self.endpoint);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(Duration::from_secs(5))
            .send()
            .await;

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

    /// Send a chat completion request with automatic retry.
    pub async fn chat_with_retry(&self, messages: Vec<Message>) -> Result<ChatResponse> {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            // Apply rate limiting
            let waited = self.rate_limiter.acquire().await;
            if !waited.is_zero() {
                info!("Rate limited, waited {:?}", waited);
            }

            // Apply backoff on retry attempts
            if attempt > 0 {
                let delay = Self::backoff_delay(attempt - 1);
                debug!("Retry attempt {}, waiting {:?}", attempt, delay);
                tokio::time::sleep(delay).await;
            }

            match self.chat_internal(&messages).await {
                Ok(response) => {
                    // Record success
                    if let Some(usage) = &response.usage {
                        self.stats.record_success(
                            usage.prompt_tokens as u64,
                            usage.completion_tokens as u64,
                        );
                    } else {
                        self.stats.record_success(0, 0);
                    }
                    return Ok(response);
                }
                Err(e) => {
                    warn!("Request failed (attempt {}): {}", attempt + 1, e);
                    last_error = Some(e);

                    // Don't retry on certain errors
                    if let Some(Error::Synthesis(msg)) = last_error.as_ref() {
                        if msg.contains("401") || msg.contains("403") {
                            // Auth errors - no point retrying
                            self.stats.record_failure();
                            return Err(last_error.unwrap());
                        }
                    }
                }
            }
        }

        // All retries exhausted
        self.stats.record_failure();
        Err(last_error.unwrap_or_else(|| Error::Synthesis("Max retries exceeded".to_string())))
    }

    /// Internal chat implementation without retry logic.
    async fn chat_internal(&self, messages: &[Message]) -> Result<ChatResponse> {
        let request = ChatRequest {
            model: self.model.as_str().to_string(),
            messages: messages.to_vec(),
            max_tokens: None,
            temperature: None,
            stream: Some(false),
        };

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::Synthesis(format!("API error {}: {}", status, text)));
        }

        Ok(response.json().await?)
    }

    /// Send a chat completion request (no retry).
    pub async fn chat(&self, messages: Vec<Message>) -> Result<ChatResponse> {
        // Apply rate limiting
        self.rate_limiter.acquire().await;

        let result = self.chat_internal(&messages).await;

        match &result {
            Ok(response) => {
                if let Some(usage) = &response.usage {
                    self.stats
                        .record_success(usage.prompt_tokens as u64, usage.completion_tokens as u64);
                } else {
                    self.stats.record_success(0, 0);
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
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let response = self.chat_with_retry(messages).await?;

        response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| Error::Synthesis("No response from API".to_string()))
    }

    /// Simple text completion (no retry).
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
            .ok_or_else(|| Error::Synthesis("No response from API".to_string()))
    }

    /// Stream a chat completion response, yielding tokens as they arrive.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let client = GeminiClient::new();
    /// let messages = vec![Message { role: "user".to_string(), content: "Hello!".to_string() }];
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
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        use futures::StreamExt;

        // Apply rate limiting
        self.rate_limiter.acquire().await;

        let request = ChatRequest {
            model: self.model.as_str().to_string(),
            messages,
            max_tokens: None,
            temperature: None,
            stream: Some(true),
        };

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            self.stats.record_failure();
            return Err(Error::Synthesis(format!("API error {}: {}", status, text)));
        }

        let stats = Arc::clone(&self.stats);
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
                                match serde_json::from_str::<StreamChunk>(json_str) {
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
            stats.record_success(0, total_tokens);
        };

        Ok(Box::pin(stream))
    }

    /// Stream a simple text completion.
    pub async fn complete_streaming(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        self.chat_streaming(messages).await
    }
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

/// Chat completion request.
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Chat completion response.
#[derive(Debug, Deserialize)]

pub struct ChatResponse {
    /// Response choices
    pub choices: Vec<Choice>,
    /// Token usage
    pub usage: Option<Usage>,
}

/// A response choice.
#[derive(Debug, Deserialize)]

pub struct Choice {
    /// Choice index
    pub index: usize,
    /// Response message
    pub message: Message,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// Streaming response chunk.
#[derive(Debug, Deserialize)]

struct StreamChunk {
    choices: Vec<StreamChoice>,
}

/// Streaming response choice.
#[derive(Debug, Deserialize)]

struct StreamChoice {
    delta: StreamDelta,
}

/// Streaming response delta.
#[derive(Debug, Deserialize)]

struct StreamDelta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_model_as_str() {
        assert_eq!(GeminiModel::Flash.as_str(), "gemini-2.0-flash");
        assert_eq!(GeminiModel::Pro.as_str(), "gemini-pro");
        assert_eq!(
            GeminiModel::ProThinking.as_str(),
            "gemini-2.0-flash-thinking"
        );
        assert_eq!(GeminiModel::Flash3.as_str(), "gemini-3-flash");
        assert_eq!(GeminiModel::Pro3High.as_str(), "gemini-3-pro-high");
    }

    #[test]
    fn test_gemini_model_display() {
        assert_eq!(format!("{}", GeminiModel::Flash), "gemini-2.0-flash");
        assert_eq!(format!("{}", GeminiModel::Flash3), "gemini-3-flash");
    }

    #[test]
    fn test_backoff_delay() {
        assert_eq!(GeminiClient::backoff_delay(0), Duration::from_secs(1));
        assert_eq!(GeminiClient::backoff_delay(1), Duration::from_secs(2));
        assert_eq!(GeminiClient::backoff_delay(2), Duration::from_secs(4));
        assert_eq!(GeminiClient::backoff_delay(3), Duration::from_secs(8));
        assert_eq!(GeminiClient::backoff_delay(4), Duration::from_secs(16));
        assert_eq!(GeminiClient::backoff_delay(5), Duration::from_secs(32));
        // Capped at 32 seconds
        assert_eq!(GeminiClient::backoff_delay(6), Duration::from_secs(32));
        assert_eq!(GeminiClient::backoff_delay(100), Duration::from_secs(32));
    }

    #[test]
    fn test_usage_stats() {
        let stats = UsageStats::new();

        stats.record_success(100, 50);
        stats.record_success(200, 100);
        stats.record_failure();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 3);
        assert_eq!(snapshot.successful_requests, 2);
        assert_eq!(snapshot.failed_requests, 1);
        assert_eq!(snapshot.total_input_tokens, 300);
        assert_eq!(snapshot.total_output_tokens, 150);
        assert_eq!(snapshot.total_tokens(), 450);
        assert!((snapshot.success_rate() - 66.666).abs() < 0.01);
    }

    #[test]
    fn test_usage_stats_reset() {
        let stats = UsageStats::new();
        stats.record_success(100, 50);
        stats.reset();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.total_tokens(), 0);
    }

    #[test]
    fn test_usage_stats_snapshot_success_rate_empty() {
        let snapshot = UsageStatsSnapshot::default();
        assert_eq!(snapshot.success_rate(), 100.0);
    }

    #[test]
    fn test_client_builder_pattern() {
        let client = GeminiClient::new()
            .with_endpoint("http://localhost:9000")
            .with_api_key("test-key")
            .with_model(GeminiModel::Pro)
            .with_timeout(Duration::from_secs(60))
            .with_max_retries(5)
            .with_rate_limit(30);

        assert_eq!(client.endpoint, "http://localhost:9000");
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.model, GeminiModel::Pro);
        assert_eq!(client.timeout, Duration::from_secs(60));
        assert_eq!(client.max_retries, 5);
    }

    #[test]
    fn test_client_clone_shares_stats() {
        let client1 = GeminiClient::new();
        let client2 = client1.clone();

        client1.stats.record_success(100, 50);

        // Both clients should share the same stats
        let snapshot1 = client1.get_stats();
        let snapshot2 = client2.get_stats();

        assert_eq!(snapshot1.total_requests, snapshot2.total_requests);
        assert_eq!(snapshot1.successful_requests, snapshot2.successful_requests);
    }

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let limiter = RateLimiter::new(60);

        // First request should not wait
        let waited = limiter.acquire().await;
        assert_eq!(waited, Duration::ZERO);

        assert_eq!(limiter.current_count().await, 1);
    }

    #[tokio::test]
    async fn test_rate_limiter_multiple_requests() {
        let limiter = RateLimiter::new(100);

        for _ in 0..10 {
            let waited = limiter.acquire().await;
            assert_eq!(waited, Duration::ZERO);
        }

        assert_eq!(limiter.current_count().await, 10);
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message {
            role: "user".to_string(),
            content: "Hello, world!".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello, world!\""));
    }

    #[test]
    fn test_chat_response_deserialization() {
        let json = r#"{
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

        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, "Hello!");
        assert_eq!(response.usage.unwrap().total_tokens, 15);
    }
    #[test]
    fn test_gemini_model_from_str_short_names() {
        assert_eq!("flash".parse::<GeminiModel>().unwrap(), GeminiModel::Flash);
        assert_eq!("pro".parse::<GeminiModel>().unwrap(), GeminiModel::Pro);
        assert_eq!(
            "thinking".parse::<GeminiModel>().unwrap(),
            GeminiModel::ProThinking
        );
        assert_eq!(
            "flash3".parse::<GeminiModel>().unwrap(),
            GeminiModel::Flash3
        );
        assert_eq!(
            "pro3-high".parse::<GeminiModel>().unwrap(),
            GeminiModel::Pro3High
        );
    }

    #[test]
    fn test_gemini_model_from_str_full_identifiers() {
        assert_eq!(
            "gemini-2.0-flash".parse::<GeminiModel>().unwrap(),
            GeminiModel::Flash
        );
        assert_eq!(
            "gemini-pro".parse::<GeminiModel>().unwrap(),
            GeminiModel::Pro
        );
        assert_eq!(
            "gemini-2.0-flash-thinking".parse::<GeminiModel>().unwrap(),
            GeminiModel::ProThinking
        );
        assert_eq!(
            "gemini-3-flash".parse::<GeminiModel>().unwrap(),
            GeminiModel::Flash3
        );
        assert_eq!(
            "gemini-3-pro-high".parse::<GeminiModel>().unwrap(),
            GeminiModel::Pro3High
        );
    }

    #[test]
    fn test_gemini_model_from_str_case_insensitive() {
        assert_eq!("FLASH".parse::<GeminiModel>().unwrap(), GeminiModel::Flash);
        assert_eq!(
            "Flash3".parse::<GeminiModel>().unwrap(),
            GeminiModel::Flash3
        );
        assert_eq!(
            "GEMINI-PRO".parse::<GeminiModel>().unwrap(),
            GeminiModel::Pro
        );
    }

    #[test]
    fn test_gemini_model_from_str_invalid() {
        let result = "invalid-model".parse::<GeminiModel>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unknown Gemini model"));
    }

    #[test]
    fn test_gemini_model_from_str_lossy() {
        // Valid model
        assert_eq!(GeminiModel::from_str_lossy("flash"), GeminiModel::Flash);
        // Invalid model falls back to default
        assert_eq!(GeminiModel::from_str_lossy("invalid"), GeminiModel::Flash3);
    }

    // =========================================================================
    // ANTIGRAVITY CONFIGURATION TESTS
    // =========================================================================

    #[test]
    fn test_antigravity_config_default_values() {
        // Temporarily clear env vars for this test
        let saved_endpoint = env::var("ANTIGRAVITY_ENDPOINT").ok();
        let saved_api_key = env::var("ANTIGRAVITY_API_KEY").ok();
        env::remove_var("ANTIGRAVITY_ENDPOINT");
        env::remove_var("ANTIGRAVITY_API_KEY");

        let config = AntigravityConfig::default();

        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(config.api_key, DEFAULT_API_KEY);
        assert_eq!(config.model, crate::constants::DEFAULT_MODEL);
        assert_eq!(config.timeout_secs, DEFAULT_TIMEOUT_SECS);

        // Restore env vars
        if let Some(val) = saved_endpoint {
            env::set_var("ANTIGRAVITY_ENDPOINT", val);
        }
        if let Some(val) = saved_api_key {
            env::set_var("ANTIGRAVITY_API_KEY", val);
        }
    }

    #[test]
    fn test_antigravity_config_builder_pattern() {
        let config = AntigravityConfig::default()
            .with_endpoint("http://custom:9000")
            .with_api_key("custom-key")
            .with_model("gemini-pro")
            .with_timeout(60);

        assert_eq!(config.endpoint, "http://custom:9000");
        assert_eq!(config.api_key, "custom-key");
        assert_eq!(config.model, "gemini-pro");
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn test_is_antigravity_enabled_default() {
        // Clear the env var to test default behavior
        let saved = env::var("ANTIGRAVITY_ENABLED").ok();
        env::remove_var("ANTIGRAVITY_ENABLED");

        assert!(is_antigravity_enabled());

        // Restore
        if let Some(val) = saved {
            env::set_var("ANTIGRAVITY_ENABLED", val);
        }
    }

    #[test]
    fn test_is_antigravity_enabled_false_values() {
        let saved = env::var("ANTIGRAVITY_ENABLED").ok();

        for value in ["false", "False", "FALSE", "0", "no", "off"] {
            env::set_var("ANTIGRAVITY_ENABLED", value);
            assert!(
                !is_antigravity_enabled(),
                "Expected disabled for value: {}",
                value
            );
        }

        // Restore
        if let Some(val) = saved {
            env::set_var("ANTIGRAVITY_ENABLED", val);
        } else {
            env::remove_var("ANTIGRAVITY_ENABLED");
        }
    }

    #[test]
    fn test_is_antigravity_enabled_true_values() {
        let saved = env::var("ANTIGRAVITY_ENABLED").ok();

        for value in ["true", "True", "TRUE", "1", "yes", "on", "anything"] {
            env::set_var("ANTIGRAVITY_ENABLED", value);
            assert!(
                is_antigravity_enabled(),
                "Expected enabled for value: {}",
                value
            );
        }

        // Restore
        if let Some(val) = saved {
            env::set_var("ANTIGRAVITY_ENABLED", val);
        } else {
            env::remove_var("ANTIGRAVITY_ENABLED");
        }
    }

    #[test]
    fn test_client_fallback_methods() {
        let client = GeminiClient::new()
            .with_fallback(true)
            .with_fallback_endpoint("https://custom-gemini.example.com")
            .with_fallback_api_key("custom-api-key");

        assert!(client.fallback_enabled);
        assert_eq!(
            client.fallback_endpoint,
            Some("https://custom-gemini.example.com".to_string())
        );
        assert_eq!(
            client.fallback_api_key,
            Some("custom-api-key".to_string())
        );
    }

    #[test]
    fn test_client_has_fallback() {
        // With fallback API key
        let client_with_fallback = GeminiClient::new()
            .with_fallback(true)
            .with_fallback_api_key("test-key");
        assert!(client_with_fallback.has_fallback());

        // Without fallback API key
        let mut client_without_key = GeminiClient::new().with_fallback(true);
        client_without_key.fallback_api_key = None;
        assert!(!client_without_key.has_fallback());

        // Fallback disabled
        let client_disabled = GeminiClient::new()
            .with_fallback(false)
            .with_fallback_api_key("test-key");
        assert!(!client_disabled.has_fallback());
    }

    #[test]
    fn test_gemini_direct_endpoint_constant() {
        assert_eq!(
            GEMINI_DIRECT_ENDPOINT,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn test_get_gemini_api_key() {
        let saved = env::var("GEMINI_API_KEY").ok();

        // Set a test value
        env::set_var("GEMINI_API_KEY", "test-gemini-key");
        assert_eq!(get_gemini_api_key(), Some("test-gemini-key".to_string()));

        // Clear and test None
        env::remove_var("GEMINI_API_KEY");
        assert_eq!(get_gemini_api_key(), None);

        // Restore
        if let Some(val) = saved {
            env::set_var("GEMINI_API_KEY", val);
        }
    }

    #[test]
    fn test_client_from_config() {
        let config = AntigravityConfig::default()
            .with_endpoint("http://test:8000")
            .with_api_key("test-api-key")
            .with_model("gemini-3-flash")
            .with_timeout(45);

        let client = GeminiClient::from_config(config);

        assert_eq!(client.endpoint, "http://test:8000");
        assert_eq!(client.api_key, "test-api-key");
        assert_eq!(client.model, GeminiModel::Flash3);
        assert_eq!(client.timeout, Duration::from_secs(45));
    }

    #[tokio::test]
    async fn test_is_antigravity_available_with_config_unreachable() {
        // Test with an endpoint that won't be reachable
        let config = AntigravityConfig::default()
            .with_endpoint("http://127.0.0.1:59999"); // Unlikely to be running

        let result = is_antigravity_available_with_config(&config).await;
        // This should return false as nothing is listening on that port
        assert!(!result);
    }
}
