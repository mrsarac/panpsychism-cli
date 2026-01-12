//! Rate Limiter Agent module for Project Panpsychism.
//!
//! Agent 27: The Flow Controller - Manages request rates and resource allocation.
//!
//! This module implements rate limiting with multiple strategies to protect
//! system resources and ensure fair access across clients. Like a wise gatekeeper,
//! the Rate Limiter Agent controls the flow of requests through the system.
//!
//! ## Philosophy
//!
//! In the Spinoza framework, rate limiting embodies balance and sustainability:
//!
//! - **CONATUS**: Self-preservation through resource protection
//! - **RATIO**: Logical allocation based on fair rules
//! - **NATURA**: Natural flow control like a river's tributaries
//! - **LAETITIA**: Joy through consistent, predictable service
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |   Client Request | --> |  Rate Limiter    | --> |    Decision      |
//! |   (with ClientId)|     | (Strategy-Based) |     | (Allow/Throttle) |
//! +------------------+     +------------------+     +------------------+
//!                                  |
//!                                  v
//!                         +------------------+
//!                         |   Usage Stats    |
//!                         |   (Per-Client)   |
//!                         +------------------+
//! ```
//!
//! ## Strategies
//!
//! | Strategy | Description | Best For |
//! |----------|-------------|----------|
//! | TokenBucket | Allows bursts, refills over time | API rate limiting |
//! | SlidingWindow | Smooth rate over rolling window | Real-time systems |
//! | FixedWindow | Simple count per time window | Basic protection |
//! | LeakyBucket | Constant output rate | Traffic shaping |
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::rate_limiter::{RateLimiterAgent, RateLimitStrategy, ClientId};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let limiter = RateLimiterAgent::builder()
//!         .requests_per_minute(60)
//!         .burst_limit(10)
//!         .strategy(RateLimitStrategy::TokenBucket)
//!         .build();
//!
//!     let client = ClientId::new("user-123");
//!
//!     match limiter.check_rate(&client).await {
//!         RateLimitDecision::Allowed => {
//!             limiter.record_request(&client).await?;
//!             // Process request
//!         }
//!         RateLimitDecision::Throttled { retry_after } => {
//!             println!("Rate limited, retry after {:?}", retry_after);
//!         }
//!         RateLimitDecision::Denied { reason } => {
//!             println!("Request denied: {}", reason);
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};

// =============================================================================
// CLIENT IDENTIFIER
// =============================================================================

/// Unique identifier for a client making requests.
///
/// Used to track per-client rate limits and usage statistics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClientId(String);

impl ClientId {
    /// Create a new client identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the string representation of this client ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Create a default/anonymous client ID.
    pub fn anonymous() -> Self {
        Self("anonymous".to_string())
    }

    /// Create a global client ID (for global rate limits).
    pub fn global() -> Self {
        Self("__global__".to_string())
    }
}

impl std::fmt::Display for ClientId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ClientId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for ClientId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// =============================================================================
// RATE LIMIT STRATEGY
// =============================================================================

/// Strategy for rate limiting requests.
///
/// Each strategy offers different trade-offs:
///
/// - **TokenBucket**: Flexible, allows bursts up to bucket capacity
/// - **SlidingWindow**: Smooth limiting over a rolling time window
/// - **FixedWindow**: Simple, resets at fixed intervals
/// - **LeakyBucket**: Constant rate output, queues excess requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum RateLimitStrategy {
    /// Token bucket algorithm - allows bursts, refills tokens over time.
    ///
    /// Tokens are added at a constant rate up to the bucket capacity.
    /// Each request consumes one token. Requests are allowed if tokens
    /// are available.
    #[default]
    TokenBucket,

    /// Sliding window algorithm - counts requests in a rolling time window.
    ///
    /// Provides smoother rate limiting by considering requests in the
    /// recent past rather than fixed intervals.
    SlidingWindow,

    /// Fixed window algorithm - simple count per time interval.
    ///
    /// Resets the counter at the start of each window. Simple but can
    /// allow bursts at window boundaries.
    FixedWindow,

    /// Leaky bucket algorithm - constant output rate.
    ///
    /// Requests are processed at a constant rate. Excess requests are
    /// queued (up to capacity) or rejected.
    LeakyBucket,
}

impl std::fmt::Display for RateLimitStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenBucket => write!(f, "TokenBucket"),
            Self::SlidingWindow => write!(f, "SlidingWindow"),
            Self::FixedWindow => write!(f, "FixedWindow"),
            Self::LeakyBucket => write!(f, "LeakyBucket"),
        }
    }
}

impl std::str::FromStr for RateLimitStrategy {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "token" | "token_bucket" | "tokenbucket" => Ok(Self::TokenBucket),
            "sliding" | "sliding_window" | "slidingwindow" => Ok(Self::SlidingWindow),
            "fixed" | "fixed_window" | "fixedwindow" => Ok(Self::FixedWindow),
            "leaky" | "leaky_bucket" | "leakybucket" => Ok(Self::LeakyBucket),
            _ => Err(crate::Error::Config(format!(
                "Unknown rate limit strategy: '{}'. Valid: token, sliding, fixed, leaky",
                s
            ))),
        }
    }
}

impl RateLimitStrategy {
    /// Get all available strategies.
    pub fn all() -> Vec<Self> {
        vec![
            Self::TokenBucket,
            Self::SlidingWindow,
            Self::FixedWindow,
            Self::LeakyBucket,
        ]
    }

    /// Get a description of this strategy.
    pub fn description(&self) -> &'static str {
        match self {
            Self::TokenBucket => "Allows bursts up to capacity, refills tokens over time",
            Self::SlidingWindow => "Smooth rate limiting over a rolling time window",
            Self::FixedWindow => "Simple counter that resets at fixed intervals",
            Self::LeakyBucket => "Constant output rate, queues or rejects excess",
        }
    }
}

// =============================================================================
// RATE LIMIT DECISION
// =============================================================================

/// The decision returned by the rate limiter.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitDecision {
    /// Request is allowed to proceed.
    Allowed,

    /// Request should be throttled (temporarily delayed).
    Throttled {
        /// Suggested time to wait before retrying.
        retry_after: Duration,
    },

    /// Request is denied (hard limit exceeded).
    Denied {
        /// Reason for denial.
        reason: String,
    },
}

impl RateLimitDecision {
    /// Check if this decision allows the request.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }

    /// Check if this decision throttles the request.
    pub fn is_throttled(&self) -> bool {
        matches!(self, Self::Throttled { .. })
    }

    /// Check if this decision denies the request.
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Denied { .. })
    }

    /// Get the retry-after duration if throttled.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Throttled { retry_after } => Some(*retry_after),
            _ => None,
        }
    }
}

impl std::fmt::Display for RateLimitDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allowed => write!(f, "Allowed"),
            Self::Throttled { retry_after } => {
                write!(f, "Throttled (retry after {:?})", retry_after)
            }
            Self::Denied { reason } => write!(f, "Denied: {}", reason),
        }
    }
}

// =============================================================================
// USAGE STATISTICS
// =============================================================================

/// Usage statistics for a client.
#[derive(Debug, Clone)]
pub struct UsageStats {
    /// Total requests made in the current window.
    pub requests_in_window: u64,

    /// Total requests made all time.
    pub total_requests: u64,

    /// Requests remaining in the current quota.
    pub remaining_quota: u64,

    /// Time until the current window resets (in milliseconds).
    pub window_reset_in_ms: u64,

    /// Current token count (for token bucket).
    pub current_tokens: f64,

    /// Maximum tokens allowed.
    pub max_tokens: f64,

    /// Whether the client is currently rate limited.
    pub is_rate_limited: bool,

    /// Milliseconds since last request (None if no requests).
    pub last_request_ago_ms: Option<u64>,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            requests_in_window: 0,
            total_requests: 0,
            remaining_quota: 0,
            window_reset_in_ms: 0,
            current_tokens: 0.0,
            max_tokens: 0.0,
            is_rate_limited: false,
            last_request_ago_ms: None,
        }
    }
}

impl UsageStats {
    /// Get the usage percentage (0.0 - 100.0).
    pub fn usage_percentage(&self) -> f64 {
        if self.max_tokens == 0.0 {
            0.0
        } else {
            ((self.max_tokens - self.current_tokens) / self.max_tokens) * 100.0
        }
    }

    /// Check if near the rate limit (>80% usage).
    pub fn is_near_limit(&self) -> bool {
        self.usage_percentage() > 80.0
    }
}

// =============================================================================
// RATE LIMIT CONFIGURATION
// =============================================================================

/// Configuration for the Rate Limiter Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per minute.
    #[serde(default = "default_requests_per_minute")]
    pub requests_per_minute: u32,

    /// Maximum requests per hour.
    #[serde(default = "default_requests_per_hour")]
    pub requests_per_hour: u32,

    /// Burst limit (max requests in a short burst).
    #[serde(default = "default_burst_limit")]
    pub burst_limit: u32,

    /// Cooldown period after hitting the limit.
    #[serde(default = "default_cooldown_secs")]
    pub cooldown_secs: u64,

    /// Rate limiting strategy to use.
    #[serde(default)]
    pub strategy: RateLimitStrategy,

    /// Whether to enable per-client tracking.
    #[serde(default = "default_per_client_tracking")]
    pub per_client_tracking: bool,

    /// Maximum number of clients to track.
    #[serde(default = "default_max_clients")]
    pub max_clients: usize,

    /// Token refill rate per second (for token bucket).
    #[serde(default = "default_refill_rate")]
    pub token_refill_rate: f64,

    /// Window size in seconds (for window-based strategies).
    #[serde(default = "default_window_secs")]
    pub window_secs: u64,

    /// Whether rate limiting is enabled.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_requests_per_minute() -> u32 {
    60
}

fn default_requests_per_hour() -> u32 {
    1000
}

fn default_burst_limit() -> u32 {
    10
}

fn default_cooldown_secs() -> u64 {
    60
}

fn default_per_client_tracking() -> bool {
    true
}

fn default_max_clients() -> usize {
    10000
}

fn default_refill_rate() -> f64 {
    1.0
}

fn default_window_secs() -> u64 {
    60
}

fn default_enabled() -> bool {
    true
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: default_requests_per_minute(),
            requests_per_hour: default_requests_per_hour(),
            burst_limit: default_burst_limit(),
            cooldown_secs: default_cooldown_secs(),
            strategy: RateLimitStrategy::default(),
            per_client_tracking: default_per_client_tracking(),
            max_clients: default_max_clients(),
            token_refill_rate: default_refill_rate(),
            window_secs: default_window_secs(),
            enabled: default_enabled(),
        }
    }
}

impl RateLimitConfig {
    /// Create a permissive configuration (high limits).
    pub fn permissive() -> Self {
        Self {
            requests_per_minute: 1000,
            requests_per_hour: 10000,
            burst_limit: 100,
            cooldown_secs: 10,
            strategy: RateLimitStrategy::TokenBucket,
            per_client_tracking: true,
            max_clients: 100000,
            token_refill_rate: 10.0,
            window_secs: 60,
            enabled: true,
        }
    }

    /// Create a strict configuration (low limits).
    pub fn strict() -> Self {
        Self {
            requests_per_minute: 10,
            requests_per_hour: 100,
            burst_limit: 3,
            cooldown_secs: 120,
            strategy: RateLimitStrategy::SlidingWindow,
            per_client_tracking: true,
            max_clients: 1000,
            token_refill_rate: 0.5,
            window_secs: 60,
            enabled: true,
        }
    }

    /// Create a disabled configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Get the cooldown duration.
    pub fn cooldown(&self) -> Duration {
        Duration::from_secs(self.cooldown_secs)
    }

    /// Get the window duration.
    pub fn window(&self) -> Duration {
        Duration::from_secs(self.window_secs)
    }
}

// =============================================================================
// CLIENT STATE
// =============================================================================

/// Internal state for tracking a client's rate limit status.
#[derive(Debug, Clone)]
struct ClientState {
    /// Current token count (for token bucket).
    tokens: f64,

    /// Last token refill time.
    last_refill: Instant,

    /// Request timestamps for sliding window.
    request_times: Vec<Instant>,

    /// Request count for fixed window.
    window_count: u64,

    /// Window start time.
    window_start: Instant,

    /// Total requests made.
    total_requests: u64,

    /// Whether in cooldown.
    in_cooldown: bool,

    /// Cooldown end time.
    cooldown_until: Option<Instant>,

    /// Last request time.
    last_request: Option<Instant>,
}

impl ClientState {
    fn new(config: &RateLimitConfig) -> Self {
        Self {
            tokens: config.burst_limit as f64,
            last_refill: Instant::now(),
            request_times: Vec::new(),
            window_count: 0,
            window_start: Instant::now(),
            total_requests: 0,
            in_cooldown: false,
            cooldown_until: None,
            last_request: None,
        }
    }
}

// =============================================================================
// RATE LIMITER AGENT
// =============================================================================

/// The Flow Controller - Agent 27 of Project Panpsychism.
///
/// The Rate Limiter Agent manages request rates and resource allocation,
/// protecting the system from overload while ensuring fair access.
///
/// ## Capabilities
///
/// - **Multiple Strategies**: Token bucket, sliding window, fixed window, leaky bucket
/// - **Per-Client Tracking**: Individual limits for each client
/// - **Burst Handling**: Allow controlled bursts above normal rate
/// - **Usage Statistics**: Detailed metrics per client
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::rate_limiter::{RateLimiterAgent, RateLimitStrategy, ClientId};
///
/// let limiter = RateLimiterAgent::builder()
///     .requests_per_minute(100)
///     .burst_limit(20)
///     .strategy(RateLimitStrategy::TokenBucket)
///     .build();
///
/// let client = ClientId::new("api-client-1");
///
/// // Check and record in one operation
/// if limiter.check_and_record(&client).await?.is_allowed() {
///     // Process the request
/// }
/// ```
#[derive(Debug)]
pub struct RateLimiterAgent {
    /// Configuration.
    config: RateLimitConfig,

    /// Per-client state.
    clients: Arc<RwLock<HashMap<ClientId, ClientState>>>,

    /// Global state (for non-per-client limiting).
    global_state: Arc<RwLock<ClientState>>,
}

impl Clone for RateLimiterAgent {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            clients: Arc::clone(&self.clients),
            global_state: Arc::clone(&self.global_state),
        }
    }
}

impl Default for RateLimiterAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiterAgent {
    /// Create a new Rate Limiter Agent with default configuration.
    pub fn new() -> Self {
        let config = RateLimitConfig::default();
        Self {
            global_state: Arc::new(RwLock::new(ClientState::new(&config))),
            clients: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create a new Rate Limiter Agent with custom configuration.
    pub fn with_config(config: RateLimitConfig) -> Self {
        Self {
            global_state: Arc::new(RwLock::new(ClientState::new(&config))),
            clients: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> RateLimiterAgentBuilder {
        RateLimiterAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    // =========================================================================
    // MAIN RATE LIMITING METHODS
    // =========================================================================

    /// Check if a request is allowed for the given client.
    ///
    /// This method does not consume a token/increment counters. Use
    /// `record_request` after processing to update usage.
    ///
    /// # Arguments
    ///
    /// * `client_id` - The client making the request
    ///
    /// # Returns
    ///
    /// A `RateLimitDecision` indicating whether the request is allowed.
    pub async fn check_rate(&self, client_id: &ClientId) -> RateLimitDecision {
        if !self.config.enabled {
            return RateLimitDecision::Allowed;
        }

        let state = if self.config.per_client_tracking {
            self.get_or_create_client_state(client_id)
        } else {
            self.global_state.read().unwrap().clone()
        };

        // Check cooldown
        if state.in_cooldown {
            if let Some(until) = state.cooldown_until {
                if Instant::now() < until {
                    let retry_after = until.duration_since(Instant::now());
                    debug!(
                        "Client {} in cooldown, retry after {:?}",
                        client_id, retry_after
                    );
                    return RateLimitDecision::Throttled { retry_after };
                }
            }
        }

        match self.config.strategy {
            RateLimitStrategy::TokenBucket => self.check_token_bucket(&state),
            RateLimitStrategy::SlidingWindow => self.check_sliding_window(&state),
            RateLimitStrategy::FixedWindow => self.check_fixed_window(&state),
            RateLimitStrategy::LeakyBucket => self.check_leaky_bucket(&state),
        }
    }

    /// Record a request for the given client.
    ///
    /// Call this after successfully processing a request to update
    /// rate limit counters.
    ///
    /// # Arguments
    ///
    /// * `client_id` - The client that made the request
    pub async fn record_request(&self, client_id: &ClientId) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        if self.config.per_client_tracking {
            self.update_client_state(client_id);
        } else {
            self.update_global_state();
        }

        trace!("Recorded request for client {}", client_id);
        Ok(())
    }

    /// Check rate and record in one operation.
    ///
    /// Returns the decision. If allowed, the request is automatically recorded.
    pub async fn check_and_record(&self, client_id: &ClientId) -> Result<RateLimitDecision> {
        let decision = self.check_rate(client_id).await;

        if decision.is_allowed() {
            self.record_request(client_id).await?;
        }

        Ok(decision)
    }

    /// Get usage statistics for a client.
    pub async fn get_usage(&self, client_id: &ClientId) -> UsageStats {
        if !self.config.per_client_tracking {
            return self.build_usage_stats(&self.global_state.read().unwrap());
        }

        let clients = self.clients.read().unwrap();
        if let Some(state) = clients.get(client_id) {
            self.build_usage_stats(state)
        } else {
            UsageStats::default()
        }
    }

    /// Reset rate limits for a specific client.
    pub async fn reset_client(&self, client_id: &ClientId) {
        if self.config.per_client_tracking {
            let mut clients = self.clients.write().unwrap();
            clients.remove(client_id);
            info!("Reset rate limits for client {}", client_id);
        }
    }

    /// Reset all rate limits.
    pub async fn reset_all(&self) {
        {
            let mut clients = self.clients.write().unwrap();
            clients.clear();
        }
        {
            let mut global = self.global_state.write().unwrap();
            *global = ClientState::new(&self.config);
        }
        info!("Reset all rate limits");
    }

    /// Get the number of tracked clients.
    pub fn tracked_clients(&self) -> usize {
        self.clients.read().unwrap().len()
    }

    /// Check if a client is currently rate limited.
    pub async fn is_rate_limited(&self, client_id: &ClientId) -> bool {
        !self.check_rate(client_id).await.is_allowed()
    }

    // =========================================================================
    // STRATEGY IMPLEMENTATIONS
    // =========================================================================

    fn check_token_bucket(&self, state: &ClientState) -> RateLimitDecision {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill);
        let refilled = elapsed.as_secs_f64() * self.config.token_refill_rate;
        let current_tokens = (state.tokens + refilled).min(self.config.burst_limit as f64);

        if current_tokens >= 1.0 {
            RateLimitDecision::Allowed
        } else {
            let tokens_needed = 1.0 - current_tokens;
            let wait_secs = tokens_needed / self.config.token_refill_rate;
            RateLimitDecision::Throttled {
                retry_after: Duration::from_secs_f64(wait_secs),
            }
        }
    }

    fn check_sliding_window(&self, state: &ClientState) -> RateLimitDecision {
        let now = Instant::now();
        let window_start = now - self.config.window();

        let requests_in_window = state
            .request_times
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u32;

        if requests_in_window < self.config.requests_per_minute {
            RateLimitDecision::Allowed
        } else if let Some(&oldest_in_window) = state
            .request_times
            .iter()
            .find(|&&t| t >= window_start)
        {
            let retry_after = oldest_in_window + self.config.window() - now;
            RateLimitDecision::Throttled { retry_after }
        } else {
            RateLimitDecision::Allowed
        }
    }

    fn check_fixed_window(&self, state: &ClientState) -> RateLimitDecision {
        let now = Instant::now();
        let window_elapsed = now.duration_since(state.window_start);

        if window_elapsed >= self.config.window() {
            // Window has reset
            RateLimitDecision::Allowed
        } else if state.window_count < self.config.requests_per_minute as u64 {
            RateLimitDecision::Allowed
        } else {
            let retry_after = self.config.window() - window_elapsed;
            RateLimitDecision::Throttled { retry_after }
        }
    }

    fn check_leaky_bucket(&self, state: &ClientState) -> RateLimitDecision {
        if let Some(last) = state.last_request {
            let elapsed = last.elapsed();
            let min_interval =
                Duration::from_secs_f64(60.0 / self.config.requests_per_minute as f64);

            if elapsed < min_interval {
                let retry_after = min_interval - elapsed;
                return RateLimitDecision::Throttled { retry_after };
            }
        }

        // Check burst capacity
        let now = Instant::now();
        let window_start = now - Duration::from_secs(1);
        let recent_requests = state
            .request_times
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u32;

        if recent_requests >= self.config.burst_limit {
            RateLimitDecision::Denied {
                reason: "Burst limit exceeded".to_string(),
            }
        } else {
            RateLimitDecision::Allowed
        }
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    fn get_or_create_client_state(&self, client_id: &ClientId) -> ClientState {
        let clients = self.clients.read().unwrap();
        if let Some(state) = clients.get(client_id) {
            return state.clone();
        }
        drop(clients);

        // Create new state
        let mut clients = self.clients.write().unwrap();

        // Check max clients limit
        if clients.len() >= self.config.max_clients {
            // Evict oldest client (LRU-style)
            if let Some(oldest) = clients
                .iter()
                .min_by_key(|(_, s)| s.last_request.unwrap_or(Instant::now()))
                .map(|(k, _)| k.clone())
            {
                clients.remove(&oldest);
                warn!("Evicted oldest client {} due to max_clients limit", oldest);
            }
        }

        let state = ClientState::new(&self.config);
        clients.insert(client_id.clone(), state.clone());
        state
    }

    fn update_client_state(&self, client_id: &ClientId) {
        let mut clients = self.clients.write().unwrap();
        let state = clients
            .entry(client_id.clone())
            .or_insert_with(|| ClientState::new(&self.config));

        self.update_state(state);
    }

    fn update_global_state(&self) {
        let mut state = self.global_state.write().unwrap();
        self.update_state(&mut state);
    }

    fn update_state(&self, state: &mut ClientState) {
        let now = Instant::now();

        // Update tokens
        let elapsed = now.duration_since(state.last_refill);
        let refilled = elapsed.as_secs_f64() * self.config.token_refill_rate;
        state.tokens = (state.tokens + refilled - 1.0).max(0.0);
        state.last_refill = now;

        // Update sliding window
        state.request_times.push(now);
        let window_start = now - self.config.window();
        state.request_times.retain(|&t| t >= window_start);

        // Update fixed window
        if now.duration_since(state.window_start) >= self.config.window() {
            state.window_start = now;
            state.window_count = 1;
        } else {
            state.window_count += 1;
        }

        // Check if should enter cooldown
        if state.window_count >= self.config.requests_per_minute as u64 {
            state.in_cooldown = true;
            state.cooldown_until = Some(now + self.config.cooldown());
        } else if state.in_cooldown {
            if let Some(until) = state.cooldown_until {
                if now >= until {
                    state.in_cooldown = false;
                    state.cooldown_until = None;
                }
            }
        }

        state.total_requests += 1;
        state.last_request = Some(now);
    }

    fn build_usage_stats(&self, state: &ClientState) -> UsageStats {
        let now = Instant::now();
        let window_start = now - self.config.window();

        let requests_in_window = state
            .request_times
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u64;

        let elapsed = now.duration_since(state.last_refill);
        let refilled = elapsed.as_secs_f64() * self.config.token_refill_rate;
        let current_tokens = (state.tokens + refilled).min(self.config.burst_limit as f64);

        let remaining = if current_tokens >= 1.0 {
            self.config.requests_per_minute as u64 - requests_in_window
        } else {
            0
        };

        let window_reset_in_ms = if now.duration_since(state.window_start) >= self.config.window() {
            0
        } else {
            (self.config.window() - now.duration_since(state.window_start)).as_millis() as u64
        };

        let last_request_ago_ms = state.last_request.map(|t| t.elapsed().as_millis() as u64);

        UsageStats {
            requests_in_window,
            total_requests: state.total_requests,
            remaining_quota: remaining,
            window_reset_in_ms,
            current_tokens,
            max_tokens: self.config.burst_limit as f64,
            is_rate_limited: state.in_cooldown,
            last_request_ago_ms,
        }
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom RateLimiterAgent configuration.
#[derive(Debug, Default)]
pub struct RateLimiterAgentBuilder {
    config: Option<RateLimitConfig>,
}

impl RateLimiterAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: RateLimitConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set requests per minute limit.
    pub fn requests_per_minute(mut self, rpm: u32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.requests_per_minute = rpm;
        self.config = Some(config);
        self
    }

    /// Set requests per hour limit.
    pub fn requests_per_hour(mut self, rph: u32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.requests_per_hour = rph;
        self.config = Some(config);
        self
    }

    /// Set burst limit.
    pub fn burst_limit(mut self, limit: u32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.burst_limit = limit;
        self.config = Some(config);
        self
    }

    /// Set cooldown period in seconds.
    pub fn cooldown_secs(mut self, secs: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.cooldown_secs = secs;
        self.config = Some(config);
        self
    }

    /// Set the rate limiting strategy.
    pub fn strategy(mut self, strategy: RateLimitStrategy) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.strategy = strategy;
        self.config = Some(config);
        self
    }

    /// Enable or disable per-client tracking.
    pub fn per_client_tracking(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.per_client_tracking = enabled;
        self.config = Some(config);
        self
    }

    /// Set maximum number of clients to track.
    pub fn max_clients(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_clients = max;
        self.config = Some(config);
        self
    }

    /// Set token refill rate per second.
    pub fn token_refill_rate(mut self, rate: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.token_refill_rate = rate;
        self.config = Some(config);
        self
    }

    /// Set window size in seconds.
    pub fn window_secs(mut self, secs: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.window_secs = secs;
        self.config = Some(config);
        self
    }

    /// Enable or disable rate limiting.
    pub fn enabled(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.enabled = enabled;
        self.config = Some(config);
        self
    }

    /// Build the RateLimiterAgent.
    pub fn build(self) -> RateLimiterAgent {
        RateLimiterAgent::with_config(self.config.unwrap_or_default())
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // -------------------------------------------------------------------------
    // ClientId Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_client_id_new() {
        let id = ClientId::new("user-123");
        assert_eq!(id.as_str(), "user-123");
    }

    #[test]
    fn test_client_id_display() {
        let id = ClientId::new("test-client");
        assert_eq!(format!("{}", id), "test-client");
    }

    #[test]
    fn test_client_id_from_str() {
        let id: ClientId = "from-str".into();
        assert_eq!(id.as_str(), "from-str");
    }

    #[test]
    fn test_client_id_from_string() {
        let id: ClientId = String::from("from-string").into();
        assert_eq!(id.as_str(), "from-string");
    }

    #[test]
    fn test_client_id_anonymous() {
        let id = ClientId::anonymous();
        assert_eq!(id.as_str(), "anonymous");
    }

    #[test]
    fn test_client_id_global() {
        let id = ClientId::global();
        assert_eq!(id.as_str(), "__global__");
    }

    #[test]
    fn test_client_id_equality() {
        let id1 = ClientId::new("test");
        let id2 = ClientId::new("test");
        let id3 = ClientId::new("other");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_client_id_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ClientId::new("a"));
        set.insert(ClientId::new("b"));
        set.insert(ClientId::new("a")); // duplicate

        assert_eq!(set.len(), 2);
    }

    // -------------------------------------------------------------------------
    // RateLimitStrategy Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_strategy_display() {
        assert_eq!(RateLimitStrategy::TokenBucket.to_string(), "TokenBucket");
        assert_eq!(RateLimitStrategy::SlidingWindow.to_string(), "SlidingWindow");
        assert_eq!(RateLimitStrategy::FixedWindow.to_string(), "FixedWindow");
        assert_eq!(RateLimitStrategy::LeakyBucket.to_string(), "LeakyBucket");
    }

    #[test]
    fn test_strategy_from_str() {
        assert_eq!(
            "token".parse::<RateLimitStrategy>().unwrap(),
            RateLimitStrategy::TokenBucket
        );
        assert_eq!(
            "sliding".parse::<RateLimitStrategy>().unwrap(),
            RateLimitStrategy::SlidingWindow
        );
        assert_eq!(
            "fixed".parse::<RateLimitStrategy>().unwrap(),
            RateLimitStrategy::FixedWindow
        );
        assert_eq!(
            "leaky".parse::<RateLimitStrategy>().unwrap(),
            RateLimitStrategy::LeakyBucket
        );
    }

    #[test]
    fn test_strategy_from_str_aliases() {
        assert_eq!(
            "token_bucket".parse::<RateLimitStrategy>().unwrap(),
            RateLimitStrategy::TokenBucket
        );
        assert_eq!(
            "tokenbucket".parse::<RateLimitStrategy>().unwrap(),
            RateLimitStrategy::TokenBucket
        );
        assert_eq!(
            "sliding_window".parse::<RateLimitStrategy>().unwrap(),
            RateLimitStrategy::SlidingWindow
        );
    }

    #[test]
    fn test_strategy_from_str_invalid() {
        let result = "invalid".parse::<RateLimitStrategy>();
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_default() {
        assert_eq!(RateLimitStrategy::default(), RateLimitStrategy::TokenBucket);
    }

    #[test]
    fn test_strategy_all() {
        let all = RateLimitStrategy::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&RateLimitStrategy::TokenBucket));
        assert!(all.contains(&RateLimitStrategy::SlidingWindow));
        assert!(all.contains(&RateLimitStrategy::FixedWindow));
        assert!(all.contains(&RateLimitStrategy::LeakyBucket));
    }

    #[test]
    fn test_strategy_description() {
        let desc = RateLimitStrategy::TokenBucket.description();
        assert!(desc.contains("burst"));

        let desc = RateLimitStrategy::SlidingWindow.description();
        assert!(desc.contains("rolling"));
    }

    // -------------------------------------------------------------------------
    // RateLimitDecision Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_decision_allowed() {
        let decision = RateLimitDecision::Allowed;
        assert!(decision.is_allowed());
        assert!(!decision.is_throttled());
        assert!(!decision.is_denied());
        assert!(decision.retry_after().is_none());
    }

    #[test]
    fn test_decision_throttled() {
        let decision = RateLimitDecision::Throttled {
            retry_after: Duration::from_secs(5),
        };
        assert!(!decision.is_allowed());
        assert!(decision.is_throttled());
        assert!(!decision.is_denied());
        assert_eq!(decision.retry_after(), Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_decision_denied() {
        let decision = RateLimitDecision::Denied {
            reason: "Quota exceeded".to_string(),
        };
        assert!(!decision.is_allowed());
        assert!(!decision.is_throttled());
        assert!(decision.is_denied());
        assert!(decision.retry_after().is_none());
    }

    #[test]
    fn test_decision_display() {
        assert_eq!(format!("{}", RateLimitDecision::Allowed), "Allowed");

        let throttled = RateLimitDecision::Throttled {
            retry_after: Duration::from_secs(10),
        };
        let display = format!("{}", throttled);
        assert!(display.contains("Throttled"));
        assert!(display.contains("10"));

        let denied = RateLimitDecision::Denied {
            reason: "Test reason".to_string(),
        };
        let display = format!("{}", denied);
        assert!(display.contains("Denied"));
        assert!(display.contains("Test reason"));
    }

    // -------------------------------------------------------------------------
    // UsageStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_usage_stats_default() {
        let stats = UsageStats::default();
        assert_eq!(stats.requests_in_window, 0);
        assert_eq!(stats.total_requests, 0);
        assert!(!stats.is_rate_limited);
    }

    #[test]
    fn test_usage_stats_percentage() {
        let stats = UsageStats {
            max_tokens: 100.0,
            current_tokens: 80.0,
            ..Default::default()
        };
        assert!((stats.usage_percentage() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_stats_percentage_zero_max() {
        let stats = UsageStats {
            max_tokens: 0.0,
            current_tokens: 0.0,
            ..Default::default()
        };
        assert!((stats.usage_percentage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_stats_near_limit() {
        let stats_near = UsageStats {
            max_tokens: 100.0,
            current_tokens: 15.0,
            ..Default::default()
        };
        assert!(stats_near.is_near_limit());

        let stats_ok = UsageStats {
            max_tokens: 100.0,
            current_tokens: 50.0,
            ..Default::default()
        };
        assert!(!stats_ok.is_near_limit());
    }

    // -------------------------------------------------------------------------
    // RateLimitConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.requests_per_minute, 60);
        assert_eq!(config.requests_per_hour, 1000);
        assert_eq!(config.burst_limit, 10);
        assert_eq!(config.cooldown_secs, 60);
        assert!(config.enabled);
        assert!(config.per_client_tracking);
    }

    #[test]
    fn test_config_permissive() {
        let config = RateLimitConfig::permissive();
        assert_eq!(config.requests_per_minute, 1000);
        assert_eq!(config.burst_limit, 100);
        assert!(config.enabled);
    }

    #[test]
    fn test_config_strict() {
        let config = RateLimitConfig::strict();
        assert_eq!(config.requests_per_minute, 10);
        assert_eq!(config.burst_limit, 3);
        assert!(config.enabled);
    }

    #[test]
    fn test_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_cooldown() {
        let config = RateLimitConfig {
            cooldown_secs: 120,
            ..Default::default()
        };
        assert_eq!(config.cooldown(), Duration::from_secs(120));
    }

    #[test]
    fn test_config_window() {
        let config = RateLimitConfig {
            window_secs: 30,
            ..Default::default()
        };
        assert_eq!(config.window(), Duration::from_secs(30));
    }

    // -------------------------------------------------------------------------
    // RateLimiterAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_new() {
        let agent = RateLimiterAgent::new();
        assert_eq!(agent.config().requests_per_minute, 60);
        assert!(agent.config().enabled);
    }

    #[test]
    fn test_agent_with_config() {
        let config = RateLimitConfig {
            requests_per_minute: 100,
            ..Default::default()
        };
        let agent = RateLimiterAgent::with_config(config);
        assert_eq!(agent.config().requests_per_minute, 100);
    }

    #[test]
    fn test_agent_clone() {
        let agent1 = RateLimiterAgent::new();
        let agent2 = agent1.clone();
        assert_eq!(agent1.config().requests_per_minute, agent2.config().requests_per_minute);
    }

    #[tokio::test]
    async fn test_agent_disabled_always_allows() {
        let agent = RateLimiterAgent::builder().enabled(false).build();
        let client = ClientId::new("test");

        for _ in 0..1000 {
            let decision = agent.check_rate(&client).await;
            assert!(decision.is_allowed());
        }
    }

    #[tokio::test]
    async fn test_agent_token_bucket_allows_burst() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::TokenBucket)
            .burst_limit(5)
            .token_refill_rate(1.0)
            .build();

        let client = ClientId::new("test");

        // Should allow burst
        for i in 0..5 {
            let decision = agent.check_and_record(&client).await.unwrap();
            assert!(decision.is_allowed(), "Request {} should be allowed", i);
        }

        // Next should be throttled
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_throttled());
    }

    #[tokio::test]
    async fn test_agent_token_bucket_refills() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::TokenBucket)
            .burst_limit(2)
            .token_refill_rate(10.0) // Fast refill for test
            .build();

        let client = ClientId::new("test");

        // Use all tokens
        agent.check_and_record(&client).await.unwrap();
        agent.check_and_record(&client).await.unwrap();

        // Should be throttled
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_throttled());

        // Wait for refill
        thread::sleep(Duration::from_millis(200));

        // Should be allowed now
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_allowed());
    }

    #[tokio::test]
    async fn test_agent_fixed_window_limit() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::FixedWindow)
            .requests_per_minute(3)
            .window_secs(1)
            .build();

        let client = ClientId::new("test");

        // Should allow first 3
        for i in 0..3 {
            let decision = agent.check_and_record(&client).await.unwrap();
            assert!(decision.is_allowed(), "Request {} should be allowed", i);
        }

        // Fourth should be throttled
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_throttled());
    }

    #[tokio::test]
    async fn test_agent_fixed_window_resets() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::FixedWindow)
            .requests_per_minute(2)
            .window_secs(1)
            .cooldown_secs(0)
            .build();

        let client = ClientId::new("test");

        // Use up quota
        agent.check_and_record(&client).await.unwrap();
        agent.check_and_record(&client).await.unwrap();

        // Wait for window reset
        thread::sleep(Duration::from_millis(1100));

        // Should be allowed after window reset
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_allowed());
    }

    #[tokio::test]
    async fn test_agent_sliding_window_limit() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::SlidingWindow)
            .requests_per_minute(3)
            .window_secs(1)
            .build();

        let client = ClientId::new("test");

        // Should allow first 3
        for _ in 0..3 {
            agent.check_and_record(&client).await.unwrap();
        }

        // Fourth should be throttled
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_throttled());
    }

    #[tokio::test]
    async fn test_agent_leaky_bucket_spacing() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::LeakyBucket)
            .requests_per_minute(60)
            .burst_limit(100) // High burst to avoid burst limit
            .build();

        let client = ClientId::new("test");

        // First request allowed
        let decision = agent.check_and_record(&client).await.unwrap();
        assert!(decision.is_allowed());

        // Immediate second request should be throttled (need 1 sec gap at 60rpm)
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_throttled());
    }

    #[tokio::test]
    async fn test_agent_per_client_tracking() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::TokenBucket)
            .burst_limit(2)
            .per_client_tracking(true)
            .build();

        let client1 = ClientId::new("client1");
        let client2 = ClientId::new("client2");

        // Exhaust client1's quota
        agent.check_and_record(&client1).await.unwrap();
        agent.check_and_record(&client1).await.unwrap();

        // Client1 should be throttled
        let decision = agent.check_rate(&client1).await;
        assert!(decision.is_throttled());

        // Client2 should still be allowed
        let decision = agent.check_rate(&client2).await;
        assert!(decision.is_allowed());
    }

    #[tokio::test]
    async fn test_agent_global_tracking() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::TokenBucket)
            .burst_limit(2)
            .per_client_tracking(false)
            .build();

        let client1 = ClientId::new("client1");
        let client2 = ClientId::new("client2");

        // Exhaust global quota
        agent.check_and_record(&client1).await.unwrap();
        agent.check_and_record(&client2).await.unwrap();

        // Both should be throttled (shared quota)
        let decision1 = agent.check_rate(&client1).await;
        let decision2 = agent.check_rate(&client2).await;

        assert!(decision1.is_throttled());
        assert!(decision2.is_throttled());
    }

    #[tokio::test]
    async fn test_agent_get_usage() {
        let agent = RateLimiterAgent::builder()
            .burst_limit(10)
            .requests_per_minute(100)
            .build();

        let client = ClientId::new("test");

        // Initial usage
        let usage = agent.get_usage(&client).await;
        assert_eq!(usage.total_requests, 0);

        // Make some requests
        agent.check_and_record(&client).await.unwrap();
        agent.check_and_record(&client).await.unwrap();

        let usage = agent.get_usage(&client).await;
        assert_eq!(usage.total_requests, 2);
        assert_eq!(usage.requests_in_window, 2);
    }

    #[tokio::test]
    async fn test_agent_reset_client() {
        let agent = RateLimiterAgent::builder()
            .burst_limit(2)
            .build();

        let client = ClientId::new("test");

        // Exhaust quota
        agent.check_and_record(&client).await.unwrap();
        agent.check_and_record(&client).await.unwrap();

        // Should be throttled
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_throttled());

        // Reset client
        agent.reset_client(&client).await;

        // Should be allowed again
        let decision = agent.check_rate(&client).await;
        assert!(decision.is_allowed());
    }

    #[tokio::test]
    async fn test_agent_reset_all() {
        let agent = RateLimiterAgent::builder()
            .burst_limit(1)
            .build();

        let client1 = ClientId::new("client1");
        let client2 = ClientId::new("client2");

        // Exhaust both
        agent.check_and_record(&client1).await.unwrap();
        agent.check_and_record(&client2).await.unwrap();

        // Reset all
        agent.reset_all().await;

        // Both should be allowed
        assert!(agent.check_rate(&client1).await.is_allowed());
        assert!(agent.check_rate(&client2).await.is_allowed());
    }

    #[tokio::test]
    async fn test_agent_tracked_clients() {
        let agent = RateLimiterAgent::builder()
            .per_client_tracking(true)
            .build();

        assert_eq!(agent.tracked_clients(), 0);

        agent.check_and_record(&ClientId::new("a")).await.unwrap();
        assert_eq!(agent.tracked_clients(), 1);

        agent.check_and_record(&ClientId::new("b")).await.unwrap();
        assert_eq!(agent.tracked_clients(), 2);

        // Same client doesn't add new entry
        agent.check_and_record(&ClientId::new("a")).await.unwrap();
        assert_eq!(agent.tracked_clients(), 2);
    }

    #[tokio::test]
    async fn test_agent_max_clients_eviction() {
        let agent = RateLimiterAgent::builder()
            .max_clients(2)
            .per_client_tracking(true)
            .build();

        agent.check_and_record(&ClientId::new("a")).await.unwrap();
        thread::sleep(Duration::from_millis(10));
        agent.check_and_record(&ClientId::new("b")).await.unwrap();
        thread::sleep(Duration::from_millis(10));

        assert_eq!(agent.tracked_clients(), 2);

        // Adding third should evict oldest
        agent.check_and_record(&ClientId::new("c")).await.unwrap();
        assert_eq!(agent.tracked_clients(), 2);
    }

    #[tokio::test]
    async fn test_agent_is_rate_limited() {
        let agent = RateLimiterAgent::builder()
            .burst_limit(1)
            .build();

        let client = ClientId::new("test");

        assert!(!agent.is_rate_limited(&client).await);

        agent.check_and_record(&client).await.unwrap();

        assert!(agent.is_rate_limited(&client).await);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let agent = RateLimiterAgentBuilder::default().build();
        assert_eq!(agent.config().requests_per_minute, 60);
    }

    #[test]
    fn test_builder_requests_per_minute() {
        let agent = RateLimiterAgent::builder()
            .requests_per_minute(100)
            .build();
        assert_eq!(agent.config().requests_per_minute, 100);
    }

    #[test]
    fn test_builder_requests_per_hour() {
        let agent = RateLimiterAgent::builder()
            .requests_per_hour(5000)
            .build();
        assert_eq!(agent.config().requests_per_hour, 5000);
    }

    #[test]
    fn test_builder_burst_limit() {
        let agent = RateLimiterAgent::builder().burst_limit(20).build();
        assert_eq!(agent.config().burst_limit, 20);
    }

    #[test]
    fn test_builder_cooldown_secs() {
        let agent = RateLimiterAgent::builder().cooldown_secs(120).build();
        assert_eq!(agent.config().cooldown_secs, 120);
    }

    #[test]
    fn test_builder_strategy() {
        let agent = RateLimiterAgent::builder()
            .strategy(RateLimitStrategy::SlidingWindow)
            .build();
        assert_eq!(agent.config().strategy, RateLimitStrategy::SlidingWindow);
    }

    #[test]
    fn test_builder_per_client_tracking() {
        let agent = RateLimiterAgent::builder()
            .per_client_tracking(false)
            .build();
        assert!(!agent.config().per_client_tracking);
    }

    #[test]
    fn test_builder_max_clients() {
        let agent = RateLimiterAgent::builder().max_clients(500).build();
        assert_eq!(agent.config().max_clients, 500);
    }

    #[test]
    fn test_builder_token_refill_rate() {
        let agent = RateLimiterAgent::builder().token_refill_rate(2.5).build();
        assert!((agent.config().token_refill_rate - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder_window_secs() {
        let agent = RateLimiterAgent::builder().window_secs(30).build();
        assert_eq!(agent.config().window_secs, 30);
    }

    #[test]
    fn test_builder_enabled() {
        let agent = RateLimiterAgent::builder().enabled(false).build();
        assert!(!agent.config().enabled);
    }

    #[test]
    fn test_builder_chaining() {
        let agent = RateLimiterAgent::builder()
            .requests_per_minute(100)
            .burst_limit(20)
            .strategy(RateLimitStrategy::SlidingWindow)
            .cooldown_secs(30)
            .enabled(true)
            .build();

        assert_eq!(agent.config().requests_per_minute, 100);
        assert_eq!(agent.config().burst_limit, 20);
        assert_eq!(agent.config().strategy, RateLimitStrategy::SlidingWindow);
        assert_eq!(agent.config().cooldown_secs, 30);
        assert!(agent.config().enabled);
    }

    #[test]
    fn test_builder_with_config() {
        let config = RateLimitConfig::strict();
        let agent = RateLimiterAgent::builder().config(config).build();
        assert_eq!(agent.config().requests_per_minute, 10);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_multiple_strategies_same_limits() {
        let strategies = RateLimitStrategy::all();

        for strategy in strategies {
            let agent = RateLimiterAgent::builder()
                .strategy(strategy)
                .burst_limit(3)
                .requests_per_minute(3)
                .window_secs(1)
                .build();

            let client = ClientId::new("test");

            // Each strategy should allow at least some requests
            let decision = agent.check_and_record(&client).await.unwrap();
            assert!(
                decision.is_allowed(),
                "Strategy {} should allow first request",
                strategy
            );
        }
    }

    #[tokio::test]
    async fn test_concurrent_clients() {
        let agent = RateLimiterAgent::builder()
            .burst_limit(5)
            .per_client_tracking(true)
            .build();

        // Simulate multiple clients
        let clients: Vec<ClientId> = (0..10).map(|i| ClientId::new(format!("client-{}", i))).collect();

        for client in &clients {
            let decision = agent.check_and_record(client).await.unwrap();
            assert!(decision.is_allowed());
        }

        assert_eq!(agent.tracked_clients(), 10);
    }

    #[tokio::test]
    async fn test_usage_stats_accuracy() {
        let agent = RateLimiterAgent::builder()
            .burst_limit(10)
            .requests_per_minute(100)
            .token_refill_rate(1.0)
            .build();

        let client = ClientId::new("test");

        // Make 5 requests
        for _ in 0..5 {
            agent.check_and_record(&client).await.unwrap();
        }

        let usage = agent.get_usage(&client).await;
        assert_eq!(usage.total_requests, 5);
        assert_eq!(usage.requests_in_window, 5);
        assert!(usage.current_tokens < 10.0); // Some tokens consumed
    }
}
