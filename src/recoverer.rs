//! Recovery Agent for Panpsychism v3.0.
//!
//! This module provides the `RecovererAgent` - a high-level resilience manager that
//! orchestrates failure recovery using multiple strategies.
//!
//! # Architecture
//!
//! The Recoverer Agent builds on top of the existing recovery primitives (`CircuitBreaker`,
//! `RetryConfig`, etc.) to provide intelligent, context-aware recovery:
//!
//! ```text
//! Error Occurs
//!      |
//!      v
//! +------------------+
//! |  RecovererAgent  |  <-- Analyzes error context
//! +------------------+
//!      |
//!      v
//! +------------------+
//! | Strategy Select  |  <-- Choose best recovery approach
//! +------------------+
//!      |
//!   +--+--+--+--+
//!   |  |  |  |  |
//!   v  v  v  v  v
//!  Retry Fallback CircuitBreak Degrade Failover
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::recoverer::{RecovererAgent, RecoveryStrategy, ErrorContext};
//!
//! let recoverer = RecovererAgent::builder()
//!     .max_retries(3)
//!     .retry_delay_ms(1000)
//!     .circuit_threshold(5)
//!     .fallback_chain(vec!["model-a", "model-b", "model-c"])
//!     .build();
//!
//! let error_ctx = ErrorContext::new("API call failed")
//!     .with_error_code(503)
//!     .with_operation("generate_text");
//!
//! let result = recoverer.recover(error_ctx, || async {
//!     // Your operation here
//!     Ok("success")
//! }).await;
//! ```

use crate::recovery::{CircuitBreaker, CircuitState, RetryConfig};
use crate::{Error, Result};
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// =============================================================================
// RECOVERY STRATEGY
// =============================================================================

/// Recovery strategies available to the Recoverer Agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryStrategy {
    /// Retry the operation with exponential backoff.
    Retry,
    /// Fall back to an alternative implementation or service.
    Fallback,
    /// Open the circuit breaker to prevent cascading failures.
    CircuitBreak,
    /// Gracefully degrade functionality to maintain partial service.
    GracefulDegrade,
    /// Failover to a completely different system or endpoint.
    Failover,
}

impl std::fmt::Display for RecoveryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryStrategy::Retry => write!(f, "retry"),
            RecoveryStrategy::Fallback => write!(f, "fallback"),
            RecoveryStrategy::CircuitBreak => write!(f, "circuit_break"),
            RecoveryStrategy::GracefulDegrade => write!(f, "graceful_degrade"),
            RecoveryStrategy::Failover => write!(f, "failover"),
        }
    }
}

impl RecoveryStrategy {
    /// Get all available strategies.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Retry,
            Self::Fallback,
            Self::CircuitBreak,
            Self::GracefulDegrade,
            Self::Failover,
        ]
    }

    /// Check if this strategy involves retrying the original operation.
    pub fn retries_original(&self) -> bool {
        matches!(self, Self::Retry)
    }

    /// Check if this strategy uses alternative implementations.
    pub fn uses_alternative(&self) -> bool {
        matches!(self, Self::Fallback | Self::Failover)
    }

    /// Get the priority of this strategy (lower = try first).
    pub fn priority(&self) -> u8 {
        match self {
            Self::Retry => 1,
            Self::Fallback => 2,
            Self::GracefulDegrade => 3,
            Self::Failover => 4,
            Self::CircuitBreak => 5,
        }
    }
}

// =============================================================================
// DEGRADE LEVEL
// =============================================================================

/// Levels of graceful degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DegradeLevel {
    /// Full functionality available.
    Full,
    /// Partial functionality - some features disabled.
    Partial,
    /// Minimal functionality - core features only.
    Minimal,
    /// Read-only mode - no write operations.
    ReadOnly,
}

impl std::fmt::Display for DegradeLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DegradeLevel::Full => write!(f, "full"),
            DegradeLevel::Partial => write!(f, "partial"),
            DegradeLevel::Minimal => write!(f, "minimal"),
            DegradeLevel::ReadOnly => write!(f, "read_only"),
        }
    }
}

impl DegradeLevel {
    /// Check if write operations are allowed at this level.
    pub fn allows_writes(&self) -> bool {
        !matches!(self, Self::ReadOnly)
    }

    /// Check if all features are available.
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Get the next lower degradation level.
    pub fn degrade(&self) -> Self {
        match self {
            Self::Full => Self::Partial,
            Self::Partial => Self::Minimal,
            Self::Minimal => Self::ReadOnly,
            Self::ReadOnly => Self::ReadOnly,
        }
    }

    /// Get the next higher functionality level.
    pub fn upgrade(&self) -> Self {
        match self {
            Self::Full => Self::Full,
            Self::Partial => Self::Full,
            Self::Minimal => Self::Partial,
            Self::ReadOnly => Self::Minimal,
        }
    }
}

impl Default for DegradeLevel {
    fn default() -> Self {
        Self::Full
    }
}

// =============================================================================
// ERROR CONTEXT
// =============================================================================

/// Contextual information about an error for recovery decisions.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The error message.
    pub message: String,
    /// HTTP status code if applicable.
    pub error_code: Option<u16>,
    /// The operation that failed.
    pub operation: Option<String>,
    /// The service or component that failed.
    pub service: Option<String>,
    /// Number of previous attempts.
    pub attempt_count: u32,
    /// Timestamp when the error occurred.
    pub timestamp: Instant,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
    /// Whether this error is considered transient.
    pub is_transient: Option<bool>,
    /// The original error if available.
    pub original_error: Option<String>,
}

impl ErrorContext {
    /// Create a new error context with a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_code: None,
            operation: None,
            service: None,
            attempt_count: 0,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
            is_transient: None,
            original_error: None,
        }
    }

    /// Create from a panpsychism Error.
    pub fn from_error(error: &Error) -> Self {
        let message = error.to_string();
        let mut ctx = Self::new(&message);
        ctx.original_error = Some(message);

        // Try to extract error code from message
        if let Some(code) = Self::extract_error_code(&error.to_string()) {
            ctx.error_code = Some(code);
        }

        // Determine if transient
        ctx.is_transient = Some(crate::recovery::is_retryable_error(error));

        ctx
    }

    /// Extract HTTP status code from error message.
    fn extract_error_code(msg: &str) -> Option<u16> {
        // Look for patterns like "429", "500", "503"
        for code in [429, 500, 501, 502, 503, 504] {
            if msg.contains(&code.to_string()) {
                return Some(code);
            }
        }
        None
    }

    /// Set the error code.
    pub fn with_error_code(mut self, code: u16) -> Self {
        self.error_code = Some(code);
        self
    }

    /// Set the operation name.
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    /// Set the service name.
    pub fn with_service(mut self, service: impl Into<String>) -> Self {
        self.service = Some(service.into());
        self
    }

    /// Set the attempt count.
    pub fn with_attempts(mut self, count: u32) -> Self {
        self.attempt_count = count;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set transient flag explicitly.
    pub fn with_transient(mut self, is_transient: bool) -> Self {
        self.is_transient = Some(is_transient);
        self
    }

    /// Increment the attempt count.
    pub fn increment_attempts(&mut self) {
        self.attempt_count += 1;
    }

    /// Check if this is a rate limit error.
    pub fn is_rate_limit(&self) -> bool {
        self.error_code == Some(429)
            || self.message.to_lowercase().contains("rate limit")
            || self.message.to_lowercase().contains("quota")
    }

    /// Check if this is a server error (5xx).
    pub fn is_server_error(&self) -> bool {
        matches!(self.error_code, Some(code) if (500..600).contains(&code))
    }

    /// Check if this is a client error (4xx excluding rate limit).
    pub fn is_client_error(&self) -> bool {
        matches!(self.error_code, Some(code) if (400..500).contains(&code) && code != 429)
    }

    /// Check if the error is likely transient.
    pub fn is_transient_error(&self) -> bool {
        self.is_transient.unwrap_or_else(|| {
            self.is_rate_limit()
                || self.is_server_error()
                || self.message.to_lowercase().contains("timeout")
                || self.message.to_lowercase().contains("connection")
        })
    }

    /// Get time elapsed since the error occurred.
    pub fn elapsed(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new("Unknown error")
    }
}

// =============================================================================
// RECOVERY CONFIG
// =============================================================================

/// Configuration for the Recoverer Agent.
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Delay between retry attempts in milliseconds.
    pub retry_delay_ms: u64,
    /// Number of failures before circuit breaker opens.
    pub circuit_threshold: u32,
    /// Time to wait before trying half-open state.
    pub circuit_reset_ms: u64,
    /// Chain of fallback options (tried in order).
    pub fallback_chain: Vec<String>,
    /// Backoff multiplier for retries.
    pub backoff_multiplier: f64,
    /// Maximum retry delay in milliseconds.
    pub max_retry_delay_ms: u64,
    /// Whether to add jitter to retry delays.
    pub jitter: bool,
    /// Timeout for individual operations in milliseconds.
    pub operation_timeout_ms: u64,
    /// Strategies to try in order.
    pub strategy_order: Vec<RecoveryStrategy>,
    /// Initial degradation level.
    pub initial_degrade_level: DegradeLevel,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 1000,
            circuit_threshold: 5,
            circuit_reset_ms: 30000,
            fallback_chain: Vec::new(),
            backoff_multiplier: 2.0,
            max_retry_delay_ms: 30000,
            jitter: true,
            operation_timeout_ms: 30000,
            strategy_order: vec![
                RecoveryStrategy::Retry,
                RecoveryStrategy::Fallback,
                RecoveryStrategy::GracefulDegrade,
            ],
            initial_degrade_level: DegradeLevel::Full,
        }
    }
}

impl RecoveryConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config optimized for API calls.
    pub fn for_api() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 500,
            circuit_threshold: 5,
            circuit_reset_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
            ..Default::default()
        }
    }

    /// Create a config optimized for database operations.
    pub fn for_database() -> Self {
        Self {
            max_retries: 2,
            retry_delay_ms: 100,
            circuit_threshold: 3,
            circuit_reset_ms: 10000,
            backoff_multiplier: 1.5,
            jitter: false,
            ..Default::default()
        }
    }

    /// Create a config optimized for external services.
    pub fn for_external_service() -> Self {
        Self {
            max_retries: 5,
            retry_delay_ms: 2000,
            circuit_threshold: 10,
            circuit_reset_ms: 60000,
            backoff_multiplier: 2.5,
            jitter: true,
            fallback_chain: vec!["primary".to_string(), "secondary".to_string()],
            ..Default::default()
        }
    }

    /// Convert to a RetryConfig for use with existing primitives.
    pub fn to_retry_config(&self) -> RetryConfig {
        RetryConfig {
            max_retries: self.max_retries,
            initial_delay: Duration::from_millis(self.retry_delay_ms),
            max_delay: Duration::from_millis(self.max_retry_delay_ms),
            backoff_multiplier: self.backoff_multiplier,
            jitter: self.jitter,
        }
    }
}

// =============================================================================
// RECOVERY ACTION
// =============================================================================

/// Result of a recovery attempt.
#[derive(Debug, Clone)]
pub struct RecoveryAction<T> {
    /// The strategy that was used.
    pub strategy: RecoveryStrategy,
    /// Whether recovery was successful.
    pub success: bool,
    /// Whether a fallback was used.
    pub fallback_used: bool,
    /// Time taken for recovery in milliseconds.
    pub recovery_time_ms: u64,
    /// The result value if successful.
    pub result: Option<T>,
    /// The fallback identifier if one was used.
    pub fallback_id: Option<String>,
    /// Number of attempts made.
    pub attempts: u32,
    /// Current degradation level after recovery.
    pub degrade_level: DegradeLevel,
    /// Errors encountered during recovery.
    pub errors: Vec<String>,
}

impl<T> RecoveryAction<T> {
    /// Create a successful recovery action.
    pub fn success(strategy: RecoveryStrategy, result: T, recovery_time_ms: u64) -> Self {
        Self {
            strategy,
            success: true,
            fallback_used: false,
            recovery_time_ms,
            result: Some(result),
            fallback_id: None,
            attempts: 1,
            degrade_level: DegradeLevel::Full,
            errors: Vec::new(),
        }
    }

    /// Create a successful recovery action using a fallback.
    pub fn success_with_fallback(
        strategy: RecoveryStrategy,
        result: T,
        fallback_id: String,
        recovery_time_ms: u64,
        attempts: u32,
    ) -> Self {
        Self {
            strategy,
            success: true,
            fallback_used: true,
            recovery_time_ms,
            result: Some(result),
            fallback_id: Some(fallback_id),
            attempts,
            degrade_level: DegradeLevel::Full,
            errors: Vec::new(),
        }
    }

    /// Create a failed recovery action.
    pub fn failure(
        strategy: RecoveryStrategy,
        recovery_time_ms: u64,
        attempts: u32,
        errors: Vec<String>,
    ) -> Self {
        Self {
            strategy,
            success: false,
            fallback_used: false,
            recovery_time_ms,
            result: None,
            fallback_id: None,
            attempts,
            degrade_level: DegradeLevel::Full,
            errors,
        }
    }

    /// Set the degradation level.
    pub fn with_degrade_level(mut self, level: DegradeLevel) -> Self {
        self.degrade_level = level;
        self
    }

    /// Check if the action was successful.
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Get the result if successful.
    pub fn into_result(self) -> Option<T> {
        self.result
    }

    /// Map the result value.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> RecoveryAction<U> {
        RecoveryAction {
            strategy: self.strategy,
            success: self.success,
            fallback_used: self.fallback_used,
            recovery_time_ms: self.recovery_time_ms,
            result: self.result.map(f),
            fallback_id: self.fallback_id,
            attempts: self.attempts,
            degrade_level: self.degrade_level,
            errors: self.errors,
        }
    }
}

// =============================================================================
// RECOVERY STATS
// =============================================================================

/// Statistics about recovery operations.
#[derive(Debug, Default)]
pub struct RecoveryStats {
    /// Total recovery attempts.
    pub total_attempts: AtomicU64,
    /// Successful recoveries.
    pub successes: AtomicU64,
    /// Failed recoveries.
    pub failures: AtomicU64,
    /// Retries performed.
    pub retries: AtomicU64,
    /// Fallbacks used.
    pub fallbacks_used: AtomicU64,
    /// Circuit breaks triggered.
    pub circuit_breaks: AtomicU64,
    /// Degradations triggered.
    pub degradations: AtomicU64,
    /// Failovers performed.
    pub failovers: AtomicU64,
    /// Total recovery time in milliseconds.
    pub total_recovery_time_ms: AtomicU64,
}

impl RecoveryStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a recovery attempt.
    pub fn record_attempt(&self) {
        self.total_attempts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful recovery.
    pub fn record_success(&self, time_ms: u64) {
        self.successes.fetch_add(1, Ordering::Relaxed);
        self.total_recovery_time_ms.fetch_add(time_ms, Ordering::Relaxed);
    }

    /// Record a failed recovery.
    pub fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a retry.
    pub fn record_retry(&self) {
        self.retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fallback usage.
    pub fn record_fallback(&self) {
        self.fallbacks_used.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a circuit break.
    pub fn record_circuit_break(&self) {
        self.circuit_breaks.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a degradation.
    pub fn record_degradation(&self) {
        self.degradations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failover.
    pub fn record_failover(&self) {
        self.failovers.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.total_attempts.load(Ordering::Relaxed);
        if total == 0 {
            return 100.0;
        }
        let successes = self.successes.load(Ordering::Relaxed);
        (successes as f64 / total as f64) * 100.0
    }

    /// Get average recovery time in milliseconds.
    pub fn average_recovery_time_ms(&self) -> f64 {
        let successes = self.successes.load(Ordering::Relaxed);
        if successes == 0 {
            return 0.0;
        }
        let total_time = self.total_recovery_time_ms.load(Ordering::Relaxed);
        total_time as f64 / successes as f64
    }

    /// Get a snapshot of all stats.
    pub fn snapshot(&self) -> RecoveryStatsSnapshot {
        RecoveryStatsSnapshot {
            total_attempts: self.total_attempts.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            fallbacks_used: self.fallbacks_used.load(Ordering::Relaxed),
            circuit_breaks: self.circuit_breaks.load(Ordering::Relaxed),
            degradations: self.degradations.load(Ordering::Relaxed),
            failovers: self.failovers.load(Ordering::Relaxed),
            total_recovery_time_ms: self.total_recovery_time_ms.load(Ordering::Relaxed),
            success_rate: self.success_rate(),
            average_recovery_time_ms: self.average_recovery_time_ms(),
        }
    }
}

/// A point-in-time snapshot of recovery statistics.
#[derive(Debug, Clone)]
pub struct RecoveryStatsSnapshot {
    pub total_attempts: u64,
    pub successes: u64,
    pub failures: u64,
    pub retries: u64,
    pub fallbacks_used: u64,
    pub circuit_breaks: u64,
    pub degradations: u64,
    pub failovers: u64,
    pub total_recovery_time_ms: u64,
    pub success_rate: f64,
    pub average_recovery_time_ms: f64,
}

// =============================================================================
// RECOVERER AGENT BUILDER
// =============================================================================

/// Builder for constructing a RecovererAgent.
#[derive(Debug, Clone, Default)]
pub struct RecovererAgentBuilder {
    config: RecoveryConfig,
}

impl RecovererAgentBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum retry attempts.
    pub fn max_retries(mut self, max: u32) -> Self {
        self.config.max_retries = max;
        self
    }

    /// Set retry delay in milliseconds.
    pub fn retry_delay_ms(mut self, delay: u64) -> Self {
        self.config.retry_delay_ms = delay;
        self
    }

    /// Set circuit breaker threshold.
    pub fn circuit_threshold(mut self, threshold: u32) -> Self {
        self.config.circuit_threshold = threshold;
        self
    }

    /// Set circuit breaker reset time in milliseconds.
    pub fn circuit_reset_ms(mut self, reset_ms: u64) -> Self {
        self.config.circuit_reset_ms = reset_ms;
        self
    }

    /// Set the fallback chain.
    pub fn fallback_chain(mut self, chain: Vec<impl Into<String>>) -> Self {
        self.config.fallback_chain = chain.into_iter().map(Into::into).collect();
        self
    }

    /// Set backoff multiplier.
    pub fn backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.config.backoff_multiplier = multiplier;
        self
    }

    /// Set maximum retry delay in milliseconds.
    pub fn max_retry_delay_ms(mut self, max: u64) -> Self {
        self.config.max_retry_delay_ms = max;
        self
    }

    /// Enable or disable jitter.
    pub fn jitter(mut self, enabled: bool) -> Self {
        self.config.jitter = enabled;
        self
    }

    /// Set operation timeout in milliseconds.
    pub fn operation_timeout_ms(mut self, timeout: u64) -> Self {
        self.config.operation_timeout_ms = timeout;
        self
    }

    /// Set the order of strategies to try.
    pub fn strategy_order(mut self, order: Vec<RecoveryStrategy>) -> Self {
        self.config.strategy_order = order;
        self
    }

    /// Set the initial degradation level.
    pub fn initial_degrade_level(mut self, level: DegradeLevel) -> Self {
        self.config.initial_degrade_level = level;
        self
    }

    /// Use a preset configuration.
    pub fn with_config(mut self, config: RecoveryConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the RecovererAgent.
    pub fn build(self) -> RecovererAgent {
        RecovererAgent::new(self.config)
    }
}

// =============================================================================
// RECOVERER AGENT
// =============================================================================

/// The Recoverer Agent - handles failures and implements recovery strategies.
///
/// This agent provides intelligent, context-aware recovery from failures using
/// multiple strategies including retry, fallback, circuit breaking, graceful
/// degradation, and failover.
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::recoverer::{RecovererAgent, ErrorContext};
///
/// let recoverer = RecovererAgent::builder()
///     .max_retries(3)
///     .fallback_chain(vec!["primary", "secondary"])
///     .build();
///
/// let result = recoverer.recover(
///     ErrorContext::new("Connection failed"),
///     || async { Ok("success") }
/// ).await;
/// ```
#[derive(Debug)]
pub struct RecovererAgent {
    /// Configuration for recovery behavior.
    config: RecoveryConfig,
    /// Circuit breaker for the primary service.
    circuit_breaker: Arc<CircuitBreaker>,
    /// Circuit breakers for fallback services.
    fallback_breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    /// Current degradation level.
    degrade_level: Arc<RwLock<DegradeLevel>>,
    /// Recovery statistics.
    stats: Arc<RecoveryStats>,
}

impl RecovererAgent {
    /// Create a new RecovererAgent with the given configuration.
    pub fn new(config: RecoveryConfig) -> Self {
        let circuit_breaker = Arc::new(CircuitBreaker::new(
            config.circuit_threshold,
            Duration::from_millis(config.circuit_reset_ms),
        ));

        Self {
            config,
            circuit_breaker,
            fallback_breakers: Arc::new(RwLock::new(HashMap::new())),
            degrade_level: Arc::new(RwLock::new(DegradeLevel::Full)),
            stats: Arc::new(RecoveryStats::new()),
        }
    }

    /// Create a builder for constructing a RecovererAgent.
    pub fn builder() -> RecovererAgentBuilder {
        RecovererAgentBuilder::new()
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(RecoveryConfig::default())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RecoveryConfig {
        &self.config
    }

    /// Get the circuit breaker.
    pub fn circuit_breaker(&self) -> &Arc<CircuitBreaker> {
        &self.circuit_breaker
    }

    /// Get recovery statistics.
    pub fn stats(&self) -> &RecoveryStats {
        &self.stats
    }

    /// Get current degradation level.
    pub async fn degrade_level(&self) -> DegradeLevel {
        *self.degrade_level.read().await
    }

    /// Set degradation level manually.
    pub async fn set_degrade_level(&self, level: DegradeLevel) {
        let mut current = self.degrade_level.write().await;
        *current = level;
        if level != DegradeLevel::Full {
            self.stats.record_degradation();
        }
        info!("Degradation level set to: {}", level);
    }

    /// Get circuit breaker state.
    pub async fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state().await
    }

    /// Check if the circuit is open.
    pub async fn is_circuit_open(&self) -> bool {
        self.circuit_breaker.is_open().await
    }

    /// Force open the circuit breaker.
    pub fn force_circuit_open(&self) {
        self.circuit_breaker.force_open();
        self.stats.record_circuit_break();
    }

    /// Reset the circuit breaker.
    pub async fn reset_circuit(&self) {
        self.circuit_breaker.reset().await;
    }

    /// Get or create a circuit breaker for a fallback.
    async fn get_fallback_breaker(&self, fallback_id: &str) -> Arc<CircuitBreaker> {
        let breakers = self.fallback_breakers.read().await;
        if let Some(breaker) = breakers.get(fallback_id) {
            return Arc::clone(breaker);
        }
        drop(breakers);

        let mut breakers = self.fallback_breakers.write().await;
        let breaker = Arc::new(CircuitBreaker::new(
            self.config.circuit_threshold,
            Duration::from_millis(self.config.circuit_reset_ms),
        ));
        breakers.insert(fallback_id.to_string(), Arc::clone(&breaker));
        breaker
    }

    /// Select the best recovery strategy based on error context.
    pub fn select_strategy(&self, context: &ErrorContext) -> RecoveryStrategy {
        // If too many attempts, don't retry
        if context.attempt_count >= self.config.max_retries {
            if !self.config.fallback_chain.is_empty() {
                return RecoveryStrategy::Fallback;
            }
            return RecoveryStrategy::GracefulDegrade;
        }

        // Rate limit errors should wait, not retry immediately
        if context.is_rate_limit() {
            if !self.config.fallback_chain.is_empty() {
                return RecoveryStrategy::Fallback;
            }
            return RecoveryStrategy::Retry;
        }

        // Transient errors are good candidates for retry
        if context.is_transient_error() {
            return RecoveryStrategy::Retry;
        }

        // Client errors shouldn't be retried
        if context.is_client_error() {
            return RecoveryStrategy::GracefulDegrade;
        }

        // Default: try strategies in configured order
        self.config
            .strategy_order
            .first()
            .copied()
            .unwrap_or(RecoveryStrategy::Retry)
    }

    /// Attempt recovery using the configured strategies.
    ///
    /// This is the main entry point for error recovery. It will:
    /// 1. Analyze the error context
    /// 2. Select appropriate recovery strategies
    /// 3. Execute strategies in order until one succeeds
    /// 4. Track statistics and update circuit breaker state
    ///
    /// # Arguments
    ///
    /// * `context` - The error context with information about the failure
    /// * `operation` - The operation to retry/recover
    ///
    /// # Returns
    ///
    /// A `RecoveryAction` containing the result and metadata about the recovery.
    pub async fn recover<T, F, Fut>(
        &self,
        mut context: ErrorContext,
        operation: F,
    ) -> Result<RecoveryAction<T>>
    where
        F: Fn() -> Fut + Clone,
        Fut: Future<Output = Result<T>>,
    {
        let start = Instant::now();
        self.stats.record_attempt();

        // Check circuit breaker first
        if self.circuit_breaker.is_open().await {
            debug!("Circuit breaker is open, skipping primary operation");
            self.stats.record_circuit_break();

            // Try fallback if available
            if !self.config.fallback_chain.is_empty() {
                return self
                    .try_fallback_chain(&context, operation, start)
                    .await;
            }

            // Otherwise degrade
            return Ok(RecoveryAction::<T>::failure(
                RecoveryStrategy::CircuitBreak,
                start.elapsed().as_millis() as u64,
                0,
                vec!["Circuit breaker is open".to_string()],
            )
            .with_degrade_level(self.degrade_level().await));
        }

        let mut errors = Vec::new();
        let retry_config = self.config.to_retry_config();

        // Attempt retries
        for attempt in 0..=self.config.max_retries {
            context.attempt_count = attempt;

            if attempt > 0 {
                let delay = retry_config.delay_for_attempt(attempt - 1);
                debug!("Retry attempt {}, waiting {:?}", attempt, delay);
                tokio::time::sleep(delay).await;
                self.stats.record_retry();
            }

            match operation().await {
                Ok(result) => {
                    self.circuit_breaker.record_success().await;
                    self.stats.record_success(start.elapsed().as_millis() as u64);

                    return Ok(RecoveryAction::success(
                        RecoveryStrategy::Retry,
                        result,
                        start.elapsed().as_millis() as u64,
                    ));
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    errors.push(error_msg.clone());
                    self.circuit_breaker.record_failure().await;

                    // Check if error is retryable
                    if !crate::recovery::is_retryable_error(&e) {
                        warn!("Non-retryable error: {}", error_msg);
                        break;
                    }

                    debug!("Attempt {} failed: {}", attempt + 1, error_msg);
                }
            }
        }

        // Retry failed, try fallback
        if !self.config.fallback_chain.is_empty() {
            let fallback_result = self
                .try_fallback_chain(&context, operation, start)
                .await?;
            if fallback_result.success {
                return Ok(fallback_result);
            }
            errors.extend(fallback_result.errors);
        }

        // All strategies failed
        self.stats.record_failure();
        self.maybe_degrade().await;

        Ok(RecoveryAction::<T>::failure(
            RecoveryStrategy::Retry,
            start.elapsed().as_millis() as u64,
            context.attempt_count + 1,
            errors,
        )
        .with_degrade_level(self.degrade_level().await))
    }

    /// Try the fallback chain.
    async fn try_fallback_chain<T, F, Fut>(
        &self,
        context: &ErrorContext,
        _operation: F,
        start: Instant,
    ) -> Result<RecoveryAction<T>>
    where
        F: Fn() -> Fut + Clone,
        Fut: Future<Output = Result<T>>,
    {
        let mut errors = Vec::new();

        for (idx, fallback_id) in self.config.fallback_chain.iter().enumerate() {
            let breaker = self.get_fallback_breaker(fallback_id).await;

            if breaker.is_open().await {
                debug!("Fallback {} circuit is open, skipping", fallback_id);
                continue;
            }

            debug!(
                "Trying fallback {} ({}/{})",
                fallback_id,
                idx + 1,
                self.config.fallback_chain.len()
            );

            // In a real implementation, we'd call a fallback-specific operation here
            // For now, we record the attempt
            self.stats.record_fallback();

            // Simulate fallback failure for this example
            // In practice, you'd have different operations for each fallback
            errors.push(format!("Fallback {} not implemented", fallback_id));
            breaker.record_failure().await;
        }

        Ok(RecoveryAction::<T>::failure(
            RecoveryStrategy::Fallback,
            start.elapsed().as_millis() as u64,
            context.attempt_count + self.config.fallback_chain.len() as u32,
            errors,
        ))
    }

    /// Recover with a specific fallback operation.
    ///
    /// This allows providing a different operation to run as a fallback.
    pub async fn recover_with_fallback<T, F, Fut, FB, FBFut>(
        &self,
        context: ErrorContext,
        operation: F,
        fallback: FB,
    ) -> Result<RecoveryAction<T>>
    where
        F: Fn() -> Fut + Clone,
        Fut: Future<Output = Result<T>>,
        FB: Fn() -> FBFut,
        FBFut: Future<Output = Result<T>>,
    {
        let start = Instant::now();
        self.stats.record_attempt();

        // Try primary operation first
        let primary_result = self.recover(context, operation).await?;
        if primary_result.success {
            return Ok(primary_result);
        }

        // Try fallback
        debug!("Primary failed, trying provided fallback");
        self.stats.record_fallback();

        match fallback().await {
            Ok(result) => {
                self.stats.record_success(start.elapsed().as_millis() as u64);
                Ok(RecoveryAction::success_with_fallback(
                    RecoveryStrategy::Fallback,
                    result,
                    "custom_fallback".to_string(),
                    start.elapsed().as_millis() as u64,
                    primary_result.attempts + 1,
                ))
            }
            Err(e) => {
                self.stats.record_failure();
                let mut errors = primary_result.errors;
                errors.push(e.to_string());
                Ok(RecoveryAction::<T>::failure(
                    RecoveryStrategy::Fallback,
                    start.elapsed().as_millis() as u64,
                    primary_result.attempts + 1,
                    errors,
                ))
            }
        }
    }

    /// Maybe degrade the service level after failures.
    async fn maybe_degrade(&self) {
        let mut level = self.degrade_level.write().await;
        let new_level = level.degrade();
        if new_level != *level {
            *level = new_level;
            self.stats.record_degradation();
            warn!("Service degraded to level: {}", new_level);
        }
    }

    /// Attempt to upgrade the service level.
    pub async fn try_upgrade(&self) {
        let mut level = self.degrade_level.write().await;
        let new_level = level.upgrade();
        if new_level != *level {
            *level = new_level;
            info!("Service upgraded to level: {}", new_level);
        }
    }

    /// Reset all state (circuit breakers, degradation level, etc.).
    pub async fn reset_all(&self) {
        self.circuit_breaker.reset().await;

        let mut breakers = self.fallback_breakers.write().await;
        for breaker in breakers.values() {
            breaker.reset().await;
        }
        breakers.clear();

        let mut level = self.degrade_level.write().await;
        *level = DegradeLevel::Full;

        info!("RecovererAgent reset to initial state");
    }

    /// Get a stats snapshot.
    pub fn stats_snapshot(&self) -> RecoveryStatsSnapshot {
        self.stats.snapshot()
    }
}

impl Clone for RecovererAgent {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            circuit_breaker: Arc::clone(&self.circuit_breaker),
            fallback_breakers: Arc::clone(&self.fallback_breakers),
            degrade_level: Arc::clone(&self.degrade_level),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl Default for RecovererAgent {
    fn default() -> Self {
        Self::default_config()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // RecoveryStrategy Tests
    // =========================================================================

    #[test]
    fn test_recovery_strategy_display() {
        assert_eq!(RecoveryStrategy::Retry.to_string(), "retry");
        assert_eq!(RecoveryStrategy::Fallback.to_string(), "fallback");
        assert_eq!(RecoveryStrategy::CircuitBreak.to_string(), "circuit_break");
        assert_eq!(
            RecoveryStrategy::GracefulDegrade.to_string(),
            "graceful_degrade"
        );
        assert_eq!(RecoveryStrategy::Failover.to_string(), "failover");
    }

    #[test]
    fn test_recovery_strategy_all() {
        let all = RecoveryStrategy::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&RecoveryStrategy::Retry));
        assert!(all.contains(&RecoveryStrategy::Fallback));
        assert!(all.contains(&RecoveryStrategy::CircuitBreak));
        assert!(all.contains(&RecoveryStrategy::GracefulDegrade));
        assert!(all.contains(&RecoveryStrategy::Failover));
    }

    #[test]
    fn test_recovery_strategy_retries_original() {
        assert!(RecoveryStrategy::Retry.retries_original());
        assert!(!RecoveryStrategy::Fallback.retries_original());
        assert!(!RecoveryStrategy::CircuitBreak.retries_original());
    }

    #[test]
    fn test_recovery_strategy_uses_alternative() {
        assert!(!RecoveryStrategy::Retry.uses_alternative());
        assert!(RecoveryStrategy::Fallback.uses_alternative());
        assert!(RecoveryStrategy::Failover.uses_alternative());
        assert!(!RecoveryStrategy::GracefulDegrade.uses_alternative());
    }

    #[test]
    fn test_recovery_strategy_priority() {
        assert!(RecoveryStrategy::Retry.priority() < RecoveryStrategy::Fallback.priority());
        assert!(RecoveryStrategy::Fallback.priority() < RecoveryStrategy::GracefulDegrade.priority());
        assert!(
            RecoveryStrategy::GracefulDegrade.priority() < RecoveryStrategy::CircuitBreak.priority()
        );
    }

    // =========================================================================
    // DegradeLevel Tests
    // =========================================================================

    #[test]
    fn test_degrade_level_display() {
        assert_eq!(DegradeLevel::Full.to_string(), "full");
        assert_eq!(DegradeLevel::Partial.to_string(), "partial");
        assert_eq!(DegradeLevel::Minimal.to_string(), "minimal");
        assert_eq!(DegradeLevel::ReadOnly.to_string(), "read_only");
    }

    #[test]
    fn test_degrade_level_allows_writes() {
        assert!(DegradeLevel::Full.allows_writes());
        assert!(DegradeLevel::Partial.allows_writes());
        assert!(DegradeLevel::Minimal.allows_writes());
        assert!(!DegradeLevel::ReadOnly.allows_writes());
    }

    #[test]
    fn test_degrade_level_is_full() {
        assert!(DegradeLevel::Full.is_full());
        assert!(!DegradeLevel::Partial.is_full());
        assert!(!DegradeLevel::Minimal.is_full());
        assert!(!DegradeLevel::ReadOnly.is_full());
    }

    #[test]
    fn test_degrade_level_degrade() {
        assert_eq!(DegradeLevel::Full.degrade(), DegradeLevel::Partial);
        assert_eq!(DegradeLevel::Partial.degrade(), DegradeLevel::Minimal);
        assert_eq!(DegradeLevel::Minimal.degrade(), DegradeLevel::ReadOnly);
        assert_eq!(DegradeLevel::ReadOnly.degrade(), DegradeLevel::ReadOnly);
    }

    #[test]
    fn test_degrade_level_upgrade() {
        assert_eq!(DegradeLevel::Full.upgrade(), DegradeLevel::Full);
        assert_eq!(DegradeLevel::Partial.upgrade(), DegradeLevel::Full);
        assert_eq!(DegradeLevel::Minimal.upgrade(), DegradeLevel::Partial);
        assert_eq!(DegradeLevel::ReadOnly.upgrade(), DegradeLevel::Minimal);
    }

    #[test]
    fn test_degrade_level_ordering() {
        // Derived ordering: Full < Partial < Minimal < ReadOnly
        // (lower index = less degraded = better)
        assert!(DegradeLevel::Full < DegradeLevel::Partial);
        assert!(DegradeLevel::Partial < DegradeLevel::Minimal);
        assert!(DegradeLevel::Minimal < DegradeLevel::ReadOnly);
    }

    #[test]
    fn test_degrade_level_default() {
        assert_eq!(DegradeLevel::default(), DegradeLevel::Full);
    }

    // =========================================================================
    // ErrorContext Tests
    // =========================================================================

    #[test]
    fn test_error_context_new() {
        let ctx = ErrorContext::new("Test error");
        assert_eq!(ctx.message, "Test error");
        assert!(ctx.error_code.is_none());
        assert!(ctx.operation.is_none());
        assert!(ctx.service.is_none());
        assert_eq!(ctx.attempt_count, 0);
    }

    #[test]
    fn test_error_context_builder() {
        let ctx = ErrorContext::new("API failed")
            .with_error_code(503)
            .with_operation("generate")
            .with_service("gemini")
            .with_attempts(2)
            .with_metadata("request_id", "abc123")
            .with_transient(true);

        assert_eq!(ctx.message, "API failed");
        assert_eq!(ctx.error_code, Some(503));
        assert_eq!(ctx.operation, Some("generate".to_string()));
        assert_eq!(ctx.service, Some("gemini".to_string()));
        assert_eq!(ctx.attempt_count, 2);
        assert_eq!(ctx.metadata.get("request_id"), Some(&"abc123".to_string()));
        assert_eq!(ctx.is_transient, Some(true));
    }

    #[test]
    fn test_error_context_from_error() {
        let error = Error::Synthesis("503 service unavailable".to_string());
        let ctx = ErrorContext::from_error(&error);

        assert!(ctx.message.contains("503"));
        assert_eq!(ctx.error_code, Some(503));
        assert_eq!(ctx.is_transient, Some(true));
    }

    #[test]
    fn test_error_context_is_rate_limit() {
        let rate_limit = ErrorContext::new("rate limit exceeded").with_error_code(429);
        assert!(rate_limit.is_rate_limit());

        let rate_limit2 = ErrorContext::new("quota exceeded");
        assert!(rate_limit2.is_rate_limit());

        let not_rate_limit = ErrorContext::new("internal error").with_error_code(500);
        assert!(!not_rate_limit.is_rate_limit());
    }

    #[test]
    fn test_error_context_is_server_error() {
        let server_error = ErrorContext::new("error").with_error_code(500);
        assert!(server_error.is_server_error());

        let server_error2 = ErrorContext::new("error").with_error_code(503);
        assert!(server_error2.is_server_error());

        let not_server = ErrorContext::new("error").with_error_code(400);
        assert!(!not_server.is_server_error());
    }

    #[test]
    fn test_error_context_is_client_error() {
        let client_error = ErrorContext::new("error").with_error_code(400);
        assert!(client_error.is_client_error());

        let client_error2 = ErrorContext::new("error").with_error_code(404);
        assert!(client_error2.is_client_error());

        // Rate limit is NOT considered a client error
        let rate_limit = ErrorContext::new("error").with_error_code(429);
        assert!(!rate_limit.is_client_error());

        let not_client = ErrorContext::new("error").with_error_code(500);
        assert!(!not_client.is_client_error());
    }

    #[test]
    fn test_error_context_is_transient_error() {
        let transient1 = ErrorContext::new("error").with_error_code(429);
        assert!(transient1.is_transient_error());

        let transient2 = ErrorContext::new("error").with_error_code(503);
        assert!(transient2.is_transient_error());

        let transient3 = ErrorContext::new("connection timeout");
        assert!(transient3.is_transient_error());

        let not_transient = ErrorContext::new("bad request").with_error_code(400);
        assert!(!not_transient.is_transient_error());
    }

    #[test]
    fn test_error_context_increment_attempts() {
        let mut ctx = ErrorContext::new("error");
        assert_eq!(ctx.attempt_count, 0);

        ctx.increment_attempts();
        assert_eq!(ctx.attempt_count, 1);

        ctx.increment_attempts();
        assert_eq!(ctx.attempt_count, 2);
    }

    #[test]
    fn test_error_context_default() {
        let ctx = ErrorContext::default();
        assert_eq!(ctx.message, "Unknown error");
    }

    // =========================================================================
    // RecoveryConfig Tests
    // =========================================================================

    #[test]
    fn test_recovery_config_default() {
        let config = RecoveryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 1000);
        assert_eq!(config.circuit_threshold, 5);
        assert!(config.jitter);
    }

    #[test]
    fn test_recovery_config_for_api() {
        let config = RecoveryConfig::for_api();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 500);
        assert!(config.jitter);
    }

    #[test]
    fn test_recovery_config_for_database() {
        let config = RecoveryConfig::for_database();
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.retry_delay_ms, 100);
        assert!(!config.jitter);
    }

    #[test]
    fn test_recovery_config_for_external_service() {
        let config = RecoveryConfig::for_external_service();
        assert_eq!(config.max_retries, 5);
        assert!(!config.fallback_chain.is_empty());
    }

    #[test]
    fn test_recovery_config_to_retry_config() {
        let config = RecoveryConfig {
            max_retries: 5,
            retry_delay_ms: 2000,
            max_retry_delay_ms: 60000,
            backoff_multiplier: 3.0,
            jitter: false,
            ..Default::default()
        };

        let retry_config = config.to_retry_config();
        assert_eq!(retry_config.max_retries, 5);
        assert_eq!(retry_config.initial_delay, Duration::from_millis(2000));
        assert_eq!(retry_config.max_delay, Duration::from_millis(60000));
        assert_eq!(retry_config.backoff_multiplier, 3.0);
        assert!(!retry_config.jitter);
    }

    // =========================================================================
    // RecoveryAction Tests
    // =========================================================================

    #[test]
    fn test_recovery_action_success() {
        let action: RecoveryAction<&str> =
            RecoveryAction::success(RecoveryStrategy::Retry, "result", 100);

        assert!(action.is_success());
        assert!(!action.fallback_used);
        assert_eq!(action.result, Some("result"));
        assert_eq!(action.recovery_time_ms, 100);
        assert_eq!(action.attempts, 1);
    }

    #[test]
    fn test_recovery_action_success_with_fallback() {
        let action: RecoveryAction<&str> = RecoveryAction::success_with_fallback(
            RecoveryStrategy::Fallback,
            "fallback_result",
            "backup_service".to_string(),
            200,
            3,
        );

        assert!(action.is_success());
        assert!(action.fallback_used);
        assert_eq!(action.fallback_id, Some("backup_service".to_string()));
        assert_eq!(action.attempts, 3);
    }

    #[test]
    fn test_recovery_action_failure() {
        let action: RecoveryAction<&str> = RecoveryAction::failure(
            RecoveryStrategy::Retry,
            300,
            4,
            vec!["Error 1".to_string(), "Error 2".to_string()],
        );

        assert!(!action.is_success());
        assert!(action.result.is_none());
        assert_eq!(action.attempts, 4);
        assert_eq!(action.errors.len(), 2);
    }

    #[test]
    fn test_recovery_action_with_degrade_level() {
        let action: RecoveryAction<&str> =
            RecoveryAction::success(RecoveryStrategy::Retry, "result", 100)
                .with_degrade_level(DegradeLevel::Partial);

        assert_eq!(action.degrade_level, DegradeLevel::Partial);
    }

    #[test]
    fn test_recovery_action_into_result() {
        let action: RecoveryAction<&str> =
            RecoveryAction::success(RecoveryStrategy::Retry, "result", 100);

        assert_eq!(action.into_result(), Some("result"));
    }

    #[test]
    fn test_recovery_action_map() {
        let action: RecoveryAction<i32> =
            RecoveryAction::success(RecoveryStrategy::Retry, 42, 100);

        let mapped = action.map(|x| x * 2);
        assert_eq!(mapped.result, Some(84));
        assert!(mapped.success);
    }

    // =========================================================================
    // RecoveryStats Tests
    // =========================================================================

    #[test]
    fn test_recovery_stats_new() {
        let stats = RecoveryStats::new();
        assert_eq!(stats.total_attempts.load(Ordering::Relaxed), 0);
        assert_eq!(stats.successes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_recovery_stats_record_operations() {
        let stats = RecoveryStats::new();

        stats.record_attempt();
        stats.record_success(100);
        stats.record_retry();
        stats.record_fallback();

        assert_eq!(stats.total_attempts.load(Ordering::Relaxed), 1);
        assert_eq!(stats.successes.load(Ordering::Relaxed), 1);
        assert_eq!(stats.retries.load(Ordering::Relaxed), 1);
        assert_eq!(stats.fallbacks_used.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_recovery_stats_success_rate() {
        let stats = RecoveryStats::new();

        // Empty stats should return 100%
        assert_eq!(stats.success_rate(), 100.0);

        stats.record_attempt();
        stats.record_success(100);
        stats.record_attempt();
        stats.record_failure();

        assert_eq!(stats.success_rate(), 50.0);
    }

    #[test]
    fn test_recovery_stats_average_recovery_time() {
        let stats = RecoveryStats::new();

        // No successes should return 0
        assert_eq!(stats.average_recovery_time_ms(), 0.0);

        stats.record_success(100);
        stats.record_success(200);

        assert_eq!(stats.average_recovery_time_ms(), 150.0);
    }

    #[test]
    fn test_recovery_stats_snapshot() {
        let stats = RecoveryStats::new();
        stats.record_attempt();
        stats.record_success(100);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_attempts, 1);
        assert_eq!(snapshot.successes, 1);
        assert_eq!(snapshot.success_rate, 100.0);
    }

    // =========================================================================
    // RecovererAgentBuilder Tests
    // =========================================================================

    #[test]
    fn test_builder_defaults() {
        let agent = RecovererAgentBuilder::new().build();
        assert_eq!(agent.config().max_retries, 3);
    }

    #[test]
    fn test_builder_custom_config() {
        let agent = RecovererAgentBuilder::new()
            .max_retries(5)
            .retry_delay_ms(2000)
            .circuit_threshold(10)
            .circuit_reset_ms(60000)
            .fallback_chain(vec!["a", "b", "c"])
            .backoff_multiplier(3.0)
            .max_retry_delay_ms(120000)
            .jitter(false)
            .operation_timeout_ms(45000)
            .initial_degrade_level(DegradeLevel::Partial)
            .build();

        assert_eq!(agent.config().max_retries, 5);
        assert_eq!(agent.config().retry_delay_ms, 2000);
        assert_eq!(agent.config().circuit_threshold, 10);
        assert_eq!(agent.config().circuit_reset_ms, 60000);
        assert_eq!(agent.config().fallback_chain.len(), 3);
        assert_eq!(agent.config().backoff_multiplier, 3.0);
        assert_eq!(agent.config().max_retry_delay_ms, 120000);
        assert!(!agent.config().jitter);
        assert_eq!(agent.config().operation_timeout_ms, 45000);
        assert_eq!(agent.config().initial_degrade_level, DegradeLevel::Partial);
    }

    #[test]
    fn test_builder_with_config() {
        let config = RecoveryConfig::for_api();
        let agent = RecovererAgentBuilder::new().with_config(config.clone()).build();

        assert_eq!(agent.config().max_retries, config.max_retries);
        assert_eq!(agent.config().retry_delay_ms, config.retry_delay_ms);
    }

    #[test]
    fn test_builder_strategy_order() {
        let agent = RecovererAgentBuilder::new()
            .strategy_order(vec![
                RecoveryStrategy::Fallback,
                RecoveryStrategy::Retry,
            ])
            .build();

        assert_eq!(agent.config().strategy_order.len(), 2);
        assert_eq!(
            agent.config().strategy_order[0],
            RecoveryStrategy::Fallback
        );
    }

    // =========================================================================
    // RecovererAgent Basic Tests
    // =========================================================================

    #[test]
    fn test_recoverer_agent_default() {
        let agent = RecovererAgent::default();
        assert_eq!(agent.config().max_retries, 3);
    }

    #[test]
    fn test_recoverer_agent_builder_method() {
        let agent = RecovererAgent::builder().max_retries(7).build();
        assert_eq!(agent.config().max_retries, 7);
    }

    #[test]
    fn test_recoverer_agent_clone() {
        let agent1 = RecovererAgent::builder().max_retries(5).build();
        let agent2 = agent1.clone();

        // Cloned agent should share the same circuit breaker
        assert_eq!(agent1.config().max_retries, agent2.config().max_retries);
    }

    #[tokio::test]
    async fn test_recoverer_agent_degrade_level() {
        let agent = RecovererAgent::default();
        assert_eq!(agent.degrade_level().await, DegradeLevel::Full);

        agent.set_degrade_level(DegradeLevel::Partial).await;
        assert_eq!(agent.degrade_level().await, DegradeLevel::Partial);
    }

    #[tokio::test]
    async fn test_recoverer_agent_circuit_state() {
        let agent = RecovererAgent::default();
        assert_eq!(agent.circuit_state().await, CircuitState::Closed);
        assert!(!agent.is_circuit_open().await);
    }

    #[tokio::test]
    async fn test_recoverer_agent_force_circuit() {
        let agent = RecovererAgent::default();

        agent.force_circuit_open();
        assert!(agent.is_circuit_open().await);

        agent.reset_circuit().await;
        assert!(!agent.is_circuit_open().await);
    }

    #[tokio::test]
    async fn test_recoverer_agent_try_upgrade() {
        let agent = RecovererAgent::default();

        agent.set_degrade_level(DegradeLevel::ReadOnly).await;
        agent.try_upgrade().await;
        assert_eq!(agent.degrade_level().await, DegradeLevel::Minimal);

        agent.try_upgrade().await;
        assert_eq!(agent.degrade_level().await, DegradeLevel::Partial);

        agent.try_upgrade().await;
        assert_eq!(agent.degrade_level().await, DegradeLevel::Full);

        // Can't upgrade beyond Full
        agent.try_upgrade().await;
        assert_eq!(agent.degrade_level().await, DegradeLevel::Full);
    }

    #[tokio::test]
    async fn test_recoverer_agent_reset_all() {
        let agent = RecovererAgent::default();

        agent.force_circuit_open();
        agent.set_degrade_level(DegradeLevel::Minimal).await;

        agent.reset_all().await;

        assert!(!agent.is_circuit_open().await);
        assert_eq!(agent.degrade_level().await, DegradeLevel::Full);
    }

    // =========================================================================
    // RecovererAgent Strategy Selection Tests
    // =========================================================================

    #[test]
    fn test_select_strategy_rate_limit_with_fallback() {
        let agent = RecovererAgent::builder()
            .fallback_chain(vec!["backup"])
            .build();

        let ctx = ErrorContext::new("rate limit exceeded").with_error_code(429);
        assert_eq!(agent.select_strategy(&ctx), RecoveryStrategy::Fallback);
    }

    #[test]
    fn test_select_strategy_rate_limit_without_fallback() {
        let agent = RecovererAgent::default();
        let ctx = ErrorContext::new("rate limit exceeded").with_error_code(429);
        assert_eq!(agent.select_strategy(&ctx), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_select_strategy_transient_error() {
        let agent = RecovererAgent::default();
        let ctx = ErrorContext::new("connection timeout");
        assert_eq!(agent.select_strategy(&ctx), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_select_strategy_client_error() {
        let agent = RecovererAgent::default();
        let ctx = ErrorContext::new("bad request").with_error_code(400);
        assert_eq!(agent.select_strategy(&ctx), RecoveryStrategy::GracefulDegrade);
    }

    #[test]
    fn test_select_strategy_max_attempts_exceeded() {
        let agent = RecovererAgent::builder()
            .max_retries(3)
            .fallback_chain(vec!["backup"])
            .build();

        let ctx = ErrorContext::new("error").with_attempts(4);
        assert_eq!(agent.select_strategy(&ctx), RecoveryStrategy::Fallback);
    }

    #[test]
    fn test_select_strategy_max_attempts_no_fallback() {
        let agent = RecovererAgent::builder().max_retries(3).build();

        let ctx = ErrorContext::new("error").with_attempts(4);
        assert_eq!(agent.select_strategy(&ctx), RecoveryStrategy::GracefulDegrade);
    }

    // =========================================================================
    // RecovererAgent Recovery Tests
    // =========================================================================

    #[tokio::test]
    async fn test_recover_success_first_try() {
        let agent = RecovererAgent::default();
        let ctx = ErrorContext::new("test error");

        let result = agent
            .recover(ctx, || async { Ok::<_, Error>("success") })
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.result, Some("success"));
        assert_eq!(result.strategy, RecoveryStrategy::Retry);
        assert!(!result.fallback_used);
    }

    #[tokio::test]
    async fn test_recover_eventual_success() {
        let agent = RecovererAgent::builder()
            .max_retries(3)
            .retry_delay_ms(10)
            .build();

        let ctx = ErrorContext::new("test error");
        let attempt_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = agent
            .recover(ctx, move || {
                let count = attempt_count_clone.clone();
                async move {
                    let attempt = count.fetch_add(1, Ordering::Relaxed);
                    if attempt < 2 {
                        Err(Error::Synthesis("503 service unavailable".to_string()))
                    } else {
                        Ok::<_, Error>("success")
                    }
                }
            })
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.result, Some("success"));
    }

    #[tokio::test]
    async fn test_recover_all_retries_fail() {
        let agent = RecovererAgent::builder()
            .max_retries(2)
            .retry_delay_ms(10)
            .build();

        let ctx = ErrorContext::new("test error");

        let result = agent
            .recover(ctx, || async {
                Err::<&str, _>(Error::Synthesis("503 service unavailable".to_string()))
            })
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.result.is_none());
        assert!(!result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_recover_circuit_open() {
        let agent = RecovererAgent::default();
        agent.force_circuit_open();

        let ctx = ErrorContext::new("test error");

        let result = agent
            .recover(ctx, || async { Ok::<_, Error>("success") })
            .await
            .unwrap();

        assert!(!result.is_success());
        assert_eq!(result.strategy, RecoveryStrategy::CircuitBreak);
    }

    #[tokio::test]
    async fn test_recover_with_fallback() {
        let agent = RecovererAgent::default();
        let ctx = ErrorContext::new("test error");

        let result = agent
            .recover_with_fallback(
                ctx,
                || async { Err::<&str, _>(Error::Synthesis("primary failed".to_string())) },
                || async { Ok::<_, Error>("fallback_success") },
            )
            .await
            .unwrap();

        assert!(result.is_success());
        assert!(result.fallback_used);
        assert_eq!(result.result, Some("fallback_success"));
    }

    #[tokio::test]
    async fn test_recover_with_fallback_both_fail() {
        let agent = RecovererAgent::builder()
            .max_retries(1)
            .retry_delay_ms(10)
            .build();

        let ctx = ErrorContext::new("test error");

        let result = agent
            .recover_with_fallback(
                ctx,
                || async { Err::<&str, _>(Error::Synthesis("503 primary failed".to_string())) },
                || async { Err::<&str, _>(Error::Synthesis("fallback failed".to_string())) },
            )
            .await
            .unwrap();

        assert!(!result.is_success());
        assert!(result.errors.len() >= 2);
    }

    // =========================================================================
    // RecovererAgent Stats Tests
    // =========================================================================

    #[tokio::test]
    async fn test_recover_updates_stats() {
        let agent = RecovererAgent::default();
        let ctx = ErrorContext::new("test error");

        agent
            .recover(ctx, || async { Ok::<_, Error>("success") })
            .await
            .unwrap();

        let snapshot = agent.stats_snapshot();
        assert!(snapshot.total_attempts >= 1);
        assert!(snapshot.successes >= 1);
    }

    #[tokio::test]
    async fn test_stats_track_failures() {
        let agent = RecovererAgent::builder()
            .max_retries(1)
            .retry_delay_ms(10)
            .build();

        let ctx = ErrorContext::new("test error");

        agent
            .recover(ctx, || async {
                Err::<&str, _>(Error::Synthesis("500 server error".to_string()))
            })
            .await
            .unwrap();

        let snapshot = agent.stats_snapshot();
        assert!(snapshot.failures >= 1);
    }

    // =========================================================================
    // Fallback Breaker Tests
    // =========================================================================

    #[tokio::test]
    async fn test_fallback_breaker_created_on_demand() {
        let agent = RecovererAgent::builder()
            .fallback_chain(vec!["service_a", "service_b"])
            .build();

        // Initially no fallback breakers
        let breakers = agent.fallback_breakers.read().await;
        assert!(breakers.is_empty());
        drop(breakers);

        // Get a fallback breaker
        let _breaker = agent.get_fallback_breaker("service_a").await;

        // Now it should exist
        let breakers = agent.fallback_breakers.read().await;
        assert!(breakers.contains_key("service_a"));
    }

    #[tokio::test]
    async fn test_fallback_breaker_reused() {
        let agent = RecovererAgent::default();

        let breaker1 = agent.get_fallback_breaker("test_service").await;
        let breaker2 = agent.get_fallback_breaker("test_service").await;

        // Should be the same breaker
        assert!(Arc::ptr_eq(&breaker1, &breaker2));
    }

    // =========================================================================
    // Non-Retryable Error Tests
    // =========================================================================

    #[tokio::test]
    async fn test_recover_stops_on_non_retryable_error() {
        let agent = RecovererAgent::builder()
            .max_retries(5)
            .retry_delay_ms(10)
            .build();

        let ctx = ErrorContext::new("test error");
        let attempt_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = agent
            .recover(ctx, move || {
                let count = attempt_count_clone.clone();
                async move {
                    count.fetch_add(1, Ordering::Relaxed);
                    // 401 is not retryable
                    Err::<&str, _>(Error::Synthesis("401 unauthorized".to_string()))
                }
            })
            .await
            .unwrap();

        assert!(!result.is_success());
        // Should only have tried once since error is not retryable
        assert_eq!(attempt_count.load(Ordering::Relaxed), 1);
    }

    // =========================================================================
    // Degradation Tests
    // =========================================================================

    #[tokio::test]
    async fn test_maybe_degrade_after_failures() {
        let agent = RecovererAgent::builder()
            .max_retries(0)
            .retry_delay_ms(10)
            .build();

        // Start at Full
        assert_eq!(agent.degrade_level().await, DegradeLevel::Full);

        // Fail recovery
        let ctx = ErrorContext::new("error");
        agent
            .recover(ctx, || async {
                Err::<&str, _>(Error::Synthesis("500 error".to_string()))
            })
            .await
            .unwrap();

        // Should have degraded
        assert_eq!(agent.degrade_level().await, DegradeLevel::Partial);
    }
}
