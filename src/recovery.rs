//! Error recovery and graceful degradation module for Project Panpsychism.
//!
//! This module provides comprehensive error handling mechanisms including:
//!
//! - **Circuit Breaker**: Prevents cascading failures by temporarily blocking requests
//! - **API Fallback**: Automatic model degradation when primary model fails
//! - **Partial Results**: Return what we can even when some operations fail
//! - **Timeout Wrappers**: Configurable timeouts with graceful handling
//! - **Graceful Shutdown**: Save state before termination
//!
//! # Architecture
//!
//! The recovery system implements the "fail gracefully" principle:
//!
//! ```text
//! Request
//!    |
//!    v
//! +------------------+
//! | Circuit Breaker  |  <-- Block if too many failures
//! +------------------+
//!    |
//!    v
//! +------------------+
//! | Timeout Wrapper  |  <-- Enforce time limits
//! +------------------+
//!    |
//!    v
//! +------------------+
//! |  Primary Call    |  <-- Try main operation
//! +------------------+
//!    |
//!    v (on failure)
//! +------------------+
//! | Fallback Chain   |  <-- Try alternative models/methods
//! +------------------+
//!    |
//!    v
//! +------------------+
//! | Partial Results  |  <-- Return what succeeded
//! +------------------+
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::recovery::{CircuitBreaker, with_fallback, with_timeout};
//! use std::time::Duration;
//!
//! // Create a circuit breaker
//! let breaker = CircuitBreaker::new(5, Duration::from_secs(30));
//!
//! // Call with fallback and timeout
//! let result = with_timeout(
//!     Duration::from_secs(10),
//!     with_fallback(&client, prompt, &[GeminiModel::Flash3, GeminiModel::Flash])
//! ).await?;
//! ```

use crate::{Error, Result};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

// =============================================================================
// CIRCUIT BREAKER
// =============================================================================

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are blocked
    Open,
    /// Circuit is half-open, testing if service recovered
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

/// Circuit breaker for preventing cascading failures.
///
/// The circuit breaker pattern prevents an application from repeatedly
/// trying to execute an operation that's likely to fail.
///
/// # States
///
/// - **Closed**: Normal operation, failures are counted
/// - **Open**: Requests are rejected immediately
/// - **Half-Open**: Allowing a test request through
///
/// # Example
///
/// ```rust,ignore
/// let breaker = CircuitBreaker::new(5, Duration::from_secs(30));
///
/// if breaker.is_open() {
///     return Err(Error::Synthesis("Service unavailable".to_string()));
/// }
///
/// match client.call().await {
///     Ok(result) => {
///         breaker.record_success();
///         Ok(result)
///     }
///     Err(e) => {
///         breaker.record_failure();
///         Err(e)
///     }
/// }
/// ```
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Number of consecutive failures
    failures: AtomicU32,
    /// Failure threshold before opening circuit
    threshold: u32,
    /// Duration to wait before trying again
    reset_after: Duration,
    /// Timestamp of last failure
    last_failure: Mutex<Option<Instant>>,
    /// Whether circuit is forcibly open
    force_open: AtomicBool,
    /// Current state (for half-open logic)
    state: Mutex<CircuitState>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Number of failures before opening the circuit
    /// * `reset_after` - Duration to wait before attempting recovery
    pub fn new(threshold: u32, reset_after: Duration) -> Self {
        Self {
            failures: AtomicU32::new(0),
            threshold,
            reset_after,
            last_failure: Mutex::new(None),
            force_open: AtomicBool::new(false),
            state: Mutex::new(CircuitState::Closed),
        }
    }

    /// Create a circuit breaker with default settings.
    ///
    /// Defaults: 5 failures threshold, 30 second reset window.
    pub fn default_config() -> Self {
        Self::new(5, Duration::from_secs(30))
    }

    /// Check if the circuit is currently open (blocking requests).
    pub async fn is_open(&self) -> bool {
        // Check force open flag
        if self.force_open.load(Ordering::Relaxed) {
            return true;
        }

        let failures = self.failures.load(Ordering::Relaxed);

        // Below threshold means closed
        if failures < self.threshold {
            return false;
        }

        // Check if enough time has passed for half-open
        let last_failure = self.last_failure.lock().await;
        if let Some(last) = *last_failure {
            if last.elapsed() >= self.reset_after {
                // Time for half-open state
                drop(last_failure);
                let mut state = self.state.lock().await;
                *state = CircuitState::HalfOpen;
                return false; // Allow one request through
            }
        }

        true
    }

    /// Check if circuit allows requests (inverse of is_open).
    pub async fn allows_request(&self) -> bool {
        !self.is_open().await
    }

    /// Get the current state of the circuit.
    pub async fn state(&self) -> CircuitState {
        if self.force_open.load(Ordering::Relaxed) {
            return CircuitState::Open;
        }

        let failures = self.failures.load(Ordering::Relaxed);
        if failures < self.threshold {
            return CircuitState::Closed;
        }

        let last_failure = self.last_failure.lock().await;
        if let Some(last) = *last_failure {
            if last.elapsed() >= self.reset_after {
                return CircuitState::HalfOpen;
            }
        }

        CircuitState::Open
    }

    /// Record a successful operation.
    ///
    /// Resets the failure counter and closes the circuit.
    pub async fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
        let mut state = self.state.lock().await;
        *state = CircuitState::Closed;
        debug!("Circuit breaker: success recorded, circuit closed");
    }

    /// Record a failed operation.
    ///
    /// Increments the failure counter and may open the circuit.
    pub async fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        let mut last_failure = self.last_failure.lock().await;
        *last_failure = Some(Instant::now());

        if failures >= self.threshold {
            let mut state = self.state.lock().await;
            *state = CircuitState::Open;
            warn!("Circuit breaker: {} failures, circuit opened", failures);
        } else {
            debug!(
                "Circuit breaker: failure recorded ({}/{})",
                failures, self.threshold
            );
        }
    }

    /// Get the current failure count.
    pub fn failure_count(&self) -> u32 {
        self.failures.load(Ordering::Relaxed)
    }

    /// Get the failure threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Forcibly open the circuit.
    pub fn force_open(&self) {
        self.force_open.store(true, Ordering::Relaxed);
        warn!("Circuit breaker: forcibly opened");
    }

    /// Clear the force open flag.
    pub fn clear_force_open(&self) {
        self.force_open.store(false, Ordering::Relaxed);
        info!("Circuit breaker: force open cleared");
    }

    /// Reset the circuit breaker to initial state.
    pub async fn reset(&self) {
        self.failures.store(0, Ordering::Relaxed);
        self.force_open.store(false, Ordering::Relaxed);
        let mut last_failure = self.last_failure.lock().await;
        *last_failure = None;
        let mut state = self.state.lock().await;
        *state = CircuitState::Closed;
        info!("Circuit breaker: reset to initial state");
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::default_config()
    }
}

// =============================================================================
// TIMEOUT WRAPPER
// =============================================================================

/// Error types for timeout operations.
#[derive(Debug, Clone)]
pub enum TimeoutError {
    /// Operation timed out
    Timeout(Duration),
    /// Operation was cancelled
    Cancelled,
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeoutError::Timeout(d) => write!(f, "operation timed out after {:?}", d),
            TimeoutError::Cancelled => write!(f, "operation was cancelled"),
        }
    }
}

impl std::error::Error for TimeoutError {}

/// Execute a future with a timeout.
///
/// # Arguments
///
/// * `duration` - Maximum time to wait
/// * `future` - The async operation to execute
///
/// # Returns
///
/// The result of the future, or a timeout error.
///
/// # Example
///
/// ```rust,ignore
/// use std::time::Duration;
/// use panpsychism::recovery::with_timeout;
///
/// let result = with_timeout(
///     Duration::from_secs(10),
///     async { client.generate("prompt").await }
/// ).await?;
/// ```
pub async fn with_timeout<T, F: Future<Output = Result<T>>>(
    duration: Duration,
    future: F,
) -> Result<T> {
    tokio::time::timeout(duration, future)
        .await
        .map_err(|_| Error::Synthesis(format!("Request timed out after {:?}", duration)))?
}

/// Execute a future with a timeout, returning None on timeout instead of error.
///
/// Useful when timeout is an acceptable outcome.
pub async fn with_timeout_optional<T, F: Future<Output = Result<T>>>(
    duration: Duration,
    future: F,
) -> Option<Result<T>> {
    tokio::time::timeout(duration, future).await.ok()
}

/// Execute multiple futures with individual timeouts.
///
/// Returns results for all futures that completed within their timeout.
pub async fn with_timeouts<T, F>(operations: Vec<(Duration, F)>) -> Vec<(usize, Result<T>)>
where
    F: Future<Output = Result<T>>,
{
    use futures::future::join_all;

    let indexed_futures: Vec<_> = operations
        .into_iter()
        .enumerate()
        .map(|(idx, (duration, future))| async move {
            let result = tokio::time::timeout(duration, future).await;
            match result {
                Ok(r) => Some((idx, r)),
                Err(_) => None, // Timed out
            }
        })
        .collect();

    let results = join_all(indexed_futures).await;
    results.into_iter().flatten().collect()
}

// =============================================================================
// PARTIAL RESULTS
// =============================================================================

/// Container for partial results when some operations succeed and others fail.
///
/// # Example
///
/// ```rust,ignore
/// let mut partial = PartialResults::new();
///
/// for source in sources {
///     match search(source).await {
///         Ok(results) => partial.add_results(results),
///         Err(e) => partial.add_error(format!("Source {} failed: {}", source.name, e)),
///     }
/// }
///
/// if !partial.is_empty() {
///     println!("Found {} results (partial: {})", partial.results.len(), partial.is_partial());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PartialResults<T> {
    /// Successfully retrieved results
    pub results: Vec<T>,
    /// Whether results are incomplete due to errors
    pub partial: bool,
    /// Errors that occurred during retrieval
    pub errors: Vec<String>,
    /// Number of sources that succeeded
    pub sources_succeeded: usize,
    /// Number of sources that failed
    pub sources_failed: usize,
}

impl<T> PartialResults<T> {
    /// Create a new empty partial results container.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            partial: false,
            errors: Vec::new(),
            sources_succeeded: 0,
            sources_failed: 0,
        }
    }

    /// Create from a complete result set (no errors).
    pub fn complete(results: Vec<T>) -> Self {
        let count = results.len();
        Self {
            results,
            partial: false,
            errors: Vec::new(),
            sources_succeeded: if count > 0 { 1 } else { 0 },
            sources_failed: 0,
        }
    }

    /// Add results from a successful operation.
    pub fn add_results(&mut self, mut results: Vec<T>) {
        self.results.append(&mut results);
        self.sources_succeeded += 1;
    }

    /// Add a single result.
    pub fn add_result(&mut self, result: T) {
        self.results.push(result);
    }

    /// Record an error from a failed operation.
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
        self.partial = true;
        self.sources_failed += 1;
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Check if results are partial (some errors occurred).
    pub fn is_partial(&self) -> bool {
        self.partial
    }

    /// Check if all operations failed.
    pub fn all_failed(&self) -> bool {
        self.sources_succeeded == 0 && self.sources_failed > 0
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Get success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.sources_succeeded + self.sources_failed;
        if total == 0 {
            100.0
        } else {
            (self.sources_succeeded as f64 / total as f64) * 100.0
        }
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        if self.partial {
            format!(
                "{} results (partial, {} errors, {:.0}% success)",
                self.results.len(),
                self.errors.len(),
                self.success_rate()
            )
        } else {
            format!("{} results (complete)", self.results.len())
        }
    }
}

impl<T> Default for PartialResults<T> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// API FALLBACK
// =============================================================================

/// Configuration for API fallback behavior.
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Models to try in order (primary first)
    pub models: Vec<String>,
    /// Whether to retry on rate limit errors
    pub retry_on_rate_limit: bool,
    /// Delay between fallback attempts
    pub fallback_delay: Duration,
    /// Maximum total attempts across all models
    pub max_attempts: usize,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            models: vec!["gemini-3-flash".to_string(), "gemini-2.0-flash".to_string()],
            retry_on_rate_limit: true,
            fallback_delay: Duration::from_millis(500),
            max_attempts: 6,
        }
    }
}

impl FallbackConfig {
    /// Create a new fallback config with specified models.
    pub fn with_models(models: Vec<String>) -> Self {
        Self {
            models,
            ..Default::default()
        }
    }

    /// Set retry on rate limit behavior.
    pub fn retry_on_rate_limit(mut self, retry: bool) -> Self {
        self.retry_on_rate_limit = retry;
        self
    }

    /// Set delay between fallback attempts.
    pub fn fallback_delay(mut self, delay: Duration) -> Self {
        self.fallback_delay = delay;
        self
    }
}

/// Result of a fallback operation.
#[derive(Debug)]
pub struct FallbackResult<T> {
    /// The successful result
    pub value: T,
    /// Which model succeeded
    pub model_used: String,
    /// How many attempts were made
    pub attempts: usize,
    /// Whether fallback was needed
    pub used_fallback: bool,
    /// Errors from failed attempts
    pub errors: Vec<String>,
}

impl<T> FallbackResult<T> {
    /// Create a result from primary model (no fallback needed).
    pub fn primary(value: T, model: String) -> Self {
        Self {
            value,
            model_used: model,
            attempts: 1,
            used_fallback: false,
            errors: Vec::new(),
        }
    }

    /// Create a result from fallback model.
    pub fn fallback(value: T, model: String, attempts: usize, errors: Vec<String>) -> Self {
        Self {
            value,
            model_used: model,
            attempts,
            used_fallback: true,
            errors,
        }
    }
}

/// Check if an error indicates rate limiting.
pub fn is_rate_limit_error(error: &Error) -> bool {
    match error {
        Error::Synthesis(msg) => {
            msg.contains("429")
                || msg.contains("rate limit")
                || msg.contains("Rate limit")
                || msg.contains("quota")
                || msg.contains("too many requests")
        }
        Error::Http(e) => e.status().map(|s| s.as_u16() == 429).unwrap_or(false),
        _ => false,
    }
}

/// Check if an error is retryable.
pub fn is_retryable_error(error: &Error) -> bool {
    match error {
        Error::Synthesis(msg) => {
            // Rate limits
            if msg.contains("429") || msg.contains("rate limit") || msg.contains("quota") {
                return true;
            }
            // Server errors
            if msg.contains("500")
                || msg.contains("502")
                || msg.contains("503")
                || msg.contains("504")
            {
                return true;
            }
            // Timeouts
            if msg.contains("timeout") || msg.contains("timed out") {
                return true;
            }
            false
        }
        Error::Http(e) => {
            if let Some(status) = e.status() {
                let code = status.as_u16();
                // Retry on 429 (rate limit), 500, 502, 503, 504
                return code == 429 || (500..=504).contains(&code);
            }
            // Also retry on connection errors
            e.is_connect() || e.is_timeout()
        }
        Error::Io(_) => true, // I/O errors are often transient
        _ => false,
    }
}

// =============================================================================
// GRACEFUL SHUTDOWN
// =============================================================================

/// State that can be saved during graceful shutdown.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ShutdownState {
    /// Current session ID
    pub session_id: Option<String>,
    /// Pending question (if any)
    pub pending_question: Option<String>,
    /// Partial response (if generation was interrupted)
    pub partial_response: Option<String>,
    /// Timestamp of shutdown
    pub timestamp: Option<String>,
    /// Any additional context to preserve
    pub context: std::collections::HashMap<String, String>,
}

impl ShutdownState {
    /// Create a new empty shutdown state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the session ID.
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set a pending question.
    pub fn with_question(mut self, question: impl Into<String>) -> Self {
        self.pending_question = Some(question.into());
        self
    }

    /// Set a partial response.
    pub fn with_partial_response(mut self, response: impl Into<String>) -> Self {
        self.partial_response = Some(response.into());
        self
    }

    /// Add context data.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Update timestamp to now.
    pub fn stamp(mut self) -> Self {
        self.timestamp = Some(chrono::Utc::now().to_rfc3339());
        self
    }

    /// Check if there's anything worth saving.
    pub fn has_data(&self) -> bool {
        self.pending_question.is_some()
            || self.partial_response.is_some()
            || !self.context.is_empty()
    }
}

/// Manager for graceful shutdown operations.
pub struct ShutdownManager {
    /// Path to save state
    state_path: std::path::PathBuf,
    /// Current state
    state: Mutex<ShutdownState>,
    /// Whether shutdown has been initiated
    shutdown_initiated: AtomicBool,
}

impl ShutdownManager {
    /// Create a new shutdown manager.
    pub fn new(state_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            state_path: state_path.into(),
            state: Mutex::new(ShutdownState::new()),
            shutdown_initiated: AtomicBool::new(false),
        }
    }

    /// Create with default path (~/.panpsychism_state.json).
    pub fn default_path() -> Self {
        let path = dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".panpsychism_state.json");
        Self::new(path)
    }

    /// Update the current state.
    pub async fn update_state(&self, state: ShutdownState) {
        let mut current = self.state.lock().await;
        *current = state;
    }

    /// Update pending question.
    pub async fn set_pending_question(&self, question: impl Into<String>) {
        let mut state = self.state.lock().await;
        state.pending_question = Some(question.into());
    }

    /// Update partial response.
    pub async fn set_partial_response(&self, response: impl Into<String>) {
        let mut state = self.state.lock().await;
        state.partial_response = Some(response.into());
    }

    /// Clear pending question.
    pub async fn clear_pending(&self) {
        let mut state = self.state.lock().await;
        state.pending_question = None;
        state.partial_response = None;
    }

    /// Save current state to disk.
    pub async fn save_state(&self) -> Result<()> {
        let state = self.state.lock().await;

        if !state.has_data() {
            debug!("No state to save");
            return Ok(());
        }

        let state_with_timestamp = state.clone().stamp();

        let json = serde_json::to_string_pretty(&state_with_timestamp)
            .map_err(|e| Error::Synthesis(format!("Failed to serialize state: {}", e)))?;

        tokio::fs::write(&self.state_path, json)
            .await
            .map_err(Error::Io)?;

        info!("State saved to {:?}", self.state_path);
        Ok(())
    }

    /// Load state from disk.
    pub async fn load_state(&self) -> Result<Option<ShutdownState>> {
        if !self.state_path.exists() {
            return Ok(None);
        }

        let json = tokio::fs::read_to_string(&self.state_path)
            .await
            .map_err(Error::Io)?;

        let state: ShutdownState = serde_json::from_str(&json)
            .map_err(|e| Error::Synthesis(format!("Failed to parse state: {}", e)))?;

        Ok(Some(state))
    }

    /// Clear saved state file.
    pub async fn clear_state(&self) -> Result<()> {
        if self.state_path.exists() {
            tokio::fs::remove_file(&self.state_path)
                .await
                .map_err(Error::Io)?;
            info!("State file cleared");
        }
        Ok(())
    }

    /// Initiate graceful shutdown.
    pub async fn initiate_shutdown(&self) -> Result<()> {
        if self.shutdown_initiated.swap(true, Ordering::Relaxed) {
            // Already initiated
            return Ok(());
        }

        info!("Initiating graceful shutdown...");
        self.save_state().await?;
        Ok(())
    }

    /// Check if shutdown was initiated.
    pub fn is_shutdown_initiated(&self) -> bool {
        self.shutdown_initiated.load(Ordering::Relaxed)
    }
}

/// Create a shutdown handler that saves state on Ctrl+C.
///
/// # Example
///
/// ```rust,ignore
/// let manager = Arc::new(ShutdownManager::default_path());
/// setup_shutdown_handler(manager.clone());
///
/// // Main loop
/// loop {
///     tokio::select! {
///         result = process_question(&question) => { ... }
///         _ = tokio::signal::ctrl_c() => {
///             manager.initiate_shutdown().await?;
///             break;
///         }
///     }
/// }
/// ```
pub fn setup_shutdown_handler(manager: Arc<ShutdownManager>) {
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            warn!("Failed to listen for Ctrl+C: {}", e);
            return;
        }

        info!("Received shutdown signal");
        if let Err(e) = manager.initiate_shutdown().await {
            warn!("Failed to save state on shutdown: {}", e);
        }
    });
}

// =============================================================================
// RETRY UTILITIES
// =============================================================================

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Add jitter to delays
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Calculate delay for a given attempt.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_delay =
            self.initial_delay.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);

        let delay_secs = base_delay.min(self.max_delay.as_secs_f64());

        let final_delay = if self.jitter {
            // Add up to 25% jitter
            let jitter_factor = 1.0 + (rand_jitter() * 0.25);
            delay_secs * jitter_factor
        } else {
            delay_secs
        };

        Duration::from_secs_f64(final_delay)
    }
}

/// Simple pseudo-random jitter (0.0 to 1.0).
fn rand_jitter() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

/// Execute a function with retries.
///
/// # Arguments
///
/// * `config` - Retry configuration
/// * `operation` - The async function to retry
///
/// # Returns
///
/// The result of the first successful attempt, or the last error.
pub async fn with_retry<T, F, Fut>(config: &RetryConfig, mut operation: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        if attempt > 0 {
            let delay = config.delay_for_attempt(attempt - 1);
            debug!("Retry attempt {}, waiting {:?}", attempt, delay);
            tokio::time::sleep(delay).await;
        }

        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if !is_retryable_error(&e) {
                    warn!("Non-retryable error: {}", e);
                    return Err(e);
                }
                debug!("Attempt {} failed: {}", attempt + 1, e);
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| Error::Synthesis("Max retries exceeded".to_string())))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // Circuit Breaker Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_circuit_breaker_initial_state() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));
        assert_eq!(breaker.state().await, CircuitState::Closed);
        assert!(breaker.allows_request().await);
        assert_eq!(breaker.failure_count(), 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_threshold() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));

        // Record failures up to threshold
        for _ in 0..3 {
            breaker.record_failure().await;
        }

        assert_eq!(breaker.failure_count(), 3);
        assert_eq!(breaker.state().await, CircuitState::Open);
        assert!(!breaker.allows_request().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_success_resets() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));

        // Record some failures
        breaker.record_failure().await;
        breaker.record_failure().await;
        assert_eq!(breaker.failure_count(), 2);

        // Success should reset
        breaker.record_success().await;
        assert_eq!(breaker.failure_count(), 0);
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_force_open() {
        let breaker = CircuitBreaker::new(5, Duration::from_secs(10));

        assert!(breaker.allows_request().await);

        breaker.force_open();
        assert!(!breaker.allows_request().await);

        breaker.clear_force_open();
        assert!(breaker.allows_request().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reset() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));

        // Open the circuit
        for _ in 0..3 {
            breaker.record_failure().await;
        }
        breaker.force_open();

        assert!(!breaker.allows_request().await);

        // Reset
        breaker.reset().await;
        assert!(breaker.allows_request().await);
        assert_eq!(breaker.failure_count(), 0);
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    // ==========================================================================
    // Timeout Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_with_timeout_success() {
        let result =
            with_timeout(Duration::from_secs(1), async { Ok::<_, Error>("success") }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_with_timeout_failure() {
        let result = with_timeout(Duration::from_millis(10), async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            Ok::<_, Error>("success")
        })
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn test_with_timeout_optional() {
        // Should return None on timeout
        let result = with_timeout_optional(Duration::from_millis(10), async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            Ok::<_, Error>("success")
        })
        .await;

        assert!(result.is_none());

        // Should return Some on success
        let result =
            with_timeout_optional(Duration::from_secs(1), async { Ok::<_, Error>("success") })
                .await;

        assert!(result.is_some());
    }

    // ==========================================================================
    // Partial Results Tests
    // ==========================================================================

    #[test]
    fn test_partial_results_complete() {
        let results = PartialResults::complete(vec![1, 2, 3]);
        assert!(!results.is_partial());
        assert!(!results.is_empty());
        assert_eq!(results.len(), 3);
        assert_eq!(results.success_rate(), 100.0);
    }

    #[test]
    fn test_partial_results_with_errors() {
        let mut results = PartialResults::<i32>::new();
        results.add_results(vec![1, 2]);
        results.add_error("Source A failed");
        results.add_results(vec![3]);
        results.add_error("Source B failed");

        assert!(results.is_partial());
        assert_eq!(results.len(), 3);
        assert_eq!(results.errors.len(), 2);
        assert_eq!(results.sources_succeeded, 2);
        assert_eq!(results.sources_failed, 2);
        assert_eq!(results.success_rate(), 50.0);
    }

    #[test]
    fn test_partial_results_all_failed() {
        let mut results = PartialResults::<i32>::new();
        results.add_error("Error 1");
        results.add_error("Error 2");

        assert!(results.is_partial());
        assert!(results.is_empty());
        assert!(results.all_failed());
    }

    #[test]
    fn test_partial_results_summary() {
        let mut results = PartialResults::<i32>::new();
        results.add_results(vec![1, 2, 3]);
        results.add_error("Error");

        let summary = results.summary();
        assert!(summary.contains("3 results"));
        assert!(summary.contains("partial"));
    }

    // ==========================================================================
    // Error Classification Tests
    // ==========================================================================

    #[test]
    fn test_is_rate_limit_error() {
        let rate_limit = Error::Synthesis("API error 429: rate limit exceeded".to_string());
        assert!(is_rate_limit_error(&rate_limit));

        let other = Error::Synthesis("API error 500: internal error".to_string());
        assert!(!is_rate_limit_error(&other));
    }

    #[test]
    fn test_is_retryable_error() {
        // Rate limit is retryable
        let rate_limit = Error::Synthesis("429 rate limit".to_string());
        assert!(is_retryable_error(&rate_limit));

        // Server errors are retryable
        let server_error = Error::Synthesis("500 internal error".to_string());
        assert!(is_retryable_error(&server_error));

        // Timeout is retryable
        let timeout = Error::Synthesis("request timed out".to_string());
        assert!(is_retryable_error(&timeout));

        // Auth errors are not retryable
        let auth = Error::Synthesis("401 unauthorized".to_string());
        assert!(!is_retryable_error(&auth));
    }

    // ==========================================================================
    // Retry Config Tests
    // ==========================================================================

    #[test]
    fn test_retry_config_delay_calculation() {
        let config = RetryConfig {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: false,
            ..Default::default()
        };

        // Without jitter, delays should be predictable
        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(4));
        assert_eq!(config.delay_for_attempt(3), Duration::from_secs(8));
        assert_eq!(config.delay_for_attempt(4), Duration::from_secs(16));
        // Should cap at max_delay
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(30));
    }

    // ==========================================================================
    // Shutdown State Tests
    // ==========================================================================

    #[test]
    fn test_shutdown_state_builder() {
        let state = ShutdownState::new()
            .with_session("session-123")
            .with_question("What is Spinoza?")
            .with_partial_response("Spinoza was a philosopher...")
            .with_context("model", "gemini-3-flash")
            .stamp();

        assert_eq!(state.session_id, Some("session-123".to_string()));
        assert_eq!(state.pending_question, Some("What is Spinoza?".to_string()));
        assert!(state.has_data());
        assert!(state.timestamp.is_some());
    }

    #[test]
    fn test_shutdown_state_has_data() {
        let empty = ShutdownState::new();
        assert!(!empty.has_data());

        let with_question = ShutdownState::new().with_question("test");
        assert!(with_question.has_data());

        let with_context = ShutdownState::new().with_context("key", "value");
        assert!(with_context.has_data());
    }

    // ==========================================================================
    // Fallback Config Tests
    // ==========================================================================

    #[test]
    fn test_fallback_config_default() {
        let config = FallbackConfig::default();
        assert!(config.retry_on_rate_limit);
        assert!(!config.models.is_empty());
    }

    #[test]
    fn test_fallback_result() {
        let primary: FallbackResult<&str> =
            FallbackResult::primary("result", "gemini-3-flash".to_string());
        assert!(!primary.used_fallback);
        assert_eq!(primary.attempts, 1);

        let fallback: FallbackResult<&str> = FallbackResult::fallback(
            "result",
            "gemini-2.0-flash".to_string(),
            3,
            vec!["Error 1".to_string()],
        );
        assert!(fallback.used_fallback);
        assert_eq!(fallback.attempts, 3);
    }

    // ==========================================================================
    // Integration Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_circuit_breaker_half_open() {
        let breaker = CircuitBreaker::new(2, Duration::from_millis(100));

        // Open the circuit
        breaker.record_failure().await;
        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        // Wait for reset window
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be half-open now
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);
        assert!(breaker.allows_request().await);

        // Success should close it
        breaker.record_success().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_with_retry_success_first_try() {
        let config = RetryConfig::default();
        let result = with_retry(&config, || async { Ok::<_, Error>("success") }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_with_retry_eventual_success() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(10),
            ..Default::default()
        };

        let attempt_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = with_retry(&config, || {
            let count = attempt_count_clone.clone();
            async move {
                let attempt = count.fetch_add(1, Ordering::Relaxed);
                if attempt < 2 {
                    // Fail first two attempts with retryable error
                    Err(Error::Synthesis("503 service unavailable".to_string()))
                } else {
                    Ok::<_, Error>("success")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(attempt_count.load(Ordering::Relaxed), 3);
    }
}
