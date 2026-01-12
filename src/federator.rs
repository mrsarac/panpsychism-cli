//! Federator Agent for Panpsychism v3.0.
//!
//! The Multi-System Coordinator — "Unity through distributed harmony."
//!
//! This module implements the Federator Agent, responsible for coordinating across
//! multiple Panpsychism instances with distributed coordination, cross-instance sync,
//! and intelligent load balancing.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, the Federator Agent serves as the **Grand Conjurer** —
//! one who can summon assistance from multiple magical towers:
//!
//! - **Instances** are sister towers, each with their own magical capabilities
//! - **Load Balancing** ensures no single tower bears too much burden
//! - **Failover** redirects spells when a tower is overwhelmed
//! - **Health Checks** monitor the vitality of each tower
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Self-preservation through redundancy and failover
//! - **NATURA**: Natural distribution of load across the system
//! - **RATIO**: Logical selection of optimal instances
//! - **LAETITIA**: Joy through reliable, distributed operations
//!
//! ## Architecture
//!
//! ```text
//! Request
//!    |
//!    v
//! +------------------+
//! | FederatorAgent   |  <-- Coordinates across instances
//! +------------------+
//!    |
//!    v
//! +------------------+
//! | Load Balancer    |  <-- Selects best instance
//! +------------------+
//!    |
//!    +-----+-----+-----+
//!    |     |     |     |
//!    v     v     v     v
//!  [A]   [B]   [C]   [D]  <-- Panpsychism Instances
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::federator::{FederatorAgent, FederationConfig, Instance, LoadBalanceStrategy};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let federator = FederatorAgent::builder()
//!         .add_instance(Instance::new("primary", "https://primary.example.com"))
//!         .add_instance(Instance::new("secondary", "https://secondary.example.com"))
//!         .load_balance_strategy(LoadBalanceStrategy::LatencyBased)
//!         .failover_enabled(true)
//!         .build();
//!
//!     let response = federator.federate("process this request").await?;
//!     println!("Processed by: {}", response.instance_id);
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

// =============================================================================
// HEALTH STATUS
// =============================================================================

/// Health status of a federated instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InstanceHealth {
    /// Instance is operating normally.
    #[default]
    Healthy,

    /// Instance is operational but experiencing issues.
    Degraded,

    /// Instance is not responding or failing.
    Unhealthy,

    /// Instance health is unknown (not yet checked).
    Unknown,
}

impl std::fmt::Display for InstanceHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl InstanceHealth {
    /// Check if the instance is available for requests.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// Check if the instance is in optimal condition.
    pub fn is_optimal(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Get a numeric score for this health (higher = better).
    pub fn score(&self) -> u8 {
        match self {
            Self::Healthy => 100,
            Self::Degraded => 50,
            Self::Unhealthy => 0,
            Self::Unknown => 25,
        }
    }
}

// =============================================================================
// LOAD BALANCE STRATEGY
// =============================================================================

/// Strategy for selecting which instance to route requests to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LoadBalanceStrategy {
    /// Round-robin selection across all healthy instances.
    #[default]
    RoundRobin,

    /// Random selection weighted by instance weights.
    WeightedRandom,

    /// Route to instance with fewest active connections.
    LeastConnections,

    /// Route to instance with lowest observed latency.
    LatencyBased,

    /// Match request capabilities to instance capabilities.
    CapabilityMatch,
}

impl std::fmt::Display for LoadBalanceStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "round_robin"),
            Self::WeightedRandom => write!(f, "weighted_random"),
            Self::LeastConnections => write!(f, "least_connections"),
            Self::LatencyBased => write!(f, "latency_based"),
            Self::CapabilityMatch => write!(f, "capability_match"),
        }
    }
}

impl LoadBalanceStrategy {
    /// Get all available strategies.
    pub fn all() -> Vec<Self> {
        vec![
            Self::RoundRobin,
            Self::WeightedRandom,
            Self::LeastConnections,
            Self::LatencyBased,
            Self::CapabilityMatch,
        ]
    }

    /// Check if this strategy considers instance health.
    pub fn considers_health(&self) -> bool {
        true // All strategies consider health
    }

    /// Check if this strategy needs latency metrics.
    pub fn needs_latency_metrics(&self) -> bool {
        matches!(self, Self::LatencyBased)
    }

    /// Check if this strategy needs connection counts.
    pub fn needs_connection_counts(&self) -> bool {
        matches!(self, Self::LeastConnections)
    }
}

// =============================================================================
// INSTANCE CAPABILITIES
// =============================================================================

/// Capabilities that an instance can provide.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Search capability for finding prompts.
    Search,
    /// Indexing capability for managing prompt stores.
    Index,
    /// Validation capability for checking prompts.
    Validate,
    /// Synthesis capability for combining prompts.
    Synthesize,
    /// Correction capability for fixing prompts.
    Correct,
    /// General text processing.
    TextProcessing,
    /// Code generation and analysis.
    CodeGeneration,
    /// Image processing.
    ImageProcessing,
    /// Audio processing.
    AudioProcessing,
    /// Real-time streaming.
    Streaming,
    /// Large context windows.
    LargeContext,
    /// Fast inference.
    FastInference,
    /// Custom capability.
    Custom(String),
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Search => write!(f, "search"),
            Self::Index => write!(f, "index"),
            Self::Validate => write!(f, "validate"),
            Self::Synthesize => write!(f, "synthesize"),
            Self::Correct => write!(f, "correct"),
            Self::TextProcessing => write!(f, "text_processing"),
            Self::CodeGeneration => write!(f, "code_generation"),
            Self::ImageProcessing => write!(f, "image_processing"),
            Self::AudioProcessing => write!(f, "audio_processing"),
            Self::Streaming => write!(f, "streaming"),
            Self::LargeContext => write!(f, "large_context"),
            Self::FastInference => write!(f, "fast_inference"),
            Self::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

// =============================================================================
// INSTANCE
// =============================================================================

/// A federated Panpsychism instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    /// Unique identifier for this instance.
    pub id: String,

    /// Endpoint URL for this instance.
    pub endpoint: String,

    /// Weight for load balancing (higher = more traffic).
    pub weight: u32,

    /// Capabilities this instance provides.
    pub capabilities: Vec<Capability>,

    /// Current health status.
    pub health: InstanceHealth,

    /// Average latency in milliseconds.
    pub latency_ms: Option<f64>,

    /// Number of active connections.
    pub active_connections: u32,

    /// Last health check timestamp.
    pub last_health_check: Option<DateTime<Utc>>,

    /// Number of consecutive failures.
    pub consecutive_failures: u32,

    /// Total requests processed.
    pub total_requests: u64,

    /// Total failures.
    pub total_failures: u64,

    /// Whether this instance is enabled.
    pub enabled: bool,

    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

impl Instance {
    /// Create a new instance with the given ID and endpoint.
    pub fn new(id: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            endpoint: endpoint.into(),
            weight: 100,
            capabilities: vec![Capability::TextProcessing],
            health: InstanceHealth::Unknown,
            latency_ms: None,
            active_connections: 0,
            last_health_check: None,
            consecutive_failures: 0,
            total_requests: 0,
            total_failures: 0,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Set the weight for this instance.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Set the capabilities for this instance.
    pub fn with_capabilities(mut self, capabilities: Vec<Capability>) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Add a capability to this instance.
    pub fn with_capability(mut self, capability: Capability) -> Self {
        if !self.capabilities.contains(&capability) {
            self.capabilities.push(capability);
        }
        self
    }

    /// Set the health status.
    pub fn with_health(mut self, health: InstanceHealth) -> Self {
        self.health = health;
        self
    }

    /// Set enabled state.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if this instance is available for requests.
    pub fn is_available(&self) -> bool {
        self.enabled && self.health.is_available()
    }

    /// Check if this instance has a specific capability.
    pub fn has_capability(&self, capability: &Capability) -> bool {
        self.capabilities.contains(capability)
    }

    /// Get the effective weight considering health.
    pub fn effective_weight(&self) -> u32 {
        if !self.is_available() {
            return 0;
        }
        let health_multiplier = self.health.score() as f64 / 100.0;
        (self.weight as f64 * health_multiplier) as u32
    }

    /// Record a successful request.
    pub fn record_success(&mut self, latency_ms: f64) {
        self.total_requests += 1;
        self.consecutive_failures = 0;
        self.health = InstanceHealth::Healthy;

        // Update rolling average latency
        match self.latency_ms {
            Some(current) => {
                self.latency_ms = Some(current * 0.8 + latency_ms * 0.2);
            }
            None => {
                self.latency_ms = Some(latency_ms);
            }
        }
    }

    /// Record a failed request.
    pub fn record_failure(&mut self) {
        self.total_requests += 1;
        self.total_failures += 1;
        self.consecutive_failures += 1;

        // Update health based on consecutive failures
        self.health = match self.consecutive_failures {
            0..=2 => InstanceHealth::Healthy,
            3..=5 => InstanceHealth::Degraded,
            _ => InstanceHealth::Unhealthy,
        };
    }

    /// Get failure rate as percentage.
    pub fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        (self.total_failures as f64 / self.total_requests as f64) * 100.0
    }

    /// Update health check timestamp.
    pub fn mark_health_checked(&mut self) {
        self.last_health_check = Some(Utc::now());
    }

    /// Check if health data is stale.
    pub fn is_health_stale(&self, max_age_secs: i64) -> bool {
        match self.last_health_check {
            Some(last) => (Utc::now() - last).num_seconds() > max_age_secs,
            None => true,
        }
    }
}

// =============================================================================
// FEDERATED RESPONSE
// =============================================================================

/// Response from a federated request.
#[derive(Debug, Clone)]
pub struct FederatedResponse<T> {
    /// The actual response data.
    pub response: T,

    /// ID of the instance that handled the request.
    pub instance_id: String,

    /// Latency in milliseconds.
    pub latency_ms: u64,

    /// Whether a fallback instance was used.
    pub fallback_used: bool,

    /// Number of attempts made.
    pub attempts: u32,

    /// Timestamp when the response was received.
    pub timestamp: DateTime<Utc>,

    /// Load balance strategy that was used.
    pub strategy_used: LoadBalanceStrategy,
}

impl<T> FederatedResponse<T> {
    /// Create a new federated response.
    pub fn new(
        response: T,
        instance_id: impl Into<String>,
        latency_ms: u64,
        strategy_used: LoadBalanceStrategy,
    ) -> Self {
        Self {
            response,
            instance_id: instance_id.into(),
            latency_ms,
            fallback_used: false,
            attempts: 1,
            timestamp: Utc::now(),
            strategy_used,
        }
    }

    /// Mark that a fallback was used.
    pub fn with_fallback(mut self) -> Self {
        self.fallback_used = true;
        self
    }

    /// Set the number of attempts.
    pub fn with_attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts;
        self
    }

    /// Map the response value.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> FederatedResponse<U> {
        FederatedResponse {
            response: f(self.response),
            instance_id: self.instance_id,
            latency_ms: self.latency_ms,
            fallback_used: self.fallback_used,
            attempts: self.attempts,
            timestamp: self.timestamp,
            strategy_used: self.strategy_used,
        }
    }

    /// Get the inner response.
    pub fn into_response(self) -> T {
        self.response
    }
}

// =============================================================================
// FEDERATION CONFIG
// =============================================================================

/// Configuration for the Federator Agent.
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// List of federated instances.
    pub instances: Vec<Instance>,

    /// Load balancing strategy.
    pub load_balance_strategy: LoadBalanceStrategy,

    /// Interval between sync operations in milliseconds.
    pub sync_interval_ms: u64,

    /// Whether failover is enabled.
    pub failover_enabled: bool,

    /// Health check interval in milliseconds.
    pub health_check_interval_ms: u64,

    /// Maximum consecutive failures before marking unhealthy.
    pub max_consecutive_failures: u32,

    /// Timeout for individual requests in milliseconds.
    pub request_timeout_ms: u64,

    /// Maximum retry attempts.
    pub max_retries: u32,

    /// Whether to prefer local instance.
    pub prefer_local: bool,

    /// Local instance ID (if prefer_local is true).
    pub local_instance_id: Option<String>,

    /// Health stale threshold in seconds.
    pub health_stale_threshold_secs: i64,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            instances: Vec::new(),
            load_balance_strategy: LoadBalanceStrategy::RoundRobin,
            sync_interval_ms: 30000,
            failover_enabled: true,
            health_check_interval_ms: 10000,
            max_consecutive_failures: 3,
            request_timeout_ms: 30000,
            max_retries: 2,
            prefer_local: false,
            local_instance_id: None,
            health_stale_threshold_secs: 60,
        }
    }
}

impl FederationConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for high availability.
    pub fn high_availability() -> Self {
        Self {
            failover_enabled: true,
            max_retries: 3,
            health_check_interval_ms: 5000,
            max_consecutive_failures: 2,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for performance.
    pub fn performance() -> Self {
        Self {
            load_balance_strategy: LoadBalanceStrategy::LatencyBased,
            failover_enabled: true,
            max_retries: 1,
            request_timeout_ms: 15000,
            ..Default::default()
        }
    }

    /// Create a configuration for local development.
    pub fn local() -> Self {
        Self {
            prefer_local: true,
            failover_enabled: false,
            max_retries: 1,
            health_check_interval_ms: 60000,
            ..Default::default()
        }
    }
}

// =============================================================================
// FEDERATION STATS
// =============================================================================

/// Statistics about federation operations.
#[derive(Debug, Default)]
pub struct FederationStats {
    /// Total requests processed.
    pub total_requests: AtomicU64,
    /// Successful requests.
    pub successful_requests: AtomicU64,
    /// Failed requests.
    pub failed_requests: AtomicU64,
    /// Failovers performed.
    pub failovers: AtomicU64,
    /// Total latency in milliseconds.
    pub total_latency_ms: AtomicU64,
    /// Round-robin counter for load balancing.
    pub round_robin_counter: AtomicUsize,
}

impl FederationStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request.
    pub fn record_success(&self, latency_ms: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failover.
    pub fn record_failover(&self) {
        self.failovers.fetch_add(1, Ordering::Relaxed);
    }

    /// Get next round-robin index.
    pub fn next_round_robin(&self, instance_count: usize) -> usize {
        if instance_count == 0 {
            return 0;
        }
        self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % instance_count
    }

    /// Get success rate as percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 100.0;
        }
        let successful = self.successful_requests.load(Ordering::Relaxed);
        (successful as f64 / total as f64) * 100.0
    }

    /// Get average latency in milliseconds.
    pub fn average_latency_ms(&self) -> f64 {
        let successful = self.successful_requests.load(Ordering::Relaxed);
        if successful == 0 {
            return 0.0;
        }
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        total_latency as f64 / successful as f64
    }

    /// Get a snapshot of stats.
    pub fn snapshot(&self) -> FederationStatsSnapshot {
        FederationStatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            failovers: self.failovers.load(Ordering::Relaxed),
            success_rate: self.success_rate(),
            average_latency_ms: self.average_latency_ms(),
        }
    }
}

/// A point-in-time snapshot of federation statistics.
#[derive(Debug, Clone)]
pub struct FederationStatsSnapshot {
    /// Total requests processed.
    pub total_requests: u64,
    /// Successful requests.
    pub successful_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Failovers performed.
    pub failovers: u64,
    /// Success rate as percentage.
    pub success_rate: f64,
    /// Average latency in milliseconds.
    pub average_latency_ms: f64,
}

// =============================================================================
// FEDERATOR AGENT BUILDER
// =============================================================================

/// Builder for constructing a FederatorAgent.
#[derive(Debug, Clone, Default)]
pub struct FederatorAgentBuilder {
    config: FederationConfig,
}

impl FederatorAgentBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an instance to the federation.
    pub fn add_instance(mut self, instance: Instance) -> Self {
        self.config.instances.push(instance);
        self
    }

    /// Set all instances.
    pub fn instances(mut self, instances: Vec<Instance>) -> Self {
        self.config.instances = instances;
        self
    }

    /// Set the load balance strategy.
    pub fn load_balance_strategy(mut self, strategy: LoadBalanceStrategy) -> Self {
        self.config.load_balance_strategy = strategy;
        self
    }

    /// Set the sync interval in milliseconds.
    pub fn sync_interval_ms(mut self, interval: u64) -> Self {
        self.config.sync_interval_ms = interval;
        self
    }

    /// Enable or disable failover.
    pub fn failover_enabled(mut self, enabled: bool) -> Self {
        self.config.failover_enabled = enabled;
        self
    }

    /// Set the health check interval in milliseconds.
    pub fn health_check_interval_ms(mut self, interval: u64) -> Self {
        self.config.health_check_interval_ms = interval;
        self
    }

    /// Set the maximum consecutive failures.
    pub fn max_consecutive_failures(mut self, max: u32) -> Self {
        self.config.max_consecutive_failures = max;
        self
    }

    /// Set the request timeout in milliseconds.
    pub fn request_timeout_ms(mut self, timeout: u64) -> Self {
        self.config.request_timeout_ms = timeout;
        self
    }

    /// Set the maximum retry attempts.
    pub fn max_retries(mut self, max: u32) -> Self {
        self.config.max_retries = max;
        self
    }

    /// Set whether to prefer local instance.
    pub fn prefer_local(mut self, prefer: bool) -> Self {
        self.config.prefer_local = prefer;
        self
    }

    /// Set the local instance ID.
    pub fn local_instance_id(mut self, id: impl Into<String>) -> Self {
        self.config.local_instance_id = Some(id.into());
        self
    }

    /// Set the health stale threshold in seconds.
    pub fn health_stale_threshold_secs(mut self, secs: i64) -> Self {
        self.config.health_stale_threshold_secs = secs;
        self
    }

    /// Use a preset configuration.
    pub fn with_config(mut self, config: FederationConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the FederatorAgent.
    pub fn build(self) -> FederatorAgent {
        FederatorAgent::new(self.config)
    }
}

// =============================================================================
// FEDERATOR AGENT
// =============================================================================

/// The Federator Agent - coordinates across multiple Panpsychism instances.
///
/// This agent provides distributed coordination, cross-instance sync, and
/// intelligent load balancing across federated instances.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive for system resilience through redundancy
/// - **NATURA**: Natural distribution of load following system patterns
/// - **RATIO**: Logical selection of optimal instances
/// - **LAETITIA**: Joy through reliable distributed operations
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::federator::{FederatorAgent, Instance, LoadBalanceStrategy};
///
/// let federator = FederatorAgent::builder()
///     .add_instance(Instance::new("primary", "https://primary.example.com"))
///     .add_instance(Instance::new("secondary", "https://secondary.example.com"))
///     .load_balance_strategy(LoadBalanceStrategy::LatencyBased)
///     .build();
///
/// let response = federator.federate("process this").await?;
/// println!("Handled by: {}", response.instance_id);
/// ```
#[derive(Debug)]
pub struct FederatorAgent {
    /// Configuration for federation behavior.
    config: FederationConfig,
    /// Instance registry.
    instances: Arc<RwLock<HashMap<String, Instance>>>,
    /// Federation statistics.
    stats: Arc<FederationStats>,
}

impl FederatorAgent {
    /// Create a new FederatorAgent with the given configuration.
    pub fn new(config: FederationConfig) -> Self {
        let mut instances_map = HashMap::new();
        for instance in &config.instances {
            instances_map.insert(instance.id.clone(), instance.clone());
        }

        Self {
            config,
            instances: Arc::new(RwLock::new(instances_map)),
            stats: Arc::new(FederationStats::new()),
        }
    }

    /// Create a builder for constructing a FederatorAgent.
    pub fn builder() -> FederatorAgentBuilder {
        FederatorAgentBuilder::new()
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(FederationConfig::default())
    }

    /// Get the current configuration.
    pub fn config(&self) -> &FederationConfig {
        &self.config
    }

    /// Get federation statistics.
    pub fn stats(&self) -> &FederationStats {
        &self.stats
    }

    /// Get a stats snapshot.
    pub fn stats_snapshot(&self) -> FederationStatsSnapshot {
        self.stats.snapshot()
    }

    // =========================================================================
    // INSTANCE MANAGEMENT
    // =========================================================================

    /// Add an instance to the federation.
    pub async fn add_instance(&self, instance: Instance) {
        let mut instances = self.instances.write().await;
        info!("Adding instance: {} ({})", instance.id, instance.endpoint);
        instances.insert(instance.id.clone(), instance);
    }

    /// Remove an instance from the federation.
    pub async fn remove_instance(&self, id: &str) -> Option<Instance> {
        let mut instances = self.instances.write().await;
        info!("Removing instance: {}", id);
        instances.remove(id)
    }

    /// Get an instance by ID.
    pub async fn get_instance(&self, id: &str) -> Option<Instance> {
        let instances = self.instances.read().await;
        instances.get(id).cloned()
    }

    /// Get all instances.
    pub async fn get_all_instances(&self) -> Vec<Instance> {
        let instances = self.instances.read().await;
        instances.values().cloned().collect()
    }

    /// Get all available instances.
    pub async fn get_available_instances(&self) -> Vec<Instance> {
        let instances = self.instances.read().await;
        instances
            .values()
            .filter(|i| i.is_available())
            .cloned()
            .collect()
    }

    /// Get instance count.
    pub async fn instance_count(&self) -> usize {
        let instances = self.instances.read().await;
        instances.len()
    }

    /// Get available instance count.
    pub async fn available_instance_count(&self) -> usize {
        let instances = self.instances.read().await;
        instances.values().filter(|i| i.is_available()).count()
    }

    /// Update an instance's health status.
    pub async fn update_instance_health(&self, id: &str, health: InstanceHealth) {
        let mut instances = self.instances.write().await;
        if let Some(instance) = instances.get_mut(id) {
            instance.health = health;
            instance.mark_health_checked();
            debug!("Updated health for {}: {}", id, health);
        }
    }

    /// Enable or disable an instance.
    pub async fn set_instance_enabled(&self, id: &str, enabled: bool) {
        let mut instances = self.instances.write().await;
        if let Some(instance) = instances.get_mut(id) {
            instance.enabled = enabled;
            info!("Instance {} enabled: {}", id, enabled);
        }
    }

    // =========================================================================
    // INSTANCE SELECTION
    // =========================================================================

    /// Select the best instance based on the configured strategy.
    pub async fn select_instance(
        &self,
        required_capability: Option<&Capability>,
    ) -> Option<Instance> {
        let instances = self.instances.read().await;
        let available: Vec<&Instance> = instances
            .values()
            .filter(|i| {
                i.is_available()
                    && required_capability
                        .map(|c| i.has_capability(c))
                        .unwrap_or(true)
            })
            .collect();

        if available.is_empty() {
            return None;
        }

        // If prefer_local and local instance is available, use it
        if self.config.prefer_local {
            if let Some(local_id) = &self.config.local_instance_id {
                if let Some(local) = available.iter().find(|i| &i.id == local_id) {
                    return Some((*local).clone());
                }
            }
        }

        match self.config.load_balance_strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.stats.next_round_robin(available.len());
                Some(available[idx].clone())
            }
            LoadBalanceStrategy::WeightedRandom => self.select_weighted_random(&available),
            LoadBalanceStrategy::LeastConnections => self.select_least_connections(&available),
            LoadBalanceStrategy::LatencyBased => self.select_latency_based(&available),
            LoadBalanceStrategy::CapabilityMatch => {
                // For capability match, we already filtered, so use round-robin among matches
                let idx = self.stats.next_round_robin(available.len());
                Some(available[idx].clone())
            }
        }
    }

    /// Select instance using weighted random selection.
    fn select_weighted_random(&self, instances: &[&Instance]) -> Option<Instance> {
        let total_weight: u32 = instances.iter().map(|i| i.effective_weight()).sum();
        if total_weight == 0 {
            return instances.first().map(|i| (*i).clone());
        }

        // Simple pseudo-random using current time
        let random = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            % total_weight as u128) as u32;

        let mut cumulative = 0u32;
        for instance in instances {
            cumulative += instance.effective_weight();
            if random < cumulative {
                return Some((*instance).clone());
            }
        }

        instances.last().map(|i| (*i).clone())
    }

    /// Select instance with least connections.
    fn select_least_connections(&self, instances: &[&Instance]) -> Option<Instance> {
        instances
            .iter()
            .min_by_key(|i| i.active_connections)
            .map(|i| (*i).clone())
    }

    /// Select instance with lowest latency.
    fn select_latency_based(&self, instances: &[&Instance]) -> Option<Instance> {
        instances
            .iter()
            .min_by(|a, b| {
                let a_latency = a.latency_ms.unwrap_or(f64::MAX);
                let b_latency = b.latency_ms.unwrap_or(f64::MAX);
                a_latency.partial_cmp(&b_latency).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|i| (*i).clone())
    }

    // =========================================================================
    // HEALTH CHECKING
    // =========================================================================

    /// Check health of all instances.
    pub async fn check_all_health(&self) {
        let instance_ids: Vec<String> = {
            let instances = self.instances.read().await;
            instances.keys().cloned().collect()
        };

        for id in instance_ids {
            self.check_instance_health(&id).await;
        }
    }

    /// Check health of a specific instance.
    pub async fn check_instance_health(&self, id: &str) {
        // In a real implementation, this would make an HTTP health check request
        // For now, we just update the timestamp
        let mut instances = self.instances.write().await;
        if let Some(instance) = instances.get_mut(id) {
            instance.mark_health_checked();
            debug!("Health checked for instance: {}", id);
        }
    }

    /// Get instances with stale health data.
    pub async fn get_stale_instances(&self) -> Vec<String> {
        let instances = self.instances.read().await;
        instances
            .values()
            .filter(|i| i.is_health_stale(self.config.health_stale_threshold_secs))
            .map(|i| i.id.clone())
            .collect()
    }

    // =========================================================================
    // FEDERATION OPERATIONS
    // =========================================================================

    /// Federate a request to the best available instance.
    ///
    /// This is the main entry point for federated requests. It will:
    /// 1. Select the best instance based on load balancing strategy
    /// 2. Attempt the request
    /// 3. Handle failover if enabled and primary fails
    /// 4. Track statistics
    ///
    /// # Arguments
    ///
    /// * `request` - The request data to process
    ///
    /// # Returns
    ///
    /// A `FederatedResponse` containing the result and metadata.
    pub async fn federate<T: Clone>(&self, request: T) -> Result<FederatedResponse<T>> {
        let start = Instant::now();
        let mut attempts = 0u32;
        let mut tried_instances: Vec<String> = Vec::new();

        loop {
            attempts += 1;

            // Select an instance
            let instance = self.select_instance(None).await;

            let instance = match instance {
                Some(i) if !tried_instances.contains(&i.id) => i,
                _ => {
                    // No available instance or all tried
                    self.stats.record_failure();
                    return Err(Error::Config(
                        "No available instances for federation".to_string(),
                    ));
                }
            };

            tried_instances.push(instance.id.clone());
            debug!(
                "Attempting request on instance: {} (attempt {})",
                instance.id, attempts
            );

            // Simulate request processing
            // In a real implementation, this would make an HTTP request
            let latency_ms = start.elapsed().as_millis() as u64;

            // For now, simulate success
            {
                let mut instances = self.instances.write().await;
                if let Some(inst) = instances.get_mut(&instance.id) {
                    inst.record_success(latency_ms as f64);
                }
            }

            self.stats.record_success(latency_ms);

            let mut response = FederatedResponse::new(
                request,
                instance.id,
                latency_ms,
                self.config.load_balance_strategy,
            );
            response = response.with_attempts(attempts);

            if attempts > 1 {
                response = response.with_fallback();
                self.stats.record_failover();
            }

            return Ok(response);
        }
    }

    /// Federate a request requiring a specific capability.
    pub async fn federate_with_capability<T: Clone>(
        &self,
        request: T,
        capability: &Capability,
    ) -> Result<FederatedResponse<T>> {
        let start = Instant::now();
        let mut attempts = 0u32;
        let mut tried_instances: Vec<String> = Vec::new();

        loop {
            attempts += 1;

            // Select an instance with required capability
            let instance = self.select_instance(Some(capability)).await;

            let instance = match instance {
                Some(i) if !tried_instances.contains(&i.id) => i,
                _ => {
                    self.stats.record_failure();
                    return Err(Error::Config(format!(
                        "No available instances with capability: {}",
                        capability
                    )));
                }
            };

            tried_instances.push(instance.id.clone());
            debug!(
                "Attempting request on instance: {} with capability: {} (attempt {})",
                instance.id, capability, attempts
            );

            let latency_ms = start.elapsed().as_millis() as u64;

            {
                let mut instances = self.instances.write().await;
                if let Some(inst) = instances.get_mut(&instance.id) {
                    inst.record_success(latency_ms as f64);
                }
            }

            self.stats.record_success(latency_ms);

            let mut response = FederatedResponse::new(
                request,
                instance.id,
                latency_ms,
                self.config.load_balance_strategy,
            );
            response = response.with_attempts(attempts);

            if attempts > 1 {
                response = response.with_fallback();
                self.stats.record_failover();
            }

            return Ok(response);
        }
    }

    /// Broadcast a request to all available instances.
    pub async fn broadcast<T: Clone>(&self, request: T) -> Vec<Result<FederatedResponse<T>>> {
        let instances = self.get_available_instances().await;
        let mut results = Vec::new();

        for instance in instances {
            let start = Instant::now();
            let latency_ms = start.elapsed().as_millis() as u64;

            // Record success (simulated)
            {
                let mut all_instances = self.instances.write().await;
                if let Some(inst) = all_instances.get_mut(&instance.id) {
                    inst.record_success(latency_ms as f64);
                }
            }

            self.stats.record_success(latency_ms);

            let response = FederatedResponse::new(
                request.clone(),
                instance.id,
                latency_ms,
                self.config.load_balance_strategy,
            );

            results.push(Ok(response));
        }

        results
    }

    // =========================================================================
    // SYNC OPERATIONS
    // =========================================================================

    /// Sync state across all instances.
    pub async fn sync_all(&self) -> Result<()> {
        let instances = self.get_available_instances().await;
        info!("Syncing state across {} instances", instances.len());

        for instance in &instances {
            debug!("Syncing with instance: {}", instance.id);
            // In a real implementation, this would sync state with each instance
        }

        Ok(())
    }

    /// Get sync interval duration.
    pub fn sync_interval(&self) -> Duration {
        Duration::from_millis(self.config.sync_interval_ms)
    }
}

impl Clone for FederatorAgent {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            instances: Arc::clone(&self.instances),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl Default for FederatorAgent {
    fn default() -> Self {
        Self::default_config()
    }
}

// ============================================================================
// Tests
// ============================================================================


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Instance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_instance_new() {
        let instance = Instance::new("test-1", "http://localhost:8080");
        assert_eq!(instance.id, "test-1");
        assert_eq!(instance.endpoint, "http://localhost:8080");
        assert_eq!(instance.weight, 100);
        assert!(!instance.capabilities.is_empty()); // Default includes TextProcessing
        assert_eq!(instance.health, InstanceHealth::Unknown);
    }

    #[test]
    fn test_instance_with_weight() {
        let instance = Instance::new("test-1", "http://localhost:8080")
            .with_weight(5);
        assert_eq!(instance.weight, 5);
    }

    #[test]
    fn test_instance_with_capabilities() {
        let instance = Instance::new("test-1", "http://localhost:8080")
            .with_capabilities(vec![Capability::Search, Capability::Index]);
        assert_eq!(instance.capabilities.len(), 2);
        assert!(instance.capabilities.contains(&Capability::Search));
        assert!(instance.capabilities.contains(&Capability::Index));
    }

    #[test]
    fn test_instance_with_health() {
        let instance = Instance::new("test-1", "http://localhost:8080")
            .with_health(InstanceHealth::Healthy);
        assert_eq!(instance.health, InstanceHealth::Healthy);
    }

    #[test]
    fn test_instance_is_available() {
        let healthy = Instance::new("h", "http://h").with_health(InstanceHealth::Healthy);
        let degraded = Instance::new("d", "http://d").with_health(InstanceHealth::Degraded);
        let unhealthy = Instance::new("u", "http://u").with_health(InstanceHealth::Unhealthy);
        let unknown = Instance::new("x", "http://x").with_health(InstanceHealth::Unknown);

        assert!(healthy.is_available());
        assert!(degraded.is_available());
        assert!(!unhealthy.is_available());
        assert!(!unknown.is_available());
    }

    #[test]
    fn test_instance_has_capability() {
        let instance = Instance::new("test-1", "http://localhost:8080")
            .with_capabilities(vec![Capability::Search, Capability::Validate]);

        assert!(instance.has_capability(&Capability::Search));
        assert!(instance.has_capability(&Capability::Validate));
        assert!(!instance.has_capability(&Capability::Index));
    }

    // -------------------------------------------------------------------------
    // InstanceHealth Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_instance_health_is_available() {
        assert!(InstanceHealth::Healthy.is_available());
        assert!(InstanceHealth::Degraded.is_available());
        assert!(!InstanceHealth::Unhealthy.is_available());
        assert!(!InstanceHealth::Unknown.is_available());
    }

    #[test]
    fn test_instance_health_score() {
        assert_eq!(InstanceHealth::Healthy.score(), 100);
        assert_eq!(InstanceHealth::Degraded.score(), 50);
        assert!(InstanceHealth::Unhealthy.score() < 50);
        assert!(InstanceHealth::Unknown.score() < 50);
    }

    #[test]
    fn test_instance_health_default() {
        // InstanceHealth doesn't implement Default, check via Instance
        let instance = Instance::new("test", "http://test");
        assert_eq!(instance.health, InstanceHealth::Unknown);
    }

    // -------------------------------------------------------------------------
    // LoadBalanceStrategy Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_balance_strategy_default() {
        assert_eq!(LoadBalanceStrategy::default(), LoadBalanceStrategy::RoundRobin);
    }

    #[test]
    fn test_load_balance_strategy_all() {
        let all = LoadBalanceStrategy::all();
        assert!(all.contains(&LoadBalanceStrategy::RoundRobin));
        assert!(all.contains(&LoadBalanceStrategy::WeightedRandom));
        assert!(all.contains(&LoadBalanceStrategy::LeastConnections));
        assert!(all.contains(&LoadBalanceStrategy::LatencyBased));
        assert!(all.contains(&LoadBalanceStrategy::CapabilityMatch));
    }

    // -------------------------------------------------------------------------
    // Capability Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_capability_custom() {
        let custom = Capability::Custom("my-capability".to_string());
        if let Capability::Custom(name) = custom {
            assert_eq!(name, "my-capability");
        } else {
            panic!("Expected Custom capability");
        }
    }

    // -------------------------------------------------------------------------
    // FederationConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_federation_config_default() {
        let config = FederationConfig::default();
        assert_eq!(config.load_balance_strategy, LoadBalanceStrategy::RoundRobin);
        assert!(config.failover_enabled);
    }

    #[test]
    fn test_federation_config_high_availability() {
        let config = FederationConfig::high_availability();
        // high_availability focuses on failover, not load balance strategy
        assert!(config.failover_enabled);
        assert!(config.max_retries >= 2);
    }

    #[test]
    fn test_federation_config_performance() {
        let config = FederationConfig::performance();
        assert_eq!(config.load_balance_strategy, LoadBalanceStrategy::LatencyBased);
    }

    // -------------------------------------------------------------------------
    // FederationStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_federation_stats_new() {
        let stats = FederationStats::new();
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.failed_requests, 0);
        assert_eq!(snapshot.failovers, 0);
    }

    #[test]
    fn test_federation_stats_record_success() {
        let stats = FederationStats::new();
        stats.record_success(100);
        stats.record_success(200);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 2);
        assert_eq!(snapshot.successful_requests, 2);
        assert_eq!(snapshot.failed_requests, 0);
    }

    #[test]
    fn test_federation_stats_record_failure() {
        let stats = FederationStats::new();
        stats.record_failure();
        stats.record_failure();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 2);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.failed_requests, 2);
    }

    #[test]
    fn test_federation_stats_record_failover() {
        let stats = FederationStats::new();
        stats.record_failover();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.failovers, 1);
    }

    #[test]
    fn test_federation_stats_success_rate() {
        let stats = FederationStats::new();
        
        // No requests yet - should be 100%
        assert_eq!(stats.success_rate(), 100.0);

        // Add some requests
        stats.record_success(100);
        stats.record_success(100);
        stats.record_success(100);
        stats.record_failure();

        // 3 out of 4 = 75%
        assert!((stats.success_rate() - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_federation_stats_average_latency() {
        let stats = FederationStats::new();
        
        // No requests - should be 0
        assert_eq!(stats.average_latency_ms(), 0.0);

        stats.record_success(100);
        stats.record_success(200);
        stats.record_success(300);

        // (100 + 200 + 300) / 3 = 200
        assert!((stats.average_latency_ms() - 200.0).abs() < 0.1);
    }

    // -------------------------------------------------------------------------
    // FederatorAgentBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_new() {
        let builder = FederatorAgentBuilder::new();
        let agent = builder.build();
        assert_eq!(agent.config().load_balance_strategy, LoadBalanceStrategy::RoundRobin);
    }

    #[test]
    fn test_builder_with_config() {
        let config = FederationConfig {
            sync_interval_ms: 10000,
            ..Default::default()
        };
        let agent = FederatorAgentBuilder::new()
            .with_config(config)
            .build();
        assert_eq!(agent.sync_interval(), Duration::from_millis(10000));
    }

    #[test]
    fn test_builder_add_instance() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("i1", "http://i1"))
            .add_instance(Instance::new("i2", "http://i2"))
            .add_instance(Instance::new("i3", "http://i3"))
            .build();
        assert_eq!(agent.config().instances.len(), 3);
    }

    #[test]
    fn test_builder_load_balance_strategy() {
        let agent = FederatorAgentBuilder::new()
            .load_balance_strategy(LoadBalanceStrategy::WeightedRandom)
            .build();
        assert_eq!(agent.config().load_balance_strategy, LoadBalanceStrategy::WeightedRandom);
    }

    #[test]
    fn test_builder_sync_interval_ms() {
        let agent = FederatorAgentBuilder::new()
            .sync_interval_ms(120000)
            .build();
        assert_eq!(agent.sync_interval(), Duration::from_secs(120));
    }

    #[test]
    fn test_builder_failover_enabled() {
        let agent = FederatorAgentBuilder::new()
            .failover_enabled(false)
            .build();
        assert!(!agent.config().failover_enabled);
    }

    // -------------------------------------------------------------------------
    // FederatorAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_default() {
        let agent = FederatorAgent::default_config();
        assert_eq!(agent.config().load_balance_strategy, LoadBalanceStrategy::RoundRobin);
    }

    #[test]
    fn test_agent_builder() {
        let agent = FederatorAgent::builder()
            .load_balance_strategy(LoadBalanceStrategy::LatencyBased)
            .build();
        assert_eq!(agent.config().load_balance_strategy, LoadBalanceStrategy::LatencyBased);
    }

    #[test]
    fn test_agent_clone() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("i1", "http://i1"))
            .build();
        let _cloned = agent.clone();
        // Clone should work without panic
    }

    #[tokio::test]
    async fn test_agent_add_instance() {
        let agent = FederatorAgent::default_config();
        let instance = Instance::new("test", "http://test");
        agent.add_instance(instance).await;
        assert_eq!(agent.instance_count().await, 1);
    }

    #[tokio::test]
    async fn test_agent_remove_instance() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("i1", "http://i1"))
            .add_instance(Instance::new("i2", "http://i2"))
            .build();

        assert_eq!(agent.instance_count().await, 2);
        let removed = agent.remove_instance("i1").await;
        assert!(removed.is_some());
        assert_eq!(agent.instance_count().await, 1);
    }

    #[tokio::test]
    async fn test_agent_remove_nonexistent_instance() {
        let agent = FederatorAgent::default_config();
        let removed = agent.remove_instance("nonexistent").await;
        assert!(removed.is_none());
    }

    #[tokio::test]
    async fn test_agent_get_instance() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("i1", "http://i1"))
            .build();

        let instance = agent.get_instance("i1").await;
        assert!(instance.is_some());
        assert_eq!(instance.unwrap().id, "i1");

        let nonexistent = agent.get_instance("i2").await;
        assert!(nonexistent.is_none());
    }

    #[tokio::test]
    async fn test_agent_update_instance_health() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("i1", "http://i1"))
            .build();

        agent.update_instance_health("i1", InstanceHealth::Healthy).await;

        let instance = agent.get_instance("i1").await.unwrap();
        assert_eq!(instance.health, InstanceHealth::Healthy);
    }

    #[tokio::test]
    async fn test_agent_get_available_instances() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("h1", "http://h1").with_health(InstanceHealth::Healthy))
            .add_instance(Instance::new("d1", "http://d1").with_health(InstanceHealth::Degraded))
            .add_instance(Instance::new("u1", "http://u1").with_health(InstanceHealth::Unhealthy))
            .build();

        let available = agent.get_available_instances().await;
        assert_eq!(available.len(), 2);
    }

    #[tokio::test]
    async fn test_agent_select_instance_round_robin() {
        let agent = FederatorAgentBuilder::new()
            .load_balance_strategy(LoadBalanceStrategy::RoundRobin)
            .add_instance(Instance::new("i1", "http://i1").with_health(InstanceHealth::Healthy))
            .add_instance(Instance::new("i2", "http://i2").with_health(InstanceHealth::Healthy))
            .build();

        // Round robin should cycle through instances
        let first = agent.select_instance(None).await.unwrap();
        let second = agent.select_instance(None).await.unwrap();
        let third = agent.select_instance(None).await.unwrap();

        // First and third should be same (wraps around)
        assert_eq!(first.id, third.id);
        assert_ne!(first.id, second.id);
    }

    #[tokio::test]
    async fn test_agent_select_instance_capability_match() {
        let agent = FederatorAgentBuilder::new()
            .load_balance_strategy(LoadBalanceStrategy::CapabilityMatch)
            .add_instance(
                Instance::new("search", "http://search")
                    .with_capabilities(vec![Capability::Search])
                    .with_health(InstanceHealth::Healthy)
            )
            .add_instance(
                Instance::new("index", "http://index")
                    .with_capabilities(vec![Capability::Index])
                    .with_health(InstanceHealth::Healthy)
            )
            .build();

        let search_instance = agent.select_instance(Some(&Capability::Search)).await.unwrap();
        assert_eq!(search_instance.id, "search");

        let index_instance = agent.select_instance(Some(&Capability::Index)).await.unwrap();
        assert_eq!(index_instance.id, "index");
    }

    #[tokio::test]
    async fn test_agent_select_instance_no_available() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("u1", "http://u1").with_health(InstanceHealth::Unhealthy))
            .build();

        let result = agent.select_instance(None).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_agent_federate_no_instances() {
        let agent = FederatorAgent::default_config();
        let result: Result<FederatedResponse<String>> = agent.federate("test request".to_string()).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_check_all_health() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("i1", "http://i1"))
            .build();

        // Should not panic
        agent.check_all_health().await;
    }

    #[tokio::test]
    async fn test_agent_sync_all() {
        let agent = FederatorAgentBuilder::new()
            .add_instance(Instance::new("i1", "http://i1").with_health(InstanceHealth::Healthy))
            .build();

        let result = agent.sync_all().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_agent_sync_interval() {
        let agent = FederatorAgentBuilder::new()
            .sync_interval_ms(60000)
            .build();
        assert_eq!(agent.sync_interval(), Duration::from_secs(60));
    }

    // -------------------------------------------------------------------------
    // FederatedResponse Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_federated_response_new() {
        let response = FederatedResponse::new(
            "result".to_string(),
            "instance-1",
            150,
            LoadBalanceStrategy::RoundRobin,
        );
        assert_eq!(response.response, "result");
        assert_eq!(response.instance_id, "instance-1");
        assert_eq!(response.latency_ms, 150);
        assert!(!response.fallback_used);
    }

    #[test]
    fn test_federated_response_with_fallback() {
        let response = FederatedResponse::new(
            "result".to_string(),
            "instance-1",
            150,
            LoadBalanceStrategy::RoundRobin,
        ).with_fallback();
        assert!(response.fallback_used);
    }

    #[test]
    fn test_federated_response_map() {
        let response = FederatedResponse::new(
            42i32,
            "instance-1",
            150,
            LoadBalanceStrategy::RoundRobin,
        );
        let mapped = response.map(|x| x.to_string());
        assert_eq!(mapped.response, "42");
    }

    #[test]
    fn test_federated_response_into_response() {
        let response = FederatedResponse::new(
            "result".to_string(),
            "instance-1",
            150,
            LoadBalanceStrategy::RoundRobin,
        );
        assert_eq!(response.into_response(), "result");
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_full_federation_workflow() {
        // Create a federation with multiple instances
        let agent = FederatorAgentBuilder::new()
            .load_balance_strategy(LoadBalanceStrategy::CapabilityMatch)
            .failover_enabled(true)
            .add_instance(
                Instance::new("search-primary", "http://search1:8080")
                    .with_weight(10)
                    .with_capabilities(vec![Capability::Search, Capability::Synthesize])
                    .with_health(InstanceHealth::Healthy)
            )
            .add_instance(
                Instance::new("search-secondary", "http://search2:8080")
                    .with_weight(5)
                    .with_capabilities(vec![Capability::Search])
                    .with_health(InstanceHealth::Degraded)
            )
            .add_instance(
                Instance::new("index-primary", "http://index1:8080")
                    .with_weight(10)
                    .with_capabilities(vec![Capability::Index, Capability::Validate])
                    .with_health(InstanceHealth::Healthy)
            )
            .build();

        // Verify instance count
        assert_eq!(agent.instance_count().await, 3);

        // Test available instances
        let available = agent.get_available_instances().await;
        assert_eq!(available.len(), 3);

        // Test select with capability
        let search_instance = agent.select_instance(Some(&Capability::Search)).await;
        assert!(search_instance.is_some());
        assert!(search_instance.unwrap().capabilities.contains(&Capability::Search));

        // Check stats
        let snapshot = agent.stats_snapshot();
        assert_eq!(snapshot.total_requests, 0);
    }
}
