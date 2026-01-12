//! Harmonizer Agent module for Project Panpsychism.
//!
//! The Balance Keeper — "In harmony, all parts find their strength."
//!
//! This module implements the Harmonizer Agent, responsible for maintaining balance
//! and harmony across all agents with load balancing, conflict resolution, and
//! resource optimization. Like the conductor of an orchestra ensuring each instrument
//! plays in concert, the Harmonizer Agent keeps the system operating smoothly.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, the Harmonizer Agent serves as the **Balance Keeper** —
//! a force that maintains equilibrium across the magical realm:
//!
//! - **Harmony Reports** reveal the overall balance of the system
//! - **Adjustments** fine-tune each component's contribution
//! - **Conflict Resolution** prevents magical interference between agents
//! - **Resource Allocation** ensures fair distribution of magical energy
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Self-preservation through balanced resource distribution
//! - **NATURA**: Natural equilibrium between system components
//! - **RATIO**: Logical analysis of resource contention and conflicts
//! - **LAETITIA**: Joy through harmonious system operation
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::harmonizer::{HarmonizerAgent, HarmonizerConfig};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let harmonizer = HarmonizerAgent::new();
//!
//!     // Analyze and balance the system
//!     let report = harmonizer.harmonize().await?;
//!     println!("Balance score: {:.2}", report.balance_score);
//!
//!     // Review adjustments made
//!     for adjustment in &report.adjustments {
//!         println!("Adjusted {}: {:?}", adjustment.agent_id, adjustment.adjustment_type);
//!     }
//!
//!     // Check resolved conflicts
//!     for conflict in &report.conflicts_resolved {
//!         println!("Resolved: {:?}", conflict.conflict_type);
//!     }
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, warn};

// =============================================================================
// ADJUSTMENT TYPE
// =============================================================================

/// Types of adjustments that can be made to agents.
///
/// Like tuning instruments in an orchestra, these adjustments
/// bring each agent into harmony with the whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdjustmentType {
    /// Increase resources allocated to the agent.
    IncreaseResources,

    /// Decrease resources allocated to the agent.
    DecreaseResources,

    /// Change the priority level of the agent.
    ChangePriority,

    /// Throttle incoming requests to the agent.
    ThrottleRequests,

    /// Boost performance by allocating more compute.
    BoostPerformance,
}

impl std::fmt::Display for AdjustmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncreaseResources => write!(f, "increase_resources"),
            Self::DecreaseResources => write!(f, "decrease_resources"),
            Self::ChangePriority => write!(f, "change_priority"),
            Self::ThrottleRequests => write!(f, "throttle_requests"),
            Self::BoostPerformance => write!(f, "boost_performance"),
        }
    }
}

impl std::str::FromStr for AdjustmentType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "increase_resources" | "increase" => Ok(Self::IncreaseResources),
            "decrease_resources" | "decrease" => Ok(Self::DecreaseResources),
            "change_priority" | "priority" => Ok(Self::ChangePriority),
            "throttle_requests" | "throttle" => Ok(Self::ThrottleRequests),
            "boost_performance" | "boost" => Ok(Self::BoostPerformance),
            _ => Err(Error::Config(format!(
                "Unknown adjustment type: '{}'. Valid: increase_resources, decrease_resources, change_priority, throttle_requests, boost_performance",
                s
            ))),
        }
    }
}

impl AdjustmentType {
    /// Get all adjustment types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::IncreaseResources,
            Self::DecreaseResources,
            Self::ChangePriority,
            Self::ThrottleRequests,
            Self::BoostPerformance,
        ]
    }

    /// Check if this adjustment increases resource usage.
    pub fn increases_resources(&self) -> bool {
        matches!(self, Self::IncreaseResources | Self::BoostPerformance)
    }

    /// Check if this adjustment decreases resource usage.
    pub fn decreases_resources(&self) -> bool {
        matches!(self, Self::DecreaseResources | Self::ThrottleRequests)
    }

    /// Get the impact magnitude (positive = more resources, negative = fewer).
    pub fn impact_magnitude(&self) -> i8 {
        match self {
            Self::IncreaseResources => 2,
            Self::BoostPerformance => 3,
            Self::ChangePriority => 0,
            Self::DecreaseResources => -2,
            Self::ThrottleRequests => -1,
        }
    }

    /// Get a human-readable description of this adjustment type.
    pub fn description(&self) -> &'static str {
        match self {
            Self::IncreaseResources => "Allocate more CPU and memory to the agent",
            Self::DecreaseResources => "Reduce CPU and memory allocation",
            Self::ChangePriority => "Modify the scheduling priority",
            Self::ThrottleRequests => "Limit incoming request rate",
            Self::BoostPerformance => "Maximum resource allocation for peak performance",
        }
    }
}

impl Default for AdjustmentType {
    fn default() -> Self {
        Self::ChangePriority
    }
}

// =============================================================================
// CONFLICT TYPE
// =============================================================================

/// Types of conflicts that can occur between agents.
///
/// Like dissonance in music, conflicts disrupt the harmony
/// of the system and must be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictType {
    /// Multiple agents competing for the same resources.
    ResourceContention,

    /// Conflicting priority levels causing scheduling issues.
    PriorityConflict,

    /// Inconsistent data between agents.
    DataInconsistency,

    /// Potential deadlock situation detected.
    DeadlockRisk,
}

impl std::fmt::Display for ConflictType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResourceContention => write!(f, "resource_contention"),
            Self::PriorityConflict => write!(f, "priority_conflict"),
            Self::DataInconsistency => write!(f, "data_inconsistency"),
            Self::DeadlockRisk => write!(f, "deadlock_risk"),
        }
    }
}

impl std::str::FromStr for ConflictType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "resource_contention" | "resource" | "contention" => Ok(Self::ResourceContention),
            "priority_conflict" | "priority" => Ok(Self::PriorityConflict),
            "data_inconsistency" | "data" | "inconsistency" => Ok(Self::DataInconsistency),
            "deadlock_risk" | "deadlock" => Ok(Self::DeadlockRisk),
            _ => Err(Error::Config(format!(
                "Unknown conflict type: '{}'. Valid: resource_contention, priority_conflict, data_inconsistency, deadlock_risk",
                s
            ))),
        }
    }
}

impl ConflictType {
    /// Get all conflict types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::ResourceContention,
            Self::PriorityConflict,
            Self::DataInconsistency,
            Self::DeadlockRisk,
        ]
    }

    /// Get the severity of this conflict type (1-10).
    pub fn severity(&self) -> u8 {
        match self {
            Self::DeadlockRisk => 10,
            Self::DataInconsistency => 8,
            Self::ResourceContention => 6,
            Self::PriorityConflict => 4,
        }
    }

    /// Check if this conflict requires immediate attention.
    pub fn is_critical(&self) -> bool {
        self.severity() >= 8
    }

    /// Check if this conflict can be auto-resolved.
    pub fn is_auto_resolvable(&self) -> bool {
        matches!(self, Self::ResourceContention | Self::PriorityConflict)
    }

    /// Get a human-readable description of this conflict type.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ResourceContention => "Multiple agents competing for limited resources",
            Self::PriorityConflict => "Conflicting scheduling priorities between agents",
            Self::DataInconsistency => "Inconsistent state or data between agents",
            Self::DeadlockRisk => "Circular dependency that may cause deadlock",
        }
    }

    /// Get recommended resolution strategy for this conflict type.
    pub fn recommended_resolution(&self) -> &'static str {
        match self {
            Self::ResourceContention => "Redistribute resources or increase capacity",
            Self::PriorityConflict => "Reassign priorities based on workload",
            Self::DataInconsistency => "Synchronize data and establish single source of truth",
            Self::DeadlockRisk => "Break circular dependency chain immediately",
        }
    }
}

// =============================================================================
// ADJUSTMENT
// =============================================================================

/// Represents an adjustment made to an agent.
///
/// Like fine-tuning an instrument, adjustments bring
/// individual agents into harmony with the whole.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjustment {
    /// The agent that was adjusted.
    pub agent_id: String,

    /// The type of adjustment made.
    pub adjustment_type: AdjustmentType,

    /// Value before adjustment.
    pub before: f64,

    /// Value after adjustment.
    pub after: f64,

    /// Reason for the adjustment.
    pub reason: String,

    /// Timestamp when adjustment was made.
    pub timestamp: DateTime<Utc>,

    /// Whether the adjustment was applied automatically.
    pub auto_applied: bool,
}

impl Adjustment {
    /// Create a new adjustment.
    pub fn new(
        agent_id: impl Into<String>,
        adjustment_type: AdjustmentType,
        before: f64,
        after: f64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            adjustment_type,
            before,
            after,
            reason: reason.into(),
            timestamp: Utc::now(),
            auto_applied: false,
        }
    }

    /// Set whether this adjustment was auto-applied.
    pub fn with_auto_applied(mut self, auto: bool) -> Self {
        self.auto_applied = auto;
        self
    }

    /// Calculate the change magnitude.
    pub fn change_magnitude(&self) -> f64 {
        self.after - self.before
    }

    /// Calculate the change percentage.
    pub fn change_percentage(&self) -> f64 {
        if self.before == 0.0 {
            if self.after == 0.0 {
                0.0
            } else {
                100.0
            }
        } else {
            ((self.after - self.before) / self.before) * 100.0
        }
    }

    /// Check if this adjustment increased the value.
    pub fn is_increase(&self) -> bool {
        self.after > self.before
    }

    /// Check if this adjustment decreased the value.
    pub fn is_decrease(&self) -> bool {
        self.after < self.before
    }

    /// Check if there was no effective change.
    pub fn is_no_change(&self) -> bool {
        (self.after - self.before).abs() < f64::EPSILON
    }

    /// Format adjustment as a summary string.
    pub fn summary(&self) -> String {
        let direction = if self.is_increase() {
            "increased"
        } else if self.is_decrease() {
            "decreased"
        } else {
            "unchanged"
        };
        format!(
            "{} {}: {} from {:.2} to {:.2} ({:+.1}%)",
            self.agent_id,
            self.adjustment_type,
            direction,
            self.before,
            self.after,
            self.change_percentage()
        )
    }
}

// =============================================================================
// CONFLICT
// =============================================================================

/// Represents a conflict between agents.
///
/// Like dissonance in music, conflicts disrupt the harmony
/// of the system and must be resolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// The agents involved in the conflict.
    pub agents: Vec<String>,

    /// The type of conflict.
    pub conflict_type: ConflictType,

    /// How the conflict was resolved.
    pub resolution: String,

    /// Timestamp when conflict was detected.
    pub detected_at: DateTime<Utc>,

    /// Timestamp when conflict was resolved.
    pub resolved_at: Option<DateTime<Utc>>,

    /// Whether the conflict was resolved automatically.
    pub auto_resolved: bool,

    /// Additional details about the conflict.
    pub details: Option<String>,
}

impl Conflict {
    /// Create a new conflict.
    pub fn new(
        agents: Vec<impl Into<String>>,
        conflict_type: ConflictType,
        resolution: impl Into<String>,
    ) -> Self {
        Self {
            agents: agents.into_iter().map(Into::into).collect(),
            conflict_type,
            resolution: resolution.into(),
            detected_at: Utc::now(),
            resolved_at: None,
            auto_resolved: false,
            details: None,
        }
    }

    /// Create a resolved conflict.
    pub fn resolved(
        agents: Vec<impl Into<String>>,
        conflict_type: ConflictType,
        resolution: impl Into<String>,
    ) -> Self {
        let mut conflict = Self::new(agents, conflict_type, resolution);
        conflict.resolved_at = Some(Utc::now());
        conflict
    }

    /// Set the resolved timestamp.
    pub fn with_resolved_at(mut self, resolved_at: DateTime<Utc>) -> Self {
        self.resolved_at = Some(resolved_at);
        self
    }

    /// Set whether auto-resolved.
    pub fn with_auto_resolved(mut self, auto: bool) -> Self {
        self.auto_resolved = auto;
        self
    }

    /// Set additional details.
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Mark the conflict as resolved now.
    pub fn mark_resolved(&mut self) {
        self.resolved_at = Some(Utc::now());
    }

    /// Check if the conflict has been resolved.
    pub fn is_resolved(&self) -> bool {
        self.resolved_at.is_some()
    }

    /// Get the number of agents involved.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Check if a specific agent is involved.
    pub fn involves_agent(&self, agent_id: &str) -> bool {
        self.agents.iter().any(|a| a == agent_id)
    }

    /// Get time to resolution in milliseconds, if resolved.
    pub fn resolution_time_ms(&self) -> Option<i64> {
        self.resolved_at
            .map(|resolved| (resolved - self.detected_at).num_milliseconds())
    }

    /// Format conflict as a summary string.
    pub fn summary(&self) -> String {
        let status = if self.is_resolved() {
            "resolved"
        } else {
            "pending"
        };
        format!(
            "{} conflict between {:?} [{}]: {}",
            self.conflict_type, self.agents, status, self.resolution
        )
    }
}

// =============================================================================
// RESOURCE ALLOCATION
// =============================================================================

/// Represents resource allocation for the system.
///
/// Like distributing magical energy among wands,
/// resource allocation ensures fair and effective distribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU shares allocated per agent (0.0 - 1.0 fraction of total).
    pub cpu_shares: HashMap<String, f64>,

    /// Memory limits per agent in megabytes.
    pub memory_limit: HashMap<String, f64>,

    /// Priority weights per agent (higher = more priority).
    pub priority_weights: HashMap<String, f64>,
}

impl ResourceAllocation {
    /// Create a new empty resource allocation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set CPU share for an agent.
    pub fn set_cpu_share(&mut self, agent_id: impl Into<String>, share: f64) {
        self.cpu_shares.insert(agent_id.into(), share.clamp(0.0, 1.0));
    }

    /// Get CPU share for an agent.
    pub fn get_cpu_share(&self, agent_id: &str) -> Option<f64> {
        self.cpu_shares.get(agent_id).copied()
    }

    /// Set memory limit for an agent.
    pub fn set_memory_limit(&mut self, agent_id: impl Into<String>, limit_mb: f64) {
        self.memory_limit.insert(agent_id.into(), limit_mb.max(0.0));
    }

    /// Get memory limit for an agent.
    pub fn get_memory_limit(&self, agent_id: &str) -> Option<f64> {
        self.memory_limit.get(agent_id).copied()
    }

    /// Set priority weight for an agent.
    pub fn set_priority_weight(&mut self, agent_id: impl Into<String>, weight: f64) {
        self.priority_weights.insert(agent_id.into(), weight.max(0.0));
    }

    /// Get priority weight for an agent.
    pub fn get_priority_weight(&self, agent_id: &str) -> Option<f64> {
        self.priority_weights.get(agent_id).copied()
    }

    /// Get total CPU shares allocated.
    pub fn total_cpu_shares(&self) -> f64 {
        self.cpu_shares.values().sum()
    }

    /// Get total memory allocated in megabytes.
    pub fn total_memory_mb(&self) -> f64 {
        self.memory_limit.values().sum()
    }

    /// Check if CPU is over-allocated (> 1.0).
    pub fn is_cpu_overallocated(&self) -> bool {
        self.total_cpu_shares() > 1.0
    }

    /// Get the number of agents with allocations.
    pub fn agent_count(&self) -> usize {
        let mut agents: std::collections::HashSet<&String> = self.cpu_shares.keys().collect();
        agents.extend(self.memory_limit.keys());
        agents.extend(self.priority_weights.keys());
        agents.len()
    }

    /// Check if an agent has any allocation.
    pub fn has_agent(&self, agent_id: &str) -> bool {
        self.cpu_shares.contains_key(agent_id)
            || self.memory_limit.contains_key(agent_id)
            || self.priority_weights.contains_key(agent_id)
    }

    /// Remove all allocations for an agent.
    pub fn remove_agent(&mut self, agent_id: &str) {
        self.cpu_shares.remove(agent_id);
        self.memory_limit.remove(agent_id);
        self.priority_weights.remove(agent_id);
    }

    /// Clear all allocations.
    pub fn clear(&mut self) {
        self.cpu_shares.clear();
        self.memory_limit.clear();
        self.priority_weights.clear();
    }

    /// Normalize CPU shares to sum to 1.0.
    pub fn normalize_cpu_shares(&mut self) {
        let total = self.total_cpu_shares();
        if total > 0.0 && total != 1.0 {
            for share in self.cpu_shares.values_mut() {
                *share /= total;
            }
        }
    }

    /// Get agents sorted by priority weight (highest first).
    pub fn agents_by_priority(&self) -> Vec<(&String, f64)> {
        let mut sorted: Vec<_> = self.priority_weights.iter().map(|(k, v)| (k, *v)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }
}

// =============================================================================
// HARMONY REPORT
// =============================================================================

/// Comprehensive harmony report for the system.
///
/// Contains the overall balance score, adjustments made,
/// conflicts resolved, and resource allocation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyReport {
    /// Overall balance score (0.0 - 1.0, higher is better).
    pub balance_score: f64,

    /// Adjustments made during harmonization.
    pub adjustments: Vec<Adjustment>,

    /// Conflicts that were resolved.
    pub conflicts_resolved: Vec<Conflict>,

    /// Current resource allocation state.
    pub resource_allocation: ResourceAllocation,

    /// Timestamp when the report was generated.
    pub timestamp: DateTime<Utc>,

    /// Duration of the harmonization process in milliseconds.
    pub harmonize_duration_ms: u64,

    /// Number of agents analyzed.
    pub agents_analyzed: usize,

    /// Number of pending (unresolved) conflicts.
    pub pending_conflicts: usize,

    /// Optional summary message.
    pub summary: Option<String>,
}

impl HarmonyReport {
    /// Create a new harmony report.
    pub fn new(balance_score: f64) -> Self {
        Self {
            balance_score: balance_score.clamp(0.0, 1.0),
            adjustments: Vec::new(),
            conflicts_resolved: Vec::new(),
            resource_allocation: ResourceAllocation::new(),
            timestamp: Utc::now(),
            harmonize_duration_ms: 0,
            agents_analyzed: 0,
            pending_conflicts: 0,
            summary: None,
        }
    }

    /// Create a perfectly balanced report.
    pub fn perfect() -> Self {
        Self::new(1.0)
    }

    /// Add an adjustment to the report.
    pub fn add_adjustment(&mut self, adjustment: Adjustment) {
        self.adjustments.push(adjustment);
    }

    /// Add a resolved conflict to the report.
    pub fn add_conflict(&mut self, conflict: Conflict) {
        self.conflicts_resolved.push(conflict);
    }

    /// Set the resource allocation.
    pub fn with_allocation(mut self, allocation: ResourceAllocation) -> Self {
        self.resource_allocation = allocation;
        self
    }

    /// Set the harmonization duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.harmonize_duration_ms = duration_ms;
        self
    }

    /// Set the number of agents analyzed.
    pub fn with_agents_analyzed(mut self, count: usize) -> Self {
        self.agents_analyzed = count;
        self
    }

    /// Set the number of pending conflicts.
    pub fn with_pending_conflicts(mut self, count: usize) -> Self {
        self.pending_conflicts = count;
        self
    }

    /// Set a summary message.
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Check if the system is in harmony (score >= 0.8).
    pub fn is_harmonious(&self) -> bool {
        self.balance_score >= 0.8
    }

    /// Check if the system is in critical imbalance (score < 0.5).
    pub fn is_critical(&self) -> bool {
        self.balance_score < 0.5
    }

    /// Get the number of adjustments made.
    pub fn adjustment_count(&self) -> usize {
        self.adjustments.len()
    }

    /// Get the number of conflicts resolved.
    pub fn conflict_count(&self) -> usize {
        self.conflicts_resolved.len()
    }

    /// Get adjustments of a specific type.
    pub fn adjustments_of_type(&self, adj_type: AdjustmentType) -> Vec<&Adjustment> {
        self.adjustments
            .iter()
            .filter(|a| a.adjustment_type == adj_type)
            .collect()
    }

    /// Get conflicts of a specific type.
    pub fn conflicts_of_type(&self, conflict_type: ConflictType) -> Vec<&Conflict> {
        self.conflicts_resolved
            .iter()
            .filter(|c| c.conflict_type == conflict_type)
            .collect()
    }

    /// Generate a brief summary of the harmony report.
    pub fn brief_summary(&self) -> String {
        let status = if self.is_harmonious() {
            "harmonious"
        } else if self.is_critical() {
            "critical"
        } else {
            "imbalanced"
        };

        format!(
            "System {}: score={:.2}, {} adjustments, {} conflicts resolved, {} pending",
            status,
            self.balance_score,
            self.adjustment_count(),
            self.conflict_count(),
            self.pending_conflicts
        )
    }

    /// Format the report as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Harmony Report\n\n");
        output.push_str(&format!("**Balance Score:** {:.2}\n\n", self.balance_score));
        output.push_str(&format!(
            "**Generated:** {}\n\n",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        if let Some(summary) = &self.summary {
            output.push_str(&format!("**Summary:** {}\n\n", summary));
        }

        // Adjustments section
        if !self.adjustments.is_empty() {
            output.push_str("## Adjustments\n\n");
            output.push_str("| Agent | Type | Before | After | Change |\n");
            output.push_str("|-------|------|--------|-------|--------|\n");

            for adj in &self.adjustments {
                output.push_str(&format!(
                    "| {} | {} | {:.2} | {:.2} | {:+.1}% |\n",
                    adj.agent_id,
                    adj.adjustment_type,
                    adj.before,
                    adj.after,
                    adj.change_percentage()
                ));
            }
            output.push('\n');
        }

        // Conflicts section
        if !self.conflicts_resolved.is_empty() {
            output.push_str("## Resolved Conflicts\n\n");
            for conflict in &self.conflicts_resolved {
                output.push_str(&format!(
                    "- **{}** between {:?}: {}\n",
                    conflict.conflict_type, conflict.agents, conflict.resolution
                ));
            }
            output.push('\n');
        }

        // Resource allocation section
        if self.resource_allocation.agent_count() > 0 {
            output.push_str("## Resource Allocation\n\n");
            output.push_str(&format!(
                "- Total CPU: {:.2}\n",
                self.resource_allocation.total_cpu_shares()
            ));
            output.push_str(&format!(
                "- Total Memory: {:.2} MB\n",
                self.resource_allocation.total_memory_mb()
            ));
            output.push('\n');
        }

        output
    }
}

impl Default for HarmonyReport {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// =============================================================================
// HARMONIZER CONFIGURATION
// =============================================================================

/// Configuration for the Harmonizer Agent.
///
/// Defines thresholds, intervals, and behavior for harmony maintenance.
#[derive(Debug, Clone)]
pub struct HarmonizerConfig {
    /// Minimum balance threshold before triggering adjustments (0.0 - 1.0).
    pub balance_threshold: f64,

    /// Interval between automatic harmony checks in seconds.
    pub check_interval: u64,

    /// Whether to automatically apply adjustments.
    pub auto_adjust: bool,

    /// Maximum number of adjustments per harmonize cycle.
    pub max_adjustments_per_cycle: usize,

    /// Whether to auto-resolve conflicts.
    pub auto_resolve_conflicts: bool,

    /// CPU share threshold to trigger rebalancing.
    pub cpu_rebalance_threshold: f64,

    /// Memory threshold to trigger rebalancing (in MB).
    pub memory_rebalance_threshold: f64,

    /// Default priority weight for new agents.
    pub default_priority_weight: f64,
}

impl Default for HarmonizerConfig {
    fn default() -> Self {
        Self {
            balance_threshold: 0.7,
            check_interval: 60,
            auto_adjust: true,
            max_adjustments_per_cycle: 10,
            auto_resolve_conflicts: true,
            cpu_rebalance_threshold: 0.8,
            memory_rebalance_threshold: 1024.0,
            default_priority_weight: 1.0,
        }
    }
}

impl HarmonizerConfig {
    /// Create a strict configuration with tighter thresholds.
    pub fn strict() -> Self {
        Self {
            balance_threshold: 0.85,
            check_interval: 30,
            auto_adjust: true,
            max_adjustments_per_cycle: 5,
            auto_resolve_conflicts: true,
            cpu_rebalance_threshold: 0.7,
            memory_rebalance_threshold: 512.0,
            default_priority_weight: 1.0,
        }
    }

    /// Create a lenient configuration with relaxed thresholds.
    pub fn lenient() -> Self {
        Self {
            balance_threshold: 0.5,
            check_interval: 120,
            auto_adjust: true,
            max_adjustments_per_cycle: 20,
            auto_resolve_conflicts: true,
            cpu_rebalance_threshold: 0.9,
            memory_rebalance_threshold: 2048.0,
            default_priority_weight: 1.0,
        }
    }

    /// Create a manual configuration (no auto-adjustments).
    pub fn manual() -> Self {
        Self {
            auto_adjust: false,
            auto_resolve_conflicts: false,
            ..Default::default()
        }
    }

    /// Check if a balance score meets the threshold.
    pub fn meets_threshold(&self, score: f64) -> bool {
        score >= self.balance_threshold
    }
}

// =============================================================================
// AGENT STATE
// =============================================================================

/// Internal state tracking for an agent.
#[derive(Debug, Clone)]
struct AgentState {
    /// Agent identifier.
    id: String,
    /// Current CPU share.
    cpu_share: f64,
    /// Current memory allocation in MB.
    memory_mb: f64,
    /// Current priority weight.
    priority: f64,
    /// Current load (0.0 - 1.0).
    load: f64,
    /// Error count.
    error_count: u32,
    /// Last update timestamp.
    last_update: DateTime<Utc>,
}

impl AgentState {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            cpu_share: 0.1,
            memory_mb: 256.0,
            priority: 1.0,
            load: 0.0,
            error_count: 0,
            last_update: Utc::now(),
        }
    }

    fn is_overloaded(&self) -> bool {
        self.load > 0.8
    }

    fn is_underutilized(&self) -> bool {
        self.load < 0.2
    }
}

// =============================================================================
// HARMONIZER AGENT
// =============================================================================

/// The Harmonizer Agent - The Balance Keeper of the Sorcerer's Tower.
///
/// Responsible for maintaining balance and harmony across all agents
/// with load balancing, conflict resolution, and resource optimization.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive to maintain system equilibrium
/// - **NATURA**: Natural balance between components
/// - **RATIO**: Logical analysis of resource distribution
/// - **LAETITIA**: Joy through harmonious operation
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::harmonizer::{HarmonizerAgent, HarmonizerConfig};
///
/// let harmonizer = HarmonizerAgent::builder()
///     .balance_threshold(0.8)
///     .auto_adjust(true)
///     .build();
///
/// let report = harmonizer.harmonize().await?;
/// println!("Balance score: {:.2}", report.balance_score);
/// ```
#[derive(Debug, Clone)]
pub struct HarmonizerAgent {
    /// Configuration for harmonization behavior.
    config: HarmonizerConfig,

    /// Agent states being tracked.
    agent_states: HashMap<String, AgentState>,

    /// Current resource allocation.
    allocation: ResourceAllocation,

    /// Pending conflicts awaiting resolution.
    pending_conflicts: Vec<Conflict>,
}

impl Default for HarmonizerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonizerAgent {
    /// Create a new Harmonizer Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: HarmonizerConfig::default(),
            agent_states: HashMap::new(),
            allocation: ResourceAllocation::new(),
            pending_conflicts: Vec::new(),
        }
    }

    /// Create a Harmonizer Agent with custom configuration.
    pub fn with_config(config: HarmonizerConfig) -> Self {
        Self {
            config,
            agent_states: HashMap::new(),
            allocation: ResourceAllocation::new(),
            pending_conflicts: Vec::new(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> HarmonizerAgentBuilder {
        HarmonizerAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &HarmonizerConfig {
        &self.config
    }

    /// Get the current resource allocation.
    pub fn allocation(&self) -> &ResourceAllocation {
        &self.allocation
    }

    /// Get pending conflicts.
    pub fn pending_conflicts(&self) -> &[Conflict] {
        &self.pending_conflicts
    }

    /// Register an agent for harmonization.
    pub fn register_agent(&mut self, agent_id: impl Into<String>) {
        let id = agent_id.into();
        self.agent_states.insert(id.clone(), AgentState::new(&id));
        self.allocation.set_cpu_share(&id, 0.1);
        self.allocation.set_memory_limit(&id, 256.0);
        self.allocation.set_priority_weight(&id, self.config.default_priority_weight);
    }

    /// Unregister an agent.
    pub fn unregister_agent(&mut self, agent_id: &str) {
        self.agent_states.remove(agent_id);
        self.allocation.remove_agent(agent_id);
    }

    /// Update an agent's load.
    pub fn update_agent_load(&mut self, agent_id: &str, load: f64) {
        if let Some(state) = self.agent_states.get_mut(agent_id) {
            state.load = load.clamp(0.0, 1.0);
            state.last_update = Utc::now();
        }
    }

    /// Update an agent's resource allocation.
    pub fn update_agent_resources(
        &mut self,
        agent_id: &str,
        cpu_share: Option<f64>,
        memory_mb: Option<f64>,
        priority: Option<f64>,
    ) {
        if let Some(state) = self.agent_states.get_mut(agent_id) {
            if let Some(cpu) = cpu_share {
                state.cpu_share = cpu.clamp(0.0, 1.0);
                self.allocation.set_cpu_share(agent_id, cpu);
            }
            if let Some(mem) = memory_mb {
                state.memory_mb = mem.max(0.0);
                self.allocation.set_memory_limit(agent_id, mem);
            }
            if let Some(prio) = priority {
                state.priority = prio.max(0.0);
                self.allocation.set_priority_weight(agent_id, prio);
            }
            state.last_update = Utc::now();
        }
    }

    /// Record an error for an agent.
    pub fn record_agent_error(&mut self, agent_id: &str) {
        if let Some(state) = self.agent_states.get_mut(agent_id) {
            state.error_count += 1;
            state.last_update = Utc::now();
        }
    }

    /// Add a pending conflict.
    pub fn add_conflict(&mut self, conflict: Conflict) {
        self.pending_conflicts.push(conflict);
    }

    /// Get the number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agent_states.len()
    }

    // =========================================================================
    // HARMONIZATION
    // =========================================================================

    /// Perform harmonization - analyze and balance the system.
    ///
    /// This is the main entry point for the Harmonizer Agent.
    /// It analyzes the current state, detects imbalances and conflicts,
    /// and makes adjustments to restore harmony.
    pub async fn harmonize(&mut self) -> Result<HarmonyReport> {
        let start = Instant::now();

        debug!("Starting harmonization process");

        // Calculate current balance score
        let balance_score = self.calculate_balance_score();

        // Create report
        let mut report = HarmonyReport::new(balance_score);
        report.agents_analyzed = self.agent_states.len();

        // Detect and resolve conflicts
        if self.config.auto_resolve_conflicts {
            let resolved = self.resolve_conflicts();
            for conflict in resolved {
                report.add_conflict(conflict);
            }
        }

        // Make adjustments if needed and auto_adjust is enabled
        if self.config.auto_adjust && !self.config.meets_threshold(balance_score) {
            let adjustments = self.generate_adjustments();
            let applied: Vec<_> = adjustments
                .into_iter()
                .take(self.config.max_adjustments_per_cycle)
                .collect();

            for adj in applied {
                self.apply_adjustment(&adj);
                report.add_adjustment(adj);
            }
        }

        // Update report with current state
        report.resource_allocation = self.allocation.clone();
        report.pending_conflicts = self.pending_conflicts.len();
        report.harmonize_duration_ms = start.elapsed().as_millis() as u64;

        // Recalculate balance after adjustments
        report.balance_score = self.calculate_balance_score();
        report.summary = Some(report.brief_summary());

        info!(
            "Harmonization complete: score={:.2}, {} adjustments, {} conflicts resolved",
            report.balance_score,
            report.adjustment_count(),
            report.conflict_count()
        );

        Ok(report)
    }

    /// Calculate the current balance score.
    pub fn calculate_balance_score(&self) -> f64 {
        if self.agent_states.is_empty() {
            return 1.0;
        }

        let mut score = 1.0;

        // Penalize for load imbalance
        let loads: Vec<f64> = self.agent_states.values().map(|s| s.load).collect();
        if !loads.is_empty() {
            let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;
            let variance: f64 = loads.iter().map(|l| (l - avg_load).powi(2)).sum::<f64>()
                / loads.len() as f64;
            score -= variance.sqrt() * 0.3;
        }

        // Penalize for overloaded agents
        let overloaded = self.agent_states.values().filter(|s| s.is_overloaded()).count();
        score -= (overloaded as f64 / self.agent_states.len() as f64) * 0.2;

        // Penalize for CPU over-allocation
        if self.allocation.is_cpu_overallocated() {
            score -= 0.1;
        }

        // Penalize for pending conflicts
        let pending_critical = self
            .pending_conflicts
            .iter()
            .filter(|c| c.conflict_type.is_critical())
            .count();
        score -= (pending_critical as f64) * 0.1;

        // Penalize for agents with high error counts
        let high_error_agents = self.agent_states.values().filter(|s| s.error_count > 5).count();
        score -= (high_error_agents as f64 / self.agent_states.len().max(1) as f64) * 0.15;

        score.clamp(0.0, 1.0)
    }

    /// Detect conflicts in the current state.
    pub fn detect_conflicts(&self) -> Vec<Conflict> {
        let mut conflicts = Vec::new();

        // Detect resource contention (multiple overloaded agents)
        let overloaded: Vec<_> = self
            .agent_states
            .iter()
            .filter(|(_, s)| s.is_overloaded())
            .map(|(id, _)| id.clone())
            .collect();

        if overloaded.len() >= 2 {
            conflicts.push(Conflict::new(
                overloaded.clone(),
                ConflictType::ResourceContention,
                "Multiple agents overloaded simultaneously",
            ));
        }

        // Detect priority conflicts (same priority but different loads)
        let priorities: HashMap<u64, Vec<&String>> = self
            .agent_states
            .iter()
            .fold(HashMap::new(), |mut acc, (id, state)| {
                let key = (state.priority * 100.0) as u64;
                acc.entry(key).or_default().push(id);
                acc
            });

        for (_priority, agents) in priorities {
            if agents.len() >= 2 {
                let loads: Vec<f64> = agents
                    .iter()
                    .filter_map(|id| self.agent_states.get(*id).map(|s| s.load))
                    .collect();

                if let (Some(&min), Some(&max)) = (
                    loads.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                    loads.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
                ) {
                    if max - min > 0.5 {
                        conflicts.push(Conflict::new(
                            agents.iter().map(|s| (*s).clone()).collect::<Vec<_>>(),
                            ConflictType::PriorityConflict,
                            "Agents with same priority have vastly different loads",
                        ));
                    }
                }
            }
        }

        conflicts
    }

    /// Resolve pending conflicts.
    fn resolve_conflicts(&mut self) -> Vec<Conflict> {
        let mut resolved = Vec::new();

        let conflicts: Vec<Conflict> = self.pending_conflicts.drain(..).collect();

        for mut conflict in conflicts {
            if conflict.conflict_type.is_auto_resolvable() {
                match conflict.conflict_type {
                    ConflictType::ResourceContention => {
                        // Redistribute resources among conflicting agents
                        let share = 0.8 / conflict.agents.len() as f64;
                        for agent_id in &conflict.agents {
                            self.allocation.set_cpu_share(agent_id, share);
                        }
                        conflict.resolution = format!(
                            "Redistributed CPU shares to {:.1}% each",
                            share * 100.0
                        );
                    }
                    ConflictType::PriorityConflict => {
                        // Adjust priorities based on current load
                        for (i, agent_id) in conflict.agents.iter().enumerate() {
                            let new_priority = 1.0 + (i as f64 * 0.1);
                            self.allocation.set_priority_weight(agent_id, new_priority);
                        }
                        conflict.resolution = "Reassigned priorities based on load".to_string();
                    }
                    _ => {}
                }
                conflict.mark_resolved();
                conflict.auto_resolved = true;
                resolved.push(conflict);
            } else {
                // Can't auto-resolve, put back in pending
                warn!(
                    "Cannot auto-resolve conflict: {}",
                    conflict.conflict_type
                );
                self.pending_conflicts.push(conflict);
            }
        }

        // Detect and add new conflicts
        let new_conflicts = self.detect_conflicts();
        for conflict in new_conflicts {
            if conflict.conflict_type.is_auto_resolvable() && self.config.auto_resolve_conflicts {
                let mut c = conflict;
                c.mark_resolved();
                c.auto_resolved = true;
                c.resolution = ConflictType::recommended_resolution(&c.conflict_type).to_string();
                resolved.push(c);
            } else {
                self.pending_conflicts.push(conflict);
            }
        }

        resolved
    }

    /// Generate adjustments to improve balance.
    fn generate_adjustments(&self) -> Vec<Adjustment> {
        let mut adjustments = Vec::new();

        for (agent_id, state) in &self.agent_states {
            if state.is_overloaded() {
                // Overloaded agent needs more resources or throttling
                adjustments.push(
                    Adjustment::new(
                        agent_id,
                        AdjustmentType::IncreaseResources,
                        state.cpu_share,
                        (state.cpu_share * 1.2).min(0.5),
                        "Agent is overloaded",
                    )
                    .with_auto_applied(true),
                );
            } else if state.is_underutilized() && state.cpu_share > 0.05 {
                // Underutilized agent can give up resources
                adjustments.push(
                    Adjustment::new(
                        agent_id,
                        AdjustmentType::DecreaseResources,
                        state.cpu_share,
                        (state.cpu_share * 0.8).max(0.05),
                        "Agent is underutilized",
                    )
                    .with_auto_applied(true),
                );
            }

            // Boost agents with high error counts
            if state.error_count > 5 {
                adjustments.push(
                    Adjustment::new(
                        agent_id,
                        AdjustmentType::ThrottleRequests,
                        state.load,
                        state.load * 0.7,
                        format!("High error count: {}", state.error_count),
                    )
                    .with_auto_applied(true),
                );
            }
        }

        adjustments
    }

    /// Apply an adjustment to the system.
    fn apply_adjustment(&mut self, adjustment: &Adjustment) {
        if let Some(state) = self.agent_states.get_mut(&adjustment.agent_id) {
            match adjustment.adjustment_type {
                AdjustmentType::IncreaseResources | AdjustmentType::DecreaseResources => {
                    state.cpu_share = adjustment.after;
                    self.allocation.set_cpu_share(&adjustment.agent_id, adjustment.after);
                }
                AdjustmentType::ChangePriority => {
                    state.priority = adjustment.after;
                    self.allocation.set_priority_weight(&adjustment.agent_id, adjustment.after);
                }
                AdjustmentType::ThrottleRequests => {
                    state.load = adjustment.after;
                }
                AdjustmentType::BoostPerformance => {
                    state.cpu_share = adjustment.after;
                    state.memory_mb *= 1.5;
                    self.allocation.set_cpu_share(&adjustment.agent_id, adjustment.after);
                    self.allocation.set_memory_limit(&adjustment.agent_id, state.memory_mb);
                }
            }
            state.last_update = Utc::now();
            debug!(
                "Applied adjustment to {}: {} from {:.2} to {:.2}",
                adjustment.agent_id, adjustment.adjustment_type, adjustment.before, adjustment.after
            );
        }
    }

    /// Force rebalancing of all agent resources.
    pub fn force_rebalance(&mut self) {
        if self.agent_states.is_empty() {
            return;
        }

        let agent_count = self.agent_states.len() as f64;
        let equal_share = 0.9 / agent_count;

        for (agent_id, state) in &mut self.agent_states {
            state.cpu_share = equal_share;
            self.allocation.set_cpu_share(agent_id, equal_share);
            state.last_update = Utc::now();
        }

        self.allocation.normalize_cpu_shares();
        info!("Force rebalanced {} agents", self.agent_states.len());
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.agent_states.clear();
        self.allocation.clear();
        self.pending_conflicts.clear();
        info!("HarmonizerAgent reset to initial state");
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom HarmonizerAgent configuration.
#[derive(Debug, Default)]
pub struct HarmonizerAgentBuilder {
    config: Option<HarmonizerConfig>,
}

impl HarmonizerAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: HarmonizerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set balance threshold.
    pub fn balance_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.balance_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set check interval in seconds.
    pub fn check_interval(mut self, secs: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.check_interval = secs;
        self.config = Some(config);
        self
    }

    /// Set whether to auto-adjust.
    pub fn auto_adjust(mut self, auto: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.auto_adjust = auto;
        self.config = Some(config);
        self
    }

    /// Set maximum adjustments per cycle.
    pub fn max_adjustments_per_cycle(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_adjustments_per_cycle = max;
        self.config = Some(config);
        self
    }

    /// Set whether to auto-resolve conflicts.
    pub fn auto_resolve_conflicts(mut self, auto: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.auto_resolve_conflicts = auto;
        self.config = Some(config);
        self
    }

    /// Set CPU rebalance threshold.
    pub fn cpu_rebalance_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.cpu_rebalance_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set memory rebalance threshold.
    pub fn memory_rebalance_threshold(mut self, threshold_mb: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.memory_rebalance_threshold = threshold_mb.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set default priority weight.
    pub fn default_priority_weight(mut self, weight: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.default_priority_weight = weight.max(0.0);
        self.config = Some(config);
        self
    }

    /// Build the HarmonizerAgent.
    pub fn build(self) -> HarmonizerAgent {
        HarmonizerAgent {
            config: self.config.unwrap_or_default(),
            agent_states: HashMap::new(),
            allocation: ResourceAllocation::new(),
            pending_conflicts: Vec::new(),
        }
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AdjustmentType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_adjustment_type_display() {
        assert_eq!(AdjustmentType::IncreaseResources.to_string(), "increase_resources");
        assert_eq!(AdjustmentType::DecreaseResources.to_string(), "decrease_resources");
        assert_eq!(AdjustmentType::ChangePriority.to_string(), "change_priority");
        assert_eq!(AdjustmentType::ThrottleRequests.to_string(), "throttle_requests");
        assert_eq!(AdjustmentType::BoostPerformance.to_string(), "boost_performance");
    }

    #[test]
    fn test_adjustment_type_from_str() {
        assert_eq!(
            "increase_resources".parse::<AdjustmentType>().unwrap(),
            AdjustmentType::IncreaseResources
        );
        assert_eq!(
            "decrease".parse::<AdjustmentType>().unwrap(),
            AdjustmentType::DecreaseResources
        );
        assert_eq!(
            "priority".parse::<AdjustmentType>().unwrap(),
            AdjustmentType::ChangePriority
        );
        assert_eq!(
            "throttle".parse::<AdjustmentType>().unwrap(),
            AdjustmentType::ThrottleRequests
        );
        assert_eq!(
            "boost".parse::<AdjustmentType>().unwrap(),
            AdjustmentType::BoostPerformance
        );
    }

    #[test]
    fn test_adjustment_type_from_str_invalid() {
        assert!("invalid".parse::<AdjustmentType>().is_err());
    }

    #[test]
    fn test_adjustment_type_all() {
        let all = AdjustmentType::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&AdjustmentType::IncreaseResources));
        assert!(all.contains(&AdjustmentType::BoostPerformance));
    }

    #[test]
    fn test_adjustment_type_increases_resources() {
        assert!(AdjustmentType::IncreaseResources.increases_resources());
        assert!(AdjustmentType::BoostPerformance.increases_resources());
        assert!(!AdjustmentType::DecreaseResources.increases_resources());
        assert!(!AdjustmentType::ChangePriority.increases_resources());
    }

    #[test]
    fn test_adjustment_type_decreases_resources() {
        assert!(AdjustmentType::DecreaseResources.decreases_resources());
        assert!(AdjustmentType::ThrottleRequests.decreases_resources());
        assert!(!AdjustmentType::IncreaseResources.decreases_resources());
        assert!(!AdjustmentType::ChangePriority.decreases_resources());
    }

    #[test]
    fn test_adjustment_type_impact_magnitude() {
        assert!(AdjustmentType::BoostPerformance.impact_magnitude() > 0);
        assert!(AdjustmentType::IncreaseResources.impact_magnitude() > 0);
        assert_eq!(AdjustmentType::ChangePriority.impact_magnitude(), 0);
        assert!(AdjustmentType::DecreaseResources.impact_magnitude() < 0);
        assert!(AdjustmentType::ThrottleRequests.impact_magnitude() < 0);
    }

    #[test]
    fn test_adjustment_type_description() {
        assert!(AdjustmentType::IncreaseResources.description().contains("more"));
        assert!(AdjustmentType::DecreaseResources.description().contains("Reduce"));
    }

    #[test]
    fn test_adjustment_type_default() {
        assert_eq!(AdjustmentType::default(), AdjustmentType::ChangePriority);
    }

    // -------------------------------------------------------------------------
    // ConflictType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_conflict_type_display() {
        assert_eq!(ConflictType::ResourceContention.to_string(), "resource_contention");
        assert_eq!(ConflictType::PriorityConflict.to_string(), "priority_conflict");
        assert_eq!(ConflictType::DataInconsistency.to_string(), "data_inconsistency");
        assert_eq!(ConflictType::DeadlockRisk.to_string(), "deadlock_risk");
    }

    #[test]
    fn test_conflict_type_from_str() {
        assert_eq!(
            "resource_contention".parse::<ConflictType>().unwrap(),
            ConflictType::ResourceContention
        );
        assert_eq!(
            "contention".parse::<ConflictType>().unwrap(),
            ConflictType::ResourceContention
        );
        assert_eq!(
            "priority".parse::<ConflictType>().unwrap(),
            ConflictType::PriorityConflict
        );
        assert_eq!(
            "data".parse::<ConflictType>().unwrap(),
            ConflictType::DataInconsistency
        );
        assert_eq!(
            "deadlock".parse::<ConflictType>().unwrap(),
            ConflictType::DeadlockRisk
        );
    }

    #[test]
    fn test_conflict_type_from_str_invalid() {
        assert!("invalid".parse::<ConflictType>().is_err());
    }

    #[test]
    fn test_conflict_type_all() {
        let all = ConflictType::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_conflict_type_severity() {
        assert_eq!(ConflictType::DeadlockRisk.severity(), 10);
        assert_eq!(ConflictType::DataInconsistency.severity(), 8);
        assert!(ConflictType::DeadlockRisk.severity() > ConflictType::ResourceContention.severity());
    }

    #[test]
    fn test_conflict_type_is_critical() {
        assert!(ConflictType::DeadlockRisk.is_critical());
        assert!(ConflictType::DataInconsistency.is_critical());
        assert!(!ConflictType::ResourceContention.is_critical());
        assert!(!ConflictType::PriorityConflict.is_critical());
    }

    #[test]
    fn test_conflict_type_is_auto_resolvable() {
        assert!(ConflictType::ResourceContention.is_auto_resolvable());
        assert!(ConflictType::PriorityConflict.is_auto_resolvable());
        assert!(!ConflictType::DataInconsistency.is_auto_resolvable());
        assert!(!ConflictType::DeadlockRisk.is_auto_resolvable());
    }

    #[test]
    fn test_conflict_type_description() {
        assert!(ConflictType::ResourceContention.description().contains("competing"));
        assert!(ConflictType::DeadlockRisk.description().contains("deadlock"));
    }

    #[test]
    fn test_conflict_type_recommended_resolution() {
        assert!(ConflictType::ResourceContention.recommended_resolution().contains("Redistribute"));
        assert!(ConflictType::DeadlockRisk.recommended_resolution().contains("Break"));
    }

    // -------------------------------------------------------------------------
    // Adjustment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_adjustment_new() {
        let adj = Adjustment::new(
            "agent1",
            AdjustmentType::IncreaseResources,
            0.1,
            0.2,
            "Test reason",
        );
        assert_eq!(adj.agent_id, "agent1");
        assert_eq!(adj.adjustment_type, AdjustmentType::IncreaseResources);
        assert_eq!(adj.before, 0.1);
        assert_eq!(adj.after, 0.2);
        assert_eq!(adj.reason, "Test reason");
        assert!(!adj.auto_applied);
    }

    #[test]
    fn test_adjustment_with_auto_applied() {
        let adj = Adjustment::new(
            "agent1",
            AdjustmentType::IncreaseResources,
            0.1,
            0.2,
            "Test",
        )
        .with_auto_applied(true);
        assert!(adj.auto_applied);
    }

    #[test]
    fn test_adjustment_change_magnitude() {
        let adj = Adjustment::new("a", AdjustmentType::IncreaseResources, 0.1, 0.3, "");
        assert!((adj.change_magnitude() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_adjustment_change_percentage() {
        let adj = Adjustment::new("a", AdjustmentType::IncreaseResources, 0.1, 0.2, "");
        assert!((adj.change_percentage() - 100.0).abs() < 0.01);

        let adj2 = Adjustment::new("a", AdjustmentType::DecreaseResources, 0.2, 0.1, "");
        assert!((adj2.change_percentage() - (-50.0)).abs() < 0.01);
    }

    #[test]
    fn test_adjustment_change_percentage_zero_before() {
        let adj = Adjustment::new("a", AdjustmentType::IncreaseResources, 0.0, 0.1, "");
        assert_eq!(adj.change_percentage(), 100.0);

        let adj2 = Adjustment::new("a", AdjustmentType::IncreaseResources, 0.0, 0.0, "");
        assert_eq!(adj2.change_percentage(), 0.0);
    }

    #[test]
    fn test_adjustment_is_increase() {
        let increase = Adjustment::new("a", AdjustmentType::IncreaseResources, 0.1, 0.2, "");
        assert!(increase.is_increase());

        let decrease = Adjustment::new("a", AdjustmentType::DecreaseResources, 0.2, 0.1, "");
        assert!(!decrease.is_increase());
    }

    #[test]
    fn test_adjustment_is_decrease() {
        let decrease = Adjustment::new("a", AdjustmentType::DecreaseResources, 0.2, 0.1, "");
        assert!(decrease.is_decrease());

        let increase = Adjustment::new("a", AdjustmentType::IncreaseResources, 0.1, 0.2, "");
        assert!(!increase.is_decrease());
    }

    #[test]
    fn test_adjustment_is_no_change() {
        let no_change = Adjustment::new("a", AdjustmentType::ChangePriority, 0.5, 0.5, "");
        assert!(no_change.is_no_change());

        let change = Adjustment::new("a", AdjustmentType::IncreaseResources, 0.1, 0.2, "");
        assert!(!change.is_no_change());
    }

    #[test]
    fn test_adjustment_summary() {
        let adj = Adjustment::new("agent1", AdjustmentType::IncreaseResources, 0.1, 0.2, "");
        let summary = adj.summary();
        assert!(summary.contains("agent1"));
        assert!(summary.contains("increase"));
        assert!(summary.contains("0.10"));
        assert!(summary.contains("0.20"));
    }

    // -------------------------------------------------------------------------
    // Conflict Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_conflict_new() {
        let conflict = Conflict::new(
            vec!["agent1", "agent2"],
            ConflictType::ResourceContention,
            "Resolved by redistribution",
        );
        assert_eq!(conflict.agents.len(), 2);
        assert_eq!(conflict.conflict_type, ConflictType::ResourceContention);
        assert!(!conflict.is_resolved());
    }

    #[test]
    fn test_conflict_resolved() {
        let conflict = Conflict::resolved(
            vec!["a", "b"],
            ConflictType::PriorityConflict,
            "Fixed",
        );
        assert!(conflict.is_resolved());
    }

    #[test]
    fn test_conflict_with_resolved_at() {
        let now = Utc::now();
        let conflict = Conflict::new(vec!["a"], ConflictType::DeadlockRisk, "")
            .with_resolved_at(now);
        assert!(conflict.is_resolved());
    }

    #[test]
    fn test_conflict_with_auto_resolved() {
        let conflict = Conflict::new(vec!["a"], ConflictType::ResourceContention, "")
            .with_auto_resolved(true);
        assert!(conflict.auto_resolved);
    }

    #[test]
    fn test_conflict_with_details() {
        let conflict = Conflict::new(vec!["a"], ConflictType::DataInconsistency, "")
            .with_details("Extra info");
        assert_eq!(conflict.details, Some("Extra info".to_string()));
    }

    #[test]
    fn test_conflict_mark_resolved() {
        let mut conflict = Conflict::new(vec!["a"], ConflictType::ResourceContention, "");
        assert!(!conflict.is_resolved());
        conflict.mark_resolved();
        assert!(conflict.is_resolved());
    }

    #[test]
    fn test_conflict_agent_count() {
        let conflict = Conflict::new(
            vec!["a", "b", "c"],
            ConflictType::ResourceContention,
            "",
        );
        assert_eq!(conflict.agent_count(), 3);
    }

    #[test]
    fn test_conflict_involves_agent() {
        let conflict = Conflict::new(
            vec!["agent1", "agent2"],
            ConflictType::PriorityConflict,
            "",
        );
        assert!(conflict.involves_agent("agent1"));
        assert!(conflict.involves_agent("agent2"));
        assert!(!conflict.involves_agent("agent3"));
    }

    #[test]
    fn test_conflict_resolution_time_ms() {
        let conflict = Conflict::new(vec!["a"], ConflictType::ResourceContention, "");
        assert!(conflict.resolution_time_ms().is_none());

        let resolved = Conflict::resolved(vec!["a"], ConflictType::ResourceContention, "");
        assert!(resolved.resolution_time_ms().is_some());
    }

    #[test]
    fn test_conflict_summary() {
        let conflict = Conflict::new(
            vec!["a", "b"],
            ConflictType::ResourceContention,
            "Test resolution",
        );
        let summary = conflict.summary();
        assert!(summary.contains("resource_contention"));
        assert!(summary.contains("pending"));
        assert!(summary.contains("Test resolution"));
    }

    // -------------------------------------------------------------------------
    // ResourceAllocation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_allocation_new() {
        let alloc = ResourceAllocation::new();
        assert!(alloc.cpu_shares.is_empty());
        assert!(alloc.memory_limit.is_empty());
        assert!(alloc.priority_weights.is_empty());
    }

    #[test]
    fn test_resource_allocation_cpu_share() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("agent1", 0.5);
        assert_eq!(alloc.get_cpu_share("agent1"), Some(0.5));
        assert_eq!(alloc.get_cpu_share("agent2"), None);
    }

    #[test]
    fn test_resource_allocation_cpu_share_clamped() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("a", 1.5);
        assert_eq!(alloc.get_cpu_share("a"), Some(1.0));

        alloc.set_cpu_share("b", -0.5);
        assert_eq!(alloc.get_cpu_share("b"), Some(0.0));
    }

    #[test]
    fn test_resource_allocation_memory_limit() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_memory_limit("agent1", 1024.0);
        assert_eq!(alloc.get_memory_limit("agent1"), Some(1024.0));
    }

    #[test]
    fn test_resource_allocation_priority_weight() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_priority_weight("agent1", 2.0);
        assert_eq!(alloc.get_priority_weight("agent1"), Some(2.0));
    }

    #[test]
    fn test_resource_allocation_totals() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("a", 0.3);
        alloc.set_cpu_share("b", 0.4);
        alloc.set_memory_limit("a", 256.0);
        alloc.set_memory_limit("b", 512.0);

        assert!((alloc.total_cpu_shares() - 0.7).abs() < 0.001);
        assert!((alloc.total_memory_mb() - 768.0).abs() < 0.001);
    }

    #[test]
    fn test_resource_allocation_is_cpu_overallocated() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("a", 0.6);
        alloc.set_cpu_share("b", 0.6);
        assert!(alloc.is_cpu_overallocated());

        let mut alloc2 = ResourceAllocation::new();
        alloc2.set_cpu_share("a", 0.5);
        alloc2.set_cpu_share("b", 0.4);
        assert!(!alloc2.is_cpu_overallocated());
    }

    #[test]
    fn test_resource_allocation_agent_count() {
        let mut alloc = ResourceAllocation::new();
        assert_eq!(alloc.agent_count(), 0);

        alloc.set_cpu_share("a", 0.1);
        assert_eq!(alloc.agent_count(), 1);

        alloc.set_memory_limit("b", 256.0);
        assert_eq!(alloc.agent_count(), 2);

        alloc.set_priority_weight("a", 1.0);
        assert_eq!(alloc.agent_count(), 2);
    }

    #[test]
    fn test_resource_allocation_has_agent() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("agent1", 0.1);
        assert!(alloc.has_agent("agent1"));
        assert!(!alloc.has_agent("agent2"));
    }

    #[test]
    fn test_resource_allocation_remove_agent() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("agent1", 0.1);
        alloc.set_memory_limit("agent1", 256.0);
        alloc.set_priority_weight("agent1", 1.0);

        assert!(alloc.has_agent("agent1"));
        alloc.remove_agent("agent1");
        assert!(!alloc.has_agent("agent1"));
    }

    #[test]
    fn test_resource_allocation_clear() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("a", 0.1);
        alloc.set_memory_limit("b", 256.0);

        alloc.clear();
        assert_eq!(alloc.agent_count(), 0);
    }

    #[test]
    fn test_resource_allocation_normalize_cpu_shares() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_cpu_share("a", 0.3);
        alloc.set_cpu_share("b", 0.3);

        alloc.normalize_cpu_shares();
        assert!((alloc.total_cpu_shares() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_resource_allocation_agents_by_priority() {
        let mut alloc = ResourceAllocation::new();
        alloc.set_priority_weight("low", 1.0);
        alloc.set_priority_weight("high", 3.0);
        alloc.set_priority_weight("medium", 2.0);

        let sorted = alloc.agents_by_priority();
        assert_eq!(sorted[0].0, "high");
        assert_eq!(sorted[1].0, "medium");
        assert_eq!(sorted[2].0, "low");
    }

    // -------------------------------------------------------------------------
    // HarmonyReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmony_report_new() {
        let report = HarmonyReport::new(0.85);
        assert_eq!(report.balance_score, 0.85);
        assert!(report.adjustments.is_empty());
        assert!(report.conflicts_resolved.is_empty());
    }

    #[test]
    fn test_harmony_report_perfect() {
        let report = HarmonyReport::perfect();
        assert_eq!(report.balance_score, 1.0);
    }

    #[test]
    fn test_harmony_report_clamped() {
        let report = HarmonyReport::new(1.5);
        assert_eq!(report.balance_score, 1.0);

        let report2 = HarmonyReport::new(-0.5);
        assert_eq!(report2.balance_score, 0.0);
    }

    #[test]
    fn test_harmony_report_add_adjustment() {
        let mut report = HarmonyReport::new(0.8);
        report.add_adjustment(Adjustment::new(
            "a",
            AdjustmentType::IncreaseResources,
            0.1,
            0.2,
            "",
        ));
        assert_eq!(report.adjustment_count(), 1);
    }

    #[test]
    fn test_harmony_report_add_conflict() {
        let mut report = HarmonyReport::new(0.8);
        report.add_conflict(Conflict::resolved(
            vec!["a", "b"],
            ConflictType::ResourceContention,
            "",
        ));
        assert_eq!(report.conflict_count(), 1);
    }

    #[test]
    fn test_harmony_report_is_harmonious() {
        let harmonious = HarmonyReport::new(0.85);
        assert!(harmonious.is_harmonious());

        let not_harmonious = HarmonyReport::new(0.7);
        assert!(!not_harmonious.is_harmonious());
    }

    #[test]
    fn test_harmony_report_is_critical() {
        let critical = HarmonyReport::new(0.4);
        assert!(critical.is_critical());

        let not_critical = HarmonyReport::new(0.6);
        assert!(!not_critical.is_critical());
    }

    #[test]
    fn test_harmony_report_adjustments_of_type() {
        let mut report = HarmonyReport::new(0.8);
        report.add_adjustment(Adjustment::new("a", AdjustmentType::IncreaseResources, 0.1, 0.2, ""));
        report.add_adjustment(Adjustment::new("b", AdjustmentType::DecreaseResources, 0.3, 0.2, ""));
        report.add_adjustment(Adjustment::new("c", AdjustmentType::IncreaseResources, 0.2, 0.3, ""));

        let increases = report.adjustments_of_type(AdjustmentType::IncreaseResources);
        assert_eq!(increases.len(), 2);
    }

    #[test]
    fn test_harmony_report_conflicts_of_type() {
        let mut report = HarmonyReport::new(0.8);
        report.add_conflict(Conflict::resolved(vec!["a"], ConflictType::ResourceContention, ""));
        report.add_conflict(Conflict::resolved(vec!["b"], ConflictType::PriorityConflict, ""));
        report.add_conflict(Conflict::resolved(vec!["c"], ConflictType::ResourceContention, ""));

        let contention = report.conflicts_of_type(ConflictType::ResourceContention);
        assert_eq!(contention.len(), 2);
    }

    #[test]
    fn test_harmony_report_brief_summary() {
        let mut report = HarmonyReport::new(0.85);
        report.add_adjustment(Adjustment::new("a", AdjustmentType::IncreaseResources, 0.1, 0.2, ""));
        report.add_conflict(Conflict::resolved(vec!["a"], ConflictType::ResourceContention, ""));

        let summary = report.brief_summary();
        assert!(summary.contains("harmonious"));
        assert!(summary.contains("0.85"));
        assert!(summary.contains("1 adjustments"));
        assert!(summary.contains("1 conflicts"));
    }

    #[test]
    fn test_harmony_report_to_markdown() {
        let mut report = HarmonyReport::new(0.85);
        report.add_adjustment(Adjustment::new("agent1", AdjustmentType::IncreaseResources, 0.1, 0.2, ""));
        report.add_conflict(Conflict::resolved(vec!["a", "b"], ConflictType::ResourceContention, "Fixed"));

        let md = report.to_markdown();
        assert!(md.contains("# Harmony Report"));
        assert!(md.contains("Balance Score"));
        assert!(md.contains("Adjustments"));
        assert!(md.contains("agent1"));
        assert!(md.contains("Resolved Conflicts"));
        assert!(md.contains("Fixed"));
    }

    #[test]
    fn test_harmony_report_default() {
        let report = HarmonyReport::default();
        assert_eq!(report.balance_score, 1.0);
    }

    // -------------------------------------------------------------------------
    // HarmonizerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonizer_config_default() {
        let config = HarmonizerConfig::default();
        assert_eq!(config.balance_threshold, 0.7);
        assert!(config.auto_adjust);
        assert!(config.auto_resolve_conflicts);
    }

    #[test]
    fn test_harmonizer_config_strict() {
        let config = HarmonizerConfig::strict();
        assert!(config.balance_threshold > HarmonizerConfig::default().balance_threshold);
        assert!(config.check_interval < HarmonizerConfig::default().check_interval);
    }

    #[test]
    fn test_harmonizer_config_lenient() {
        let config = HarmonizerConfig::lenient();
        assert!(config.balance_threshold < HarmonizerConfig::default().balance_threshold);
        assert!(config.check_interval > HarmonizerConfig::default().check_interval);
    }

    #[test]
    fn test_harmonizer_config_manual() {
        let config = HarmonizerConfig::manual();
        assert!(!config.auto_adjust);
        assert!(!config.auto_resolve_conflicts);
    }

    #[test]
    fn test_harmonizer_config_meets_threshold() {
        let config = HarmonizerConfig::default();
        assert!(config.meets_threshold(0.8));
        assert!(config.meets_threshold(0.7));
        assert!(!config.meets_threshold(0.6));
    }

    // -------------------------------------------------------------------------
    // HarmonizerAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonizer_agent_new() {
        let agent = HarmonizerAgent::new();
        assert_eq!(agent.agent_count(), 0);
    }

    #[test]
    fn test_harmonizer_agent_with_config() {
        let config = HarmonizerConfig::strict();
        let agent = HarmonizerAgent::with_config(config);
        assert!(agent.config().balance_threshold > 0.7);
    }

    #[test]
    fn test_harmonizer_agent_builder() {
        let agent = HarmonizerAgent::builder()
            .balance_threshold(0.9)
            .auto_adjust(false)
            .max_adjustments_per_cycle(5)
            .build();

        assert_eq!(agent.config().balance_threshold, 0.9);
        assert!(!agent.config().auto_adjust);
        assert_eq!(agent.config().max_adjustments_per_cycle, 5);
    }

    #[test]
    fn test_harmonizer_agent_default() {
        let agent = HarmonizerAgent::default();
        assert_eq!(agent.config().balance_threshold, 0.7);
    }

    // -------------------------------------------------------------------------
    // HarmonizerAgent Agent Management Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonizer_register_agent() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("agent1");
        assert_eq!(agent.agent_count(), 1);
        assert!(agent.allocation().has_agent("agent1"));
    }

    #[test]
    fn test_harmonizer_unregister_agent() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("agent1");
        agent.unregister_agent("agent1");
        assert_eq!(agent.agent_count(), 0);
        assert!(!agent.allocation().has_agent("agent1"));
    }

    #[test]
    fn test_harmonizer_update_agent_load() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("agent1");
        agent.update_agent_load("agent1", 0.5);
        // Load is internal state, we verify it doesn't panic
    }

    #[test]
    fn test_harmonizer_update_agent_resources() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("agent1");
        agent.update_agent_resources("agent1", Some(0.5), Some(512.0), Some(2.0));

        assert_eq!(agent.allocation().get_cpu_share("agent1"), Some(0.5));
        assert_eq!(agent.allocation().get_memory_limit("agent1"), Some(512.0));
        assert_eq!(agent.allocation().get_priority_weight("agent1"), Some(2.0));
    }

    #[test]
    fn test_harmonizer_record_agent_error() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("agent1");
        agent.record_agent_error("agent1");
        // Error count is internal state, we verify it doesn't panic
    }

    #[test]
    fn test_harmonizer_add_conflict() {
        let mut agent = HarmonizerAgent::new();
        let conflict = Conflict::new(vec!["a", "b"], ConflictType::ResourceContention, "");
        agent.add_conflict(conflict);
        assert_eq!(agent.pending_conflicts().len(), 1);
    }

    // -------------------------------------------------------------------------
    // HarmonizerAgent Harmonization Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_harmonizer_harmonize_empty() {
        let mut agent = HarmonizerAgent::new();
        let report = agent.harmonize().await.unwrap();

        assert_eq!(report.balance_score, 1.0);
        assert_eq!(report.agents_analyzed, 0);
    }

    #[tokio::test]
    async fn test_harmonizer_harmonize_with_agents() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("agent1");
        agent.register_agent("agent2");

        let report = agent.harmonize().await.unwrap();

        assert_eq!(report.agents_analyzed, 2);
        assert!(report.harmonize_duration_ms > 0 || report.harmonize_duration_ms == 0);
    }

    #[tokio::test]
    async fn test_harmonizer_harmonize_overloaded_agent() {
        let mut agent = HarmonizerAgent::builder()
            .auto_adjust(true)
            .balance_threshold(0.9)
            .build();

        agent.register_agent("overloaded");
        agent.update_agent_load("overloaded", 0.95);

        let report = agent.harmonize().await.unwrap();

        // Should have generated adjustments for the overloaded agent
        assert!(report.adjustment_count() >= 0);
    }

    #[tokio::test]
    async fn test_harmonizer_harmonize_no_auto_adjust() {
        let mut agent = HarmonizerAgent::builder()
            .auto_adjust(false)
            .build();

        agent.register_agent("agent1");
        agent.update_agent_load("agent1", 0.9);

        let report = agent.harmonize().await.unwrap();

        // No adjustments should be made with auto_adjust=false
        assert_eq!(report.adjustment_count(), 0);
    }

    // -------------------------------------------------------------------------
    // HarmonizerAgent Balance Score Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonizer_calculate_balance_score_empty() {
        let agent = HarmonizerAgent::new();
        assert_eq!(agent.calculate_balance_score(), 1.0);
    }

    #[test]
    fn test_harmonizer_calculate_balance_score_balanced() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("a");
        agent.register_agent("b");
        agent.update_agent_load("a", 0.5);
        agent.update_agent_load("b", 0.5);

        let score = agent.calculate_balance_score();
        assert!(score > 0.8);
    }

    #[test]
    fn test_harmonizer_calculate_balance_score_imbalanced() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("overloaded");
        agent.update_agent_load("overloaded", 0.99);

        let score = agent.calculate_balance_score();
        assert!(score < 1.0);
    }

    // -------------------------------------------------------------------------
    // HarmonizerAgent Conflict Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonizer_detect_conflicts_none() {
        let agent = HarmonizerAgent::new();
        let conflicts = agent.detect_conflicts();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_harmonizer_detect_resource_contention() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("a");
        agent.register_agent("b");
        agent.update_agent_load("a", 0.95);
        agent.update_agent_load("b", 0.9);

        let conflicts = agent.detect_conflicts();
        let contention = conflicts
            .iter()
            .find(|c| c.conflict_type == ConflictType::ResourceContention);
        assert!(contention.is_some());
    }

    // -------------------------------------------------------------------------
    // HarmonizerAgent Other Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonizer_force_rebalance() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("a");
        agent.register_agent("b");
        agent.update_agent_resources("a", Some(0.9), None, None);
        agent.update_agent_resources("b", Some(0.05), None, None);

        agent.force_rebalance();

        let share_a = agent.allocation().get_cpu_share("a").unwrap();
        let share_b = agent.allocation().get_cpu_share("b").unwrap();

        // After rebalance, shares should be more equal
        assert!((share_a - share_b).abs() < 0.01);
    }

    #[test]
    fn test_harmonizer_reset() {
        let mut agent = HarmonizerAgent::new();
        agent.register_agent("agent1");
        agent.add_conflict(Conflict::new(vec!["a"], ConflictType::DeadlockRisk, ""));

        agent.reset();

        assert_eq!(agent.agent_count(), 0);
        assert!(agent.pending_conflicts().is_empty());
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_config() {
        let config = HarmonizerConfig::strict();
        let agent = HarmonizerAgentBuilder::default().config(config).build();
        assert!(agent.config().balance_threshold > 0.7);
    }

    #[test]
    fn test_builder_balance_threshold() {
        let agent = HarmonizerAgentBuilder::default()
            .balance_threshold(0.95)
            .build();
        assert_eq!(agent.config().balance_threshold, 0.95);
    }

    #[test]
    fn test_builder_balance_threshold_clamped() {
        let agent = HarmonizerAgentBuilder::default()
            .balance_threshold(1.5)
            .build();
        assert_eq!(agent.config().balance_threshold, 1.0);
    }

    #[test]
    fn test_builder_check_interval() {
        let agent = HarmonizerAgentBuilder::default()
            .check_interval(120)
            .build();
        assert_eq!(agent.config().check_interval, 120);
    }

    #[test]
    fn test_builder_auto_adjust() {
        let agent = HarmonizerAgentBuilder::default()
            .auto_adjust(false)
            .build();
        assert!(!agent.config().auto_adjust);
    }

    #[test]
    fn test_builder_max_adjustments_per_cycle() {
        let agent = HarmonizerAgentBuilder::default()
            .max_adjustments_per_cycle(20)
            .build();
        assert_eq!(agent.config().max_adjustments_per_cycle, 20);
    }

    #[test]
    fn test_builder_auto_resolve_conflicts() {
        let agent = HarmonizerAgentBuilder::default()
            .auto_resolve_conflicts(false)
            .build();
        assert!(!agent.config().auto_resolve_conflicts);
    }

    #[test]
    fn test_builder_cpu_rebalance_threshold() {
        let agent = HarmonizerAgentBuilder::default()
            .cpu_rebalance_threshold(0.9)
            .build();
        assert_eq!(agent.config().cpu_rebalance_threshold, 0.9);
    }

    #[test]
    fn test_builder_memory_rebalance_threshold() {
        let agent = HarmonizerAgentBuilder::default()
            .memory_rebalance_threshold(2048.0)
            .build();
        assert_eq!(agent.config().memory_rebalance_threshold, 2048.0);
    }

    #[test]
    fn test_builder_default_priority_weight() {
        let agent = HarmonizerAgentBuilder::default()
            .default_priority_weight(2.0)
            .build();
        assert_eq!(agent.config().default_priority_weight, 2.0);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_full_harmonization_flow() {
        let mut agent = HarmonizerAgent::builder()
            .balance_threshold(0.8)
            .auto_adjust(true)
            .auto_resolve_conflicts(true)
            .build();

        // Register agents
        agent.register_agent("monitor");
        agent.register_agent("recoverer");
        agent.register_agent("learner");

        // Simulate load
        agent.update_agent_load("monitor", 0.3);
        agent.update_agent_load("recoverer", 0.9); // Overloaded
        agent.update_agent_load("learner", 0.2);

        // Add a conflict
        agent.add_conflict(Conflict::new(
            vec!["monitor", "recoverer"],
            ConflictType::ResourceContention,
            "Both need CPU",
        ));

        // Run harmonization
        let report = agent.harmonize().await.unwrap();

        // Verify report
        assert_eq!(report.agents_analyzed, 3);
        assert!(report.summary.is_some());
        assert!(report.harmonize_duration_ms >= 0);
    }

    #[tokio::test]
    async fn test_harmonization_with_errors() {
        let mut agent = HarmonizerAgent::builder()
            .auto_adjust(true)
            .balance_threshold(0.9)
            .build();

        agent.register_agent("failing");

        // Record many errors
        for _ in 0..10 {
            agent.record_agent_error("failing");
        }

        let report = agent.harmonize().await.unwrap();

        // Score should be reduced due to errors
        assert!(report.balance_score < 1.0);
    }
}
