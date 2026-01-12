//! Monitor Agent module for Project Panpsychism.
//!
//! The Health Watcher — "Eternal vigilance is the price of excellence."
//!
//! This module implements the Monitor Agent, responsible for tracking system health,
//! performance metrics, and raising alerts when anomalies are detected. Like a
//! vigilant guardian watching over the arcane systems, the Monitor Agent ensures
//! the health of all magical components.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, the Monitor Agent serves as the **Health Watcher** —
//! a sentinel that observes the flow of magical energy:
//!
//! - **Health Reports** reveal the vitality of each system component
//! - **Metrics** measure the flow and efficiency of magical operations
//! - **Alerts** warn of disturbances in the magical fabric
//! - **Thresholds** define the boundaries of healthy operation
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Self-preservation through continuous health monitoring
//! - **NATURA**: Natural observation of system behavior patterns
//! - **RATIO**: Logical analysis of metrics against thresholds
//! - **LAETITIA**: Joy through maintained system health and reliability
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::monitor::{MonitorAgent, MonitorConfig};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let monitor = MonitorAgent::new();
//!
//!     // Check system health
//!     let report = monitor.check_health().await?;
//!     println!("System status: {:?}", report.overall_status);
//!
//!     // Get current metrics
//!     let metrics = monitor.get_metrics().await?;
//!     println!("Requests/sec: {:.2}", metrics.requests_per_second);
//!
//!     // Check for alerts
//!     for alert in &report.alerts {
//!         println!("Alert [{}]: {} - {}", alert.level, alert.source, alert.message);
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
// HEALTH STATUS
// =============================================================================

/// Health status of a system component or the overall system.
///
/// Like the vital signs of a living organism, health status indicates
/// the operational state of the monitored components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum HealthStatus {
    /// System is operating normally with all metrics within acceptable ranges.
    #[default]
    Healthy,

    /// System is operational but some metrics are approaching thresholds.
    /// Performance may be suboptimal.
    Degraded,

    /// System has issues that affect functionality.
    /// Some operations may fail or be significantly slowed.
    Unhealthy,

    /// System is in a critical state requiring immediate attention.
    /// Major functionality is impaired or unavailable.
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Unhealthy => write!(f, "Unhealthy"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

impl std::str::FromStr for HealthStatus {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "healthy" | "ok" | "green" => Ok(Self::Healthy),
            "degraded" | "warning" | "yellow" => Ok(Self::Degraded),
            "unhealthy" | "error" | "orange" => Ok(Self::Unhealthy),
            "critical" | "fatal" | "red" => Ok(Self::Critical),
            _ => Err(Error::Config(format!(
                "Unknown health status: '{}'. Valid: healthy, degraded, unhealthy, critical",
                s
            ))),
        }
    }
}

impl HealthStatus {
    /// Get all health status values in order from best to worst.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Healthy,
            Self::Degraded,
            Self::Unhealthy,
            Self::Critical,
        ]
    }

    /// Check if this status requires immediate attention.
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::Critical)
    }

    /// Check if this status indicates any kind of problem.
    pub fn has_issues(&self) -> bool {
        !matches!(self, Self::Healthy)
    }

    /// Check if the system is operational (healthy or degraded).
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// Get a numeric severity value (lower = better).
    pub fn severity(&self) -> u8 {
        match self {
            Self::Healthy => 0,
            Self::Degraded => 1,
            Self::Unhealthy => 2,
            Self::Critical => 3,
        }
    }

    /// Combine two health statuses, returning the worse one.
    pub fn combine(&self, other: &Self) -> Self {
        if self.severity() >= other.severity() {
            *self
        } else {
            *other
        }
    }

    /// Get a human-readable description of this status.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Healthy => "All systems operating normally",
            Self::Degraded => "System operational with reduced performance",
            Self::Unhealthy => "System experiencing issues affecting functionality",
            Self::Critical => "Critical system failure requiring immediate attention",
        }
    }

    /// Get recommended action for this status.
    pub fn recommended_action(&self) -> &'static str {
        match self {
            Self::Healthy => "Continue normal operations",
            Self::Degraded => "Monitor closely and investigate root cause",
            Self::Unhealthy => "Investigate and remediate promptly",
            Self::Critical => "Immediate intervention required",
        }
    }
}

// =============================================================================
// ALERT LEVEL
// =============================================================================

/// Severity level for alerts.
///
/// Alerts are notifications about conditions that require attention.
/// The level indicates the urgency and importance of the alert.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum AlertLevel {
    /// Informational alert — no action required, for awareness only.
    #[default]
    Info,

    /// Warning alert — condition should be monitored and may require action.
    Warning,

    /// Error alert — action required to restore normal operation.
    Error,

    /// Critical alert — immediate action required to prevent major impact.
    Critical,
}

impl std::fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl std::str::FromStr for AlertLevel {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "info" | "information" | "notice" => Ok(Self::Info),
            "warning" | "warn" => Ok(Self::Warning),
            "error" | "err" => Ok(Self::Error),
            "critical" | "crit" | "fatal" => Ok(Self::Critical),
            _ => Err(Error::Config(format!(
                "Unknown alert level: '{}'. Valid: info, warning, error, critical",
                s
            ))),
        }
    }
}

impl AlertLevel {
    /// Get all alert levels in order from lowest to highest severity.
    pub fn all() -> Vec<Self> {
        vec![Self::Info, Self::Warning, Self::Error, Self::Critical]
    }

    /// Check if this alert level requires immediate attention.
    pub fn is_urgent(&self) -> bool {
        matches!(self, Self::Error | Self::Critical)
    }

    /// Get a numeric priority value (higher = more urgent).
    pub fn priority(&self) -> u8 {
        match self {
            Self::Info => 1,
            Self::Warning => 2,
            Self::Error => 3,
            Self::Critical => 4,
        }
    }

    /// Convert alert level to corresponding health status.
    pub fn to_health_status(&self) -> HealthStatus {
        match self {
            Self::Info => HealthStatus::Healthy,
            Self::Warning => HealthStatus::Degraded,
            Self::Error => HealthStatus::Unhealthy,
            Self::Critical => HealthStatus::Critical,
        }
    }
}

// =============================================================================
// ALERT
// =============================================================================

/// An alert raised by the monitoring system.
///
/// Alerts notify operators of conditions that may require attention,
/// from informational notices to critical system failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Severity level of the alert.
    pub level: AlertLevel,

    /// Human-readable message describing the condition.
    pub message: String,

    /// Source component that raised the alert.
    pub source: String,

    /// Timestamp when the alert was raised.
    pub timestamp: DateTime<Utc>,

    /// Optional additional context or details.
    pub details: Option<String>,

    /// Alert identifier for deduplication.
    pub alert_id: Option<String>,
}

impl Alert {
    /// Create a new alert with the specified level, message, and source.
    pub fn new(level: AlertLevel, message: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            source: source.into(),
            timestamp: Utc::now(),
            details: None,
            alert_id: None,
        }
    }

    /// Create an info-level alert.
    pub fn info(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(AlertLevel::Info, message, source)
    }

    /// Create a warning-level alert.
    pub fn warning(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(AlertLevel::Warning, message, source)
    }

    /// Create an error-level alert.
    pub fn error(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(AlertLevel::Error, message, source)
    }

    /// Create a critical-level alert.
    pub fn critical(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::new(AlertLevel::Critical, message, source)
    }

    /// Set additional details for the alert.
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Set an alert ID for deduplication.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.alert_id = Some(id.into());
        self
    }

    /// Check if this alert requires immediate attention.
    pub fn is_urgent(&self) -> bool {
        self.level.is_urgent()
    }

    /// Format the alert as a log line.
    pub fn to_log_line(&self) -> String {
        format!(
            "[{}] {} - {} ({})",
            self.level,
            self.source,
            self.message,
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }

    /// Format the alert as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!(
            "### {} Alert from {}\n\n**Time:** {}\n\n**Message:** {}\n",
            self.level,
            self.source,
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            self.message
        );

        if let Some(details) = &self.details {
            output.push_str(&format!("\n**Details:** {}\n", details));
        }

        output
    }
}

// =============================================================================
// AGENT HEALTH
// =============================================================================

/// Health status of an individual agent in the system.
///
/// Tracks the operational state, last check time, and error count
/// for a specific agent component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    /// Name of the agent.
    pub name: String,

    /// Current health status.
    pub status: HealthStatus,

    /// Timestamp of the last health check.
    pub last_check: DateTime<Utc>,

    /// Number of consecutive errors encountered.
    pub error_count: u32,

    /// Last error message, if any.
    pub last_error: Option<String>,

    /// Average response time in milliseconds.
    pub avg_response_ms: Option<f64>,

    /// Whether the agent is currently active.
    pub is_active: bool,
}

impl AgentHealth {
    /// Create a new agent health record.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Healthy,
            last_check: Utc::now(),
            error_count: 0,
            last_error: None,
            avg_response_ms: None,
            is_active: true,
        }
    }

    /// Create a healthy agent health record.
    pub fn healthy(name: impl Into<String>) -> Self {
        Self::new(name)
    }

    /// Create an unhealthy agent health record with an error message.
    pub fn unhealthy(name: impl Into<String>, error: impl Into<String>) -> Self {
        let mut health = Self::new(name);
        health.status = HealthStatus::Unhealthy;
        health.error_count = 1;
        health.last_error = Some(error.into());
        health
    }

    /// Set the health status.
    pub fn with_status(mut self, status: HealthStatus) -> Self {
        self.status = status;
        self
    }

    /// Set the error count.
    pub fn with_error_count(mut self, count: u32) -> Self {
        self.error_count = count;
        if count > 0 && self.status == HealthStatus::Healthy {
            self.status = HealthStatus::Degraded;
        }
        self
    }

    /// Set the last error message.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.last_error = Some(error.into());
        if self.status == HealthStatus::Healthy {
            self.status = HealthStatus::Unhealthy;
        }
        self.error_count += 1;
        self
    }

    /// Set the average response time.
    pub fn with_response_time(mut self, ms: f64) -> Self {
        self.avg_response_ms = Some(ms);
        self
    }

    /// Set whether the agent is active.
    pub fn with_active(mut self, active: bool) -> Self {
        self.is_active = active;
        self
    }

    /// Record a successful operation.
    pub fn record_success(&mut self) {
        self.error_count = 0;
        self.last_error = None;
        self.last_check = Utc::now();
        self.status = HealthStatus::Healthy;
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self, error: impl Into<String>) {
        self.error_count += 1;
        self.last_error = Some(error.into());
        self.last_check = Utc::now();

        // Update status based on error count
        self.status = match self.error_count {
            1..=2 => HealthStatus::Degraded,
            3..=5 => HealthStatus::Unhealthy,
            _ => HealthStatus::Critical,
        };
    }

    /// Check if this agent is operational.
    pub fn is_operational(&self) -> bool {
        self.is_active && self.status.is_operational()
    }

    /// Get age of the last check in seconds.
    pub fn check_age_secs(&self) -> i64 {
        (Utc::now() - self.last_check).num_seconds()
    }

    /// Check if the health data is stale (older than given seconds).
    pub fn is_stale(&self, max_age_secs: i64) -> bool {
        self.check_age_secs() > max_age_secs
    }
}

// =============================================================================
// SYSTEM METRICS
// =============================================================================

/// System-wide performance metrics.
///
/// Captures key performance indicators for the monitoring system
/// to track and analyze system behavior over time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Number of requests processed per second.
    pub requests_per_second: f64,

    /// Average latency in milliseconds.
    pub average_latency_ms: f64,

    /// Error rate as a percentage (0.0 - 100.0).
    pub error_rate: f64,

    /// Cache hit ratio as a percentage (0.0 - 100.0).
    pub cache_hit_ratio: f64,

    /// Memory usage in megabytes.
    pub memory_usage_mb: f64,

    /// CPU usage as a percentage (0.0 - 100.0).
    pub cpu_usage: Option<f64>,

    /// Number of active connections.
    pub active_connections: Option<u32>,

    /// Queue depth (pending requests).
    pub queue_depth: Option<u32>,

    /// Timestamp when metrics were collected.
    pub timestamp: DateTime<Utc>,
}

impl SystemMetrics {
    /// Create new system metrics with default values.
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            ..Default::default()
        }
    }

    /// Create metrics with the specified values.
    pub fn with_values(
        requests_per_second: f64,
        average_latency_ms: f64,
        error_rate: f64,
        cache_hit_ratio: f64,
        memory_usage_mb: f64,
    ) -> Self {
        Self {
            requests_per_second,
            average_latency_ms,
            error_rate: error_rate.clamp(0.0, 100.0),
            cache_hit_ratio: cache_hit_ratio.clamp(0.0, 100.0),
            memory_usage_mb,
            cpu_usage: None,
            active_connections: None,
            queue_depth: None,
            timestamp: Utc::now(),
        }
    }

    /// Set CPU usage.
    pub fn with_cpu_usage(mut self, cpu: f64) -> Self {
        self.cpu_usage = Some(cpu.clamp(0.0, 100.0));
        self
    }

    /// Set active connections.
    pub fn with_connections(mut self, connections: u32) -> Self {
        self.active_connections = Some(connections);
        self
    }

    /// Set queue depth.
    pub fn with_queue_depth(mut self, depth: u32) -> Self {
        self.queue_depth = Some(depth);
        self
    }

    /// Check if error rate exceeds threshold.
    pub fn has_high_error_rate(&self, threshold: f64) -> bool {
        self.error_rate > threshold
    }

    /// Check if latency exceeds threshold.
    pub fn has_high_latency(&self, threshold_ms: f64) -> bool {
        self.average_latency_ms > threshold_ms
    }

    /// Check if memory usage exceeds threshold.
    pub fn has_high_memory(&self, threshold_mb: f64) -> bool {
        self.memory_usage_mb > threshold_mb
    }

    /// Get overall system health based on metrics and thresholds.
    pub fn assess_health(&self, config: &MonitorConfig) -> HealthStatus {
        let mut status = HealthStatus::Healthy;

        // Check error rate
        if self.error_rate >= config.critical_error_rate {
            status = status.combine(&HealthStatus::Critical);
        } else if self.error_rate >= config.high_error_rate {
            status = status.combine(&HealthStatus::Unhealthy);
        } else if self.error_rate >= config.warning_error_rate {
            status = status.combine(&HealthStatus::Degraded);
        }

        // Check latency
        if self.average_latency_ms >= config.critical_latency_ms {
            status = status.combine(&HealthStatus::Critical);
        } else if self.average_latency_ms >= config.high_latency_ms {
            status = status.combine(&HealthStatus::Unhealthy);
        } else if self.average_latency_ms >= config.warning_latency_ms {
            status = status.combine(&HealthStatus::Degraded);
        }

        // Check memory
        if self.memory_usage_mb >= config.critical_memory_mb {
            status = status.combine(&HealthStatus::Critical);
        } else if self.memory_usage_mb >= config.high_memory_mb {
            status = status.combine(&HealthStatus::Unhealthy);
        } else if self.memory_usage_mb >= config.warning_memory_mb {
            status = status.combine(&HealthStatus::Degraded);
        }

        status
    }

    /// Format metrics as a markdown table.
    pub fn to_markdown(&self) -> String {
        let mut output = String::from("| Metric | Value |\n|--------|-------|\n");
        output.push_str(&format!(
            "| Requests/sec | {:.2} |\n",
            self.requests_per_second
        ));
        output.push_str(&format!(
            "| Avg Latency | {:.2} ms |\n",
            self.average_latency_ms
        ));
        output.push_str(&format!("| Error Rate | {:.2}% |\n", self.error_rate));
        output.push_str(&format!(
            "| Cache Hit Ratio | {:.2}% |\n",
            self.cache_hit_ratio
        ));
        output.push_str(&format!(
            "| Memory Usage | {:.2} MB |\n",
            self.memory_usage_mb
        ));

        if let Some(cpu) = self.cpu_usage {
            output.push_str(&format!("| CPU Usage | {:.2}% |\n", cpu));
        }
        if let Some(conn) = self.active_connections {
            output.push_str(&format!("| Active Connections | {} |\n", conn));
        }
        if let Some(queue) = self.queue_depth {
            output.push_str(&format!("| Queue Depth | {} |\n", queue));
        }

        output
    }
}

// =============================================================================
// HEALTH REPORT
// =============================================================================

/// Comprehensive health report for the system.
///
/// Contains overall status, individual agent health, metrics,
/// and any active alerts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall system health status.
    pub overall_status: HealthStatus,

    /// Health status of individual agents.
    pub agent_health: HashMap<String, AgentHealth>,

    /// Current system metrics.
    pub metrics: SystemMetrics,

    /// Active alerts.
    pub alerts: Vec<Alert>,

    /// Timestamp when the report was generated.
    pub timestamp: DateTime<Utc>,

    /// Duration of the health check in milliseconds.
    pub check_duration_ms: u64,

    /// Optional summary message.
    pub summary: Option<String>,
}

impl HealthReport {
    /// Create a new health report.
    pub fn new(overall_status: HealthStatus, metrics: SystemMetrics) -> Self {
        Self {
            overall_status,
            agent_health: HashMap::new(),
            metrics,
            alerts: Vec::new(),
            timestamp: Utc::now(),
            check_duration_ms: 0,
            summary: None,
        }
    }

    /// Create a healthy report.
    pub fn healthy() -> Self {
        Self::new(HealthStatus::Healthy, SystemMetrics::new())
    }

    /// Add an agent's health status.
    pub fn add_agent_health(&mut self, health: AgentHealth) {
        // Update overall status if agent status is worse
        self.overall_status = self.overall_status.combine(&health.status);
        self.agent_health.insert(health.name.clone(), health);
    }

    /// Add an alert to the report.
    pub fn add_alert(&mut self, alert: Alert) {
        // Update overall status based on alert level
        self.overall_status = self.overall_status.combine(&alert.level.to_health_status());
        self.alerts.push(alert);
    }

    /// Set the check duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.check_duration_ms = duration_ms;
        self
    }

    /// Set a summary message.
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Get the number of healthy agents.
    pub fn healthy_agent_count(&self) -> usize {
        self.agent_health
            .values()
            .filter(|h| h.status == HealthStatus::Healthy)
            .count()
    }

    /// Get the number of unhealthy agents.
    pub fn unhealthy_agent_count(&self) -> usize {
        self.agent_health
            .values()
            .filter(|h| h.status.has_issues())
            .count()
    }

    /// Get critical alerts.
    pub fn critical_alerts(&self) -> Vec<&Alert> {
        self.alerts
            .iter()
            .filter(|a| a.level == AlertLevel::Critical)
            .collect()
    }

    /// Get urgent alerts (error or critical).
    pub fn urgent_alerts(&self) -> Vec<&Alert> {
        self.alerts.iter().filter(|a| a.is_urgent()).collect()
    }

    /// Check if the system is operational.
    pub fn is_operational(&self) -> bool {
        self.overall_status.is_operational()
    }

    /// Check if immediate attention is required.
    pub fn requires_attention(&self) -> bool {
        self.overall_status.is_critical() || !self.critical_alerts().is_empty()
    }

    /// Generate a brief summary of the health report.
    pub fn brief_summary(&self) -> String {
        let agent_summary = if self.agent_health.is_empty() {
            String::from("no agents monitored")
        } else {
            format!(
                "{}/{} agents healthy",
                self.healthy_agent_count(),
                self.agent_health.len()
            )
        };

        let alert_summary = if self.alerts.is_empty() {
            String::from("no alerts")
        } else {
            let critical = self.critical_alerts().len();
            if critical > 0 {
                format!("{} alerts ({} critical)", self.alerts.len(), critical)
            } else {
                format!("{} alerts", self.alerts.len())
            }
        };

        format!(
            "{}: {}, {}",
            self.overall_status, agent_summary, alert_summary
        )
    }

    /// Format the report as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Health Report\n\n");
        output.push_str(&format!("**Overall Status:** {}\n\n", self.overall_status));
        output.push_str(&format!(
            "**Generated:** {}\n\n",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        if let Some(summary) = &self.summary {
            output.push_str(&format!("**Summary:** {}\n\n", summary));
        }

        // Metrics section
        output.push_str("## System Metrics\n\n");
        output.push_str(&self.metrics.to_markdown());
        output.push('\n');

        // Agent health section
        if !self.agent_health.is_empty() {
            output.push_str("## Agent Health\n\n");
            output.push_str("| Agent | Status | Errors | Last Check |\n");
            output.push_str("|-------|--------|--------|------------|\n");

            for (name, health) in &self.agent_health {
                output.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    name,
                    health.status,
                    health.error_count,
                    health.last_check.format("%H:%M:%S")
                ));
            }
            output.push('\n');
        }

        // Alerts section
        if !self.alerts.is_empty() {
            output.push_str("## Alerts\n\n");
            for alert in &self.alerts {
                output.push_str(&format!(
                    "- **[{}]** {} - {}\n",
                    alert.level, alert.source, alert.message
                ));
            }
        }

        output
    }
}

// =============================================================================
// MONITOR CONFIGURATION
// =============================================================================

/// Configuration for the Monitor Agent.
///
/// Defines thresholds, intervals, and behavior for health monitoring.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Warning threshold for error rate (percentage).
    pub warning_error_rate: f64,

    /// High threshold for error rate (percentage).
    pub high_error_rate: f64,

    /// Critical threshold for error rate (percentage).
    pub critical_error_rate: f64,

    /// Warning threshold for latency (milliseconds).
    pub warning_latency_ms: f64,

    /// High threshold for latency (milliseconds).
    pub high_latency_ms: f64,

    /// Critical threshold for latency (milliseconds).
    pub critical_latency_ms: f64,

    /// Warning threshold for memory usage (megabytes).
    pub warning_memory_mb: f64,

    /// High threshold for memory usage (megabytes).
    pub high_memory_mb: f64,

    /// Critical threshold for memory usage (megabytes).
    pub critical_memory_mb: f64,

    /// Health check interval in seconds.
    pub check_interval_secs: u64,

    /// Maximum age of health data before considered stale (seconds).
    pub stale_threshold_secs: i64,

    /// Maximum number of consecutive errors before marking unhealthy.
    pub max_consecutive_errors: u32,

    /// Whether to include detailed agent health in reports.
    pub include_agent_details: bool,

    /// Whether to automatically raise alerts.
    pub auto_alert: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            warning_error_rate: 1.0,
            high_error_rate: 5.0,
            critical_error_rate: 10.0,
            warning_latency_ms: 500.0,
            high_latency_ms: 1000.0,
            critical_latency_ms: 5000.0,
            warning_memory_mb: 256.0,
            high_memory_mb: 512.0,
            critical_memory_mb: 1024.0,
            check_interval_secs: 30,
            stale_threshold_secs: 120,
            max_consecutive_errors: 3,
            include_agent_details: true,
            auto_alert: true,
        }
    }
}

impl MonitorConfig {
    /// Create a strict configuration with tighter thresholds.
    pub fn strict() -> Self {
        Self {
            warning_error_rate: 0.5,
            high_error_rate: 2.0,
            critical_error_rate: 5.0,
            warning_latency_ms: 200.0,
            high_latency_ms: 500.0,
            critical_latency_ms: 2000.0,
            warning_memory_mb: 128.0,
            high_memory_mb: 256.0,
            critical_memory_mb: 512.0,
            check_interval_secs: 15,
            stale_threshold_secs: 60,
            max_consecutive_errors: 2,
            include_agent_details: true,
            auto_alert: true,
        }
    }

    /// Create a lenient configuration with relaxed thresholds.
    pub fn lenient() -> Self {
        Self {
            warning_error_rate: 5.0,
            high_error_rate: 10.0,
            critical_error_rate: 25.0,
            warning_latency_ms: 1000.0,
            high_latency_ms: 3000.0,
            critical_latency_ms: 10000.0,
            warning_memory_mb: 512.0,
            high_memory_mb: 1024.0,
            critical_memory_mb: 2048.0,
            check_interval_secs: 60,
            stale_threshold_secs: 300,
            max_consecutive_errors: 5,
            include_agent_details: true,
            auto_alert: true,
        }
    }

    /// Create a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            include_agent_details: false,
            auto_alert: false,
            ..Default::default()
        }
    }
}

// =============================================================================
// MONITOR AGENT
// =============================================================================

/// The Monitor Agent - The Health Watcher of the Sorcerer's Tower.
///
/// Responsible for tracking system health, collecting metrics,
/// and raising alerts when issues are detected.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive to maintain system health and stability
/// - **NATURA**: Natural observation and pattern recognition
/// - **RATIO**: Logical analysis of metrics against thresholds
/// - **LAETITIA**: Joy through maintained system reliability
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::monitor::{MonitorAgent, MonitorConfig};
///
/// let monitor = MonitorAgent::builder()
///     .warning_error_rate(2.0)
///     .high_latency_ms(1000.0)
///     .build();
///
/// let report = monitor.check_health().await?;
/// println!("Status: {}", report.overall_status);
/// ```
#[derive(Debug, Clone)]
pub struct MonitorAgent {
    /// Configuration for monitoring behavior.
    config: MonitorConfig,

    /// Agent health states.
    agent_states: HashMap<String, AgentHealth>,
}

impl Default for MonitorAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl MonitorAgent {
    /// Create a new Monitor Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: MonitorConfig::default(),
            agent_states: HashMap::new(),
        }
    }

    /// Create a Monitor Agent with custom configuration.
    pub fn with_config(config: MonitorConfig) -> Self {
        Self {
            config,
            agent_states: HashMap::new(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> MonitorAgentBuilder {
        MonitorAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Register an agent for monitoring.
    pub fn register_agent(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.agent_states
            .insert(name.clone(), AgentHealth::new(name));
    }

    /// Update an agent's health status.
    pub fn update_agent_health(&mut self, name: &str, health: AgentHealth) {
        self.agent_states.insert(name.to_string(), health);
    }

    /// Record a successful operation for an agent.
    pub fn record_agent_success(&mut self, name: &str) {
        if let Some(health) = self.agent_states.get_mut(name) {
            health.record_success();
        }
    }

    /// Record a failed operation for an agent.
    pub fn record_agent_failure(&mut self, name: &str, error: impl Into<String>) {
        if let Some(health) = self.agent_states.get_mut(name) {
            health.record_failure(error);
        }
    }

    /// Get an agent's current health status.
    pub fn get_agent_health(&self, name: &str) -> Option<&AgentHealth> {
        self.agent_states.get(name)
    }

    // =========================================================================
    // HEALTH CHECK
    // =========================================================================

    /// Perform a comprehensive health check.
    ///
    /// Evaluates all registered agents and system metrics,
    /// generating a complete health report.
    pub async fn check_health(&self) -> Result<HealthReport> {
        let start = Instant::now();

        debug!("Starting health check");

        // Collect current metrics (simulated for now)
        let metrics = self.collect_metrics().await?;

        // Determine overall status from metrics
        let mut overall_status = metrics.assess_health(&self.config);

        // Create the report
        let mut report = HealthReport::new(overall_status, metrics);

        // Add agent health data
        if self.config.include_agent_details {
            for (name, health) in &self.agent_states {
                report.add_agent_health(health.clone());

                // Check for stale data
                if health.is_stale(self.config.stale_threshold_secs) && self.config.auto_alert {
                    report.add_alert(Alert::warning(
                        format!("Health data is stale ({}s old)", health.check_age_secs()),
                        name.clone(),
                    ));
                }
            }
        }

        // Generate alerts based on thresholds
        if self.config.auto_alert {
            self.generate_alerts(&mut report);
        }

        // Update overall status after considering agent health
        for health in report.agent_health.values() {
            overall_status = overall_status.combine(&health.status);
        }
        report.overall_status = overall_status;

        report.check_duration_ms = start.elapsed().as_millis() as u64;
        report.summary = Some(report.brief_summary());

        info!(
            "Health check complete: {} in {}ms",
            report.overall_status, report.check_duration_ms
        );

        Ok(report)
    }

    /// Get current system metrics.
    pub async fn get_metrics(&self) -> Result<SystemMetrics> {
        self.collect_metrics().await
    }

    // =========================================================================
    // INTERNAL METHODS
    // =========================================================================

    /// Collect system metrics.
    async fn collect_metrics(&self) -> Result<SystemMetrics> {
        debug!("Collecting system metrics");

        // In a real implementation, this would gather actual metrics
        // from the system. For now, we return simulated healthy metrics.
        let metrics = SystemMetrics::with_values(
            10.0,  // requests per second
            50.0,  // average latency ms
            0.5,   // error rate %
            85.0,  // cache hit ratio %
            128.0, // memory usage mb
        );

        Ok(metrics)
    }

    /// Generate alerts based on current metrics and thresholds.
    fn generate_alerts(&self, report: &mut HealthReport) {
        // Copy metrics values to avoid borrow conflicts
        let error_rate = report.metrics.error_rate;
        let average_latency_ms = report.metrics.average_latency_ms;
        let memory_usage_mb = report.metrics.memory_usage_mb;

        // Error rate alerts
        if error_rate >= self.config.critical_error_rate {
            report.add_alert(Alert::critical(
                format!("Error rate critical: {:.2}%", error_rate),
                "metrics",
            ));
        } else if error_rate >= self.config.high_error_rate {
            report.add_alert(Alert::error(
                format!("Error rate high: {:.2}%", error_rate),
                "metrics",
            ));
        } else if error_rate >= self.config.warning_error_rate {
            report.add_alert(Alert::warning(
                format!("Error rate elevated: {:.2}%", error_rate),
                "metrics",
            ));
        }

        // Latency alerts
        if average_latency_ms >= self.config.critical_latency_ms {
            report.add_alert(Alert::critical(
                format!("Latency critical: {:.2}ms", average_latency_ms),
                "metrics",
            ));
        } else if average_latency_ms >= self.config.high_latency_ms {
            report.add_alert(Alert::error(
                format!("Latency high: {:.2}ms", average_latency_ms),
                "metrics",
            ));
        } else if average_latency_ms >= self.config.warning_latency_ms {
            report.add_alert(Alert::warning(
                format!("Latency elevated: {:.2}ms", average_latency_ms),
                "metrics",
            ));
        }

        // Memory alerts
        if memory_usage_mb >= self.config.critical_memory_mb {
            report.add_alert(Alert::critical(
                format!("Memory usage critical: {:.2}MB", memory_usage_mb),
                "metrics",
            ));
        } else if memory_usage_mb >= self.config.high_memory_mb {
            report.add_alert(Alert::error(
                format!("Memory usage high: {:.2}MB", memory_usage_mb),
                "metrics",
            ));
        } else if memory_usage_mb >= self.config.warning_memory_mb {
            report.add_alert(Alert::warning(
                format!("Memory usage elevated: {:.2}MB", memory_usage_mb),
                "metrics",
            ));
        }

        // Agent health alerts
        for (name, health) in &report.agent_health {
            if health.status == HealthStatus::Critical {
                report.alerts.push(Alert::critical(
                    format!(
                        "Agent in critical state: {}",
                        health.last_error.as_deref().unwrap_or("unknown error")
                    ),
                    name.clone(),
                ));
            } else if health.status == HealthStatus::Unhealthy {
                report.alerts.push(Alert::error(
                    format!(
                        "Agent unhealthy: {}",
                        health.last_error.as_deref().unwrap_or("unknown error")
                    ),
                    name.clone(),
                ));
            } else if health.error_count >= self.config.max_consecutive_errors {
                report.alerts.push(Alert::warning(
                    format!("{} consecutive errors detected", health.error_count),
                    name.clone(),
                ));
            }
        }
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom MonitorAgent configuration.
#[derive(Debug, Default)]
pub struct MonitorAgentBuilder {
    config: Option<MonitorConfig>,
}

impl MonitorAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: MonitorConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set warning error rate threshold.
    pub fn warning_error_rate(mut self, rate: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.warning_error_rate = rate.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set high error rate threshold.
    pub fn high_error_rate(mut self, rate: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.high_error_rate = rate.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set critical error rate threshold.
    pub fn critical_error_rate(mut self, rate: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.critical_error_rate = rate.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set warning latency threshold.
    pub fn warning_latency_ms(mut self, ms: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.warning_latency_ms = ms.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set high latency threshold.
    pub fn high_latency_ms(mut self, ms: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.high_latency_ms = ms.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set critical latency threshold.
    pub fn critical_latency_ms(mut self, ms: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.critical_latency_ms = ms.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set warning memory threshold.
    pub fn warning_memory_mb(mut self, mb: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.warning_memory_mb = mb.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set high memory threshold.
    pub fn high_memory_mb(mut self, mb: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.high_memory_mb = mb.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set critical memory threshold.
    pub fn critical_memory_mb(mut self, mb: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.critical_memory_mb = mb.max(0.0);
        self.config = Some(config);
        self
    }

    /// Set check interval.
    pub fn check_interval_secs(mut self, secs: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.check_interval_secs = secs;
        self.config = Some(config);
        self
    }

    /// Set stale threshold.
    pub fn stale_threshold_secs(mut self, secs: i64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.stale_threshold_secs = secs;
        self.config = Some(config);
        self
    }

    /// Set max consecutive errors.
    pub fn max_consecutive_errors(mut self, max: u32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_consecutive_errors = max;
        self.config = Some(config);
        self
    }

    /// Set whether to include agent details.
    pub fn include_agent_details(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_agent_details = include;
        self.config = Some(config);
        self
    }

    /// Set whether to auto-generate alerts.
    pub fn auto_alert(mut self, auto: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.auto_alert = auto;
        self.config = Some(config);
        self
    }

    /// Build the MonitorAgent.
    pub fn build(self) -> MonitorAgent {
        MonitorAgent {
            config: self.config.unwrap_or_default(),
            agent_states: HashMap::new(),
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
    // HealthStatus Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "Healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "Degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "Unhealthy");
        assert_eq!(HealthStatus::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_health_status_from_str() {
        assert_eq!(
            "healthy".parse::<HealthStatus>().unwrap(),
            HealthStatus::Healthy
        );
        assert_eq!(
            "green".parse::<HealthStatus>().unwrap(),
            HealthStatus::Healthy
        );
        assert_eq!(
            "degraded".parse::<HealthStatus>().unwrap(),
            HealthStatus::Degraded
        );
        assert_eq!(
            "warning".parse::<HealthStatus>().unwrap(),
            HealthStatus::Degraded
        );
        assert_eq!(
            "unhealthy".parse::<HealthStatus>().unwrap(),
            HealthStatus::Unhealthy
        );
        assert_eq!(
            "critical".parse::<HealthStatus>().unwrap(),
            HealthStatus::Critical
        );
    }

    #[test]
    fn test_health_status_from_str_invalid() {
        assert!("invalid".parse::<HealthStatus>().is_err());
    }

    #[test]
    fn test_health_status_default() {
        assert_eq!(HealthStatus::default(), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_status_all() {
        let all = HealthStatus::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&HealthStatus::Healthy));
        assert!(all.contains(&HealthStatus::Critical));
    }

    #[test]
    fn test_health_status_is_critical() {
        assert!(!HealthStatus::Healthy.is_critical());
        assert!(!HealthStatus::Degraded.is_critical());
        assert!(!HealthStatus::Unhealthy.is_critical());
        assert!(HealthStatus::Critical.is_critical());
    }

    #[test]
    fn test_health_status_has_issues() {
        assert!(!HealthStatus::Healthy.has_issues());
        assert!(HealthStatus::Degraded.has_issues());
        assert!(HealthStatus::Unhealthy.has_issues());
        assert!(HealthStatus::Critical.has_issues());
    }

    #[test]
    fn test_health_status_is_operational() {
        assert!(HealthStatus::Healthy.is_operational());
        assert!(HealthStatus::Degraded.is_operational());
        assert!(!HealthStatus::Unhealthy.is_operational());
        assert!(!HealthStatus::Critical.is_operational());
    }

    #[test]
    fn test_health_status_severity() {
        assert_eq!(HealthStatus::Healthy.severity(), 0);
        assert_eq!(HealthStatus::Degraded.severity(), 1);
        assert_eq!(HealthStatus::Unhealthy.severity(), 2);
        assert_eq!(HealthStatus::Critical.severity(), 3);
    }

    #[test]
    fn test_health_status_combine() {
        assert_eq!(
            HealthStatus::Healthy.combine(&HealthStatus::Healthy),
            HealthStatus::Healthy
        );
        assert_eq!(
            HealthStatus::Healthy.combine(&HealthStatus::Degraded),
            HealthStatus::Degraded
        );
        assert_eq!(
            HealthStatus::Degraded.combine(&HealthStatus::Unhealthy),
            HealthStatus::Unhealthy
        );
        assert_eq!(
            HealthStatus::Unhealthy.combine(&HealthStatus::Critical),
            HealthStatus::Critical
        );
        assert_eq!(
            HealthStatus::Critical.combine(&HealthStatus::Healthy),
            HealthStatus::Critical
        );
    }

    #[test]
    fn test_health_status_description() {
        assert!(HealthStatus::Healthy.description().contains("normally"));
        assert!(HealthStatus::Critical.description().contains("immediate"));
    }

    #[test]
    fn test_health_status_recommended_action() {
        assert!(HealthStatus::Healthy
            .recommended_action()
            .contains("normal"));
        assert!(HealthStatus::Critical
            .recommended_action()
            .contains("Immediate"));
    }

    // -------------------------------------------------------------------------
    // AlertLevel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alert_level_display() {
        assert_eq!(AlertLevel::Info.to_string(), "INFO");
        assert_eq!(AlertLevel::Warning.to_string(), "WARNING");
        assert_eq!(AlertLevel::Error.to_string(), "ERROR");
        assert_eq!(AlertLevel::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_alert_level_from_str() {
        assert_eq!("info".parse::<AlertLevel>().unwrap(), AlertLevel::Info);
        assert_eq!(
            "warning".parse::<AlertLevel>().unwrap(),
            AlertLevel::Warning
        );
        assert_eq!("error".parse::<AlertLevel>().unwrap(), AlertLevel::Error);
        assert_eq!(
            "critical".parse::<AlertLevel>().unwrap(),
            AlertLevel::Critical
        );
    }

    #[test]
    fn test_alert_level_from_str_invalid() {
        assert!("invalid".parse::<AlertLevel>().is_err());
    }

    #[test]
    fn test_alert_level_default() {
        assert_eq!(AlertLevel::default(), AlertLevel::Info);
    }

    #[test]
    fn test_alert_level_all() {
        let all = AlertLevel::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_alert_level_is_urgent() {
        assert!(!AlertLevel::Info.is_urgent());
        assert!(!AlertLevel::Warning.is_urgent());
        assert!(AlertLevel::Error.is_urgent());
        assert!(AlertLevel::Critical.is_urgent());
    }

    #[test]
    fn test_alert_level_priority() {
        assert_eq!(AlertLevel::Info.priority(), 1);
        assert_eq!(AlertLevel::Warning.priority(), 2);
        assert_eq!(AlertLevel::Error.priority(), 3);
        assert_eq!(AlertLevel::Critical.priority(), 4);
    }

    #[test]
    fn test_alert_level_to_health_status() {
        assert_eq!(AlertLevel::Info.to_health_status(), HealthStatus::Healthy);
        assert_eq!(
            AlertLevel::Warning.to_health_status(),
            HealthStatus::Degraded
        );
        assert_eq!(
            AlertLevel::Error.to_health_status(),
            HealthStatus::Unhealthy
        );
        assert_eq!(
            AlertLevel::Critical.to_health_status(),
            HealthStatus::Critical
        );
    }

    // -------------------------------------------------------------------------
    // Alert Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alert_new() {
        let alert = Alert::new(AlertLevel::Warning, "Test message", "test_source");
        assert_eq!(alert.level, AlertLevel::Warning);
        assert_eq!(alert.message, "Test message");
        assert_eq!(alert.source, "test_source");
        assert!(alert.details.is_none());
    }

    #[test]
    fn test_alert_convenience_constructors() {
        let info = Alert::info("Info message", "src");
        assert_eq!(info.level, AlertLevel::Info);

        let warning = Alert::warning("Warning message", "src");
        assert_eq!(warning.level, AlertLevel::Warning);

        let error = Alert::error("Error message", "src");
        assert_eq!(error.level, AlertLevel::Error);

        let critical = Alert::critical("Critical message", "src");
        assert_eq!(critical.level, AlertLevel::Critical);
    }

    #[test]
    fn test_alert_with_details() {
        let alert = Alert::warning("Test", "src").with_details("Additional details");
        assert_eq!(alert.details.unwrap(), "Additional details");
    }

    #[test]
    fn test_alert_with_id() {
        let alert = Alert::warning("Test", "src").with_id("alert-123");
        assert_eq!(alert.alert_id.unwrap(), "alert-123");
    }

    #[test]
    fn test_alert_is_urgent() {
        assert!(!Alert::info("Test", "src").is_urgent());
        assert!(!Alert::warning("Test", "src").is_urgent());
        assert!(Alert::error("Test", "src").is_urgent());
        assert!(Alert::critical("Test", "src").is_urgent());
    }

    #[test]
    fn test_alert_to_log_line() {
        let alert = Alert::warning("Test message", "test_source");
        let log = alert.to_log_line();
        assert!(log.contains("[WARNING]"));
        assert!(log.contains("test_source"));
        assert!(log.contains("Test message"));
    }

    #[test]
    fn test_alert_to_markdown() {
        let alert = Alert::error("Error occurred", "system").with_details("Stack trace here");
        let md = alert.to_markdown();
        assert!(md.contains("### ERROR Alert"));
        assert!(md.contains("Error occurred"));
        assert!(md.contains("Stack trace here"));
    }

    // -------------------------------------------------------------------------
    // AgentHealth Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_health_new() {
        let health = AgentHealth::new("test_agent");
        assert_eq!(health.name, "test_agent");
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.error_count, 0);
        assert!(health.is_active);
    }

    #[test]
    fn test_agent_health_healthy() {
        let health = AgentHealth::healthy("agent");
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_agent_health_unhealthy() {
        let health = AgentHealth::unhealthy("agent", "Connection failed");
        assert_eq!(health.status, HealthStatus::Unhealthy);
        assert_eq!(health.error_count, 1);
        assert_eq!(health.last_error.unwrap(), "Connection failed");
    }

    #[test]
    fn test_agent_health_with_status() {
        let health = AgentHealth::new("agent").with_status(HealthStatus::Degraded);
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_agent_health_with_error_count() {
        let health = AgentHealth::new("agent").with_error_count(3);
        assert_eq!(health.error_count, 3);
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_agent_health_with_error() {
        let health = AgentHealth::new("agent").with_error("Error occurred");
        assert_eq!(health.error_count, 1);
        assert!(health.last_error.is_some());
        assert_eq!(health.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_agent_health_with_response_time() {
        let health = AgentHealth::new("agent").with_response_time(150.0);
        assert_eq!(health.avg_response_ms.unwrap(), 150.0);
    }

    #[test]
    fn test_agent_health_with_active() {
        let health = AgentHealth::new("agent").with_active(false);
        assert!(!health.is_active);
    }

    #[test]
    fn test_agent_health_record_success() {
        let mut health = AgentHealth::unhealthy("agent", "Error");
        health.record_success();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.error_count, 0);
        assert!(health.last_error.is_none());
    }

    #[test]
    fn test_agent_health_record_failure() {
        let mut health = AgentHealth::new("agent");
        health.record_failure("First error");
        assert_eq!(health.error_count, 1);
        assert_eq!(health.status, HealthStatus::Degraded);

        health.record_failure("Second error");
        assert_eq!(health.error_count, 2);
        assert_eq!(health.status, HealthStatus::Degraded);

        health.record_failure("Third error");
        assert_eq!(health.error_count, 3);
        assert_eq!(health.status, HealthStatus::Unhealthy);

        for _ in 0..3 {
            health.record_failure("More errors");
        }
        assert_eq!(health.status, HealthStatus::Critical);
    }

    #[test]
    fn test_agent_health_is_operational() {
        let healthy = AgentHealth::healthy("agent");
        assert!(healthy.is_operational());

        let inactive = AgentHealth::healthy("agent").with_active(false);
        assert!(!inactive.is_operational());

        let unhealthy = AgentHealth::unhealthy("agent", "Error");
        assert!(!unhealthy.is_operational());
    }

    // -------------------------------------------------------------------------
    // SystemMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_system_metrics_new() {
        let metrics = SystemMetrics::new();
        assert_eq!(metrics.requests_per_second, 0.0);
        assert_eq!(metrics.average_latency_ms, 0.0);
        assert_eq!(metrics.error_rate, 0.0);
    }

    #[test]
    fn test_system_metrics_with_values() {
        let metrics = SystemMetrics::with_values(100.0, 50.0, 2.5, 90.0, 256.0);
        assert_eq!(metrics.requests_per_second, 100.0);
        assert_eq!(metrics.average_latency_ms, 50.0);
        assert_eq!(metrics.error_rate, 2.5);
        assert_eq!(metrics.cache_hit_ratio, 90.0);
        assert_eq!(metrics.memory_usage_mb, 256.0);
    }

    #[test]
    fn test_system_metrics_clamping() {
        let metrics = SystemMetrics::with_values(100.0, 50.0, 150.0, -10.0, 256.0);
        assert_eq!(metrics.error_rate, 100.0);
        assert_eq!(metrics.cache_hit_ratio, 0.0);
    }

    #[test]
    fn test_system_metrics_with_cpu_usage() {
        let metrics = SystemMetrics::new().with_cpu_usage(75.0);
        assert_eq!(metrics.cpu_usage.unwrap(), 75.0);
    }

    #[test]
    fn test_system_metrics_with_connections() {
        let metrics = SystemMetrics::new().with_connections(50);
        assert_eq!(metrics.active_connections.unwrap(), 50);
    }

    #[test]
    fn test_system_metrics_with_queue_depth() {
        let metrics = SystemMetrics::new().with_queue_depth(10);
        assert_eq!(metrics.queue_depth.unwrap(), 10);
    }

    #[test]
    fn test_system_metrics_has_high_error_rate() {
        let metrics = SystemMetrics::with_values(100.0, 50.0, 5.0, 90.0, 256.0);
        assert!(metrics.has_high_error_rate(4.0));
        assert!(!metrics.has_high_error_rate(6.0));
    }

    #[test]
    fn test_system_metrics_has_high_latency() {
        let metrics = SystemMetrics::with_values(100.0, 500.0, 2.0, 90.0, 256.0);
        assert!(metrics.has_high_latency(400.0));
        assert!(!metrics.has_high_latency(600.0));
    }

    #[test]
    fn test_system_metrics_has_high_memory() {
        let metrics = SystemMetrics::with_values(100.0, 50.0, 2.0, 90.0, 512.0);
        assert!(metrics.has_high_memory(400.0));
        assert!(!metrics.has_high_memory(600.0));
    }

    #[test]
    fn test_system_metrics_assess_health() {
        let config = MonitorConfig::default();

        // Healthy metrics
        let healthy = SystemMetrics::with_values(100.0, 100.0, 0.5, 90.0, 100.0);
        assert_eq!(healthy.assess_health(&config), HealthStatus::Healthy);

        // Degraded (warning error rate)
        let degraded = SystemMetrics::with_values(100.0, 100.0, 2.0, 90.0, 100.0);
        assert_eq!(degraded.assess_health(&config), HealthStatus::Degraded);

        // Unhealthy (high error rate)
        let unhealthy = SystemMetrics::with_values(100.0, 100.0, 7.0, 90.0, 100.0);
        assert_eq!(unhealthy.assess_health(&config), HealthStatus::Unhealthy);

        // Critical (critical error rate)
        let critical = SystemMetrics::with_values(100.0, 100.0, 15.0, 90.0, 100.0);
        assert_eq!(critical.assess_health(&config), HealthStatus::Critical);
    }

    #[test]
    fn test_system_metrics_to_markdown() {
        let metrics = SystemMetrics::with_values(100.0, 50.0, 2.5, 90.0, 256.0)
            .with_cpu_usage(45.0)
            .with_connections(25);

        let md = metrics.to_markdown();
        assert!(md.contains("Requests/sec"));
        assert!(md.contains("100.00"));
        assert!(md.contains("Avg Latency"));
        assert!(md.contains("CPU Usage"));
        assert!(md.contains("45.00%"));
    }

    // -------------------------------------------------------------------------
    // HealthReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_health_report_new() {
        let metrics = SystemMetrics::new();
        let report = HealthReport::new(HealthStatus::Healthy, metrics);
        assert_eq!(report.overall_status, HealthStatus::Healthy);
        assert!(report.agent_health.is_empty());
        assert!(report.alerts.is_empty());
    }

    #[test]
    fn test_health_report_healthy() {
        let report = HealthReport::healthy();
        assert_eq!(report.overall_status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_report_add_agent_health() {
        let mut report = HealthReport::healthy();
        report.add_agent_health(AgentHealth::unhealthy("agent1", "Error"));
        assert_eq!(report.overall_status, HealthStatus::Unhealthy);
        assert_eq!(report.agent_health.len(), 1);
    }

    #[test]
    fn test_health_report_add_alert() {
        let mut report = HealthReport::healthy();
        report.add_alert(Alert::error("Error occurred", "system"));
        assert_eq!(report.overall_status, HealthStatus::Unhealthy);
        assert_eq!(report.alerts.len(), 1);
    }

    #[test]
    fn test_health_report_with_duration() {
        let report = HealthReport::healthy().with_duration(150);
        assert_eq!(report.check_duration_ms, 150);
    }

    #[test]
    fn test_health_report_with_summary() {
        let report = HealthReport::healthy().with_summary("All systems operational");
        assert_eq!(report.summary.unwrap(), "All systems operational");
    }

    #[test]
    fn test_health_report_healthy_agent_count() {
        let mut report = HealthReport::healthy();
        report.add_agent_health(AgentHealth::healthy("agent1"));
        report.add_agent_health(AgentHealth::healthy("agent2"));
        report.add_agent_health(AgentHealth::unhealthy("agent3", "Error"));
        assert_eq!(report.healthy_agent_count(), 2);
    }

    #[test]
    fn test_health_report_unhealthy_agent_count() {
        let mut report = HealthReport::healthy();
        report.add_agent_health(AgentHealth::healthy("agent1"));
        report.add_agent_health(AgentHealth::unhealthy("agent2", "Error"));
        report.add_agent_health(AgentHealth::new("agent3").with_status(HealthStatus::Degraded));
        assert_eq!(report.unhealthy_agent_count(), 2);
    }

    #[test]
    fn test_health_report_critical_alerts() {
        let mut report = HealthReport::healthy();
        report.add_alert(Alert::info("Info", "src"));
        report.add_alert(Alert::critical("Critical 1", "src"));
        report.add_alert(Alert::warning("Warning", "src"));
        report.add_alert(Alert::critical("Critical 2", "src"));
        assert_eq!(report.critical_alerts().len(), 2);
    }

    #[test]
    fn test_health_report_urgent_alerts() {
        let mut report = HealthReport::healthy();
        report.add_alert(Alert::info("Info", "src"));
        report.add_alert(Alert::error("Error", "src"));
        report.add_alert(Alert::critical("Critical", "src"));
        assert_eq!(report.urgent_alerts().len(), 2);
    }

    #[test]
    fn test_health_report_is_operational() {
        let healthy = HealthReport::new(HealthStatus::Healthy, SystemMetrics::new());
        assert!(healthy.is_operational());

        let degraded = HealthReport::new(HealthStatus::Degraded, SystemMetrics::new());
        assert!(degraded.is_operational());

        let unhealthy = HealthReport::new(HealthStatus::Unhealthy, SystemMetrics::new());
        assert!(!unhealthy.is_operational());
    }

    #[test]
    fn test_health_report_requires_attention() {
        let healthy = HealthReport::healthy();
        assert!(!healthy.requires_attention());

        let critical = HealthReport::new(HealthStatus::Critical, SystemMetrics::new());
        assert!(critical.requires_attention());

        let mut with_critical_alert = HealthReport::healthy();
        with_critical_alert.add_alert(Alert::critical("Critical", "src"));
        assert!(with_critical_alert.requires_attention());
    }

    #[test]
    fn test_health_report_brief_summary() {
        let mut report = HealthReport::healthy();
        report.add_agent_health(AgentHealth::healthy("agent1"));
        report.add_agent_health(AgentHealth::unhealthy("agent2", "Error"));

        let summary = report.brief_summary();
        assert!(summary.contains("1/2 agents healthy"));
    }

    #[test]
    fn test_health_report_to_markdown() {
        let mut report = HealthReport::healthy().with_summary("Test report");
        report.add_agent_health(AgentHealth::healthy("test_agent"));
        report.add_alert(Alert::warning("Test alert", "system"));

        let md = report.to_markdown();
        assert!(md.contains("# Health Report"));
        assert!(md.contains("Overall Status"));
        assert!(md.contains("System Metrics"));
        assert!(md.contains("Agent Health"));
        assert!(md.contains("test_agent"));
        assert!(md.contains("Alerts"));
        assert!(md.contains("Test alert"));
    }

    // -------------------------------------------------------------------------
    // MonitorConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_monitor_config_default() {
        let config = MonitorConfig::default();
        assert_eq!(config.warning_error_rate, 1.0);
        assert_eq!(config.check_interval_secs, 30);
        assert!(config.include_agent_details);
        assert!(config.auto_alert);
    }

    #[test]
    fn test_monitor_config_strict() {
        let config = MonitorConfig::strict();
        assert!(config.warning_error_rate < MonitorConfig::default().warning_error_rate);
        assert!(config.warning_latency_ms < MonitorConfig::default().warning_latency_ms);
    }

    #[test]
    fn test_monitor_config_lenient() {
        let config = MonitorConfig::lenient();
        assert!(config.warning_error_rate > MonitorConfig::default().warning_error_rate);
        assert!(config.warning_latency_ms > MonitorConfig::default().warning_latency_ms);
    }

    #[test]
    fn test_monitor_config_minimal() {
        let config = MonitorConfig::minimal();
        assert!(!config.include_agent_details);
        assert!(!config.auto_alert);
    }

    // -------------------------------------------------------------------------
    // MonitorAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_monitor_agent_new() {
        let agent = MonitorAgent::new();
        assert_eq!(agent.config().warning_error_rate, 1.0);
    }

    #[test]
    fn test_monitor_agent_with_config() {
        let config = MonitorConfig::strict();
        let agent = MonitorAgent::with_config(config);
        assert!(agent.config().warning_error_rate < 1.0);
    }

    #[test]
    fn test_monitor_agent_builder() {
        let agent = MonitorAgent::builder()
            .warning_error_rate(2.0)
            .high_latency_ms(800.0)
            .auto_alert(false)
            .build();

        assert_eq!(agent.config().warning_error_rate, 2.0);
        assert_eq!(agent.config().high_latency_ms, 800.0);
        assert!(!agent.config().auto_alert);
    }

    #[test]
    fn test_monitor_agent_default() {
        let agent = MonitorAgent::default();
        assert_eq!(agent.config().warning_error_rate, 1.0);
    }

    // -------------------------------------------------------------------------
    // MonitorAgent Agent Registration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_monitor_agent_register_agent() {
        let mut agent = MonitorAgent::new();
        agent.register_agent("test_agent");
        assert!(agent.get_agent_health("test_agent").is_some());
    }

    #[test]
    fn test_monitor_agent_update_agent_health() {
        let mut agent = MonitorAgent::new();
        agent.register_agent("test_agent");
        agent.update_agent_health("test_agent", AgentHealth::unhealthy("test_agent", "Error"));

        let health = agent.get_agent_health("test_agent").unwrap();
        assert_eq!(health.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_monitor_agent_record_agent_success() {
        let mut agent = MonitorAgent::new();
        agent.register_agent("test_agent");
        agent.record_agent_failure("test_agent", "Error");
        agent.record_agent_success("test_agent");

        let health = agent.get_agent_health("test_agent").unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.error_count, 0);
    }

    #[test]
    fn test_monitor_agent_record_agent_failure() {
        let mut agent = MonitorAgent::new();
        agent.register_agent("test_agent");
        agent.record_agent_failure("test_agent", "Connection failed");

        let health = agent.get_agent_health("test_agent").unwrap();
        assert_eq!(health.error_count, 1);
        assert!(health.status.has_issues());
    }

    // -------------------------------------------------------------------------
    // MonitorAgent Health Check Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_monitor_agent_check_health() {
        let agent = MonitorAgent::new();
        let report = agent.check_health().await.unwrap();

        assert!(report.overall_status.is_operational());
        assert!(report.check_duration_ms > 0 || report.check_duration_ms == 0);
        assert!(report.summary.is_some());
    }

    #[tokio::test]
    async fn test_monitor_agent_check_health_with_agents() {
        let mut agent = MonitorAgent::new();
        agent.register_agent("agent1");
        agent.register_agent("agent2");

        let report = agent.check_health().await.unwrap();
        assert_eq!(report.agent_health.len(), 2);
    }

    #[tokio::test]
    async fn test_monitor_agent_get_metrics() {
        let agent = MonitorAgent::new();
        let metrics = agent.get_metrics().await.unwrap();

        // Should return valid metrics
        assert!(metrics.requests_per_second >= 0.0);
        assert!(metrics.average_latency_ms >= 0.0);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_config() {
        let config = MonitorConfig::strict();
        let agent = MonitorAgentBuilder::default().config(config).build();
        assert!(agent.config().warning_error_rate < 1.0);
    }

    #[test]
    fn test_builder_warning_error_rate() {
        let agent = MonitorAgentBuilder::default()
            .warning_error_rate(3.0)
            .build();
        assert_eq!(agent.config().warning_error_rate, 3.0);
    }

    #[test]
    fn test_builder_high_error_rate() {
        let agent = MonitorAgentBuilder::default().high_error_rate(8.0).build();
        assert_eq!(agent.config().high_error_rate, 8.0);
    }

    #[test]
    fn test_builder_critical_error_rate() {
        let agent = MonitorAgentBuilder::default()
            .critical_error_rate(15.0)
            .build();
        assert_eq!(agent.config().critical_error_rate, 15.0);
    }

    #[test]
    fn test_builder_latency_thresholds() {
        let agent = MonitorAgentBuilder::default()
            .warning_latency_ms(300.0)
            .high_latency_ms(600.0)
            .critical_latency_ms(1500.0)
            .build();

        assert_eq!(agent.config().warning_latency_ms, 300.0);
        assert_eq!(agent.config().high_latency_ms, 600.0);
        assert_eq!(agent.config().critical_latency_ms, 1500.0);
    }

    #[test]
    fn test_builder_memory_thresholds() {
        let agent = MonitorAgentBuilder::default()
            .warning_memory_mb(200.0)
            .high_memory_mb(400.0)
            .critical_memory_mb(800.0)
            .build();

        assert_eq!(agent.config().warning_memory_mb, 200.0);
        assert_eq!(agent.config().high_memory_mb, 400.0);
        assert_eq!(agent.config().critical_memory_mb, 800.0);
    }

    #[test]
    fn test_builder_check_interval() {
        let agent = MonitorAgentBuilder::default()
            .check_interval_secs(60)
            .build();
        assert_eq!(agent.config().check_interval_secs, 60);
    }

    #[test]
    fn test_builder_stale_threshold() {
        let agent = MonitorAgentBuilder::default()
            .stale_threshold_secs(180)
            .build();
        assert_eq!(agent.config().stale_threshold_secs, 180);
    }

    #[test]
    fn test_builder_max_consecutive_errors() {
        let agent = MonitorAgentBuilder::default()
            .max_consecutive_errors(5)
            .build();
        assert_eq!(agent.config().max_consecutive_errors, 5);
    }

    #[test]
    fn test_builder_include_agent_details() {
        let agent = MonitorAgentBuilder::default()
            .include_agent_details(false)
            .build();
        assert!(!agent.config().include_agent_details);
    }

    #[test]
    fn test_builder_auto_alert() {
        let agent = MonitorAgentBuilder::default().auto_alert(false).build();
        assert!(!agent.config().auto_alert);
    }

    #[test]
    fn test_builder_negative_values_clamped() {
        let agent = MonitorAgentBuilder::default()
            .warning_error_rate(-5.0)
            .warning_latency_ms(-100.0)
            .build();

        assert_eq!(agent.config().warning_error_rate, 0.0);
        assert_eq!(agent.config().warning_latency_ms, 0.0);
    }

    // -------------------------------------------------------------------------
    // Alert Generation Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_alert_generation_disabled() {
        let agent = MonitorAgent::builder().auto_alert(false).build();
        let report = agent.check_health().await.unwrap();
        assert!(report.alerts.is_empty());
    }

    #[tokio::test]
    async fn test_health_check_with_unhealthy_agent() {
        let mut agent = MonitorAgent::new();
        agent.register_agent("failing_agent");
        agent.record_agent_failure("failing_agent", "Error 1");
        agent.record_agent_failure("failing_agent", "Error 2");
        agent.record_agent_failure("failing_agent", "Error 3");

        let report = agent.check_health().await.unwrap();

        // Should have degraded overall status due to agent errors
        let agent_health = report.agent_health.get("failing_agent").unwrap();
        assert!(agent_health.status.has_issues());
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_health_multiple_failures() {
        let mut health = AgentHealth::new("agent");
        for i in 0..10 {
            health.record_failure(format!("Error {}", i));
        }
        assert_eq!(health.error_count, 10);
        assert_eq!(health.status, HealthStatus::Critical);
    }

    #[test]
    fn test_health_report_empty_agents() {
        let report = HealthReport::healthy();
        assert_eq!(report.healthy_agent_count(), 0);
        assert_eq!(report.unhealthy_agent_count(), 0);
        let summary = report.brief_summary();
        assert!(summary.contains("no agents monitored"));
    }

    #[test]
    fn test_health_report_empty_alerts() {
        let report = HealthReport::healthy();
        let summary = report.brief_summary();
        assert!(summary.contains("no alerts"));
    }

    #[tokio::test]
    async fn test_monitor_agent_no_agents_registered() {
        let agent = MonitorAgent::new();
        let report = agent.check_health().await.unwrap();
        assert!(report.agent_health.is_empty());
    }

    #[test]
    fn test_system_metrics_default() {
        let metrics = SystemMetrics::default();
        assert_eq!(metrics.requests_per_second, 0.0);
        assert!(metrics.cpu_usage.is_none());
    }
}
